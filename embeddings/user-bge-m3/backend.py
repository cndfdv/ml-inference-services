"""Self-contained pinned backend for this model service."""
from __future__ import annotations
import hashlib
import io
import json
import math
import re
from pathlib import Path
from typing import Any

LOCK_PATH = Path(__file__).resolve().parent / "model.lock.json"

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(model_dir: Path, model_id: str, lock: dict | None = None) -> dict:
    """Validate an immutable prepared bundle against its manifest and lock."""
    lock = lock or json.loads(LOCK_PATH.read_text())
    root = model_dir / model_id
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"model manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    entry = lock[model_id]
    for field in ("revision", "weights", "contract_fingerprint"):
        if manifest.get(field) != entry[field]:
            raise RuntimeError(f"{model_id} manifest {field} does not match model.lock.json")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError(f"{model_id} manifest has no file checksums")
    for rel, expected in files.items():
        path = (root / rel).resolve()
        if root.resolve() not in path.parents or not path.is_file():
            raise RuntimeError(f"invalid or missing model file: {rel}")
        if file_sha256(path) != expected:
            raise RuntimeError(f"model checksum mismatch: {rel}")
    return manifest


def plan_batches(
    lengths: list[int], max_batch_size: int, max_batch_tokens: int, max_tokens: int
) -> list[list[int]]:
    """Greedily split item indices by padded token budget, preserving order.

    One item may use more than the batch budget, but never exceed its model limit.
    """
    result: list[list[int]] = []
    batch: list[int] = []
    longest = 0
    for index, length in enumerate(lengths):
        if length < 1 or length > max_tokens:
            raise ValueError(f"token length {length} outside valid range 1..{max_tokens}")
        new_longest = max(longest, length)
        if batch and (
            len(batch) >= max_batch_size or new_longest * (len(batch) + 1) > max_batch_tokens
        ):
            result.append(batch)
            batch, longest = [], 0
        batch.append(index)
        longest = max(longest, length)
    if batch:
        result.append(batch)
    return result


def _settings_model_root(settings) -> Path:
    return Path(settings.model_dir) / settings.model_id


def _metadata(
    settings,
    manifest: dict,
    *,
    runtime: str,
    precision: str,
    preprocessing: dict,
    dimension: int,
    max_tokens: int,
    providers: list[str] | None = None,
) -> dict:
    lock = json.loads(LOCK_PATH.read_text())[settings.model_id]
    return {
        "model_id": settings.model_id,
        "checkpoint_revision": manifest["revision"],
        "weights": manifest["weights"],
        "weight_files": manifest["files"],
        "runtime": runtime,
        "precision": precision,
        "preprocessing": preprocessing,
        "device": settings.device,
        "providers": providers or [],
        "dimensions": dimension,
        "max_tokens": max_tokens,
        "contract_fingerprint": lock["contract_fingerprint"],
    }


class _EmbeddingBackend:
    model_max_tokens: int

    def _chunks(self, lengths: list[int]) -> list[list[int]]:
        return plan_batches(
            lengths,
            self.settings.max_batch_size,
            self.settings.max_batch_tokens,
            self.model_max_tokens,
        )


class UserBGEM3Backend(_EmbeddingBackend):
    model_max_tokens = 8192
    dimension = 1024

    def __init__(self, settings):
        import onnxruntime as ort
        import torch
        from transformers import AutoTokenizer

        self.settings = settings
        self.manifest = validate_manifest(Path(settings.model_dir), settings.model_id)
        if settings.device == "cuda":
            # CPU torch wheels do not preload CUDA libraries for ONNX Runtime.
            ort.preload_dlls()
        torch.set_num_threads(settings.cpu_threads)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if (
            settings.device == "cuda"
            and "CUDAExecutionProvider" not in ort.get_available_providers()
        ):
            raise RuntimeError(
                "DEVICE=cuda requested but ONNX Runtime has no CUDAExecutionProvider"
            )
        if settings.device == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            providers = [("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"]
        root = _settings_model_root(settings)
        graph = root / ("model-int8.onnx" if settings.user_variant == "int8" else "model-fp32.onnx")
        options = ort.SessionOptions()
        options.intra_op_num_threads = settings.cpu_threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(graph), sess_options=options, providers=providers)
        actual = self.session.get_providers()
        if settings.device == "cuda" and "CUDAExecutionProvider" not in actual:
            raise RuntimeError(f"CUDA provider failed to initialize; actual providers: {actual}")
        self.tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
        self.inputs = [i.name for i in self.session.get_inputs()]
        self.info = _metadata(
            settings,
            self.manifest,
            runtime=f"onnxruntime {ort.__version__}",
            precision=settings.user_variant,
            preprocessing={
                "pooling": "CLS token",
                "normalization": "L2",
                "prefixes": None,
                "max_length": self.model_max_tokens,
            },
            dimension=self.dimension,
            max_tokens=self.model_max_tokens,
            providers=actual,
        )

    def infer(self, items):
        texts = []
        for item in items:
            if not isinstance(item.get("text"), str):
                raise ValueError("embedding items require text")
            texts.append(item["text"])
        if not texts:
            return []
        encoded = self.tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=self.model_max_tokens,
            padding=False,
        )
        lengths = [len(ids) for ids in encoded["input_ids"]]
        outputs = [None] * len(items)
        for indices in self._chunks(lengths):
            width = max(lengths[i] for i in indices)
            feeds = {}
            for name in self.inputs:
                key = "token_type_ids" if name == "token_type_ids" else name
                if key in encoded:
                    feeds[name] = self._pad(
                        encoded[key],
                        indices,
                        width,
                        self.tokenizer.pad_token_id if key == "input_ids" else 0,
                    )
            vectors = self.session.run(None, feeds)[0]
            for index, vector in zip(indices, vectors):
                norm = math.sqrt(sum(float(x) * float(x) for x in vector))
                outputs[index] = [float(x) / norm for x in vector] if norm else [0.0] * len(vector)
        return outputs

    @staticmethod
    def _pad(values, indices, width, pad_value):
        import numpy as np

        result = np.full((len(indices), width), pad_value, dtype=np.int64)
        for row, index in enumerate(indices):
            value = values[index]
            result[row, : len(value)] = value
        return result

    def metadata(self):
        return dict(self.info)

def create_backend(settings):
    settings.validate()
    return UserBGEM3Backend(settings)
