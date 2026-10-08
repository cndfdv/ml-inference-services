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


class E5Backend(_EmbeddingBackend):
    model_max_tokens = 512
    dimension = 384

    def __init__(self, settings):
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer

        self.settings = settings
        self.manifest = validate_manifest(Path(settings.model_dir), settings.model_id)
        torch.set_num_threads(settings.cpu_threads)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if settings.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("DEVICE=cuda requested but torch.cuda.is_available() is false")
        root = _settings_model_root(settings)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        self.model = (
            AutoModel.from_pretrained(root, local_files_only=True).to(settings.device).eval()
        )
        self.info = _metadata(
            settings,
            self.manifest,
            runtime=f"torch {torch.__version__} / transformers {transformers.__version__}",
            precision="fp32",
            preprocessing={
                "pooling": "attention-mask weighted mean",
                "normalization": "L2",
                "prefixes": {"query": "query: ", "passage": "passage: "},
                "max_length": self.model_max_tokens,
            },
            dimension=self.dimension,
            max_tokens=self.model_max_tokens,
            providers=[settings.device],
        )

    def infer(self, items):
        texts = []
        for item in items:
            role = item.get("role")
            if role not in ("query", "passage") or not isinstance(item.get("text"), str):
                raise ValueError("embedding items require text and role=query|passage")
            texts.append(f"{role}: " + item["text"])
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
        outputs: list[Any] = [None] * len(items)
        torch = self.torch
        with torch.inference_mode():
            for indices in self._chunks(lengths):
                width = max(lengths[i] for i in indices)
                inputs = self.tokenizer.pad(
                    {k: [v[i] for i in indices] for k, v in encoded.items()},
                    padding="max_length",
                    max_length=width,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.settings.device) for k, v in inputs.items()}
                hidden = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                vectors = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
                for index, vector in zip(indices, vectors.cpu().tolist()):
                    outputs[index] = vector
        return outputs

    def metadata(self):
        return dict(self.info)

def create_backend(settings):
    settings.validate()
    return E5Backend(settings)
