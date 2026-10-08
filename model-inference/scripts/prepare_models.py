#!/usr/bin/env python3
"""Prepare the exact, checksum-manifested model bundles under /models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCK = json.loads((REPO / "models.lock.json").read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(
    repo: str, revision: str, cache: Path | None, patterns: list[str] | None = None
) -> Path:
    if cache:
        candidates = [cache / "snapshots" / revision]
        candidates.extend(cache.glob(f"models--*/snapshots/{revision}"))
        candidates.extend(cache.glob(f"**/snapshots/{revision}"))
        if cache.name == revision:
            candidates.append(cache)
        for candidate in candidates:
            if candidate.is_dir() and (candidate / "config.json").is_file():
                return candidate
    from huggingface_hub import snapshot_download

    patterns = patterns or [
        "config.json",
        "*.safetensors",
        "*.safetensors.index.json",
        "pytorch_model*.bin",
        "pytorch_model*.bin.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
        "spiece.model",
        "sentencepiece.bpe.model",
    ]
    return Path(snapshot_download(repo_id=repo, revision=revision, allow_patterns=patterns))


def copy_hf_snapshot(source: Path, destination: Path, *, include_weights: bool):
    destination.mkdir(parents=True, exist_ok=True)
    patterns = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
        "spiece.model",
        "sentencepiece.bpe.model",
    ]
    if include_weights:
        patterns.extend(
            [
                "*.safetensors",
                "*.safetensors.index.json",
                "pytorch_model*.bin",
                "pytorch_model*.bin.index.json",
            ]
        )
    copied = 0
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(source)
        if any(rel.match(pattern) for pattern in patterns):
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path.resolve(), target)
            copied += 1
    if not copied:
        raise ValueError(f"no pinned model files found in snapshot {source}")


def copy_whisper_snapshot(source: Path, destination: Path):
    files = (
        "model.bin",
        "config.json",
        "tokenizer.json",
        "preprocessor_config.json",
        "vocabulary.json",
    )
    destination.mkdir(parents=True, exist_ok=True)
    for name in files:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(f"pinned Whisper snapshot is missing {name}: {source}")
        shutil.copy2(path.resolve(), destination / name)


def export_user(source: Path, dest: Path, make_int8: bool):
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModel.from_pretrained(
        source, local_files_only=True, attn_implementation="eager"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)

    class DenseCLS(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            return self.inner(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=False,
            )[0][:, 0, :]

    wrapper = DenseCLS(model).eval()
    sample = tokenizer(
        ["export sample"], padding="max_length", max_length=8, truncation=True, return_tensors="pt"
    )
    args = (sample["input_ids"], sample["attention_mask"])
    torch.onnx.export(
        wrapper,
        args,
        dest / "model-fp32.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["sentence_embedding"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "sentence_embedding": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )
    if make_int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(
            str(dest / "model-fp32.onnx"),
            str(dest / "model-int8.onnx"),
            weight_type=QuantType.QInt8,
        )


def manifest(model_id: str, root: Path):
    entry = LOCK[model_id]
    weights = entry["weights"]
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            rel = path.relative_to(root).as_posix()
            files[rel] = sha256(path)
    data = {
        "model_id": model_id,
        "revision": entry["revision"],
        "weights": weights,
        "contract_fingerprint": entry["contract_fingerprint"],
        "files": files,
    }
    target = root / "manifest.json"
    temporary = root / ".manifest.json.tmp"
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, target)


def prepare(
    model_id: str, output: Path, benchmark_cache: Path | None, ocr_cache: Path | None, int8: bool
):
    target = output / model_id
    target.mkdir(parents=True, exist_ok=True)
    entry = LOCK[model_id]
    with tempfile.TemporaryDirectory(prefix=f"prepare-{model_id}-", dir=output) as tmp_name:
        tmp = Path(tmp_name)
        if model_id == "e5-small":
            source = snapshot(entry["weights"], entry["revision"], benchmark_cache)
            copy_hf_snapshot(source, tmp, include_weights=True)
        elif model_id == "user-bge-m3":
            source = snapshot(entry["weights"], entry["revision"], benchmark_cache)
            copy_hf_snapshot(source, tmp / "tokenizer", include_weights=False)
            fp32_src = (
                next(benchmark_cache.rglob("cls-op17-fp32.onnx"), None) if benchmark_cache else None
            )
            int8_src = (
                next(benchmark_cache.rglob("cls-op17-int8.onnx"), None) if benchmark_cache else None
            )
            if fp32_src and fp32_src.is_file():
                shutil.copy2(fp32_src, tmp / "model-fp32.onnx")
            else:
                export_user(source, tmp, int8)
            if int8_src and int8_src.is_file():
                shutil.copy2(int8_src, tmp / "model-int8.onnx")
            elif int8 and not (tmp / "model-int8.onnx").exists():
                from onnxruntime.quantization import QuantType, quantize_dynamic

                quantize_dynamic(
                    str(tmp / "model-fp32.onnx"),
                    str(tmp / "model-int8.onnx"),
                    weight_type=QuantType.QInt8,
                )
        elif model_id == "rapid-v5-mobile":
            urls = entry["source"]
            for name, digest in entry["weights"].items():
                candidates = (
                    [p for p in ocr_cache.rglob("*.onnx") if p.is_file()] if ocr_cache else []
                )
                source = next((path for path in candidates if sha256(path) == digest), None)
                destination = tmp / name
                if source is not None:
                    shutil.copy2(source, destination)
                else:
                    request = urllib.request.Request(
                        urls[name], headers={"User-Agent": "model-inference-preparer/1"}
                    )
                    with (
                        urllib.request.urlopen(request, timeout=120) as response,
                        destination.open("wb") as stream,
                    ):
                        shutil.copyfileobj(response, stream)
                if sha256(destination) != digest:
                    raise ValueError(f"pinned OCR checksum mismatch for {name}")
        else:
            whisper_files = [
                "model.bin",
                "config.json",
                "tokenizer.json",
                "preprocessor_config.json",
                "vocabulary.json",
            ]
            source = snapshot(
                entry["weights"], entry["revision"], benchmark_cache, patterns=whisper_files
            )
            copy_whisper_snapshot(source, tmp)
        manifest(model_id, tmp)
        # tempfile creates mode 0700; the non-root runtime must read the bundle.
        for path in tmp.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)
        tmp.chmod(0o755)
        # Stage first, then swap the complete bundle into place. Restore an existing
        # bundle if the final rename fails.
        backup = output / f".{model_id}.previous"
        if backup.exists():
            shutil.rmtree(backup)
        if target.exists():
            os.replace(target, backup)
        try:
            os.replace(tmp, target)
        except Exception:
            if backup.exists():
                os.replace(backup, target)
            raise
        if backup.exists():
            shutil.rmtree(backup)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=LOCK.keys(), action="append")
    parser.add_argument("--output", type=Path, default=Path("/models"))
    parser.add_argument("--benchmark-cache", type=Path)
    parser.add_argument("--ocr-cache", type=Path)
    parser.add_argument("--int8", action="store_true", help="also export USER dynamic INT8")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for model_id in args.model or list(LOCK):
        prepare(model_id, args.output, args.benchmark_cache, args.ocr_cache, args.int8)


if __name__ == "__main__":
    main()
