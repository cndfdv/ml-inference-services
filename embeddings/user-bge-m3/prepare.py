"""Prepare this service's pinned, checksum-manifested model bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from .backend import LOCK_PATH, validate_manifest

LOCK = json.loads(LOCK_PATH.read_text())
MODEL_ID = "user-bge-m3"


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _quantize_int8(fp32_path: Path, output_path: Path):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(fp32_path), str(output_path), weight_type=QuantType.QInt8)


def _extend_with_int8(output: Path, target: Path):
    manifest_data = validate_manifest(output, MODEL_ID)
    variant = target / "model-int8.onnx"
    if variant.exists():
        if "model-int8.onnx" not in manifest_data["files"]:
            raise RuntimeError("existing INT8 graph is not covered by the model manifest")
        return target

    fp32 = target / "model-fp32.onnx"
    if not fp32.is_file() or "model-fp32.onnx" not in manifest_data["files"]:
        raise RuntimeError("valid FP32 graph is required to prepare USER INT8")

    old_manifest = (target / "manifest.json").read_bytes()
    with tempfile.TemporaryDirectory(prefix="prepare-user-bge-m3-int8-", dir=output) as tmp_name:
        staged = Path(tmp_name) / "model-int8.onnx"
        _quantize_int8(fp32, staged)
        if not staged.is_file() or staged.stat().st_size == 0:
            raise RuntimeError("INT8 quantizer did not produce a model graph")
        os.replace(staged, variant)
    try:
        manifest(MODEL_ID, target)
        validate_manifest(output, MODEL_ID)
    except Exception:
        variant.unlink(missing_ok=True)
        restore = target / ".manifest.restore.tmp"
        restore.write_bytes(old_manifest)
        os.replace(restore, target / "manifest.json")
        raise
    return target


def prepare(output: Path, benchmark_cache: Path | None = None, int8: bool = False):
    output.mkdir(parents=True, exist_ok=True)
    target = output / MODEL_ID
    if target.exists():
        validate_manifest(output, MODEL_ID)
        return _extend_with_int8(output, target) if int8 else target
    entry = LOCK[MODEL_ID]
    with tempfile.TemporaryDirectory(prefix=f"prepare-{MODEL_ID}-", dir=output) as tmp_name:
        tmp = Path(tmp_name)
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
            _quantize_int8(tmp / "model-fp32.onnx", tmp / "model-int8.onnx")
        manifest(MODEL_ID, tmp)
        for path in tmp.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)
        tmp.chmod(0o755)
        os.replace(tmp, target)
    return target


def ensure_prepared(settings):
    """Reuse a valid bundle, prepare a missing bundle, or fail in offline mode."""
    settings.validate()
    if settings.model_id != MODEL_ID:
        raise ValueError(f"this preparation module only supports {MODEL_ID}")
    output = Path(settings.model_dir)
    target = output / MODEL_ID
    mode = os.getenv("PREPARE_MODE", "auto").lower()
    if mode not in {"auto", "offline"}:
        raise ValueError("PREPARE_MODE must be auto or offline")
    if target.exists():
        validate_manifest(output, MODEL_ID)
        if settings.user_variant == "int8":
            if mode == "offline" and not (target / "model-int8.onnx").is_file():
                raise RuntimeError(
                    f"prepared USER INT8 graph is missing: {target / 'model-int8.onnx'}"
                )
            if not (target / "model-int8.onnx").is_file():
                return _extend_with_int8(output, target)
            manifest_data = validate_manifest(output, MODEL_ID)
            if "model-int8.onnx" not in manifest_data["files"]:
                raise RuntimeError("existing INT8 graph is not covered by the model manifest")
        return target
    if mode == "offline":
        raise RuntimeError(f"prepared model bundle is missing: {target}")
    prepared = prepare(output, int8=settings.user_variant == "int8")
    manifest_data = validate_manifest(output, MODEL_ID)
    if settings.user_variant == "int8":
        if not (prepared / "model-int8.onnx").is_file():
            raise RuntimeError(f"prepared USER INT8 graph is missing: {prepared / 'model-int8.onnx'}")
        if "model-int8.onnx" not in manifest_data["files"]:
            raise RuntimeError("prepared INT8 graph is not covered by the model manifest")
    return prepared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/models"))
    parser.add_argument("--benchmark-cache", type=Path)
    parser.add_argument("--int8", action="store_true", help="also export USER dynamic INT8")
    args = parser.parse_args()
    prepare(args.output, args.benchmark_cache, args.int8)


if __name__ == "__main__":
    main()
