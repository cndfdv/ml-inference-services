"""Prepare this service's pinned, checksum-manifested model bundle."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from .backend import LOCK_PATH, validate_manifest
LOCK = json.loads(LOCK_PATH.read_text())
MODEL_ID = "rapid-v5-mobile"

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

def prepare(output: Path, ocr_cache: Path | None = None):
    output.mkdir(parents=True, exist_ok=True)
    target = output / MODEL_ID
    if target.exists():
        validate_manifest(output, MODEL_ID)
        return target
    entry = LOCK[MODEL_ID]
    with tempfile.TemporaryDirectory(prefix=f"prepare-{MODEL_ID}-", dir=output) as tmp_name:
        tmp = Path(tmp_name)
        urls = entry["source"]
        for name, digest in entry["weights"].items():
            candidates = [p for p in ocr_cache.rglob("*.onnx") if p.is_file()] if ocr_cache else []
            cached = next((path for path in candidates if sha256(path) == digest), None)
            destination = tmp / name
            if cached is not None:
                shutil.copy2(cached, destination)
            else:
                request = urllib.request.Request(urls[name], headers={"User-Agent": "model-inference-preparer/1"})
                with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
            if sha256(destination) != digest:
                raise ValueError(f"pinned OCR checksum mismatch for {name}")
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
        return target
    if mode == "offline":
        raise RuntimeError(f"prepared model bundle is missing: {target}")
    prepared = prepare(output)
    validate_manifest(output, MODEL_ID)
    return prepared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/models"))
    parser.add_argument("--ocr-cache", type=Path)
    args = parser.parse_args()
    prepare(args.output, args.ocr_cache)


if __name__ == "__main__":
    main()
