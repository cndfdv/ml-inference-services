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


class RapidOCRBackend:
    """RapidOCR adapter. Initialization asserts the requested engine provider."""

    def __init__(self, settings):
        from importlib.metadata import version

        import numpy as np
        import onnxruntime as ort
        import rapidocr
        from PIL import Image
        from rapidocr import LangDet, LangRec, ModelType, OCRVersion, RapidOCR

        if settings.device == "cuda":
            ort.preload_dlls()

        self.np, self.Image = np, Image
        self.settings = settings
        self.manifest = validate_manifest(Path(settings.model_dir), settings.model_id)
        self.root = _settings_model_root(settings)
        self.provider_by_stage = {"detector": [], "recognizer": []}
        self.engine = RapidOCR(
            params={
                "Global.model_root_dir": str(self.root),
                "Global.use_cls": False,
                # RapidOCR eagerly constructs the classifier even when disabled.
                "Cls.model_path": str(self.root / "cls.onnx"),
                "Det.ocr_version": OCRVersion("PP-OCRv5"),
                "Det.model_type": ModelType.MOBILE,
                "Det.lang_type": LangDet.CH,
                "Det.model_path": str(self.root / "det.onnx"),
                "Det.mean": [0.485, 0.456, 0.406],
                "Det.std": [0.229, 0.224, 0.225],
                "Rec.ocr_version": OCRVersion("PP-OCRv5"),
                "Rec.model_type": ModelType.MOBILE,
                "Rec.lang_type": LangRec("cyrillic"),
                "Rec.model_path": str(self.root / "rec.onnx"),
                "Rec.rec_batch_num": settings.rec_batch_size,
                "EngineConfig.onnxruntime.use_cuda": settings.device == "cuda",
                "EngineConfig.onnxruntime.intra_op_num_threads": settings.cpu_threads,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
                "EngineConfig.onnxruntime.cuda_ep_cfg.use_tf32": False,
            }
        )
        self._configure_runtime(settings.device)
        self.info = _metadata(
            settings,
            self.manifest,
            runtime=f"rapidocr {getattr(rapidocr, '__version__', version('rapidocr'))}",
            precision="fp32",
            preprocessing={
                "decode": "Pillow RGB then OpenCV BGR",
                "detector_mean": [0.485, 0.456, 0.406],
                "detector_std": [0.229, 0.224, 0.225],
                "classifier": False,
                "recognizer_batch_size": settings.rec_batch_size,
                "recognizer": "PP-OCRv5 mobile Cyrillic",
            },
            dimension=None,
            max_tokens=None,
            providers=self._runtime_providers(),
        )

    def _configure_runtime(self, device):
        # Configuration attribute paths vary across RapidOCR releases. Fail closed if the
        # pinned API does not expose its ONNX session; never claim CUDA from config alone.
        # RapidOCR 3.9.2 constructs both stage sessions eagerly in __init__.
        self._engine_impl = self.engine
        self._runtime_providers()
        required = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
        for stage, stage_providers in self.provider_by_stage.items():
            if required not in stage_providers:
                raise RuntimeError(f"RapidOCR {stage} missing {required}: {stage_providers}")

    def _runtime_providers(self):
        providers = []
        impl = getattr(self, "_engine_impl", None)
        if impl is not None:
            for name in ("text_det", "text_rec"):
                stage = getattr(impl, name, None)
                wrapper = getattr(stage, "session", None)
                session = getattr(wrapper, "session", None)
                if session is None:
                    raise RuntimeError(f"RapidOCR {name} ONNX session is unavailable")
                actual = session.get_providers()
                key = "detector" if name == "text_det" else "recognizer"
                self.provider_by_stage[key] = actual
                providers.extend(actual)
        return list(dict.fromkeys(providers))

    def infer(self, items):
        results = []
        for item in items:
            image_bytes = (
                item if isinstance(item, (bytes, bytearray, memoryview)) else item.get("image")
            )
            if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
                raise ValueError("OCR items require image bytes in 'image'")
            with self.Image.open(__import__("io").BytesIO(image_bytes)) as image:
                rgb = image.convert("RGB")
                width, height = rgb.size
                bgr = self.np.asarray(rgb)[:, :, ::-1].copy()
            raw = self.engine(bgr)
            results.append(self._format(raw, width, height))
        return results

    @staticmethod
    def _format(raw, width, height):
        lines = []
        # RapidOCR v3 returns OCRResult; support its explicit text boxes and preserve blanks.
        boxes = getattr(raw, "boxes", None)
        texts = getattr(raw, "txts", None)
        scores = getattr(raw, "scores", None)
        if boxes is None and isinstance(raw, tuple) and raw:
            value = raw[0]
            if value:
                boxes = [row[0] for row in value]
                texts = [row[1] for row in value]
                scores = [row[2] for row in value]
        for box, text, confidence in zip(
            [] if boxes is None else boxes,
            [] if texts is None else texts,
            [] if scores is None else scores,
        ):
            polygon = [[float(p[0]), float(p[1])] for p in box]
            lines.append(
                {
                    "text": str(text),
                    "bbox": [
                        min(p[0] for p in polygon),
                        min(p[1] for p in polygon),
                        max(p[0] for p in polygon),
                        max(p[1] for p in polygon),
                    ],
                    "polygon": polygon,
                    "confidence": float(confidence),
                }
            )
        joined = "\n".join(line["text"] for line in lines)
        return {"text": joined, "lines": lines, "width": width, "height": height}

    def metadata(self):
        return dict(self.info)

def create_backend(settings):
    settings.validate()
    return RapidOCRBackend(settings)
