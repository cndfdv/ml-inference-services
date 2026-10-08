"""Pinned model backends. Heavy runtimes are imported only when a backend starts."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from pathlib import Path
from typing import Any

LOCK_PATH = Path(__file__).resolve().parent.parent / "models.lock.json"


class AudioDurationExceeded(ValueError):
    pass


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
            raise RuntimeError(f"{model_id} manifest {field} does not match models.lock.json")
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


def _read_bounded_audio_frames(frames, max_samples: int):
    """Collect resampled s16 frames, rejecting an over-limit frame before conversion."""
    raw_buffer = io.BytesIO()
    dtype = None
    sample_count = 0
    for frame in frames:
        frame_samples = int(frame.samples)
        if frame_samples < 0 or sample_count + frame_samples > max_samples:
            raise AudioDurationExceeded("audio duration exceeds configured maximum")
        array = frame.to_ndarray()
        dtype = array.dtype
        raw_buffer.write(array.tobytes())
        sample_count += frame_samples
    return raw_buffer.getvalue(), dtype, sample_count


def decode_whisper_audio_bounded(input_file, max_duration_s: float, sampling_rate: int = 16000):
    """Decode like faster-whisper 1.2.1 while bounding the materialized waveform."""
    import av
    import numpy as np

    if not math.isfinite(max_duration_s) or max_duration_s <= 0:
        raise ValueError("max audio duration must be finite and positive")
    max_samples = math.floor(max_duration_s * sampling_rate)
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sampling_rate)

    def ignore_invalid_frames(frames):
        iterator = iter(frames)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                return
            except av.error.InvalidDataError:
                continue

    def group_frames(frames, num_samples=500000):
        fifo = av.audio.fifo.AudioFifo()
        for frame in frames:
            frame.pts = None
            fifo.write(frame)
            if fifo.samples >= num_samples:
                yield fifo.read()
        if fifo.samples > 0:
            yield fifo.read()

    def resampled_frames():
        with av.open(input_file, mode="r", metadata_errors="ignore") as container:
            for grouped in group_frames(ignore_invalid_frames(container.decode(audio=0))):
                yield from resampler.resample(grouped)
            yield from resampler.resample(None)

    raw, dtype, sample_count = _read_bounded_audio_frames(resampled_frames(), max_samples)
    if sample_count == 0 or dtype is None:
        raise ValueError("audio contains no samples")
    return np.frombuffer(raw, dtype=dtype).astype(np.float32) / 32768.0


def decode_whisper_item(item, *, max_bytes: int, max_duration_s: float, decoder):
    """Validate and decode one audio item to faster-whisper's 16 kHz waveform."""
    if not isinstance(item, (bytes, bytearray, memoryview, dict)):
        raise ValueError("audio item must be bytes or an object containing audio bytes")
    audio_bytes = item if isinstance(item, (bytes, bytearray, memoryview)) else item.get("audio")
    if not isinstance(audio_bytes, (bytes, bytearray, memoryview)) or not audio_bytes:
        raise ValueError("audio items require non-empty audio bytes")
    if len(audio_bytes) > max_bytes:
        raise ValueError("audio exceeds maximum byte size")
    if not math.isfinite(max_duration_s) or max_duration_s <= 0:
        raise ValueError("max audio duration must be finite and positive")
    language = (
        "ru" if isinstance(item, (bytes, bytearray, memoryview)) else item.get("language", "ru")
    )
    task = (
        "transcribe"
        if isinstance(item, (bytes, bytearray, memoryview))
        else item.get("task", "transcribe")
    )
    if language is not None and (
        not isinstance(language, str) or re.fullmatch(r"[a-z]{2,3}", language) is None
    ):
        raise ValueError("language must be a lowercase ISO 639 code or null")
    if task not in {"transcribe", "translate"}:
        raise ValueError("task must be transcribe or translate")
    try:
        waveform = decoder(io.BytesIO(bytes(audio_bytes)), sampling_rate=16000)
    except AudioDurationExceeded as exc:
        raise ValueError(f"audio duration exceeds maximum of {max_duration_s:g} seconds") from exc
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("audio could not be decoded") from exc
    sample_count = len(waveform)
    duration = sample_count / 16000.0
    if sample_count == 0:
        raise ValueError("audio contains no samples")
    if duration > max_duration_s:
        raise ValueError(f"audio duration exceeds maximum of {max_duration_s:g} seconds")
    return waveform, language, task, duration


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


class UserBGEM3Backend(_EmbeddingBackend):
    model_max_tokens = 8192
    dimension = 1024

    def __init__(self, settings):
        import onnxruntime as ort
        import torch
        from transformers import AutoTokenizer

        self.settings = settings
        self.manifest = validate_manifest(Path(settings.model_dir), settings.model_id)
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


class WhisperBackend:
    """Sequential-file Whisper transcription using the same FP32 bundle on CPU/CUDA."""

    model_max_tokens = None
    decode_options = {
        "beam_size": 5,
        "temperature": 0.0,
        "condition_on_previous_text": True,
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "suppress_blank": True,
        "suppress_tokens": [-1],
        "without_timestamps": False,
        "word_timestamps": False,
        "vad_filter": False,
    }

    def __init__(self, settings):
        from importlib.metadata import version

        import ctranslate2
        from faster_whisper import WhisperModel

        self.settings = settings
        self.manifest = validate_manifest(Path(settings.model_dir), settings.model_id)
        self.root = _settings_model_root(settings)
        if settings.device == "cuda" and ctranslate2.get_cuda_device_count() < 1:
            raise RuntimeError("DEVICE=cuda requested but CTranslate2 has no CUDA device")
        max_duration = getattr(settings, "max_audio_duration_s", 600)

        def bounded_decoder(stream, sampling_rate=16000):
            return decode_whisper_audio_bounded(stream, max_duration, sampling_rate)

        self.decoder = bounded_decoder
        self.model = WhisperModel(
            str(self.root),
            device=settings.device,
            compute_type="float32",
            cpu_threads=settings.cpu_threads,
            num_workers=1,
        )
        self.info = _metadata(
            settings,
            self.manifest,
            runtime=f"faster-whisper {version('faster-whisper')} / CTranslate2 {ctranslate2.__version__}",
            precision="float32",
            preprocessing={
                "audio_sample_rate": 16000,
                "default_language": "ru",
                "default_task": "transcribe",
                "decode_options": dict(self.decode_options),
                "batching": "sequential files; faster-whisper internal chunk processing",
            },
            dimension=None,
            max_tokens=None,
            providers=[settings.device],
        )

    def infer(self, items):
        max_bytes = getattr(self.settings, "max_audio_bytes", 64 * 1024 * 1024)
        max_duration = getattr(self.settings, "max_audio_duration_s", 600)
        results = []
        for item in items:
            waveform, language, task, duration = decode_whisper_item(
                item,
                max_bytes=max_bytes,
                max_duration_s=max_duration,
                decoder=self.decoder,
            )
            segments, info = self.model.transcribe(
                waveform,
                language=language,
                task=task,
                **self.decode_options,
            )
            rows = [
                {"start": float(segment.start), "end": float(segment.end), "text": segment.text}
                for segment in segments
            ]
            results.append(
                {
                    "text": "".join(row["text"] for row in rows).strip(),
                    "segments": rows,
                    "language": info.language,
                    "language_probability": float(info.language_probability),
                    "duration": duration,
                }
            )
        return results

    def metadata(self):
        return dict(self.info)


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
    """Create the pinned backend for settings.model_id."""
    settings.validate()
    cls = {
        "e5-small": E5Backend,
        "user-bge-m3": UserBGEM3Backend,
        "rapid-v5-mobile": RapidOCRBackend,
        "whisper-large-v3": WhisperBackend,
    }[settings.model_id]
    return cls(settings)
