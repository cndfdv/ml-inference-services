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


class AudioDurationExceeded(ValueError):
    """Decoded audio exceeds the configured duration before materialization."""


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

def create_backend(settings):
    settings.validate()
    return WhisperBackend(settings)
