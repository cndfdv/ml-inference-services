import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_id: str = "e5-small"
    model_dir: str = "/models"
    device: str = "cuda"
    user_variant: str = "fp32"
    max_batch_size: int = 16
    max_batch_tokens: int = 8192
    batch_wait_ms: int = 10
    max_queue_items: int = 256
    request_timeout_s: float = 600
    workers: int = 1
    cpu_threads: int = 4
    max_request_items: int = 64
    max_text_chars: int = 65536
    max_image_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 20_000_000
    max_request_body_bytes: int = 64 * 1024 * 1024
    max_audio_bytes: int = 64 * 1024 * 1024
    max_audio_duration_s: float = 600
    rec_batch_size: int = 6
    pdf_dpi: int = 150

    @classmethod
    def from_env(cls):
        def number(name, default, cast):
            return cast(os.getenv(name, default))

        s = cls(
            model_id=os.getenv("MODEL_ID", "e5-small"),
            model_dir=os.getenv("MODEL_DIR", "/models"),
            device=os.getenv("DEVICE", "cuda"),
            user_variant=os.getenv("USER_VARIANT", "fp32"),
            max_batch_size=number("MAX_BATCH_SIZE", 16, int),
            max_batch_tokens=number("MAX_BATCH_TOKENS", 8192, int),
            batch_wait_ms=number("BATCH_WAIT_MS", 10, int),
            max_queue_items=number("MAX_QUEUE_ITEMS", 256, int),
            request_timeout_s=number("REQUEST_TIMEOUT_S", 600, float),
            workers=number("WORKERS", 1, int),
            cpu_threads=number("CPU_THREADS", 4, int),
            max_request_items=number("MAX_REQUEST_ITEMS", 64, int),
            max_text_chars=number("MAX_TEXT_CHARS", 65536, int),
            max_image_bytes=number("MAX_IMAGE_BYTES", 10 * 1024 * 1024, int),
            max_image_pixels=number("MAX_IMAGE_PIXELS", 20_000_000, int),
            max_request_body_bytes=number("MAX_REQUEST_BODY_BYTES", 64 * 1024 * 1024, int),
            max_audio_bytes=number("MAX_AUDIO_BYTES", 64 * 1024 * 1024, int),
            max_audio_duration_s=number("MAX_AUDIO_DURATION_S", 600, float),
            rec_batch_size=number("REC_BATCH_SIZE", 6, int),
            pdf_dpi=number("PDF_DPI", 150, int),
        )
        s.validate()
        return s

    def validate(self):
        allowed = {"e5-small", "user-bge-m3", "rapid-v5-mobile", "whisper-large-v3"}
        if self.model_id not in allowed:
            raise ValueError(f"MODEL_ID must be one of {sorted(allowed)}")
        if self.device not in {"cuda", "cpu"}:
            raise ValueError("DEVICE must be cuda or cpu")
        if self.user_variant not in {"fp32", "int8"}:
            raise ValueError("USER_VARIANT must be fp32 or int8")
        if self.user_variant == "int8" and self.device != "cpu":
            raise ValueError("USER_VARIANT=int8 requires DEVICE=cpu")
        for name in (
            "max_batch_size",
            "max_batch_tokens",
            "max_queue_items",
            "workers",
            "cpu_threads",
            "max_request_items",
            "max_text_chars",
            "max_image_bytes",
            "max_image_pixels",
            "max_request_body_bytes",
            "max_audio_bytes",
            "rec_batch_size",
            "pdf_dpi",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.batch_wait_ms < 0:
            raise ValueError("batch_wait_ms must be non-negative")
        if not math.isfinite(self.request_timeout_s) or self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be finite and positive")
        if not math.isfinite(self.max_audio_duration_s) or self.max_audio_duration_s <= 0:
            raise ValueError("max_audio_duration_s must be finite and positive")
        if not self.model_dir:
            raise ValueError("model_dir must be non-empty")

    @property
    def model(self):
        return self.model_id
