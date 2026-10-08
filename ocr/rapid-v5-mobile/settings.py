import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_id: str = "rapid-v5-mobile"
    model_dir: str = "/models"
    device: str = "cuda"
    max_batch_size: int = 16
    max_batch_tokens: int = 8192
    batch_wait_ms: int = 10
    max_queue_items: int = 256
    request_timeout_s: float = 600
    workers: int = 1
    cpu_threads: int = 4
    max_request_items: int = 64
    max_request_body_bytes: int = 64 * 1024 * 1024
    max_image_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 20_000_000
    rec_batch_size: int = 6
    pdf_dpi: int = 150

    @classmethod
    def from_env(cls):
        values = {
            "model_id": os.getenv("MODEL_ID", "rapid-v5-mobile"),
            "model_dir": os.getenv("MODEL_DIR", "/models"),
            "device": os.getenv("DEVICE", "cuda"),
            "max_batch_size": int(os.getenv("MAX_BATCH_SIZE", str(16))),
            "max_batch_tokens": int(os.getenv("MAX_BATCH_TOKENS", str(8192))),
            "batch_wait_ms": int(os.getenv("BATCH_WAIT_MS", str(10))),
            "max_queue_items": int(os.getenv("MAX_QUEUE_ITEMS", str(256))),
            "request_timeout_s": float(os.getenv("REQUEST_TIMEOUT_S", str(600))),
            "workers": int(os.getenv("WORKERS", str(1))),
            "cpu_threads": int(os.getenv("CPU_THREADS", str(4))),
            "max_request_items": int(os.getenv("MAX_REQUEST_ITEMS", str(64))),
            "max_request_body_bytes": int(os.getenv("MAX_REQUEST_BODY_BYTES", str(64 * 1024 * 1024))),
            "max_image_bytes": int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024))),
            "max_image_pixels": int(os.getenv("MAX_IMAGE_PIXELS", str(20_000_000))),
            "rec_batch_size": int(os.getenv("REC_BATCH_SIZE", str(6))),
            "pdf_dpi": int(os.getenv("PDF_DPI", str(150))),
        }
        settings = cls(**values)
        settings.validate()
        return settings

    def validate(self):
        if self.model_id != "rapid-v5-mobile":
            raise ValueError("MODEL_ID must be rapid-v5-mobile for this service")
        if self.device not in {"cuda", "cpu"}:
            raise ValueError("DEVICE must be cuda or cpu")
        for name in (
            "max_batch_size",
            "max_batch_tokens",
            "max_queue_items",
            "workers",
            "cpu_threads",
            "max_request_items",
            "max_request_body_bytes",
            "max_image_bytes",
            "max_image_pixels",
            "rec_batch_size",
            "pdf_dpi",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.batch_wait_ms < 0:
            raise ValueError("batch_wait_ms must be non-negative")
        if not math.isfinite(self.request_timeout_s) or self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be finite and positive")
        if not self.model_dir:
            raise ValueError("model_dir must be non-empty")

    @property
    def model(self):
        return self.model_id
