"""HTTP contract tests for the standalone CPU GigaAM and EasyOCR services."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from concurrent.futures import Future
from pathlib import Path
from types import ModuleType

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


class FakeWorker:
    """Small worker double that never imports or loads an ML backend."""

    instances: list["FakeWorker"] = []
    result: str | list[str] = "stub transcript"

    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.load_attempts = 0
        self._submitted: list[str] = []
        self.__class__.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    @property
    def loaded(self) -> bool:
        return False

    @property
    def queue_size(self) -> int:
        return 0

    def submit(self, path: str) -> Future:
        self._submitted.append(path)
        future = Future()
        future.set_result(self.result)
        Path(path).unlink(missing_ok=True)
        return future


def load_app(monkeypatch: pytest.MonkeyPatch, service: str):
    service_dir = ROOT / ("asr/gigaam" if service == "gigaam" else "ocr/easyocr")
    worker_module_name = "transcriber" if service == "gigaam" else "ocr"
    worker_class_name = "TranscriberWorker" if service == "gigaam" else "OcrWorker"

    # The legacy services use flat imports because each folder is a standalone
    # Docker build context. Clear those names between imports to avoid cross-app
    # module reuse in this test process.
    for name in ("app", "config", "schemas", worker_module_name):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.syspath_prepend(str(service_dir))

    FakeWorker.instances = []
    FakeWorker.result = "stub transcript" if service == "gigaam" else ["stub text"]
    backend_stub = ModuleType(worker_module_name)
    setattr(backend_stub, worker_class_name, FakeWorker)
    monkeypatch.setitem(sys.modules, worker_module_name, backend_stub)

    return importlib.import_module("app")


@pytest.mark.parametrize(
    ("service", "named_route", "legacy_route", "filename", "expected"),
    [
        ("gigaam", "/gigaam/transcribe", "/transcribe", "sample.wav", {"text": "stub transcript"}),
        (
            "easyocr",
            "/easyocr/ocr",
            "/ocr",
            "sample.png",
            {"langs": ["ru", "en"], "pages": 1, "text": "stub text", "page_texts": ["stub text"]},
        ),
    ],
)
def test_named_and_legacy_inference_routes(
    monkeypatch: pytest.MonkeyPatch,
    service: str,
    named_route: str,
    legacy_route: str,
    filename: str,
    expected: dict,
) -> None:
    module = load_app(monkeypatch, service)

    with TestClient(module.app) as client:
        named = client.post(named_route, files={"file": (filename, b"fake input")})
        legacy = client.post(legacy_route, files={"file": (filename, b"fake input")})

    assert named.status_code == legacy.status_code == 200
    assert named.json() == legacy.json() == expected
    assert len(FakeWorker.instances[0]._submitted) == 2
    assert FakeWorker.instances[0].started and FakeWorker.instances[0].stopped

    # The new route is the documented route; compatibility aliases stay out of
    # generated API docs while continuing to serve existing clients.
    schema = module.app.openapi()
    assert named_route in schema["paths"]
    assert legacy_route not in schema["paths"]


@pytest.mark.parametrize(
    ("service", "named_route", "legacy_route", "expected_model"),
    [
        ("gigaam", "/gigaam/health", "/health", "gigaam-v3-ctc"),
        ("easyocr", "/easyocr/health", "/health", "ru,en"),
    ],
)
def test_health_aliases_do_not_load_the_model(
    monkeypatch: pytest.MonkeyPatch,
    service: str,
    named_route: str,
    legacy_route: str,
    expected_model: str,
) -> None:
    module = load_app(monkeypatch, service)

    with TestClient(module.app) as client:
        named = client.get(named_route)
        legacy = client.get(legacy_route)

    assert named.status_code == legacy.status_code == 200
    assert named.json() == legacy.json()
    assert named.json() == {
        "status": "ok",
        "model": expected_model,
        "loaded": False,
        "queue": 0,
    }
    worker = FakeWorker.instances[0]
    assert worker.load_attempts == 0
    assert worker._submitted == []


@pytest.mark.parametrize(
    ("service", "prefix", "port"),
    [("gigaam", "GIGAAM", 8000), ("easyocr", "EASYOCR", 8000)],
)
def test_entrypoint_uses_configured_worker_count(
    monkeypatch: pytest.MonkeyPatch, service: str, prefix: str, port: int
) -> None:
    entrypoint_path = ROOT / ("asr/gigaam" if service == "gigaam" else "ocr/easyocr") / "__main__.py"
    spec = importlib.util.spec_from_file_location(f"{service}_entrypoint_test", entrypoint_path)
    assert spec and spec.loader
    entrypoint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entrypoint)

    calls: list[dict] = []
    monkeypatch.setattr(entrypoint.uvicorn, "run", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setenv(f"{prefix}_WORKERS", "3")
    monkeypatch.setenv(f"{prefix}_PORT", str(port + 1))

    entrypoint.main()

    assert calls == [{"host": "0.0.0.0", "port": port, "workers": 3}]


@pytest.mark.parametrize(("service", "prefix"), [("gigaam", "GIGAAM"), ("easyocr", "EASYOCR")])
def test_entrypoint_rejects_nonpositive_worker_count(
    monkeypatch: pytest.MonkeyPatch, service: str, prefix: str
) -> None:
    entrypoint_path = ROOT / ("asr/gigaam" if service == "gigaam" else "ocr/easyocr") / "__main__.py"
    spec = importlib.util.spec_from_file_location(f"{service}_entrypoint_invalid_test", entrypoint_path)
    assert spec and spec.loader
    entrypoint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entrypoint)
    monkeypatch.setenv(f"{prefix}_WORKERS", "0")

    with pytest.raises(ValueError, match="positive integer"):
        entrypoint.main()


@pytest.mark.parametrize(("service", "prefix"), [("gigaam", "GIGAAM"), ("easyocr", "EASYOCR")])
def test_legacy_env_fallback_and_prefixed_priority(monkeypatch, service, prefix):
    # Existing local .env files survive the directory move unchanged.
    monkeypatch.delenv(f"{prefix}_IDLE_TTL", raising=False)
    monkeypatch.setenv("IDLE_TTL", "19")
    load_app(monkeypatch, service)
    config = sys.modules["config"]
    assert config.Settings.from_env().idle_ttl == 19
    monkeypatch.setenv(f"{prefix}_IDLE_TTL", "23")
    assert config.Settings.from_env().idle_ttl == 23
