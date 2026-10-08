import asyncio
import base64
import io

from fastapi.testclient import TestClient
from PIL import Image

from conftest import load_service


class FakeBackend:
    def __init__(self, model_id):
        self.model_id = model_id

    def infer(self, items):
        results = []
        for item in items:
            if isinstance(item, dict) and "text" in item:
                results.append([len(item["text"]), 2.0])
            elif isinstance(item, dict) and "audio" in item:
                results.append({"text": "heard", "language": item["language"] or "ru", "segments": []})
            elif isinstance(item, bytes):
                results.append({"text": f"image:{len(item)}", "lines": []})
            else:
                raise AssertionError(f"unexpected item {type(item)}")
        return results

    def metadata(self):
        return {"weights": self.model_id}


def client(model_id, **overrides):
    service = load_service(model_id)
    settings = service.settings.Settings(device="cpu", **overrides)
    backend = FakeBackend(model_id)
    return TestClient(service.app.create_app(settings, lambda _: backend))


def png_bytes(size=(4, 3)):
    output = io.BytesIO()
    Image.new("RGB", size).save(output, format="PNG")
    return output.getvalue()


def test_each_application_registers_only_its_model_inference_routes():
    models = {
        "e5-small": ("/{route_model}/embed",),
        "user-bge-m3": ("/{route_model}/embed",),
        "rapid-v5-mobile": ("/rapid-v5-mobile/ocr", "/rapid-v5-mobile/ocr/batch"),
        "whisper-large-v3": (
            "/whisper-large-v3/transcribe",
            "/whisper-large-v3/transcribe/batch",
        ),
    }
    for model_id, expected in models.items():
        service = load_service(model_id)
        settings = service.settings.Settings(device="cpu")
        app = service.app.create_app(settings, lambda _: FakeBackend(model_id))
        paths = {route.path for route in app.routes}
        assert set(expected) <= paths
        assert app.title == {
            "e5-small": "E5 small embedding service",
            "user-bge-m3": "USER BGE M3 embedding service",
            "rapid-v5-mobile": "Rapid OCR inference service",
            "whisper-large-v3": "Whisper inference service",
        }[model_id]
        all_inference_paths = {
            path for values in models.values() for path in values
        }
        assert paths & all_inference_paths == set(expected)


def test_embedding_http_behavior_and_model_isolation():
    with client("e5-small") as c:
        assert c.get("/healthz").json() == {"status": "ok"}
        assert c.get("/readyz").json() == {"status": "ready"}
        assert c.get("/e5-small/health").json()["model"] == "e5-small"
        response = c.post("/e5-small/embed", json={"texts": ["one", "two"], "role": "passage"})
        assert response.status_code == 200
        assert response.json()["embeddings"] == [[3, 2.0], [3, 2.0]]
        assert response.json()["count"] == 2
        assert c.post("/user-bge-m3/embed", json={"texts": ["x"]}).status_code == 404
        assert c.post("/e5-small/embed", json={"texts": ["x"], "role": "other"}).status_code == 422
    with client("user-bge-m3") as c:
        assert c.post("/user-bge-m3/embed", json={"texts": ["x"]}).status_code == 200
        assert c.post("/e5-small/embed", json={"texts": ["x"]}).status_code == 404


def test_ocr_batch_file_and_validation():
    encoded = base64.b64encode(png_bytes()).decode()
    with client("rapid-v5-mobile") as c:
        batch = c.post("/rapid-v5-mobile/ocr/batch", json={"images": [encoded, encoded]})
        assert batch.status_code == 200 and len(batch.json()["pages"]) == 2
        assert c.post("/e5-small/embed", json={"texts": ["x"]}).status_code == 404
        image = c.post("/rapid-v5-mobile/ocr", files={"file": ("x.png", png_bytes(), "image/png")})
        assert image.status_code == 200 and image.json()["count"] == 1
        assert c.post("/rapid-v5-mobile/ocr/batch", json={"images": ["%%%"]}).status_code == 422


def test_whisper_single_batch_and_validation():
    with client("whisper-large-v3") as c:
        one = c.post("/whisper-large-v3/transcribe", files={"file": ("one.wav", b"audio", "audio/wav")})
        assert one.status_code == 200 and one.json()["text"] == "heard"
        batch = c.post(
            "/whisper-large-v3/transcribe/batch",
            files=[("files", ("one.wav", b"a", "audio/wav")), ("files", ("two.wav", b"b", "audio/wav"))],
            data={"language": "en", "task": "translate"},
        )
        assert batch.status_code == 200 and len(batch.json()["results"]) == 2
        assert batch.json()["results"][0]["language"] == "en"
        invalid = c.post(
            "/whisper-large-v3/transcribe", files={"file": ("x.wav", b"a", "audio/wav")}, data={"task": "summarize"}
        )
        assert invalid.status_code == 422
        assert c.post("/e5-small/embed", json={"texts": ["x"]}).status_code == 404


def test_settings_validate_only_own_model_and_constraints():
    settings = load_service("whisper-large-v3").settings.Settings(device="cpu")
    settings.validate()
    try:
        load_service("e5-small").settings.Settings(model_id="user-bge-m3").validate()
    except ValueError as exc:
        assert "e5-small" in str(exc)
    else:
        raise AssertionError("a model service must reject another model ID")
    try:
        load_service("user-bge-m3").settings.Settings(device="cuda", user_variant="int8").validate()
    except ValueError as exc:
        assert "cpu" in str(exc).lower()
    else:
        raise AssertionError("int8 user model mode must require CPU")
    for settings in (
        load_service("e5-small").settings.Settings(max_request_body_bytes=0),
        load_service("whisper-large-v3").settings.Settings(max_audio_bytes=0),
        load_service("whisper-large-v3").settings.Settings(max_audio_duration_s=float("inf")),
    ):
        try:
            settings.validate()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid setting should be rejected")


def test_streamed_request_body_limit():
    service = load_service("e5-small")
    app = service.app.create_app(
        service.settings.Settings(device="cpu", max_request_body_bytes=8), lambda _: FakeBackend("e5-small")
    )

    async def invoke():
        events = iter([
            {"type": "http.request", "body": b"1234", "more_body": True},
            {"type": "http.request", "body": b"56789", "more_body": False},
        ])
        sent = []
        async def receive():
            return next(events)
        async def send(message):
            sent.append(message)
        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
            "scheme": "http", "path": "/e5-small/embed", "raw_path": b"/e5-small/embed", "query_string": b"",
            "root_path": "", "headers": [(b"content-type", b"application/json")],
            "client": ("test", 1), "server": ("test", 80),
        }
        await app(scope, receive, send)
        return sent

    assert asyncio.run(invoke())[0]["status"] == 413


def test_entrypoint_prepares_once_before_starting_uvicorn(monkeypatch):
    import importlib

    for model_id in ("e5-small", "user-bge-m3", "rapid-v5-mobile", "whisper-large-v3"):
        package = load_service(model_id)
        prepare = importlib.import_module(package.__name__ + ".prepare")
        entrypoint = importlib.import_module(package.__name__ + ".__main__")
        events = []
        monkeypatch.setattr(prepare, "ensure_prepared", lambda settings: events.append(("prepare", settings.model_id)))
        monkeypatch.setattr(entrypoint.uvicorn, "run", lambda *args, **kwargs: events.append(("run", kwargs["workers"])))
        entrypoint.main()
        assert events == [("prepare", model_id), ("run", 1)]
