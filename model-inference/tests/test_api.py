import asyncio
import base64
import io

from fastapi.testclient import TestClient
from PIL import Image

from inference.app import create_app
from inference.settings import Settings


class FakeBackend:
    def infer(self, items):
        results = []
        for item in items:
            if isinstance(item, dict) and "text" in item:
                results.append([len(item["text"]), 2.0])
            elif isinstance(item, dict) and "audio" in item:
                results.append(
                    {"text": "heard", "language": item["language"] or "ru", "segments": []}
                )
            elif isinstance(item, bytes):
                results.append({"text": f"image:{len(item)}", "lines": []})
            else:
                raise AssertionError(f"unexpected item {type(item)}")
        return results

    def metadata(self):
        model = (
            "deepvk/USER-bge-m3"
            if self.model_id == "user-bge-m3"
            else {
                "e5-small": "intfloat/multilingual-e5-small",
                "rapid-v5-mobile": "rapid-v5-mobile",
                "whisper-large-v3": "Systran/faster-whisper-large-v3",
            }[self.model_id]
        )
        return {"weights": model}


def client(model_id="e5-small", **overrides):
    settings = Settings(device="cpu", model_id=model_id, **overrides)

    def factory(config):
        backend = FakeBackend()
        backend.model_id = config.model_id
        return backend

    return TestClient(create_app(settings, factory))


def png_bytes(size=(4, 3)):
    output = io.BytesIO()
    Image.new("RGB", size).save(output, format="PNG")
    return output.getvalue()


def test_model_named_embeddings_and_health():
    with client() as c:
        assert c.get("/healthz").json() == {"status": "ok"}
        assert c.get("/readyz").json() == {"status": "ready"}
        health = c.get("/e5-small/health").json()
        assert health == {
            "status": "ok",
            "model": "e5-small",
            "loaded": True,
            "queue": 0,
        }
        result = c.post("/e5-small/embed", json={"texts": ["one", "two"], "role": "passage"})
        assert result.status_code == 200
        assert result.json()["model"] == "e5-small"
        assert result.json()["dim"] == 2
        assert result.json()["count"] == 2
        assert result.json()["embeddings"] == [[3, 2.0], [3, 2.0]]
        assert c.post("/user-bge-m3/embed", json={"texts": ["x"]}).status_code == 404
        assert c.post("/e5-small/embed", json={"texts": ["x"], "role": "other"}).status_code == 422


def test_ocr_json_batch_multipart_image_and_pdf():
    encoded = base64.b64encode(png_bytes()).decode()
    with client(model_id="rapid-v5-mobile") as c:
        batch = c.post("/rapid-v5-mobile/ocr/batch", json={"images": [encoded, encoded]})
        assert batch.status_code == 200
        assert len(batch.json()["pages"]) == 2
        assert c.post("/e5-small/embed", json={"texts": ["x"]}).status_code == 404
        image = c.post(
            "/rapid-v5-mobile/ocr", files={"file": ("test.png", png_bytes(), "image/png")}
        )
        assert image.status_code == 200
        assert image.json()["count"] == 1
        assert image.json()["pages"][0]["lines"] == []
        assert image.json()["text"].startswith("image:")
        pdf_output = io.BytesIO()
        Image.new("RGB", (8, 8), "white").save(pdf_output, format="PDF")
        pdf = c.post(
            "/rapid-v5-mobile/ocr",
            files={"file": ("test.pdf", pdf_output.getvalue(), "application/pdf")},
        )
        assert pdf.status_code == 200, pdf.text
        assert pdf.json()["count"] == 1
        assert pdf.json()["pages"][0]["lines"] == []
        assert c.post("/rapid-v5-mobile/ocr/batch", json={"images": ["%%%"]}).status_code == 422


def test_whisper_single_and_batch_form_fields():
    with client(model_id="whisper-large-v3") as c:
        one = c.post(
            "/whisper-large-v3/transcribe", files={"file": ("one.wav", b"audio", "audio/wav")}
        )
        assert one.status_code == 200
        assert one.json()["text"] == "heard"
        assert one.json()["metadata"]["weights"] == "Systran/faster-whisper-large-v3"
        batch = c.post(
            "/whisper-large-v3/transcribe/batch",
            files=[
                ("files", ("one.wav", b"a", "audio/wav")),
                ("files", ("two.wav", b"b", "audio/wav")),
            ],
            data={"language": "en", "task": "translate"},
        )
        assert batch.status_code == 200
        assert len(batch.json()["results"]) == 2
        assert batch.json()["results"][0]["language"] == "en"
        assert c.post("/e5-small/embed", json={"texts": ["x"]}).status_code == 404
        bad = c.post(
            "/whisper-large-v3/transcribe",
            files={"file": ("one.wav", b"audio", "audio/wav")},
            data={"task": "summarize"},
        )
        assert bad.status_code == 422


def test_settings_validate_model_audio_limits_and_variant():
    Settings(model_id="whisper-large-v3", device="cpu").validate()
    for settings in (
        Settings(max_request_body_bytes=0),
        Settings(max_audio_bytes=0),
        Settings(max_audio_duration_s=float("inf")),
    ):
        try:
            settings.validate()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid setting should be rejected")
    try:
        Settings(device="cuda", user_variant="int8").validate()
    except ValueError as exc:
        assert "cpu" in str(exc).lower()
    else:
        raise AssertionError("int8 on CUDA should be rejected")


def test_streamed_body_limit_without_content_length():
    app = create_app(Settings(device="cpu", max_request_body_bytes=8), lambda _: FakeBackend())

    async def invoke():
        events = iter(
            [
                {"type": "http.request", "body": b"1234", "more_body": True},
                {"type": "http.request", "body": b"56789", "more_body": False},
            ]
        )
        sent = []

        async def receive():
            return next(events)

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/e5-small/embed",
            "raw_path": b"/e5-small/embed",
            "query_string": b"",
            "root_path": "",
            "headers": [(b"content-type", b"application/json")],
            "client": ("test", 1),
            "server": ("test", 80),
        }
        await app(scope, receive, send)
        return sent

    messages = asyncio.run(invoke())
    assert messages[0]["status"] == 413
