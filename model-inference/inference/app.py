import asyncio
import base64
import binascii
import io
import math
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .batching import BatchScheduler, QueueFull
from .settings import Settings


class _RequestBodyLimitMiddleware:
    def __init__(self, app, max_body):
        self.app = app
        self.max_body = max_body

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared = 0
        if declared > self.max_body:
            await self._too_large(send)
            return
        chunks = []
        total = 0
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > self.max_body:
                await self._too_large(send)
                return
            chunks.append(chunk)
            more_body = message.get("more_body", False)
        body = b"".join(chunks)
        sent = False

        async def replay_receive():
            nonlocal sent
            if sent:
                return {"type": "http.request", "body": b"", "more_body": False}
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay_receive, send)

    @staticmethod
    async def _too_large(send):
        body = b'{"detail":"request body too large"}'
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def _validate_image(raw, limit):
    from PIL import Image

    try:
        with Image.open(io.BytesIO(raw)) as image:
            if getattr(image, "is_animated", False):
                raise ValueError("animated images are not supported")
            if image.format not in {"PNG", "JPEG", "WEBP", "TIFF", "BMP"}:
                raise ValueError("unsupported image format")
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > limit:
                raise ValueError("image dimensions exceed limit")
            image.verify()
        return raw
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("invalid image") from exc


class EmbeddingBody(BaseModel):
    texts: list[str]
    role: str = "query"


class OCRBody(BaseModel):
    images: list[str]


def _render_pdf(raw, settings):
    import pypdfium2 as pdfium
    import pypdfium2.raw as pdfium_raw

    try:
        document = pdfium.PdfDocument(raw)
    except Exception as exc:
        raise ValueError("invalid or encrypted PDF") from exc
    images = []
    total_bytes = 0
    try:
        if pdfium_raw.FPDF_GetSecurityHandlerRevision(document.raw) >= 0:
            raise ValueError("encrypted PDFs are not supported")
        count = len(document)
        if count < 1:
            raise ValueError("PDF has no pages")
        if count > settings.max_request_items:
            raise OverflowError("PDF has too many pages")
        scale = settings.pdf_dpi / 72.0
        for index in range(count):
            page = document[index]
            width_pt, height_pt = page.get_size()
            width, height = math.ceil(width_pt * scale), math.ceil(height_pt * scale)
            if width <= 0 or height <= 0 or width * height > settings.max_image_pixels:
                raise OverflowError(f"PDF page {index + 1} exceeds pixel limit")
            rendered = page.render(scale=scale).to_pil().convert("RGB")
            output = io.BytesIO()
            rendered.save(output, format="PNG")
            image_bytes = output.getvalue()
            total_bytes += len(image_bytes)
            if len(image_bytes) > settings.max_image_bytes:
                raise OverflowError(f"rendered PDF page {index + 1} exceeds byte limit")
            if total_bytes > settings.max_request_body_bytes:
                raise OverflowError("rendered PDF exceeds total byte limit")
            images.append(image_bytes)
            page.close()
        return images
    except (ValueError, OverflowError):
        raise
    except Exception as exc:
        raise ValueError("malformed PDF") from exc
    finally:
        document.close()


def create_app(settings=None, backend_factory=None):
    settings = settings or Settings.from_env()
    settings.validate()
    if backend_factory is None:

        def backend_factory(config):
            from .backends import create_backend

            return create_backend(config)

    @asynccontextmanager
    async def lifespan(app):
        backend = await asyncio.to_thread(backend_factory, settings)
        app.state.backend = backend
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference-backend")

        async def infer(items):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(executor, backend.infer, items)

        app.state.scheduler = BatchScheduler(
            infer,
            settings.max_batch_size,
            settings.max_queue_items,
            settings.batch_wait_ms,
            settings.max_batch_tokens,
        )
        await app.state.scheduler.start()
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False
            await app.state.scheduler.shutdown()
            executor.shutdown(wait=True, cancel_futures=False)

    app = FastAPI(lifespan=lifespan)
    app.state.ready = False

    app.add_middleware(_RequestBodyLimitMiddleware, max_body=settings.max_request_body_bytes)

    @app.exception_handler(QueueFull)
    async def queue_full(_, __):
        return JSONResponse({"detail": "inference queue is full"}, status_code=429)

    def check_route_model(route_model, allowed):
        if route_model not in allowed or settings.model_id != route_model:
            raise HTTPException(404, "model route does not match this service")

    def check_role(role):
        if role not in {"query", "passage"}:
            raise HTTPException(422, "role must be query or passage")

    def check_texts(texts):
        if not texts:
            raise HTTPException(422, "input must contain at least one item")
        if len(texts) > settings.max_request_items:
            raise HTTPException(422, "too many input items")
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise HTTPException(422, "texts must be non-empty strings")
            if len(text) > settings.max_text_chars:
                raise HTTPException(422, "text exceeds maximum length")

    async def submit(items):
        if not app.state.ready:
            raise HTTPException(503, "inference service is not ready")
        try:
            return await app.state.scheduler.submit(items, settings.request_timeout_s)
        except asyncio.TimeoutError as exc:
            raise HTTPException(504, "inference request timed out") from exc
        except QueueFull as exc:
            raise HTTPException(429, "inference queue is full") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, "inference service unavailable") from exc
        except Exception as exc:
            raise HTTPException(503, "inference backend failed") from exc

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    @app.get("/readyz")
    async def ready():
        if not app.state.ready:
            raise HTTPException(503, "model is not ready")
        return {"status": "ready"}

    @app.get("/{route_model}/health")
    async def model_health(route_model: str):
        check_route_model(route_model, {settings.model_id})
        return {
            "status": "ok",
            "model": settings.model_id,
            "loaded": bool(app.state.ready),
            "queue": app.state.scheduler.stats["queued_items"] if app.state.ready else 0,
        }

    @app.get("/v1/models")
    async def models():
        if not app.state.ready:
            raise HTTPException(503, "model is not ready")
        metadata = await asyncio.to_thread(app.state.backend.metadata)
        return {
            "object": "list",
            "data": [
                {
                    "id": settings.model_id,
                    "object": "model",
                    "metadata": metadata,
                    "worker_pid": os.getpid(),
                    "queue": app.state.scheduler.stats,
                }
            ],
        }

    @app.post("/{route_model}/embed")
    async def embeddings(route_model: str, body: EmbeddingBody):
        check_route_model(route_model, {"e5-small", "user-bge-m3"})
        check_role(body.role)
        check_texts(body.texts)
        values = await submit([{"text": text, "role": body.role} for text in body.texts])
        metadata = await asyncio.to_thread(app.state.backend.metadata)
        return {
            "embeddings": values,
            "model": settings.model_id,
            "dim": len(values[0]) if values else 0,
            "count": len(values),
            "metadata": metadata,
        }

    async def validate_ocr_batch(body: OCRBody):
        if not body.images or len(body.images) > settings.max_request_items:
            raise HTTPException(413, "images must contain between 1 and the configured maximum")
        pages = []
        for encoded in body.images:
            try:
                raw = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError, TypeError) as exc:
                raise HTTPException(422, "images must be valid base64") from exc
            if len(raw) > settings.max_image_bytes:
                raise HTTPException(413, "image exceeds maximum byte size")
            try:
                pages.append(
                    await asyncio.to_thread(_validate_image, raw, settings.max_image_pixels)
                )
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
        result = await submit(pages)
        return {"model": settings.model_id, "pages": result}

    @app.post("/rapid-v5-mobile/ocr/batch")
    async def ocr_batch(body: OCRBody):
        check_route_model("rapid-v5-mobile", {"rapid-v5-mobile"})
        return await validate_ocr_batch(body)

    @app.post("/rapid-v5-mobile/ocr")
    async def ocr_file(
        file: UploadFile = File(...),
    ):
        check_route_model("rapid-v5-mobile", {"rapid-v5-mobile"})
        raw = await file.read(settings.max_request_body_bytes + 1)
        await file.close()
        if len(raw) > settings.max_request_body_bytes:
            raise HTTPException(413, "uploaded file exceeds request body limit")
        is_pdf = (file.filename or "").lower().endswith(".pdf") or raw.startswith(b"%PDF-")
        try:
            if is_pdf:
                pages = await asyncio.to_thread(_render_pdf, raw, settings)
            else:
                if len(raw) > settings.max_image_bytes:
                    raise OverflowError("image exceeds maximum byte size")
                pages = [await asyncio.to_thread(_validate_image, raw, settings.max_image_pixels)]
        except OverflowError as exc:
            raise HTTPException(413, str(exc)) from exc
        except ValueError as exc:
            if not is_pdf and "unsupported image format" in str(exc):
                raise HTTPException(415, str(exc)) from exc
            raise HTTPException(422, str(exc)) from exc
        page_results = await submit(pages)
        page_texts = [r.get("text", "") if isinstance(r, dict) else str(r) for r in page_results]
        return {
            "model": settings.model_id,
            "pages": page_results,
            "count": len(page_results),
            "text": "\n\n".join(page_texts),
        }

    @app.post("/whisper-large-v3/transcribe")
    async def transcribe(
        file: UploadFile = File(...),
        language: str | None = Form("ru"),
        task: str = Form("transcribe"),
    ):
        check_route_model("whisper-large-v3", {"whisper-large-v3"})
        result = (await transcribe_files([file], language, task))[0]
        return {**result, "metadata": await asyncio.to_thread(app.state.backend.metadata)}

    @app.post("/whisper-large-v3/transcribe/batch")
    async def transcribe_batch(
        files: list[UploadFile] = File(...),
        language: str | None = Form("ru"),
        task: str = Form("transcribe"),
    ):
        check_route_model("whisper-large-v3", {"whisper-large-v3"})
        if not files or len(files) > settings.max_request_items:
            raise HTTPException(413, "files must contain between 1 and the configured maximum")
        results = await transcribe_files(files, language, task)
        return {
            "model": "whisper-large-v3",
            "results": results,
            "metadata": await asyncio.to_thread(app.state.backend.metadata),
        }

    async def transcribe_files(files, language, task):
        if task not in {"transcribe", "translate"}:
            raise HTTPException(422, "task must be transcribe or translate")
        if language == "auto":
            language = None
        if language is not None and (
            len(language) not in (2, 3) or not language.isalpha() or not language.islower()
        ):
            raise HTTPException(422, "language must be a lowercase ISO 639 code or auto")
        items = []
        for upload in files:
            audio = await upload.read(settings.max_audio_bytes + 1)
            await upload.close()
            if len(audio) > settings.max_audio_bytes:
                raise HTTPException(413, "audio exceeds maximum byte size")
            if not audio:
                raise HTTPException(422, "audio file is empty")
            items.append({"audio": audio, "language": language, "task": task})
        return await submit(items)

    return app


app = create_app()
