import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

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
        total = 0

        async def limited_receive():
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.max_body:
                    # FastAPI handles this HTTPException before inference starts.
                    # Multipart parsing can spool files without buffering the
                    # whole upload a second time in this middleware.
                    raise HTTPException(413, "request body too large")
            return message

        await self.app(scope, limited_receive, send)

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


def create_app(settings=None, backend_factory=None):
    settings = settings or Settings.from_env()
    settings.validate()
    if backend_factory is None:

        def backend_factory(config):
            from .backend import create_backend

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

    app = FastAPI(title="Whisper inference service", lifespan=lifespan)
    app.state.ready = False

    app.add_middleware(_RequestBodyLimitMiddleware, max_body=settings.max_request_body_bytes)

    @app.exception_handler(QueueFull)
    async def queue_full(_, __):
        return JSONResponse({"detail": "inference queue is full"}, status_code=429)

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

    @app.get("/whisper-large-v3/health")
    async def model_health():
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

    @app.post("/whisper-large-v3/transcribe")
    async def transcribe(
        file: UploadFile = File(...),
        language: str | None = Form("ru"),
        task: str = Form("transcribe"),
    ):
        result = (await transcribe_files([file], language, task))[0]
        return {**result, "metadata": await asyncio.to_thread(app.state.backend.metadata)}

    @app.post("/whisper-large-v3/transcribe/batch")
    async def transcribe_batch(
        files: list[UploadFile] = File(...),
        language: str | None = Form("ru"),
        task: str = Form("transcribe"),
    ):
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
