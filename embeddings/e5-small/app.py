import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
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


class EmbeddingBody(BaseModel):
    texts: list[str]
    role: str = "query"


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

    app = FastAPI(title="E5 small embedding service", lifespan=lifespan)
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

    return app


app = create_app()
