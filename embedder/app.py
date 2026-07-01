"""
Сервис эмбеддингов текста: HTTP-слой (FastAPI).

Кодирование выполняет EmbedderWorker (см. embedder.py) — один поток с очередью
запросов и выгрузкой модели по простою. Здесь только приём текстов, постановка в
очередь и ожидание результата. Конфиг — в config.py, контракты API — в schemas.py.

Интерактивная документация API доступна после запуска на `/docs` (Swagger UI)
и `/redoc`, машиночитаемая схема — на `/openapi.json`.

Запуск (один воркер uvicorn — это принципиально, см. README):
    uvicorn app:app --host 0.0.0.0 --port 8001 --workers 1
"""

import asyncio
from contextlib import asynccontextmanager

from config import settings
from fastapi import FastAPI, HTTPException
from schemas import EmbedRequest, EmbedResponse, HealthResponse

from embedder import EmbedderWorker

worker = EmbedderWorker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Старт: поднимаем поток-воркер (модель грузится лениво по первому запросу).
    worker.start()
    yield
    # Остановка: гасим воркер (кладём «отравленную пилюлю» и ждём завершения).
    worker.stop()


app = FastAPI(
    title="Text Embedder (sentence-transformers, CPU)",
    description=(
        "Кодирование текста в векторы на sentence-transformers. Инференс на CPU, "
        "запросы обрабатываются по очереди одним воркером, модель выгружается из "
        "RAM по простою и поднимается обратно по запросу."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Проверка состояния сервиса",
    description=(
        "Возвращает статус процесса, имя модели, загружена ли модель в RAM "
        "и длину очереди. Не поднимает модель — безопасно дёргать как healthcheck."
    ),
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model=settings.model_name,
        loaded=worker.loaded,
        queue=worker.queue_size,
    )


@app.post(
    "/embed",
    response_model=EmbedResponse,
    summary="Закодировать тексты в векторы",
    description=(
        "Принимает список текстов, ставит их в очередь и возвращает эмбеддинги "
        "(по одному вектору на текст). Первый запрос после простоя дольше "
        "обычного — поднимается модель (cold start)."
    ),
    responses={
        422: {"description": "Пустой список или превышен лимит MAX_TEXTS"},
        504: {"description": "Истекло время ожидания результата (REQUEST_TIMEOUT)"},
        500: {"description": "Ошибка инференса"},
    },
)
async def embed(req: EmbedRequest) -> EmbedResponse:
    if len(req.texts) > settings.max_texts:
        raise HTTPException(
            status_code=422,
            detail=f"Слишком много текстов: {len(req.texts)} > {settings.max_texts} "
            "(лимит MAX_TEXTS)",
        )

    future = worker.submit(req.texts)

    # Ждём результат, не блокируя event loop.
    try:
        vectors = await asyncio.wait_for(
            asyncio.wrap_future(future), timeout=settings.request_timeout
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Время ожидания кодирования истекло") from None
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return EmbedResponse(
        model=settings.model_name,
        dim=len(vectors[0]) if vectors else 0,
        count=len(vectors),
        embeddings=vectors,
    )
