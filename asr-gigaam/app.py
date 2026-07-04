"""
ASR-сервис на GigaAM: HTTP-слой (FastAPI).

Вся тяжёлая работа — в TranscriberWorker (transcriber.py): один поток, очередь,
выгрузка модели по простою. Здесь дело простое — принять файл, поставить в
очередь, дождаться текста. Настройки в config.py, схемы ответов в schemas.py.

После запуска живая документация — на `/docs` (Swagger UI) и `/redoc`, сырая
OpenAPI-схема — на `/openapi.json`.

Запуск (ровно один воркер uvicorn — это важно, см. README):
    uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
"""

import asyncio
import os
import shutil
import tempfile
from contextlib import asynccontextmanager

from config import settings
from fastapi import FastAPI, File, HTTPException, UploadFile
from schemas import HealthResponse, TranscribeResponse
from transcriber import TranscriberWorker

worker = TranscriberWorker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Старт: заводим временную папку и запускаем воркер.
    os.makedirs(settings.tmp_dir, exist_ok=True)
    worker.start()
    yield
    # Стоп: гасим воркер и ждём, пока он завершится.
    worker.stop()


app = FastAPI(
    title="GigaAM ASR (ONNX, CPU)",
    description=(
        "Распознавание речи на GigaAM. Модель в ONNX, инференс на CPU, запросы "
        "обрабатываются по очереди одним воркером, модель выгружается из RAM по "
        "простою и поднимается обратно по запросу."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Проверка состояния сервиса",
    description=(
        "Возвращает статус процесса, версию модели, загружена ли модель в RAM "
        "и длину очереди. Не поднимает модель — безопасно дёргать как healthcheck."
    ),
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model=settings.model_version,
        loaded=worker.loaded,
        queue=worker.queue_size,
    )


@app.post(
    "/transcribe",
    response_model=TranscribeResponse,
    summary="Распознать речь из аудиофайла",
    description=(
        "Принимает аудиофайл (wav/mp3/mp4/m4a/flac/ogg/opus/aac/webm/wma), "
        "ставит его в очередь и возвращает распознанный текст. Первый запрос "
        "после простоя дольше обычного — поднимается модель (cold start)."
    ),
    responses={
        415: {"description": "Неподдерживаемый формат файла"},
        504: {"description": "Истекло время ожидания результата (REQUEST_TIMEOUT)"},
        500: {"description": "Ошибка инференса"},
    },
)
async def transcribe(file: UploadFile = File(...)) -> TranscribeResponse:
    suffix = os.path.splitext(file.filename or "")[1].lower() or ".wav"
    if suffix not in settings.accepted_exts:
        raise HTTPException(
            status_code=415,
            detail=f"Неподдерживаемый формат '{suffix}'. Принимаем: "
            + ", ".join(sorted(settings.accepted_exts)),
        )

    # Пишем загрузку на диск потоком, не держа целиком в RAM, во временный файл
    # с уникальным именем. Дальше его удалит воркер — аудио мы не храним, и
    # декодирует (mp3/mp4 через ffmpeg) тоже он. Запись уводим в to_thread:
    # copyfileobj синхронный и на большом файле подвесил бы event loop.
    with tempfile.NamedTemporaryFile(suffix=suffix, dir=settings.tmp_dir, delete=False) as tmp:
        await asyncio.to_thread(shutil.copyfileobj, file.file, tmp)
        audio_path = tmp.name

    future = worker.submit(audio_path)

    # Ждём результат, не подвешивая event loop.
    try:
        text = await asyncio.wait_for(asyncio.wrap_future(future), timeout=settings.request_timeout)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504, detail="Время ожидания распознавания истекло"
        ) from None
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return TranscribeResponse(text=text)
