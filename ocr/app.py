"""
OCR-сервис на EasyOCR (torch, CPU): HTTP-слой (FastAPI).

Вся тяжёлая работа — в OcrWorker (ocr.py): один поток, очередь, выгрузка ридера
по простою. Здесь дело простое — принять файл, поставить в очередь, дождаться
текста. Настройки в config.py, схемы ответов в schemas.py.

После запуска живая документация — на `/docs` (Swagger UI) и `/redoc`, сырая
OpenAPI-схема — на `/openapi.json`.

Запуск (ровно один воркер uvicorn — это важно, см. README):
    uvicorn app:app --host 0.0.0.0 --port 8002 --workers 1
"""

import asyncio
import os
import shutil
import tempfile
from contextlib import asynccontextmanager

from config import settings
from fastapi import FastAPI, File, HTTPException, UploadFile
from schemas import HealthResponse, OcrResponse

from ocr import OcrWorker

worker = OcrWorker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Старт: заводим временную папку и запускаем воркер.
    os.makedirs(settings.tmp_dir, exist_ok=True)
    worker.start()
    yield
    # Стоп: гасим воркер и ждём, пока он завершится.
    worker.stop()


app = FastAPI(
    title="EasyOCR OCR (CPU)",
    description=(
        "Распознавание текста на изображениях и в PDF через EasyOCR (torch). "
        "Инференс на CPU, запросы обрабатываются по очереди одним воркером, "
        "ридер выгружается из RAM по простою и поднимается обратно по запросу."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Проверка состояния сервиса",
    description=(
        "Возвращает статус процесса, языки распознавания, загружен ли ридер в RAM "
        "и длину очереди. Не поднимает ридер — безопасно дёргать как healthcheck."
    ),
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model=",".join(settings.langs),
        loaded=worker.loaded,
        queue=worker.queue_size,
    )


@app.post(
    "/ocr",
    response_model=OcrResponse,
    summary="Распознать текст на изображении или в PDF",
    description=(
        "Принимает изображение (png/jpg/tiff/bmp/webp) или PDF, ставит его в "
        "очередь и возвращает распознанный текст, постранично. Первый запрос "
        "после простоя дольше обычного — поднимается ридер (cold start)."
    ),
    responses={
        415: {"description": "Неподдерживаемый формат файла"},
        504: {"description": "Истекло время ожидания результата (REQUEST_TIMEOUT)"},
        500: {"description": "Ошибка инференса"},
    },
)
async def ocr(file: UploadFile = File(...)) -> OcrResponse:
    suffix = os.path.splitext(file.filename or "")[1].lower() or ".png"
    if suffix not in settings.accepted_exts:
        raise HTTPException(
            status_code=415,
            detail=f"Неподдерживаемый формат '{suffix}'. Принимаем: "
            + ", ".join(sorted(settings.accepted_exts)),
        )

    # Пишем загрузку на диск потоком, не держа целиком в RAM, во временный файл
    # с уникальным именем. Дальше его удалит воркер — файлы мы не храним. Запись
    # уводим в to_thread: copyfileobj синхронный и на большом файле подвесил бы
    # event loop.
    with tempfile.NamedTemporaryFile(suffix=suffix, dir=settings.tmp_dir, delete=False) as tmp:
        await asyncio.to_thread(shutil.copyfileobj, file.file, tmp)
        file_path = tmp.name

    future = worker.submit(file_path)

    # Ждём результат, не подвешивая event loop.
    try:
        page_texts = await asyncio.wait_for(
            asyncio.wrap_future(future), timeout=settings.request_timeout
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504, detail="Время ожидания распознавания истекло"
        ) from None
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return OcrResponse(
        langs=list(settings.langs),
        pages=len(page_texts),
        text=settings.page_separator.join(page_texts),
        page_texts=page_texts,
    )
