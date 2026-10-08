# ML-сервисы

Набор автономных inference-сервисов. Исходные CPU-сервисы ниже используют общую архитектуру:
FastAPI + один поток-воркер с очередью, ленивая загрузка модели и **выгрузка из
RAM по простою**, вся конфигурация в `.env` каждого сервиса.

| Сервис | Папка | Порт | Что делает | Бэкенд |
|--------|-------|------|-----------|--------|
| ASR | [`asr-gigaam/`](asr-gigaam/) | 8000 | речь → текст (`POST /transcribe`) | GigaAM, onnx-asr |
| Эмбеддер | [`embedder/`](embedder/) | 8001 | текст → векторы (`POST /embed`) | sentence-transformers |
| OCR | [`ocr/`](ocr/) | 8002 | изображение/PDF → текст (`POST /ocr`) | EasyOCR (torch) |

У каждого сервиса свой `README.md`, `docs/` и автономный `docker-compose.yml` —
их можно поднимать по отдельности. Корневой `docker-compose.yml` собирает все три
вместе.

## Запуск всех сразу

```bash
# 1. настройки: в каждой папке скопируй .env из шаблона
cp asr-gigaam/.env.example asr-gigaam/.env
cp embedder/.env.example   embedder/.env
cp ocr/.env.example        ocr/.env
# при необходимости впиши в .env свой HF_TOKEN (для приватных/gated-моделей
# HuggingFace и снятия лимитов загрузки) — .env в gitignore, наружу не уйдёт

# 2. (опционально) прогреть кеши моделей заранее, чтобы не качать на первом запросе
docker compose --profile prepare run --rm asr-prepare
docker compose --profile prepare run --rm embedder-prepare
docker compose --profile prepare run --rm ocr-prepare

# 3. поднять все три
docker compose up -d --build
```

Один сервис: `docker compose up -d --build embedder`.

Порты на хосте по умолчанию `8000/8001/8002`, переопределяются переменными
`ASR_PORT` / `EMBEDDER_PORT` / `OCR_PORT` (например в корневом `.env` рядом с этим
compose или в окружении).

Проверка:

```bash
curl http://localhost:8000/health   # asr
curl http://localhost:8001/health   # embedder
curl http://localhost:8002/health   # ocr
```

Здоровье каждого контейнера отслеживает healthcheck (`GET /health`, модель не
поднимает). Общие принципы (очередь, cold start, выгрузка по простою, `--workers
1`) — в README и `docs/` каждого сервиса.

## Закреплённые модели на CPU и GPU

Каталог [`model-inference/`](model-inference/) содержит отдельный Compose-проект:
multilingual-e5-small, deepvk/USER-bge-m3, RapidOCR PP-OCRv5 mobile Cyrillic
и Whisper large-v3. У каждой модели свои HTTP-ручки, CPU/GPU-режим и число
процессов-воркеров. Порты: 18101–18104. Поддержаны NVIDIA Container Toolkit
и native Docker в WSL2. Веса подготовлены заранее, startup проверяет SHA-256.

[Запуск, API и батчи](model-inference/README.md) ·
[Проверки CPU/GPU и качества](model-inference/docs/VALIDATION.md) ·
[Источники моделей](model-inference/docs/MODELS.md).
