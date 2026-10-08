# RapidOCR PP-OCRv5 mobile Cyrillic

Самостоятельный HTTP-контейнер. В этой папке — код только этой модели:
`app.py`, `backend.py`, `settings.py`, `batching.py`, `prepare.py`,
`model.lock.json`, Dockerfile и requirements. Другие модели не импортируются.
Источник весов: **RapidAI/RapidOCR 3.9.2**; точная ревизия — в `model.lock.json`.

## Запуск

Из корня репозитория:

```bash
bash scripts/run.sh ocr/rapid-v5-mobile --device cuda --workers 1
bash scripts/run.sh ocr/rapid-v5-mobile --device cpu --workers 2
```

Для постоянных настроек: `OCR_DEVICE=cpu|cuda`, `OCR_WORKERS=N`,
`OCR_PORT=18103` в корневой `.env`. Из этой папки обычный
`docker compose up -d --build rapid-v5-mobile` запускает модель на CPU;
локальная `.env.example` содержит её настройки. CUDA/WSL overlays выбираются
через корневой `scripts/compose.sh`.

Веса при отсутствии скачиваются/готовятся до старта HTTP и сохраняются
в Docker volume `ml-services-rapid-v5-mobile-weights`. Повторный старт проверяет манифест.
`OCR_PREPARE_MODE=offline` требует готовый bundle. Готовность: `/readyz`;
живой процесс: `/healthz`; информация о весах/устройстве: `/v1/models`;
Swagger: `http://127.0.0.1:18103/docs`.

## API

```bash
curl http://127.0.0.1:18103/rapid-v5-mobile/ocr -F 'file=@page.png'
curl http://127.0.0.1:18103/rapid-v5-mobile/ocr -F 'file=@document.pdf'
```

Multipart возвращает `model`, `pages`, `count`, общий `text`. В странице:
`text`, `lines`, `width`, `height`; в строке — `text`, `bbox`, `polygon`,
`confidence`. PDF растеризуется при 150 DPI. JSON-пакет:
`POST /rapid-v5-mobile/ocr/batch`, `{"images":["<base64 PNG/JPEG>","<base64>"]}`.

Detector и кириллический recognizer — PP-OCRv5 **mobile**, classifier отключён.
Нормализация detector соответствует бенчмарку; распознавание строк батчами
`REC_BATCH_SIZE=6`. Страницы detector обрабатывает последовательно.

## Очередь и ресурсы

Каждый worker имеет свою модель и очередь. `MAX_QUEUE_ITEMS=256`,
`MAX_REQUEST_ITEMS=64`, `REQUEST_TIMEOUT_S=600`, тело до 64 MiB.
Переполнение очереди — 429, слишком большой вход — 413, неверные данные — 422,
таймаут — 504. Таймаут не прерывает уже выполняющееся вычисление.

[Все модели](../../README.md) · [Проверки качества](../../docs/VALIDATION.md).
