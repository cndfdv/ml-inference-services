# multilingual-e5-small

Самостоятельный HTTP-контейнер. В этой папке — код только этой модели:
`app.py`, `backend.py`, `settings.py`, `batching.py`, `prepare.py`,
`model.lock.json`, Dockerfile и requirements. Другие модели не импортируются.
Источник весов: **intfloat/multilingual-e5-small**; точная ревизия — в `model.lock.json`.

## Запуск

Из корня репозитория:

```bash
bash scripts/run.sh embeddings/e5-small --device cuda --workers 1
bash scripts/run.sh embeddings/e5-small --device cpu --workers 2
```

Для постоянных настроек: `E5_DEVICE=cpu|cuda`, `E5_WORKERS=N`,
`E5_PORT=18101` в корневой `.env`. Из этой папки обычный
`docker compose up -d --build e5-small` запускает модель на CPU;
локальная `.env.example` содержит её настройки. CUDA/WSL overlays выбираются
через корневой `scripts/compose.sh`.

Веса при отсутствии скачиваются/готовятся до старта HTTP и сохраняются
в Docker volume `ml-services-e5-small-weights`. Повторный старт проверяет манифест.
`E5_PREPARE_MODE=offline` требует готовый bundle. Готовность: `/readyz`;
живой процесс: `/healthz`; информация о весах/устройстве: `/v1/models`;
Swagger: `http://127.0.0.1:18101/docs`.

## API

```bash
curl http://127.0.0.1:18101/e5-small/embed \
  -H 'Content-Type: application/json' \
  -d '{"texts":["Когда вернут командировочные?","Срок хранения договоров"],"role":"query"}'
```

Ответ: `model`, `dim=384`, `count`, `embeddings`, `metadata`.
Тексты передавайте без префикса: по `role=query|passage` сервис добавляет
`query: ` или `passage: `. Masked mean pooling, L2-нормализация, FP32;
контекст до 512 токенов, длинный текст усекается закреплённым токенизатором.

## Очередь и ресурсы

Каждый worker имеет свою модель и очередь. `MAX_QUEUE_ITEMS=256`,
`MAX_REQUEST_ITEMS=64`, `REQUEST_TIMEOUT_S=600`, тело до 64 MiB.
Переполнение очереди — 429, слишком большой вход — 413, неверные данные — 422,
таймаут — 504. Таймаут не прерывает уже выполняющееся вычисление.

Микробатчи: `MAX_BATCH_SIZE=16`, `MAX_BATCH_TOKENS=8192`, `BATCH_WAIT_MS=10`.
Порядок векторов соответствует входам. Бюджет учитывает реальные padded-токены.

[Все модели](../../README.md) · [Проверки качества](../../docs/VALIDATION.md).
