# USER-bge-m3

Самостоятельный HTTP-контейнер. В этой папке — код только этой модели:
`app.py`, `backend.py`, `settings.py`, `batching.py`, `prepare.py`,
`model.lock.json`, Dockerfile и requirements. Другие модели не импортируются.
Источник весов: **deepvk/USER-bge-m3**; точная ревизия — в `model.lock.json`.

## Запуск

Из корня репозитория:

```bash
bash scripts/run.sh embeddings/user-bge-m3 --device cuda --workers 1
bash scripts/run.sh embeddings/user-bge-m3 --device cpu --workers 2
```

Для постоянных настроек: `USER_DEVICE=cpu|cuda`, `USER_WORKERS=N`,
`USER_PORT=18102` в корневой `.env`. Из этой папки обычный
`docker compose up -d --build user-bge-m3` запускает модель на CPU;
локальная `.env.example` содержит её настройки. CUDA/WSL overlays выбираются
через корневой `scripts/compose.sh`.

Веса при отсутствии скачиваются/готовятся до старта HTTP и сохраняются
в Docker volume `ml-services-user-bge-m3-weights`. Повторный старт проверяет манифест.
`USER_PREPARE_MODE=offline` требует готовый bundle. Готовность: `/readyz`;
живой процесс: `/healthz`; информация о весах/устройстве: `/v1/models`;
Swagger: `http://127.0.0.1:18102/docs`.

## API

```bash
curl http://127.0.0.1:18102/user-bge-m3/embed \
  -H 'Content-Type: application/json' \
  -d '{"texts":["Положение: возмещение в течение 10 рабочих дней."],"role":"passage"}'
```

Ответ: `model`, `dim=1024`, `count`, `embeddings`, `metadata`.
ONNX opset 17, CLS pooling, L2, без префикса. `role=query|passage` принимается
для общего поискового контракта. Контекст до 8192 токенов, хвост усекается.
FP32 одинаков на CPU/CUDA. `USER_CPU_VARIANT=int8` доступен только на CPU:
при первом старте готовится отдельный INT8-граф; FP32-файлы сохраняются.
INT8 может изменить результаты поиска; сверяйте качество на своих данных.

## Очередь и ресурсы

Каждый worker имеет свою модель и очередь. `MAX_QUEUE_ITEMS=256`,
`MAX_REQUEST_ITEMS=64`, `REQUEST_TIMEOUT_S=600`, тело до 64 MiB.
Переполнение очереди — 429, слишком большой вход — 413, неверные данные — 422,
таймаут — 504. Таймаут не прерывает уже выполняющееся вычисление.

Микробатчи: `MAX_BATCH_SIZE=16`, `MAX_BATCH_TOKENS=8192`, `BATCH_WAIT_MS=10`.
Порядок векторов соответствует входам. Бюджет учитывает реальные padded-токены.

[Все модели](../../README.md) · [Проверки качества](../../docs/VALIDATION.md).
