# Whisper large-v3

Самостоятельный HTTP-контейнер. В этой папке — код только этой модели:
`app.py`, `backend.py`, `settings.py`, `batching.py`, `prepare.py`,
`model.lock.json`, Dockerfile и requirements. Другие модели не импортируются.
Источник весов: **Systran/faster-whisper-large-v3**; точная ревизия — в `model.lock.json`.

## Запуск

Из корня репозитория:

```bash
bash scripts/run.sh asr/whisper --device cuda --workers 1
bash scripts/run.sh asr/whisper --device cpu --workers 2
```

Для постоянных настроек: `WHISPER_DEVICE=cpu|cuda`, `WHISPER_WORKERS=N`,
`WHISPER_PORT=18104` в корневой `.env`. Из этой папки обычный
`docker compose up -d --build whisper-large-v3` запускает модель на CPU;
локальная `.env.example` содержит её настройки. CUDA/WSL overlays выбираются
через корневой `scripts/compose.sh`.

Веса при отсутствии скачиваются/готовятся до старта HTTP и сохраняются
в Docker volume `ml-services-whisper-large-v3-weights`. Повторный старт проверяет манифест.
`WHISPER_PREPARE_MODE=offline` требует готовый bundle. Готовность: `/readyz`;
живой процесс: `/healthz`; информация о весах/устройстве: `/v1/models`;
Swagger: `http://127.0.0.1:18104/docs`.

## API

```bash
curl http://127.0.0.1:18104/whisper-large-v3/transcribe \
  -F 'file=@speech.wav' -F 'language=ru' -F 'task=transcribe'
curl http://127.0.0.1:18104/whisper-large-v3/transcribe/batch \
  -F 'files=@first.wav' -F 'files=@second.mp3' -F 'language=ru'
```

Один файл: `model`, `text`, `language`, `language_probability`, `duration`,
`segments` с `start/end/text`, `metadata`. Пакет: `model`, ordered `results`,
`metadata`. `language=auto` определяет язык; по умолчанию `ru`.
`task=translate` переводит речь на английский.

Используется large-v3, FP32 на обоих устройствах, beam size 5, temperature 0,
VAD выключен. PyAV декодирует mono 16 kHz и ограничивает длительность в процессе
декодирования. `MAX_AUDIO_DURATION_S=600`. Пакет файлов выполняется последовательно
внутри процесса, параллелизм между файлами дают отдельные workers.

## Очередь и ресурсы

Каждый worker имеет свою модель и очередь. `MAX_QUEUE_ITEMS=256`,
`MAX_REQUEST_ITEMS=64`, `REQUEST_TIMEOUT_S=600`, тело до 64 MiB.
Переполнение очереди — 429, слишком большой вход — 413, неверные данные — 422,
таймаут — 504. Таймаут не прерывает уже выполняющееся вычисление.

[Все модели](../../README.md) · [Проверки качества](../../docs/VALIDATION.md).
