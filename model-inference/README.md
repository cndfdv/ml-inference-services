# Model inference: CPU и GPU

Четыре самостоятельных контейнера для разработки и тестирования сервисов.
Каждая модель запускается на CPU или NVIDIA GPU с теми же закреплёнными весами
и обработкой входов. Этот каталог дополняет существующие CPU-сервисы родительского
репозитория.

| Модель | HTTP-ручка | Порт хоста | Представление |
| --- | --- | --- | --- |
| multilingual-e5-small | `POST /e5-small/embed` | 18101 | 384, mean pooling, L2 |
| deepvk/USER-bge-m3 | `POST /user-bge-m3/embed` | 18102 | 1024, ONNX CLS, L2 |
| RapidOCR PP-OCRv5 mobile Cyrillic | `POST /rapid-v5-mobile/ocr` | 18103 | текст, координаты и уверенность |
| Whisper large-v3 | `POST /whisper-large-v3/transcribe` | 18104 | текст и сегменты с временем |

У каждого контейнера также есть `/healthz`, `/readyz`, `/<model>/health`,
`/v1/models` и Swagger на `/docs`. Чужая модель в URL получает 404.

## Быстрый запуск

Требуются Docker Engine + Compose и свободные порты. Для GPU: NVIDIA Container
Toolkit либо WSL2 с `/dev/dxg` и `/usr/lib/wsl`. Python в образе — 3.11.

```bash
cd model-inference
cp .env.example .env
mkdir -p models
bash scripts/compose.sh build e5-small
# Подготовка моделей: скачивает закреплённые публичные снимки и проверяет хеши.
bash scripts/compose.sh run --rm --no-deps --user "$(id -u):$(id -g)" \
  -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \
  -e HF_HOME=/models/.hf-cache \
  -v "$PWD/models:/models:rw" \
  e5-small python scripts/prepare_models.py --output /models
bash scripts/compose.sh up -d
bash scripts/compose.sh ps
```

Веса не входят
в Git и образ. После подготовки сервисы работают автономно, без загрузки весов
во время HTTP-запросов. Манифесты SHA-256 проверяются при каждом запуске воркера.
Образ общий, но процессы и API у моделей отдельные.

Для `home` веса уже подготовлены. Рабочий вход — `/home/knyze/model-inference`.
Это ссылка на Git checkout `/home/knyze/ml-inference-services/model-inference`.
Запуск с Mac: `ssh home 'cd /home/knyze/model-inference && bash scripts/compose.sh up -d'`.

## Устройство и воркеры при запуске

```bash
bash scripts/run.sh e5-small --device cuda --workers 1
bash scripts/run.sh user-bge-m3 --device cpu --workers 2
bash scripts/run.sh rapid-v5-mobile --device cuda --workers 1
bash scripts/run.sh whisper-large-v3 --device cpu --workers 1

# Все четыре на CPU:
bash scripts/compose.sh --mode cpu up -d
# NVIDIA Container Toolkit:
bash scripts/compose.sh --mode gpu up -d
# WSL2 native Docker:
bash scripts/compose.sh --mode wsl up -d
```

Для постоянных настроек используйте `.env`: `E5_DEVICE`, `USER_DEVICE`,
`OCR_DEVICE`, `WHISPER_DEVICE` (`cpu`/`cuda`) и соответствующие `*_WORKERS`.
`run.sh` задаёт их для одного запуска; сохраните значения в `.env`, чтобы
следующий общий `up` их воспроизвёл. CPU-overlay переключает все выбранные
сервисы на CPU. CUDA-mode завершается ошибкой при отсутствии GPU; автоматической
подмены на CPU нет.

Каждый процесс-воркер загружает свою копию модели и имеет свою очередь.
По умолчанию — один воркер на сервис. Сначала увеличивайте батч, затем проверяйте
дополнительные воркеры по памяти и задержке. Whisper large-v3 особенно требователен
к RAM/VRAM; число воркеров не означает столько же бесплатных копий модели.

## Эмбеддинги

```bash
curl http://127.0.0.1:18101/e5-small/embed \
  -H 'Content-Type: application/json' \
  -d '{"texts":["Когда вернут командировочные?","Что закупают по заявке 884?"],"role":"query"}'

curl http://127.0.0.1:18102/user-bge-m3/embed \
  -H 'Content-Type: application/json' \
  -d '{"texts":["Положение: срок возмещения — 10 рабочих дней."],"role":"passage"}'
```

Ответ: `model`, `dim`, `count`, `embeddings` в порядке входов и `metadata`.
Передавайте текст **без префикса**: E5 добавляет `query: ` или `passage: `
по `role`; USER не добавляет префикс. По умолчанию `role=query`.
Лимиты контекста: E5 — 512, USER — 8192 токена. Хвост более длинного текста
усекается по правилам закреплённого токенизатора.

## OCR

Изображение или PDF через multipart; PDF растеризуется при 150 DPI:

```bash
curl http://127.0.0.1:18103/rapid-v5-mobile/ocr -F 'file=@page.png'
curl http://127.0.0.1:18103/rapid-v5-mobile/ocr -F 'file=@document.pdf'
```

Пакет изображений: `POST /rapid-v5-mobile/ocr/batch`, JSON
`{"images":["<base64 PNG/JPEG/...>","<base64 второй страницы>"]}`.
В JSON нет URL-загрузки. Возвращаются страницы с текстом, строками, координатами
и уверенностью. Границы строк не являются готовой структурой таблицы.

Detector и recognizer — PP-OCRv5 **mobile**, recognizer именно кириллический.
Классификатор поворота отключён, нормализация detector совпадает с бенчмарком.
`REC_BATCH_SIZE=6` — исходный размер батча распознавателя.

## Whisper large-v3

```bash
curl http://127.0.0.1:18104/whisper-large-v3/transcribe \
  -F 'file=@speech.wav' -F 'language=ru' -F 'task=transcribe'

curl http://127.0.0.1:18104/whisper-large-v3/transcribe/batch \
  -F 'files=@first.wav' -F 'files=@second.mp3' -F 'language=ru'
```

Ответ содержит текст, язык, длительность и сегменты `start/end/text`.
`task=translate` переводит речь на английский. Язык по умолчанию — `ru`;
вариант автоматического определения описан в Swagger.
Аудио декодируется PyAV в mono 16 kHz; длительность ограничивается во время
декодирования. По умолчанию — 600 секунд на файл.

Используется **large-v3**, без замены на turbo или distilled. Один и тот же
закреплённый CTranslate2 snapshot, FP32-вычисления на CPU/GPU, beam size 5,
temperature 0, VAD выключен. Веса snapshot могут храниться в иной исходной
точности; `float32` описывает вычисления runtime.

## Батчи, очереди и лимиты

Эмбеддинги объединяются между HTTP-запросами в микробатч, с сохранением порядка
результатов. Реальный батч дополнительно ограничивается произведением числа
текстов и максимальной токенизированной длины (`MAX_BATCH_TOKENS`).
OCR принимает пакет страниц; detector выполняется по странице, recognizer
обрабатывает батчи найденных строк. Whisper принимает пакет файлов и обрабатывает
файлы последовательно внутри процесса; параллельные файлы распределяются между
процессами-воркерами. Пакет файлов не меняет алгоритм декодирования Whisper.

| Настройка | По умолчанию | Значение |
| --- | ---: | --- |
| `MAX_BATCH_SIZE` | 16 | максимум элементов микробатча |
| `MAX_BATCH_TOKENS` | 8192 | бюджет padded-токенов |
| `BATCH_WAIT_MS` | 10 | окно объединения запросов |
| `MAX_QUEUE_ITEMS` | 256 | очередь + выполняющиеся элементы, на процесс |
| `MAX_REQUEST_ITEMS` | 64 | максимум элементов одного запроса |
| `REQUEST_TIMEOUT_S` | 600 | ожидание результата |
| `MAX_REQUEST_BODY_BYTES` | 67108864 | максимум HTTP-тела |
| `WHISPER_BATCH_SIZE` | 1 | элементы очереди Whisper за один вызов |

Переполнение очереди — 429, слишком большое тело — 413, неверные данные —
422, таймаут — 504. Таймаут отменяет ожидание, но не прерывает уже запущенное
GPU-ядро. Лимиты применяются к каждому процессу отдельно.

## Подключение сервисов

В общей сети `model-inference` используйте адреса
`http://e5-small:8000/e5-small/embed`,
`http://user-bge-m3:8000/user-bge-m3/embed`,
`http://rapid-v5-mobile:8000/rapid-v5-mobile/ocr`,
`http://whisper-large-v3:8000/whisper-large-v3/transcribe`.
Сеть можно объявить `external: true` в Compose потребителя.

Порты хоста привязаны к localhost. С Mac можно сделать SSH-туннель:

```bash
ssh -N -L 18101:127.0.0.1:18101 -L 18102:127.0.0.1:18102 \
  -L 18103:127.0.0.1:18103 -L 18104:127.0.0.1:18104 home
```

## Совпадение теста и прода

Для наиболее близкого поведения используйте **этот же каталог и model bundle**
в CPU-режиме в проде. Сверяйте `checkpoint_revision`, `weight_files`,
`contract_fingerprint`, preprocessing и precision через `/v1/models`.
FP32 и отсутствие TF32 уменьшают численные различия, но не гарантируют
побитовое совпадение разных устройств.

`USER_CPU_VARIANT=int8` — отдельная CPU-опция. Подготовьте её с `--int8` и
повторите проверку поиска: INT8 не тождествен FP32. Старый CPU EasyOCR также
не эквивалентен RapidOCR. Само переключение устройства не должно менять модель,
префиксы, язык OCR или decoder options.

Проверки и фактические результаты: [docs/VALIDATION.md](docs/VALIDATION.md).
Лицензии и источники: [docs/MODELS.md](docs/MODELS.md).

```bash
# Лёгкие unit-тесты: модельные зависимости не нужны.
python3 -m pip install -r requirements-dev.txt fastapi==0.115.12 pydantic==2.11.4 \
  httpx==0.28.1 Pillow==11.2.1 numpy==2.2.6 python-multipart==0.0.20 pypdfium2==4.30.0
make test
make smoke
bash scripts/compose.sh logs --tail=80
bash scripts/compose.sh down
```

`down` останавливает только этот Compose-проект; модели на диске сохраняются.
