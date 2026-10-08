# ML-сервисы

Один репозиторий: **задача → модель → самостоятельный контейнер**.
У каждой модели собственные HTTP-приложение, код инференса, очередь,
настройки, Dockerfile и зависимости. Контейнеры общаются с потребителями по HTTP.

```text
asr/
  gigaam/               GigaAM v3 CTC, CPU
  whisper/              Whisper large-v3, CPU / GPU
embeddings/
  e5-small/             multilingual-e5-small, CPU / GPU
  user-bge-m3/          deepvk/USER-bge-m3, CPU / GPU
ocr/
  easyocr/              EasyOCR ru + en, CPU
  rapid-v5-mobile/      PP-OCRv5 mobile Cyrillic, CPU / GPU
scripts/                запуск и проверки контейнеров
tests/                 HTTP, очереди, подготовка весов
docs/                  миграция, источники, результаты проверок
```

## Модели и API

| Задача | Модель и инструкция | Режим | Порт | HTTP-ручка |
| --- | --- | --- | ---: | --- |
| ASR | [GigaAM](asr/gigaam/README.md) | CPU | 8000 | `POST /gigaam/transcribe` |
| ASR | [Whisper large-v3](asr/whisper/README.md) | CPU / CUDA | 18104 | `POST /whisper-large-v3/transcribe` |
| Эмбеддинги | [E5-small](embeddings/e5-small/README.md) | CPU / CUDA | 18101 | `POST /e5-small/embed` |
| Эмбеддинги | [USER-bge-m3](embeddings/user-bge-m3/README.md) | CPU / CUDA | 18102 | `POST /user-bge-m3/embed` |
| OCR | [EasyOCR](ocr/easyocr/README.md) | CPU | 8002 | `POST /easyocr/ocr` |
| OCR | [RapidOCR v5 mobile](ocr/rapid-v5-mobile/README.md) | CPU / CUDA | 18103 | `POST /rapid-v5-mobile/ocr` |

Порты хоста по умолчанию доступны на `127.0.0.1`. Swagger каждого контейнера —
`http://127.0.0.1:<порт>/docs`. Все контейнеры слушают внутренний порт 8000.

## Быстрый запуск

Нужны Docker Engine и Compose **2.24+**. Для CUDA — NVIDIA Container Toolkit
или WSL2 с `/dev/dxg` и `/usr/lib/wsl`. На CPU GPU-инструменты не нужны.

```bash
git clone https://github.com/cndfdv/ml-inference-services.git
cd ml-inference-services

# Одна модель. Путь соответствует папке в репозитории.
bash scripts/run.sh embeddings/e5-small --device cuda --workers 1
bash scripts/run.sh asr/whisper --device cpu --workers 1
bash scripts/run.sh asr/gigaam --workers 2

# Все модели одной задачи.
bash scripts/compose.sh --mode cpu --profile embeddings up -d --build
bash scripts/compose.sh --mode wsl --profile ocr up -d --build

# Все шесть моделей на CPU.
bash scripts/compose.sh --mode cpu --profile all up -d --build
```

Без `.env` работают значения по умолчанию. Для постоянных настроек скопируйте
`.env.example` в `.env` в корне; индивидуальная `.env` в папке модели тоже поддержана.
`run.sh` задаёт устройство и workers на один запуск; для следующего общего `up`
сохраните соответствующие `*_DEVICE` и `*_WORKERS` в `.env`.

`--mode auto` выбирает WSL, NVIDIA GPU или CPU по доступному оборудованию.
`run.sh --device cuda` требует GPU и не переключается на CPU. Для NVIDIA Toolkit
используйте `--mode gpu`, для native Docker в WSL2 — `--mode wsl`.

Модели выбираются профилями `asr`, `embeddings`, `ocr`, `all` или именем сервиса.
Обычный `docker compose up` без выбора профиля ничего не запускает.

## Веса при первом запуске

E5, USER, RapidOCR и Whisper **сами готовят недостающие веса при старте**,
до запуска HTTP-воркеров. Закреплены checkpoint и обработка входов; каждый файл
проверяется по SHA-256. Повторный запуск использует готовые веса.

Веса сохраняются в отдельном Docker volume для каждой модели: например,
`ml-services-e5-small-weights`. Они не входят в Git или Docker-образ.
Первое скачивание крупных моделей и экспорт USER в ONNX могут занять несколько
минут. Проверяйте `/readyz` перед отправкой запросов:

```bash
bash scripts/compose.sh logs -f user-bge-m3
curl http://127.0.0.1:18102/readyz
```

`*_PREPARE_MODE=offline` запрещает скачивание отсутствующего bundle.
Повреждённый или несовместимый существующий bundle вызывает ошибку; сервис
не подменяет его другой моделью. GigaAM и EasyOCR сохраняют прежнюю CPU-схему:
скачивание при первом запросе, выгрузка из RAM по простою; их можно подготовить
заранее командой из README модели.

## Батчи и workers

E5 и USER объединяют запросы в микробатчи. RapidOCR принимает пакет страниц
и распознаёт найденные строки батчами. Whisper принимает пакет файлов и
обрабатывает их последовательно внутри воркера. GigaAM распознаёт батчи
30-секундных окон; EasyOCR обрабатывает страницы последовательно.

Каждый процесс-воркер держит собственную модель и очередь. По умолчанию — один.
Сначала подберите размер батча, затем увеличивайте workers по доступной памяти.
Четыре FP32-модели вместе занимают почти всю 16 GB VRAM после полной нагрузки.

## Подключение своих сервисов

В Docker-сети `model-inference` адреса имеют вид
`http://e5-small:8000/e5-small/embed` или
`http://whisper-large-v3:8000/whisper-large-v3/transcribe`.
Подключите контейнер потребителя к этой сети как к external network.

На `home` рабочий checkout — `/home/knyze/ml-inference-services`.
Для доступа с Mac используйте SSH-туннель:

```bash
ssh -N -L 18101:127.0.0.1:18101 -L 18102:127.0.0.1:18102 \
  -L 18103:127.0.0.1:18103 -L 18104:127.0.0.1:18104 home
```

## Разработка и проверки

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
make test PYTHON=.venv/bin/python
make config
make smoke PYTHON=.venv/bin/python   # четыре запущенных E5/USER/Rapid/Whisper
```

[Переход со старой структуры](docs/MIGRATION.md) ·
[Модели, ревизии и лицензии](docs/MODELS.md) ·
[Проверки CPU/GPU и качества](docs/VALIDATION.md).
