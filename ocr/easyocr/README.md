# EasyOCR OCR — standalone-сервис (CPU, Docker)

HTTP-сервис распознавания текста на изображениях и в PDF на **EasyOCR** (torch).
Работает **только на CPU**, запросы обрабатываются по очереди одним воркером, а
ридер **выгружается из RAM при простое** и поднимается обратно по запросу.
Принимает картинки и PDF. Языки — в `.env`, всё запускается через Docker.

Сделан по образцу соседних сервисов `../../asr/gigaam` и `../../embeddings/e5-small` (та же
архитектура: очередь + поток-воркер + ленивая загрузка/выгрузка модели).

> **Почему EasyOCR.** Лёгкий (модели ~90 МБ), быстрый на CPU (доли секунды на
> страницу), хорошо распознаёт русский, простой API. Полноценные VLM-OCR
> (PaddleOCR-VL и т.п.) под этот сервис не подошли: их transformers-код
> несовместим со стабильными версиями transformers, а на CPU они на порядки
> медленнее. EasyOCR даёт детекцию + распознавание строк (не layout-разбор
> документа), чего для сканов и фото достаточно.

## Состав

| Файл                 | Назначение                                                  |
|----------------------|-------------------------------------------------------------|
| `app.py`             | HTTP-слой (FastAPI: роуты `/easyocr/ocr`, `/easyocr/health` (legacy aliases `/ocr`, `/health`))               |
| `ocr.py`             | поток-воркер: очередь, загрузка/выгрузка ридера, инференс  |
| `config.py`          | конфигурация из env (`Settings`-dataclass) + логгер        |
| `schemas.py`         | типы тел ответов API (dataclass'ы для OpenAPI)             |
| `download_model.py`  | разовое скачивание моделей EasyOCR в кеш                  |
| `Dockerfile`         | рантайм-образ (torch CPU-only + easyocr)                 |
| `docker-compose.yml` | сервис + профиль `prepare` (скачивание моделей)          |
| `.env.example`       | шаблон настроек                                            |
| `requirements.txt`   | зависимости рантайма                                       |
| `docs/`              | техническая документация (архитектура, API, dataclasses)   |

Подробная техническая документация — в [`docs/`](docs/README.md): как всё
устроено внутри ([architecture](docs/architecture.md)), полное описание
[API](docs/api.md) и [dataclasses](docs/dataclasses.md).

## Быстрый старт

```bash
# 1. настройки
cp .env.example .env        # при желании поправь EASYOCR_OCR_LANGS, EASYOCR_PORT, EASYOCR_OMP_NUM_THREADS

# 2. разовое скачивание моделей EasyOCR в кеш (./models)
docker compose --profile prepare run --rm prepare

# 3. поднять сервис
docker compose up -d --build
```

Шаг 2 объявлен под профилем `prepare`, поэтому обычным `docker compose up` он не
поднимается — запускается только явной командой (разово, `run --rm`). Он лишь
прогревает кеш, чтобы сервис не качал модели во время первого запроса; можно и
пропустить — тогда модели скачаются лениво при первом `/easyocr/ocr`. Шаг 3 поднимает сам
сервис `ocr`.

Проверка:

```bash
curl -F "file=@scan.png" http://{host}:{port}/easyocr/ocr
# {"langs":["ru","en"],"pages":1,"text":"...","page_texts":["..."]}

curl http://{host}:{port}/easyocr/health
# {"status":"ok","model":"ru,en","loaded":false,"queue":0}
```

## API

Интерактивная документация поднимается вместе с сервисом:

- **Swagger UI** — `http://localhost:8002/docs`
- **ReDoc** — `http://localhost:8002/redoc`
- **OpenAPI-схема** — `http://localhost:8002/openapi.json`

Типы тел ответов описаны dataclass'ами в `schemas.py` и автоматически попадают
в OpenAPI-схему.

### `POST /easyocr/ocr`

Распознать текст на изображении или в PDF.

- **Тело:** `multipart/form-data`, поле `file` — изображение или PDF.
- **Принимаемые форматы:** `png`, `jpg`, `jpeg`, `tiff`, `tif`, `bmp`, `webp`,
  `pdf`.
- **Ответ `200`** (`OcrResponse`):

  ```json
  {
    "langs": ["ru", "en"],
    "pages": 2,
    "text": "страница 1\n\n---\n\nстраница 2",
    "page_texts": ["страница 1", "страница 2"]
  }
  ```

  | Поле         | Тип         | Значение                                         |
  |--------------|-------------|--------------------------------------------------|
  | `langs`      | list[string]| языки распознавания (`EASYOCR_OCR_LANGS`)                |
  | `pages`      | int         | число распознанных страниц                       |
  | `text`       | string      | весь текст, страницы через `EASYOCR_PAGE_SEPARATOR`      |
  | `page_texts` | list[string]| текст по одной строке на страницу                |

- **Ошибки:**

  | Код   | Когда                                                        |
  |-------|--------------------------------------------------------------|
  | `415` | неподдерживаемое расширение файла                            |
  | `422` | не передан файл (валидация запроса, добавляет FastAPI)       |
  | `500` | инференс упал                                                |
  | `504` | результат не пришёл за `EASYOCR_REQUEST_TIMEOUT`                      |

  ```bash
  curl -F "file=@scan.png" http://localhost:8002/easyocr/ocr
  ```

### `GET /easyocr/health`

Состояние сервиса и воркера. Ридер не поднимает — годится как healthcheck.

- **Ответ `200`** (`HealthResponse`):

  ```json
  { "status": "ok", "model": "ru,en", "loaded": false, "queue": 0 }
  ```

  | Поле     | Тип    | Значение                                              |
  |----------|--------|-------------------------------------------------------|
  | `status` | string | всегда `"ok"`, если процесс отвечает                  |
  | `model`  | string | языки распознавания (`EASYOCR_OCR_LANGS`)                    |
  | `loaded` | bool   | загружен ли ридер в RAM прямо сейчас                 |
  | `queue`  | int    | сколько запросов ждёт в очереди воркера               |

## Настройки (.env)

| Переменная            | По умолчанию   | Назначение                                      |
|-----------------------|----------------|-------------------------------------------------|
| `EASYOCR_OCR_LANGS`           | `ru,en`        | языки распознавания EasyOCR (через запятую)     |
| `EASYOCR_MODULE_PATH` | `/app/models`  | кеш моделей EasyOCR (том `./models`)            |
| `EASYOCR_PARAGRAPH`           | `false`        | группировать строки в абзацы                     |
| `EASYOCR_QUANTIZE`            | `true`         | int8-квантизация (быстрее, меньше RAM); требует AVX2 — на CPU без AVX2 авто-откат на fp32 |
| `EASYOCR_PDF_DPI`             | `150`          | DPI рендера страниц PDF перед OCR                |
| `EASYOCR_TMP_DIR`             | `/app/tmp`     | временная папка для файлов; удаляются сразу после распознавания |
| `EASYOCR_IDLE_TTL`            | `300`          | сек простоя до выгрузки ридера из RAM            |
| `EASYOCR_REQUEST_TIMEOUT`     | `300`          | макс. ожидание результата, сек; дольше — `504`  |
| `EASYOCR_PAGE_SEPARATOR`      | `\n\n---\n\n`  | разделитель страниц в общем `text` (берётся из env литерально) |
| `EASYOCR_OMP_NUM_THREADS`     | `4`            | потоки CPU для torch (≈ число физ. ядер)        |
| `EASYOCR_PORT`                | `8002`         | порт на хосте                                   |

> Языки меняются переменной `EASYOCR_OCR_LANGS` (напр. `ru,en`). EasyOCR совмещает
> кириллические языки с латиницей в одном ридере; полный список кодов — в
> документации EasyOCR. Поменял языки? Прогони шаг 2 (или просто перезапусти —
> модели скачаются лениво), затем `docker compose up -d`.

## Как это работает

- Запросы складываются в очередь (без ограничения по размеру); обрабатывает их
  один поток-воркер, который единолично владеет ридером. Параллельный инференс
  на CPU смысла не имеет, а единоличное владение избавляет от блокировок и гонок.
- Ридер не грузится при старте контейнера. Первый запрос поднимает его в RAM
  (cold start; если моделей ещё нет в кеше — добавится время на скачивание,
  поэтому есть шаг `prepare`).
- Инференс — на **EasyOCR** поверх `torch` (CPU): детекция текстовых областей +
  распознавание строк. EasyOCR сам качает модели по языкам (`EASYOCR_OCR_LANGS`).
- PDF рендерится в картинки постранично через `pypdfium2` (DPI = `EASYOCR_PDF_DPI`),
  каждая страница распознаётся отдельно; тексты собираются в `page_texts`.
- Тот же воркер выгружает ридер из RAM, если запросов не было дольше
  `EASYOCR_IDLE_TTL`. Следующий запрос поднимает его заново.
- Результат не пришёл за `EASYOCR_REQUEST_TIMEOUT` → `504`.
- Загруженный файл пишется во временную папку (`EASYOCR_TMP_DIR`) и **удаляется сразу
  после обработки** — успешной или нет. Сервис ничего не хранит; хранение и БД
  живут в отдельном сервисе. Удаляет файл воркер (не HTTP-слой): при `504`
  запрос перестаёт ждать, но файл ещё нужен воркеру до конца распознавания.

`EASYOCR_WORKERS=1` is the default. Each additional worker holds another reader copy in RAM.

## Про выгрузку из RAM — что реально происходит

При простое сервис отпускает ридер (`self._reader = None`) и зовёт `gc.collect()`.
Крупные аллокации моделей в большинстве случаев возвращаются ОС; остаётся базовый
«хвост» процесса (интерпретатор + torch + библиотеки). То есть основная память
освобождается, но контейнер не падает в ноль.

## Локальный запуск без Docker (опционально)

```bash
# torch/torchvision ставятся CPU-only отдельным индексом
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt python-dotenv
# модели EasyOCR скачаются в кеш при первом запросе
python __main__.py
```


Each worker owns its own model copy. Keep `EASYOCR_WORKERS=1` for memory-constrained deployments; budget about 1 GiB RAM per worker, plus request buffers. Set the variable to a positive integer to run multiple independent CPU workers.
