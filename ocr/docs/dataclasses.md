# Dataclasses

Все структуры данных проекта описаны через `dataclass` из стандартной библиотеки.
Их три группы: конфигурация (`config.py`), контракты API (`schemas.py`) и
внутренняя единица работы воркера (`ocr.py`).

---

## Settings

`config.py` — вся конфигурация сервиса в одном неизменяемом объекте.

```python
@dataclass(frozen=True)
class Settings:
    langs: tuple[str, ...] = ("ru", "en")
    paragraph: bool = False
    pdf_dpi: int = 150
    idle_ttl: int = 300
    request_timeout: float = 300.0
    tmp_dir: str = "tmp"
    accepted_exts: frozenset[str] = DEFAULT_ACCEPTED_EXTS
    page_separator: str = "\n\n---\n\n"
```

| Поле | Тип | Дефолт | env-переменная | Значение |
|------|-----|--------|----------------|----------|
| `langs` | `tuple[str, ...]` | `("ru", "en")` | `OCR_LANGS` | Языки распознавания EasyOCR (в env — через запятую). |
| `paragraph` | `bool` | `False` | `PARAGRAPH` | Группировать распознанные строки в абзацы. |
| `pdf_dpi` | `int` | `150` | `PDF_DPI` | DPI рендера страниц PDF в картинку перед OCR. |
| `idle_ttl` | `int` | `300` | `IDLE_TTL` | Секунд простоя до выгрузки ридера из RAM. |
| `request_timeout` | `float` | `300.0` | `REQUEST_TIMEOUT` | Макс. ожидание результата (очередь + инференс), сек; дольше — клиент получает `504`. |
| `tmp_dir` | `str` | `"tmp"` | `TMP_DIR` | Временная папка для загруженных файлов; файл удаляется сразу после обработки. |
| `accepted_exts` | `frozenset[str]` | `DEFAULT_ACCEPTED_EXTS` | — | Допустимые расширения входных файлов. |
| `page_separator` | `str` | `"\n\n---\n\n"` | `PAGE_SEPARATOR` | Разделитель страниц при склейке многостраничного PDF в общий `text`. |

**Особенности:**

- **`frozen=True`** — конфигурация неизменяема после старта. Случайно перезаписать
  поле в рантайме нельзя.
- **`from_env()`** — classmethod-конструктор: читает окружение, подставляя дефолты
  полей, и приводит типы (`OCR_LANGS` → кортеж, `PARAGRAPH` → bool, `int`/`float`).
  Именно он используется на старте: `settings = Settings.from_env()` — единственный
  экземпляр на весь процесс.
- **`accepted_exts`** не читается из env: список задан в коде
  (`DEFAULT_ACCEPTED_EXTS`), потому что это скорее контракт API, чем настройка
  развёртывания.
- **`page_separator`** берётся из env **литерально**: значение
  `PAGE_SEPARATOR=\n` даст последовательность из двух символов `\` и `n`, а не
  перенос строки. Дефолт в коде — с настоящими переносами. Обычно менять не нужно.
- **`slots=True` намеренно не используется.** Со `slots` обращение
  `cls.idle_ttl` в `from_env` возвращало бы дескриптор слота, а не значение по
  умолчанию, и приведение типа падало бы.

> `EASYOCR_MODULE_PATH` (каталог кеша моделей) и `OMP_NUM_THREADS` в `Settings` не
> попадают: их читают напрямую EasyOCR и torch из окружения.

---

## Контракты API

`schemas.py` — типы тел ответов. Обычные (не frozen) dataclass'ы: FastAPI
понимает их как `response_model`, сериализует в JSON и добавляет в OpenAPI-схему
(Swagger). Полное описание эндпоинтов — в [api.md](api.md).

### OcrResponse

Ответ `POST /ocr`.

```python
@dataclass
class OcrResponse:
    langs: list[str]                 # языки распознавания (OCR_LANGS)
    pages: int                       # число распознанных страниц
    text: str                        # весь текст, страницы через PAGE_SEPARATOR
    page_texts: list[str] = field(default_factory=list)  # текст по одной строке на страницу
```

### HealthResponse

Ответ `GET /health`.

```python
@dataclass
class HealthResponse:
    status: str   # всегда "ok", если процесс отвечает
    model: str    # языки распознавания (OCR_LANGS), напр. "ru,en"
    loaded: bool  # загружен ли ридер в RAM прямо сейчас
    queue: int    # сколько запросов ждёт в очереди воркера
```

---

## _Job

`ocr.py` — внутренняя единица работы воркера. Приватный (с `_`), наружу не торчит;
связывает файл с future, куда воркер положит результат.

```python
@dataclass
class _Job:
    file_path: str               # путь к временному файлу (воркер удалит после обработки)
    future: "Future[list[str]]"  # куда воркер положит список текстов страниц или исключение
```

Как используется: HTTP-поток создаёт `Future`, оборачивает его вместе с путём в
`_Job`, кладёт в очередь (`worker.submit`) и ждёт future. Воркер, обработав
задачу, вызывает `future.set_result(page_texts)` либо `future.set_exception(...)` —
это будит ожидающий HTTP-поток. Роль этого моста подробно — в
[architecture.md#мост-между-потоками-queue--future](architecture.md#мост-между-потоками-queue--future).
