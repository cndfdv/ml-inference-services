# Dataclasses

Все структуры данных проекта описаны через `dataclass` из стандартной библиотеки.
Их три группы: конфигурация (`config.py`), контракты API (`schemas.py`) и
внутренняя единица работы воркера (`transcriber.py`).

---

## Settings

`config.py` — вся конфигурация сервиса в одном неизменяемом объекте.

```python
@dataclass(frozen=True)
class Settings:
    model_version: str = "gigaam-v3-ctc"
    idle_ttl: int = 60
    request_timeout: float = 3000.0
    tmp_dir: str = "tmp"
    accepted_exts: frozenset[str] = DEFAULT_ACCEPTED_EXTS
    chunk_sec: int = 30
    chunk_batch_size: int = 8
    audio_filters: str = "highpass=f=100,dynaudnorm"
```

| Поле | Тип | Дефолт | env-переменная | Значение |
|------|-----|--------|----------------|----------|
| `model_version` | `str` | `"gigaam-v3-ctc"` | `MODEL_VERSION` | Имя модели для onnx-asr; качается с HuggingFace по имени. |
| `idle_ttl` | `int` | `60` | `IDLE_TTL` | Секунд простоя до выгрузки модели из RAM. |
| `request_timeout` | `float` | `3000.0` | `REQUEST_TIMEOUT` | Макс. ожидание результата (очередь + инференс), сек; дольше — клиент получает `504`. |
| `tmp_dir` | `str` | `"tmp"` | `TMP_DIR` | Временная папка для загруженных аудио; файл удаляется сразу после обработки. |
| `accepted_exts` | `frozenset[str]` | `DEFAULT_ACCEPTED_EXTS` | — | Допустимые расширения входных файлов. |
| `chunk_sec` | `int` | `30` | `CHUNK_SEC` | Длина окна нарезки длинного аудио, сек. `0` — не резать. |
| `chunk_batch_size` | `int` | `8` | `CHUNK_BATCH_SIZE` | Сколько окон гонится через инференс одним батчем. |
| `audio_filters` | `str` | `"highpass=f=100,dynaudnorm"` | `AUDIO_FILTERS` | Цепочка аудиофильтров ffmpeg (`-af`), предобработка перед распознаванием; пусто — без фильтров. |

**Особенности:**

- **`frozen=True`** — конфигурация неизменяема после старта. Случайно перезаписать
  поле в рантайме нельзя.
- **`from_env()`** — classmethod-конструктор: читает окружение, подставляя дефолты
  полей, и приводит типы (`int()`, `float()`). Именно он используется на старте:
  `settings = Settings.from_env()` — единственный экземпляр на весь процесс.
- **`accepted_exts`** не читается из env: список задан в коде
  (`DEFAULT_ACCEPTED_EXTS`), потому что это скорее контракт API, чем настройка
  развёртывания.
- **`slots=True` намеренно не используется.** Со `slots` обращение `cls.idle_ttl`
  в `from_env` возвращало бы дескриптор слота, а не значение по умолчанию, и
  приведение типа падало бы.

> Дефолты в коде и в `.env.example` могут отличаться (например `idle_ttl`,
> `request_timeout`): код задаёт значения для локального запуска без докера,
> `.env.example` — рекомендованные для развёртывания. В докере всегда выигрывает
> `.env`.

---

## Контракты API

`schemas.py` — типы тел ответов. Обычные (не frozen) dataclass'ы: FastAPI
понимает их как `response_model`, сериализует в JSON и добавляет в OpenAPI-схему
(Swagger). Полное описание эндпоинтов — в [api.md](api.md).

### HealthResponse

Ответ `GET /health`.

```python
@dataclass
class HealthResponse:
    status: str   # всегда "ok", если процесс отвечает
    model: str    # версия модели (MODEL_VERSION)
    loaded: bool  # загружена ли модель в RAM прямо сейчас
    queue: int    # сколько запросов ждёт в очереди воркера
```

### TranscribeResponse

Ответ `POST /transcribe`.

```python
@dataclass
class TranscribeResponse:
    text: str  # распознанный текст
```

---

## _Job

`transcriber.py` — внутренняя единица работы воркера. Приватный (с `_`), наружу
не торчит; связывает аудио с future, куда воркер положит результат.

```python
@dataclass
class _Job:
    audio_path: str        # путь к временному файлу (воркер удалит после обработки)
    future: "Future[str]"  # куда воркер положит текст или исключение
```

Как используется: HTTP-поток создаёт `Future`, оборачивает его вместе с путём в
`_Job`, кладёт в очередь (`worker.submit`) и ждёт future. Воркер, обработав
задачу, вызывает `future.set_result(text)` либо `future.set_exception(...)` — это
будит ожидающий HTTP-поток. Роль этого моста подробно — в
[architecture.md#мост-между-потоками-queue--future](architecture.md#мост-между-потоками-queue--future).
