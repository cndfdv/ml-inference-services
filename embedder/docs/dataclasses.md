# Структуры данных

Структуры проекта делятся на три группы: конфигурация (`config.py`, dataclass),
контракты API (`schemas.py`: pydantic-модель запроса + dataclass'ы ответов) и
внутренняя единица работы воркера (`embedder.py`, dataclass).

---

## Settings

`config.py` — вся конфигурация сервиса в одном неизменяемом объекте.

```python
@dataclass(frozen=True)
class Settings:
    model_name: str = "intfloat/multilingual-e5-base"
    idle_ttl: int = 120
    request_timeout: float = 300.0
    batch_size: int = 32
    normalize: bool = True
    embed_prefix: str = ""
    max_texts: int = 256
```

| Поле | Тип | Дефолт | env-переменная | Значение |
|------|-----|--------|----------------|----------|
| `model_name` | `str` | `"intfloat/multilingual-e5-base"` | `MODEL_NAME` | Имя HF-модели для sentence-transformers; качается с HuggingFace по имени. |
| `idle_ttl` | `int` | `120` | `IDLE_TTL` | Секунд простоя до выгрузки модели из RAM. |
| `request_timeout` | `float` | `300.0` | `REQUEST_TIMEOUT` | Макс. ожидание результата (очередь + инференс), сек; дольше — клиент получает `504`. |
| `batch_size` | `int` | `32` | `BATCH_SIZE` | Размер батча при кодировании; больше — быстрее, выше пик памяти. |
| `normalize` | `bool` | `True` | `NORMALIZE_EMBEDDINGS` | L2-нормализация выходных векторов (удобно для косинуса). |
| `embed_prefix` | `str` | `""` | `EMBED_PREFIX` | Префикс к каждому тексту перед кодированием (для e5 — `query: `); пусто — без префикса. |
| `max_texts` | `int` | `256` | `MAX_TEXTS` | Предел числа текстов в одном запросе (защита от OOM); больше — `422`. |

**Особенности:**

- **`frozen=True`** — конфигурация неизменяема после старта. Случайно перезаписать
  поле в рантайме нельзя.
- **`from_env()`** — classmethod-конструктор: читает окружение, подставляя дефолты
  полей, и приводит типы (`int()`, `float()`, булев флаг через `_env_bool`). Именно
  он используется на старте: `settings = Settings.from_env()` — единственный
  экземпляр на весь процесс.
- **`_env_bool`** — хелпер для `NORMALIZE_EMBEDDINGS`: истина — `1/true/yes/on`
  (без учёта регистра), остальное — ложь.
- **`slots=True` намеренно не используется.** Со `slots` обращение `cls.idle_ttl`
  в `from_env` возвращало бы дескриптор слота, а не значение по умолчанию, и
  приведение типа падало бы.

> `HF_HOME`, `HF_TOKEN` и `OMP_NUM_THREADS` в `Settings` не попадают: их читают
> напрямую библиотеки (huggingface_hub и torch) из окружения, дублировать в
> конфиге незачем. `HF_TOKEN` — токен доступа HuggingFace для приватных/gated-
> моделей и снятия лимитов загрузки.

---

## Контракты API

`schemas.py` — типы тел запроса и ответов. Полное описание эндпоинтов — в
[api.md](api.md).

### EmbedRequest

Тело `POST /embed`. Pydantic-модель — нужна валидация входа (в отличие от ответов,
которые сервис формирует сам).

```python
class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)  # непустой список текстов
```

`min_length=1` отсекает пустой список на уровне валидации FastAPI (→ `422`).
Верхний предел (`MAX_TEXTS`) проверяется в обработчике, потому что он настраивается
через env.

### EmbedResponse

Ответ `POST /embed`. Обычный (не frozen) dataclass: FastAPI понимает его как
`response_model`, сериализует в JSON и добавляет в OpenAPI-схему.

```python
@dataclass
class EmbedResponse:
    model: str                       # имя модели (MODEL_NAME)
    dim: int                         # размерность вектора (0, если текстов не было)
    count: int                       # число векторов (== числу входных текстов)
    embeddings: list[list[float]]    # векторы, по одному на текст
```

### HealthResponse

Ответ `GET /health`.

```python
@dataclass
class HealthResponse:
    status: str   # всегда "ok", если процесс отвечает
    model: str    # имя модели (MODEL_NAME)
    loaded: bool  # загружена ли модель в RAM прямо сейчас
    queue: int    # сколько запросов ждёт в очереди воркера
```

---

## _Job

`embedder.py` — внутренняя единица работы воркера. Приватный (с `_`), наружу не
торчит; связывает входные тексты с future, куда воркер положит результат.

```python
@dataclass
class _Job:
    texts: list[str]                      # тексты для кодирования
    future: "Future[list[list[float]]]"   # куда воркер положит векторы или исключение
```

Как используется: HTTP-поток создаёт `Future`, оборачивает его вместе с текстами в
`_Job`, кладёт в очередь (`worker.submit`) и ждёт future. Воркер, обработав задачу,
вызывает `future.set_result(vectors)` либо `future.set_exception(...)` — это будит
ожидающий HTTP-поток. Роль этого моста подробно — в
[architecture.md#мост-между-потоками-queue--future](architecture.md#мост-между-потоками-queue--future).
