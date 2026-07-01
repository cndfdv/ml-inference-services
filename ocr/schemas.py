"""
Контракты HTTP-API: типы тел ответов.

Описаны обычными dataclass'ами — FastAPI понимает их как `response_model`,
сериализует и добавляет в OpenAPI-схему (Swagger на `/docs`). Так типы ответов
задокументированы в одном месте и видны клиентам.
"""

from dataclasses import dataclass, field


@dataclass
class OcrResponse:
    """Результат распознавания (ответ `POST /ocr`).

    Attributes:
        model: имя HF-модели из конфигурации (`MODEL_NAME`).
        pages: число распознанных страниц (для картинки — 1, для PDF — по числу
            страниц).
        text: весь распознанный текст, страницы склеены через `PAGE_SEPARATOR`.
        page_texts: распознанный текст по одной строке на страницу.
    """

    model: str
    pages: int
    text: str
    page_texts: list[str] = field(default_factory=list)


@dataclass
class HealthResponse:
    """Состояние сервиса и воркера (ответ `GET /health`).

    Attributes:
        status: всегда "ok", если процесс жив и отвечает.
        model: имя HF-модели из конфигурации (`MODEL_NAME`).
        loaded: загружена ли модель в RAM прямо сейчас (False — выгружена по
            простою либо ещё не было ни одного запроса).
        queue: сколько запросов сейчас ждёт в очереди воркера.
    """

    status: str
    model: str
    loaded: bool
    queue: int
