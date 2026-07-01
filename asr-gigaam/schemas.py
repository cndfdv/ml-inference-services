"""
Контракты HTTP-API: типы тел ответов.

Описаны обычными dataclass'ами — FastAPI понимает их как `response_model`,
сериализует и добавляет в OpenAPI-схему (Swagger на `/docs`). Так типы ответов
задокументированы в одном месте и видны клиентам.
"""

from dataclasses import dataclass


@dataclass
class HealthResponse:
    """Состояние сервиса и воркера (ответ `GET /health`).

    Attributes:
        status: всегда "ok", если процесс жив и отвечает.
        model: версия модели из конфигурации (`MODEL_VERSION`).
        loaded: загружена ли модель в RAM прямо сейчас (False — выгружена по
            простою либо ещё не было ни одного запроса).
        queue: сколько запросов сейчас ждёт в очереди воркера.
    """

    status: str
    model: str
    loaded: bool
    queue: int


@dataclass
class TranscribeResponse:
    """Результат распознавания (ответ `POST /transcribe`).

    Attributes:
        text: распознанный текст аудио.
    """

    text: str
