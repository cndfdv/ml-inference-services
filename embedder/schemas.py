"""
Формы запросов и ответов HTTP-API.

Тело `POST /embed` — pydantic-модель (нужна валидация входа), ответы — обычные
dataclass'ы. FastAPI берёт и то, и другое как схему, сериализует ответы и
кладёт всё в OpenAPI (Swagger на `/docs`). Контракт в одном месте и виден
клиентам.
"""

from dataclasses import dataclass

from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    """Тело запроса `POST /embed`.

    Attributes:
        texts: список текстов для кодирования (минимум один).
    """

    texts: list[str] = Field(..., min_length=1)


@dataclass
class EmbedResponse:
    """Результат кодирования (ответ `POST /embed`).

    Attributes:
        model: имя модели из конфигурации (`MODEL_NAME`).
        dim: размерность одного вектора (0, если текстов не было).
        count: сколько векторов вернули (== числу входных текстов).
        embeddings: список векторов, по одному на каждый входной текст.
    """

    model: str
    dim: int
    count: int
    embeddings: list[list[float]]


@dataclass
class HealthResponse:
    """Состояние сервиса и воркера (ответ `GET /health`).

    Attributes:
        status: всегда "ok", если процесс жив и отвечает.
        model: имя модели из конфигурации (`MODEL_NAME`).
        loaded: загружена ли модель в RAM прямо сейчас (False — выгружена по
            простою либо ещё не было ни одного запроса).
        queue: сколько запросов сейчас ждёт в очереди воркера.
    """

    status: str
    model: str
    loaded: bool
    queue: int
