"""
Настройки сервиса и общий логгер.

Конфиг целиком через переменные окружения: в докере — из env_file, локально —
из .env (если установлен python-dotenv). Складываем всё в типизированный
`Settings`, готовый `settings` дальше используют остальные модули.
"""

import logging
import os
from dataclasses import dataclass

# Локальный запуск без докера: подтянем .env, если он рядом. В докере переменные
# уже в окружении, и этот блок просто ничего не находит.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log: logging.Logger = logging.getLogger("embedder-svc")


def _env_bool(name: str, default: bool) -> bool:
    """Булев флаг из env: 1/true/yes/on — да, всё прочее — нет."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Настройки сервиса, замороженные на время жизни процесса.

    Поля берутся из окружения в `Settings.from_env()`. Дефолты здесь — те же, что
    в `.env.example`, чтобы сервис поднимался локально и без своего `.env`.

    Attributes:
        model_name: имя HF-модели для sentence-transformers (напр.
            deepvk/USER-bge-m3). Библиотека сама качает её с HuggingFace в кеш
            (HF_HOME) при первом обращении.
        idle_ttl: секунд простоя до выгрузки модели из RAM.
        request_timeout: макс. ожидание результата (очередь + инференс), сек;
            дольше — клиент получает 504.
        batch_size: размер батча при кодировании текстов.
        normalize: L2-нормализовать ли выходные векторы (удобно для косинуса).
        embed_prefix: префикс, добавляемый к каждому тексту перед кодированием.
            Дефолтная bge-m3 симметричная — префикс ей не нужен, поэтому пусто.
            Нужен для асимметричных моделей e5 (`query: ` / `passage: `, см.
            README). Пустая строка — префикс не добавлять.
        max_texts: предел числа текстов в одном запросе (защита от OOM).
    """

    model_name: str = "deepvk/USER-bge-m3"
    idle_ttl: int = 120
    request_timeout: float = 300.0
    batch_size: int = 32
    normalize: bool = True
    embed_prefix: str = ""
    max_texts: int = 256

    @classmethod
    def from_env(cls) -> "Settings":
        """Собрать конфигурацию из переменных окружения."""
        return cls(
            model_name=os.environ.get("MODEL_NAME", cls.model_name),
            idle_ttl=int(os.environ.get("IDLE_TTL", cls.idle_ttl)),
            request_timeout=float(os.environ.get("REQUEST_TIMEOUT", cls.request_timeout)),
            batch_size=int(os.environ.get("BATCH_SIZE", cls.batch_size)),
            normalize=_env_bool("NORMALIZE_EMBEDDINGS", cls.normalize),
            embed_prefix=os.environ.get("EMBED_PREFIX", cls.embed_prefix),
            max_texts=int(os.environ.get("MAX_TEXTS", cls.max_texts)),
        )


# Один экземпляр настроек на весь процесс.
settings: Settings = Settings.from_env()
