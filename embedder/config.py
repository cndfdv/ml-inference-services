"""
Конфигурация сервиса и общий логгер.

Всё настраивается через переменные окружения. В докере они приходят через
env_file, при локальном запуске — подхватываются из .env (если установлен
python-dotenv). Значения собраны в один типизированный `Settings` (dataclass),
а готовый экземпляр `settings` импортируют остальные модули.
"""

import logging
import os
from dataclasses import dataclass

# Локальный запуск без докера: подхватываем .env, если он есть. В докере
# переменные приходят через env_file, и этот блок просто ничего не находит.
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
    """Прочитать булев флаг из env: 1/true/yes/on — истина, остальное — ложь."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Конфигурация сервиса (неизменяемая после старта).

    Поля заполняются из переменных окружения через `Settings.from_env()`.
    Значения по умолчанию совпадают с тем, что прописано в `.env.example`
    для локального запуска без докера.

    Attributes:
        model_name: имя HF-модели для sentence-transformers (напр.
            intfloat/multilingual-e5-base). Библиотека сама качает её с
            HuggingFace в кеш (HF_HOME) при первом обращении.
        idle_ttl: секунд простоя до выгрузки модели из RAM.
        request_timeout: макс. ожидание результата (очередь + инференс), сек;
            дольше — клиент получает 504.
        batch_size: размер батча при кодировании текстов.
        normalize: L2-нормализовать ли выходные векторы (удобно для косинуса).
        embed_prefix: префикс, добавляемый к каждому тексту перед кодированием.
            Для асимметричных моделей e5 обычно `query: ` (см. README). Пустая
            строка — префикс не добавлять.
        max_texts: предел числа текстов в одном запросе (защита от OOM).
    """

    model_name: str = "intfloat/multilingual-e5-base"
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


# Единый экземпляр конфигурации на весь процесс.
settings: Settings = Settings.from_env()
