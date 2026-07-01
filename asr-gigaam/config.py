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
log: logging.Logger = logging.getLogger("gigaam-svc")


# Расширения, которые принимаем. Декодирует всё равно ffmpeg, ему формат
# контейнера почти безразличен, но явный список отсекает мусорные загрузки.
DEFAULT_ACCEPTED_EXTS: frozenset[str] = frozenset(
    {
        ".wav",
        ".mp3",
        ".mp4",
        ".m4a",
        ".flac",
        ".ogg",
        ".opus",
        ".aac",
        ".webm",
        ".wma",
    }
)


@dataclass(frozen=True)
class Settings:
    """Конфигурация сервиса (неизменяемая после старта).

    Поля заполняются из переменных окружения через `Settings.from_env()`.
    Значения по умолчанию совпадают с тем, что прописано в `.env.example`
    для локального запуска без докера.

    Attributes:
        model_version: имя модели для onnx-asr (напр. gigaam-v3-ctc,
            gigaam-v3-e2e-ctc). onnx-asr сам качает её с HuggingFace в кеш.
        idle_ttl: секунд простоя до выгрузки модели из RAM.
        request_timeout: макс. ожидание результата (очередь + инференс), сек;
            дольше — клиент получает 504.
        tmp_dir: временная папка для загруженных аудио; файл удаляется сразу
            после обработки — сервис ничего не хранит.
        accepted_exts: допустимые расширения входных файлов.
        chunk_sec: длина окна, на которые режется длинное аудио перед
            инференсом, сек. Энкодер по памяти растёт ~квадратично от длины,
            поэтому длинный файл целиком уводит процесс в OOM — режем на окна.
            0 — не резать (только для коротких файлов).
        chunk_batch_size: сколько окон гонится через инференс одним батчем.
            Больше — быстрее, но и пик памяти выше.
        audio_filters: цепочка аудиофильтров ffmpeg (значение `-af`),
            применяется к входному аудио перед распознаванием. Пустая строка —
            фильтры не применять (штатный декод gigaam). Помогает на тихой/шумной
            записи; на чистом аудио эффект небольшой.
    """

    model_version: str = "gigaam-v3-ctc"
    idle_ttl: int = 60
    request_timeout: float = 3000.0
    tmp_dir: str = "tmp"
    accepted_exts: frozenset[str] = DEFAULT_ACCEPTED_EXTS
    chunk_sec: int = 30
    chunk_batch_size: int = 8
    audio_filters: str = "highpass=f=100,dynaudnorm"

    @classmethod
    def from_env(cls) -> "Settings":
        """Собрать конфигурацию из переменных окружения."""
        return cls(
            model_version=os.environ.get("MODEL_VERSION", cls.model_version),
            idle_ttl=int(os.environ.get("IDLE_TTL", cls.idle_ttl)),
            request_timeout=float(
                os.environ.get("REQUEST_TIMEOUT", cls.request_timeout)
            ),
            tmp_dir=os.environ.get("TMP_DIR", cls.tmp_dir),
            chunk_sec=int(os.environ.get("CHUNK_SEC", cls.chunk_sec)),
            chunk_batch_size=int(
                os.environ.get("CHUNK_BATCH_SIZE", cls.chunk_batch_size)
            ),
            audio_filters=os.environ.get("AUDIO_FILTERS", cls.audio_filters),
        )


# Единый экземпляр конфигурации на весь процесс.
settings: Settings = Settings.from_env()
