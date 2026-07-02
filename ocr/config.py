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
log: logging.Logger = logging.getLogger("ocr-svc")


# Расширения, которые принимаем. Картинки читаем через Pillow, PDF рендерим
# постранично через pypdfium2 (без системного poppler). Явный список отсекает
# мусорные загрузки.
DEFAULT_ACCEPTED_EXTS: frozenset[str] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".tiff",
        ".tif",
        ".bmp",
        ".webp",
        ".pdf",
    }
)


@dataclass(frozen=True)
class Settings:
    """Конфигурация сервиса (неизменяемая после старта).

    Поля заполняются из переменных окружения через `Settings.from_env()`.
    Значения по умолчанию совпадают с тем, что прописано в `.env.example`
    для локального запуска без докера.

    Attributes:
        langs: языки распознавания для EasyOCR (напр. ("ru", "en")). EasyOCR сам
            качает нужные модели при первом запуске. Русский совместим с
            английским в одном ридере.
        paragraph: группировать ли распознанные строки в абзацы (EasyOCR
            `paragraph`). False — по строкам (есть уверенность по каждой).
        pdf_dpi: с каким DPI рендерить страницы PDF в картинку перед OCR.
        idle_ttl: секунд простоя до выгрузки ридера из RAM.
        request_timeout: макс. ожидание результата (очередь + инференс), сек;
            дольше — клиент получает 504.
        tmp_dir: временная папка для загруженных файлов; файл удаляется сразу
            после обработки — сервис ничего не хранит.
        accepted_exts: допустимые расширения входных файлов.
        page_separator: разделитель, которым склеиваются страницы многостраничного
            PDF в общий текст ответа.
    """

    langs: tuple[str, ...] = ("ru", "en")
    paragraph: bool = False
    pdf_dpi: int = 150
    idle_ttl: int = 300
    request_timeout: float = 300.0
    tmp_dir: str = "tmp"
    accepted_exts: frozenset[str] = DEFAULT_ACCEPTED_EXTS
    page_separator: str = "\n\n---\n\n"

    @classmethod
    def from_env(cls) -> "Settings":
        """Собрать конфигурацию из переменных окружения."""
        langs_raw = os.environ.get("OCR_LANGS")
        langs = (
            tuple(x.strip() for x in langs_raw.split(",") if x.strip()) if langs_raw else cls.langs
        )
        paragraph_raw = os.environ.get("PARAGRAPH")
        paragraph = (
            paragraph_raw.strip().lower() in {"1", "true", "yes", "on"}
            if paragraph_raw is not None
            else cls.paragraph
        )
        return cls(
            langs=langs,
            paragraph=paragraph,
            pdf_dpi=int(os.environ.get("PDF_DPI", cls.pdf_dpi)),
            idle_ttl=int(os.environ.get("IDLE_TTL", cls.idle_ttl)),
            request_timeout=float(os.environ.get("REQUEST_TIMEOUT", cls.request_timeout)),
            tmp_dir=os.environ.get("TMP_DIR", cls.tmp_dir),
            page_separator=os.environ.get("PAGE_SEPARATOR", cls.page_separator),
        )


# Единый экземпляр конфигурации на весь процесс.
settings: Settings = Settings.from_env()
