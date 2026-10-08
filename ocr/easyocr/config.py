"""
Настройки сервиса и общий логгер.

Всё задаётся переменными окружения: в докере — через env_file, локально —
через .env (если установлен python-dotenv). Держим их в одном типизированном
`Settings`, а `settings` дальше берут остальные модули.
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
log: logging.Logger = logging.getLogger("ocr-svc")


# Что принимаем на вход. Картинки читаем через Pillow, PDF рендерим постранично
# через pypdfium2 (системный poppler не нужен). Явный список отсекает заведомо
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
    """Настройки сервиса, замороженные на время жизни процесса.

    Поля берутся из окружения в `Settings.from_env()`. Дефолты здесь — те же, что
    в `.env.example`, чтобы сервис поднимался локально и без своего `.env`.

    Attributes:
        langs: языки распознавания для EasyOCR (напр. ("ru", "en")). EasyOCR сам
            качает нужные модели при первом запуске. Русский совместим с
            английским в одном ридере.
        paragraph: группировать ли распознанные строки в абзацы (EasyOCR
            `paragraph`). False — по строкам (есть уверенность по каждой).
        quantize: включать int8-квантизацию распознавателя EasyOCR (быстрее и
            меньше RAM). Бэкенд квантизации fbgemm требует AVX2 — на CPU без
            AVX2 (напр. дефолтный QEMU-CPU) квантизованные операции падают с
            SIGILL. Поэтому фактическое значение дополнительно гейтится наличием
            AVX2 (см. ocr.py): здесь True лишь разрешает квантизацию там, где она
            поддерживается.
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
    quantize: bool = True
    pdf_dpi: int = 150
    idle_ttl: int = 300
    request_timeout: float = 300.0
    tmp_dir: str = "tmp"
    accepted_exts: frozenset[str] = DEFAULT_ACCEPTED_EXTS
    page_separator: str = "\n\n---\n\n"

    @classmethod
    def from_env(cls) -> "Settings":
        """Собрать конфигурацию из переменных окружения."""
        langs_raw = os.environ.get("EASYOCR_OCR_LANGS", os.environ.get("OCR_LANGS"))
        langs = (
            tuple(x.strip() for x in langs_raw.split(",") if x.strip()) if langs_raw else cls.langs
        )
        paragraph_raw = os.environ.get("EASYOCR_PARAGRAPH", os.environ.get("PARAGRAPH"))
        paragraph = (
            paragraph_raw.strip().lower() in {"1", "true", "yes", "on"}
            if paragraph_raw is not None
            else cls.paragraph
        )
        quantize_raw = os.environ.get("EASYOCR_QUANTIZE", os.environ.get("QUANTIZE"))
        quantize = (
            quantize_raw.strip().lower() in {"1", "true", "yes", "on"}
            if quantize_raw is not None
            else cls.quantize
        )
        return cls(
            langs=langs,
            paragraph=paragraph,
            quantize=quantize,
            pdf_dpi=int(os.environ.get("EASYOCR_PDF_DPI", os.environ.get("PDF_DPI", cls.pdf_dpi))),
            idle_ttl=int(os.environ.get("EASYOCR_IDLE_TTL", os.environ.get("IDLE_TTL", cls.idle_ttl))),
            request_timeout=float(os.environ.get("EASYOCR_REQUEST_TIMEOUT", os.environ.get("REQUEST_TIMEOUT", cls.request_timeout))),
            tmp_dir=os.environ.get("EASYOCR_TMP_DIR", os.environ.get("TMP_DIR", cls.tmp_dir)),
            page_separator=os.environ.get("EASYOCR_PAGE_SEPARATOR", os.environ.get("PAGE_SEPARATOR", cls.page_separator)),
        )


# Один экземпляр настроек на весь процесс.
settings: Settings = Settings.from_env()
