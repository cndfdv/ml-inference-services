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


# Расширения, которые принимаем. Картинки открываем через Pillow, PDF рендерим
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
        model_name: имя HF-модели PaddleOCR-VL для transformers (напр.
            PaddlePaddle/PaddleOCR-VL, PaddlePaddle/PaddleOCR-VL-1.5). transformers
            сам качает её с HuggingFace в кеш (HF_HOME).
        model_class: какой Auto-класс использовать для загрузки —
            `auto` (пробуем image-text-to-text, затем causal-lm), либо явно
            `image_text_to_text` / `causal_lm` (разные версии модели требуют
            разные классы).
        device: устройство инференса (`cpu`, `cuda`). Сервис рассчитан на CPU.
        torch_dtype: тип весов (`float32`/`bfloat16`/`float16`). На CPU — float32
            (bfloat16/float16 на CPU медленные и не везде поддержаны).
        prompt: текстовый промпт задачи для VLM. `OCR:` — распознать текст; также
            бывают `Table Recognition:`, `Formula Recognition:`, `Chart Recognition:`.
        max_new_tokens: потолок генерации на одну страницу (больше — можно длиннее
            текст, но медленнее; на CPU каждый токен дорогой).
        pdf_dpi: с каким DPI рендерить страницы PDF в картинку перед OCR.
        idle_ttl: секунд простоя до выгрузки модели из RAM.
        request_timeout: макс. ожидание результата (очередь + инференс), сек;
            дольше — клиент получает 504. VLM на CPU медленный — таймаут большой.
        tmp_dir: временная папка для загруженных файлов; файл удаляется сразу
            после обработки — сервис ничего не хранит.
        accepted_exts: допустимые расширения входных файлов.
        page_separator: разделитель, которым склеиваются страницы многостраничного
            PDF в общий текст ответа.
    """

    model_name: str = "PaddlePaddle/PaddleOCR-VL"
    model_class: str = "auto"
    device: str = "cpu"
    torch_dtype: str = "float32"
    prompt: str = "OCR:"
    max_new_tokens: int = 2048
    pdf_dpi: int = 150
    idle_ttl: int = 300
    request_timeout: float = 600.0
    tmp_dir: str = "tmp"
    accepted_exts: frozenset[str] = DEFAULT_ACCEPTED_EXTS
    page_separator: str = "\n\n---\n\n"

    @classmethod
    def from_env(cls) -> "Settings":
        """Собрать конфигурацию из переменных окружения."""
        return cls(
            model_name=os.environ.get("MODEL_NAME", cls.model_name),
            model_class=os.environ.get("MODEL_CLASS", cls.model_class),
            device=os.environ.get("DEVICE", cls.device),
            torch_dtype=os.environ.get("TORCH_DTYPE", cls.torch_dtype),
            prompt=os.environ.get("OCR_PROMPT", cls.prompt),
            max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", cls.max_new_tokens)),
            pdf_dpi=int(os.environ.get("PDF_DPI", cls.pdf_dpi)),
            idle_ttl=int(os.environ.get("IDLE_TTL", cls.idle_ttl)),
            request_timeout=float(os.environ.get("REQUEST_TIMEOUT", cls.request_timeout)),
            tmp_dir=os.environ.get("TMP_DIR", cls.tmp_dir),
            page_separator=os.environ.get("PAGE_SEPARATOR", cls.page_separator),
        )


# Единый экземпляр конфигурации на весь процесс.
settings: Settings = Settings.from_env()
