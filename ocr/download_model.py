"""
Разовая загрузка моделей EasyOCR в кеш.

EasyOCR сам качает модели детекции и распознавания по языкам (OCR_LANGS) при
первом создании Reader — каталог задаётся EASYOCR_MODULE_PATH. Этот скрипт просто
прогревает кеш заранее, чтобы сервис не качал модели во время первого запроса.
Запускать один раз.

Запуск:
    python download_model.py
"""

import os

import easyocr

LANGS = [x.strip() for x in os.environ.get("OCR_LANGS", "ru,en").split(",") if x.strip()]


def main() -> None:
    cache = os.environ.get("EASYOCR_MODULE_PATH", "по умолчанию (~/.EasyOCR)")
    print(f"Готовлю EasyOCR ({','.join(LANGS)}) — качаю модели в кеш ({cache})...")
    # Создание Reader скачивает все нужные модели.
    easyocr.Reader(
        LANGS, gpu=False, model_storage_directory=os.environ.get("EASYOCR_MODULE_PATH") or None
    )
    print("Готово. Модели лежат в кеше — сервис возьмёт их оттуда.")


if __name__ == "__main__":
    main()
