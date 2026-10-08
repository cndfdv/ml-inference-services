"""
Разовый прогрев кеша: тянем модели EasyOCR заранее.

EasyOCR сам качает модели детекции и распознавания по языкам (OCR_LANGS) при
первом создании Reader — каталог задаётся EASYOCR_MODULE_PATH. Гоняем один раз
до старта, чтобы первый запрос к сервису не ждал скачивания.

Запуск:
    python download_model.py
"""

import os

import easyocr

LANGS = [x.strip() for x in os.environ.get("EASYOCR_OCR_LANGS", os.environ.get("OCR_LANGS", "ru,en")).split(",") if x.strip()]


def main() -> None:
    cache = os.environ.get("EASYOCR_MODULE_PATH", "по умолчанию (~/.EasyOCR)")
    print(f"Готовлю EasyOCR ({','.join(LANGS)}) — качаю модели в кеш ({cache})...")
    # Сам факт создания Reader стягивает все нужные модели.
    easyocr.Reader(
        LANGS, gpu=False, model_storage_directory=os.environ.get("EASYOCR_MODULE_PATH") or None
    )
    print("Готово. Модели в кеше — сервис подхватит их оттуда.")


if __name__ == "__main__":
    main()
