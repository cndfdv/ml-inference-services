"""
Разовая загрузка модели PaddleOCR-VL с HuggingFace в кеш.

transformers сам качает веса и процессор по имени модели (MODEL_NAME) в кеш
HuggingFace (путь задаётся HF_HOME) при первом обращении. Этот скрипт просто
прогревает кеш заранее, чтобы сервис не качал веса во время первого запроса.
Запускать один раз.

Запуск:
    python download_model.py
"""

import os

from huggingface_hub import snapshot_download

MODEL_NAME = os.environ.get("MODEL_NAME", "PaddlePaddle/PaddleOCR-VL")


def main() -> None:
    print(
        f"Готовлю {MODEL_NAME} — качаю веса и процессор с HuggingFace в кеш "
        f"(HF_HOME={os.environ.get('HF_HOME', 'по умолчанию')})..."
    )
    # Скачиваем весь репозиторий модели (веса + конфиг + кастомный код + токенизатор)
    # в кеш HuggingFace — дальше сервис грузит модель оттуда без обращения к сети.
    snapshot_download(MODEL_NAME)
    print("Готово. Веса лежат в кеше HuggingFace — сервис возьмёт их оттуда.")


if __name__ == "__main__":
    main()
