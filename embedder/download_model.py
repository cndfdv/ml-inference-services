"""
Разовая загрузка модели эмбеддера с HuggingFace в кеш.

sentence-transformers сам качает модель по имени (напр. intfloat/multilingual-
e5-base) с HuggingFace в кеш — путь задаётся переменной HF_HOME. Этот скрипт
просто прогревает кеш заранее, чтобы сервис не качал модель во время первого
запроса. Запускать один раз.

Запуск:
    python download_model.py
"""

import os

from sentence_transformers import SentenceTransformer

MODEL_NAME = os.environ.get("MODEL_NAME", "intfloat/multilingual-e5-base")


def main() -> None:
    cache = os.environ.get("HF_HOME", "по умолчанию")
    print(f"Скачиваю модель {MODEL_NAME} с HuggingFace (кеш: {cache})...")
    SentenceTransformer(MODEL_NAME, device="cpu")
    print("Готово. Модель лежит в кеше HuggingFace — сервис возьмёт её оттуда.")


if __name__ == "__main__":
    main()
