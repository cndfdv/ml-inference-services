"""
Разовый прогрев кеша: тянем модель эмбеддера с HuggingFace заранее.

sentence-transformers сам качает модель по имени (напр. deepvk/USER-bge-m3) в
кеш HuggingFace (путь — HF_HOME). Гоняем один раз до старта, чтобы первый запрос
к сервису не ждал скачивания.

Запуск:
    python download_model.py
"""

import os

from sentence_transformers import SentenceTransformer

MODEL_NAME = os.environ.get("MODEL_NAME", "deepvk/USER-bge-m3")


def main() -> None:
    cache = os.environ.get("HF_HOME", "по умолчанию")
    print(f"Скачиваю модель {MODEL_NAME} с HuggingFace (кеш: {cache})...")
    SentenceTransformer(MODEL_NAME, device="cpu")
    print("Готово. Модель в кеше HuggingFace — сервис подхватит её оттуда.")


if __name__ == "__main__":
    main()
