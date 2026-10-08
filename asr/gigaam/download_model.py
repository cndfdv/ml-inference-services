"""
Разовый прогрев кеша: тянем ONNX-модель GigaAM с HuggingFace заранее.

onnx-asr сам качает модель по имени (напр. gigaam-v3-ctc) из репозитория
istupakov/gigaam-v3-onnx в кеш HuggingFace (путь — HF_HOME). Гоняем это один
раз до старта, чтобы первый запрос к сервису не ждал скачивания.

Запуск:
    python download_model.py
"""

import os

import onnx_asr

os.environ.setdefault("HF_HOME", os.environ.get("GIGAAM_HF_HOME", "/app/models"))

MODEL_VERSION = os.environ.get("GIGAAM_MODEL_VERSION", os.environ.get("MODEL_VERSION", "gigaam-v3-ctc"))


def main() -> None:
    print(
        f"Скачиваю модель {MODEL_VERSION} с HuggingFace (кеш: {os.environ.get('GIGAAM_HF_HOME', 'по умолчанию')})..."
    )
    onnx_asr.load_model(MODEL_VERSION, providers=["CPUExecutionProvider"])
    print("Готово. Модель в кеше HuggingFace — сервис подхватит её оттуда.")


if __name__ == "__main__":
    main()
