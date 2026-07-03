"""
Разовая загрузка ONNX-модели GigaAM с HuggingFace в кеш.

onnx-asr сам качает модель по имени (напр. gigaam-v3-ctc) с HuggingFace
(репозиторий istupakov/gigaam-v3-onnx) в кеш HuggingFace — путь задаётся
переменной HF_HOME. Этот скрипт просто прогревает кеш заранее, чтобы сервис не
качал модель во время первого запроса. Запускать один раз.

Запуск:
    python download_model.py
"""

import os

import onnx_asr

MODEL_VERSION = os.environ.get("MODEL_VERSION", "gigaam-v3-ctc")


def main() -> None:
    print(
        f"Скачиваю модель {MODEL_VERSION} с HuggingFace (кеш: {os.environ.get('HF_HOME', 'по умолчанию')})..."
    )
    onnx_asr.load_model(MODEL_VERSION, providers=["CPUExecutionProvider"])
    print("Готово. Модель лежит в кеше HuggingFace — сервис возьмёт её оттуда.")


if __name__ == "__main__":
    main()
