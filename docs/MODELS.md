# Закреплённые модели

Источник истины — `model.lock.json` в каждой папке E5/USER/Rapid/Whisper.
SHA-256 записывается в `/models/<model>/manifest.json` внутри volume контейнера. В Git нет весов и документов.

| API-модель | Источник | Ревизия |
| --- | --- | --- |
| e5-small | intfloat/multilingual-e5-small | 614241f622f53c4eeff9890bdc4f31cfecc418b3 |
| user-bge-m3 | deepvk/USER-bge-m3 | 0cc6cfe48e260fb0474c753087a69369e88709ae |
| rapid-v5-mobile | RapidOCR 3.9.2, PP-OCRv5 mobile Cyrillic | SHA-256 detector/recognizer в lock |
| whisper-large-v3 | Systran/faster-whisper-large-v3 | edaa852ec7e145841d8ffdb056a99866b5f0a478 |

E5 использует masked mean pooling, USER — первый CLS-токен. Векторы нормализованы.
USER FP32 — ONNX opset 17 с динамическими размерами batch/sequence.
Публичный preparation может экспортировать граф из закреплённого checkpoint;
на home скопирован уже измеренный граф. Манифест фиксирует конкретные байты.

Динамический INT8-граф USER предназначен для CPU. Его integer-операции не
поддерживаются CUDA Execution Provider так же, как FP32-граф:
[официальное руководство Hugging Face](https://huggingface.co/docs/optimum-onnx/onnxruntime/usage_guides/gpu).
CUDA и cuDNN должны соответствовать ONNX Runtime:
[таблица совместимости](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

Источники и условия моделей:

- [E5-small, model card, MIT](https://huggingface.co/intfloat/multilingual-e5-small).
- [USER-bge-m3, model card](https://huggingface.co/deepvk/USER-bge-m3).
- [RapidOCR, Apache-2.0](https://github.com/RapidAI/RapidOCR/tree/v3.9.2).
- [PaddleOCR, Apache-2.0](https://github.com/PaddlePaddle/PaddleOCR).
- [Whisper large-v3 conversion, MIT](https://huggingface.co/Systran/faster-whisper-large-v3).
- [faster-whisper, MIT](https://github.com/SYSTRAN/faster-whisper).

Условия сторонних весов действуют независимо от лицензии репозитория.
Публикация кода не распространяет веса и не меняет их условия.

GigaAM и EasyOCR сохраняют прежние CPU backend и схему выбора весов. Их
checkpoint не закреплён SHA-256 так же, как у четырёх перечисленных моделей.
См. README [GigaAM](../asr/gigaam/README.md) и [EasyOCR](../ocr/easyocr/README.md).
