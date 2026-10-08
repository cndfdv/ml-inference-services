# Распознавание документов

| Модель | Устройство | Особенности |
| --- | --- | --- |
| [EasyOCR](easyocr/README.md) | CPU | ru/en, квантование с AVX2, выгрузка по простою |
| [RapidOCR v5 mobile](rapid-v5-mobile/README.md) | CPU / CUDA | кириллический recognizer, координаты строк, батчи |

Обе модели принимают изображения и PDF. Их веса и алгоритмы разные; они не
являются взаимозаменяемыми при проверке одинакового качества теста и прода.
Все OCR: `bash scripts/compose.sh --mode cpu --profile ocr up -d --build`.
