# Распознавание речи

| Модель | Устройство | Особенности |
| --- | --- | --- |
| [GigaAM](gigaam/README.md) | CPU | v3 CTC, окна 30 секунд, выгрузка по простою |
| [Whisper](whisper/README.md) | CPU / CUDA | large-v3, языки, перевод, сегменты с временем |

Один контейнер: `bash scripts/run.sh asr/whisper --device cuda --workers 1` из корня.
Все ASR: `bash scripts/compose.sh --mode cpu --profile asr up -d --build`.
