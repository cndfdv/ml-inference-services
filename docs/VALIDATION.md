# Проверки на home

Проверено 8 октября 2026 года: Ubuntu 24.04 / WSL2, RTX 4060 Ti 16 GB,
Docker 29.8.1, Compose 5.5.1. GPU доступен через `/dev/dxg` и read-only
`/usr/lib/wsl`; применяется `compose.wsl.yaml`. Другие проекты продолжают работать.

## Исходная проверка до рефакторинга: API и запуск

- 17 unit-тестов: маршруты четырёх моделей, порядок результатов, очереди,
  таймауты, ограничение HTTP-тела и аудио, OCR/PDF, манифесты весов,
  токенизированные батчи и проверка CUDA для обоих этапов RapidOCR.
- Собран общий Docker-образ. Четыре GPU-контейнера проходят readiness;
  `/v1/models` сообщает `cuda` и CUDA Execution Provider для ONNX.
- `scripts/smoke.py`: явные батчи, нормализованные векторы ожидаемой размерности,
  по 16 HTTP-запросов с восемью клиентами к каждому эмбеддеру, пакет пустых OCR-страниц.
- После переноса в Git checkout повторный smoke прошёл. Реальный multipart OCR
  вернул 14 строк с geometry/confidence; Whisper file batch из двух JFK-записей
  вернул две одинаковые непустые расшифровки. Проверки GitHub Actions прошли.
- Реальные CPU backend-проверки всех четырёх моделей в контейнерах:
  `scripts/check_devices.py`. На CPU/GPU одинаковы ревизии, SHA-256 весов,
  preprocessing, precision и contract fingerprint.
- Отдельный E5 CPU-контейнер с `WORKERS=2`: оба процесса ответили на HTTP-запросы
  и выполнили эмбеддинг (`scripts/check_workers.py`, два разных PID).
  GPU запускается с одним воркером на модель; увеличение GPU workers по памяти
  при одновременном запуске четырёх моделей здесь не проверялось.
- USER дополнительно экспортирован заново из закреплённого HF snapshot в
  отдельный bundle с UID хоста. Закреплён legacy ONNX exporter (`dynamo=False`),
  как в бенчмарке. Проверены батчи 1/3/4, короткий и длинный текст, обе роли:
  минимальный cosine нового CPU-графа против GPU reference — 0.9999999999856.

## Поиск: CPU reference против GPU HTTP

Использован сохранённый корпус локального `russian-embedding-benchmark`:
168 документов, 45 tune-запросов и 20 holdout-запросов. Хеши входов проверены
по summary. Метрика — точный dense NDCG@10 по бинарным relevant-документам,
без индекса ANN и без прикладного сервиса поиска. Минимальный cosine между
CPU/GPU-векторами: 0.99999988. Изменений top-1 нет.

| Модель | Раздел | CPU FP32 NDCG@10 | GPU FP32 NDCG@10 | CPU INT8 NDCG@10 |
| --- | --- | ---: | ---: | ---: |
| E5-small | tune | 0.861854 | 0.861854 | — |
| E5-small | holdout | 0.859582 | 0.859582 | — |
| USER-bge-m3 | tune | 0.867455 | 0.867455 | 0.857050 |
| USER-bge-m3 | holdout | 0.960658 | 0.960658 | 0.958424 |

На этих данных FP32 не ухудшил CPU INT8: разница +1.0404 и +0.2234 процентного
пункта соответственно. Это не означает тождество INT8 и FP32 на других данных.
Для одинакового контракта в тесте и проде выбирайте FP32 на обоих устройствах.

## OCR: CPU predictions против GPU HTTP

На 60 страницах нормализованный текст совпал: 36 синтетических и 24 публичных
скана. Нормализация — Unicode NFKC, lowercase и пробелы; CER — расстояние
Левенштейна по символам относительно эталонного текста, делённое на число символов.

| Корпус | Раздел | Страниц | CPU/GPU CER |
| --- | --- | ---: | ---: |
| synthetic | tune | 24 | 0.044674 |
| synthetic | holdout | 12 | 0.053108 |
| public | tune | 12 | 0.185941 |
| public | holdout | 12 | 0.103293 |

Деградация CER — 0. Координаты и confidence возвращаются API, но численное
равенство геометрии на всём корпусе здесь не оценивалось.

## Whisper

Одна 11-секундная английская запись JFK из
[upstream faster-whisper v1.2.1](https://github.com/SYSTRAN/faster-whisper/blob/v1.2.1/tests/data/jfk.flac),
SHA-256 `63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715`.
CPU и GPU выдали одинаковую непустую расшифровку; нормализованный WER между ними
равен 0. Ограниченный по длительности PyAV decoder дал тот же массив mono 16 kHz,
что upstream `decode_audio`. Язык для этого теста задан явно: `en`.

Это проверка устройства и decoder, а не оценка качества русского ASR.
Для прикладной гарантии нужен русский корпус с эталонными транскриптами.

## Повторение

Ниже зафиксированы результаты исходных checkpoint перед переносом кода.
Проверки независимых контейнеров после рефакторинга добавляются отдельным разделом.

JSON-отчёты на home находятся в `artifacts/` и исключены из Git; исходные корпуса
читаются на месте и не публикуются. Порог поиска/CER по умолчанию допускает
0.5 процентного пункта деградации; фактическая деградация FP32 составила 0.

```bash
cd /home/knyze/ml-inference-services
bash scripts/compose.sh --mode wsl config --quiet
python3 scripts/smoke.py > artifacts/smoke.json
# Для quality используйте отдельное окружение с лёгкими зависимостями:
python3 -m venv .venv-check
.venv-check/bin/pip install -r requirements-dev.txt
.venv-check/bin/python scripts/check_quality.py \
  --benchmark /home/knyze/russian-embedding-benchmark \
  --ocr-root /home/knyze/russian-ocr-benchmark > artifacts/quality.json

# Реальный CPU backend против работающего GPU HTTP. Подставьте модель и порт.
DOCKER_CONFIG="$PWD/.docker-client" docker run --rm --network host \
  -v "$PWD/scripts:/checks:ro" -v ml-services-e5-small-weights:/models:ro \
  ml-services-e5-small:local \
  python /checks/check_devices.py --model e5-small --gpu-base http://127.0.0.1:18101
```

Для OCR добавьте read-only image mount и `--image`; для Whisper — audio mount,
`--audio` и `--language en` для JFK. Запускайте CPU-проверки последовательно:
large-v3 требует существенного объёма RAM даже без GPU.

Гарантия ограничена этими данными. Продолжительные аудиозаписи, несколько GPU
воркеров и конкурентные USER-запросы по 8192 токена не проходили нагрузочный тест.
На 16 GB GPU после обработки всех корпусов четыре FP32-сервиса почти полностью
занимают доступную VRAM. Перед увеличением батчей или workers измеряйте память.
