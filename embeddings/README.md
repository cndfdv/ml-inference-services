# Эмбеддинги

| Модель | Размерность | Контекст | Подготовка текста |
| --- | ---: | ---: | --- |
| [E5-small](e5-small/README.md) | 384 | 512 токенов | `query:` / `passage:` по роли |
| [USER-bge-m3](user-bge-m3/README.md) | 1024 | 8192 токена | без префикса, CLS + L2 |

Обе модели поддерживают CPU/CUDA, явные батчи, микробатчи между запросами и workers.
Все эмбеддеры: `bash scripts/compose.sh --mode cpu --profile embeddings up -d --build`.
