"""
Поток-воркер, владеющий моделью эмбеддера.

Инференс на CPU тяжёлый и по своей сути однопоточный (внутри — BLAS/torch на
нескольких ядрах), поэтому гнать несколько кодирований параллельно смысла нет:
они только мешали бы друг другу за те же ядра. Вместо этого все запросы
выстраиваются в очередь, а обрабатывает их один выделенный поток-воркер. Приятный
побочный эффект: модель живёт только внутри воркера, к ней больше никто не лезет,
и никакие блокировки для защиты от гонок не нужны.

Модель не грузится на старте. Первый запрос поднимает её в RAM (cold start —
несколько секунд). Если запросов не было дольше IDLE_TTL, воркер сам выгружает
модель и освобождает память. Следующий запрос поднимет её заново.

Инференс — на sentence-transformers (torch, CPU): библиотека сама скачивает
модель с HuggingFace по имени (MODEL_NAME) в кеш HF_HOME.
"""

from __future__ import annotations

import gc
import os
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass

from config import log, settings

# Число потоков CPU фиксируем один раз при импорте модуля. torch читает
# OMP_NUM_THREADS и сам, но зададим явно для предсказуемости инференса.
_omp = os.environ.get("OMP_NUM_THREADS")
if _omp:
    try:
        import torch

        torch.set_num_threads(int(_omp))
    except Exception as exc:  # torch ещё не поставлен / кривое значение — не падаем
        log.warning("Не удалось задать число потоков torch (%s): %s", _omp, exc)


@dataclass
class _Job:
    """Одна единица работы для воркера: тексты и куда положить результат."""

    texts: list[str]
    future: Future[list[list[float]]]


class EmbedderWorker:
    """Поток-воркер: владеет моделью эмбеддера и обрабатывает очередь запросов.

    Всё, что касается модели (загрузка, инференс, выгрузка), происходит только
    здесь, в одном потоке. Поэтому состояние можно трогать без блокировок.
    """

    def __init__(self):
        # Очередь без ограничения по размеру: запросы под наплывом просто ждут
        # своей очереди, а не получают отказ.
        self._jobs: queue.Queue[_Job | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="embedder", daemon=True)
        # Доступ к модели — только из потока воркера. _loaded читает /health
        # из другого потока, поэтому это отдельный потокобезопасный флаг.
        self._model = None
        self._loaded = threading.Event()
        self._last_used = 0.0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        # Кладём «отравленную пилюлю», чтобы цикл воркера корректно завершился.
        self._jobs.put(None)
        self._thread.join(timeout=10)

    @property
    def loaded(self) -> bool:
        return self._loaded.is_set()

    @property
    def queue_size(self) -> int:
        return self._jobs.qsize()

    def submit(self, texts: list[str]) -> Future[list[list[float]]]:
        """Поставить тексты в очередь и получить future с результатом."""
        future: Future[list[list[float]]] = Future()
        self._jobs.put(_Job(texts, future))
        return future

    # ---- внутренняя кухня потока ----

    def _run(self) -> None:
        while True:
            # Спим на get, пока не придёт задача. Таймаут ставим ровно на момент
            # истечения простоя: проснуться раньше незачем, а проснувшись точно
            # в срок — сразу выгружаем модель. Если модель не загружена, timeout
            # = None: выгружать нечего, спим до задачи без холостых пробуждений.
            try:
                job = self._jobs.get(timeout=self._idle_timeout())
            except queue.Empty:
                # Проснулись по таймауту — значит простой истёк, пора выгружать.
                self._unload_if_idle()
                continue

            if job is None:  # сигнал на остановку
                break

            self._process(job)

    def _idle_timeout(self) -> float | None:
        """Сколько секунд спать на get. None — модель не загружена, ждём задачу."""
        if self._model is None:
            return None
        remaining = settings.idle_ttl - (time.monotonic() - self._last_used)
        return max(0.0, remaining)

    def _process(self, job: _Job) -> None:
        try:
            self._ensure_loaded()
            self._last_used = time.monotonic()
            vectors = self._embed(job.texts)
            self._last_used = time.monotonic()  # инференс долгий, обновляем после
            job.future.set_result(vectors)
        except Exception as exc:  # пробрасываем ошибку ожидающему запросу
            log.exception("Инференс упал")
            job.future.set_exception(exc)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        # Опциональный префикс к каждому тексту (для асимметричных моделей e5
        # обычно `query: ` — см. README и EMBED_PREFIX).
        inputs = [settings.embed_prefix + t for t in texts] if settings.embed_prefix else texts
        vecs = self._model.encode(
            inputs,
            batch_size=settings.batch_size,
            normalize_embeddings=settings.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # float32 → обычные списки Python для JSON-ответа.
        return vecs.astype("float32").tolist()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        log.info("Поднимаю модель %s...", settings.model_name)
        t0 = time.monotonic()
        from sentence_transformers import SentenceTransformer

        # Жёстко фиксируем CPU: сервис рассчитан только на него. sentence-
        # transformers сам скачает модель с HuggingFace по имени (в кеш HF_HOME)
        # при первом обращении, если её там ещё нет.
        self._model = SentenceTransformer(settings.model_name, device="cpu")
        self._loaded.set()
        log.info("Модель загружена за %.1f c.", time.monotonic() - t0)

    def _unload_if_idle(self) -> None:
        if self._model is None:
            return
        idle = time.monotonic() - self._last_used
        if idle < settings.idle_ttl:
            return
        log.info("Простой %.0f c >= %d c — выгружаю модель из RAM.", idle, settings.idle_ttl)
        self._model = None
        self._loaded.clear()
        gc.collect()
