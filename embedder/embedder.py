"""
Поток-воркер с моделью эмбеддера.

Кодирование на CPU и так занимает все ядра через BLAS/torch — гнать несколько
запросов разом бессмысленно, они лишь мешали бы друг другу. Поэтому один воркер
и одна очередь: модель ни с кем не делится, гонок нет, локи не нужны.

Модель поднимается лениво по первому запросу (cold start — несколько секунд) и
выгружается из RAM после IDLE_TTL простоя; следующий запрос грузит её заново.

Бэкенд — sentence-transformers (torch, CPU): сам качает модель с HuggingFace по
имени (MODEL_NAME) в кеш HF_HOME.
"""

from __future__ import annotations

import gc
import os
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass

import torch
from config import log, settings
from sentence_transformers import SentenceTransformer

# Число CPU-потоков torch задаём один раз при импорте. Он и сам читает
# OMP_NUM_THREADS, но выставим явно — так инференс предсказуемее.
_omp = os.environ.get("OMP_NUM_THREADS")
if _omp:
    try:
        torch.set_num_threads(int(_omp))
    except ValueError as exc:  # мусор в переменной — не роняем сервис
        log.warning("Не удалось задать число потоков torch (%s): %s", _omp, exc)


@dataclass
class _Job:
    """Задача для воркера: тексты и future, куда вернуть векторы."""

    texts: list[str]
    future: Future[list[list[float]]]


class EmbedderWorker:
    """Поток-воркер: держит модель эмбеддера и разгребает очередь запросов.

    Модель трогает только этот поток — грузит, кодирует, выгружает. Раз доступ
    из одного потока, состояние живёт без блокировок.
    """

    def __init__(self):
        # Очередь без лимита: при наплыве запросы ждут, а не отбиваются отказом.
        self._jobs: queue.Queue[_Job | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="embedder", daemon=True)
        # К модели ходит только воркер. А флаг _loaded читает /health из другого
        # потока — поэтому потокобезопасный Event, а не просто bool.
        self._model = None
        self._loaded = threading.Event()
        self._last_used = 0.0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        # Кидаем None — воркер увидит его в очереди и завершит цикл.
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

    # ---- потроха воркера ----

    def _run(self) -> None:
        while True:
            # Ждём задачу на get. Таймаут — ровно до конца простоя: проснуться
            # раньше незачем, а точно в срок — сразу выгрузим модель. Модель не
            # загружена → timeout=None: выгружать нечего, спим до задачи.
            try:
                job = self._jobs.get(timeout=self._idle_timeout())
            except queue.Empty:
                # Проснулись по таймауту — значит простой вышел, выгружаем.
                self._unload_if_idle()
                continue

            if job is None:  # сигнал остановки из stop()
                break

            self._process(job)

    def _idle_timeout(self) -> float | None:
        """Сколько спать на get. None — модель не загружена, ждём задачу."""
        if self._model is None:
            return None
        remaining = settings.idle_ttl - (time.monotonic() - self._last_used)
        return max(0.0, remaining)

    def _process(self, job: _Job) -> None:
        try:
            self._ensure_loaded()
            self._last_used = time.monotonic()
            vectors = self._embed(job.texts)
            self._last_used = time.monotonic()  # инференс долгий, отметимся ещё раз
            job.future.set_result(vectors)
        except Exception as exc:  # отдаём ошибку тому, кто ждёт результат
            log.exception("Инференс упал")
            job.future.set_exception(exc)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        # Необязательный префикс перед каждым текстом. Дефолтной bge-m3 он не
        # нужен; пригодится для асимметричных e5 (`query: `, см. EMBED_PREFIX).
        inputs = [settings.embed_prefix + t for t in texts] if settings.embed_prefix else texts
        vecs = self._model.encode(
            inputs,
            batch_size=settings.batch_size,
            normalize_embeddings=settings.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # float32 → обычные python-списки, чтобы уехало в JSON.
        return vecs.astype("float32").tolist()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        log.info("Поднимаю модель %s...", settings.model_name)
        t0 = time.monotonic()
        # Только CPU — под него сервис и рассчитан. sentence-transformers сам
        # стянет модель с HuggingFace по имени (в кеш HF_HOME), если её там нет.
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
