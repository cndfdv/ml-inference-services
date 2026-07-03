"""
Поток-воркер, владеющий ASR-моделью.

Распознавание на CPU тяжёлое и однопоточное по своей сути, поэтому гнать
несколько инференсов параллельно смысла нет — они только мешали бы друг другу.
Вместо этого все запросы выстраиваются в очередь, а обрабатывает их один
выделенный поток-воркер. У такого подхода приятный побочный эффект: модель
живёт только внутри воркера, к ней больше никто не лезет, и никакие блокировки
для защиты от гонок не нужны.

Модель не грузится на старте. Первый запрос поднимает её в RAM (cold start —
несколько секунд). Если запросов не было дольше IDLE_TTL, воркер сам выгружает
модель и освобождает память. Следующий запрос поднимет её заново.

Инференс — на onnxruntime через библиотеку onnx-asr: она сама скачивает ONNX-
модель GigaAM с HuggingFace по имени (напр. gigaam-v3-ctc) и делает препроцессинг
без torch.
"""

from __future__ import annotations

import gc
import os
import queue
import subprocess
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass

import numpy as np
import onnx_asr
from config import log, settings

# Частота дискретизации, к которой приводим любое входное аудио. Совпадает с той,
# на которой обучалась модель; onnx-asr ожидает 16 кГц моно.
SAMPLE_RATE = 16000


@dataclass
class _Job:
    """Одна единица работы для воркера: путь к аудио и куда положить результат."""

    audio_path: str
    future: Future[str]


class TranscriberWorker:
    """Поток-воркер: владеет ASR-моделью и обрабатывает очередь запросов.

    Всё, что касается модели (загрузка, инференс, выгрузка), происходит только
    здесь, в одном потоке. Поэтому состояние можно трогать без блокировок.
    """

    def __init__(self):
        # Очередь без ограничения по размеру: запросы под наплывом просто ждут
        # своей очереди, а не получают отказ.
        self._jobs: queue.Queue[_Job | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="transcriber", daemon=True)
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

    def submit(self, audio_path: str) -> Future[str]:
        """Поставить аудио в очередь и получить future с результатом."""
        future: Future[str] = Future()
        self._jobs.put(_Job(audio_path, future))
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
            text = self._transcribe(job.audio_path)
            self._last_used = time.monotonic()  # инференс долгий, обновляем после
            job.future.set_result(text)
        except Exception as exc:  # пробрасываем ошибку ожидающему запросу
            log.exception("Инференс упал")
            job.future.set_exception(exc)
        finally:
            # Аудио — временное. Сервис ничего не хранит (хранение и БД — в
            # другом сервисе), поэтому файл удаляем сразу после обработки —
            # успешной или нет. Удаляет именно воркер, а не HTTP-слой: при 504
            # запрос перестаёт ждать результат, но файл всё ещё нужен воркеру,
            # пока тот не дочитает его в _transcribe.
            self._delete(job.audio_path)

    @staticmethod
    def _delete(path: str) -> None:
        try:
            os.remove(path)
        except OSError as exc:
            log.warning("Не удалось удалить временный файл %s: %s", path, exc)

    @staticmethod
    def _decode_audio(audio_path: str):
        """Декодировать аудио в waveform (numpy float32, моно, 16 кГц) через ffmpeg.

        Почему сами, а не отдаём путь в onnx-asr: recognize по пути читает только
        PCM-wav, а мы принимаем mp3/mp4/m4a и пр. ffmpeg декодирует любой
        контейнер и на том же проходе применяет предобработку (AUDIO_FILTERS).
        Выдаём сырой PCM f32 на stdout и читаем его в массив — это и есть формат,
        который ждёт recognize (float32 в диапазоне [-1, 1]).
        """
        cmd = ["ffmpeg", "-nostdin", "-threads", "1", "-i", audio_path]
        filters = settings.audio_filters.strip()
        if filters:
            cmd += ["-af", filters]
        cmd += ["-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]

        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            tail = proc.stderr.decode("utf-8", "replace")[-500:]
            raise RuntimeError(f"ffmpeg не смог обработать аудио: {tail}")

        # .copy(): frombuffer даёт read-only view поверх bytes, а дальше массив
        # может резаться/копироваться — нужен writable буфер.
        return np.frombuffer(proc.stdout, dtype=np.float32).copy()

    def _transcribe(self, audio_path: str) -> str:
        waveform = self._decode_audio(audio_path)

        # Режем длинное аудио на окна по chunk_sec: энкодер по памяти растёт
        # ~квадратично от длины, поэтому файл целиком (десятки минут) уводит
        # процесс в OOM. Окна держат пик памяти ограниченным независимо от
        # длины записи.
        win = settings.chunk_sec * SAMPLE_RATE
        if win <= 0 or waveform.shape[0] <= win:
            chunks = [waveform]
        else:
            chunks = [waveform[i : i + win] for i in range(0, waveform.shape[0], win)]

        # Гоним окна батчами по chunk_batch_size: recognize принимает список
        # массивов и возвращает список текстов. Батчим вручную, чтобы пик памяти
        # не рос с числом окон (recognize паддит батч под самое длинное окно).
        results: list[str] = []
        batch_size = max(1, settings.chunk_batch_size)
        for i in range(0, len(chunks), batch_size):
            out = self._model.recognize(chunks[i : i + batch_size], sample_rate=SAMPLE_RATE)
            results.extend(out if isinstance(out, list) else [out])

        return " ".join(part for part in results if part).strip()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        log.info("Поднимаю модель %s...", settings.model_version)
        t0 = time.monotonic()
        # Жёстко фиксируем CPU: сервис рассчитан только на него. onnx-asr сам
        # скачает ONNX-модель с HuggingFace по имени (в кеш HF_HOME) при первом
        # обращении, если её там ещё нет.
        self._model = onnx_asr.load_model(
            settings.model_version, providers=["CPUExecutionProvider"]
        )
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
