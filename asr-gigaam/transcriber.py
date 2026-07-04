"""
Поток-воркер с ASR-моделью GigaAM.

CPU-инференс однопоточный, параллелить запросы смысла нет — только дрались бы
за ядра. Поэтому всё идёт через одну очередь и единственный воркер; заодно
модель никто не шарит между потоками, и локи не нужны.

Модель поднимается лениво по первому запросу (cold start — несколько секунд) и
выгружается из RAM после IDLE_TTL простоя; следующий запрос грузит её заново.

Под капотом — onnx-asr поверх onnxruntime: сам качает ONNX-модель GigaAM с
HuggingFace по имени (напр. gigaam-v3-ctc), препроцессинг без torch.
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

# 16 кГц моно — на этом модель училась, столько же ждёт onnx-asr. Любой вход
# приводим к этой частоте.
SAMPLE_RATE = 16000


@dataclass
class _Job:
    """Задача для воркера: путь к аудио и future, куда вернуть текст."""

    audio_path: str
    future: Future[str]


class TranscriberWorker:
    """Поток-воркер: держит ASR-модель и разгребает очередь запросов.

    Модель трогает только этот поток — грузит, гоняет инференс, выгружает.
    Раз доступ из одного потока, состояние живёт без блокировок.
    """

    def __init__(self):
        # Очередь без лимита: при наплыве запросы ждут, а не отбиваются отказом.
        self._jobs: queue.Queue[_Job | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="transcriber", daemon=True)
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

    def submit(self, audio_path: str) -> Future[str]:
        """Поставить аудио в очередь и получить future с результатом."""
        future: Future[str] = Future()
        self._jobs.put(_Job(audio_path, future))
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
            text = self._transcribe(job.audio_path)
            self._last_used = time.monotonic()  # инференс долгий, отметимся ещё раз
            job.future.set_result(text)
        except Exception as exc:  # отдаём ошибку тому, кто ждёт результат
            log.exception("Инференс упал")
            job.future.set_exception(exc)
        finally:
            # Файл временный: хранением занят другой сервис, здесь ничего не
            # держим. Удаляет именно воркер, а не HTTP-слой: при 504 клиент уже
            # ушёл, но файл ещё нужен нам, пока _transcribe его не дочитает.
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

        Сами, а не через onnx-asr: его recognize по пути читает только PCM-wav, а
        мы принимаем mp3/mp4/m4a и прочее. ffmpeg вскроет любой контейнер и на том
        же проходе прогонит предобработку (AUDIO_FILTERS). На выходе — сырой PCM
        f32 в stdout; ровно это recognize и ждёт (float32 в диапазоне [-1, 1]).
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

        # .copy(): frombuffer отдаёт read-only view поверх bytes, а массив дальше
        # режется и копируется — нужен writable буфер.
        return np.frombuffer(proc.stdout, dtype=np.float32).copy()

    def _transcribe(self, audio_path: str) -> str:
        waveform = self._decode_audio(audio_path)

        # Режем длинное аудио на окна по chunk_sec: память энкодера растёт
        # ~квадратично от длины, так что файл на десятки минут целиком уводит
        # процесс в OOM. С окнами пик памяти не зависит от длины записи.
        win = settings.chunk_sec * SAMPLE_RATE
        if win <= 0 or waveform.shape[0] <= win:
            chunks = [waveform]
        else:
            chunks = [waveform[i : i + win] for i in range(0, waveform.shape[0], win)]

        # Гоним окна батчами по chunk_batch_size: recognize принимает список
        # массивов и возвращает список текстов. Батчим вручную, иначе пик памяти
        # рос бы с числом окон (recognize паддит батч под самое длинное окно).
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
        # Только CPU — под него сервис и рассчитан. onnx-asr сам стянет ONNX-модель
        # с HuggingFace по имени (в кеш HF_HOME), если её там ещё нет.
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
