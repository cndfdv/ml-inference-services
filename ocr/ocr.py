"""
Поток-воркер с OCR-моделью EasyOCR.

OCR на CPU — последовательная работа, параллельные запросы только толкались бы
за ядра. Поэтому одна очередь и один воркер; ридер при этом никто не шарит,
и локи не нужны.

Ридер поднимается лениво по первому запросу (cold start — несколько секунд плюс
скачивание моделей при первом обращении) и выгружается из RAM после IDLE_TTL
простоя; следующий запрос грузит его заново.

Бэкенд — EasyOCR (torch, CPU): детекция + распознавание, модели качает сам по
языкам (LANGS). PDF рендерим в картинки постранично через pypdfium2 — без
системного poppler.
"""

from __future__ import annotations

import gc
import os
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass

import easyocr
import numpy as np
import pypdfium2 as pdfium
import torch
from config import log, settings
from PIL import Image

# Число CPU-потоков torch задаём один раз при импорте. Он и сам читает
# OMP_NUM_THREADS, но выставим явно — так инференс предсказуемее.
_omp = os.environ.get("OMP_NUM_THREADS")
if _omp:
    try:
        torch.set_num_threads(int(_omp))
    except ValueError as exc:  # мусор в переменной — не роняем сервис
        log.warning("Не удалось задать число потоков torch (%s): %s", _omp, exc)


def _avx2_available() -> bool:
    """Есть ли у CPU инструкции AVX2.

    Квантизация распознавателя EasyOCR идёт через fbgemm, а тому нужен AVX2. На
    CPU без него (напр. дефолтный QEMU Virtual CPU) квантованные операции падают
    с SIGILL прямо в forward-проходе — поэтому квантуем только при живом AVX2.
    """
    try:
        with open("/proc/cpuinfo") as f:
            return " avx2 " in f.read().replace("\n", " ")
    except OSError:
        return False


@dataclass
class _Job:
    """Задача для воркера: путь к файлу и future, куда вернуть тексты страниц."""

    file_path: str
    future: Future[list[str]]


class OcrWorker:
    """Поток-воркер: держит OCR-ридер и разгребает очередь запросов.

    Ридер трогает только этот поток — грузит, гоняет инференс, выгружает. Раз
    доступ из одного потока, состояние живёт без блокировок.
    """

    def __init__(self):
        # Очередь без лимита: при наплыве запросы ждут, а не отбиваются отказом.
        self._jobs: queue.Queue[_Job | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="ocr", daemon=True)
        # К ридеру ходит только воркер. А флаг _loaded читает /health из другого
        # потока — поэтому потокобезопасный Event, а не просто bool.
        self._reader = None
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

    def submit(self, file_path: str) -> Future[list[str]]:
        """Поставить файл в очередь и получить future со списком текстов страниц."""
        future: Future[list[str]] = Future()
        self._jobs.put(_Job(file_path, future))
        return future

    # ---- потроха воркера ----

    def _run(self) -> None:
        while True:
            # Ждём задачу на get. Таймаут — ровно до конца простоя: проснуться
            # раньше незачем, а точно в срок — сразу выгрузим ридер. Ридер не
            # загружен → timeout=None: выгружать нечего, спим до задачи.
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
        """Сколько спать на get. None — ридер не загружен, ждём задачу."""
        if self._reader is None:
            return None
        remaining = settings.idle_ttl - (time.monotonic() - self._last_used)
        return max(0.0, remaining)

    def _process(self, job: _Job) -> None:
        try:
            self._ensure_loaded()
            self._last_used = time.monotonic()
            page_texts = self._ocr(job.file_path)
            self._last_used = time.monotonic()  # инференс долгий, отметимся ещё раз
        except Exception as exc:  # отдаём ошибку тому, кто ждёт результат
            log.exception("Инференс упал")
            self._settle(job.future, exc=exc)
        else:
            self._settle(job.future, result=page_texts)
        finally:
            # Файл временный: хранением занят другой сервис, здесь ничего не
            # держим. Удаляет именно воркер, а не HTTP-слой: при 504 клиент уже
            # ушёл, но файл ещё нужен нам, пока _ocr его не дочитает.
            self._delete(job.file_path)

    @staticmethod
    def _settle(future: Future, *, result=None, exc: Exception | None = None) -> None:
        # future мог быть уже отменён (клиент отвалился по таймауту 504) — тогда
        # set_result/set_exception бросят InvalidStateError и уронят поток-воркер,
        # а с ним встанет вся очередь. Поэтому трогаем future только если он ещё жив.
        if future.done():
            return
        if exc is not None:
            future.set_exception(exc)
        else:
            future.set_result(result)

    @staticmethod
    def _delete(path: str) -> None:
        try:
            os.remove(path)
        except OSError as exc:
            log.warning("Не удалось удалить временный файл %s: %s", path, exc)

    @staticmethod
    def _iter_images(file_path: str):
        """Отдавать картинки страниц по одной (RGB numpy), а не все разом.

        Картинка — это одна страница. Страницы PDF рендерим и выдаём по одной
        (генератор), чтобы в памяти лежала только текущая: у большого PDF рендер
        всех страниц сразу — это гигабайты RAM и путь в OOM. DPI — из настроек,
        системный poppler не нужен.
        """
        if os.path.splitext(file_path)[1].lower() == ".pdf":
            pdf = pdfium.PdfDocument(file_path)
            try:
                scale = settings.pdf_dpi / 72.0  # pypdfium2 считает масштаб от 72 DPI
                for i in range(len(pdf)):
                    yield np.asarray(pdf[i].render(scale=scale).to_pil().convert("RGB"))
            finally:
                pdf.close()
        else:
            yield np.asarray(Image.open(file_path).convert("RGB"))

    def _ocr(self, file_path: str) -> list[str]:
        """Распознать файл: одна строка результата на страницу.

        Идём постранично — распознаём картинку и отпускаем её перед следующей,
        поэтому пик памяти держится одной страницей, а не всем документом.
        """
        return [self._recognize(image) for image in self._iter_images(file_path)]

    def _recognize(self, image) -> str:
        """Прогнать картинку через EasyOCR и склеить найденные строки.

        readtext отдаёт список результатов; сам текст лежит в [1] — и в режиме
        paragraph, и без него.
        """
        results = self._reader.readtext(image, detail=1, paragraph=settings.paragraph)
        return "\n".join(item[1] for item in results if item[1]).strip()

    def _ensure_loaded(self) -> None:
        if self._reader is not None:
            return
        log.info("Поднимаю EasyOCR (%s)...", ",".join(settings.langs))
        t0 = time.monotonic()
        # Квантизация (int8, fbgemm) требует AVX2, иначе распознаватель падает с
        # SIGILL. Включаем её, только когда она разрешена настройкой И CPU умеет
        # AVX2; без AVX2 тихо катимся на fp32.
        quantize = settings.quantize and _avx2_available()
        if settings.quantize and not quantize:
            log.warning("CPU без AVX2 — отключаю квантизацию EasyOCR (fbgemm требует AVX2), fp32.")

        # Жёстко на CPU (gpu=False). Модели EasyOCR качает сам по языкам в
        # EASYOCR_MODULE_PATH (в docker — том), если их там ещё нет.
        self._reader = easyocr.Reader(
            list(settings.langs),
            gpu=False,
            quantize=quantize,
            model_storage_directory=os.environ.get("EASYOCR_MODULE_PATH") or None,
        )
        self._loaded.set()
        log.info("Ридер загружен за %.1f c.", time.monotonic() - t0)

    def _unload_if_idle(self) -> None:
        if self._reader is None:
            return
        idle = time.monotonic() - self._last_used
        if idle < settings.idle_ttl:
            return
        log.info("Простой %.0f c >= %d c — выгружаю ридер из RAM.", idle, settings.idle_ttl)
        self._reader = None
        self._loaded.clear()
        gc.collect()
