"""
Поток-воркер, владеющий OCR-моделью.

Распознавание на CPU по своей сути последовательное, поэтому гнать несколько
запросов параллельно смысла нет: они только конкурировали бы за те же ядра.
Вместо этого все запросы выстраиваются в очередь, а обрабатывает их один
выделенный поток-воркер. Приятный побочный эффект: ридер живёт только внутри
воркера, к нему больше никто не лезет, и никакие блокировки не нужны.

Ридер не грузится на старте. Первый запрос поднимает его в RAM (cold start —
несколько секунд + скачивание моделей при первом обращении). Если запросов не
было дольше IDLE_TTL, воркер сам выгружает ридер и освобождает память. Следующий
запрос поднимет его заново.

Инференс — на EasyOCR (torch, CPU): детекция + распознавание текста. EasyOCR сам
качает модели по языкам (LANGS). PDF рендерим в картинки постранично через
pypdfium2 (без системного poppler).
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


def _avx2_available() -> bool:
    """Есть ли у CPU инструкции AVX2.

    Квантизация распознавателя EasyOCR идёт через fbgemm, а он требует AVX2.
    На CPU без AVX2 (напр. дефолтный QEMU Virtual CPU) квантованные операции
    падают с SIGILL прямо в forward-проходе. Поэтому квантизацию включаем
    только когда AVX2 реально есть.
    """
    try:
        with open("/proc/cpuinfo") as f:
            return " avx2 " in f.read().replace("\n", " ")
    except OSError:
        return False


@dataclass
class _Job:
    """Одна единица работы для воркера: путь к файлу и куда положить результат."""

    file_path: str
    future: Future[list[str]]


class OcrWorker:
    """Поток-воркер: владеет OCR-ридером и обрабатывает очередь запросов.

    Всё, что касается ридера (загрузка, инференс, выгрузка), происходит только
    здесь, в одном потоке. Поэтому состояние можно трогать без блокировок.
    """

    def __init__(self):
        # Очередь без ограничения по размеру: запросы под наплывом просто ждут
        # своей очереди, а не получают отказ.
        self._jobs: queue.Queue[_Job | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="ocr", daemon=True)
        # Доступ к ридеру — только из потока воркера. _loaded читает /health
        # из другого потока, поэтому это отдельный потокобезопасный флаг.
        self._reader = None
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

    def submit(self, file_path: str) -> Future[list[str]]:
        """Поставить файл в очередь и получить future со списком текстов страниц."""
        future: Future[list[str]] = Future()
        self._jobs.put(_Job(file_path, future))
        return future

    # ---- внутренняя кухня потока ----

    def _run(self) -> None:
        while True:
            # Спим на get, пока не придёт задача. Таймаут ставим ровно на момент
            # истечения простоя: проснуться раньше незачем, а проснувшись точно
            # в срок — сразу выгружаем ридер. Если ридер не загружен, timeout
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
        """Сколько секунд спать на get. None — ридер не загружен, ждём задачу."""
        if self._reader is None:
            return None
        remaining = settings.idle_ttl - (time.monotonic() - self._last_used)
        return max(0.0, remaining)

    def _process(self, job: _Job) -> None:
        try:
            self._ensure_loaded()
            self._last_used = time.monotonic()
            page_texts = self._ocr(job.file_path)
            self._last_used = time.monotonic()  # инференс долгий, обновляем после
            job.future.set_result(page_texts)
        except Exception as exc:  # пробрасываем ошибку ожидающему запросу
            log.exception("Инференс упал")
            job.future.set_exception(exc)
        finally:
            # Загрузка — временная. Сервис ничего не хранит (хранение и БД — в
            # другом сервисе), поэтому файл удаляем сразу после обработки —
            # успешной или нет. Удаляет именно воркер, а не HTTP-слой: при 504
            # запрос перестаёт ждать результат, но файл всё ещё нужен воркеру,
            # пока тот не дочитает его в _ocr.
            self._delete(job.file_path)

    @staticmethod
    def _delete(path: str) -> None:
        try:
            os.remove(path)
        except OSError as exc:
            log.warning("Не удалось удалить временный файл %s: %s", path, exc)

    @staticmethod
    def _load_images(file_path: str):
        """Загрузить файл как список numpy-картинок (RGB): по одной на страницу.

        Картинка → одна страница. PDF рендерим постранично через pypdfium2 с DPI
        из настроек (без системного poppler).
        """
        import numpy as np
        from PIL import Image

        if os.path.splitext(file_path)[1].lower() == ".pdf":
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(file_path)
            try:
                scale = settings.pdf_dpi / 72.0  # pypdfium2 масштабирует от 72 DPI
                images = [
                    np.asarray(pdf[i].render(scale=scale).to_pil().convert("RGB"))
                    for i in range(len(pdf))
                ]
            finally:
                pdf.close()
            return images

        return [np.asarray(Image.open(file_path).convert("RGB"))]

    def _ocr(self, file_path: str) -> list[str]:
        """Распознать файл и вернуть текст по одной строке на страницу."""
        return [self._recognize(image) for image in self._load_images(file_path)]

    def _recognize(self, image) -> str:
        """Прогнать одну картинку через EasyOCR и склеить строки в текст.

        readtext возвращает список результатов; текст лежит на позиции [1] и в
        режиме paragraph, и без него.
        """
        results = self._reader.readtext(image, detail=1, paragraph=settings.paragraph)
        return "\n".join(item[1] for item in results if item[1]).strip()

    def _ensure_loaded(self) -> None:
        if self._reader is not None:
            return
        log.info("Поднимаю EasyOCR (%s)...", ",".join(settings.langs))
        t0 = time.monotonic()
        import easyocr

        # Квантизация (int8, fbgemm) требует AVX2 — иначе распознаватель падает
        # с SIGILL. Разрешаем её только когда включена в настройках И CPU
        # поддерживает AVX2; на CPU без AVX2 тихо откатываемся на fp32.
        quantize = settings.quantize and _avx2_available()
        if settings.quantize and not quantize:
            log.warning("CPU без AVX2 — отключаю квантизацию EasyOCR (fbgemm требует AVX2), fp32.")

        # Жёстко фиксируем CPU (gpu=False). Модели EasyOCR качает сам по языкам в
        # каталог EASYOCR_MODULE_PATH (в docker — том), если их там ещё нет.
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
