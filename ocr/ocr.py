"""
Поток-воркер, владеющий OCR-моделью.

Инференс на CPU тяжёлый (PaddleOCR-VL — VL-модель ~1B параметров), поэтому гнать
несколько распознаваний параллельно смысла нет: они только конкурировали бы за те
же ядра и память. Вместо этого все запросы выстраиваются в очередь, а
обрабатывает их один выделенный поток-воркер. Приятный побочный эффект: модель
живёт только внутри воркера, к ней больше никто не лезет, и никакие блокировки для
защиты от гонок не нужны.

Модель не грузится на старте. Первый запрос поднимает её в RAM (cold start —
несколько секунд + скачивание весов при первом обращении). Если запросов не было
дольше IDLE_TTL, воркер сам выгружает модель и освобождает память. Следующий
запрос поднимет её заново.

Инференс — на PaddleOCR-VL через HuggingFace transformers (torch, CPU): модель
качается с HuggingFace по имени (MODEL_NAME). Это VLM: на вход — картинка + промпт
(`OCR:`), на выход — распознанный текст. Layout-пайплайна (разметка блоков, порядок
чтения) в transformers-пути нет — распознаётся картинка целиком (element-level).
PDF рендерим в картинки постранично сами через pypdfium2 (без системного poppler).
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
    """Одна единица работы для воркера: путь к файлу и куда положить результат."""

    file_path: str
    future: Future[list[str]]


class OcrWorker:
    """Поток-воркер: владеет OCR-моделью и обрабатывает очередь запросов.

    Всё, что касается модели (загрузка, инференс, выгрузка), происходит только
    здесь, в одном потоке. Поэтому состояние можно трогать без блокировок.
    """

    def __init__(self):
        # Очередь без ограничения по размеру: запросы под наплывом просто ждут
        # своей очереди, а не получают отказ.
        self._jobs: queue.Queue[_Job | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="ocr", daemon=True)
        # Доступ к модели — только из потока воркера. _loaded читает /health
        # из другого потока, поэтому это отдельный потокобезопасный флаг.
        self._model = None
        self._processor = None
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
        """Загрузить файл как список PIL-картинок (RGB): по одной на страницу.

        Картинка → одна страница. PDF рендерим постранично через pypdfium2 с DPI
        из настроек (без системного poppler).
        """
        from PIL import Image

        if os.path.splitext(file_path)[1].lower() == ".pdf":
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(file_path)
            try:
                scale = settings.pdf_dpi / 72.0  # pypdfium2 масштабирует от 72 DPI
                images = [
                    pdf[i].render(scale=scale).to_pil().convert("RGB") for i in range(len(pdf))
                ]
            finally:
                pdf.close()
            return images

        return [Image.open(file_path).convert("RGB")]

    def _ocr(self, file_path: str) -> list[str]:
        """Распознать файл и вернуть текст по одной строке на страницу."""
        return [self._recognize(image) for image in self._load_images(file_path)]

    def _recognize(self, image) -> str:
        """Прогнать одну картинку через VLM и вернуть распознанный текст."""
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": settings.prompt},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(settings.device)

        with torch.inference_mode():
            outputs = self._model.generate(**inputs, max_new_tokens=settings.max_new_tokens)

        # Отрезаем токены промпта, чтобы в ответ не попал сам вопрос — декодируем
        # только сгенерированную моделью часть.
        prompt_len = inputs["input_ids"].shape[1]
        generated = outputs[:, prompt_len:]
        text = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
        return text.strip()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        log.info(
            "Поднимаю PaddleOCR-VL %s (%s, %s)...",
            settings.model_name,
            settings.device,
            settings.torch_dtype,
        )
        t0 = time.monotonic()
        import torch
        from transformers import AutoProcessor

        dtype = getattr(torch, settings.torch_dtype)
        model = self._load_model(settings.model_name, dtype)
        model = model.to(settings.device).eval()
        self._processor = AutoProcessor.from_pretrained(settings.model_name, trust_remote_code=True)
        self._model = model
        self._loaded.set()
        log.info("Модель загружена за %.1f c.", time.monotonic() - t0)

    @staticmethod
    def _load_model(name: str, dtype):
        """Загрузить веса нужным Auto-классом.

        Разные версии PaddleOCR-VL зарегистрированы под разные классы: базовая —
        под causal-lm, новые (1.6) — под image-text-to-text. `auto` пробует
        сначала image-text-to-text, затем causal-lm.
        """
        from transformers import AutoModelForCausalLM

        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:  # старая версия transformers без этого класса
            AutoModelForImageTextToText = None

        kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
        choice = settings.model_class

        if choice == "causal_lm":
            return AutoModelForCausalLM.from_pretrained(name, **kwargs)
        if choice == "image_text_to_text":
            if AutoModelForImageTextToText is None:
                raise RuntimeError("transformers слишком старый: нет AutoModelForImageTextToText")
            return AutoModelForImageTextToText.from_pretrained(name, **kwargs)

        # auto: сперва image-text-to-text (для новых версий), потом causal-lm.
        if AutoModelForImageTextToText is not None:
            try:
                return AutoModelForImageTextToText.from_pretrained(name, **kwargs)
            except Exception as exc:
                log.info("image-text-to-text не подошёл (%s), пробую causal-lm.", exc)
        return AutoModelForCausalLM.from_pretrained(name, **kwargs)

    def _unload_if_idle(self) -> None:
        if self._model is None:
            return
        idle = time.monotonic() - self._last_used
        if idle < settings.idle_ttl:
            return
        log.info("Простой %.0f c >= %d c — выгружаю модель из RAM.", idle, settings.idle_ttl)
        self._model = None
        self._processor = None
        self._loaded.clear()
        gc.collect()
