# Рантайм-образ сервиса. Инференс на onnxruntime через onnx-asr — без torch.
FROM python:3.11-slim

# ffmpeg нужен для декодирования входного аудио (в т.ч. mp3/mp4) и предобработки.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt python-dotenv

COPY app.py config.py schemas.py transcriber.py download_model.py ./

# Веса модели onnx-asr качает с HuggingFace в кеш (HF_HOME); он монтируется как
# volume (см. docker-compose.yml), поэтому в образ модель не зашиваем.

EXPOSE 8000

# --workers 1 принципиально: иначе каждый воркер держит свою копию модели в RAM
# и экономия памяти теряется.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
