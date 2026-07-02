# Рантайм-образ сервиса. Инференс на PaddleOCR-VL через transformers (torch, CPU).
FROM python:3.11-slim

WORKDIR /app

# torch и torchvision ставим CPU-only отдельным индексом, чтобы не тянуть CUDA-
# колёса (иначе образ раздувается на несколько ГБ) и получить совместимую пару
# версий. torchvision нужен image-процессору PaddleOCR-VL (его remote-код падает
# без него). Ставим до остальных зависимостей, чтобы transformers подхватил уже
# установленный CPU-torch.
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt python-dotenv

COPY app.py config.py schemas.py ocr.py download_model.py ./

# Веса PaddleOCR-VL качает с HuggingFace в кеш (HF_HOME); он монтируется как
# volume (см. docker-compose.yml), поэтому в образ веса не зашиваем.

EXPOSE 8000

# --workers 1 принципиально: иначе каждый воркер держит свою копию модели в RAM
# и экономия памяти теряется.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
