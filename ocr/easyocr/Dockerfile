# Рантайм-образ сервиса. Инференс на EasyOCR (torch, CPU).
FROM python:3.11-slim

# Системная библиотека для opencv-python-headless (нужна easyocr). GL/GUI-зависимостей
# у headless-сборки нет, поэтому хватает libglib2.0-0.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch и torchvision ставим CPU-only отдельным индексом, чтобы не тянуть CUDA-
# колёса (иначе образ раздувается на несколько ГБ). Ставим до easyocr, чтобы он
# подхватил уже установленный CPU-torch, а не притащил дефолтный.
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt python-dotenv

COPY __main__.py app.py config.py schemas.py ocr.py download_model.py ./

# Модели EasyOCR качает сам в кеш (EASYOCR_MODULE_PATH); он монтируется как volume
# (см. docker-compose.yml), поэтому в образ модели не зашиваем.

EXPOSE 8000

# Каждый worker держит отдельный ридер в RAM; по умолчанию один.
CMD ["python", "__main__.py"]
