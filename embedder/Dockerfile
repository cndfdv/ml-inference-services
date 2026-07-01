# Рантайм-образ сервиса. Инференс на sentence-transformers (torch, CPU).
FROM python:3.11-slim

WORKDIR /app

# torch ставим CPU-only отдельным индексом, чтобы не тянуть CUDA-колёса (иначе
# образ раздувается на несколько ГБ). Ставим до остальных зависимостей, чтобы
# sentence-transformers подхватил уже установленный CPU-torch.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt python-dotenv

COPY app.py config.py schemas.py embedder.py download_model.py ./

# Веса модели sentence-transformers качает с HuggingFace в кеш (HF_HOME); он
# монтируется как volume (см. docker-compose.yml), поэтому в образ модель не
# зашиваем.

EXPOSE 8000

# --workers 1 принципиально: иначе каждый воркер держит свою копию модели в RAM
# и экономия памяти теряется.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
