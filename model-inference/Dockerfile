FROM python:3.11.13-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=4
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/
# RapidOCR declares the CPU ORT package. Replace its wheel after installation
# so only one distribution owns the shared onnxruntime module.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y onnxruntime onnxruntime-gpu \
    && pip install --no-cache-dir onnxruntime-gpu==1.21.0
COPY requirements-extra.txt /app/
RUN pip install --no-cache-dir -r requirements-extra.txt \
    && pip uninstall -y onnxruntime onnxruntime-gpu \
    && pip install --no-cache-dir onnxruntime-gpu==1.21.0
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib
RUN pip uninstall -y opencv-python opencv-python-headless \
    && pip install --no-cache-dir opencv-python-headless==5.0.0.93
COPY inference /app/inference
COPY scripts /app/scripts
COPY models.lock.json /app/
RUN useradd --uid 10001 --create-home inference
USER inference
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=4)"
CMD ["python", "-m", "inference"]
