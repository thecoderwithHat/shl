FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system dependencies needed by FAISS and builds
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       libopenblas-dev \
       liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# Install Python deps and gunicorn (if not already present)
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

EXPOSE 8000

ENV FAISS_INDEX_PATH=/app/faiss_index

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
