FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Durable SQLite file lives on a mounted volume in production.
RUN mkdir -p /data
ENV MAILROOM_DB_PATH=/data/mailroom.db

EXPOSE 8080



CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --timeout-keep-alive 60"]
