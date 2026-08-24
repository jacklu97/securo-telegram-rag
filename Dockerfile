FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

# Model cache + sqlite live here; mount a volume.
ENV HF_HOME=/data/models DB_PATH=/data/messages.db
VOLUME /data
EXPOSE 8900

CMD ["python", "-m", "telegram_rag.main"]
