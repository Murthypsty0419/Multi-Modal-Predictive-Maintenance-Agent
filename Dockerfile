FROM python:3.11-slim

WORKDIR /app

# System deps for psycopg2 and lightgbm
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential libpq-dev libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY . .

# Run DB migrations then start the API
CMD ["sh", "-c", "python -m app.db.migrate && uvicorn app.api.routes:app --host 0.0.0.0 --port 8000"]
