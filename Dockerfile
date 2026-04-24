# Author: Bradley R. Kinnard
FROM python:3.12-slim AS base

WORKDIR /app

# System deps for spacy and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY backend/ backend/
COPY baselines/ baselines/
COPY beliefs/ beliefs/
COPY interfaces/ interfaces/
COPY metrics/ metrics/
COPY data/ data/
COPY examples/ examples/
COPY configs/ configs/

RUN pip install --no-cache-dir -e ".[serve]"

EXPOSE 8000

ENV PYTHONPATH=/app
ENV STORAGE_BACKEND=memory
ENV LLM_PROVIDER=none

CMD ["uvicorn", "backend.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
