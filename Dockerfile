FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

ARG PDF_EMBEDDING_MODEL
ENV PDF_EMBEDDING_MODEL=${PDF_EMBEDDING_MODEL} \
    FASTEMBED_CACHE_PATH=/opt/fastembed_cache
RUN uv run python -c "import os; from fastembed import TextEmbedding; TextEmbedding(model_name=os.environ['PDF_EMBEDDING_MODEL'], cache_dir=os.environ['FASTEMBED_CACHE_PATH'])"

FROM python:3.11-slim AS runtime

ARG PDF_EMBEDDING_MODEL
ENV PATH="/app/.venv/bin:$PATH" \
    PDF_EMBEDDING_MODEL=${PDF_EMBEDDING_MODEL} \
    FASTEMBED_CACHE_PATH=/opt/fastembed_cache \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --chown=10001:10001 --from=builder /opt/fastembed_cache /opt/fastembed_cache
COPY app ./app
COPY migrations ./migrations
COPY alembic.ini ./
COPY data ./data

USER 10001:10001

EXPOSE 5000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
