.PHONY: server migrate ingest

DATA ?= data

server:
	uv run uvicorn app.main:app --reload

migrate:
	uv run alembic upgrade head

ingest:
	uv run python -m app.tools.ingest --data-dir "$(DATA)"