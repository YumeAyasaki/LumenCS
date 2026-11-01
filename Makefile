.PHONY: server migrate ingest

DATA ?= data

server:
	uv run uvicorn app.main:app --reload

migrate:
	uv run alembic upgrade head

ingest:
	uv run python -m app.tools.ingest --data-dir "$(DATA)"

test:
	uv run python scripts/run_sample_questions.py

up:
	docker compose --env-file .env.example --env-file .env up --build

down:
	docker compose --env-file .env.example --env-file .env down