# Lumen Home CS assistant

## API placeholder

Run the API locally with:

```sh
uv run uvicorn app.main:app --reload
```

The server exposes `GET /health` and `POST /ask`. The ask endpoint currently
validates the request and returns a Vietnamese placeholder answer with an empty
`citations` list; material-backed answer generation is not implemented yet.
Its request body accepts a question, an ISO 8601 `asked_at` timestamp, and
optional user/assistant conversation history:

```sh
curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Chính sách bảo hành như thế nào?","asked_at":"2026-10-29T21:00:00+07:00"}'
```

OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

## Spreadsheet reader

The server-facing reader is [spreadsheet_reader.py](app/tools/spreadsheet_reader.py).
It returns Pydantic models containing worksheet columns, row values, and source
file/sheet/Excel-row provenance. It does not connect to a database or print
output; a server can call `read_spreadsheet_directory()` and consume the returned
models directly.

Worksheet layouts are configured in
[`app/schema_mapping.json`](app/schema_mapping.json). To preview the parsed data
as JSON while adjusting the mapping, run:

```sh
uv run python -m app.tools.preview_spreadsheets --data-dir data
```

Pre-header rules and titles are returned in `header_notes`; configured footnotes
are returned in `footnotes`, separately from table `rows`. Sheets include an
`is_hidden` flag. Missing, invalid, or mismatched schema entries are reported in
`warnings`; those sheets are still read as unstructured rows rather than
stopping the reader.

This preview command is only a development aid; database persistence can be
added later as a separate server integration.

## Docker

Build and run the API alongside local Postgres and Qdrant:

```sh
docker compose up --build
```

The API is available at `http://127.0.0.1:8000`; it does not yet connect to
Postgres, Qdrant, or the source materials. The databases continue running with
persistent Docker volumes.

To run only the databases, use `docker compose up -d postgres qdrant`. Their
ports are bound to localhost (`5432`, `6333`, and `6334`). Postgres defaults
are intended for local development and can be overridden with `POSTGRES_DB`,
`POSTGRES_USER`, and `POSTGRES_PASSWORD` environment variables.
