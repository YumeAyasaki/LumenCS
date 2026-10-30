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
curl -X POST http://127.0.0.1:5000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Chính sách bảo hành như thế nào?","asked_at":"2026-10-29T21:00:00+07:00"}'
```

OpenAPI documentation is available at `http://127.0.0.1:5000/docs`.

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

This preview command is a development aid. To persist the parsed rows in
PostgreSQL, run `make migrate` followed by `make ingest DATA=data`.

## PostgreSQL and Qdrant

PostgreSQL stores the spreadsheet hierarchy in three related tables:

- `source_workbooks` tracks each relative source filename, SHA-256 digest, and
  ingestion warnings.
- `source_sheets` stores worksheet names, visibility, column labels, notes,
  footnotes, and sheet warnings.
- `spreadsheet_rows` stores each non-empty row's original Excel row number and
  cell values. Join the row to its sheet and workbook to recover its
  `file/sheet/row` provenance.

The columns and cell values are JSONB so newly added workbook columns and
worksheets can be ingested without changing database migrations. Re-ingesting
unchanged files skips them by content hash; changed files replace their
previous records atomically.

The application connects to PostgreSQL using `DATABASE_URL` and to Qdrant using
`QDRANT_URL`. `GET /health` checks both stores and returns HTTP 503 if either
is unavailable. Alembic migrations run with `uv run alembic upgrade head`;
`make ingest DATA=<dir>` imports the mapped Excel files from the chosen
directory. Set `DATABASE_URL` to a reachable PostgreSQL instance and
`QDRANT_URL` to a reachable Qdrant instance when running outside Compose.

## Policy PDF reader

The server-facing reader is [pdf_reader.py](app/tools/pdf_reader.py). Call
`read_pdf()` for one policy PDF or `read_pdf_directory()` to recursively read
PDFs under a data directory. It returns typed document and page models with the
original page text, source-relative path, and page-level `cite_id`; it does not
force policy prose into a rigid field schema. Pages without extractable text
remain in the result and are flagged with a warning because scanned pages may
need OCR.

Preview the extracted text and provenance as formatted JSON:

```sh
uv run python -m app.tools.preview_pdf --data-dir data
```

The reader is independent of FastAPI and persistence so a server can consume
the models directly and add its own indexing or storage layer.
PDF text extraction uses PyMuPDF; malformed PDFs raise a parsing error rather
than being silently skipped.

## Docker

Build and run the API alongside local Postgres and Qdrant:

```sh
docker compose up --build
```

The app container applies Alembic migrations, ingests the source workbooks, and
then starts the API at `http://127.0.0.1:5000`. PostgreSQL and Qdrant use
persistent Docker volumes. The placeholder `/ask` response is not yet backed by
retrieval from these stores. If port 5000 is occupied, set `APP_PORT` to choose
a different host port.

To run only the databases, use `docker compose up -d postgres qdrant`. Their
ports are bound to localhost (`5432`, `6333`, and `6334`). Postgres defaults
are intended for local development and can be overridden with `POSTGRES_DB`,
`POSTGRES_USER`, and `POSTGRES_PASSWORD` environment variables.
