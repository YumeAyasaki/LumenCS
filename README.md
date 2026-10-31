# Lumen Home CS assistant

## API

Run the API locally with:

```sh
uv run uvicorn app.main:app --reload
```

The server exposes `GET /health`, `POST /ask`, and `POST /ingest`. The ask
endpoint embeds the question and asynchronously retrieves up to three relevant
policy excerpts from Qdrant. It returns the source text and clause citations;
it filters out policies not yet effective at the request's `asked_at` time and
does not use a language model to synthesize an answer. Its request body accepts
a question, an ISO 8601 `asked_at` timestamp, and optional user/assistant
conversation history:

```sh
curl -X POST http://127.0.0.1:5000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Chính sách bảo hành như thế nào?","asked_at":"2026-10-29T21:00:00+07:00"}'
```

OpenAPI documentation is available at `http://127.0.0.1:5000/docs`.

`POST /ingest` processes the configured `DATA_DIR` in one request. It stores
Excel workbook rows in PostgreSQL and embeds clause-level PDF chunks into a
model-specific Qdrant collection. The response includes counts, PDF warnings,
and per-stage/total latency. Repeated calls use stable point IDs and remove
stale chunks. The API loads the multilingual embedding model during startup.
The Docker image downloads and includes the default model weights at build
time, avoiding a first-request download. Configure `DATA_DIR`,
`QDRANT_COLLECTION`, and `PDF_EMBEDDING_MODEL` with environment variables.

## Spreadsheet reader

The reader is [spreadsheet_reader.py](app/tools/spreadsheet_reader.py).
It returns Pydantic models containing worksheet columns, row values, and source
file/sheet/Excel-row provenance. It does not connect to a database or print
output; a server can call `read_spreadsheet_directory()` and consume the returned
models directly.

Worksheet layouts are configured in
[`app/schema_mapping.json`](app/schema_mapping.json).

Pre-header rules and titles are returned in `header_notes`; configured footnotes
are returned in `footnotes`, separately from table `rows`. Sheets include an
`is_hidden` flag. Missing, invalid, or mismatched schema entries are reported in
`warnings`; those sheets are still read as unstructured rows rather than
stopping the reader. The matching PostgreSQL ingester is
[spreadsheet_ingester.py](app/tools/spreadsheet_ingester.py).

## PostgreSQL and Qdrant

PostgreSQL stores the spreadsheet hierarchy in three related tables:

- `source_workbooks` tracks each relative source filename, SHA-256 digest, and
  schema-mapping digest, and ingestion warnings.
- `source_sheets` stores worksheet names, visibility, column labels, notes,
  footnotes, and sheet warnings.
- `spreadsheet_rows` stores each non-empty row's original Excel row number and
  cell values. Join the row to its sheet and workbook to recover its
  `file/sheet/row` provenance.

The raw columns and cell values are JSONB so newly added workbook columns
remain auditable without changing migrations. A normalized layer stores
products, regional effective-dated prices, promotions, warranty terms, regions,
service centres, and service fees. Every typed row has a `source_row_id` foreign
key to `spreadsheet_rows`; its ORM `cite_id` resolves to the source file, sheet,
and Excel row. The schema and parse rules for each table are reviewed in
[`app/schema_mapping.json`](app/schema_mapping.json). Archive sheets remain in
the raw layer and are never normalized.

The database query helpers in [normalized_data.py](app/tools/normalized_data.py)
select typed models only and eager-load raw provenance for citations. After
updating field mappings or parsers, call
`renormalize_spreadsheet_rows(session_factory)` to rebuild typed rows from
persisted raw data without reopening workbooks. Normal ingestion skips a
workbook only when both the workbook and mapping hashes match; changed workbook
records are replaced atomically.

The application connects to PostgreSQL using `DATABASE_URL` and to Qdrant using
`QDRANT_URL`. `GET /health` checks both stores and returns HTTP 503 if either
is unavailable. Alembic migrations run with `uv run alembic upgrade head`.
`make ingest DATA=<dir>` imports the mapped Excel files from the chosen
directory and indexes PDFs in Qdrant. Set `DATABASE_URL` to a reachable
PostgreSQL instance and `QDRANT_URL` to a reachable Qdrant instance when
running outside Compose.

## Policy PDF reader

The reader is [pdf_reader.py](app/tools/pdf_reader.py). Call
`read_pdf()` for one policy PDF or `read_pdf_directory()` to recursively read
PDFs under a data directory. It returns typed document and page models with the
original page text, source-relative path, and page-level `cite_id`; it does not
force policy prose into a rigid field schema. Pages without extractable text
remain in the result and are flagged with a warning because scanned pages may
need OCR.

The matching Qdrant ingester is [pdf_ingester.py](app/tools/pdf_ingester.py).
It splits numbered clauses and bullets into distinct chunks, keeps tables
separate with their rows preserved, and prefixes each chunk with its section
title. Clause citations use IDs such as `policy_handbook.pdf#3.2`; the payload
also stores the page, source and chunk SHA-256 digests, handbook version, and
effective date. PDF text extraction uses PyMuPDF; malformed PDFs raise a
parsing error rather than being silently skipped.

The configured
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` model is
multilingual (rather than FastEmbed's English default). Model identity is
included in both the Qdrant collection name and point ID so changing models
cannot mix incompatible vector spaces. To evaluate the top-three clause
retrieval on ten Vietnamese handbook questions, run:

```sh
uv run python -m eval.evaluate_policy_retrieval
```

The current ten-question handbook set achieves Recall@3 of 10/10 with the
configured MiniLM model.

## Docker

Build and run the API alongside local Postgres and Qdrant:

```sh
docker compose up --build
```

The app container applies Alembic migrations and starts the API at
`http://127.0.0.1:5000`. Call `POST /ingest` to import the configured sources.
PostgreSQL and Qdrant use persistent Docker volumes. If port 5000 is occupied,
set `APP_PORT` to choose a different host port.

To run only the databases, use `docker compose up -d postgres qdrant`. Their
ports are bound to localhost (`5432`, `6333`, and `6334`). Postgres defaults
are intended for local development and can be overridden with `POSTGRES_DB`,
`POSTGRES_USER`, and `POSTGRES_PASSWORD` environment variables.
