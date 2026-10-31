# Lumen Home CS assistant

## API

Run the API locally with:

```sh
uv run uvicorn app.main:app --reload
```

The server exposes `GET /health`, `POST /ask`, and `POST /ingest`. The ask
endpoint uses a chat-completions LLM with typed tools for PostgreSQL and Qdrant.
All tool calls use the request's `asked_at` timestamp for effective-date
filtering. Final citations are restricted to IDs returned by tools in that
request, with one retry for invalid answers or citations. It accepts a
question, an ISO 8601 `asked_at` timestamp, and optional user/assistant
conversation history. The system prompt includes the asked date converted to
`Asia/Ho_Chi_Minh` and formatted `dd/mm/yyyy`; timestamps without a timezone
offset are treated as Vietnam local time. JSON responses and requests accepting
`text/event-stream` include each tool call's name and arguments along with its
structured tool answer; the Streamlit chat displays these in a
"Các công cụ đã tra cứu" expander. SSE responses also include validated answer
chunks, citations, per-step metrics, and latency events. Each LLM request and
typed-tool call reports its duration and provider-reported token counts when
available. Missing token counts are omitted. Parallel tool durations are
individual elapsed times and therefore can overlap:

```sh
curl -X POST http://127.0.0.1:5000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Chính sách bảo hành như thế nào?","asked_at":"2026-10-29T21:00:00+07:00"}'
```

OpenAPI documentation is available at `http://127.0.0.1:5000/docs`.

## Streamlit chat UI

Start the API, databases, and Streamlit chat interface together with
`docker compose up --build`. Open `http://127.0.0.1:8501` to chat; the UI
connects to the API over the Compose network. Set `STREAMLIT_PORT` to change
the host port. To run the interface locally, install the `ui` dependency group
with `uv sync --group ui` and run
`LUMEN_API_URL=http://127.0.0.1:5000 uv run --group ui streamlit run ui/streamlit_app.py`.
The chat keeps its conversation in the Streamlit session, shows returned source
citations, and sends the local current time as `asked_at`.

## Agent tools

`GET /tools` returns model-ready function declarations with JSON schemas.
Invoke a declared tool with `POST /tools/{tool_name}` and an `arguments` object;
each tool requires the question's ISO 8601 `as_of` timestamp. Responses contain
both `call` (the invoked name and arguments) and `answer` (the structured
result with citations). For example:

```sh
curl -X POST http://127.0.0.1:5000/tools/quote_price \
  -H 'Content-Type: application/json' \
  -d '{"arguments":{"as_of":"2026-10-29T21:00:00+07:00","model":"LH-AC09N","province":"Hà Nội"}}'
```

The registry includes `get_product_details`, `quote_price`, `get_promotions`,
`get_warranty`, `get_region`, `find_service_centres`, `get_service_fee`,
`search_policy`, `calculate`, and `get_date_context`. Spreadsheet tools query
normalized typed tables and return source-row citations; policy search returns
handbook clause citations and expands each matching clause to its complete
section text. All clause citations included in that section are returned to
the answer step. Spreadsheet tool results also include the source worksheet's
flattened headers (including multi-row header titles), pre-header notes, and
configured footer rows in `source_contexts`; sheet and note references are
included in the tool's citation allowlist. `calculate` accepts arithmetic only
and does not execute model-provided code. Tools are provider-neutral
declarations and an executor used by `/ask`; the LLM cannot issue SQL or cite
sources that were not returned by a tool.

## LLM providers

The default provider is Gemini's OpenAI-compatible chat-completions endpoint
using `gemini-3.5-flash-lite`.
Put `GEMINI_KEY` in the local `.env` file (which is gitignored), then run the
server normally. Set `LLM_PROVIDER=openai` and `OPENAI_API_KEY` to use OpenAI.
For other compatible services, set `LLM_PROVIDER=openai-compatible`,
`LLM_BASE_URL`, and `LLM_MODEL`; `LLM_API_KEY` is optional for local services.
Provider-reported input/output token counts and per-stage timings are logged
without logging credentials or prompts. Pricing is not collected or calculated.
Provider response reads time out after 180 seconds by default and retry
transient timeouts up to three attempts. Set `LLM_READ_TIMEOUT_SECONDS` from
1 to 600 to change that limit; the Streamlit client allows up to ten minutes
for an answer. The same environment settings can be used with Docker Compose.

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
