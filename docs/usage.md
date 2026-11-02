# Usage and configuration

## Environment

The committed [`.env.example`](../.env.example) contains safe defaults for the
application, PostgreSQL, Qdrant, embedding model, LLM provider, and host ports.
Create `.env` for local settings and credentials; it is gitignored. If a local
`.env` already exists, merge any missing settings rather than overwriting it.
Compose reads `.env.example` first and `.env` second, so local values override
the template.

The default provider is Gemini (`gemini-3.5-flash-lite`); set `GEMINI_KEY` to
authenticate. Optionally set `GEMINI_KEY_BACKUP` to a second Gemini API key.
When Gemini returns HTTP 429 for the primary key, the current completion is
retried immediately with the backup key; later completions still start with
the primary key and fall back again if it is rate limited. The backup must
have access to the configured model and API. For OpenAI, set
`LLM_PROVIDER=openai` and `OPENAI_API_KEY`. For another OpenAI-compatible
provider, set `LLM_PROVIDER=openai-compatible`, `LLM_BASE_URL`, and
`LLM_MODEL`; `LLM_API_KEY` is optional for local providers.

Change `APP_PORT`, `STREAMLIT_PORT`, `POSTGRES_HOST_PORT`, `QDRANT_HTTP_PORT`,
or `QDRANT_GRPC_PORT` to adjust host ports. If changing the Qdrant HTTP port
while running the API outside Compose, also update `QDRANT_URL`.

## Running locally

Install dependencies and apply database migrations:

```sh
uv sync
docker compose --env-file .env.example --env-file .env up -d postgres qdrant
uv run alembic upgrade head
```

Start the API in one terminal:

```sh
uv run uvicorn app.main:app --reload
```

For the chat UI, install its optional dependencies and start Streamlit in
another terminal:

```sh
uv sync --group ui
LUMEN_API_URL=http://127.0.0.1:5000 uv run --group ui streamlit run ui/streamlit_app.py
```

Alternatively, run the full stack with
`docker compose --env-file .env.example --env-file .env up --build`. Docker
Compose applies migrations when starting the API. PostgreSQL and Qdrant use
persistent volumes. The sample database password is for local development;
replace it before exposing services beyond the local machine.

## API

- `GET /health` checks PostgreSQL and Qdrant.
- `POST /ask` answers a question using the given `asked_at` timestamp and
  optional user/assistant history.
- `POST /ingest` processes the configured `DATA_DIR`.
- `GET /tools` lists available model-facing typed tools.
- `POST /tools/{tool_name}` invokes a typed tool directly.

Example question:

```sh
curl -X POST http://127.0.0.1:5000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Chính sách bảo hành như thế nào?","asked_at":"2026-10-29T21:00:00+07:00"}'
```

The `/ask` endpoint accepts JSON by default and supports
`Accept: text/event-stream` for server-sent events. Responses include citations,
tool activity, and timing metrics. Timestamps without a timezone offset are
treated as Vietnam local time. Interactive OpenAPI documentation is available
at `/docs`.

## Agent tools

The registered tools are `get_product_details`, `quote_price`,
`get_promotions`, `get_warranty`, `find_service_centres`, `get_service_fee`,
`search_policy`, `calculate`, and `add_days`. Tool schemas are typed, and the
server supplies the request date as `as_of`; the model cannot issue SQL.
`calculate` evaluates arithmetic expressions only. `quote_price` includes
applicable promotion and installment information when supported by the source
data.

The agent searches the policy handbook with the original question before tool
selection. It validates final citations against IDs returned during the
request and retries an invalid answer once. Spreadsheet citations resolve to
source file, sheet, and Excel row; policy citations identify handbook clauses.

## Source ingestion

Put supported workbooks and PDFs under `DATA_DIR` (default: `data`). Ingest
with:

```sh
make ingest DATA=data
```

Or call `POST /ingest` when the API is running. Excel layouts and field parsers
are configured in [`app/schema_mapping.json`](../app/schema_mapping.json).
Raw workbook rows and provenance are stored in PostgreSQL, alongside typed
tables used by lookups. Workbook contents and the mapping file are hashed to
skip unchanged workbooks. Use `renormalize_spreadsheet_rows(session_factory)`
after changing normalization logic to rebuild typed rows from persisted
sources.

PDF text and tables are split into citeable sections and indexed in Qdrant
using the configured multilingual embedding model. Ingestion reports warnings
for pages without extractable text; scanned PDFs are not OCRed. The response
includes item counts, warnings, and processing times.

## Persistence and monitoring

PostgreSQL stores the raw workbook/sheet/row hierarchy and normalized product,
price, promotion, warranty, region, service-centre, and fee records. Qdrant
stores the handbook index. Run migrations with `uv run alembic upgrade head`.

Per-stage timings and provider-reported token counts are logged when available.
Provider dollar costs are not estimated. The LLM read timeout is configurable
with `LLM_READ_TIMEOUT_SECONDS` (default: 180 seconds).
