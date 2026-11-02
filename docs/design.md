# Lumen Home CS Assistant — Design

## Architecture

The service is a Vietnamese, retrieval-backed customer-support assistant. FastAPI exposes `/ask`, `/ingest`, `/health`, and typed-tool endpoints; requests and responses are validated with Pydantic. The Streamlit UI calls the API and displays answers, citations, and tool activity. See [the API](../app/main.py) and [the Streamlit UI](../ui/streamlit_app.py).

The system separates structured facts from handbook prose:

- **PostgreSQL** stores spreadsheet source workbooks, sheets, and rows, plus normalized tables for products, prices, promotions, warranty, regions, service centres, and service fees. Each normalized record links to its raw source row.
- **Qdrant** stores handbook clauses and tables as embedded, citeable chunks. The multilingual FastEmbed model is loaded at application startup.
- **The agent** calls a configured OpenAI-compatible chat-completions provider. It receives registered typed tools, not database access or free-form SQL.
- **Docker Compose** runs the API, Streamlit, PostgreSQL, and Qdrant. Alembic migrations run when the API container starts.

This split follows the shape of the source material. Spreadsheet rows are treated
as structured facts that need exact filtering, joins, and traceable provenance,
so they are normalized and queried in PostgreSQL. Handbook policies express
meaning in natural language; semantic similarity is a better retrieval fit for
those clauses, so they are indexed in Qdrant and searched on demand.

## Processing the materials

Spreadsheet ingestion reads configured Excel files and preserves the original workbook/sheet structure, headers, notes, footnotes, visibility, cell values, and Excel row numbers. [The schema mapping](../app/schema_mapping.json) specifies sheet layouts, field mappings, and parsers. The normalizer converts mapped rows into typed records; parse or mapping problems are reported as warnings rather than silently turned into facts. Raw rows remain available for auditing and re-normalization.

The spreadsheet ingester compares both the workbook SHA-256 and mapping SHA-256 to skip unchanged inputs. A changed workbook is replaced transactionally. Run ingestion with `make ingest DATA=<dir>`.

PDF ingestion extracts page text and tables with PyMuPDF, then groups content into sections, clauses, bullets, and table chunks. Chunks retain source file, clause, page, handbook version, effective date, and hashes. They are embedded into a collection associated with the embedding model; stable point IDs support replacement and stale-point cleanup. The reader warns about pages with no extractable text; it does not currently OCR scanned pages. See [the PDF reader](../app/tools/pdf_reader.py) and [the PDF ingester](../app/tools/pdf_ingester.py).

## How the agent decides what to do

The API normalizes `asked_at` to Vietnam time and supplies the question,
optional conversation history, and date context to the LLM. No handbook search
runs automatically before tool selection; the model can request a focused
`search_policy` call when relevant. History helps resolve follow-up references,
but factual claims must still be verified with tools during the current request.

The LLM may request registered tools such as `quote_price`, `get_warranty`, `find_service_centres`, `get_service_fee`, or a focused `search_policy`. Tool arguments are Pydantic-validated; the server supplies the request timestamp as `as_of`. Independent calls in a step run concurrently, each database-backed call using its own session. The agent maps results back to their originating calls and collects citations from successful results. Arithmetic uses `calculate`; date arithmetic uses `add_days`.

The loop currently permits up to 10 LLM iterations and up to 8 parallel tool calls per step. The 10-iteration limit differs from the intended maximum of 3 and should be aligned. The agent stops when the LLM returns a final answer or fails explicitly if it exhausts the limit.

## Grounding, time, and speed

Every spreadsheet result carries provenance that resolves to its source file, sheet, and Excel row. Handbook results carry clause citations and are expanded to the matching section, with the represented clause citations returned alongside it. The LLM is instructed to answer in Vietnamese using tool evidence only. Before returning a response, code validates its JSON shape and rejects citations not returned during that request; it retries an invalid answer or citation once, then raises an error.

Effective-date rules are enforced in application queries, not left to the prompt. Prices are filtered by `valid_from`; promotions by their start and end dates; handbook chunks by effective date. Some spreadsheet categories have no validity dates in the source materials, so their lookups accept `as_of` for a consistent tool interface but cannot filter by it.

For latency, the embedding model is loaded at startup, PostgreSQL lookups use typed tables, and independent tool calls run concurrently. The API supports streaming responses. It records per-stage durations and provider-reported token counts when available; it does not estimate provider dollar cost. Spreadsheet hash checks avoid repeating unchanged workbook ingestion.

## Known failure modes

- Scanned or image-only PDF pages may produce no text; they are warned about but not OCRed.
- New workbook layouts or ambiguous values may need mapping changes. Sheets can still be read as unstructured rows, while unmappable values produce warnings and may be absent from typed lookups.
- Semantic search is limited to the top three matching sections. Relevant wording may rank poorly, especially for short or ambiguous questions.
- Some source tables lack effective dates, and incomplete or conflicting source material can limit the answer. The assistant must disclose missing information rather than infer it.
- PostgreSQL, Qdrant, or the LLM provider can be unavailable. Provider timeouts and transient failures can delay or fail a request; invalid final output or citations still fail after one retry.
- Parallel tool calls are capped. Calls beyond the limit are reported as tool errors.

## Next steps

1. Re-evaluate handbook retrieval on representative Vietnamese questions. Measure whether focused on-demand search and query construction using conversation context find the right clauses; avoid reintroducing a forced search on every request unless evaluation shows a clear accuracy benefit.
2. Align the agent loop with the intended three-step maximum and add tests that enforce the limit.
3. Improve mapping review and ingestion reporting for new workbook versions, including clearer summaries of rows excluded from normalization.
4. Add request-level cost estimation where provider pricing and token data are available, and monitor retrieval quality in production.
5. Define source-precedence and effective-date policies for conflicting or undated materials, then test those policies with realistic edge cases.