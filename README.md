# Lumen Home CS assistant

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

Build and run the spreadsheet preview alongside local Postgres and Qdrant:

```sh
docker compose up --build
```

The `app` container runs the existing preview command and exits when it has
printed the parsed workbook data. Postgres and Qdrant continue running with
persistent Docker volumes. The app does not yet connect to either database.
The source `data/` directory is mounted read-only, so updated workbooks are
available to the next app run.

To run only the databases, use `docker compose up -d postgres qdrant`. Their
ports are bound to localhost (`5432`, `6333`, and `6334`). Postgres defaults
are intended for local development and can be overridden with `POSTGRES_DB`,
`POSTGRES_USER`, and `POSTGRES_PASSWORD` environment variables.
