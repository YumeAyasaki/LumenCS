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

This preview command is only a development aid; database persistence can be
added later as a separate server integration.
