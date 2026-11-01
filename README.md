# Lumen Home CS Assistant

Vietnamese customer-support API and chat UI grounded in the supplied Excel
workbooks and policy handbook.

## Quick start

1. Copy `.env.example` to `.env` for a new checkout:

   ```sh
   cp -n .env.example .env
   ```

   If `.env` already exists, add any missing settings without replacing local
   credentials.
2. Set `GEMINI_KEY` in `.env`, or configure another supported LLM provider.
3. Start the API, Streamlit UI, PostgreSQL, and Qdrant:

   ```sh
   docker compose --env-file .env.example --env-file .env up --build
   ```

4. Open <http://127.0.0.1:8501>, then ingest the supplied materials:

   ```sh
   curl -X POST http://127.0.0.1:5000/ingest
   ```

API docs: <http://127.0.0.1:5000/docs>

## More information

- [Usage and configuration](docs/usage.md): local development, API, tools,
  environment variables, and ingestion.
- [Design](docs/design.md): architecture, grounding, retrieval, limitations,
  and next steps.
