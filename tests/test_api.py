"""Tests for the placeholder FastAPI endpoints."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import ASGITransport, AsyncClient, Response
from sqlalchemy.exc import SQLAlchemyError

from app.tools.ingest import IngestAllResult
from app.tools.pdf_ingester import PDFIngestionResult
from app.tools.spreadsheet_ingester import IngestionResult
from app.main import app, lifespan


class ApiTests(unittest.TestCase):
    """Verify API health, request validation, and placeholder response shape."""

    def setUp(self) -> None:
        """Provide mock database clients for endpoint tests."""
        self.connection: MagicMock = MagicMock()
        self.connection.execute = AsyncMock()
        self.connection_context: MagicMock = MagicMock()
        self.connection_context.__aenter__ = AsyncMock(
            return_value=self.connection
        )
        self.connection_context.__aexit__ = AsyncMock(return_value=None)
        self.postgres_engine: MagicMock = MagicMock()
        self.postgres_engine.connect.return_value = self.connection_context
        self.qdrant_client: MagicMock = MagicMock()
        self.qdrant_client.get_collections = AsyncMock()
        self.qdrant_sync_client: MagicMock = MagicMock()
        app.state.postgres_engine = self.postgres_engine
        app.state.postgres_session_factory = MagicMock()
        app.state.qdrant_client = self.qdrant_client
        app.state.qdrant_sync_client = self.qdrant_sync_client

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
    ) -> Response:
        """Send an HTTP request through the application ASGI interface.

        Args:
            method: HTTP method to use.
            path: Application path to request.
            payload: Optional JSON request body.

        Returns:
            Response: The response produced by the ASGI application.
        """

        async def send_request() -> Response:
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://testserver",
            ) as client:
                return await client.request(method, path, json=payload)

        return asyncio.run(send_request())

    def test_health_check(self) -> None:
        """Return successful PostgreSQL and Qdrant health checks."""
        response: Response = self.request("GET", "/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"status": "ok", "postgres": True, "qdrant": True},
        )
        self.connection.execute.assert_awaited_once()
        self.qdrant_client.get_collections.assert_awaited_once()

    def test_health_check_reports_unavailable_database(self) -> None:
        """Return HTTP 503 when PostgreSQL cannot answer a health check."""
        self.connection.execute.side_effect = SQLAlchemyError("database offline")

        response: Response = self.request("GET", "/health")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"],
            {
                "status": "unavailable",
                "postgres": False,
                "qdrant": True,
            },
        )

    def test_lifespan_creates_and_closes_database_clients(self) -> None:
        """Initialize and release both backing-store clients."""
        postgres_engine: MagicMock = MagicMock()
        postgres_engine.dispose = AsyncMock()
        qdrant_client: MagicMock = MagicMock()
        qdrant_client.close = AsyncMock()
        qdrant_sync_client: MagicMock = MagicMock()

        async def run_lifespan() -> None:
            async with lifespan(app):
                self.assertIs(app.state.postgres_engine, postgres_engine)
                self.assertIs(app.state.qdrant_client, qdrant_client)
                self.assertIs(app.state.qdrant_sync_client, qdrant_sync_client)

        with (
            patch("app.main.create_async_engine", return_value=postgres_engine),
            patch("app.main.AsyncQdrantClient", return_value=qdrant_client),
            patch("app.main.QdrantClient", return_value=qdrant_sync_client),
        ):
            asyncio.run(run_lifespan())

        postgres_engine.dispose.assert_awaited_once()
        qdrant_client.close.assert_awaited_once()
        qdrant_sync_client.close.assert_called_once()

    def test_ask_returns_placeholder_response(self) -> None:
        """Accept a valid question and return the documented response fields."""
        response: Response = self.request(
            "POST",
            "/ask",
            {
                "question": "Chính sách bảo hành như thế nào?",
                "asked_at": "2026-10-29T21:00:00+07:00",
                "history": [{"role": "user", "content": "Xin chào"}],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            set(response.json()),
            {"answer", "citations", "latency_ms"},
        )
        self.assertIn("đang được phát triển", response.json()["answer"])
        self.assertEqual(response.json()["citations"], [])
        self.assertGreaterEqual(response.json()["latency_ms"], 0)

    def test_ask_rejects_empty_question(self) -> None:
        """Reject a question containing only whitespace."""
        response: Response = self.request(
            "POST",
            "/ask",
            {
                "question": "   ",
                "asked_at": "2026-10-29T21:00:00+07:00",
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_ingest_endpoint_runs_both_source_ingesters(self) -> None:
        """Return the combined PostgreSQL and Qdrant ingestion summary."""
        result: IngestAllResult = IngestAllResult(
            spreadsheets=IngestionResult(
                processed_workbooks=2,
                skipped_workbooks=0,
                row_count=87,
            ),
            pdfs=PDFIngestionResult(
                processed_files=1,
                page_count=12,
                chunk_count=15,
                removed_count=0,
            ),
            spreadsheet_latency_ms=25,
            pdf_latency_ms=40,
            latency_ms=65,
        )

        async def run_ingestion(*args: object, **kwargs: object) -> IngestAllResult:
            return result

        with patch("app.main.ingest_all_sources", side_effect=run_ingestion) as ingest:
            response: Response = self.request("POST", "/ingest")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["spreadsheets"]["row_count"],
            87,
        )
        self.assertEqual(response.json()["pdfs"]["chunk_count"], 15)
        self.assertEqual(response.json()["latency_ms"], 65)
        ingest.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
