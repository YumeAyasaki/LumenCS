"""Tests for the FastAPI endpoints."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import ASGITransport, AsyncClient, Response
from sqlalchemy.exc import SQLAlchemyError

from app.config import Settings
from app.llm_agent import (
    AgentResult,
    AgentStepMetrics,
    AgentToolExecution,
    LLMConfigurationError,
    LLMProviderError,
)
from app.main import app, lifespan
from app.tools.ingest import IngestAllResult
from app.tools.model_tools import ToolInvocationCall, ToolResult
from app.tools.pdf_ingester import PDFIngestionResult
from app.tools.spreadsheet_ingester import IngestionResult


class ApiTests(unittest.TestCase):
    """Verify API health, retrieval, ingestion, and request validation."""

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
        app.state.pdf_embedding_model = MagicMock()
        app.state.settings = Settings(
            database_url="postgresql+asyncpg://user:pass@localhost/lumen",
            qdrant_url="http://localhost:6333",
        )

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
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
                return await client.request(
                    method,
                    path,
                    json=payload,
                    headers=headers,
                )

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
        embedding_model: MagicMock = MagicMock()

        async def run_lifespan() -> None:
            async with lifespan(app):
                self.assertIs(app.state.postgres_engine, postgres_engine)
                self.assertIs(app.state.qdrant_client, qdrant_client)
                self.assertIs(app.state.qdrant_sync_client, qdrant_sync_client)
                self.assertIs(app.state.pdf_embedding_model, embedding_model)

        with (
            patch("app.main.create_async_engine", return_value=postgres_engine),
            patch("app.main.AsyncQdrantClient", return_value=qdrant_client),
            patch("app.main.QdrantClient", return_value=qdrant_sync_client),
            patch("app.main.TextEmbedding", return_value=embedding_model),
        ):
            asyncio.run(run_lifespan())

        postgres_engine.dispose.assert_awaited_once()
        qdrant_client.close.assert_awaited_once()
        qdrant_sync_client.close.assert_called_once()

    def test_ask_returns_generated_answer_and_tool_citations(self) -> None:
        """Return the LLM answer and citations approved by its tool run."""
        result: AgentResult = AgentResult(
            answer="Bạn cần kích hoạt trong 45 ngày.",
            citations=["policy_handbook.pdf#3.2"],
            prompt_tokens=100,
            completion_tokens=25,
            tool_calls=1,
            llm_latency_ms=150,
            steps=[
                AgentStepMetrics(
                    step=1,
                    kind="llm",
                    name="answer",
                    duration_ms=150,
                    prompt_tokens=100,
                    completion_tokens=25,
                )
            ],
            tool_executions=[
                AgentToolExecution(
                    call_id="call-warranty-1",
                    call=ToolInvocationCall(
                        name="get_warranty",
                        arguments={
                            "model": "LH-AC09N",
                            "as_of": "2026-10-29T21:00:00+07:00",
                        },
                    ),
                    answer=ToolResult(
                        tool_name="get_warranty",
                        records=[],
                        citations=["product_catalog.xlsx#sheet=warranty&row=6"],
                    ),
                )
            ],
        )
        with patch(
            "app.main.answer_question",
            new=AsyncMock(return_value=result),
        ) as answer_question:
            response: Response = self.request(
                "POST",
                "/ask",
                {
                    "question": "Khi nào cần kích hoạt bảo hành?",
                    "asked_at": "2026-10-29T21:00:00Z",
                    "history": [
                        {"role": "user", "content": "Tôi đang xem LH-AC09N."},
                        {
                            "role": "assistant",
                            "content": "Anh/chị cần hỗ trợ gì về sản phẩm?",
                        },
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            set(response.json()),
            {
                "answer",
                "citations",
                "latency_ms",
                "steps",
                "tool_executions",
            },
        )
        self.assertEqual(response.json()["answer"], result.answer)
        self.assertEqual(response.json()["citations"], ["policy_handbook.pdf#3.2"])
        self.assertEqual(response.json()["steps"][0]["name"], "answer")
        self.assertEqual(
            response.json()["steps"][0]["duration_ms"],
            150,
        )
        self.assertEqual(response.json()["steps"][0]["prompt_tokens"], 100)
        self.assertEqual(response.json()["steps"][0]["completion_tokens"], 25)
        self.assertEqual(
            response.json()["tool_executions"][0]["call"]["name"],
            "get_warranty",
        )
        self.assertEqual(
            response.json()["tool_executions"][0]["answer"]["citations"],
            ["product_catalog.xlsx#sheet=warranty&row=6"],
        )
        self.assertNotIn("estimated_cost_usd", response.json())
        self.assertNotIn("estimated_cost_usd", response.json()["steps"][0])
        self.assertGreaterEqual(response.json()["latency_ms"], 0)
        answer_question.assert_awaited_once()
        self.assertEqual(
            answer_question.await_args.kwargs["asked_at"],
            "2026-10-30T04:00:00+07:00",
        )
        self.assertEqual(
            answer_question.await_args.kwargs["history"],
            [
                {"role": "user", "content": "Tôi đang xem LH-AC09N."},
                {
                    "role": "assistant",
                    "content": "Anh/chị cần hỗ trợ gì về sản phẩm?",
                },
            ],
        )

    def test_ask_reports_provider_configuration_failure(self) -> None:
        """Return a service error without exposing provider configuration."""
        with patch(
            "app.main.answer_question",
            new=AsyncMock(side_effect=LLMConfigurationError("missing key")),
        ):
            response: Response = self.request(
                "POST",
                "/ask",
                {
                    "question": "Thông tin này có trong sổ tay không?",
                    "asked_at": "2026-10-29T21:00:00+07:00",
                },
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"],
            "The LLM provider is not configured.",
        )

    def test_ask_reports_provider_failure(self) -> None:
        """Return HTTP 502 for a failed provider call."""
        with patch(
            "app.main.answer_question",
            new=AsyncMock(side_effect=LLMProviderError("provider unavailable")),
        ):
            response: Response = self.request(
                "POST",
                "/ask",
                {
                    "question": "Tôi cần hỗ trợ gì?",
                    "asked_at": "2026-10-29T21:00:00+07:00",
                },
            )

        self.assertEqual(response.status_code, 502)

    def test_ask_streams_validated_answer_when_requested(self) -> None:
        """Return server-sent answer, citation, and completion events."""
        result: AgentResult = AgentResult(
            answer="Được hỗ trợ bảo hành.",
            citations=["handbook.pdf#3.2"],
            prompt_tokens=10,
            completion_tokens=5,
            tool_calls=1,
            llm_latency_ms=25,
            steps=[
                AgentStepMetrics(
                    step=1,
                    kind="llm",
                    name="answer",
                    duration_ms=25,
                    prompt_tokens=10,
                    completion_tokens=5,
                )
            ],
            tool_executions=[
                AgentToolExecution(
                    call_id="call-policy-1",
                    call=ToolInvocationCall(
                        name="search_policy",
                        arguments={
                            "query": "kích hoạt bảo hành",
                            "as_of": "2026-10-29T21:00:00+07:00",
                        },
                    ),
                    answer=ToolResult(
                        tool_name="search_policy",
                        records=[],
                        citations=["handbook.pdf#3.2"],
                    ),
                )
            ],
        )
        with patch(
            "app.main.answer_question",
            new=AsyncMock(return_value=result),
        ):
            response: Response = self.request(
                "POST",
                "/ask",
                {
                    "question": "Tôi được bảo hành không?",
                    "asked_at": "2026-10-29T21:00:00+07:00",
                },
                headers={"Accept": "text/event-stream"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/event-stream; charset=utf-8")
        self.assertIn("event: answer", response.text)
        self.assertIn("event: tool_calls", response.text)
        self.assertIn('"call_id": "call-policy-1"', response.text)
        self.assertIn('"name": "search_policy"', response.text)
        self.assertIn('"query": "kích hoạt bảo hành"', response.text)
        self.assertIn('"tool_name": "search_policy"', response.text)
        self.assertIn('"handbook.pdf#3.2"', response.text)
        self.assertIn("event: citations", response.text)
        self.assertIn('["handbook.pdf#3.2"]', response.text)
        self.assertIn("event: metrics", response.text)
        self.assertIn('"name": "answer"', response.text)
        self.assertNotIn("estimated_cost_usd", response.text)
        self.assertIn("event: done", response.text)

    def test_ask_omits_token_counts_when_provider_does_not_report_them(self) -> None:
        """Omit unavailable token fields instead of returning zero estimates."""
        result: AgentResult = AgentResult(
            answer="Chào bạn.",
            citations=[],
            prompt_tokens=None,
            completion_tokens=None,
            tool_calls=0,
            llm_latency_ms=50,
            steps=[
                AgentStepMetrics(
                    step=1,
                    kind="llm",
                    name="answer",
                    duration_ms=50,
                    prompt_tokens=None,
                    completion_tokens=None,
                )
            ],
        )
        with patch(
            "app.main.answer_question",
            new=AsyncMock(return_value=result),
        ):
            response: Response = self.request(
                "POST",
                "/ask",
                {
                    "question": "Xin chào",
                    "asked_at": "2026-10-29T21:00:00+07:00",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("prompt_tokens", response.json())
        self.assertNotIn("completion_tokens", response.json())
        self.assertNotIn("prompt_tokens", response.json()["steps"][0])
        self.assertNotIn("completion_tokens", response.json()["steps"][0])

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

    def test_list_model_tools(self) -> None:
        """Publish model tool schemas for agent function calling."""
        response: Response = self.request("GET", "/tools")

        self.assertEqual(response.status_code, 200)
        definitions: list[dict[str, object]] = response.json()
        names: list[str] = []
        for item in definitions:
            function: object = item.get("function")
            if isinstance(function, dict):
                name: object = function.get("name")
                if isinstance(name, str):
                    names.append(name)
        self.assertIn("quote_price", names)
        self.assertIn("search_policy", names)
        self.assertIn("calculate", names)
        self.assertNotIn("get_region", names)
        self.assertNotIn("get_date_context", names)
        quote_schema: dict[str, object] = next(
            item["function"]
            for item in definitions
            if item["function"]["name"] == "quote_price"
        )["parameters"]
        self.assertNotIn("as_of", quote_schema["properties"])
        self.assertNotIn("title", str(quote_schema))

    def test_invoke_model_tool(self) -> None:
        """Execute a validated model function call through the API."""
        session: AsyncMock = AsyncMock()
        session_context: MagicMock = MagicMock()
        session_context.__aenter__ = AsyncMock(return_value=session)
        session_context.__aexit__ = AsyncMock(return_value=None)
        app.state.postgres_session_factory = MagicMock(
            return_value=session_context
        )

        response: Response = self.request(
            "POST",
            "/tools/calculate",
            {
                "arguments": {
                    "expression": "100 * 0.9",
                }
            },
        )

        self.assertEqual(response.status_code, 200)
        payload: dict[str, object] = response.json()
        call: object = payload["call"]
        self.assertIsInstance(call, dict)
        self.assertEqual(call["name"], "calculate")
        self.assertEqual(
            call["arguments"]["expression"],
            "100 * 0.9",
        )
        self.assertTrue(call["arguments"]["as_of"].endswith("+07:00"))
        answer: object = payload["answer"]
        self.assertIsInstance(answer, dict)
        self.assertEqual(answer["tool_name"], "calculate")
        self.assertEqual(answer["citations"], [])
        records: object = answer["records"]
        self.assertIsInstance(records, list)
        first_record: object = records[0]
        self.assertIsInstance(first_record, dict)
        self.assertEqual(first_record["result"], "90.0")

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
                collection_name="policy_documents_multilingual-model",
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
