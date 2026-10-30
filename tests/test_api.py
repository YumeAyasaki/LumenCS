"""Tests for the placeholder FastAPI endpoints."""

from __future__ import annotations

import asyncio
import unittest

from httpx import ASGITransport, AsyncClient, Response

from app.main import app


class ApiTests(unittest.TestCase):
    """Verify API health, request validation, and placeholder response shape."""

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
        """Return a successful health status."""
        response: Response = self.request("GET", "/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

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


if __name__ == "__main__":
    unittest.main()
