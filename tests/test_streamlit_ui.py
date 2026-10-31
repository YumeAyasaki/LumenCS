"""Tests for the Streamlit chat's FastAPI event-stream client."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import httpx

from ui.streamlit_app import _stream_answer


class StreamlitChatTests(unittest.TestCase):
    """Verify SSE parsing and request construction for the chat interface."""

    def test_stream_answer_collects_text_citations_and_latency(self) -> None:
        """Parse API answer, citation, and completion events."""
        request_details: dict[str, object] = {}
        event_stream: str = (
            'event: answer\n'
            'data: {"text":"Xin chào "}\n\n'
            'event: answer\n'
            'data: {"text":"bạn"}\n\n'
            'event: tool_calls\n'
            'data: {"tool_executions":[{"call_id":"call-1",'
            '"call":{"name":"quote_price","arguments":{"model":"LH-AC09N"}},'
            '"answer":{"tool_name":"quote_price","records":[],"citations":[]}}]}\n\n'
            'event: citations\n'
            'data: ["handbook.pdf#3.2"]\n\n'
            'event: metrics\n'
            'data: {"steps":[{"step":1,"kind":"llm","name":"answer",'
            '"duration_ms":100,"prompt_tokens":10,"completion_tokens":4}]}\n\n'
            'event: done\n'
            'data: {"latency_ms":125}\n\n'
        )

        def handler(request: httpx.Request) -> httpx.Response:
            request_details["url"] = str(request.url)
            request_details["accept"] = request.headers.get("Accept")
            request_details["payload"] = request.read().decode("utf-8")
            return httpx.Response(
                status_code=200,
                headers={"content-type": "text/event-stream"},
                text=event_stream,
            )

        transport: httpx.MockTransport = httpx.MockTransport(handler)
        client: httpx.Client = httpx.Client(transport=transport)
        with (
            patch("ui.streamlit_app.httpx.Client", return_value=client),
            patch.dict("os.environ", {"LUMEN_API_URL": "http://api:5000"}),
        ):
            answer, citations, latency_ms, metrics, executions = _stream_answer(
                question="Xin chào",
                history=[{"role": "assistant", "content": "Chào bạn"}],
            )

        self.assertEqual(answer, "Xin chào bạn")
        self.assertEqual(citations, ["handbook.pdf#3.2"])
        self.assertEqual(latency_ms, 125)
        self.assertEqual(metrics[0]["duration_ms"], 100)
        self.assertEqual(metrics[0]["prompt_tokens"], 10)
        self.assertEqual(executions[0]["call_id"], "call-1")
        self.assertEqual(executions[0]["call"]["name"], "quote_price")
        self.assertEqual(executions[0]["answer"]["tool_name"], "quote_price")
        self.assertEqual(request_details["url"], "http://api:5000/ask")
        self.assertEqual(request_details["accept"], "text/event-stream")
        payload: dict[str, object] = json.loads(str(request_details["payload"]))
        self.assertEqual(payload["question"], "Xin chào")
        self.assertEqual(
            payload["history"],
            [{"role": "assistant", "content": "Chào bạn"}],
        )
        self.assertIsNotNone(payload["asked_at"])

    def test_metrics_without_token_counts_remain_missing(self) -> None:
        """Parse durations without rendering fabricated token values."""
        event_stream: str = (
            'event: answer\n'
            'data: {"text":"Không có số liệu token."}\n\n'
            'event: metrics\n'
            'data: {"steps":[{"step":1,"kind":"llm","name":"answer",'
            '"duration_ms":100}]}\n\n'
        )
        transport: httpx.MockTransport = httpx.MockTransport(
            lambda request: httpx.Response(
                status_code=200,
                headers={"content-type": "text/event-stream"},
                text=event_stream,
            )
        )
        client: httpx.Client = httpx.Client(transport=transport)
        with (
            patch("ui.streamlit_app.httpx.Client", return_value=client),
            patch.dict("os.environ", {"LUMEN_API_URL": "http://api:5000"}),
        ):
            answer, _, _, metrics, executions = _stream_answer("Xin chào", [])

        self.assertEqual(answer, "Không có số liệu token.")
        self.assertEqual(executions, [])
        self.assertNotIn("prompt_tokens", metrics[0])
        self.assertNotIn("completion_tokens", metrics[0])

    def test_stream_error_response_is_readable_before_raising(self) -> None:
        """Keep provider API error details available for Streamlit."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=502,
                json={
                    "detail": (
                        "LLM provider returned HTTP 404: "
                        "models/gemini-3.5-flash-lite was not found."
                    )
                },
            )

        transport: httpx.MockTransport = httpx.MockTransport(handler)
        client: httpx.Client = httpx.Client(transport=transport)
        with (
            patch("ui.streamlit_app.httpx.Client", return_value=client),
            patch.dict("os.environ", {"LUMEN_API_URL": "http://api:5000"}),
        ):
            with self.assertRaises(httpx.HTTPStatusError) as raised:
                _stream_answer(question="Câu hỏi", history=[])

        self.assertIn(
            "gemini-3.5-flash-lite",
            raised.exception.response.text,
        )


if __name__ == "__main__":
    unittest.main()
