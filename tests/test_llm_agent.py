"""Tests for provider configuration and source-grounded LLM orchestration."""

from __future__ import annotations

import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from app.config import Settings
from app.llm_agent import (
    ChatChoice,
    ChatCompletion,
    ChatFunction,
    ChatMessage,
    ChatToolCall,
    LLMProviderError,
    TokenUsage,
    _complete,
    _execute_call,
    _provider_retry_delay,
    _provider_error_detail,
    answer_question,
)
from app.tools.model_tools import ToolResult


class LLMConfigurationTests(unittest.TestCase):
    """Verify environment configuration for supported provider families."""

    def test_selects_gemini_defaults_and_key(self) -> None:
        """Resolve the default Gemini OpenAI-compatible endpoint."""
        environment: dict[str, str] = {
            "DATABASE_URL": "postgresql+asyncpg://user:pass@localhost/lumen",
            "QDRANT_URL": "http://localhost:6333",
            "LLM_PROVIDER": "gemini",
            "GEMINI_KEY": "test-key",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("app.config.load_dotenv"),
        ):
            settings: Settings = Settings.from_environment()

        self.assertEqual(settings.llm_provider, "gemini")
        self.assertEqual(settings.llm_model, "gemini-3.5-flash-lite")
        self.assertEqual(settings.llm_read_timeout_seconds, 180.0)
        self.assertEqual(
            settings.llm_base_url,
            "https://generativelanguage.googleapis.com/v1beta/openai",
        )
        self.assertIsNotNone(settings.llm_api_key)
        assert settings.llm_api_key is not None
        self.assertEqual(settings.llm_api_key.get_secret_value(), "test-key")

    def test_selects_openai_defaults(self) -> None:
        """Resolve OpenAI model, URL, and provider-specific key."""
        environment: dict[str, str] = {
            "DATABASE_URL": "postgresql+asyncpg://user:pass@localhost/lumen",
            "QDRANT_URL": "http://localhost:6333",
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("app.config.load_dotenv"),
        ):
            settings: Settings = Settings.from_environment()

        self.assertEqual(settings.llm_provider, "openai")
        self.assertEqual(settings.llm_model, "gpt-4o-mini")
        self.assertEqual(settings.llm_base_url, "https://api.openai.com/v1")

    def test_compatible_provider_requires_custom_url_and_model_at_use(self) -> None:
        """Allow local compatible services to omit authentication."""
        environment: dict[str, str] = {
            "DATABASE_URL": "postgresql+asyncpg://user:pass@localhost/lumen",
            "QDRANT_URL": "http://localhost:6333",
            "LLM_PROVIDER": "openai-compatible",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("app.config.load_dotenv"),
        ):
            settings: Settings = Settings.from_environment()

        self.assertEqual(settings.llm_provider, "openai-compatible")
        self.assertIsNone(settings.llm_api_key)


class LLMAgentTests(unittest.IsolatedAsyncioTestCase):
    """Verify tool timestamp enforcement and citation-safe response retries."""

    def setUp(self) -> None:
        """Create a configured provider with mock data-store clients."""
        self.settings: Settings = Settings(
            database_url="postgresql+asyncpg://user:pass@localhost/lumen",
            qdrant_url="http://localhost:6333",
            llm_provider="openai-compatible",
            llm_model="test-model",
            llm_base_url="http://localhost:8000/v1",
        )
        self.session_factory: MagicMock = MagicMock()
        self.qdrant_client: MagicMock = MagicMock()
        self.embedding_model: MagicMock = MagicMock()

    def test_provider_error_detail_is_actionable_and_redacted(self) -> None:
        """Extract a provider error message without exposing its API key."""
        secret: str = "sensitive-test-key"
        settings: Settings = Settings(
            database_url="postgresql+asyncpg://user:pass@localhost/lumen",
            qdrant_url="http://localhost:6333",
            llm_provider="gemini",
            llm_model="gemini-3.5-flash-lite",
            llm_base_url=(
                "https://generativelanguage.googleapis.com/v1beta/openai"
            ),
            llm_api_key=secret,
        )
        response: httpx.Response = httpx.Response(
            status_code=404,
            json={
                "error": {
                    "message": f"Model unavailable for key {secret}",
                }
            },
        )

        detail: str = _provider_error_detail(response, settings)

        self.assertEqual(
            detail,
            "Model unavailable for key [redacted]",
        )

    def test_provider_503_without_body_gets_actionable_message(self) -> None:
        """Explain that an empty provider 503 is transient."""
        response: httpx.Response = httpx.Response(status_code=503)

        detail: str = _provider_error_detail(response, self.settings)

        self.assertIn("temporarily unavailable", detail)
        self.assertIn("try again shortly", detail)

    def test_retry_delay_respects_retry_after_with_a_safe_cap(self) -> None:
        """Honor a numeric retry hint without waiting unboundedly."""
        response: httpx.Response = httpx.Response(
            status_code=503,
            headers={"Retry-After": "30"},
        )

        self.assertEqual(_provider_retry_delay(response, attempt=0), 5.0)
        self.assertEqual(
            _provider_retry_delay(httpx.Response(status_code=503), attempt=1),
            1.0,
        )

    async def test_complete_retries_transient_provider_503(self) -> None:
        """Retry a transient 503 once before accepting a successful response."""
        error_response: httpx.Response = httpx.Response(
            status_code=503,
            request=httpx.Request(
                "POST",
                "http://localhost:8000/v1/chat/completions",
            ),
        )
        successful_response: httpx.Response = httpx.Response(
            status_code=200,
            request=httpx.Request(
                "POST",
                "http://localhost:8000/v1/chat/completions",
            ),
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                '{"answer":"Tôi có thể giúp.",'
                                '"citations":[]}'
                            ),
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 4,
                },
            },
        )
        mock_client: MagicMock = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(
            side_effect=[error_response, successful_response]
        )
        with (
            patch("app.llm_agent.httpx.AsyncClient", return_value=mock_client),
            patch("app.llm_agent.asyncio.sleep", new=AsyncMock()) as sleep,
        ):
            result = await _complete(
                self.settings,
                messages=[{"role": "user", "content": "Xin chào"}],
                allow_tools=False,
            )

        self.assertEqual(result.choices[0].message.content, '{"answer":"Tôi có thể giúp.","citations":[]}')
        self.assertEqual(mock_client.post.await_count, 2)
        sleep.assert_awaited_once_with(0.5)

    async def test_missing_provider_usage_stays_unavailable(self) -> None:
        """Keep token fields absent when the provider omits usage data."""
        response: ChatCompletion = ChatCompletion.model_validate(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                '{"answer":"Tôi có thể giúp.",'
                                '"citations":[]}'
                            ),
                        }
                    }
                ]
            }
        )

        self.assertIsNone(response.usage)

    async def test_complete_retries_read_timeout(self) -> None:
        """Retry a provider read timeout before returning a successful answer."""
        successful_response: httpx.Response = httpx.Response(
            status_code=200,
            request=httpx.Request(
                "POST",
                "http://localhost:8000/v1/chat/completions",
            ),
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                '{"answer":"Tôi có thể giúp.",'
                                '"citations":[]}'
                            ),
                        }
                    }
                ],
            },
        )
        mock_client: MagicMock = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(
            side_effect=[
                httpx.ReadTimeout("provider response timed out"),
                successful_response,
            ]
        )
        with (
            patch("app.llm_agent.httpx.AsyncClient", return_value=mock_client),
            patch("app.llm_agent.asyncio.sleep", new=AsyncMock()) as sleep,
        ):
            result = await _complete(
                self.settings,
                messages=[{"role": "user", "content": "Xin chào"}],
                allow_tools=False,
            )

        self.assertEqual(
            result.choices[0].message.content,
            '{"answer":"Tôi có thể giúp.","citations":[]}',
        )
        self.assertEqual(mock_client.post.await_count, 2)
        sleep.assert_awaited_once_with(0.5)

    async def test_complete_reports_read_timeout_after_all_attempts(self) -> None:
        """Return a clear error if provider read timeouts persist."""
        mock_client: MagicMock = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(
            side_effect=httpx.ReadTimeout("provider response timed out")
        )
        with (
            patch("app.llm_agent.httpx.AsyncClient", return_value=mock_client),
            patch("app.llm_agent.asyncio.sleep", new=AsyncMock()),
        ):
            with self.assertRaisesRegex(
                LLMProviderError,
                "Timed out waiting for LLM provider",
            ):
                await _complete(
                    self.settings,
                    messages=[{"role": "user", "content": "Xin chào"}],
                    allow_tools=False,
                )

        self.assertEqual(mock_client.post.await_count, 3)

    async def test_tool_call_uses_request_timestamp(self) -> None:
        """Override model-supplied as_of with the request timestamp."""
        call: ChatToolCall = ChatToolCall(
            id="call-1",
            type="function",
            function=ChatFunction(
                name="calculate",
                arguments='{"as_of":"1900-01-01T00:00:00Z","expression":"1 + 1"}',
            ),
        )
        session: AsyncMock = AsyncMock()
        session_context: MagicMock = MagicMock()
        session_context.__aenter__ = AsyncMock(return_value=session)
        session_context.__aexit__ = AsyncMock(return_value=None)
        self.session_factory.return_value = session_context
        tool_result: ToolResult = ToolResult(
            tool_name="calculate",
            records=[],
            citations=[],
        )
        with patch(
            "app.llm_agent.execute_model_tool",
            new=AsyncMock(return_value=tool_result),
        ) as execute_tool:
            result = await _execute_call(
                call=call,
                asked_at="2026-10-29T21:00:00+07:00",
                session_factory=self.session_factory,
                qdrant_client=self.qdrant_client,
                embedding_model=self.embedding_model,
                settings=self.settings,
            )

        self.assertEqual(result[:3], (call, tool_result, None))
        arguments: dict[str, object] = execute_tool.await_args.args[1]
        self.assertEqual(arguments["as_of"], "2026-10-29T21:00:00+07:00")

    async def test_retries_final_answer_with_unapproved_citation(self) -> None:
        """Retry once and return only citations received from a tool."""
        call: ChatToolCall = ChatToolCall(
            id="call-1",
            type="function",
            function=ChatFunction(
                name="search_policy",
                arguments='{"as_of":"1900-01-01T00:00:00Z","question":"warranty"}',
            ),
        )
        tool_result: ToolResult = ToolResult(
            tool_name="search_policy",
            records=[],
            citations=["handbook.pdf#3.2"],
        )
        completions: list[ChatCompletion] = [
            ChatCompletion(
                choices=[
                    ChatChoice(
                        message=ChatMessage(
                            role="assistant",
                            tool_calls=[call],
                        )
                    )
                ],
                usage=TokenUsage(prompt_tokens=10, completion_tokens=2),
            ),
            ChatCompletion(
                choices=[
                    ChatChoice(
                        message=ChatMessage(
                            role="assistant",
                            content=(
                                '{"answer":"Chính sách bảo hành.",'
                                '"citations":["invented.pdf#9"]}'
                            ),
                        )
                    )
                ],
                usage=TokenUsage(prompt_tokens=12, completion_tokens=5),
            ),
            ChatCompletion(
                choices=[
                    ChatChoice(
                        message=ChatMessage(
                            role="assistant",
                            content=(
                                '{"answer":"Chính sách bảo hành.",'
                                '"citations":["handbook.pdf#3.2"]}'
                            ),
                        )
                    )
                ],
                usage=TokenUsage(prompt_tokens=14, completion_tokens=5),
            ),
        ]
        with (
            patch(
                "app.llm_agent._complete",
                new=AsyncMock(side_effect=completions),
            ) as complete,
            patch(
                "app.llm_agent._execute_call",
                new=AsyncMock(return_value=(call, tool_result, None, 12)),
            ),
        ):
            result = await answer_question(
                question="Điều kiện bảo hành là gì?",
                asked_at="2026-10-29T21:00:00+07:00",
                history=[],
                session_factory=self.session_factory,
                qdrant_client=self.qdrant_client,
                embedding_model=self.embedding_model,
                settings=self.settings,
            )

        self.assertEqual(result.answer, "Chính sách bảo hành.")
        self.assertEqual(result.citations, ["handbook.pdf#3.2"])
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual(result.prompt_tokens, 36)
        self.assertEqual(result.completion_tokens, 12)
        self.assertEqual(complete.await_count, 3)
        self.assertEqual(
            [(metric.kind, metric.name) for metric in result.steps],
            [
                ("llm", "tool_selection"),
                ("tool", "search_policy"),
                ("llm", "answer"),
                ("llm", "citation_validation_retry"),
            ],
        )
        self.assertEqual(result.steps[0].prompt_tokens, 10)
        self.assertEqual(result.steps[0].completion_tokens, 2)
        self.assertIsNone(result.steps[1].prompt_tokens)
        self.assertIsNone(result.steps[1].completion_tokens)

    async def test_preserves_provider_tool_signature_for_followup_request(self) -> None:
        """Echo Gemini thought signatures with the original assistant call."""
        signature: str = "opaque-provider-signature"
        completion_with_tool: ChatCompletion = ChatCompletion.model_validate(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call-with-signature",
                                    "type": "function",
                                    "function": {
                                        "name": "search_policy",
                                        "arguments": (
                                            '{"as_of":"1900-01-01T00:00:00Z",'
                                            '"question":"warranty"}'
                                        ),
                                    },
                                    "extra_content": {
                                        "google": {
                                            "thought_signature": signature,
                                        }
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        )
        final_completion: ChatCompletion = ChatCompletion(
            choices=[
                ChatChoice(
                    message=ChatMessage(
                        role="assistant",
                        content='{"answer":"Có bảo hành.","citations":[]}',
                    )
                )
            ]
        )
        call: ChatToolCall = (
            completion_with_tool.choices[0].message.tool_calls[0]
        )
        tool_result: ToolResult = ToolResult(
            tool_name="search_policy",
            records=[],
            citations=[],
        )
        with (
            patch(
                "app.llm_agent._complete",
                new=AsyncMock(
                    side_effect=[completion_with_tool, final_completion]
                ),
            ) as complete,
            patch(
                "app.llm_agent._execute_call",
                new=AsyncMock(return_value=(call, tool_result, None, 12)),
            ),
        ):
            result = await answer_question(
                question="Chính sách bảo hành thế nào?",
                asked_at="2026-10-29T21:00:00+07:00",
                history=[],
                session_factory=self.session_factory,
                qdrant_client=self.qdrant_client,
                embedding_model=self.embedding_model,
                settings=self.settings,
            )

        self.assertEqual(result.answer, "Có bảo hành.")
        followup_messages: list[dict[str, object]] = (
            complete.await_args_list[1].args[1]
        )
        assistant_tool_message: dict[str, object] = followup_messages[2]
        forwarded_tool_calls: list[dict[str, object]] = (
            assistant_tool_message["tool_calls"]
        )
        self.assertEqual(
            forwarded_tool_calls[0]["extra_content"],
            {"google": {"thought_signature": signature}},
        )


if __name__ == "__main__":
    unittest.main()
