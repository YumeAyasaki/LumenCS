"""Tests for provider configuration and source-grounded LLM orchestration."""

from __future__ import annotations

import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.config import Settings
from app.llm_agent import (
    ChatChoice,
    ChatCompletion,
    ChatFunction,
    ChatMessage,
    ChatToolCall,
    TokenUsage,
    _execute_call,
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
        self.assertEqual(settings.llm_model, "gemini-2.5-flash")
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

        self.assertEqual(result, (call, tool_result, None))
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
                new=AsyncMock(return_value=(call, tool_result, None)),
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


if __name__ == "__main__":
    unittest.main()
