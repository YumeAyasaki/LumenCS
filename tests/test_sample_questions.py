"""Tests for retries in the sample-question runner."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, call, patch

import httpx

from scripts.run_sample_questions import (
    DEFAULT_QUESTIONS_FILE,
    MAX_QUESTION_RETRIES,
    SampleQuestion,
    _load_sample_questions,
    _post_question_with_retries,
    _question_payload,
)


class SampleQuestionRetryTests(unittest.TestCase):
    """Verify retry behavior for transient API request failures."""

    def test_loads_default_question_file(self) -> None:
        """Load all sample questions from the adjacent JSON file."""
        questions = _load_sample_questions(DEFAULT_QUESTIONS_FILE)

        self.assertEqual(len(questions), 32)
        self.assertEqual(
            questions[0].asked_at_local,
            "27/10/2026 10:05",
        )
        self.assertIn("LH-AC12N", questions[0].question)
        history_questions: list[SampleQuestion] = [
            question for question in questions if question.history
        ]
        self.assertEqual(
            [question.question for question in history_questions],
            [
                "Tôi ở Đà Nẵng, quận Hải Châu, cần bảo hành máy giặt.",
                "Còn ở Hà Nội thì sao?",
                "Vậy nếu tôi mua ở cửa hàng thì sao?",
            ],
        )

    def test_builds_request_payload_with_conversation_history(self) -> None:
        """Send ordered history turns to the API with the current question."""
        sample: SampleQuestion = _load_sample_questions(DEFAULT_QUESTIONS_FILE)[30]

        payload: dict[str, object] = _question_payload(sample)

        self.assertEqual(payload["question"], "Còn ở Hà Nội thì sao?")
        self.assertEqual(payload["asked_at"], "2026-10-27T10:10:00+07:00")
        self.assertEqual(
            payload["history"],
            [
                {
                    "role": "user",
                    "content": "Điều hòa LH-AC12N giá bao nhiêu ở Đà Nẵng?",
                },
                {
                    "role": "assistant",
                    "content": (
                        "LH-AC12N tại Đà Nẵng (vùng giá Miền Trung) có giá "
                        "niêm yết 11.090.000 đ. Đến hết 31/10/2026 có "
                        "KM-2610 giảm 10% (tối đa 2.000.000 đ) → còn "
                        "9.981.000 đ."
                    ),
                },
            ],
        )

    def test_rejects_invalid_question_file(self) -> None:
        """Report invalid question input with its file path."""
        with TemporaryDirectory() as temporary_directory:
            question_file: Path = Path(temporary_directory) / "questions.json"
            question_file.write_text(
                '[{"asked_at_local":"invalid","question":"Test"}]',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "questions.json"):
                _load_sample_questions(question_file)

    def test_retries_transient_response_and_honors_retry_after(self) -> None:
        """Retry a transient response after the server-requested delay."""
        request: httpx.Request = httpx.Request("POST", "http://localhost/ask")
        rate_limited: httpx.Response = httpx.Response(
            status_code=429,
            headers={"Retry-After": "15"},
            request=request,
        )
        successful: httpx.Response = httpx.Response(
            status_code=200,
            request=request,
        )
        client: Mock = Mock()
        client.post.side_effect = [rate_limited, successful]

        with patch("scripts.run_sample_questions.time.sleep") as sleep:
            response, client_latency_ms = _post_question_with_retries(
                client=client,
                url="http://localhost/ask",
                payload={"question": "Giá bao nhiêu?"},
            )

        self.assertIs(response, successful)
        self.assertGreaterEqual(client_latency_ms, 0)
        self.assertEqual(client.post.call_count, 2)
        sleep.assert_called_once_with(15.0)

    def test_client_latency_excludes_retry_waits(self) -> None:
        """Sum only HTTP attempt time, not the intervening retry pause."""
        request: httpx.Request = httpx.Request("POST", "http://localhost/ask")
        rate_limited: httpx.Response = httpx.Response(
            status_code=502,
            request=request,
        )
        successful: httpx.Response = httpx.Response(
            status_code=200,
            request=request,
        )
        client: Mock = Mock()
        client.post.side_effect = [rate_limited, successful]
        clock_values: list[float] = [1.0, 3.0, 100.0, 104.0]

        with (
            patch("scripts.run_sample_questions.time.sleep"),
            patch(
                "scripts.run_sample_questions.time.perf_counter",
                side_effect=clock_values,
            ),
        ):
            _, client_latency_ms = _post_question_with_retries(
                client=client,
                url="http://localhost/ask",
                payload={"question": "Giá bao nhiêu?"},
            )

        self.assertEqual(client_latency_ms, 6000)

    def test_stops_after_five_retries(self) -> None:
        """Return the final transient response after the retry budget is spent."""
        request: httpx.Request = httpx.Request("POST", "http://localhost/ask")
        failure: httpx.Response = httpx.Response(
            status_code=502,
            request=request,
        )
        client: Mock = Mock()
        client.post.return_value = failure

        with patch("scripts.run_sample_questions.time.sleep") as sleep:
            response, client_latency_ms = _post_question_with_retries(
                client=client,
                url="http://localhost/ask",
                payload={"question": "Giá bao nhiêu?"},
            )

        self.assertIs(response, failure)
        self.assertGreaterEqual(client_latency_ms, 0)
        self.assertEqual(client.post.call_count, MAX_QUESTION_RETRIES + 1)
        self.assertEqual(
            sleep.call_args_list,
            [call(10.0)] * MAX_QUESTION_RETRIES,
        )


if __name__ == "__main__":
    unittest.main()
