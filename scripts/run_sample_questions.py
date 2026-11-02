"""Run dated sample questions sequentially and persist each API response."""

from __future__ import annotations

import argparse
from email.utils import parsedate_to_datetime
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal
from zoneinfo import ZoneInfo

import httpx
from pydantic import BaseModel, Field, TypeAdapter, ValidationError, field_validator

DEFAULT_BASE_URL: str = "http://127.0.0.1:5000"
DEFAULT_OUTPUT: Path = Path("reports/model_answers_2026-10-27.json")
DEFAULT_QUESTIONS_FILE: Path = Path(__file__).with_name("sample_questions.json")
HEALTH_TIMEOUT_SECONDS: float = 5.0
STARTUP_POLL_SECONDS: float = 2.0
MAX_QUESTION_RETRIES: int = 10
QUESTION_RETRY_DELAY_SECONDS: float = 10.0
RETRIABLE_HTTP_STATUS_CODES: frozenset[int] = frozenset(
    {408, 429, 500, 502, 503, 504}
)
VIETNAM_TIMEZONE: ZoneInfo = ZoneInfo("Asia/Ho_Chi_Minh")


class ConversationMessage(BaseModel):
    """Represent one user or assistant turn supplied as question context.

    Attributes:
        role: Whether the message was written by the user or assistant.
        content: Text content of the conversation turn.
    """

    role: Literal["user", "assistant"]
    content: str


class SampleQuestion(BaseModel):
    """Define one dated question read from the sample-question JSON file.

    Attributes:
        asked_at_local: Vietnam-local time in ``dd/mm/yyyy HH:MM`` format.
        question: Vietnamese customer question to send to the API.
        history: Optional earlier user and assistant turns, in conversation
            order.
    """

    asked_at_local: str = Field(min_length=1)
    question: str = Field(min_length=1)
    history: list[ConversationMessage] = Field(default_factory=list)

    @field_validator("asked_at_local")
    @classmethod
    def validate_asked_at_local(cls, value: str) -> str:
        """Validate the local timestamp format used by the API request.

        Args:
            value: Timestamp string from the question file.

        Returns:
            str: The validated timestamp string.

        Raises:
            ValueError: If the timestamp is not a valid ``dd/mm/yyyy HH:MM``.
        """
        datetime.strptime(value, "%d/%m/%Y %H:%M")
        return value


class QuestionRequestError(Exception):
    """Keep request timing when all HTTP attempts fail.

    Attributes:
        original_error: The final HTTP error from the request client.
        client_latency_ms: Cumulative HTTP attempt duration, excluding waits.
    """

    def __init__(
        self,
        original_error: httpx.HTTPError,
        client_latency_ms: int,
    ) -> None:
        """Create an error with the final transport failure and attempt timing.

        Args:
            original_error: The final HTTP error raised by the client.
            client_latency_ms: Cumulative duration of HTTP attempts.
        """
        super().__init__(str(original_error))
        self.original_error: httpx.HTTPError = original_error
        self.client_latency_ms: int = client_latency_ms


def _load_sample_questions(questions_file: Path) -> list[SampleQuestion]:
    """Load and validate the question list from a JSON file.

    Args:
        questions_file: JSON file containing an array of dated questions.

    Returns:
        list[SampleQuestion]: Validated questions in file order.

    Raises:
        ValueError: If the file cannot be read, is invalid, or is empty.
    """
    try:
        questions: list[SampleQuestion] = TypeAdapter(
            list[SampleQuestion]
        ).validate_json(questions_file.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as error:
        raise ValueError(
            f"Unable to load valid questions from {questions_file}: {error}"
        ) from error
    if not questions:
        raise ValueError(f"Question file contains no questions: {questions_file}")
    return questions


class QuestionRun(BaseModel):
    """Capture one sequential request and its response diagnostics.

    Attributes:
        index: One-based question number.
        asked_at_local: Original Vietnam-local timestamp from the sample.
        asked_at: ISO 8601 timestamp sent to the API.
        question: Vietnamese customer question.
        history: Prior conversation turns supplied with the request.
        client_latency_ms: Cumulative HTTP attempt duration excluding retry waits.
        http_status: HTTP response status, if a response was received.
        server_latency_ms: API-reported total request duration.
        llm_latency_ms: Sum of model step durations reported by the API.
        prompt_tokens: Sum of model-reported prompt tokens, when available.
        completion_tokens: Sum of model-reported completion tokens, when
            available.
        answer: Final answer returned by the assistant.
        citations: References returned by the assistant.
        steps: Per-LLM/tool durations and token metrics.
        tool_executions: Tool calls and their structured results.
        response_body: Complete decoded API response.
        error: Client-side error if the response could not be fetched.
    """

    index: int = Field(ge=1)
    asked_at_local: str
    asked_at: str
    question: str
    history: list[ConversationMessage] = Field(default_factory=list)
    client_latency_ms: int | None = Field(default=None, ge=0)
    http_status: int | None = None
    server_latency_ms: int | None = Field(default=None, ge=0)
    llm_latency_ms: int | None = Field(default=None, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    answer: str | None = None
    citations: list[str] = Field(default_factory=list)
    steps: list[dict[str, Any]] = Field(default_factory=list)
    tool_executions: list[dict[str, Any]] = Field(default_factory=list)
    response_body: dict[str, Any] | None = None
    error: str | None = None


class RunReport(BaseModel):
    """Store metadata and all completed question requests.

    Attributes:
        base_url: API address used by the runner.
        output_file: Destination path for the JSON report.
        started_at: UTC time when this run began.
        completed_at: UTC time when all requests finished.
        results: Sequential question results; flushed after each request.
    """

    base_url: str
    output_file: str
    started_at: datetime
    completed_at: datetime | None = None
    results: list[QuestionRun] = Field(default_factory=list)


def _utc_now() -> datetime:
    """Return the current UTC time as a timezone-aware value.

    Returns:
        datetime: Current timestamp in UTC.
    """
    return datetime.now(timezone.utc)


def _asked_at_iso(local_timestamp: str) -> str:
    """Convert the sample's Vietnam-local timestamp to ISO 8601.

    Args:
        local_timestamp: Date and time in ``dd/mm/yyyy HH:MM`` format.

    Returns:
        str: ISO 8601 timestamp with the Vietnam UTC offset.

    Raises:
        ValueError: If the sample timestamp does not match the expected format.
    """
    parsed: datetime = datetime.strptime(local_timestamp, "%d/%m/%Y %H:%M")
    return parsed.replace(tzinfo=VIETNAM_TIMEZONE).isoformat(timespec="seconds")


def _question_payload(sample: SampleQuestion) -> dict[str, object]:
    """Build the API payload for one dated sample question.

    Args:
        sample: Validated sample question and optional conversation history.

    Returns:
        dict[str, object]: JSON-compatible question, timestamp, and history.
    """
    return {
        "question": sample.question,
        "asked_at": _asked_at_iso(sample.asked_at_local),
        "history": [
            message.model_dump(mode="json") for message in sample.history
        ],
    }


def _token_total(steps: list[dict[str, Any]], field: str) -> int | None:
    """Sum a reported token field over LLM steps.

    Args:
        steps: Per-operation metrics returned by the API.
        field: Token key to sum, either prompt or completion tokens.

    Returns:
        int | None: Sum when at least one step reports tokens; otherwise None.
    """
    values: list[int] = [
        step[field]
        for step in steps
        if step.get("kind") == "llm" and isinstance(step.get(field), int)
    ]
    return sum(values) if values else None


def _write_report(report: RunReport, output_file: Path) -> None:
    """Atomically persist current results so progress survives interruption.

    Args:
        report: Current run state.
        output_file: JSON report destination.

    Raises:
        OSError: If the destination cannot be written.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_file.parent,
        prefix=f".{output_file.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path: Path = Path(temporary_file.name)
        temporary_file.write(report.model_dump_json(indent=2))
        temporary_file.write("\n")
    temporary_path.replace(output_file)


def _wait_for_api(client: httpx.Client, base_url: str, timeout: float) -> None:
    """Wait for the API health endpoint before making model requests.

    Args:
        client: HTTP client with a short health-check timeout.
        base_url: API base address without a trailing slash.
        timeout: Maximum seconds to wait for service readiness.

    Raises:
        SystemExit: If the API does not become healthy before the timeout.
    """
    deadline: float = time.monotonic() + timeout
    last_error: str = "Health check did not return HTTP 200."
    while time.monotonic() < deadline:
        try:
            response: httpx.Response = client.get(
                f"{base_url}/health",
                timeout=HEALTH_TIMEOUT_SECONDS,
            )
            if response.is_success:
                print(f"API ready: {base_url} ({response.text})", flush=True)
                return
            last_error = f"HTTP {response.status_code}: {response.text}"
        except httpx.HTTPError as error:
            last_error = str(error)
        print("Waiting for API health check...", flush=True)
        time.sleep(STARTUP_POLL_SECONDS)
    raise SystemExit(f"API did not become ready within {timeout:g}s: {last_error}")


def _question_retry_delay(response: httpx.Response) -> float:
    """Choose a retry delay that honors server guidance without retrying early.

    Args:
        response: Transient response returned by the API.

    Returns:
        float: Delay in seconds before the next attempt.
    """
    retry_after: str | None = response.headers.get("Retry-After")
    if retry_after is None:
        return QUESTION_RETRY_DELAY_SECONDS

    try:
        requested_delay: float = float(retry_after)
    except ValueError:
        try:
            retry_at: datetime = parsedate_to_datetime(retry_after)
        except (TypeError, ValueError, OverflowError):
            return QUESTION_RETRY_DELAY_SECONDS
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        requested_delay = (retry_at - _utc_now()).total_seconds()

    if not math.isfinite(requested_delay):
        return QUESTION_RETRY_DELAY_SECONDS
    return max(QUESTION_RETRY_DELAY_SECONDS, requested_delay)


def _post_question_with_retries(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
) -> tuple[httpx.Response, int]:
    """Send one question and retry transient failures with bounded backoff.

    The returned duration sums only time spent inside HTTP attempts; retry
    delays are deliberately excluded.

    Args:
        client: HTTP client used to send the API request.
        url: Complete question endpoint URL.
        payload: JSON request payload.

    Returns:
        tuple[httpx.Response, int]: Response and cumulative HTTP attempt
            duration in milliseconds.

    Raises:
        QuestionRequestError: If every attempt fails before a response.
    """
    request_duration_seconds: float = 0.0
    for attempt in range(MAX_QUESTION_RETRIES + 1):
        request_started: float = time.perf_counter()
        try:
            response: httpx.Response = client.post(url, json=payload)
        except httpx.TransportError as error:
            request_duration_seconds += time.perf_counter() - request_started
            if attempt == MAX_QUESTION_RETRIES:
                client_latency_ms: int = round(request_duration_seconds * 1000)
                raise QuestionRequestError(error, client_latency_ms) from error
            delay_seconds: float = QUESTION_RETRY_DELAY_SECONDS
            print(
                f"Request transport error ({error}); retrying "
                f"{attempt + 1}/{MAX_QUESTION_RETRIES} in "
                f"{delay_seconds:.1f}s.",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay_seconds)
            continue
        request_duration_seconds += time.perf_counter() - request_started

        if (
            response.status_code not in RETRIABLE_HTTP_STATUS_CODES
            or attempt == MAX_QUESTION_RETRIES
        ):
            client_latency_ms = round(request_duration_seconds * 1000)
            return response, client_latency_ms

        delay_seconds = _question_retry_delay(response)
        print(
            f"HTTP {response.status_code}; retrying "
            f"{attempt + 1}/{MAX_QUESTION_RETRIES} in {delay_seconds:.1f}s.",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(delay_seconds)

    raise RuntimeError("Question request retry loop ended unexpectedly.")


def _parse_response(
    index: int,
    asked_at_local: str,
    question: str,
    response: httpx.Response,
    client_latency_ms: int,
    history: list[ConversationMessage],
) -> QuestionRun:
    """Convert one HTTP response into a report entry.

    Args:
        index: One-based question number.
        asked_at_local: Original Vietnam-local timestamp.
        question: Question sent to the API.
        response: API response.
        client_latency_ms: Cumulative HTTP attempt duration excluding retry waits.
        history: Prior conversation turns supplied with the request.

    Returns:
        QuestionRun: Response and timing detail for this question.
    """
    try:
        decoded: Any = response.json()
    except ValueError:
        decoded = {"body": response.text}
    body: dict[str, Any] = (
        decoded if isinstance(decoded, dict) else {"body": decoded}
    )
    steps_value: Any = body.get("steps", [])
    steps: list[dict[str, Any]] = (
        [step for step in steps_value if isinstance(step, dict)]
        if isinstance(steps_value, list)
        else []
    )
    executions_value: Any = body.get("tool_executions", [])
    executions: list[dict[str, Any]] = (
        [execution for execution in executions_value if isinstance(execution, dict)]
        if isinstance(executions_value, list)
        else []
    )
    citations_value: Any = body.get("citations", [])
    citations: list[str] = (
        [citation for citation in citations_value if isinstance(citation, str)]
        if isinstance(citations_value, list)
        else []
    )
    llm_durations: list[int] = [
        step["duration_ms"]
        for step in steps
        if step.get("kind") == "llm"
        and isinstance(step.get("duration_ms"), int)
    ]
    return QuestionRun(
        index=index,
        asked_at_local=asked_at_local,
        asked_at=_asked_at_iso(asked_at_local),
        question=question,
        history=history,
        client_latency_ms=client_latency_ms,
        http_status=response.status_code,
        server_latency_ms=(
            body.get("latency_ms")
            if isinstance(body.get("latency_ms"), int)
            else None
        ),
        llm_latency_ms=sum(llm_durations) if llm_durations else None,
        prompt_tokens=_token_total(steps, "prompt_tokens"),
        completion_tokens=_token_total(steps, "completion_tokens"),
        answer=body.get("answer") if isinstance(body.get("answer"), str) else None,
        citations=citations,
        steps=steps,
        tool_executions=executions,
        response_body=body,
    )


def run_questions(
    base_url: str,
    output_file: Path,
    request_timeout: float,
    startup_timeout: float,
    questions_file: Path = DEFAULT_QUESTIONS_FILE,
) -> RunReport:
    """Run sample requests sequentially with retries and save progressive results.

    Args:
        base_url: API base URL reachable from the host running this script.
        output_file: JSON path for the complete and partial report.
        request_timeout: Per-question HTTP read timeout in seconds.
        startup_timeout: Maximum health-check wait in seconds.
        questions_file: JSON file containing questions to run.

    Returns:
        RunReport: Completed results in original question order.

    Raises:
        OSError: If the output report cannot be persisted.
        ValueError: If the question file is empty or invalid.
        SystemExit: If the API is unhealthy when startup timeout expires.
    """
    questions: list[SampleQuestion] = _load_sample_questions(questions_file)
    normalized_base_url: str = base_url.rstrip("/")
    report: RunReport = RunReport(
        base_url=normalized_base_url,
        output_file=str(output_file),
        started_at=_utc_now(),
    )
    _write_report(report, output_file)
    with httpx.Client(
        timeout=httpx.Timeout(request_timeout, connect=15.0),
    ) as client:
        _wait_for_api(client, normalized_base_url, startup_timeout)
        for index, sample in enumerate(questions, start=1):
            asked_at_local: str = sample.asked_at_local
            question: str = sample.question
            asked_at: str = _asked_at_iso(asked_at_local)
            print(
                f"[{index}/{len(questions)}] Sending {asked_at_local}: "
                f"{question}",
                flush=True,
            )
            try:
                response, client_latency_ms = _post_question_with_retries(
                    client=client,
                    url=f"{normalized_base_url}/ask",
                    payload=_question_payload(sample),
                )
                result: QuestionRun = _parse_response(
                    index=index,
                    asked_at_local=asked_at_local,
                    question=question,
                    response=response,
                    client_latency_ms=client_latency_ms,
                    history=sample.history,
                )
            except QuestionRequestError as error:
                result = QuestionRun(
                    index=index,
                    asked_at_local=asked_at_local,
                    asked_at=asked_at,
                    question=question,
                    history=sample.history,
                    client_latency_ms=error.client_latency_ms,
                    error=str(error.original_error),
                )
                report.results.append(result)
                _write_report(report, output_file)
                print(f"  Request failed: {error}", file=sys.stderr, flush=True)
                break

            report.results.append(result)
            _write_report(report, output_file)
            prompt_tokens: str = (
                str(result.prompt_tokens)
                if result.prompt_tokens is not None
                else "n/a"
            )
            completion_tokens: str = (
                str(result.completion_tokens)
                if result.completion_tokens is not None
                else "n/a"
            )
            print(
                f"  HTTP {result.http_status}; client={result.client_latency_ms}ms; "
                f"server={result.server_latency_ms}ms; llm={result.llm_latency_ms}ms; "
                f"tokens={prompt_tokens}/{completion_tokens}; "
                f"tools={len(result.tool_executions)}",
                flush=True,
            )
            step_summary: str = "; ".join(
                f"{step.get('kind')}:{step.get('name')}="
                f"{step.get('duration_ms')}ms"
                for step in result.steps
            )
            if step_summary:
                print(f"  Stage timings: {step_summary}", flush=True)
            if result.answer:
                print(f"  Answer: {result.answer}", flush=True)

    report.completed_at = _utc_now()
    _write_report(report, output_file)
    print(
        f"Finished {len(report.results)}/{len(questions)} requests. "
        f"Report: {output_file}",
        flush=True,
    )
    return report


def main() -> None:
    """Parse command-line options and run the sample request sequence.

    Raises:
        SystemExit: If arguments are invalid or the API does not become ready.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(
            "Run dated questions loaded from a JSON file sequentially against "
            "the host-published Lumen API. Retry transient failures up to "
            f"{MAX_QUESTION_RETRIES} times, honoring server retry hints, then "
            "save full responses and timings."
        )
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS_FILE,
        help=(
            "JSON file containing the question list "
            f"(default: {DEFAULT_QUESTIONS_FILE})."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LUMEN_API_URL", DEFAULT_BASE_URL),
        help="Host API URL (default: LUMEN_API_URL or http://127.0.0.1:5000).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Progressively updated JSON output (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=600.0,
        help="Timeout per request in seconds (default: 600).",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=300.0,
        help="Maximum wait for a healthy API in seconds (default: 300).",
    )
    arguments: argparse.Namespace = parser.parse_args()
    run_questions(
        base_url=arguments.base_url,
        output_file=arguments.output,
        request_timeout=arguments.request_timeout,
        startup_timeout=arguments.startup_timeout,
        questions_file=arguments.questions,
    )


if __name__ == "__main__":
    main()