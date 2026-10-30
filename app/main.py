"""FastAPI application for the Lumen Home customer-support assistant."""

from __future__ import annotations

import logging
from datetime import datetime
from time import perf_counter
from typing import Annotated, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field, StringConstraints

logger: logging.Logger = logging.getLogger(__name__)

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ConversationMessage(BaseModel):
    """Represent one message in the conversation history.

    Attributes:
        role: Whether the message came from the user or assistant.
        content: Text content of the conversation message.

    Raises:
        ValidationError: If a role or content value is invalid.
    """

    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    """Validate a question sent to the assistant.

    Attributes:
        question: Non-empty question text.
        asked_at: Timestamp used to evaluate time-sensitive source material.
        history: Optional prior conversation messages.

    Raises:
        ValidationError: If any request field is invalid.
    """

    question: NonEmptyString
    asked_at: datetime
    history: list[ConversationMessage] = Field(default_factory=list)


class AskResponse(BaseModel):
    """Return the answer, source identifiers, and request latency.

    Attributes:
        answer: Vietnamese response text.
        citations: Source identifiers supporting the answer.
        latency_ms: Time spent handling the request, in milliseconds.

    Raises:
        ValidationError: If any response field is invalid.
    """

    answer: str
    citations: list[str]
    latency_ms: int = Field(ge=0)


class HealthResponse(BaseModel):
    """Report that the API process is responsive.

    Attributes:
        status: Health status string.

    Raises:
        ValidationError: If the status is not ``ok``.
    """

    status: Literal["ok"]


app: FastAPI = FastAPI(
    title="Lumen Home Assistant API",
    description="Placeholder API for the Lumen Home customer-support assistant.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return the process health status.

    Returns:
        HealthResponse: The health status with value ``ok``.

    Raises:
        None.
    """
    return HealthResponse(status="ok")


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    """Return a placeholder answer to a customer-support question.

    Args:
        request (AskRequest): Question, timestamp, and optional conversation
            history.

    Returns:
        AskResponse: A placeholder response with no citations until answer
            retrieval is implemented.

    Raises:
        RequestValidationError: FastAPI returns HTTP 422 if the request body
            fails model validation.
    """
    started_at: float = perf_counter()
    logger.info("Received assistant question")

    return AskResponse(
        answer="Tính năng trả lời tự động đang được phát triển. Vui lòng thử lại sau.",
        citations=[],
        latency_ms=round((perf_counter() - started_at) * 1000),
    )
