"""FastAPI application for the Lumen Home customer-support assistant."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from time import perf_counter
from typing import Annotated, AsyncIterator, Literal

from fastapi import FastAPI, HTTPException, Request
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ApiException, ResponseHandlingException
from pydantic import BaseModel, Field, StringConstraints
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.config import Settings

logger: logging.Logger = logging.getLogger(__name__)

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Create and dispose PostgreSQL and Qdrant clients for the server lifetime.

    Args:
        application: FastAPI application receiving the client handles.

    Yields:
        None: The application runs while both configured clients are available.

    Raises:
        ValueError: If required connection settings are empty.
    """
    settings: Settings = Settings.from_environment()
    postgres_engine: AsyncEngine = create_async_engine(
        settings.database_url,
        pool_pre_ping=True,
    )
    qdrant_client: AsyncQdrantClient = AsyncQdrantClient(url=settings.qdrant_url)
    application.state.postgres_engine = postgres_engine
    application.state.qdrant_client = qdrant_client
    logger.info("Initialized PostgreSQL and Qdrant clients")
    try:
        yield
    finally:
        try:
            await qdrant_client.close()
        finally:
            await postgres_engine.dispose()
        logger.info("Closed PostgreSQL and Qdrant clients")


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
    """Report API and backing-store readiness.

    Attributes:
        status: Overall health status string.
        postgres: Whether PostgreSQL responded to a connection check.
        qdrant: Whether Qdrant responded to a collections check.

    Raises:
        ValidationError: If the status is not a supported health state.
    """

    status: Literal["ok", "unavailable"]
    postgres: bool
    qdrant: bool


app: FastAPI = FastAPI(
    title="Lumen Home Assistant API",
    description="Placeholder API for the Lumen Home customer-support assistant.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Check that the API can reach PostgreSQL and Qdrant.

    Args:
        request: Incoming request containing the application database clients.

    Returns:
        HealthResponse: Readiness status for both backing stores.

    Raises:
        HTTPException: If either database health check fails, with HTTP 503.
    """
    postgres_engine: AsyncEngine = request.app.state.postgres_engine
    qdrant_client: AsyncQdrantClient = request.app.state.qdrant_client
    postgres_available: bool = True
    qdrant_available: bool = True

    try:
        async with postgres_engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
    except SQLAlchemyError:
        postgres_available = False
        logger.exception("PostgreSQL health check failed")

    try:
        await qdrant_client.get_collections()
    except (ApiException, ResponseHandlingException):
        qdrant_available = False
        logger.exception("Qdrant health check failed")

    response: HealthResponse = HealthResponse(
        status=(
            "ok"
            if postgres_available and qdrant_available
            else "unavailable"
        ),
        postgres=postgres_available,
        qdrant=qdrant_available,
    )
    if not postgres_available or not qdrant_available:
        raise HTTPException(
            status_code=503,
            detail=response.model_dump(),
        )
    return response


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
