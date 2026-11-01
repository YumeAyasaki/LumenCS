"""FastAPI application for the Lumen Home customer-support assistant."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from time import perf_counter
from typing import Annotated, Literal
from zoneinfo import ZoneInfo

import pymupdf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastembed import TextEmbedding
from httpx import HTTPError
from pydantic import (
    BaseModel,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
)
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http.exceptions import ApiException, ResponseHandlingException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import Settings
from app.llm_agent import (
    AgentResult,
    AgentStepMetrics,
    AgentToolExecution,
    LLMConfigurationError,
    LLMProviderError,
    answer_question,
)
from app.tools.ingest import IngestAllResult, ingest_all_sources
from app.tools.model_tools import (
    ModelToolContext,
    ModelToolDefinition,
    ToolInvocationCall,
    ToolInvocationRequest,
    ToolInvocationResponse,
    ToolResult,
    UnknownModelToolError,
    execute_model_tool,
    get_model_tool_definitions,
)

logger: logging.Logger = logging.getLogger(__name__)
VIETNAM_TIMEZONE: ZoneInfo = ZoneInfo("Asia/Ho_Chi_Minh")

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Load embeddings and create/dispose database clients for server lifetime.

    Args:
        application: FastAPI application receiving the client handles.

    Yields:
        None: The application runs while both configured clients are available.

    Raises:
        ValueError: If required connection settings or model configuration are invalid.
    """
    settings: Settings = Settings.from_environment()
    embedding_model: TextEmbedding = await asyncio.to_thread(
        TextEmbedding,
        model_name=settings.pdf_embedding_model,
        cache_dir=os.getenv("FASTEMBED_CACHE_PATH"),
    )
    postgres_engine: AsyncEngine = create_async_engine(
        settings.database_url,
        pool_pre_ping=True,
    )
    qdrant_client: AsyncQdrantClient = AsyncQdrantClient(url=settings.qdrant_url)
    qdrant_sync_client: QdrantClient = QdrantClient(url=settings.qdrant_url)
    application.state.postgres_engine = postgres_engine
    application.state.postgres_session_factory = async_sessionmaker(
        postgres_engine,
        expire_on_commit=False,
    )
    application.state.qdrant_client = qdrant_client
    application.state.qdrant_sync_client = qdrant_sync_client
    application.state.pdf_embedding_model = embedding_model
    application.state.settings = settings
    logger.info(
        "Initialized PostgreSQL, Qdrant, and embedding model %s",
        settings.pdf_embedding_model,
    )
    try:
        yield
    finally:
        try:
            await qdrant_client.close()
        finally:
            try:
                qdrant_sync_client.close()
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

    @field_validator("asked_at")
    @classmethod
    def convert_asked_at_to_vietnam_time(cls, value: datetime) -> datetime:
        """Normalize timestamps once at the API boundary.

        Args:
            value: User-provided ISO timestamp, which may be naive.

        Returns:
            Timestamp represented in ``Asia/Ho_Chi_Minh``.

        Raises:
            ValueError: If timezone conversion cannot be performed.
        """
        if value.tzinfo is None:
            return value.replace(tzinfo=VIETNAM_TIMEZONE)
        return value.astimezone(VIETNAM_TIMEZONE)


class AskResponse(BaseModel):
    """Return the answer, source identifiers, and request latency.

    Attributes:
        answer: Vietnamese response text.
        citations: Source identifiers supporting the answer.
        latency_ms: Time spent handling the request, in milliseconds.
        steps: Per-operation timing and provider-reported token counts.
        tool_executions: Tool calls and their structured answers.

    Raises:
        ValidationError: If any response field is invalid.
    """

    answer: str
    citations: list[str]
    latency_ms: int = Field(ge=0)
    steps: list[AgentStepMetrics] = Field(default_factory=list)
    tool_executions: list[AgentToolExecution] = Field(default_factory=list)


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
    description="Retrieval-backed API for the Lumen Home customer-support assistant.",
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


@app.post(
    "/ask",
    response_model=AskResponse,
    response_model_exclude_none=True,
)
async def ask(
    body: AskRequest,
    request: Request,
) -> AskResponse | StreamingResponse:
    """Answer a customer question using typed tools over approved source data.

    Args:
        body: Question, timestamp, and optional conversation history.
        request: Application state containing provider and data-store clients.

    Returns:
        AskResponse | StreamingResponse: Vietnamese answer with citations
            returned by this run's tools. Clients accepting
            ``text/event-stream`` receive validated answer chunks followed by
            citation and completion events.

    Raises:
        HTTPException: If the LLM configuration/provider or a source store fails.
    """
    started_at: float = perf_counter()
    settings: Settings = request.app.state.settings
    qdrant_client: AsyncQdrantClient = request.app.state.qdrant_client
    embedding_model: TextEmbedding = request.app.state.pdf_embedding_model
    session_factory: async_sessionmaker[AsyncSession] = (
        request.app.state.postgres_session_factory
    )
    try:
        agent_result: AgentResult = await answer_question(
            question=body.question,
            asked_at=body.asked_at.isoformat(),
            history=[message.model_dump() for message in body.history],
            session_factory=session_factory,
            qdrant_client=qdrant_client,
            embedding_model=embedding_model,
            settings=settings,
        )
    except LLMConfigurationError as error:
        logger.error("LLM configuration error: %s", error)
        raise HTTPException(
            status_code=503,
            detail="The LLM provider is not configured.",
        ) from error
    except LLMProviderError as error:
        logger.error("LLM answer generation failed: %s", error)
        raise HTTPException(
            status_code=502,
            detail=str(error),
        ) from error
    except (
        ApiException,
        HTTPError,
        ResponseHandlingException,
        SQLAlchemyError,
    ) as error:
        logger.exception("A source lookup failed while answering the question")
        raise HTTPException(
            status_code=503,
            detail="A source data service is temporarily unavailable.",
        ) from error

    latency_ms: int = round((perf_counter() - started_at) * 1000)
    logger.info(
        "Ask request completed: provider=%s llm_latency_ms=%d total_latency_ms=%d",
        settings.llm_provider,
        agent_result.llm_latency_ms,
        latency_ms,
    )
    ask_response: AskResponse = AskResponse(
        answer=agent_result.answer,
        citations=agent_result.citations,
        latency_ms=latency_ms,
        steps=agent_result.steps,
        tool_executions=agent_result.tool_executions,
    )
    if "text/event-stream" in request.headers.get("accept", "").lower():
        async def event_stream() -> AsyncIterator[str]:
            """Yield tool calls, answer text, citations, metrics, and completion.

            Yields:
                str: Server-sent events containing tool calls and answers,
                    answer chunks, citations, per-step metrics, and request
                    timing.
            """
            chunk_size: int = 32
            tool_event: str = json.dumps(
                {
                    "tool_executions": [
                        execution.model_dump(mode="json", exclude_none=True)
                        for execution in ask_response.tool_executions
                    ],
                },
                ensure_ascii=False,
            )
            yield f"event: tool_calls\ndata: {tool_event}\n\n"
            for start in range(0, len(ask_response.answer), chunk_size):
                chunk: str = ask_response.answer[start : start + chunk_size]
                event_data: str = json.dumps(
                    {"text": chunk},
                    ensure_ascii=False,
                )
                yield f"event: answer\ndata: {event_data}\n\n"
            citations_event: str = json.dumps(
                ask_response.citations,
                ensure_ascii=False,
            )
            yield f"event: citations\ndata: {citations_event}\n\n"
            metrics_event: str = json.dumps(
                {
                    "steps": [
                        step.model_dump(mode="json", exclude_none=True)
                        for step in ask_response.steps
                    ],
                },
                ensure_ascii=False,
            )
            yield f"event: metrics\ndata: {metrics_event}\n\n"
            completion_event: str = json.dumps(
                {"latency_ms": ask_response.latency_ms},
                ensure_ascii=False,
            )
            yield f"event: done\ndata: {completion_event}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
        )
    return ask_response


@app.get("/tools", response_model=list[ModelToolDefinition])
async def list_model_tools() -> list[ModelToolDefinition]:
    """List typed function declarations that an LLM or agent can call.

    Returns:
        list[ModelToolDefinition]: Model-ready names, descriptions, and schemas.
    """
    return get_model_tool_definitions()


@app.post("/tools/{tool_name}", response_model=ToolInvocationResponse)
async def invoke_model_tool(
    tool_name: str,
    body: ToolInvocationRequest,
    request: Request,
) -> ToolInvocationResponse:
    """Execute one validated data or utility tool for an agent.

    Args:
        tool_name: Registered tool name selected by the model.
        body: JSON arguments validated against the selected tool's Pydantic
            input schema.
        request: Application state containing database and policy clients.

    Returns:
        ToolInvocationResponse: Call arguments and typed answer records.

    Raises:
        HTTPException: With status 404 for an unknown tool, 422 for invalid
            arguments, or 503 if a backing data store is unavailable.
    """
    session_factory: async_sessionmaker[AsyncSession] = (
        request.app.state.postgres_session_factory
    )
    arguments: dict[str, object] = dict(body.arguments)
    if "as_of" in arguments:
        logger.info(
            "Ignoring caller-supplied as_of for direct tool %s",
            tool_name,
        )
    arguments["as_of"] = datetime.now(VIETNAM_TIMEZONE).isoformat()
    try:
        async with session_factory() as session:
            context = ModelToolContext(
                session=session,
                qdrant_client=request.app.state.qdrant_client,
                embedding_model=request.app.state.pdf_embedding_model,
                settings=request.app.state.settings,
            )
            result: ToolResult = await execute_model_tool(
                tool_name,
                arguments,
                context,
            )
    except UnknownModelToolError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=error.errors()) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except (
        ApiException,
        HTTPError,
        ResponseHandlingException,
        SQLAlchemyError,
    ) as error:
        logger.exception("Model tool %s failed", tool_name)
        raise HTTPException(
            status_code=503,
            detail=f"Tool '{tool_name}' is temporarily unavailable.",
        ) from error

    logger.info(
        "Model tool %s returned %d records and %d citations",
        tool_name,
        len(result.records),
        len(result.citations),
    )
    return ToolInvocationResponse(
        call=ToolInvocationCall(name=tool_name, arguments=arguments),
        answer=result,
    )


@app.post("/ingest", response_model=IngestAllResult)
async def ingest_sources(request: Request) -> IngestAllResult:
    """Import all Excel workbooks to PostgreSQL and PDFs to Qdrant.

    The source directory, PDF collection, and embedding model are configured by
    ``DATA_DIR``, ``QDRANT_COLLECTION``, and ``PDF_EMBEDDING_MODEL``.

    Args:
        request: Incoming request containing the application database clients.

    Returns:
        IngestAllResult: Counts and timings for both storage backends.

    Raises:
        HTTPException: If source parsing or either backing-store operation fails.
    """
    session_factory: async_sessionmaker[AsyncSession] = (
        request.app.state.postgres_session_factory
    )
    qdrant_client: QdrantClient = request.app.state.qdrant_sync_client
    try:
        settings: Settings = Settings.from_environment()
        return await ingest_all_sources(
            session_factory=session_factory,
            qdrant_client=qdrant_client,
            data_dir=settings.data_dir,
            qdrant_collection=settings.qdrant_collection,
            pdf_embedding_model=settings.pdf_embedding_model,
        )
    except (
        ApiException,
        FileNotFoundError,
        HTTPError,
        OSError,
        pymupdf.FileDataError,
        ResponseHandlingException,
        SQLAlchemyError,
        ValueError,
    ) as error:
        logger.exception("Source ingestion endpoint failed")
        status_code: int = (
            503
            if isinstance(
                error,
                (
                    ApiException,
                    HTTPError,
                    ResponseHandlingException,
                    SQLAlchemyError,
                ),
            )
            else 422
        )
        raise HTTPException(
            status_code=status_code,
            detail="Source ingestion failed; see server logs for details.",
        ) from error
