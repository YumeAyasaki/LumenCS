"""FastAPI application for the Lumen Home customer-support assistant."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from time import perf_counter
from typing import Annotated, AsyncIterator, Literal

from fastembed import TextEmbedding
import pymupdf
from fastapi import FastAPI, HTTPException, Request
from httpx import HTTPError
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http.exceptions import ApiException, ResponseHandlingException
from pydantic import BaseModel, Field, StringConstraints
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import Settings
from app.tools.ingest import IngestAllResult, ingest_all_sources
from app.tools.pdf_retriever import PDFSearchResult, retrieve_policy_chunks

logger: logging.Logger = logging.getLogger(__name__)

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


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request) -> AskResponse:
    """Return source-grounded policy excerpts retrieved for a customer question.

    Args:
        body: Question, timestamp, and optional conversation history.
        request: Application state containing the embedding and Qdrant clients.

    Returns:
        AskResponse: Up to three relevant Vietnamese policy excerpts and citations.

    Raises:
        HTTPException: If Qdrant retrieval fails, with HTTP 503.
    """
    started_at: float = perf_counter()
    settings: Settings = request.app.state.settings
    qdrant_client: AsyncQdrantClient = request.app.state.qdrant_client
    embedding_model: TextEmbedding = request.app.state.pdf_embedding_model
    try:
        results: list[PDFSearchResult] = await retrieve_policy_chunks(
            qdrant_client=qdrant_client,
            embedding_model=embedding_model,
            question=body.question,
            as_of=body.asked_at,
            base_collection_name=settings.qdrant_collection,
            embedding_model_name=settings.pdf_embedding_model,
        )
    except (ApiException, HTTPError, ResponseHandlingException) as error:
        logger.exception("Policy retrieval failed")
        raise HTTPException(
            status_code=503,
            detail="Policy retrieval is temporarily unavailable.",
        ) from error

    citations: list[str] = list(dict.fromkeys(result.cite_id for result in results))
    if results:
        excerpts: list[str] = [
            f"{result.cite_id}:\n{result.text}"
            for result in results
        ]
        answer: str = "Các nội dung liên quan trong tài liệu:\n\n" + "\n\n".join(
            excerpts
        )
    else:
        answer = "Tôi chưa tìm thấy thông tin phù hợp trong sổ tay chính sách."
    latency_ms: int = round((perf_counter() - started_at) * 1000)
    logger.info(
        "Policy retrieval completed: %d results in %d ms",
        len(results),
        latency_ms,
    )
    return AskResponse(
        answer=answer,
        citations=citations,
        latency_ms=latency_ms,
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
