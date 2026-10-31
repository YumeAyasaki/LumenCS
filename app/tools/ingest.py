"""Run spreadsheet-to-PostgreSQL and PDF-to-Qdrant ingestion together."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from time import perf_counter
from typing import Sequence

from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ApiException, ResponseHandlingException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import Settings
from app.tools.pdf_ingester import PDFIngestionResult, ingest_pdf_directory
from app.tools.spreadsheet_ingester import (
    IngestionResult,
    ingest_spreadsheet_directory,
)

logger: logging.Logger = logging.getLogger(__name__)


class IngestAllResult(BaseModel):
    """Summarize ingestion of both source types.

    Attributes:
        spreadsheets: PostgreSQL workbook-ingestion counts.
        pdfs: Qdrant PDF-ingestion counts and extraction warnings.
        spreadsheet_latency_ms: Time spent reading and storing spreadsheets.
        pdf_latency_ms: Time spent reading, embedding, and storing PDFs.
        latency_ms: Total time spent in the combined operation.
    """

    spreadsheets: IngestionResult
    pdfs: PDFIngestionResult
    spreadsheet_latency_ms: int
    pdf_latency_ms: int
    latency_ms: int


async def ingest_all_sources(
    session_factory: async_sessionmaker[AsyncSession],
    qdrant_client: QdrantClient,
    data_dir: Path,
    qdrant_collection: str = "policy_documents",
    pdf_embedding_model: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ),
) -> IngestAllResult:
    """Ingest spreadsheets into PostgreSQL and PDFs into Qdrant.

    The spreadsheet transaction completes before PDF indexing starts. If PDF
    indexing fails, the successful spreadsheet result remains committed.

    Args:
        session_factory: Factory for asynchronous PostgreSQL sessions.
        qdrant_client: Connected Qdrant synchronous client, called in a worker thread.
        data_dir: Directory containing mapped workbooks and policy PDFs.
        qdrant_collection: Dedicated collection for indexed PDF chunks.
        pdf_embedding_model: Multilingual FastEmbed model for Qdrant vectors.

    Returns:
        IngestAllResult: Counts and warnings for both ingesters.

    Raises:
        FileNotFoundError: If the directory has no workbooks or no PDFs.
        OSError: If a source file cannot be read.
        ValueError: If source parsing fails or the embedding model is invalid.
        SQLAlchemyError: If the PostgreSQL transaction fails.
        ApiException: If Qdrant rejects an operation.
        ResponseHandlingException: If Qdrant cannot be reached.
    """
    total_started_at: float = perf_counter()
    spreadsheet_started_at: float = perf_counter()
    spreadsheet_result: IngestionResult = await ingest_spreadsheet_directory(
        session_factory,
        data_dir,
    )
    spreadsheet_latency_ms: int = round(
        (perf_counter() - spreadsheet_started_at) * 1000
    )
    pdf_started_at: float = perf_counter()
    pdf_result: PDFIngestionResult = await asyncio.to_thread(
        ingest_pdf_directory,
        qdrant_client,
        data_dir,
        qdrant_collection,
        pdf_embedding_model,
    )
    pdf_latency_ms: int = round((perf_counter() - pdf_started_at) * 1000)
    result: IngestAllResult = IngestAllResult(
        spreadsheets=spreadsheet_result,
        pdfs=pdf_result,
        spreadsheet_latency_ms=spreadsheet_latency_ms,
        pdf_latency_ms=pdf_latency_ms,
        latency_ms=round((perf_counter() - total_started_at) * 1000),
    )
    logger.info("All source ingestion completed")
    return result


async def _run_cli_ingestion(data_dir: Path) -> IngestAllResult:
    """Create database clients and run ingestion for CLI use.

    Args:
        data_dir: Directory containing source workbooks and PDFs.

    Returns:
        IngestAllResult: Counts and warnings for both ingestion stages.

    Raises:
        OSError: If a source file cannot be read.
        ValueError: If configuration or source data is invalid.
        SQLAlchemyError: If PostgreSQL cannot persist spreadsheet rows.
        ApiException: If Qdrant rejects an operation.
        ResponseHandlingException: If Qdrant cannot be reached.
    """
    settings: Settings = Settings.from_environment()
    engine: AsyncEngine = create_async_engine(
        settings.database_url,
        pool_pre_ping=True,
    )
    qdrant_client: QdrantClient = QdrantClient(url=settings.qdrant_url)
    session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        engine,
        expire_on_commit=False,
    )
    try:
        return await ingest_all_sources(
            session_factory=session_factory,
            qdrant_client=qdrant_client,
            data_dir=data_dir,
            qdrant_collection=settings.qdrant_collection,
            pdf_embedding_model=settings.pdf_embedding_model,
        )
    finally:
        try:
            qdrant_client.close()
        finally:
            await engine.dispose()


def main(argv: Sequence[str] | None = None) -> None:
    """Run both configured source ingesters from the command line.

    Args:
        argv: Optional command-line argument sequence for programmatic use.

    Returns:
        None: Prints a JSON summary after successful ingestion.

    Raises:
        SystemExit: If command-line arguments are invalid.
        OSError: If a source file cannot be read.
        SQLAlchemyError: If PostgreSQL cannot persist spreadsheet rows.
        ApiException: If Qdrant rejects an operation.
        ResponseHandlingException: If Qdrant cannot be reached.
        ValueError: If configuration or source data is invalid.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Import Excel workbooks to PostgreSQL and PDFs to Qdrant."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    arguments: argparse.Namespace = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    result: IngestAllResult = asyncio.run(
        _run_cli_ingestion(arguments.data_dir)
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
