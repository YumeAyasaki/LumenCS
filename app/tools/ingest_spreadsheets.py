"""Command-line entry point for importing workbooks into PostgreSQL."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Sequence

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import Settings
from app.tools.spreadsheet_ingester import IngestionResult, ingest_spreadsheet_directory

logger: logging.Logger = logging.getLogger(__name__)


async def _ingest(data_dir: Path) -> IngestionResult:
    """Ingest workbook files and close the PostgreSQL engine.

    Args:
        data_dir: Directory containing source workbook files.

    Returns:
        IngestionResult: Counts for inserted, skipped, and stored rows.

    Raises:
        OSError: If source workbook files cannot be read.
        SQLAlchemyError: If PostgreSQL is unavailable or rejects the data.
        ValueError: If workbook schema configuration is invalid.
    """
    settings: Settings = Settings.from_environment()
    engine: AsyncEngine = create_async_engine(
        settings.database_url,
        pool_pre_ping=True,
    )
    session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        engine,
        expire_on_commit=False,
    )
    try:
        return await ingest_spreadsheet_directory(session_factory, data_dir)
    finally:
        await engine.dispose()


def main(argv: Sequence[str] | None = None) -> None:
    """Parse command-line options and ingest the selected workbook directory.

    Args:
        argv: Optional command-line argument sequence for programmatic use.

    Returns:
        None.

    Raises:
        SystemExit: If command-line arguments are invalid.
        OSError: If workbook files cannot be read.
        SQLAlchemyError: If PostgreSQL is unavailable or rejects the data.
        ValueError: If workbook schema configuration is invalid.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Import mapped Excel workbooks into PostgreSQL."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing Excel workbooks (default: data).",
    )
    arguments: argparse.Namespace = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        result: IngestionResult = asyncio.run(_ingest(arguments.data_dir))
    except (OSError, SQLAlchemyError, ValueError):
        logger.exception("Spreadsheet ingestion failed")
        raise
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
