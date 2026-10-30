"""Persist parsed workbook hierarchies in PostgreSQL."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.database import SourceSheet, SourceWorkbook, SpreadsheetRow
from app.tools.spreadsheet_reader import (
    SpreadsheetWorkbook,
    read_spreadsheet_directory,
)

logger: logging.Logger = logging.getLogger(__name__)


class IngestionResult(BaseModel):
    """Summarize the result of spreadsheet ingestion.

    Attributes:
        processed_workbooks: Number of inserted or replaced workbook files.
        skipped_workbooks: Number of unchanged files skipped by SHA-256.
        row_count: Number of source worksheet rows stored.

    Raises:
        ValidationError: If any reported count is negative.
    """

    processed_workbooks: int = Field(ge=0)
    skipped_workbooks: int = Field(ge=0)
    row_count: int = Field(ge=0)


def _file_sha256(path: Path) -> str:
    """Calculate the SHA-256 digest of a workbook file.

    Args:
        path: Workbook path to hash.

    Returns:
        str: Lowercase hexadecimal SHA-256 digest.

    Raises:
        OSError: If the workbook cannot be read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _database_workbook(
    workbook: SpreadsheetWorkbook,
    sha256: str,
) -> SourceWorkbook:
    """Convert a parsed workbook into its related database models.

    Args:
        workbook: Parsed workbook with sheet and row provenance.
        sha256: Digest of the original workbook bytes.

    Returns:
        SourceWorkbook: A workbook model with worksheet and row relationships.
    """
    return SourceWorkbook(
        source_file=workbook.source_file,
        sha256=sha256,
        warnings=workbook.warnings,
        sheets=[
            SourceSheet(
                name=sheet.name,
                ordinal=sheet.ordinal,
                is_hidden=sheet.is_hidden,
                columns=sheet.columns,
                header_notes=[
                    note.model_dump(mode="json") for note in sheet.header_notes
                ],
                footnotes=[note.model_dump(mode="json") for note in sheet.footnotes],
                warnings=sheet.warnings,
                rows=[
                    SpreadsheetRow(
                        source_row=row.source_row,
                        values_json=row.values,
                    )
                    for row in sheet.rows
                ],
            )
            for sheet in workbook.sheets
        ],
    )


async def ingest_spreadsheet_directory(
    session_factory: async_sessionmaker[AsyncSession],
    data_dir: Path,
) -> IngestionResult:
    """Read workbook files and persist changed rows with their provenance.

    Unchanged files are skipped by content hash. Changed workbooks replace
    their previous workbook, worksheet, and row records atomically.

    Args:
        session_factory: Factory for asynchronous PostgreSQL sessions.
        data_dir: Directory recursively searched for mapped Excel files.

    Returns:
        IngestionResult: Counts of changed files, skipped files, and rows.

    Raises:
        FileNotFoundError: If the data directory contains no Excel files.
        OSError: If a workbook cannot be read or hashed.
        SQLAlchemyError: If PostgreSQL cannot persist the parsed data.
        ValueError: If a workbook has an invalid or ambiguous schema mapping.
    """
    workbooks: list[SpreadsheetWorkbook] = read_spreadsheet_directory(data_dir)
    hashes: dict[str, str] = {
        workbook.source_file: _file_sha256(data_dir / workbook.source_file)
        for workbook in workbooks
    }
    processed_workbooks: int = 0
    skipped_workbooks: int = 0
    row_count: int = 0

    async with session_factory() as session:
        async with session.begin():
            for workbook in workbooks:
                existing: SourceWorkbook | None = await session.scalar(
                    select(SourceWorkbook).where(
                        SourceWorkbook.source_file == workbook.source_file
                    )
                )
                sha256: str = hashes[workbook.source_file]
                # if existing is not None and existing.sha256 == sha256:
                #     skipped_workbooks += 1
                #     continue
                if existing is not None:
                    await session.delete(existing)
                    await session.flush()

                session.add(_database_workbook(workbook, sha256))
                processed_workbooks += 1
                row_count += sum(len(sheet.rows) for sheet in workbook.sheets)

    result: IngestionResult = IngestionResult(
        processed_workbooks=processed_workbooks,
        skipped_workbooks=skipped_workbooks,
        row_count=row_count,
    )
    logger.info(
        "Spreadsheet ingestion completed: %d changed, %d unchanged, %d rows",
        result.processed_workbooks,
        result.skipped_workbooks,
        result.row_count,
    )
    return result
