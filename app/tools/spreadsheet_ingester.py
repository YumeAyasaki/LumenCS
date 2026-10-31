"""Persist parsed workbook hierarchies in PostgreSQL."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import delete, select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.database import (
    SourceSheet,
    SourceWorkbook,
    SpreadsheetRow,
)
from app.tools.spreadsheet_reader import (
    DEFAULT_MAPPING_PATH,
    SchemaMapping,
    SpreadsheetSheet,
    SpreadsheetWorkbook,
    load_schema_mapping,
    read_spreadsheet_directory,
    select_workbook_mapping,
)
from app.tools.spreadsheet_normalizer import (
    NORMALIZED_MODELS,
    normalize_source_row,
)

logger: logging.Logger = logging.getLogger(__name__)


class IngestionResult(BaseModel):
    """Summarize the result of spreadsheet ingestion.

    Attributes:
        processed_workbooks: Number of inserted or replaced workbook files.
        skipped_workbooks: Number of unchanged files skipped by SHA-256.
        row_count: Number of source worksheet rows stored.
        normalized_row_count: Number of typed rows created.
        warnings: Non-fatal mapping or value-conversion warnings.

    Raises:
        ValidationError: If any reported count is negative.
    """

    processed_workbooks: int = Field(ge=0)
    skipped_workbooks: int = Field(ge=0)
    row_count: int = Field(ge=0)
    normalized_row_count: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)


class RenormalizationResult(BaseModel):
    """Report rebuilding typed rows from already persisted raw source rows.

    Attributes:
        source_row_count: Raw rows examined from PostgreSQL.
        normalized_row_count: Typed records rebuilt from those rows.
        warnings: Non-fatal mapping or value-conversion warnings.
    """

    source_row_count: int = Field(ge=0)
    normalized_row_count: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


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
    schema_sha256: str = "",
) -> SourceWorkbook:
    """Convert a parsed workbook into its related database models.

    Args:
        workbook: Parsed workbook with sheet and row provenance.
        sha256: Digest of the original workbook bytes.
        schema_sha256: Digest of the active mapping JSON.

    Returns:
        SourceWorkbook: A workbook model with worksheet and row relationships.
    """
    return SourceWorkbook(
        source_file=workbook.source_file,
        sha256=sha256,
        schema_sha256=schema_sha256,
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


async def _clear_normalized_rows(
    session: AsyncSession,
    workbook_id: int,
) -> None:
    """Delete typed rows for one workbook while preserving raw source rows.

    Args:
        session: Active SQLAlchemy session.
        workbook_id: Database ID of the source workbook.

    Returns:
        None.

    Raises:
        SQLAlchemyError: If PostgreSQL rejects a delete statement.
    """
    source_row_ids = (
        select(SpreadsheetRow.id)
        .join(SourceSheet, SpreadsheetRow.sheet_id == SourceSheet.id)
        .where(SourceSheet.workbook_id == workbook_id)
    )
    for model in NORMALIZED_MODELS.values():
        await session.execute(
            delete(model).where(model.source_row_id.in_(source_row_ids))
        )


async def _normalize_persisted_workbook(
    session: AsyncSession,
    workbook: SourceWorkbook,
    parsed_workbook: SpreadsheetWorkbook,
) -> tuple[int, list[str]]:
    """Build normalized ORM rows linked to the persisted raw rows.

    Args:
        session: Active SQLAlchemy session.
        workbook: Persisted raw workbook hierarchy with row IDs assigned.
        parsed_workbook: Parsed workbook carrying the reviewed mappings.

    Returns:
        A pair of normalized row count and non-fatal parse warnings.
    """
    parsed_sheets: dict[str, SpreadsheetSheet] = {
        sheet.name: sheet for sheet in parsed_workbook.sheets
    }
    database_sheets: dict[str, SourceSheet] = {
        sheet.name: sheet for sheet in workbook.sheets
    }
    warnings: list[str] = []
    normalized_count: int = 0
    for sheet_name, parsed_sheet in parsed_sheets.items():
        source_sheet: SourceSheet = database_sheets[sheet_name]
        mapping = parsed_sheet.normalization
        if parsed_sheet.archive or mapping is None:
            continue
        for raw_row in source_sheet.rows:
            source: str = (
                f"{workbook.source_file}#sheet={sheet_name}"
                f"&row={raw_row.source_row}"
            )
            try:
                normalized_rows = normalize_source_row(raw_row, mapping, source)
            except (KeyError, TypeError, ValueError) as error:
                warning: str = f"Could not normalize {source}: {error}"
                source_sheet.warnings.append(warning)
                workbook.warnings.append(warning)
                warnings.append(warning)
                logger.warning("%s", warning)
                continue
            session.add_all(normalized_rows)
            normalized_count += len(normalized_rows)
    await session.flush()
    return normalized_count, warnings


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
        ValueError: If a workbook file cannot be parsed.
    """
    workbooks: list[SpreadsheetWorkbook] = read_spreadsheet_directory(data_dir)
    mapping_warnings: list[str] = []
    try:
        schema_sha256: str = _file_sha256(DEFAULT_MAPPING_PATH)
    except OSError as error:
        schema_sha256 = hashlib.sha256(b"schema-mapping-unavailable").hexdigest()
        warning: str = (
            f"Cannot hash schema mapping {DEFAULT_MAPPING_PATH}: {error}"
        )
        mapping_warnings.append(warning)
        logger.warning("%s", warning)
    hashes: dict[str, str] = {
        workbook.source_file: _file_sha256(data_dir / workbook.source_file)
        for workbook in workbooks
    }
    processed_workbooks: int = 0
    skipped_workbooks: int = 0
    row_count: int = 0
    normalized_row_count: int = 0
    warnings: list[str] = [
        warning
        for workbook in workbooks
        for warning in workbook.warnings
    ]
    warnings.extend(mapping_warnings)

    async with session_factory() as session:
        async with session.begin():
            for workbook in workbooks:
                existing: SourceWorkbook | None = await session.scalar(
                    select(SourceWorkbook)
                    .options(
                        selectinload(SourceWorkbook.sheets).selectinload(
                            SourceSheet.rows
                        )
                    )
                    .where(SourceWorkbook.source_file == workbook.source_file)
                )
                sha256: str = hashes[workbook.source_file]
                if (
                    existing is not None
                    and existing.sha256 == sha256
                    and existing.schema_sha256 == schema_sha256
                ):
                    skipped_workbooks += 1
                    continue
                if existing is not None and existing.sha256 == sha256:
                    await _clear_normalized_rows(session, existing.id)
                    existing.schema_sha256 = schema_sha256
                    existing.warnings = list(workbook.warnings)
                    for sheet in existing.sheets:
                        sheet.warnings = []
                    await session.flush()
                    normalized_count, normalization_warnings = (
                        await _normalize_persisted_workbook(
                            session,
                            existing,
                            workbook,
                        )
                    )
                    normalized_row_count += normalized_count
                    warnings.extend(normalization_warnings)
                    row_count += sum(
                        len(sheet.rows) for sheet in workbook.sheets
                    )
                    processed_workbooks += 1
                    continue
                if existing is not None:
                    await session.delete(existing)
                    await session.flush()

                database_workbook: SourceWorkbook = _database_workbook(
                    workbook,
                    sha256,
                    schema_sha256,
                )
                session.add(database_workbook)
                await session.flush()
                normalized_count, normalization_warnings = (
                    await _normalize_persisted_workbook(
                        session,
                        database_workbook,
                        workbook,
                    )
                )
                processed_workbooks += 1
                row_count += sum(len(sheet.rows) for sheet in workbook.sheets)
                normalized_row_count += normalized_count
                warnings.extend(normalization_warnings)

    result: IngestionResult = IngestionResult(
        processed_workbooks=processed_workbooks,
        skipped_workbooks=skipped_workbooks,
        row_count=row_count,
        normalized_row_count=normalized_row_count,
        warnings=warnings,
    )
    logger.info(
        "Spreadsheet ingestion completed: %d changed, %d unchanged, "
        "%d raw rows, %d normalized rows",
        result.processed_workbooks,
        result.skipped_workbooks,
        result.row_count,
        result.normalized_row_count,
    )
    return result


async def renormalize_spreadsheet_rows(
    session_factory: async_sessionmaker[AsyncSession],
) -> RenormalizationResult:
    """Rebuild typed tables from persisted spreadsheet rows only.

    No source workbook is opened. Existing normalized records are replaced
    using the current JSON schema mappings while raw audit rows retain IDs.

    Args:
        session_factory: Factory for asynchronous PostgreSQL sessions.

    Returns:
        Counts of raw rows examined and typed rows created, plus warnings.

    Raises:
        OSError: If the mapping JSON cannot be read.
        ValidationError: If the schema mapping is invalid.
        SQLAlchemyError: If PostgreSQL cannot query or update rows.
    """
    warnings: list[str] = []
    source_row_count: int = 0
    normalized_row_count: int = 0
    try:
        schema_mapping: SchemaMapping | None = load_schema_mapping(
            DEFAULT_MAPPING_PATH
        )
    except (OSError, ValidationError, ValueError) as error:
        warning: str = (
            f"Cannot re-normalize with {DEFAULT_MAPPING_PATH}: {error}"
        )
        logger.warning("%s", warning)
        return RenormalizationResult(
            source_row_count=0,
            normalized_row_count=0,
            warnings=[warning],
        )
    async with session_factory() as session:
        async with session.begin():
            workbooks: list[SourceWorkbook] = list(
                (
                    await session.scalars(
                        select(SourceWorkbook).options(
                            selectinload(SourceWorkbook.sheets).selectinload(
                                SourceSheet.rows
                            )
                        )
                    )
                ).all()
            )
            for workbook in workbooks:
                source_row_count += sum(len(sheet.rows) for sheet in workbook.sheets)
                try:
                    if schema_mapping is None:
                        raise ValueError("schema mapping is unavailable")
                    workbook_mapping = select_workbook_mapping(
                        Path(workbook.source_file).name,
                        schema_mapping,
                    )
                except ValueError as error:
                    await _clear_normalized_rows(session, workbook.id)
                    warning: str = (
                        f"Cannot re-normalize {workbook.source_file}: {error}"
                    )
                    warnings.append(warning)
                    logger.warning("%s", warning)
                    continue
                await _clear_normalized_rows(session, workbook.id)
                workbook.warnings = []
                database_sheets: dict[str, SourceSheet] = {
                    sheet.name: sheet for sheet in workbook.sheets
                }
                for source_sheet in workbook.sheets:
                    source_sheet.warnings = []
                for sheet_name, sheet_mapping in workbook_mapping.sheets.items():
                    source_sheet: SourceSheet | None = database_sheets.get(sheet_name)
                    if (
                        source_sheet is None
                        or sheet_mapping.archive
                        or sheet_mapping.normalization is None
                    ):
                        continue
                    for raw_row in source_sheet.rows:
                        source: str = (
                            f"{workbook.source_file}#sheet={sheet_name}"
                            f"&row={raw_row.source_row}"
                        )
                        try:
                            normalized_rows = normalize_source_row(
                                raw_row,
                                sheet_mapping.normalization,
                                source,
                            )
                        except (KeyError, TypeError, ValueError) as error:
                            warning = f"Could not normalize {source}: {error}"
                            source_sheet.warnings.append(warning)
                            workbook.warnings.append(warning)
                            warnings.append(warning)
                            logger.warning("%s", warning)
                            continue
                        session.add_all(normalized_rows)
                        normalized_row_count += len(normalized_rows)
                workbook.schema_sha256 = _file_sha256(DEFAULT_MAPPING_PATH)
                await session.flush()
    logger.info(
        "Re-normalized %d stored source rows into %d typed records",
        source_row_count,
        normalized_row_count,
    )
    return RenormalizationResult(
        source_row_count=source_row_count,
        normalized_row_count=normalized_row_count,
        warnings=warnings,
    )
