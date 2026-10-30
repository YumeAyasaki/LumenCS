"""PostgreSQL models reflecting the source workbook hierarchy."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(AsyncAttrs, DeclarativeBase):
    """Provide shared SQLAlchemy metadata for PostgreSQL models.

    Attributes:
        metadata: Registry of application database tables.
    """


class SourceWorkbook(Base):
    """Track an ingested workbook by relative filename and content hash.

    Attributes:
        id: Database identifier for the workbook.
        source_file: Path relative to the ingested data directory.
        sha256: Digest used to avoid reprocessing unchanged files.
        warnings: Workbook parsing and schema-mapping warnings.
        ingested_at: Timestamp of the latest successful ingestion.
        sheets: Worksheets belonging to this workbook.

    Raises:
        SQLAlchemyError: If database operations on this model fail.
    """

    __tablename__ = "source_workbooks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_file: Mapped[str] = mapped_column(String(1024), unique=True)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    warnings: Mapped[list[str]] = mapped_column(JSONB, default=list)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    sheets: Mapped[list[SourceSheet]] = relationship(
        back_populates="workbook",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="SourceSheet.ordinal",
    )


class SourceSheet(Base):
    """Store worksheet-level column layout and explanatory notes.

    Attributes:
        id: Database identifier for the worksheet.
        workbook_id: Foreign key to the source workbook.
        name: Original worksheet title.
        ordinal: Worksheet position in the workbook.
        is_hidden: Whether the source worksheet was hidden.
        columns: Ordered header labels extracted from the worksheet.
        header_notes: Non-tabular content before the header rows.
        footnotes: Non-tabular content after the table rows.
        warnings: Worksheet parsing or mapping warnings.
        workbook: Parent source workbook.
        rows: Non-empty data rows with their Excel row numbers.

    Raises:
        SQLAlchemyError: If database operations on this model fail.
    """

    __tablename__ = "source_sheets"
    __table_args__ = (
        UniqueConstraint("workbook_id", "name", name="uq_source_sheets_workbook_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    workbook_id: Mapped[int] = mapped_column(
        ForeignKey("source_workbooks.id", ondelete="CASCADE"),
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255))
    ordinal: Mapped[int] = mapped_column(Integer)
    is_hidden: Mapped[bool] = mapped_column(Boolean)
    columns: Mapped[list[str]] = mapped_column(JSONB)
    header_notes: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    footnotes: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    warnings: Mapped[list[str]] = mapped_column(JSONB, default=list)
    workbook: Mapped[SourceWorkbook] = relationship(back_populates="sheets")
    rows: Mapped[list[SpreadsheetRow]] = relationship(
        back_populates="sheet",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="SpreadsheetRow.source_row",
    )


class SpreadsheetRow(Base):
    """Store one non-empty worksheet data row and its Excel row number.

    Attributes:
        id: Database identifier for the source row.
        sheet_id: Foreign key to the source worksheet.
        source_row: Original one-based Excel row number.
        values_json: Cell values keyed by their worksheet header labels.
        sheet: Parent worksheet, which links to the source file provenance.

    Raises:
        SQLAlchemyError: If database operations on this model fail.
    """

    __tablename__ = "spreadsheet_rows"
    __table_args__ = (
        UniqueConstraint("sheet_id", "source_row", name="uq_spreadsheet_rows_sheet_row"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sheet_id: Mapped[int] = mapped_column(
        ForeignKey("source_sheets.id", ondelete="CASCADE"),
        index=True,
    )
    source_row: Mapped[int] = mapped_column(Integer)
    values_json: Mapped[dict[str, Any]] = mapped_column("values", JSONB)
    sheet: Mapped[SourceSheet] = relationship(back_populates="rows")
