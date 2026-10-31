"""PostgreSQL models reflecting the source workbook hierarchy."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any
from urllib.parse import quote

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
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
        schema_sha256: Digest of the mapping used to create its raw and typed data.
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
    schema_sha256: Mapped[str] = mapped_column(String(64), default="")
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


class SourceRowProvenance:
    """Provide the raw-row foreign key shared by normalized source tables."""

    source_row_id: Mapped[int] = mapped_column(
        ForeignKey("spreadsheet_rows.id", ondelete="CASCADE"),
        index=True,
    )

    @property
    def cite_id(self) -> str:
        """Return a stable citation path derived from the source raw row.

        Returns:
            Citation containing the workbook, sheet, and original Excel row.

        Raises:
            AttributeError: If source relationships were not loaded or assigned.
        """
        row: SpreadsheetRow = self.source_row
        sheet: SourceSheet = row.sheet
        return (
            f"{sheet.workbook.source_file}#sheet={quote(sheet.name, safe='')}"
            f"&row={row.source_row}"
        )


class Product(SourceRowProvenance, Base):
    """Store normalized product catalog details and source-row provenance."""

    __tablename__ = "products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model: Mapped[str] = mapped_column(String(100), index=True)
    name: Mapped[str] = mapped_column(Text)
    product_group: Mapped[str] = mapped_column(String(100), index=True)
    specification: Mapped[str | None] = mapped_column(Text, nullable=True)
    energy_rating: Mapped[str | None] = mapped_column(Text, nullable=True)
    dimensions: Mapped[str | None] = mapped_column(Text, nullable=True)
    weight: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str | None] = mapped_column(String(100), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_row: Mapped[SpreadsheetRow] = relationship()


class Price(SourceRowProvenance, Base):
    """Store normalized model-region prices and their effective dates."""

    __tablename__ = "prices"
    __table_args__ = (
        UniqueConstraint("source_row_id", "region", name="uq_prices_source_region"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model: Mapped[str] = mapped_column(String(100), index=True)
    region: Mapped[str] = mapped_column(String(100), index=True)
    price: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    valid_from: Mapped[date] = mapped_column(Date, index=True)
    source_row: Mapped[SpreadsheetRow] = relationship()


class Promotion(SourceRowProvenance, Base):
    """Store normalized promotion scope, value, date, and stacking details."""

    __tablename__ = "promotions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(100), index=True)
    name: Mapped[str] = mapped_column(Text)
    starts_on: Mapped[date] = mapped_column(Date, index=True)
    ends_on: Mapped[date] = mapped_column(Date, index=True)
    scope_text: Mapped[str] = mapped_column(Text)
    scope: Mapped[dict[str, list[str]]] = mapped_column(JSONB)
    channel: Mapped[str | None] = mapped_column(String(100), nullable=True)
    promotion_type: Mapped[str] = mapped_column(String(50))
    value_amount: Mapped[Decimal | None] = mapped_column(
        Numeric(14, 2),
        nullable=True,
    )
    value_unit: Mapped[str | None] = mapped_column(String(20), nullable=True)
    value_details: Mapped[str | None] = mapped_column(Text, nullable=True)
    cap_amount: Mapped[Decimal | None] = mapped_column(Numeric(14, 2), nullable=True)
    conditions: Mapped[str | None] = mapped_column(Text, nullable=True)
    can_stack: Mapped[bool] = mapped_column(Boolean)
    stacking_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_row: Mapped[SpreadsheetRow] = relationship()


class Warranty(SourceRowProvenance, Base):
    """Store product-group and component warranty durations."""

    __tablename__ = "warranty"
    __table_args__ = (
        UniqueConstraint("source_row_id", name="uq_warranty_source_row"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_group: Mapped[str] = mapped_column(String(100), index=True)
    product_months: Mapped[int] = mapped_column(Integer)
    component: Mapped[str | None] = mapped_column(Text, nullable=True)
    component_months: Mapped[int | None] = mapped_column(Integer, nullable=True)
    exclusions: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_row: Mapped[SpreadsheetRow] = relationship()


class Region(SourceRowProvenance, Base):
    """Store the province-to-price-region mapping."""

    __tablename__ = "regions"
    __table_args__ = (UniqueConstraint("source_row_id", name="uq_regions_source_row"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    province: Mapped[str] = mapped_column(String(150), index=True)
    region: Mapped[str] = mapped_column(String(100), index=True)
    source_row: Mapped[SpreadsheetRow] = relationship()


class ServiceCentre(SourceRowProvenance, Base):
    """Store normalized service-centre details and accepted product groups."""

    __tablename__ = "service_centres"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(50), index=True)
    name: Mapped[str] = mapped_column(Text)
    province: Mapped[str] = mapped_column(String(150), index=True)
    district: Mapped[str | None] = mapped_column(String(150), nullable=True)
    address: Mapped[str] = mapped_column(Text)
    phone: Mapped[str | None] = mapped_column(String(100), nullable=True)
    weekday_hours: Mapped[str | None] = mapped_column(Text, nullable=True)
    sunday_hours: Mapped[str | None] = mapped_column(Text, nullable=True)
    accepted_services: Mapped[dict[str, bool]] = mapped_column(JSONB)
    status: Mapped[str | None] = mapped_column(String(100), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_row: Mapped[SpreadsheetRow] = relationship()


class ServiceFee(SourceRowProvenance, Base):
    """Store normalized service prices, applicability, and unavailable labels."""

    __tablename__ = "service_fees"
    __table_args__ = (
        UniqueConstraint("source_row_id", name="uq_service_fees_source_row"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(50), index=True)
    service: Mapped[str] = mapped_column(Text)
    applies_to: Mapped[list[str]] = mapped_column(JSONB)
    unit: Mapped[str | None] = mapped_column(String(50), nullable=True)
    fee_in_warranty: Mapped[Decimal | None] = mapped_column(
        Numeric(14, 2),
        nullable=True,
    )
    fee_out_of_warranty: Mapped[Decimal | None] = mapped_column(
        Numeric(14, 2),
        nullable=True,
    )
    fee_in_warranty_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    fee_out_of_warranty_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_row: Mapped[SpreadsheetRow] = relationship()
