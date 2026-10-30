"""Create a provenance-preserving workbook, worksheet, and row schema.

Revision ID: 0001_workbook_hierarchy
Revises:
Create Date: 2026-10-30
"""

from __future__ import annotations

from typing import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0001_workbook_hierarchy"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create normalized workbook, worksheet, and source-row tables."""
    op.create_table(
        "source_workbooks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source_file", sa.String(length=1024), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column(
            "warnings",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("source_file", name="uq_source_workbooks_source_file"),
    )
    op.create_index(
        "ix_source_workbooks_sha256",
        "source_workbooks",
        ["sha256"],
    )
    op.create_table(
        "source_sheets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "workbook_id",
            sa.Integer(),
            sa.ForeignKey("source_workbooks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("is_hidden", sa.Boolean(), nullable=False),
        sa.Column("columns", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "header_notes",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "footnotes",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "warnings",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "workbook_id",
            "name",
            name="uq_source_sheets_workbook_name",
        ),
    )
    op.create_index("ix_source_sheets_workbook_id", "source_sheets", ["workbook_id"])
    op.create_table(
        "spreadsheet_rows",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "sheet_id",
            sa.Integer(),
            sa.ForeignKey("source_sheets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("source_row", sa.Integer(), nullable=False),
        sa.Column("values", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.UniqueConstraint(
            "sheet_id",
            "source_row",
            name="uq_spreadsheet_rows_sheet_row",
        ),
    )
    op.create_index("ix_spreadsheet_rows_sheet_id", "spreadsheet_rows", ["sheet_id"])


def downgrade() -> None:
    """Drop the workbook hierarchy in foreign-key-safe order."""
    op.drop_index("ix_spreadsheet_rows_sheet_id", table_name="spreadsheet_rows")
    op.drop_table("spreadsheet_rows")
    op.drop_index("ix_source_sheets_workbook_id", table_name="source_sheets")
    op.drop_table("source_sheets")
    op.drop_index("ix_source_workbooks_sha256", table_name="source_workbooks")
    op.drop_table("source_workbooks")
