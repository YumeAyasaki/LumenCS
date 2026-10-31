"""Add provenance-linked typed tables for workbook source data.

Revision ID: 0002_typed_source_tables
Revises: 0001_workbook_hierarchy
Create Date: 2026-10-31
"""

from __future__ import annotations

from typing import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0002_typed_source_tables"
down_revision: str | None = "0001_workbook_hierarchy"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create typed domain tables and track the mapping version per workbook."""
    op.add_column(
        "source_workbooks",
        sa.Column(
            "schema_sha256",
            sa.String(length=64),
            server_default="",
            nullable=False,
        ),
    )
    op.create_table(
        "products",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "source_row_id",
            sa.Integer(),
            sa.ForeignKey("spreadsheet_rows.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("model", sa.String(length=100), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("product_group", sa.String(length=100), nullable=False),
        sa.Column("specification", sa.Text(), nullable=True),
        sa.Column("energy_rating", sa.Text(), nullable=True),
        sa.Column("dimensions", sa.Text(), nullable=True),
        sa.Column("weight", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=100), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
    )
    op.create_index("ix_products_source_row_id", "products", ["source_row_id"])
    op.create_index("ix_products_model", "products", ["model"])
    op.create_index("ix_products_product_group", "products", ["product_group"])

    op.create_table(
        "prices",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "source_row_id",
            sa.Integer(),
            sa.ForeignKey("spreadsheet_rows.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("model", sa.String(length=100), nullable=False),
        sa.Column("region", sa.String(length=100), nullable=False),
        sa.Column("price", sa.Numeric(14, 2), nullable=False),
        sa.Column("valid_from", sa.Date(), nullable=False),
        sa.UniqueConstraint(
            "source_row_id",
            "region",
            name="uq_prices_source_region",
        ),
    )
    op.create_index("ix_prices_source_row_id", "prices", ["source_row_id"])
    op.create_index("ix_prices_model", "prices", ["model"])
    op.create_index("ix_prices_region", "prices", ["region"])
    op.create_index("ix_prices_valid_from", "prices", ["valid_from"])

    op.create_table(
        "promotions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "source_row_id",
            sa.Integer(),
            sa.ForeignKey("spreadsheet_rows.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("code", sa.String(length=100), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("starts_on", sa.Date(), nullable=False),
        sa.Column("ends_on", sa.Date(), nullable=False),
        sa.Column("scope_text", sa.Text(), nullable=False),
        sa.Column("scope", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("channel", sa.String(length=100), nullable=True),
        sa.Column("promotion_type", sa.String(length=50), nullable=False),
        sa.Column("value_amount", sa.Numeric(14, 2), nullable=True),
        sa.Column("value_unit", sa.String(length=20), nullable=True),
        sa.Column("value_details", sa.Text(), nullable=True),
        sa.Column("cap_amount", sa.Numeric(14, 2), nullable=True),
        sa.Column("conditions", sa.Text(), nullable=True),
        sa.Column("can_stack", sa.Boolean(), nullable=False),
        sa.Column("stacking_text", sa.Text(), nullable=True),
    )
    op.create_index("ix_promotions_source_row_id", "promotions", ["source_row_id"])
    op.create_index("ix_promotions_code", "promotions", ["code"])
    op.create_index("ix_promotions_starts_on", "promotions", ["starts_on"])
    op.create_index("ix_promotions_ends_on", "promotions", ["ends_on"])

    op.create_table(
        "warranty",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "source_row_id",
            sa.Integer(),
            sa.ForeignKey("spreadsheet_rows.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("product_group", sa.String(length=100), nullable=False),
        sa.Column("product_months", sa.Integer(), nullable=False),
        sa.Column("component", sa.Text(), nullable=True),
        sa.Column("component_months", sa.Integer(), nullable=True),
        sa.Column("exclusions", sa.Text(), nullable=True),
        sa.UniqueConstraint("source_row_id", name="uq_warranty_source_row"),
    )
    op.create_index("ix_warranty_source_row_id", "warranty", ["source_row_id"])
    op.create_index("ix_warranty_product_group", "warranty", ["product_group"])

    op.create_table(
        "regions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "source_row_id",
            sa.Integer(),
            sa.ForeignKey("spreadsheet_rows.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("province", sa.String(length=150), nullable=False),
        sa.Column("region", sa.String(length=100), nullable=False),
        sa.UniqueConstraint("source_row_id", name="uq_regions_source_row"),
    )
    op.create_index("ix_regions_source_row_id", "regions", ["source_row_id"])
    op.create_index("ix_regions_province", "regions", ["province"])
    op.create_index("ix_regions_region", "regions", ["region"])

    op.create_table(
        "service_centres",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "source_row_id",
            sa.Integer(),
            sa.ForeignKey("spreadsheet_rows.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("code", sa.String(length=50), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("province", sa.String(length=150), nullable=False),
        sa.Column("district", sa.String(length=150), nullable=True),
        sa.Column("address", sa.Text(), nullable=False),
        sa.Column("phone", sa.String(length=100), nullable=True),
        sa.Column("weekday_hours", sa.Text(), nullable=True),
        sa.Column("sunday_hours", sa.Text(), nullable=True),
        sa.Column(
            "accepted_services",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=100), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_service_centres_source_row_id",
        "service_centres",
        ["source_row_id"],
    )
    op.create_index("ix_service_centres_code", "service_centres", ["code"])
    op.create_index("ix_service_centres_province", "service_centres", ["province"])

    op.create_table(
        "service_fees",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "source_row_id",
            sa.Integer(),
            sa.ForeignKey("spreadsheet_rows.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("code", sa.String(length=50), nullable=False),
        sa.Column("service", sa.Text(), nullable=False),
        sa.Column("applies_to", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("unit", sa.String(length=50), nullable=True),
        sa.Column("fee_in_warranty", sa.Numeric(14, 2), nullable=True),
        sa.Column("fee_out_of_warranty", sa.Numeric(14, 2), nullable=True),
        sa.Column("fee_in_warranty_note", sa.Text(), nullable=True),
        sa.Column("fee_out_of_warranty_note", sa.Text(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.UniqueConstraint(
            "source_row_id",
            name="uq_service_fees_source_row",
        ),
    )
    op.create_index("ix_service_fees_source_row_id", "service_fees", ["source_row_id"])
    op.create_index("ix_service_fees_code", "service_fees", ["code"])


def downgrade() -> None:
    """Drop typed tables before removing the mapping hash column."""
    op.drop_index("ix_service_fees_code", table_name="service_fees")
    op.drop_index("ix_service_fees_source_row_id", table_name="service_fees")
    op.drop_table("service_fees")
    op.drop_index("ix_service_centres_province", table_name="service_centres")
    op.drop_index("ix_service_centres_code", table_name="service_centres")
    op.drop_index("ix_service_centres_source_row_id", table_name="service_centres")
    op.drop_table("service_centres")
    op.drop_index("ix_regions_region", table_name="regions")
    op.drop_index("ix_regions_province", table_name="regions")
    op.drop_index("ix_regions_source_row_id", table_name="regions")
    op.drop_table("regions")
    op.drop_index("ix_warranty_product_group", table_name="warranty")
    op.drop_index("ix_warranty_source_row_id", table_name="warranty")
    op.drop_table("warranty")
    op.drop_index("ix_promotions_ends_on", table_name="promotions")
    op.drop_index("ix_promotions_starts_on", table_name="promotions")
    op.drop_index("ix_promotions_code", table_name="promotions")
    op.drop_index("ix_promotions_source_row_id", table_name="promotions")
    op.drop_table("promotions")
    op.drop_index("ix_prices_valid_from", table_name="prices")
    op.drop_index("ix_prices_region", table_name="prices")
    op.drop_index("ix_prices_model", table_name="prices")
    op.drop_index("ix_prices_source_row_id", table_name="prices")
    op.drop_table("prices")
    op.drop_index("ix_products_product_group", table_name="products")
    op.drop_index("ix_products_model", table_name="products")
    op.drop_index("ix_products_source_row_id", table_name="products")
    op.drop_table("products")
    op.drop_column("source_workbooks", "schema_sha256")
