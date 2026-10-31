"""Tests for converting workbook data to provenance-aware database models."""

from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

from app.database import (
    Base,
    Product,
    Promotion,
    ServiceFee,
    SourceSheet,
    SourceWorkbook,
    SpreadsheetRow as DatabaseRow,
)
from app.tools.spreadsheet_ingester import _database_workbook
from app.tools.spreadsheet_normalizer import normalize_source_row
from app.tools.spreadsheet_reader import (
    SpreadsheetRow as ParsedRow,
    SpreadsheetSheet as ParsedSheet,
    SpreadsheetWorkbook,
    read_spreadsheet_directory,
)


class SpreadsheetIngesterTests(unittest.TestCase):
    """Verify workbook hierarchy and cell provenance are retained."""

    def test_database_schema_has_raw_and_typed_tables(self) -> None:
        """Create raw hierarchy tables and normalized typed domain tables."""
        self.assertEqual(
            set(Base.metadata.tables),
            {
                "source_workbooks",
                "source_sheets",
                "spreadsheet_rows",
                "products",
                "prices",
                "promotions",
                "warranty",
                "regions",
                "service_centres",
                "service_fees",
            },
        )
        for table_name in (
            "products",
            "prices",
            "promotions",
            "warranty",
            "regions",
            "service_centres",
            "service_fees",
        ):
            source_row_column = Base.metadata.tables[table_name].c.source_row_id
            self.assertEqual(
                str(next(iter(source_row_column.foreign_keys)).target_fullname),
                "spreadsheet_rows.id",
            )

    def test_conversion_preserves_sheet_columns_and_excel_row(self) -> None:
        """Keep worksheet headers, values, and source row in ORM records."""
        parsed_workbook: SpreadsheetWorkbook = SpreadsheetWorkbook(
            source_file="product_catalog.xlsx",
            sheets=[
                ParsedSheet(
                    name="Danh mục SP",
                    ordinal=1,
                    is_hidden=False,
                    columns=["Mã model", "Tên sản phẩm"],
                    rows=[
                        ParsedRow(
                            source_row=6,
                            values={
                                "Mã model": "LH-AC09N",
                                "Tên sản phẩm": "Điều hòa",
                            },
                        )
                    ],
                )
            ],
        )

        database_workbook = _database_workbook(
            parsed_workbook,
            "a" * 64,
            "b" * 64,
        )

        self.assertEqual(database_workbook.source_file, "product_catalog.xlsx")
        self.assertEqual(database_workbook.sha256, "a" * 64)
        self.assertEqual(database_workbook.schema_sha256, "b" * 64)
        self.assertEqual(database_workbook.sheets[0].columns, ["Mã model", "Tên sản phẩm"])
        self.assertEqual(database_workbook.sheets[0].rows[0].source_row, 6)
        self.assertEqual(
            database_workbook.sheets[0].rows[0].values_json,
            {"Mã model": "LH-AC09N", "Tên sản phẩm": "Điều hòa"},
        )

    def test_normalizes_source_rows_with_reviewable_rules(self) -> None:
        """Populate typed models from source rows with provenance foreign keys."""
        workbooks: list[SpreadsheetWorkbook] = read_spreadsheet_directory(
            Path("data")
        )
        counts: Counter[str] = Counter()
        promotions: list[Promotion] = []
        service_fees: list[ServiceFee] = []
        product: Product | None = None
        archive_sheets: list[tuple[str, str]] = []

        for workbook in workbooks:
            for sheet in workbook.sheets:
                if sheet.archive:
                    archive_sheets.append((workbook.source_file, sheet.name))
                mapping = sheet.normalization
                if sheet.archive or mapping is None:
                    continue
                for row in sheet.rows:
                    raw_row: DatabaseRow = DatabaseRow(
                        id=row.source_row,
                        sheet_id=sheet.ordinal,
                        source_row=row.source_row,
                        values_json=row.values,
                    )
                    normalized_rows = normalize_source_row(
                        raw_row,
                        mapping,
                        f"{workbook.source_file}#{sheet.name}#{row.source_row}",
                    )
                    counts[mapping.table] += len(normalized_rows)
                    if mapping.table == "promotions":
                        promotions.extend(normalized_rows)
                    elif mapping.table == "service_fees":
                        service_fees.extend(normalized_rows)
                    elif mapping.table == "products" and product is None:
                        product = normalized_rows[0]

        self.assertEqual(
            counts,
            {
                "products": 14,
                "prices": 48,
                "promotions": 5,
                "warranty": 4,
                "regions": 9,
                "service_centres": 16,
                "service_fees": 17,
            },
        )
        first_promotion: Promotion = promotions[0]
        self.assertEqual(first_promotion.code, "KM-2610")
        self.assertEqual(first_promotion.scope["include"], ["LH-AC*"])
        self.assertEqual(first_promotion.scope["exclude"], ["LH-AC09S"])
        self.assertEqual(first_promotion.value_amount, 10)
        self.assertEqual(first_promotion.cap_amount, 2000000)
        self.assertTrue(first_promotion.can_stack)
        self.assertEqual(first_promotion.source_row_id, 5)
        installment: Promotion = next(
            promotion for promotion in promotions if promotion.code == "KM-TRAGOP"
        )
        self.assertEqual(installment.promotion_type, "installment")
        self.assertEqual(installment.value_amount, 0)
        self.assertEqual(installment.scope["include"], ["*"])
        gift: Promotion = next(
            promotion for promotion in promotions if promotion.code == "KM-LOC"
        )
        self.assertEqual(gift.value_amount, 690000)
        self.assertIsNotNone(product)
        first_product: Product = product
        self.assertEqual(first_product.model, "LH-AC09N")
        self.assertEqual(first_product.source_row_id, 6)
        unavailable_fee: ServiceFee = next(
            fee for fee in service_fees if fee.code == "DV06"
        )
        self.assertIsNone(unavailable_fee.fee_in_warranty)
        self.assertEqual(unavailable_fee.fee_in_warranty_note, "Không phục vụ")
        self.assertEqual(
            archive_sheets,
            [("product_catalog.xlsx", "Giá 2025 (lưu trữ)")],
        )

    def test_typed_row_citation_uses_raw_source_hierarchy(self) -> None:
        """Derive a citation from the workbook, sheet, and raw Excel row."""
        raw_row: DatabaseRow = DatabaseRow(
            id=42,
            sheet_id=1,
            source_row=6,
            values_json={"Mã model": "LH-AC09N"},
        )
        source_sheet: SourceSheet = SourceSheet(
            name="Danh mục SP",
            ordinal=1,
            is_hidden=False,
            columns=["Mã model"],
            header_notes=[],
            footnotes=[],
            warnings=[],
            rows=[raw_row],
        )
        workbook: SourceWorkbook = SourceWorkbook(
            source_file="product_catalog.xlsx",
            sha256="a" * 64,
            schema_sha256="b" * 64,
            warnings=[],
            sheets=[source_sheet],
        )
        source_sheet.workbook = workbook
        product: Product = Product(
            source_row_id=42,
            model="LH-AC09N",
            name="Điều hòa",
            product_group="Điều hòa",
            source_row=raw_row,
        )

        self.assertEqual(
            product.cite_id,
            "product_catalog.xlsx#sheet=Danh%20m%E1%BB%A5c%20SP&row=6",
        )


if __name__ == "__main__":
    unittest.main()
