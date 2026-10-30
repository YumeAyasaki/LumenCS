"""Tests for converting workbook data to provenance-aware database models."""

from __future__ import annotations

import unittest

from app.database import Base
from app.tools.spreadsheet_ingester import _database_workbook
from app.tools.spreadsheet_reader import (
    SpreadsheetRow as ParsedRow,
    SpreadsheetSheet as ParsedSheet,
    SpreadsheetWorkbook,
)


class SpreadsheetIngesterTests(unittest.TestCase):
    """Verify workbook hierarchy and cell provenance are retained."""

    def test_database_schema_matches_workbook_hierarchy(self) -> None:
        """Create separate relational records for files, sheets, and rows."""
        self.assertEqual(
            set(Base.metadata.tables),
            {"source_workbooks", "source_sheets", "spreadsheet_rows"},
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
        )

        self.assertEqual(database_workbook.source_file, "product_catalog.xlsx")
        self.assertEqual(database_workbook.sha256, "a" * 64)
        self.assertEqual(database_workbook.sheets[0].columns, ["Mã model", "Tên sản phẩm"])
        self.assertEqual(database_workbook.sheets[0].rows[0].source_row, 6)
        self.assertEqual(
            database_workbook.sheets[0].rows[0].values_json,
            {"Mã model": "LH-AC09N", "Tên sản phẩm": "Điều hòa"},
        )


if __name__ == "__main__":
    unittest.main()
