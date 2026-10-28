"""Tests for the reusable spreadsheet reader component."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from openpyxl import Workbook
from openpyxl.worksheet.worksheet import Worksheet

from app.tools.spreadsheet_reader import (
    SchemaMapping,
    SheetMapping,
    SpreadsheetSheet,
    SpreadsheetWorkbook,
    WorkbookMapping,
    read_spreadsheet_directory,
    read_workbook,
    select_workbook_mapping,
)


class SpreadsheetReaderTests(unittest.TestCase):
    """Verify spreadsheet layouts, values, and source provenance."""

    def test_reads_merged_headers_and_preserves_excel_rows(self) -> None:
        """Read multi-row headers and typed values without losing row numbers."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            workbook_path: Path = Path(temporary_directory) / "products.xlsx"
            workbook: Workbook = Workbook()
            worksheet: Worksheet = workbook.active
            worksheet.title = "Products"
            worksheet["A2"] = "Product"
            worksheet.merge_cells("B2:C2")
            worksheet["B2"] = "Details"
            worksheet["A3"] = "Model"
            worksheet["B3"] = "Price"
            worksheet["C3"] = "Date"
            worksheet["A5"] = "LH-001"
            worksheet["B5"] = 1200
            worksheet["C5"] = date(2026, 10, 28)
            workbook.save(workbook_path)
            workbook.close()

            parsed: SpreadsheetWorkbook = read_workbook(
                workbook_path,
                WorkbookMapping(
                    sheets={
                        "Products": SheetMapping(
                            header_rows=(2, 3),
                            data_start_row=5,
                        )
                    }
                ),
                source_file="products.xlsx",
            )

        sheet: SpreadsheetSheet = parsed.sheets[0]
        self.assertEqual(
            sheet.columns,
            ["Product / Model", "Details / Price", "Details / Date"],
        )
        self.assertEqual(sheet.rows[0].source_row, 5)
        self.assertEqual(
            sheet.rows[0].values,
            {
                "Product / Model": "LH-001",
                "Details / Price": 1200,
                "Details / Date": "2026-10-28T00:00:00",
            },
        )
        self.assertEqual(parsed.source_file, "products.xlsx")

    def test_reads_all_configured_workbooks_from_directory(self) -> None:
        """Load workbook files using the reviewable JSON schema mapping."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir: Path = Path(temporary_directory) / "data"
            data_dir.mkdir()
            workbook_path: Path = data_dir / "catalog.xlsx"
            workbook: Workbook = Workbook()
            worksheet: Worksheet = workbook.active
            worksheet.title = "Products"
            worksheet["A1"] = "Model"
            worksheet["A2"] = "LH-001"
            workbook.save(workbook_path)
            workbook.close()
            mapping_path: Path = data_dir / "mapping.json"
            mapping_path.write_text(
                json.dumps(
                    {
                        "files": {
                            "catalog.xlsx": {
                                "sheets": {
                                    "Products": {
                                        "header_rows": [1],
                                        "data_start_row": 2,
                                    }
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            workbooks: list[SpreadsheetWorkbook] = read_spreadsheet_directory(
                data_dir,
                mapping_path,
            )

        self.assertEqual(len(workbooks), 1)
        self.assertEqual(workbooks[0].source_file, "catalog.xlsx")
        self.assertEqual(workbooks[0].sheets[0].rows[0].values["Model"], "LH-001")

    def test_selects_filename_glob_mapping(self) -> None:
        """Select a workbook mapping based on a configured filename pattern."""
        workbook_mapping: WorkbookMapping = WorkbookMapping(
            sheets={
                "Products": SheetMapping(header_rows=(1,), data_start_row=2),
            }
        )
        schema_mapping: SchemaMapping = SchemaMapping(
            files={"product_catalog*.xlsx": workbook_mapping}
        )

        self.assertIs(
            select_workbook_mapping("product_catalog_v2.xlsx", schema_mapping),
            workbook_mapping,
        )

    def test_rejects_missing_workbook_mapping(self) -> None:
        """Fail explicitly when a workbook has no configured mapping."""
        with self.assertRaisesRegex(ValueError, "found 0"):
            select_workbook_mapping("unknown.xlsx", SchemaMapping(files={}))


if __name__ == "__main__":
    unittest.main()
