"""Tests for model-facing tool schemas, execution, and safe calculations."""

from __future__ import annotations

import unittest
from datetime import date, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.database import (
    Price,
    Region,
    SourceSheet,
    SourceWorkbook,
    SpreadsheetRow,
    Warranty,
)
from app.tools.model_tools import (
    CalculationRecord,
    DateContextRecord,
    ModelToolContext,
    PriceRecord,
    PolicyRecord,
    UnknownModelToolError,
    WarrantyRecord,
    _calculate_expression,
    _make_result,
    execute_model_tool,
    get_model_tool_definitions,
)


class ModelToolTests(unittest.IsolatedAsyncioTestCase):
    """Verify tool descriptions, validated dispatch, and source citations."""

    def setUp(self) -> None:
        """Create inert database and policy clients for tool invocations."""
        self.session: AsyncMock = AsyncMock(spec=AsyncSession)
        self.context: ModelToolContext = ModelToolContext(
            session=self.session,
            qdrant_client=MagicMock(),
            embedding_model=MagicMock(),
            settings=Settings(
                database_url="postgresql+asyncpg://localhost/lumen",
                qdrant_url="http://localhost:6333",
            ),
        )

    def test_definitions_include_all_typed_tools_and_as_of(self) -> None:
        """Expose every tool with an explicit asked-at time in its schema."""
        definitions = get_model_tool_definitions()
        names: list[str] = [
            item.function.name
            for item in definitions
        ]

        self.assertEqual(
            names,
            [
                "get_product_details",
                "quote_price",
                "get_promotions",
                "get_warranty",
                "get_region",
                "find_service_centres",
                "get_service_fee",
                "search_policy",
                "calculate",
                "get_date_context",
            ],
        )
        for definition in definitions:
            self.assertIn("as_of", definition.function.parameters["properties"])
            self.assertIn("as_of", definition.function.parameters["required"])

    async def test_date_tool_uses_question_timestamp(self) -> None:
        """Return date context from as_of, not the server's wall-clock date."""
        result = await execute_model_tool(
            "get_date_context",
            {"as_of": "2026-10-31T17:02:00+07:00"},
            self.context,
        )

        record = result.records[0]
        self.assertIsInstance(record, DateContextRecord)
        self.assertEqual(record.local_date, "2026-10-31")
        self.assertEqual(record.weekday, "Thứ Bảy")
        self.assertEqual(result.citations, [])

    async def test_calculate_returns_exact_decimal(self) -> None:
        """Calculate money arithmetic without binary floating-point drift."""
        result = await execute_model_tool(
            "calculate",
            {
                "as_of": "2026-10-31T17:02:00+07:00",
                "expression": "7990000 * (1 - 10 / 100)",
            },
            self.context,
        )

        record = result.records[0]
        self.assertIsInstance(record, CalculationRecord)
        self.assertEqual(record.result, Decimal("7191000"))
        self.assertEqual(result.citations, [])

    def test_expanded_policy_section_returns_all_clause_citations(self) -> None:
        """Expose citations for every clause included in the section text."""
        record: PolicyRecord = PolicyRecord(
            source_file="policy_handbook.pdf",
            section_id="3.2",
            parent_section_id="3",
            section_title="3. Bảo hành",
            page=1,
            text="3. Bảo hành\n3.1 ...\n3.2 ...\n3.3 ...",
            section_cite_ids=[
                "policy_handbook.pdf#3.1",
                "policy_handbook.pdf#3.2",
                "policy_handbook.pdf#3.3",
            ],
            document_version="3.2",
            effective_date="2026-10-01",
            cite_id="policy_handbook.pdf#3.2",
        )

        result = _make_result("search_policy", [record])

        self.assertEqual(
            result.citations,
            [
                "policy_handbook.pdf#3.1",
                "policy_handbook.pdf#3.2",
                "policy_handbook.pdf#3.3",
            ],
        )

    def test_warranty_result_includes_sheet_headers_and_footnotes(self) -> None:
        """Return multi-row headers and warranty footer with its data."""
        workbook: SourceWorkbook = SourceWorkbook(
            source_file="product_catalog.xlsx"
        )
        sheet: SourceSheet = SourceSheet(
            id=3,
            name="Bảo hành",
            columns=[
                "Nhóm sản phẩm",
                "Linh kiện chính / Linh kiện",
                "Linh kiện chính / Thời hạn (tháng)",
            ],
            header_notes=[
                {"source_row": 2, "values": {"A": "Chính sách bảo hành"}}
            ],
            footnotes=[
                {"source_row": 12, "values": {"A": "Không áp dụng cho mục đích thương mại"}}
            ],
            workbook=workbook,
        )
        source_row: SpreadsheetRow = SpreadsheetRow(
            source_row=6,
            sheet=sheet,
        )
        warranty: Warranty = Warranty(
            product_group="Điều hòa",
            product_months=24,
            component="Máy nén",
            component_months=120,
            exclusions=None,
            source_row=source_row,
        )
        record: WarrantyRecord = WarrantyRecord(
            product_group=warranty.product_group,
            product_months=warranty.product_months,
            component=warranty.component,
            component_months=warranty.component_months,
            exclusions=warranty.exclusions,
            cite_id=warranty.cite_id,
        )

        result = _make_result("get_warranty", [record], [warranty])

        self.assertEqual(len(result.source_contexts), 1)
        source_context = result.source_contexts[0]
        self.assertEqual(
            source_context.columns,
            [
                "Nhóm sản phẩm",
                "Linh kiện chính / Linh kiện",
                "Linh kiện chính / Thời hạn (tháng)",
            ],
        )
        self.assertEqual(
            source_context.header_notes[0].values["A"],
            "Chính sách bảo hành",
        )
        self.assertEqual(source_context.header_notes[0].source_row, 2)
        self.assertEqual(
            source_context.footnotes[0].values["A"],
            "Không áp dụng cho mục đích thương mại",
        )
        self.assertEqual(source_context.footnotes[0].source_row, 12)
        self.assertIn(
            "product_catalog.xlsx#sheet=B%E1%BA%A3o%20h%C3%A0nh&row=12",
            result.citations,
        )
        self.assertIn(
            "product_catalog.xlsx#sheet=B%E1%BA%A3o%20h%C3%A0nh&row=2",
            result.citations,
        )
        self.assertIn(
            "product_catalog.xlsx#sheet=B%E1%BA%A3o%20h%C3%A0nh",
            result.citations,
        )
        serialized_result: dict[str, object] = result.model_dump(mode="json")
        serialized_contexts: object = serialized_result["source_contexts"]
        self.assertIsInstance(serialized_contexts, list)
        self.assertEqual(serialized_contexts[0]["footnotes"][0]["source_row"], 12)

    def test_calculator_rejects_code_and_division_by_zero(self) -> None:
        """Reject expressions that could execute code or fail arithmetically."""
        with self.assertRaises(ValueError):
            _calculate_expression("__import__('os').system('true')")
        with self.assertRaises(ValueError):
            _calculate_expression("1 / 0")

    def test_calculator_preserves_decimal_literal_precision(self) -> None:
        """Parse decimal literals directly instead of rounding through float."""
        self.assertEqual(
            _calculate_expression("0.123456789123456789 + 0"),
            Decimal("0.123456789123456789"),
        )

    async def test_price_tool_resolves_province_and_returns_provenance(self) -> None:
        """Resolve a province to its price region and preserve row citation."""
        workbook: SourceWorkbook = SourceWorkbook(source_file="catalog.xlsx")
        sheet: SourceSheet = SourceSheet(name="Bảng giá", workbook=workbook)
        source_row: SpreadsheetRow = SpreadsheetRow(source_row=18, sheet=sheet)
        region: Region = Region(
            province="Hà Nội",
            region="Miền Bắc",
            source_row=source_row,
        )
        price: Price = Price(
            model="LH-AC09N",
            region="Miền Bắc",
            price=Decimal("7990000"),
            valid_from=date(2026, 9, 1),
            source_row=source_row,
        )
        with (
            patch(
                "app.tools.model_tools.get_regions",
                new=AsyncMock(return_value=[region]),
            ) as query_regions,
            patch(
                "app.tools.model_tools.get_prices",
                new=AsyncMock(return_value=[price]),
            ) as query_prices,
        ):
            result = await execute_model_tool(
                "quote_price",
                {
                    "as_of": "2026-10-31T17:02:00+07:00",
                    "model": "LH-AC09N",
                    "province": "Hà Nội",
                },
                self.context,
            )

        record = result.records[0]
        self.assertIsInstance(record, PriceRecord)
        self.assertEqual(record.price, Decimal("7990000"))
        self.assertIn(
            "catalog.xlsx#sheet=B%E1%BA%A3ng%20gi%C3%A1&row=18",
            result.citations,
        )
        self.assertIn(
            "catalog.xlsx#sheet=B%E1%BA%A3ng%20gi%C3%A1",
            result.citations,
        )
        self.assertEqual(len(result.source_contexts), 1)
        query_regions.assert_awaited_once()
        query_prices.assert_awaited_once()
        self.assertEqual(
            query_prices.await_args.kwargs["region"],
            "Miền Bắc",
        )

    async def test_unknown_tool_is_explicit(self) -> None:
        """Do not silently treat an unregistered tool name as empty data."""
        with self.assertRaises(UnknownModelToolError):
            await execute_model_tool(
                "free_form_sql",
                {"as_of": datetime(2026, 10, 31)},
                self.context,
            )

    async def test_unknown_model_does_not_return_all_warranty_rows(self) -> None:
        """Return no warranty data for an unknown model instead of broad data."""
        with (
            patch(
                "app.tools.model_tools.get_products",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "app.tools.model_tools.get_warranty",
                new=AsyncMock(),
            ) as query_warranty,
        ):
            result = await execute_model_tool(
                "get_warranty",
                {
                    "as_of": "2026-10-31T17:02:00+07:00",
                    "model": "unknown-model",
                },
                self.context,
            )

        self.assertEqual(result.records, [])
        query_warranty.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
