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
    Product,
    Promotion,
    Region,
    SourceSheet,
    SourceWorkbook,
    SpreadsheetRow,
    Warranty,
)
from app.tools.model_tools import (
    AddDaysRecord,
    CalculationRecord,
    ModelToolContext,
    PolicyRecord,
    PriceRecord,
    UnknownModelToolError,
    WarrantyRecord,
    _calculate_expression,
    _make_result,
    _queried_worksheet_notes,
    _quote_price_record,
    execute_model_tool,
    get_model_tool_definitions,
)
from app.tools.normalized_data import resolve_product_group


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

    def test_definitions_hide_internal_date_and_schema_titles(self) -> None:
        """Expose concise schemas without server-owned timestamp arguments."""
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
                "find_service_centres",
                "get_service_fee",
                "search_policy",
                "calculate",
                "add_days",
            ],
        )
        for definition in definitions:
            self.assertNotIn("as_of", definition.function.parameters["properties"])
            self.assertNotIn("as_of", definition.function.parameters.get("required", []))
            self.assertNotIn("title", str(definition.function.parameters))

    async def test_add_days_returns_date_and_vietnamese_weekday(self) -> None:
        """Calculate a positive date offset and identify its weekday."""
        result = await execute_model_tool(
            "add_days",
            {
                "as_of": "2026-10-31T17:02:00+07:00",
                "start_date": "2026-10-22",
                "days": 7,
            },
            self.context,
        )

        record = result.records[0]
        self.assertIsInstance(record, AddDaysRecord)
        self.assertEqual(record.result_date, date(2026, 10, 29))
        self.assertEqual(record.weekday, "Thứ Năm")
        self.assertEqual(result.citations, [])

    async def test_calculate_returns_exact_decimal(self) -> None:
        """Calculate money arithmetic without binary floating-point drift."""
        result = await execute_model_tool(
            "calculate",
            {
                "as_of": "2026-10-31T17:02:00+07:00",
                "expression": "7990000 * (1 - 10 / 100)",
                "unknown_argument": "ignored",
            },
            self.context,
        )

        record = result.records[0]
        self.assertIsInstance(record, CalculationRecord)
        self.assertEqual(record.result, Decimal("7191000"))
        self.assertEqual(result.citations, [])

    def test_quote_price_computes_capped_discount_and_marks_channel_unknown(self) -> None:
        """Return promo math, conditional channel eligibility, and product status."""
        workbook: SourceWorkbook = SourceWorkbook(
            source_file="product_catalog.xlsx"
        )
        sheet: SourceSheet = SourceSheet(
            id=4,
            name="Khuyến mãi",
            columns=[],
            header_notes=[],
            footnotes=[],
            workbook=workbook,
        )

        def promo(
            row_number: int,
            code: str,
            channel: str | None,
            starts: date,
            ends: date,
            scope: dict[str, list[str]],
            promotion_type: str = "percentage_discount",
            value: Decimal | None = Decimal("10"),
            cap: Decimal | None = Decimal("2000000"),
            conditions: str | None = None,
        ) -> Promotion:
            """Create a cited promotion fixture."""
            return Promotion(
                code=code,
                name=f"Chương trình {code}",
                starts_on=starts,
                ends_on=ends,
                scope_text="fixture",
                scope=scope,
                channel=channel,
                promotion_type=promotion_type,
                value_amount=value,
                value_unit="percent",
                value_details=None,
                cap_amount=cap,
                conditions=conditions,
                can_stack=False,
                stacking_text="Không",
                source_row=SpreadsheetRow(
                    source_row=row_number,
                    sheet=sheet,
                ),
            )

        product_sheet: SourceSheet = SourceSheet(
            id=5,
            name="Danh mục SP",
            columns=[],
            header_notes=[],
            footnotes=[],
            workbook=workbook,
        )
        product: Product = Product(
            model="LH-AC09S",
            name="Điều hòa cũ",
            product_group="Điều hòa",
            specification=None,
            energy_rating=None,
            dimensions=None,
            weight=None,
            status="Ngừng kinh doanh",
            notes="Model thay thế: LH-AC09N",
            source_row=SpreadsheetRow(source_row=10, sheet=product_sheet),
        )
        price_sheet: SourceSheet = SourceSheet(
            id=6,
            name="Bảng giá",
            columns=[],
            header_notes=[],
            footnotes=[],
            workbook=workbook,
        )
        price: Price = Price(
            model="LH-AC09S",
            region="Miền Bắc",
            price=Decimal("30000000"),
            valid_from=date(2026, 1, 1),
            source_row=SpreadsheetRow(source_row=9, sheet=price_sheet),
        )
        promos: list[Promotion] = [
            promo(
                11,
                "KM-ONLINE",
                "Online",
                date(2026, 10, 1),
                date(2026, 10, 31),
                {"include": ["LH-AC*"], "exclude": []},
            ),
            promo(
                12,
                "KM-ENDED",
                "Tất cả",
                date(2026, 10, 1),
                date(2026, 10, 20),
                {"include": ["LH-AC*"], "exclude": []},
            ),
            promo(
                13,
                "KM-EXCLUDED",
                "Tất cả",
                date(2026, 10, 1),
                date(2026, 10, 31),
                {"include": ["LH-AC*"], "exclude": ["LH-AC09S"]},
            ),
            promo(
                14,
                "KM-TRAGOP",
                "Tất cả",
                date(2026, 10, 1),
                date(2026, 10, 31),
                {"include": ["*"], "exclude": []},
                promotion_type="installment",
                value=Decimal("0"),
                cap=None,
                conditions="Đơn hàng ≥ 5.000.000 đ; thẻ tín dụng ngân hàng đối tác",
            ),
        ]

        record: PriceRecord = _quote_price_record(
            model=product.model,
            price=price,
            product=product,
            region=price.region,
            promotions=promos,
            as_of=datetime(2026, 10, 25, 12, 0),
            channel=None,
        )

        self.assertEqual(len(record.promotions), 1)
        self.assertEqual(record.promotions[0].discount_amount, Decimal("2000000"))
        self.assertEqual(
            record.promotions[0].price_after_discount,
            Decimal("28000000"),
        )
        self.assertTrue(record.promotions[0].channel_conditional)
        self.assertEqual(record.product_status, "Ngừng kinh doanh")
        self.assertIn("LH-AC09N", record.product_notes or "")
        self.assertEqual(
            [item.reason for item in record.not_applicable],
            ["ended 20/10/2026", "excluded: LH-AC09S"],
        )
        self.assertIn("5.000.000", record.installment_note or "")
        self.assertEqual(len(record.installment_cite_ids), 1)
        tool_result = _make_result("quote_price", [record])
        self.assertIn(record.installment_cite_ids[0], tool_result.citations)

    async def test_group_resolver_matches_accentless_descriptive_phrase(self) -> None:
        """Resolve a component phrase to its canonical product group."""
        scalars: MagicMock = MagicMock()
        scalars.all.return_value = ["Điều hòa", "Máy giặt", "Tủ lạnh"]
        self.session.scalars = AsyncMock(return_value=scalars)

        result: str | None = await resolve_product_group(
            self.session,
            "MAY NEN dieu hoa",
        )

        self.assertEqual(result, "Điều hòa")

    async def test_queried_sheet_notes_are_available_for_empty_results(self) -> None:
        """Fetch source notes from configured sheets without returned data rows."""
        workbook: SourceWorkbook = SourceWorkbook(
            source_file="product_catalog.xlsx"
        )
        sheet: SourceSheet = SourceSheet(
            id=8,
            name="Vùng giá",
            columns=["Tỉnh/Thành phố", "Vùng giá"],
            header_notes=[
                {"source_row": 1, "values": {"A": "PHÂN VÙNG GIÁ"}},
            ],
            footnotes=[
                {
                    "source_row": 14,
                    "values": {
                        "A": "Tỉnh/thành không có trong danh sách: liên hệ hotline."
                    },
                }
            ],
            workbook=workbook,
        )
        scalars: MagicMock = MagicMock()
        scalars.all.return_value = [sheet]
        self.session.scalars = AsyncMock(return_value=scalars)

        notes: list[tuple[str, str]] = await _queried_worksheet_notes(
            self.session,
            "quote_price",
        )

        self.assertEqual(len(notes), 1)
        self.assertIn("liên hệ hotline", notes[0][1])
        self.assertNotIn("PHÂN VÙNG", notes[0][1])
        self.assertIn(notes[0][0], _make_result(
            "quote_price",
            [],
            source_notes=notes,
        ).citations)

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

    def test_warranty_result_includes_compact_cited_notes_only(self) -> None:
        """Return cited explanatory notes, without title rows or table headers."""
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
                {"source_row": 1, "values": {"A": "THỜI HẠN BẢO HÀNH"}},
                {"source_row": 2, "values": {"A": "Áp dụng cho hộ gia đình."}},
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

        self.assertEqual(len(result.notes), 2)
        self.assertTrue(result.notes[0].startswith("[product_catalog.xlsx#sheet="))
        self.assertIn("Áp dụng cho hộ gia đình.", result.notes[0])
        self.assertIn("Không áp dụng cho mục đích thương mại", result.notes[1])
        self.assertNotIn("THỜI HẠN BẢO HÀNH", str(result.notes))
        self.assertTrue(
            all(cite in result.citations for cite in (
                "product_catalog.xlsx#sheet=B%E1%BA%A3o%20h%C3%A0nh&row=2",
                "product_catalog.xlsx#sheet=B%E1%BA%A3o%20h%C3%A0nh&row=12",
            ))
        )
        serialized_result: dict[str, object] = result.model_dump(mode="json")
        self.assertNotIn("source_contexts", serialized_result)
        self.assertEqual(serialized_result["notes"], result.notes)

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
                "app.tools.model_tools.get_products",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "app.tools.model_tools.get_regions",
                new=AsyncMock(return_value=[region]),
            ) as query_regions,
            patch(
                "app.tools.model_tools.get_prices",
                new=AsyncMock(return_value=[price]),
            ) as query_prices,
            patch(
                "app.tools.model_tools.get_promotions_for_model",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "app.tools.model_tools._queried_worksheet_notes",
                new=AsyncMock(return_value=[]),
            ),
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
            patch(
                "app.tools.model_tools._queried_worksheet_notes",
                new=AsyncMock(return_value=[]),
            ),
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
