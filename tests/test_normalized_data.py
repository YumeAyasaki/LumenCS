"""Tests for typed-table query tools."""

from __future__ import annotations

import unittest
from datetime import date
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy.ext.asyncio import AsyncSession

from app.database import Price, Promotion
from app.tools.normalized_data import (
    find_service_centres,
    get_prices,
    get_promotions,
    get_service_fees,
)


class NormalizedDataTests(unittest.IsolatedAsyncioTestCase):
    """Verify typed lookups filter dates and promotion scopes."""

    def setUp(self) -> None:
        """Create a session double that records typed SQL statements."""
        self.session: AsyncMock = AsyncMock(spec=AsyncSession)
        self.scalar_result: MagicMock = MagicMock()
        self.scalar_result.all.return_value = []
        self.session.scalars.return_value = self.scalar_result

    async def test_get_prices_uses_only_typed_price_table(self) -> None:
        """Choose the latest price effective on the requested date."""
        newer_price: Price = Price(
            id=2,
            source_row_id=20,
            model="LH-AC09N",
            region="Miền Bắc",
            price=Decimal("7990000"),
            valid_from=date(2026, 9, 1),
        )
        older_price: Price = Price(
            id=1,
            source_row_id=10,
            model="LH-AC09N",
            region="Miền Bắc",
            price=Decimal("8490000"),
            valid_from=date(2026, 1, 1),
        )
        self.scalar_result.all.return_value = [newer_price, older_price]

        prices: list[Price] = await get_prices(
            self.session,
            date(2026, 10, 28),
            model="LH-AC09N",
            region="Miền Bắc",
        )

        self.assertEqual(prices, [newer_price])
        statement = self.session.scalars.await_args.args[0]
        sql: str = str(statement.compile(compile_kwargs={"literal_binds": True}))
        self.assertIn("FROM prices", sql)
        self.assertNotIn("spreadsheet_rows", sql)
        self.assertIn("valid_from", sql)

    async def test_get_promotions_filters_by_date_and_model_scope(self) -> None:
        """Return active promotions whose parsed scope includes the model."""
        matching: Promotion = Promotion(
            id=1,
            source_row_id=5,
            code="KM-2610",
            name="October promotion",
            starts_on=date(2026, 10, 1),
            ends_on=date(2026, 10, 31),
            scope_text="LH-AC* except LH-AC09S",
            scope={"include": ["LH-AC*"], "exclude": ["LH-AC09S"]},
            promotion_type="percentage_discount",
            can_stack=True,
        )
        self.scalar_result.all.return_value = [matching]

        promotions: list[Promotion] = await get_promotions(
            self.session,
            date(2026, 10, 28),
            model="LH-AC12N",
        )

        self.assertEqual(promotions, [matching])
        statement = self.session.scalars.await_args.args[0]
        sql: str = str(statement.compile(compile_kwargs={"literal_binds": True}))
        self.assertIn("FROM promotions", sql)
        self.assertNotIn("spreadsheet_rows", sql)
        self.assertIn("starts_on", sql)
        self.assertIn("ends_on", sql)

        self.scalar_result.all.return_value = [matching]
        excluded: list[Promotion] = await get_promotions(
            self.session,
            date(2026, 10, 28),
            model="LH-AC09S",
        )
        self.assertEqual(excluded, [])

    async def test_service_centre_filters_by_district(self) -> None:
        """Filter centres by both province and district when requested."""
        await find_service_centres(
            self.session,
            date(2026, 10, 28),
            province="Hà Nội",
            district="Cầu Giấy",
        )

        statement = self.session.scalars.await_args.args[0]
        sql: str = str(statement.compile(compile_kwargs={"literal_binds": True}))
        self.assertIn("service_centres.province = 'Hà Nội'", sql)
        self.assertIn("service_centres.district = 'Cầu Giấy'", sql)

    async def test_service_fee_filters_by_name_and_product_group(self) -> None:
        """Support user-friendly fee-name search without unsafe SQL."""
        await get_service_fees(
            self.session,
            date(2026, 10, 28),
            product_group="Điều hòa",
            service_query="vệ sinh%",
        )

        statement = self.session.scalars.await_args.args[0]
        sql: str = str(statement.compile(compile_kwargs={"literal_binds": True}))
        self.assertIn("lower(service_fees.service) LIKE lower", sql)
        self.assertIn("ESCAPE", sql)


if __name__ == "__main__":
    unittest.main()
