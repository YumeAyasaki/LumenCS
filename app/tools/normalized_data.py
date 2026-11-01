"""Query normalized spreadsheet tables with citations from raw-row provenance."""

from __future__ import annotations

import fnmatch
import unicodedata
from datetime import date, datetime
from typing import TypeAlias

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import (
    Price,
    Product,
    Promotion,
    Region,
    ServiceCentre,
    ServiceFee,
    SourceSheet,
    SpreadsheetRow,
    Warranty,
)

AsOf: TypeAlias = date | datetime


def _citation_load_options(model: type[object]) -> tuple[object, ...]:
    """Build eager-loading options for a normalized row's provenance chain.

    Args:
        model: Normalized SQLAlchemy entity class.

    Returns:
        Loader options for raw row, source sheet, and source workbook.
    """
    source_row_attribute = getattr(model, "source_row")
    return (
        selectinload(source_row_attribute)
        .selectinload(SpreadsheetRow.sheet)
        .selectinload(SourceSheet.workbook),
    )


def _as_date(value: AsOf) -> date:
    """Reduce an asked-at timestamp to the calendar date used by source data.

    Args:
        value: Date or timestamp from the calling tool.

    Returns:
        Calendar date from the Vietnam-local timestamp normalized at the API
        boundary.
    """
    return value.date() if isinstance(value, datetime) else value


def _normalize_group_text(value: str) -> str:
    """Normalize accents and case for canonical product-group matching.

    Args:
        value: User-provided or source product-group text.

    Returns:
        Accent-free, case-folded text with normalized whitespace.
    """
    decomposed: str = unicodedata.normalize("NFD", value.casefold())
    return " ".join(
        "".join(
            character
            for character in decomposed
            if unicodedata.category(character) != "Mn"
        ).replace("đ", "d").split()
    )


async def resolve_product_group(
    session: AsyncSession,
    product_group: str,
) -> str | None:
    """Resolve user wording to a canonical product group.

    Matching accepts accent/case differences and phrases containing a known
    group, for example ``"máy nén điều hòa"`` resolves to ``"Điều hòa"``.

    Args:
        session: Active asynchronous database session.
        product_group: User-supplied group name or descriptive phrase.

    Returns:
        Canonical product group, or ``None`` when no known group matches.

    Raises:
        SQLAlchemyError: If PostgreSQL cannot execute the query.
    """
    result = await session.scalars(select(Product.product_group).distinct())
    groups: list[str] = sorted(
        set(result.all()),
        key=lambda group: _normalize_group_text(group),
    )
    normalized_query: str = _normalize_group_text(product_group)
    exact_match: str | None = next(
        (
            group
            for group in groups
            if _normalize_group_text(group) == normalized_query
        ),
        None,
    )
    if exact_match is not None:
        return exact_match
    matching_groups: list[str] = [
        group
        for group in groups
        if _normalize_group_text(group) in normalized_query
    ]
    return max(
        matching_groups,
        key=lambda group: len(_normalize_group_text(group)),
        default=None,
    )


async def get_products(
    session: AsyncSession,
    as_of: AsOf,
    model: str | None = None,
    product_group: str | None = None,
) -> list[Product]:
    """Query typed product records with eager-loaded source citations.

    The source catalog has no effective-date columns, so ``as_of`` is accepted
    for a consistent typed-tool interface but does not filter product rows.

    Args:
        session: Active asynchronous database session.
        as_of: Date context supplied by the calling server tool.
        model: Optional exact model code.
        product_group: Optional exact product group.

    Returns:
        Products ordered by model, each with a usable ``cite_id`` property.

    Raises:
        SQLAlchemyError: If PostgreSQL cannot execute the query.
    """
    statement = select(Product).options(*_citation_load_options(Product))
    if model is not None:
        statement = statement.where(Product.model == model)
    if product_group is not None:
        resolved_group: str | None = await resolve_product_group(
            session,
            product_group,
        )
        if resolved_group is None:
            return []
        statement = statement.where(Product.product_group == resolved_group)
    result = await session.scalars(statement.order_by(Product.model))
    return list(result.all())


async def get_prices(
    session: AsyncSession,
    as_of: AsOf,
    model: str | None = None,
    region: str | None = None,
) -> list[Price]:
    """Return the latest model/region prices effective on the requested date.

    Args:
        session: Active asynchronous database session.
        as_of: Inclusive date used to filter ``valid_from``.
        model: Optional exact model code.
        region: Optional exact price region.

    Returns:
        Latest effective price per model and region with source citations.

    Raises:
        SQLAlchemyError: If PostgreSQL cannot execute the query.
    """
    statement = (
        select(Price)
        .options(*_citation_load_options(Price))
        .where(Price.valid_from <= _as_date(as_of))
        .order_by(
            Price.model,
            Price.region,
            Price.valid_from.desc(),
            Price.id.desc(),
        )
    )
    if model is not None:
        statement = statement.where(Price.model == model)
    if region is not None:
        statement = statement.where(Price.region == region)
    result = await session.scalars(statement)
    latest: dict[tuple[str, str], Price] = {}
    for price in result.all():
        latest.setdefault((price.model, price.region), price)
    return list(latest.values())


async def get_promotions(
    session: AsyncSession,
    as_of: AsOf,
    model: str | None = None,
    channel: str | None = None,
) -> list[Promotion]:
    """Return active typed promotions optionally applicable to a product.

    Args:
        session: Active asynchronous database session.
        as_of: Inclusive date used to filter promotion start and end dates.
        model: Optional product model to match against parsed scope patterns.
        channel: Optional sales channel; ``Tất cả`` promotions also match.

    Returns:
        Active promotions with source citations.

    Raises:
        SQLAlchemyError: If PostgreSQL cannot execute the query.
    """
    statement = (
        select(Promotion)
        .options(*_citation_load_options(Promotion))
        .where(
            Promotion.starts_on <= _as_date(as_of),
            Promotion.ends_on >= _as_date(as_of),
        )
        .order_by(Promotion.starts_on, Promotion.code)
    )
    if channel is not None:
        statement = statement.where(
            Promotion.channel.in_([channel, "Tất cả"])
        )
    result = await session.scalars(statement)
    promotions: list[Promotion] = list(result.all())
    if model is None:
        return promotions
    return [
        promotion
        for promotion in promotions
        if any(
            fnmatch.fnmatchcase(model, pattern)
            for pattern in promotion.scope.get("include", [])
        )
        and not any(
            fnmatch.fnmatchcase(model, pattern)
            for pattern in promotion.scope.get("exclude", [])
        )
    ]


async def get_promotions_for_model(
    session: AsyncSession,
    model: str,
) -> list[Promotion]:
    """Return all dated promotions whose parsed scope mentions a model.

    Unlike :func:`get_promotions`, this deliberately includes expired and
    future programs so a price quote can explain why each one is inapplicable.

    Args:
        session: Active asynchronous database session.
        model: Exact product model code.

    Returns:
        Promotions whose included or excluded patterns match the model.

    Raises:
        SQLAlchemyError: If PostgreSQL cannot execute the query.
    """
    result = await session.scalars(
        select(Promotion)
        .options(*_citation_load_options(Promotion))
        .order_by(Promotion.starts_on, Promotion.code)
    )
    return [
        promotion
        for promotion in result.all()
        if any(
            fnmatch.fnmatchcase(model, pattern)
            for pattern in [
                *promotion.scope.get("include", []),
                *promotion.scope.get("exclude", []),
            ]
        )
    ]


async def get_warranty(
    session: AsyncSession,
    as_of: AsOf,
    product_group: str | None = None,
) -> list[Warranty]:
    """Query typed warranty records by product group.

    The source warranty sheet has no effective-date columns; ``as_of`` is
    accepted for a consistent tool interface but does not filter its records.

    Args:
        session: Active asynchronous database session.
        as_of: Date context supplied by the calling server tool.
        product_group: Optional exact product group.

    Returns:
        Warranty records ordered by product group with source citations.

    Raises:
        SQLAlchemyError: If PostgreSQL cannot execute the query.
    """
    statement = select(Warranty).options(*_citation_load_options(Warranty))
    if product_group is not None:
        resolved_group: str | None = await resolve_product_group(
            session,
            product_group,
        )
        if resolved_group is None:
            return []
        statement = statement.where(Warranty.product_group == resolved_group)
    result = await session.scalars(statement.order_by(Warranty.product_group))
    return list(result.all())


async def get_regions(
    session: AsyncSession,
    as_of: AsOf,
    province: str | None = None,
) -> list[Region]:
    """Query typed province-to-price-region mappings.

    The source region sheet has no effective-date columns; ``as_of`` is
    accepted for a consistent tool interface but does not filter its records.

    Args:
        session: Active asynchronous database session.
        as_of: Date context supplied by the calling server tool.
        province: Optional exact province or city.

    Returns:
        Region records ordered by province with source citations.

    Raises:
        SQLAlchemyError: If PostgreSQL cannot execute the query.
    """
    statement = select(Region).options(*_citation_load_options(Region))
    if province is not None:
        statement = statement.where(Region.province == province)
    result = await session.scalars(statement.order_by(Region.province))
    return list(result.all())


async def find_service_centres(
    session: AsyncSession,
    as_of: AsOf,
    province: str | None = None,
    product_group: str | None = None,
    district: str | None = None,
) -> list[ServiceCentre]:
    """Query service centres, optionally filtered by location and capability.

    The source service-centre sheet has no effective-date columns; ``as_of`` is
    accepted for a consistent tool interface but does not filter its records.

    Args:
        session: Active asynchronous database session.
        as_of: Date context supplied by the calling server tool.
        province: Optional exact province or city.
        product_group: Optional product group capability to require.
        district: Optional exact district within the province.

    Returns:
        Matching centres with source citations.

    Raises:
        SQLAlchemyError: If PostgreSQL cannot execute the query.
    """
    resolved_group: str | None = None
    if product_group is not None:
        resolved_group = await resolve_product_group(session, product_group)
        if resolved_group is None:
            return []
    statement = select(ServiceCentre).options(
        *_citation_load_options(ServiceCentre)
    )
    if province is not None:
        statement = statement.where(ServiceCentre.province == province)
    if district is not None:
        statement = statement.where(ServiceCentre.district == district)
    result = await session.scalars(statement.order_by(ServiceCentre.code))
    centres: list[ServiceCentre] = list(result.all())
    if resolved_group is None:
        return centres
    return [
        centre
        for centre in centres
        if centre.accepted_services.get(resolved_group, False)
    ]


async def get_service_fees(
    session: AsyncSession,
    as_of: AsOf,
    service_code: str | None = None,
    product_group: str | None = None,
    service_query: str | None = None,
) -> list[ServiceFee]:
    """Query typed service-fee records with source citations.

    The source service-fee sheet has no per-row validity dates; ``as_of`` is
    accepted for a consistent tool interface but does not filter its records.

    Args:
        session: Active asynchronous database session.
        as_of: Date context supplied by the calling server tool.
        service_code: Optional exact service code.
        product_group: Optional product group in the applicability list.
        service_query: Optional case-insensitive substring of service name.

    Returns:
        Matching service fees with source citations.

    Raises:
        SQLAlchemyError: If PostgreSQL cannot execute the query.
    """
    resolved_group: str | None = None
    if product_group is not None:
        resolved_group = await resolve_product_group(session, product_group)
        if resolved_group is None:
            return []
    statement = select(ServiceFee).options(*_citation_load_options(ServiceFee))
    if service_code is not None:
        statement = statement.where(ServiceFee.code == service_code)
    if service_query is not None:
        escaped_query: str = (
            service_query.replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        statement = statement.where(
            ServiceFee.service.ilike(f"%{escaped_query}%", escape="\\")
        )
    result = await session.scalars(statement.order_by(ServiceFee.code))
    fees: list[ServiceFee] = list(result.all())
    if resolved_group is None:
        return fees
    return [
        fee
        for fee in fees
        if "Tất cả" in fee.applies_to or resolved_group in fee.applies_to
    ]
