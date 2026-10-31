"""Convert persisted raw spreadsheet rows into typed domain records."""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import TypeAlias

from app.database import (
    Price,
    Product,
    Promotion,
    Region,
    ServiceCentre,
    ServiceFee,
    SpreadsheetRow,
    Warranty,
)
from app.tools.spreadsheet_reader import (
    NormalizationField,
    NormalizationMapping,
)

NormalizedRecord: TypeAlias = (
    Product | Price | Promotion | Region | ServiceCentre | ServiceFee | Warranty
)
NORMALIZED_MODELS: dict[str, type[NormalizedRecord]] = {
    "products": Product,
    "prices": Price,
    "promotions": Promotion,
    "warranty": Warranty,
    "regions": Region,
    "service_centres": ServiceCentre,
    "service_fees": ServiceFee,
}

NormalizationOptions: TypeAlias = dict[
    str,
    str | list[str] | dict[str, str],
]


def _option_text(
    options: NormalizationOptions,
    key: str,
    default: str,
) -> str:
    """Read a string parser option or its default.

    Args:
        options: Options configured in the schema mapping.
        key: Option name.
        default: Value used when the option is not a string.

    Returns:
        Configured string option or the default.
    """
    value: str | list[str] | dict[str, str] | None = options.get(key)
    return value if isinstance(value, str) else default


def _option_strings(
    options: NormalizationOptions,
    key: str,
    default: list[str],
) -> list[str]:
    """Read a list-of-strings parser option or its default.

    Args:
        options: Options configured in the schema mapping.
        key: Option name.
        default: Value used when the option is absent or not a string list.

    Returns:
        Configured string list or the default.
    """
    value: str | list[str] | dict[str, str] | None = options.get(key)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    return default


def _option_mapping(
    options: NormalizationOptions,
    key: str,
    default: dict[str, str],
) -> dict[str, str]:
    """Read a string mapping parser option or its default.

    Args:
        options: Options configured in the schema mapping.
        key: Option name.
        default: Value used when the option is absent or not a string mapping.

    Returns:
        Configured string mapping or the default.
    """
    value: str | list[str] | dict[str, str] | None = options.get(key)
    return value if isinstance(value, dict) else default


def _number_pattern(options: NormalizationOptions) -> re.Pattern[str]:
    """Build the numeric cell pattern from configured locale separators.

    Args:
        options: Parser options for thousands and decimal separators.

    Returns:
        Compiled pattern matching one localized numeric amount.
    """
    thousands_separator: str = re.escape(
        _option_text(options, "thousands_separator", ".")
    )
    decimal_separator: str = re.escape(
        _option_text(options, "decimal_separator", ",")
    )
    return re.compile(
        r"(?<![\d-])\d{1,3}(?:"
        + thousands_separator
        + r"\d{3})+(?:"
        + decimal_separator
        + r"\d+)?|\d+(?:"
        + decimal_separator
        + r"\d+)?"
    )


def _decimal(
    value: object,
    source: str,
    options: NormalizationOptions | None = None,
) -> Decimal:
    """Parse a spreadsheet number into Decimal without binary rounding.

    Args:
        value: Source cell content.
        source: Provenance label used in parse errors.

    Returns:
        Parsed decimal value.

    Raises:
        ValueError: If the value does not contain one unambiguous number.
    """
    if isinstance(value, bool) or value is None:
        raise ValueError(f"Expected a number at {source}; got {value!r}")
    if isinstance(value, (int, float, Decimal)):
        return Decimal(str(value))
    parse_options: NormalizationOptions = options or {}
    thousands_separator: str = _option_text(
        parse_options,
        "thousands_separator",
        ".",
    )
    decimal_separator: str = _option_text(
        parse_options,
        "decimal_separator",
        ",",
    )
    match: re.Match[str] | None = _number_pattern(parse_options).search(str(value))
    if match is None:
        raise ValueError(f"Expected a number at {source}; got {value!r}")
    normalized: str = match.group().replace(thousands_separator, "")
    normalized = normalized.replace(decimal_separator, ".")
    try:
        return Decimal(normalized)
    except InvalidOperation as error:
        raise ValueError(f"Invalid number at {source}: {value!r}") from error


def _maybe_decimal(
    value: object,
    source: str,
    options: NormalizationOptions | None = None,
) -> Decimal | None:
    """Parse a numeric cell, returning ``None`` for text-only fee values.

    Args:
        value: Source cell content.
        source: Provenance label used in parse errors.

    Returns:
        Parsed decimal, or ``None`` when the cell is blank or non-numeric.

    Raises:
        ValueError: If a numeric-looking value is malformed.
    """
    parse_options: NormalizationOptions = options or {}
    if value is None or not _number_pattern(parse_options).search(str(value)):
        return None
    return _decimal(value, source, parse_options)


def _parse_date(
    value: object,
    source: str,
    options: NormalizationOptions,
) -> date:
    """Parse a date from an Excel date cell or configured D/M/Y text.

    Args:
        value: Source cell content.
        source: Provenance label used in parse errors.

    Returns:
        A Python date.

    Raises:
        ValueError: If the value is not a supported date.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        for date_format in _option_strings(
            options,
            "formats",
            ["%d/%m/%Y", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"],
        ):
            try:
                return datetime.strptime(value, date_format).date()
            except ValueError:
                continue
    raise ValueError(f"Invalid date at {source}: {value!r}")


def _product_scope(
    value: object,
    source: str,
    options: NormalizationOptions,
) -> dict[str, list[str]]:
    """Parse included and excluded model globs from a promotion scope.

    Args:
        value: Scope text from the promotion sheet.
        source: Provenance label used in parse errors.

    Returns:
        Scope with ``include`` and ``exclude`` model-pattern lists.

    Raises:
        ValueError: If no product pattern or universal scope can be determined.
    """
    scope_text: str = str(value or "").strip()
    if not scope_text:
        raise ValueError(f"Promotion scope is empty at {source}")
    exclude_marker: str = _option_text(options, "exclude_marker", "trừ")
    sections: list[str] = re.split(
        re.escape(exclude_marker),
        scope_text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )
    include_text: str = sections[0]
    exclude_text: str = sections[1] if len(sections) > 1 else ""
    product_pattern: str = _option_text(
        options,
        "product_pattern",
        r"(?i)\bLH-[A-Z0-9]*\*?(?![A-Z0-9])",
    )
    product_matcher: re.Pattern[str] = re.compile(product_pattern, re.IGNORECASE)
    includes: list[str] = list(dict.fromkeys(product_matcher.findall(include_text)))
    excludes: list[str] = list(dict.fromkeys(product_matcher.findall(exclude_text)))
    universal_marker: str = _option_text(options, "universal_marker", "tất cả")
    if re.search(re.escape(universal_marker), include_text, re.IGNORECASE):
        includes = ["*"]
    if not includes:
        raise ValueError(f"Cannot parse included product scope at {source}: {value!r}")
    return {"include": includes, "exclude": excludes}


def _promotion_type(
    value: object,
    options: NormalizationOptions,
) -> str:
    """Map promotion form text to a stable typed category.

    Args:
        value: Promotion form from the source sheet.

    Returns:
        One of ``percentage_discount``, ``cash_discount``, ``installment``,
        ``gift``, or ``other``.
    """
    text: str = str(value or "").casefold()
    type_rules: dict[str, str] = _option_mapping(
        options,
        "contains",
        {
            "trả góp": "installment",
            "giảm %": "percentage_discount",
            "giảm phần trăm": "percentage_discount",
            "giảm tiền": "cash_discount",
            "quà tặng": "gift",
        },
    )
    for phrase, promotion_type in type_rules.items():
        if phrase.casefold() in text:
            return promotion_type
    return "other"


def _promotion_amount(
    value: object,
    context: object,
    source: str,
    options: NormalizationOptions,
) -> Decimal | None:
    """Parse the promotion value using its form as a hint.

    Args:
        value: Promotion value cell.
        context: Promotion form cell.
        source: Provenance label used in parse errors.

    Returns:
        Percentage, cash amount, gift value, or installment percentage.
    """
    value_text: str = str(value or "")
    form_text: str = str(context or "")
    percent_marker: str = _option_text(options, "percent_marker", "%")
    number_pattern: re.Pattern[str] = _number_pattern(options)
    if percent_marker in value_text:
        percent_match: re.Match[str] | None = re.search(
            r"(\d+(?:[.,]\d+)?)\s*" + re.escape(percent_marker),
            value_text,
        )
        if percent_match is not None:
            return _decimal(percent_match.group(1), source, options)
    installment_keyword: str = _option_text(
        options,
        "installment_keyword",
        "trả góp",
    )
    if installment_keyword.casefold() in form_text.casefold():
        percent_match = re.search(
            r"(\d+(?:[.,]\d+)?)\s*" + re.escape(percent_marker),
            form_text,
        )
        if percent_match is not None:
            return _decimal(percent_match.group(1), source, options)
    gift_value_marker: str = _option_text(options, "gift_value_marker", "trị giá")
    gift_value_match: re.Match[str] | None = re.search(
        re.escape(gift_value_marker)
        + r"\s*("
        + number_pattern.pattern
        + r")",
        value_text,
        re.IGNORECASE,
    )
    if gift_value_match is not None:
        return _decimal(gift_value_match.group(1), source, options)
    return _maybe_decimal(value, source, options)


def _parse_field(
    field: NormalizationField,
    row: dict[str, object],
    source: str,
) -> object:
    """Apply one configured parser to its raw cell value.

    Args:
        field: Reviewed source-column and parser configuration.
        row: Raw spreadsheet row keyed by its source headers.
        source: File, sheet, and Excel row provenance string.

    Returns:
        The parsed typed value.

    Raises:
        KeyError: If a required source column is missing from the raw row.
        ValueError: If a configured value cannot be parsed.
    """
    if field.parser == "boolean_columns":
        if field.columns is None:
            raise ValueError(f"boolean_columns parser has no columns at {source}")
        true_marker: str = _option_text(field.options, "true_marker", "✓")
        return {
            key: row.get(column) == true_marker
            for key, column in field.columns.items()
        }
    if field.column not in row:
        if field.required:
            raise KeyError(f"Required column {field.column!r} is missing at {source}")
        return None
    value: object = row[field.column]
    if value is None or (isinstance(value, str) and not value.strip()):
        if field.required:
            raise ValueError(f"Required value {field.column!r} is empty at {source}")
        return None
    if field.parser == "text":
        return str(value).strip()
    if field.parser == "decimal":
        return _decimal(value, source, field.options)
    if field.parser == "decimal_or_null":
        return _maybe_decimal(value, source, field.options)
    if field.parser == "non_numeric_text":
        return (
            None
            if _maybe_decimal(value, source, field.options) is not None
            else str(value).strip()
        )
    if field.parser == "integer":
        return int(_decimal(value, source, field.options))
    if field.parser == "date_dmy":
        return _parse_date(value, source, field.options)
    if field.parser == "months":
        month_pattern: str = _option_text(
            field.options,
            "pattern",
            r"\d+",
        )
        month_match: re.Match[str] | None = re.search(month_pattern, str(value))
        if month_match is None:
            raise ValueError(f"Cannot parse month count at {source}: {value!r}")
        return int(month_match.group())
    if field.parser == "comma_list":
        delimiter: str = _option_text(field.options, "delimiter", ",")
        return [
            part.strip()
            for part in str(value).split(delimiter)
            if part.strip()
        ]
    if field.parser == "product_scope":
        return _product_scope(value, source, field.options)
    if field.parser == "promotion_type":
        return _promotion_type(value, field.options)
    if field.parser == "promotion_amount":
        context: object = row.get(field.context_column or "")
        return _promotion_amount(value, context, source, field.options)
    if field.parser == "promotion_unit":
        text: str = str(value).casefold()
        unit_rules: dict[str, str] = _option_mapping(
            field.options,
            "contains",
            {
                "trả góp": "percent",
                "giảm %": "percent",
                "giảm tiền": "VND",
                "quà tặng": "VND",
            },
        )
        for phrase, unit in unit_rules.items():
            if phrase.casefold() in text:
                return unit
        return None
    if field.parser == "promotion_cap":
        cap_keyword: str = _option_text(field.options, "cap_keyword", "tối đa")
        match: re.Match[str] | None = re.search(
            re.escape(cap_keyword)
            + r"\s*("
            + _number_pattern(field.options).pattern
            + r")",
            str(value),
            re.IGNORECASE,
        )
        return (
            _decimal(match.group(1), source, field.options)
            if match is not None
            else None
        )
    if field.parser == "stacking":
        normalized_value: str = str(value).strip().casefold()
        true_prefix: str = _option_text(field.options, "true_prefix", "có")
        false_prefix: str = _option_text(field.options, "false_prefix", "không")
        if normalized_value.startswith(true_prefix.casefold()):
            return True
        if normalized_value.startswith(false_prefix.casefold()):
            return False
        raise ValueError(f"Cannot parse stacking flag at {source}: {value!r}")
    raise ValueError(f"Unsupported normalization parser {field.parser!r} at {source}")


def normalize_source_row(
    raw_row: SpreadsheetRow,
    mapping: NormalizationMapping,
    source: str,
) -> list[NormalizedRecord]:
    """Build typed ORM rows that reference the persisted raw spreadsheet row.

    Args:
        raw_row: Persisted raw row with a database ID and source values.
        mapping: Reviewable typed-table and parser configuration.
        source: Human-readable source file/sheet/Excel row.

    Returns:
        One or more typed SQLAlchemy models linked to ``raw_row.id``.

    Raises:
        KeyError: If required raw columns are missing.
        ValueError: If required values or parser rules are invalid.
    """
    model: type[NormalizedRecord] = NORMALIZED_MODELS[mapping.table]
    raw_values: dict[str, object] = raw_row.values_json
    parsed_fields: dict[str, object] = {
        target: _parse_field(field, raw_values, source)
        for target, field in mapping.fields.items()
    }
    values: list[dict[str, object]] = []
    if mapping.expand is None:
        values.append(parsed_fields)
    else:
        expansion = mapping.expand
        for dimension, source_column in expansion.columns.items():
            raw_value: object = raw_values.get(source_column)
            if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
                continue
            expanded: dict[str, object] = dict(parsed_fields)
            expanded[expansion.dimension_field] = dimension
            expanded[expansion.value_field] = _parse_field(
                NormalizationField(
                    column=source_column,
                    parser=expansion.parser,
                    options=expansion.options,
                ),
                raw_values,
                source,
            )
            values.append(expanded)
    return [
        model(source_row_id=raw_row.id, **value_fields)
        for value_fields in values
    ]
