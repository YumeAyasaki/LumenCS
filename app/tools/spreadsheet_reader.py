"""Read configured Excel workbooks into provenance-aware server data models."""

from __future__ import annotations

import fnmatch
from datetime import date, datetime, time
from pathlib import Path
from typing import Literal, TypeAlias

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic import ValidationError

DEFAULT_MAPPING_PATH: Path = Path(__file__).parents[1] / "schema_mapping.json"
ExcelValue: TypeAlias = str | int | float | bool | None
NormalizedTable: TypeAlias = Literal[
    "products",
    "prices",
    "promotions",
    "warranty",
    "regions",
    "service_centres",
    "service_fees",
]
NormalizationParser: TypeAlias = Literal[
    "text",
    "decimal",
    "decimal_or_null",
    "non_numeric_text",
    "integer",
    "date_dmy",
    "months",
    "comma_list",
    "product_scope",
    "promotion_type",
    "promotion_amount",
    "promotion_unit",
    "promotion_cap",
    "stacking",
    "boolean_columns",
]

NORMALIZED_FIELDS: dict[str, set[str]] = {
    "products": {
        "model",
        "name",
        "product_group",
        "specification",
        "energy_rating",
        "dimensions",
        "weight",
        "status",
        "notes",
    },
    "prices": {"model", "valid_from", "region", "price"},
    "promotions": {
        "code",
        "name",
        "starts_on",
        "ends_on",
        "scope_text",
        "scope",
        "channel",
        "promotion_type",
        "value_amount",
        "value_unit",
        "value_details",
        "cap_amount",
        "conditions",
        "can_stack",
        "stacking_text",
    },
    "warranty": {
        "product_group",
        "product_months",
        "component",
        "component_months",
        "exclusions",
    },
    "regions": {"province", "region"},
    "service_centres": {
        "code",
        "name",
        "province",
        "district",
        "address",
        "phone",
        "weekday_hours",
        "sunday_hours",
        "accepted_services",
        "status",
        "notes",
    },
    "service_fees": {
        "code",
        "service",
        "applies_to",
        "unit",
        "fee_in_warranty",
        "fee_out_of_warranty",
        "fee_in_warranty_note",
        "fee_out_of_warranty_note",
        "notes",
    },
}
NORMALIZED_FIELD_PARSERS: dict[str, dict[str, set[str]]] = {
    "products": {
        field: {"text"}
        for field in NORMALIZED_FIELDS["products"]
    },
    "prices": {
        "model": {"text"},
        "valid_from": {"date_dmy"},
    },
    "promotions": {
        "code": {"text"},
        "name": {"text"},
        "starts_on": {"date_dmy"},
        "ends_on": {"date_dmy"},
        "scope_text": {"text"},
        "scope": {"product_scope"},
        "channel": {"text"},
        "promotion_type": {"promotion_type"},
        "value_amount": {"promotion_amount"},
        "value_unit": {"promotion_unit"},
        "value_details": {"text"},
        "cap_amount": {"promotion_cap"},
        "conditions": {"text"},
        "can_stack": {"stacking"},
        "stacking_text": {"text"},
    },
    "warranty": {
        "product_group": {"text"},
        "product_months": {"integer", "months"},
        "component": {"text"},
        "component_months": {"integer", "months"},
        "exclusions": {"text"},
    },
    "regions": {"province": {"text"}, "region": {"text"}},
    "service_centres": {
        field: {"text"}
        for field in NORMALIZED_FIELDS["service_centres"] - {"accepted_services"}
    }
    | {"accepted_services": {"boolean_columns"}},
    "service_fees": {
        "code": {"text"},
        "service": {"text"},
        "applies_to": {"comma_list"},
        "unit": {"text"},
        "fee_in_warranty": {"decimal_or_null"},
        "fee_out_of_warranty": {"decimal_or_null"},
        "fee_in_warranty_note": {"non_numeric_text"},
        "fee_out_of_warranty_note": {"non_numeric_text"},
        "notes": {"text"},
    },
}


class NormalizationField(BaseModel):
    """Map one normalized field to a raw source column and parse rule."""

    model_config = ConfigDict(extra="forbid")

    column: str
    parser: NormalizationParser = "text"
    required: bool = False
    context_column: str | None = None
    columns: dict[str, str] | None = None
    options: dict[str, str | list[str] | dict[str, str]] = Field(
        default_factory=dict
    )


class NormalizationExpansion(BaseModel):
    """Expand one source row into records for configured column dimensions."""

    model_config = ConfigDict(extra="forbid")

    dimension_field: str
    value_field: str
    parser: NormalizationParser = "decimal"
    columns: dict[str, str]
    options: dict[str, str | list[str] | dict[str, str]] = Field(
        default_factory=dict
    )


class NormalizationMapping(BaseModel):
    """Describe the typed table and field mappings for a worksheet."""

    model_config = ConfigDict(extra="forbid")

    table: NormalizedTable
    fields: dict[str, NormalizationField]
    expand: NormalizationExpansion | None = None

    @model_validator(mode="after")
    def validate_field_names(self) -> NormalizationMapping:
        """Check configured output fields against the selected typed table.

        Returns:
            The validated normalization mapping.

        Raises:
            ValueError: If a field does not exist in the typed table.
        """
        allowed_fields: set[str] = NORMALIZED_FIELDS[self.table]
        unknown_fields: set[str] = set(self.fields) - allowed_fields
        if unknown_fields:
            fields_text: str = ", ".join(sorted(unknown_fields))
            raise ValueError(
                f"Unknown fields for normalized table {self.table!r}: {fields_text}"
            )
        incompatible_fields: list[str] = [
            f"{field_name} ({field.parser})"
            for field_name, field in self.fields.items()
            if field.parser
            not in NORMALIZED_FIELD_PARSERS[self.table][field_name]
        ]
        if incompatible_fields:
            fields_text = ", ".join(sorted(incompatible_fields))
            raise ValueError(
                f"Parser does not match normalized table {self.table!r}: {fields_text}"
            )
        if self.expand is not None:
            expansion_fields: set[str] = {
                self.expand.dimension_field,
                self.expand.value_field,
            }
            unknown_expansion_fields: set[str] = expansion_fields - allowed_fields
            duplicate_fields: set[str] = expansion_fields & set(self.fields)
            if unknown_expansion_fields or duplicate_fields:
                raise ValueError(
                    "Expanded fields must be valid table fields and must not also "
                    "appear in fields"
                )
            if (
                self.expand.dimension_field != "region"
                or self.expand.value_field != "price"
                or self.expand.parser != "decimal"
            ):
                raise ValueError(
                    "Price expansion must map region names to decimal price values"
                )
        elif self.table == "prices":
            raise ValueError("Price normalization must expand one row per region")
        return self


class SheetMapping(BaseModel):
    """Configure header rows and the first data row in a worksheet.

    Attributes:
        header_rows: Excel row numbers that form the worksheet column labels.
        data_start_row: First Excel row that may contain source records.
        footnote_start_row: Optional first row containing footnotes rather than
            table records.

    Raises:
        ValueError: If data starts before or on a configured header row.
    """

    model_config = ConfigDict(extra="forbid")

    header_rows: tuple[int, ...] = Field(min_length=1)
    data_start_row: int = Field(ge=1)
    footnote_start_row: int | None = Field(default=None, ge=1)
    archive: bool = False
    normalization: NormalizationMapping | None = None

    @model_validator(mode="after")
    def validate_row_order(self) -> SheetMapping:
        """Ensure every header row precedes the configured data.

        Returns:
            The validated worksheet mapping.

        Raises:
            ValueError: If a header row is invalid or data precedes a header.
        """
        if any(row < 1 for row in self.header_rows):
            raise ValueError("header_rows must contain positive Excel row numbers")
        if self.data_start_row <= max(self.header_rows):
            raise ValueError("data_start_row must follow every configured header row")
        if (
            self.footnote_start_row is not None
            and self.footnote_start_row <= self.data_start_row
        ):
            raise ValueError("footnote_start_row must follow data_start_row")
        if self.archive and self.normalization is not None:
            raise ValueError("archive worksheets cannot have a normalization mapping")
        if self.normalization is not None and self.normalization.expand is not None:
            if self.normalization.table != "prices":
                raise ValueError("column expansion is currently supported for prices")
        return self


class WorkbookMapping(BaseModel):
    """Map worksheet names in one workbook to their row layout."""

    model_config = ConfigDict(extra="forbid")

    sheets: dict[str, SheetMapping]


class SchemaMapping(BaseModel):
    """Hold worksheet layouts keyed by workbook filename glob patterns."""

    model_config = ConfigDict(extra="forbid")

    files: dict[str, WorkbookMapping]


class SpreadsheetRow(BaseModel):
    """Hold cell values from one Excel row with its source row number."""

    source_row: int = Field(ge=1)
    values: dict[str, ExcelValue]


class SpreadsheetNote(BaseModel):
    """Represent a non-tabular worksheet note and its original row number."""

    source_row: int = Field(ge=1)
    values: dict[str, ExcelValue]


class SpreadsheetSheet(BaseModel):
    """Represent worksheet data, notes, visibility, and any read warnings."""

    name: str
    ordinal: int = Field(ge=1)
    is_hidden: bool
    columns: list[str]
    rows: list[SpreadsheetRow]
    header_notes: list[SpreadsheetNote] = Field(default_factory=list)
    footnotes: list[SpreadsheetNote] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    archive: bool = False
    normalization: NormalizationMapping | None = Field(default=None, exclude=True)


class SpreadsheetWorkbook(BaseModel):
    """Represent a parsed workbook and the source path for its worksheets."""

    source_file: str
    sheets: list[SpreadsheetSheet]
    warnings: list[str] = Field(default_factory=list)


def load_schema_mapping(mapping_path: Path = DEFAULT_MAPPING_PATH) -> SchemaMapping:
    """Read and validate the worksheet layout mapping JSON.

    Args:
        mapping_path: Path to the mapping JSON file.

    Returns:
        A validated mapping of workbook patterns and worksheet layouts.

    Raises:
        OSError: If the mapping file cannot be read.
        ValidationError: If its JSON does not match the mapping model.
    """
    return SchemaMapping.model_validate_json(mapping_path.read_bytes())


def select_workbook_mapping(
    file_name: str,
    schema_mapping: SchemaMapping,
) -> WorkbookMapping:
    """Select the single configured layout matching a workbook filename.

    Args:
        file_name: Workbook filename, without its directory path.
        schema_mapping: Available filename-pattern mappings.

    Returns:
        The matching workbook layout.

    Raises:
        ValueError: If no pattern or multiple patterns match the filename.
    """
    matches: list[WorkbookMapping] = [
        workbook_mapping
        for pattern, workbook_mapping in schema_mapping.files.items()
        if fnmatch.fnmatchcase(file_name, pattern)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one schema mapping for {file_name!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _header_cell_value(
    worksheet: Worksheet,
    row_number: int,
    column_number: int,
) -> object:
    """Return a cell value, including labels inherited from merged cells.

    Args:
        worksheet: Worksheet containing the header.
        row_number: One-based Excel row number.
        column_number: One-based Excel column number.

    Returns:
        The cell value, or ``None`` if it is blank.
    """
    cell_value: object = worksheet.cell(row_number, column_number).value
    if cell_value is not None:
        return cell_value
    for cell_range in worksheet.merged_cells.ranges:
        if (
            cell_range.min_row <= row_number <= cell_range.max_row
            and cell_range.min_col <= column_number <= cell_range.max_col
        ):
            return worksheet.cell(cell_range.min_row, cell_range.min_col).value
    return None


def _column_names(worksheet: Worksheet, mapping: SheetMapping) -> list[str]:
    """Build unique labels from the configured worksheet header rows.

    Args:
        worksheet: Worksheet whose columns are named.
        mapping: Header-row configuration for the worksheet.

    Returns:
        Ordered unique labels, using Excel column letters for blank headers.
    """
    names: list[str] = []
    seen: dict[str, int] = {}
    for column_number in range(1, worksheet.max_column + 1):
        parts: list[str] = []
        for row_number in mapping.header_rows:
            value: object = _header_cell_value(worksheet, row_number, column_number)
            if value is not None:
                label: str = str(value).strip()
                if label and (not parts or parts[-1] != label):
                    parts.append(label)
        name: str = " / ".join(parts) or f"column_{get_column_letter(column_number)}"
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name} ({seen[name]})"
        names.append(name)
    return names


def _excel_value(value: object, source: str) -> ExcelValue:
    """Convert an Excel value to a scalar suitable for JSON serialization.

    Args:
        value: Value read from an Excel cell.
        source: Workbook, worksheet, and column for actionable errors.

    Returns:
        A JSON-compatible scalar; dates and times use ISO 8601 strings.

    Raises:
        TypeError: If the cell contains an unsupported Python value.
    """
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported Excel value at {source}: {type(value).__name__}")


def _note_rows(
    worksheet: Worksheet,
    start_row: int,
    end_row: int,
    source: str,
) -> list[SpreadsheetNote]:
    """Collect non-empty raw worksheet rows outside of the data table.

    Args:
        worksheet: Worksheet containing notes.
        start_row: First inclusive row to inspect.
        end_row: Last inclusive row to inspect.
        source: Workbook and worksheet label for actionable errors.

    Returns:
        Non-empty note rows keyed by their Excel column letters.

    Raises:
        TypeError: If a note cell contains an unsupported Python value.
    """
    notes: list[SpreadsheetNote] = []
    for row_number in range(start_row, end_row + 1):
        values: dict[str, ExcelValue] = {
            get_column_letter(column_number): _excel_value(
                worksheet.cell(row_number, column_number).value,
                f"{source}!{get_column_letter(column_number)}{row_number}",
            )
            for column_number in range(1, worksheet.max_column + 1)
            if worksheet.cell(row_number, column_number).value is not None
        }
        if values:
            notes.append(SpreadsheetNote(source_row=row_number, values=values))
    return notes


def _read_unmapped_sheet(
    worksheet: Worksheet,
    sheet_ordinal: int,
    workbook_name: str,
    warning: str,
) -> SpreadsheetSheet:
    """Preserve all populated cells when a worksheet mapping is unavailable.

    Args:
        worksheet: Worksheet to read without a trusted schema.
        sheet_ordinal: One-based position in the workbook.
        workbook_name: Workbook filename used in warning context.
        warning: Explanation of why the fallback was needed.

    Returns:
        An unstructured sheet whose non-empty rows are retained as data.

    Raises:
        TypeError: If a cell contains an unsupported Python value.
    """
    columns: list[str] = [
        get_column_letter(column_number)
        for column_number in range(1, worksheet.max_column + 1)
    ]
    rows: list[SpreadsheetRow] = []
    for row_number in range(1, worksheet.max_row + 1):
        values: dict[str, ExcelValue] = {
            column: _excel_value(
                worksheet.cell(row_number, column_number).value,
                f"{workbook_name}:{worksheet.title}!{column}{row_number}",
            )
            for column_number, column in enumerate(columns, start=1)
        }
        if any(value is not None for value in values.values()):
            rows.append(SpreadsheetRow(source_row=row_number, values=values))
    return SpreadsheetSheet(
        name=worksheet.title,
        ordinal=sheet_ordinal,
        is_hidden=worksheet.sheet_state != "visible",
        columns=columns,
        rows=rows,
        warnings=[warning],
    )


def read_workbook(
    workbook_path: Path,
    workbook_mapping: WorkbookMapping | None,
    source_file: str | None = None,
    warnings: list[str] | None = None,
) -> SpreadsheetWorkbook:
    """Read a workbook into structured data without writing to a database.

    Args:
        workbook_path: Path to the source ``.xlsx`` workbook.
        workbook_mapping: Layout definitions for each worksheet, or ``None`` to
            read worksheets as unstructured rows.
        source_file: Optional source label, such as a path relative to the data
            directory. Defaults to the workbook filename.
        warnings: Optional warnings about the workbook or its schema mapping.

    Returns:
        The workbook's worksheets, cell values, and original row provenance.

    Raises:
        KeyError: If a workbook worksheet has no configured mapping.
        OSError: If the workbook cannot be opened.
        TypeError: If an Excel cell contains an unsupported value.
    """
    workbook = load_workbook(workbook_path, read_only=False, data_only=True)
    parsed_sheets: list[SpreadsheetSheet] = []
    workbook_warnings: list[str] = list(warnings or [])
    try:
        for sheet_ordinal, worksheet in enumerate(workbook.worksheets, start=1):
            mapping: SheetMapping | None = (
                workbook_mapping.sheets.get(worksheet.title)
                if workbook_mapping is not None
                else None
            )
            if mapping is None:
                warning: str = (
                    f"No valid schema mapping for worksheet {worksheet.title!r}; "
                    "read all populated rows without separating table data and notes."
                )
                workbook_warnings.append(f"{workbook_path.name}: {warning}")
                parsed_sheets.append(
                    _read_unmapped_sheet(
                        worksheet,
                        sheet_ordinal,
                        workbook_path.name,
                        warning,
                    )
                )
                continue
            if (
                max(mapping.header_rows) > worksheet.max_row
                or mapping.data_start_row > worksheet.max_row + 1
                or (
                    mapping.footnote_start_row is not None
                    and mapping.footnote_start_row > worksheet.max_row + 1
                )
            ):
                warning = (
                    f"Schema row positions are outside worksheet "
                    f"{worksheet.title!r}; read all populated rows without "
                    "separating table data and notes."
                )
                workbook_warnings.append(f"{workbook_path.name}: {warning}")
                parsed_sheets.append(
                    _read_unmapped_sheet(
                        worksheet,
                        sheet_ordinal,
                        workbook_path.name,
                        warning,
                    )
                )
                continue
            columns: list[str] = _column_names(worksheet, mapping)
            table_end_row: int = (
                mapping.footnote_start_row - 1
                if mapping.footnote_start_row is not None
                else worksheet.max_row
            )
            rows: list[SpreadsheetRow] = []
            for row_number in range(
                mapping.data_start_row,
                table_end_row + 1,
            ):
                values: dict[str, ExcelValue] = {
                    column: _excel_value(
                        worksheet.cell(row_number, column_number).value,
                        f"{workbook_path.name}:{worksheet.title}!{column}",
                    )
                    for column_number, column in enumerate(columns, start=1)
                }
                if any(value is not None for value in values.values()):
                    rows.append(SpreadsheetRow(source_row=row_number, values=values))
            first_header_row: int = min(mapping.header_rows)
            header_notes: list[SpreadsheetNote] = _note_rows(
                worksheet,
                1,
                first_header_row - 1,
                f"{workbook_path.name}:{worksheet.title}",
            )
            footnotes: list[SpreadsheetNote] = (
                _note_rows(
                    worksheet,
                    mapping.footnote_start_row,
                    worksheet.max_row,
                    f"{workbook_path.name}:{worksheet.title}",
                )
                if mapping.footnote_start_row is not None
                else []
            )
            parsed_sheets.append(
                SpreadsheetSheet(
                    name=worksheet.title,
                    ordinal=sheet_ordinal,
                    is_hidden=worksheet.sheet_state != "visible",
                    columns=columns,
                    rows=rows,
                    header_notes=header_notes,
                    footnotes=footnotes,
                    archive=mapping.archive,
                    normalization=mapping.normalization,
                )
            )
    finally:
        workbook.close()
    return SpreadsheetWorkbook(
        source_file=source_file or workbook_path.name,
        sheets=parsed_sheets,
        warnings=workbook_warnings,
    )


def read_spreadsheet_directory(
    data_dir: Path,
    mapping_path: Path = DEFAULT_MAPPING_PATH,
) -> list[SpreadsheetWorkbook]:
    """Read every mapped ``.xlsx`` workbook below a data directory.

    Args:
        data_dir: Directory recursively searched for workbooks.
        mapping_path: JSON file describing workbook and worksheet layouts.

    Returns:
        Parsed workbooks sorted by their relative paths.

    Raises:
        FileNotFoundError: If the directory is missing or contains no workbooks.
        OSError: If a workbook cannot be read.
        TypeError: If an Excel cell contains an unsupported value.
    """
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    mapping_warnings: list[str] = []
    try:
        schema_mapping: SchemaMapping | None = load_schema_mapping(mapping_path)
    except (OSError, ValidationError, ValueError) as error:
        schema_mapping = None
        mapping_warnings.append(
            f"Could not load schema mapping {mapping_path}: {error}. "
            "All workbook sheets will be read as unstructured rows."
        )
    workbook_paths: list[Path] = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".xlsx"
    )
    if not workbook_paths:
        raise FileNotFoundError(f"No .xlsx workbooks found under {data_dir}")
    workbooks: list[SpreadsheetWorkbook] = []
    for workbook_path in workbook_paths:
        relative_path: str = workbook_path.relative_to(data_dir).as_posix()
        workbook_mapping: WorkbookMapping | None = None
        workbook_warnings: list[str] = list(mapping_warnings)
        if schema_mapping is not None:
            try:
                workbook_mapping = select_workbook_mapping(
                    workbook_path.name,
                    schema_mapping,
                )
            except ValueError as error:
                workbook_warnings.append(
                    f"{error}. Workbook will be read as unstructured rows."
                )
        workbooks.append(
            read_workbook(
                workbook_path,
                workbook_mapping,
                source_file=relative_path,
                warnings=workbook_warnings,
            )
        )
    return workbooks
