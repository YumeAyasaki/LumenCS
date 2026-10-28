"""Read configured Excel workbooks into provenance-aware server data models."""

from __future__ import annotations

import fnmatch
from datetime import date, datetime, time
from pathlib import Path
from typing import TypeAlias

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_MAPPING_PATH: Path = Path(__file__).parents[1] / "schema_mapping.json"
ExcelValue: TypeAlias = str | int | float | bool | None


class SheetMapping(BaseModel):
    """Configure header rows and the first data row in a worksheet.

    Attributes:
        header_rows: Excel row numbers that form the worksheet column labels.
        data_start_row: First Excel row that may contain source records.

    Raises:
        ValueError: If data starts before or on a configured header row.
    """

    model_config = ConfigDict(extra="forbid")

    header_rows: tuple[int, ...] = Field(min_length=1)
    data_start_row: int = Field(ge=1)

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


class SpreadsheetSheet(BaseModel):
    """Represent one worksheet as columns and provenance-bearing rows."""

    name: str
    ordinal: int = Field(ge=1)
    columns: list[str]
    rows: list[SpreadsheetRow]


class SpreadsheetWorkbook(BaseModel):
    """Represent a parsed workbook and the source path for its worksheets."""

    source_file: str
    sheets: list[SpreadsheetSheet]


def load_schema_mapping(mapping_path: Path = DEFAULT_MAPPING_PATH) -> SchemaMapping:
    """Read and validate the worksheet layout mapping JSON.

    Args:
        mapping_path: Path to the mapping JSON file.

    Returns:
        A validated mapping of workbook patterns and worksheet layouts.

    Raises:
        OSError: If the mapping file cannot be read.
        ValueError: If its JSON does not match the mapping model.
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


def read_workbook(
    workbook_path: Path,
    workbook_mapping: WorkbookMapping,
    source_file: str | None = None,
) -> SpreadsheetWorkbook:
    """Read a workbook into structured data without writing to a database.

    Args:
        workbook_path: Path to the source ``.xlsx`` workbook.
        workbook_mapping: Layout definitions for each workbook worksheet.
        source_file: Optional source label, such as a path relative to the data
            directory. Defaults to the workbook filename.

    Returns:
        The workbook's worksheets, cell values, and original row provenance.

    Raises:
        KeyError: If a workbook worksheet has no configured mapping.
        OSError: If the workbook cannot be opened.
        TypeError: If an Excel cell contains an unsupported value.
        ValueError: If a configured header lies outside a worksheet.
    """
    workbook = load_workbook(workbook_path, read_only=False, data_only=True)
    parsed_sheets: list[SpreadsheetSheet] = []
    try:
        for sheet_ordinal, worksheet in enumerate(workbook.worksheets, start=1):
            if worksheet.title not in workbook_mapping.sheets:
                raise KeyError(
                    f"No schema mapping for worksheet {worksheet.title!r} "
                    f"in {workbook_path.name!r}"
                )
            mapping: SheetMapping = workbook_mapping.sheets[worksheet.title]
            if max(mapping.header_rows) > worksheet.max_row:
                raise ValueError(
                    f"Header rows {mapping.header_rows} exceed worksheet "
                    f"{worksheet.title!r} in {workbook_path.name!r}"
                )
            columns: list[str] = _column_names(worksheet, mapping)
            rows: list[SpreadsheetRow] = []
            for row_number in range(mapping.data_start_row, worksheet.max_row + 1):
                values: dict[str, ExcelValue] = {
                    column: _excel_value(
                        worksheet.cell(row_number, column_number).value,
                        f"{workbook_path.name}:{worksheet.title}!{column}",
                    )
                    for column_number, column in enumerate(columns, start=1)
                }
                if any(value is not None for value in values.values()):
                    rows.append(SpreadsheetRow(source_row=row_number, values=values))
            parsed_sheets.append(
                SpreadsheetSheet(
                    name=worksheet.title,
                    ordinal=sheet_ordinal,
                    columns=columns,
                    rows=rows,
                )
            )
    finally:
        workbook.close()
    return SpreadsheetWorkbook(
        source_file=source_file or workbook_path.name,
        sheets=parsed_sheets,
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
        OSError: If the mapping or a workbook cannot be read.
        KeyError: If a workbook worksheet has no configured mapping.
        TypeError: If an Excel cell contains an unsupported value.
        ValueError: If a workbook mapping is missing or ambiguous.
    """
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    schema_mapping: SchemaMapping = load_schema_mapping(mapping_path)
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
        workbook_mapping: WorkbookMapping = select_workbook_mapping(
            workbook_path.name,
            schema_mapping,
        )
        workbooks.append(
            read_workbook(
                workbook_path,
                workbook_mapping,
                source_file=relative_path,
            )
        )
    return workbooks
