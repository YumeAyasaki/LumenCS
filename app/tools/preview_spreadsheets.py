"""Print parsed workbook data as JSON for manual inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.tools.spreadsheet_reader import (
    DEFAULT_MAPPING_PATH,
    SpreadsheetWorkbook,
    read_spreadsheet_directory,
)


def main() -> None:
    """Print all parsed workbooks from the selected data directory.

    Returns:
        ``None`` after writing formatted JSON to standard output.

    Raises:
        FileNotFoundError: If the input directory or workbook files are missing.
        OSError: If a mapping or workbook cannot be read.
        ValueError: If a workbook mapping is missing or ambiguous.
        KeyError: If a workbook worksheet has no configured mapping.
        TypeError: If an Excel cell contains an unsupported value.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH)
    arguments: argparse.Namespace = parser.parse_args()
    workbooks: list[SpreadsheetWorkbook] = read_spreadsheet_directory(
        arguments.data_dir,
        arguments.mapping,
    )
    print(
        json.dumps(
            [workbook.model_dump(mode="json") for workbook in workbooks],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
