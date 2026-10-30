"""Print extracted PDF pages as JSON for manual inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.tools.pdf_reader import PDFDocument, read_pdf_directory


def main() -> None:
    """Print all extracted PDF pages from the selected data directory.

    Returns:
        None: Formatted JSON is written to standard output.

    Raises:
        FileNotFoundError: If the input directory is missing or contains no PDFs.
        OSError: If a PDF cannot be opened.
        ValueError: If a PDF is encrypted and cannot be read without a password.
        pymupdf.FileDataError: If a PDF is malformed or unreadable.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    arguments = parser.parse_args()
    documents: list[PDFDocument] = read_pdf_directory(arguments.data_dir)
    print(
        json.dumps(
            [document.model_dump(mode="json") for document in documents],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

documents = read_pdf_directory(Path("data"))