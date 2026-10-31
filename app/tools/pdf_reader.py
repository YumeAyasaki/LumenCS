"""Read policy PDFs into page-level, provenance-aware server data models."""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import pymupdf
from pydantic import BaseModel, Field

logger: logging.Logger = logging.getLogger(__name__)


class PDFElement(BaseModel):
    """Represent a text block or table in its original page order.

    Attributes:
        kind: Whether the element contains extracted text or a table.
        text: Element content with line breaks preserved.
        table_index: One-based table index on the source page, if applicable.

    Raises:
        ValidationError: If a field does not match the model.
    """

    kind: Literal["text", "table"]
    text: str
    table_index: int | None = Field(default=None, ge=1)


class PDFPage(BaseModel):
    """Represent extracted text from one PDF page.

    Attributes:
        page_number: One-based page number in the source PDF.
        cite_id: Stable source identifier that includes the page number.
        text: Extracted page text, which may be empty for scanned or blank pages.
        elements: Ordered text blocks and table rows with original line breaks.

    Raises:
        ValidationError: If a page number or field value is invalid.
    """

    page_number: int = Field(ge=1)
    cite_id: str
    text: str
    elements: list[PDFElement] = Field(default_factory=list)


class PDFDocument(BaseModel):
    """Represent a PDF as flexible page text with source provenance.

    Attributes:
        source_file: Source filename or path relative to the data directory.
        pages: Extracted pages in their original order.
        document_version: Version parsed from the document header, if present.
        effective_date: Effective date parsed from the document header, if present.
        warnings: Non-fatal extraction issues, such as pages without text.

    Raises:
        ValidationError: If a document field does not match the model.
    """

    source_file: str
    pages: list[PDFPage]
    document_version: str | None = None
    effective_date: date | None = None
    warnings: list[str] = Field(default_factory=list)


def _intersects(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    """Return whether two page rectangles overlap with positive area.

    Args:
        first: First rectangle as ``(x0, y0, x1, y1)``.
        second: Second rectangle as ``(x0, y0, x1, y1)``.

    Returns:
        bool: Whether the rectangles overlap.
    """
    return (
        min(first[2], second[2]) > max(first[0], second[0])
        and min(first[3], second[3]) > max(first[1], second[1])
    )


def _table_text(rows: list[list[str | None]]) -> str:
    """Serialize table rows without flattening row or cell boundaries.

    Args:
        rows: Cell values returned by PyMuPDF's table extractor.

    Returns:
        str: Rows separated by newlines and cells by arrows.
    """
    return "\n".join(
        " → ".join((cell or "").strip() for cell in row)
        for row in rows
        if any(cell and cell.strip() for cell in row)
    )


def _page_elements(page: pymupdf.Page) -> list[PDFElement]:
    """Extract ordered text blocks and table rows from one page.

    Args:
        page: Open PyMuPDF page to inspect.

    Returns:
        list[PDFElement]: Text blocks and tables ordered from top to bottom.
    """
    finder: pymupdf.table.TableFinder = page.find_tables()
    table_regions: list[tuple[float, float, float, float]] = [
        tuple(float(coordinate) for coordinate in table.bbox)
        for table in finder.tables
    ]
    positioned: list[
        tuple[float, float, int, PDFElement]
    ] = []

    for block in page.get_text("blocks"):
        x0, y0, x1, y1, content = block[:5]
        if int(block[6]) != 0 or not str(content).strip():
            continue
        block_rectangle: tuple[float, float, float, float] = (
            float(x0),
            float(y0),
            float(x1),
            float(y1),
        )
        if any(_intersects(block_rectangle, region) for region in table_regions):
            continue
        positioned.append(
            (
                float(y0),
                float(x0),
                0,
                PDFElement(kind="text", text=str(content).rstrip()),
            )
        )

    for table_index, table in enumerate(finder.tables, start=1):
        table_text: str = _table_text(table.extract())
        if table_text:
            x0, y0, _, _ = table.bbox
            positioned.append(
                (
                    float(y0),
                    float(x0),
                    table_index,
                    PDFElement(
                        kind="table",
                        text=table_text,
                        table_index=table_index,
                    ),
                )
            )

    positioned.sort(key=lambda item: (item[0], item[1], item[2]))
    return [element for _, _, _, element in positioned]


def _document_metadata(text: str) -> tuple[str | None, date | None]:
    """Extract version and effective date from the handbook header.

    Args:
        text: Text extracted from the PDF document.

    Returns:
        tuple[str | None, date | None]: Version and effective date, if found.
    """
    version_match: re.Match[str] | None = re.search(
        r"Phiên bản\s+([0-9]+(?:\.[0-9]+)*)",
        text,
        flags=re.IGNORECASE,
    )
    effective_match: re.Match[str] | None = re.search(
        r"Hiệu lực từ\s+(\d{2}/\d{2}/\d{4})",
        text,
        flags=re.IGNORECASE,
    )
    effective_date: date | None = (
        datetime.strptime(effective_match.group(1), "%d/%m/%Y").date()
        if effective_match
        else None
    )
    return (
        version_match.group(1) if version_match else None,
        effective_date,
    )


def read_pdf(pdf_path: Path, source_file: str | None = None) -> PDFDocument:
    """Extract page text from one PDF without imposing a document schema.

    Args:
        pdf_path: Path to the PDF file on disk.
        source_file: Optional source name to retain instead of the file name.

    Returns:
        PDFDocument: Ordered page text with stable page citation identifiers.

    Raises:
        FileNotFoundError: If ``pdf_path`` does not exist.
        OSError: If the PDF cannot be opened.
        ValueError: If the PDF is encrypted and cannot be read without a password.
        pymupdf.FileDataError: If the file is not a readable PDF.
    """
    resolved_source: str = source_file or pdf_path.name
    pages: list[PDFPage] = []
    empty_pages: list[int] = []
    document_text: list[str] = []
    with pymupdf.open(pdf_path) as document:
        if document.is_encrypted:
            raise ValueError(
                f"Encrypted PDF cannot be read without a password: {pdf_path}"
            )
        for page_number in range(1, document.page_count + 1):
            page: pymupdf.Page = document.load_page(page_number - 1)
            text: str = page.get_text()
            document_text.append(text)
            if not text.strip():
                empty_pages.append(page_number)
            pages.append(
                PDFPage(
                    page_number=page_number,
                    cite_id=f"{resolved_source}#page={page_number}",
                    text=text,
                    elements=_page_elements(page),
                )
            )

    warnings: list[str] = []
    if empty_pages:
        page_list: str = ", ".join(str(page_number) for page_number in empty_pages)
        warnings.append(
            f"No extractable text on page(s) {page_list} of {resolved_source}; "
            "the pages may be blank or require OCR."
        )
    logger.info(
        "Read PDF %s: %d pages, %d pages without extractable text",
        resolved_source,
        len(pages),
        len(empty_pages),
    )
    document_version, effective_date = _document_metadata("\n".join(document_text))
    return PDFDocument(
        source_file=resolved_source,
        pages=pages,
        document_version=document_version,
        effective_date=effective_date,
        warnings=warnings,
    )


def read_pdf_directory(data_dir: Path) -> list[PDFDocument]:
    """Read every PDF recursively beneath a data directory.

    Args:
        data_dir: Directory recursively searched for ``.pdf`` files.

    Returns:
        list[PDFDocument]: Parsed PDFs sorted by their relative paths.

    Raises:
        FileNotFoundError: If the directory is missing or contains no PDFs.
        OSError: If a PDF cannot be opened.
        ValueError: If a PDF is encrypted and cannot be read without a password.
        pymupdf.FileDataError: If a file is not a readable PDF.
    """
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    pdf_paths: list[Path] = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    if not pdf_paths:
        raise FileNotFoundError(f"No .pdf files found under {data_dir}")

    documents: list[PDFDocument] = []
    for pdf_path in pdf_paths:
        relative_path: str = pdf_path.relative_to(data_dir).as_posix()
        documents.append(read_pdf(pdf_path, source_file=relative_path))
    return documents
