"""Read policy PDFs into page-level, provenance-aware server data models."""

from __future__ import annotations

import logging
from pathlib import Path

import pymupdf
from pydantic import BaseModel, Field

logger: logging.Logger = logging.getLogger(__name__)


class PDFPage(BaseModel):
    """Represent extracted text from one PDF page.

    Attributes:
        page_number: One-based page number in the source PDF.
        cite_id: Stable source identifier that includes the page number.
        text: Extracted page text, which may be empty for scanned or blank pages.

    Raises:
        ValidationError: If a page number or field value is invalid.
    """

    page_number: int = Field(ge=1)
    cite_id: str
    text: str


class PDFDocument(BaseModel):
    """Represent a PDF as flexible page text with source provenance.

    Attributes:
        source_file: Source filename or path relative to the data directory.
        pages: Extracted pages in their original order.
        warnings: Non-fatal extraction issues, such as pages without text.

    Raises:
        ValidationError: If a document field does not match the model.
    """

    source_file: str
    pages: list[PDFPage]
    warnings: list[str] = Field(default_factory=list)


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
    with pymupdf.open(pdf_path) as document:
        if document.is_encrypted:
            raise ValueError(
                f"Encrypted PDF cannot be read without a password: {pdf_path}"
            )
        for page_number in range(1, document.page_count + 1):
            page: pymupdf.Page = document.load_page(page_number - 1)
            text: str = page.get_text()
            if not text.strip():
                empty_pages.append(page_number)
            pages.append(
                PDFPage(
                    page_number=page_number,
                    cite_id=f"{resolved_source}#page={page_number}",
                    text=text,
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
    return PDFDocument(
        source_file=resolved_source,
        pages=pages,
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
