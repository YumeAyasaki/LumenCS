"""Tests for the reusable policy PDF reader component."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pymupdf

from app.tools.pdf_reader import PDFDocument, read_pdf, read_pdf_directory


def _write_text_pdf(pdf_path: Path, text: str) -> None:
    """Write a minimal single-page PDF containing one line of text.

    Args:
        pdf_path: Destination path for the generated PDF.
        text: ASCII text to place on the page.

    Returns:
        None.

    Raises:
        OSError: If the destination file cannot be written.
        UnicodeEncodeError: If ``text`` contains non-ASCII characters.
    """
    content: bytes = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode("ascii")
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        (
            f"<< /Length {len(content)} >>\nstream\n".encode("ascii")
            + content
            + b"\nendstream"
        ),
    ]
    pdf: bytes = b"%PDF-1.4\n"
    offsets: list[int] = [0]
    for object_number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{object_number} 0 obj\n".encode("ascii") + body + b"\nendobj\n"

    xref_offset: int = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    pdf += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        pdf += f"{offset:010} 00000 n \n".encode("ascii")
    pdf += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode("ascii")
    pdf_path.write_bytes(pdf)


class PDFReaderTests(unittest.TestCase):
    """Verify flexible page extraction, source identifiers, and directory loading."""

    def test_reads_page_text_with_source_provenance(self) -> None:
        """Extract policy prose and identify its original source page."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            pdf_path: Path = Path(temporary_directory) / "policy.pdf"
            _write_text_pdf(pdf_path, "Warranty policy")

            parsed: PDFDocument = read_pdf(
                pdf_path,
                source_file="policy.pdf",
            )

        self.assertEqual(parsed.source_file, "policy.pdf")
        self.assertEqual(len(parsed.pages), 1)
        self.assertEqual(parsed.pages[0].page_number, 1)
        self.assertEqual(parsed.pages[0].cite_id, "policy.pdf#page=1")
        self.assertEqual(parsed.pages[0].text.strip(), "Warranty policy")
        self.assertEqual(parsed.warnings, [])

    def test_reads_nested_pdfs_in_relative_path_order(self) -> None:
        """Load all PDFs recursively and preserve directory-relative paths."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir: Path = Path(temporary_directory) / "data"
            nested_dir: Path = data_dir / "handbooks"
            nested_dir.mkdir(parents=True)
            _write_text_pdf(data_dir / "policy.pdf", "Policy")
            _write_text_pdf(nested_dir / "warranty.pdf", "Warranty")

            documents: list[PDFDocument] = read_pdf_directory(data_dir)

        self.assertEqual(
            [document.source_file for document in documents],
            ["handbooks/warranty.pdf", "policy.pdf"],
        )
        self.assertEqual(
            documents[0].pages[0].cite_id,
            "handbooks/warranty.pdf#page=1",
        )

    def test_reports_pages_without_extractable_text(self) -> None:
        """Keep blank page data and report that OCR may be needed."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            pdf_path: Path = Path(temporary_directory) / "scanned.pdf"
            _write_text_pdf(pdf_path, "")

            parsed: PDFDocument = read_pdf(pdf_path)

        self.assertEqual(parsed.pages[0].text, "")
        self.assertEqual(len(parsed.warnings), 1)
        self.assertIn("may be blank or require OCR", parsed.warnings[0])

    def test_rejects_missing_or_empty_data_directories(self) -> None:
        """Fail explicitly when the PDF input directory has no usable files."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root: Path = Path(temporary_directory)
            missing_dir: Path = root / "missing"
            empty_dir: Path = root / "empty"
            empty_dir.mkdir()

            with self.assertRaises(FileNotFoundError):
                read_pdf_directory(missing_dir)
            with self.assertRaisesRegex(FileNotFoundError, "No .pdf files"):
                read_pdf_directory(empty_dir)

    def test_rejects_invalid_pdf(self) -> None:
        """Raise an explicit PDF parsing error for invalid source files."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            pdf_path: Path = Path(temporary_directory) / "invalid.pdf"
            pdf_path.write_text("not a PDF", encoding="utf-8")

            with self.assertRaises(pymupdf.FileDataError):
                read_pdf(pdf_path)


if __name__ == "__main__":
    unittest.main()
