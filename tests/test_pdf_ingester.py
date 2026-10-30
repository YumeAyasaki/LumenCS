"""Tests for adding PDF page chunks to Qdrant."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from qdrant_client import models

from app.tools.pdf_ingester import PDFIngestionResult, _text_chunks, ingest_pdf_directory
from app.tools.pdf_reader import PDFDocument, PDFPage


class PDFIngesterTests(unittest.TestCase):
    """Verify PDF page chunks are embedded with source citations."""

    def test_splits_long_text_with_overlap(self) -> None:
        """Keep all page text across bounded, overlapping embedding chunks."""
        text: str = "word " * 700

        chunks: list[str] = _text_chunks(text)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 1200 for chunk in chunks))
        self.assertIn(chunks[0][-40:], chunks[1])

    def test_indexes_pages_with_source_and_citation_payload(self) -> None:
        """Embed PDF page text and preserve page-level citation metadata."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir: Path = Path(temporary_directory)
            (data_dir / "policy.pdf").write_bytes(b"pdf source bytes")
            document: PDFDocument = PDFDocument(
                source_file="policy.pdf",
                pages=[
                    PDFPage(
                        page_number=1,
                        cite_id="policy.pdf#page=1",
                        text="Warranty coverage and service information.",
                    )
                ],
            )
            qdrant_client: MagicMock = MagicMock()
            qdrant_client.set_model = MagicMock()
            qdrant_client.add = MagicMock(return_value=["point-1"])
            qdrant_client.delete = MagicMock()
            old_point: models.Record = models.Record(
                id="old-point",
                payload={"source_file": "policy.pdf"},
            )

            with (
                patch(
                    "app.tools.pdf_ingester.read_pdf_directory",
                    return_value=[document],
                ),
                patch(
                    "app.tools.pdf_ingester._existing_points",
                    return_value=[old_point],
                ),
            ):
                result: PDFIngestionResult = ingest_pdf_directory(
                    qdrant_client=qdrant_client,
                    data_dir=data_dir,
                    collection_name="policy_documents",
                    embedding_model="multilingual-model",
                )

        self.assertEqual(result.processed_files, 1)
        self.assertEqual(result.page_count, 1)
        self.assertEqual(result.chunk_count, 1)
        self.assertEqual(result.removed_count, 1)
        qdrant_client.set_model.assert_called_once_with("multilingual-model")
        qdrant_client.add.assert_called_once()
        add_arguments: dict[str, object] = qdrant_client.add.call_args.kwargs
        self.assertEqual(
            add_arguments["documents"],
            ["Warranty coverage and service information."],
        )
        self.assertEqual(
            add_arguments["metadata"][0]["cite_id"],
            "policy.pdf#page=1",
        )
        qdrant_client.delete.assert_called_once()


if __name__ == "__main__":
    unittest.main()
