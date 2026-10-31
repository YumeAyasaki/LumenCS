"""Tests for adding clause-level PDF chunks to Qdrant."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from qdrant_client import models

from app.tools.pdf_ingester import (
    PDFIngestionResult,
    PDFPolicyChunk,
    _policy_chunks,
    _text_chunks,
    ingest_pdf_directory,
    model_collection_name,
)
from app.tools.pdf_reader import PDFDocument, PDFPage, read_pdf


class PDFIngesterTests(unittest.TestCase):
    """Verify policy clauses retain table content and source citations."""

    def test_splits_long_text_with_overlap(self) -> None:
        """Keep all page text across bounded, overlapping embedding chunks."""
        text: str = "word " * 700

        chunks: list[str] = _text_chunks(text)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 1200 for chunk in chunks))
        self.assertIn(chunks[0][-40:], chunks[1])

    def test_indexes_pages_with_source_and_citation_payload(self) -> None:
        """Embed clause chunks and preserve model-specific payload metadata."""
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
        expected_collection: str = model_collection_name(
            "policy_documents",
            "multilingual-model",
        )
        self.assertEqual(result.collection_name, expected_collection)
        qdrant_client.set_model.assert_called_once_with("multilingual-model")
        qdrant_client.add.assert_called_once()
        add_arguments: dict[str, object] = qdrant_client.add.call_args.kwargs
        self.assertEqual(add_arguments["collection_name"], expected_collection)
        self.assertEqual(
            add_arguments["documents"],
            ["policy\nWarranty coverage and service information."],
        )
        self.assertEqual(
            add_arguments["metadata"][0]["section_id"],
            "document",
        )
        self.assertEqual(
            add_arguments["metadata"][0]["cite_id"],
            "policy.pdf#document",
        )
        self.assertIn("sha256", add_arguments["metadata"][0])
        self.assertIn("chunk_sha256", add_arguments["metadata"][0])
        self.assertEqual(
            add_arguments["metadata"][0]["embedding_model"],
            "multilingual-model",
        )
        qdrant_client.delete.assert_called_once()

    def test_chunks_handbook_by_clause_bullets_tables_and_appendix(self) -> None:
        """Preserve numbered clauses, individual bullets, and table row order."""
        handbook_path: Path = (
            Path(__file__).parents[1] / "data" / "policy_handbook.pdf"
        )
        document: PDFDocument = read_pdf(
            handbook_path,
            source_file="policy_handbook.pdf",
        )

        chunks = _policy_chunks(document)
        warranty_table: PDFPolicyChunk = next(
            chunk
            for chunk in chunks
            if chunk.cite_id == "policy_handbook.pdf#3.2"
            and "45 ngày kể từ ngày mua" in chunk.text
        )

        self.assertEqual(document.document_version, "3.2")
        self.assertEqual(
            document.effective_date.isoformat() if document.effective_date else None,
            "2026-10-01",
        )
        self.assertEqual(warranty_table.section_id, "3.2")
        self.assertEqual(warranty_table.section_title, "3. Bảo hành")
        self.assertIn(
            "Từ 01/10/2026 → 45 ngày kể từ ngày mua → Linh kiện chính: 60 tháng",
            warranty_table.text,
        )
        self.assertEqual(warranty_table.page, 1)
        self.assertEqual(warranty_table.document_version, "3.2")
        self.assertEqual(warranty_table.effective_date, "2026-10-01")
        self.assertTrue(
            any(chunk.section_id == "4.bullet-1" for chunk in chunks)
        )
        self.assertEqual(
            sum(chunk.section_id.startswith("4.bullet-") for chunk in chunks),
            5,
        )
        self.assertTrue(
            any(
                chunk.cite_id == "policy_handbook.pdf#appendix-a.table-2"
                for chunk in chunks
            )
        )

    def test_model_changes_use_distinct_collection_names(self) -> None:
        """Avoid mixing vectors generated by different embedding models."""
        first_collection: str = model_collection_name(
            "policy_documents",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        second_collection: str = model_collection_name(
            "policy_documents",
            "intfloat/multilingual-e5-large",
        )

        self.assertNotEqual(first_collection, second_collection)
        self.assertIn("MiniLM", first_collection)
        self.assertIn("e5-large", second_collection)

    def test_unnumbered_documents_fall_back_to_preserving_character_chunks(self) -> None:
        """Split unnumbered prose by size without flattening its line breaks."""
        document: PDFDocument = PDFDocument(
            source_file="policy.pdf",
            pages=[
                PDFPage(
                    page_number=1,
                    cite_id="policy.pdf#page=1",
                    text="Introductory policy text.\n" + ("A" * 2500),
                )
            ],
        )

        chunks: list[PDFPolicyChunk] = _policy_chunks(document)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 1200 for chunk in chunks))
        self.assertTrue(chunks[0].text.startswith("policy\n"))
        self.assertIn("\n", chunks[0].text)


if __name__ == "__main__":
    unittest.main()
