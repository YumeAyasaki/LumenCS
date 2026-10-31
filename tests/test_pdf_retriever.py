"""Tests for asynchronous policy retrieval and citation payload validation."""

from __future__ import annotations

import asyncio
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from qdrant_client import models

from app.tools.pdf_ingester import model_collection_name, model_vector_name
from app.tools.pdf_retriever import PDFSearchResult, retrieve_policy_chunks


class PDFRetrieverTests(unittest.TestCase):
    """Verify embedding, async queries, and whole-section expansion."""

    def test_retrieves_top_three_clause_payloads_asynchronously(self) -> None:
        """Query the model-specific collection and validate returned payloads."""
        embedding_model: MagicMock = MagicMock()
        embedding_model.query_embed.return_value = iter([[0.1, 0.2, 0.3]])
        hit_payload: dict[str, str | int] = {
            "source_file": "policy_handbook.pdf",
            "section_id": "3.2",
            "parent_section_id": "3",
            "section_title": "3. Bảo hành",
            "page": 1,
            "cite_id": "policy_handbook.pdf#3.2",
            "text": "Bảo hành điện tử cần được kích hoạt.",
            "document_version": "3.2",
            "effective_date": "2026-10-01",
        }
        other_hit_payload: dict[str, str | int] = {
            **hit_payload,
            "section_id": "3.3",
            "cite_id": "policy_handbook.pdf#3.3",
            "text": "Sản phẩm ngừng kinh doanh vẫn được bảo hành.",
        }
        section_records: list[SimpleNamespace] = [
            SimpleNamespace(
                payload={
                    **hit_payload,
                    "section_id": "3.1",
                    "cite_id": "policy_handbook.pdf#3.1",
                    "text": "3. Bảo hành\n3.1. Thời hạn bảo hành.",
                    "chunk_index": 0,
                }
            ),
            SimpleNamespace(
                payload={
                    **hit_payload,
                    "text": "3. Bảo hành\n3.2. Kích hoạt bảo hành.",
                    "chunk_index": 1,
                }
            ),
            SimpleNamespace(
                payload={
                    **other_hit_payload,
                    "chunk_index": 2,
                }
            ),
        ]
        qdrant_client: MagicMock = MagicMock()
        qdrant_client.query_points = AsyncMock(
            return_value=SimpleNamespace(
                points=[
                    SimpleNamespace(payload=hit_payload, score=0.91),
                    SimpleNamespace(payload=other_hit_payload, score=0.88),
                ]
            )
        )
        qdrant_client.scroll = AsyncMock(
            return_value=(section_records, None)
        )

        results: list[PDFSearchResult] = asyncio.run(
            retrieve_policy_chunks(
                qdrant_client=qdrant_client,
                embedding_model=embedding_model,
                question="Kích hoạt bảo hành như thế nào?",
                as_of=datetime.fromisoformat("2026-10-29T21:00:00+07:00"),
                base_collection_name="policy_documents",
                embedding_model_name="multilingual-model",
            )
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].cite_id, "policy_handbook.pdf#3.2")
        self.assertEqual(results[0].page, 1)
        self.assertEqual(results[0].score, 0.91)
        self.assertEqual(results[0].matched_section_id, "3.2")
        self.assertEqual(
            results[0].section_cite_ids,
            ["policy_handbook.pdf#3.1", "policy_handbook.pdf#3.2", "policy_handbook.pdf#3.3"],
        )
        self.assertIn("3.1. Thời hạn bảo hành.", results[0].text)
        self.assertIn("3.2. Kích hoạt bảo hành.", results[0].text)
        self.assertIn("Sản phẩm ngừng kinh doanh vẫn được bảo hành.", results[0].text)
        query_arguments: dict[str, object] = qdrant_client.query_points.call_args.kwargs
        self.assertEqual(
            query_arguments["collection_name"],
            model_collection_name("policy_documents", "multilingual-model"),
        )
        self.assertEqual(query_arguments["query"], [0.1, 0.2, 0.3])
        self.assertEqual(
            query_arguments["using"],
            model_vector_name("multilingual-model"),
        )
        self.assertEqual(query_arguments["limit"], 3)
        query_filter: models.Filter = query_arguments["query_filter"]
        self.assertEqual(len(query_filter.should), 2)
        qdrant_client.scroll.assert_awaited_once()
        scroll_filter: models.Filter = (
            qdrant_client.scroll.call_args.kwargs["scroll_filter"]
        )
        self.assertEqual(len(scroll_filter.must), 2)


if __name__ == "__main__":
    unittest.main()
