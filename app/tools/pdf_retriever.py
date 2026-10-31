"""Retrieve policy clauses asynchronously from the model-specific Qdrant index."""

from __future__ import annotations

import asyncio
from datetime import datetime

from fastembed import TextEmbedding
from pydantic import Field
from qdrant_client import AsyncQdrantClient, models

from app.tools.pdf_ingester import (
    PDFPolicyChunk,
    model_collection_name,
    model_vector_name,
)

RESULT_LIMIT: int = 3


class PDFSearchResult(PDFPolicyChunk):
    """Represent a retrieved policy section and its best matching clause.

    Attributes:
        score: Similarity score returned by Qdrant.
        matched_section_id: Clause identifier for the semantic match.
        section_cite_ids: Clause citations represented in the expanded section.
    """

    score: float = Field(ge=0)
    matched_section_id: str
    section_cite_ids: list[str]


def _embed_question(embedding_model: TextEmbedding, question: str) -> list[float]:
    """Create one query vector using the model's query embedding method.

    Args:
        embedding_model: Loaded FastEmbed model.
        question: User query to embed.

    Returns:
        list[float]: Query vector in the model's native dimensions.

    Raises:
        StopIteration: If the embedding model produces no vector.
        ValueError: If the query cannot be embedded.
    """
    vector = next(embedding_model.query_embed(question))
    return [float(value) for value in vector]


def _section_filter(
    source_file: str,
    parent_section_id: str,
    as_of: datetime,
) -> models.Filter:
    """Build a filter for all effective chunks in one source section.

    Args:
        source_file: Relative path identifying the source PDF.
        parent_section_id: Top-level section identifier.
        as_of: Timestamp used to exclude policies not yet effective.

    Returns:
        models.Filter: Qdrant filter for the requested source section.
    """
    return models.Filter(
        must=[
            models.FieldCondition(
                key="source_file",
                match=models.MatchValue(value=source_file),
            ),
            models.FieldCondition(
                key="parent_section_id",
                match=models.MatchValue(value=parent_section_id),
            ),
        ],
        should=[
            models.FieldCondition(
                key="effective_date_ordinal",
                range=models.Range(lte=as_of.date().toordinal()),
            ),
            models.IsEmptyCondition(
                is_empty=models.PayloadField(key="effective_date_ordinal")
            ),
        ],
    )


def _merge_section_text(
    records: list[models.Record],
    section_title: str,
) -> str:
    """Assemble stored clause chunks into their complete parent section.

    Args:
        records: Section points returned by Qdrant scroll.
        section_title: Section heading to include once at the beginning.

    Returns:
        str: Ordered section text containing every clause and table row.
    """
    ordered_records: list[models.Record] = sorted(
        records,
        key=lambda record: (
            int(record.payload.get("chunk_index", 0))
            if record.payload
            else 0
        ),
    )
    chunks_by_citation: dict[str, list[str]] = {}
    for record in ordered_records:
        if record.payload is None:
            continue
        citation_id: str = str(record.payload.get("cite_id", ""))
        text: str = str(record.payload.get("text", "")).strip()
        if text:
            chunks_by_citation.setdefault(citation_id, []).append(text)

    section_parts: list[str] = [section_title]
    for chunks in chunks_by_citation.values():
        merged: str = chunks[0]
        for chunk in chunks[1:]:
            maximum_overlap: int = min(len(merged), len(chunk), 160)
            overlap: int = next(
                (
                    size
                    for size in range(maximum_overlap, 0, -1)
                    if merged.endswith(chunk[:size])
                ),
                0,
            )
            merged += chunk[overlap:] if overlap else f"\n{chunk}"
        lines: list[str] = merged.splitlines()
        if lines and lines[0].strip() == section_title:
            merged = "\n".join(lines[1:]).strip()
        if merged:
            section_parts.append(merged)
    return "\n".join(section_parts)


def _section_citations(records: list[models.Record]) -> list[str]:
    """Return unique clause citations represented by section records.

    Args:
        records: Section points returned by Qdrant scroll.

    Returns:
        list[str]: Clause citations in source order.
    """
    ordered_records: list[models.Record] = sorted(
        records,
        key=lambda record: (
            int(record.payload.get("chunk_index", 0))
            if record.payload
            else 0
        ),
    )
    return list(
        dict.fromkeys(
            str(record.payload["cite_id"])
            for record in ordered_records
            if record.payload and record.payload.get("cite_id")
        )
    )


async def retrieve_policy_chunks(
    qdrant_client: AsyncQdrantClient,
    embedding_model: TextEmbedding,
    question: str,
    as_of: datetime,
    base_collection_name: str,
    embedding_model_name: str,
) -> list[PDFSearchResult]:
    """Retrieve matching clauses, expanding each hit to its whole section.

    Args:
        qdrant_client: Connected asynchronous Qdrant client.
        embedding_model: FastEmbed model initialized during application startup.
        question: User query to match against indexed policy chunks.
        as_of: Timestamp used to exclude policies not yet effective.
        base_collection_name: Configured base name for the policy collection.
        embedding_model_name: Name identifying the model used to build the index.

    Returns:
        list[PDFSearchResult]: Up to three distinct whole sections, each retaining
            the best matching clause citation and page.

    Raises:
        ApiException: If Qdrant rejects the query.
        ResponseHandlingException: If Qdrant cannot be reached.
        ValidationError: If a returned payload is malformed.
    """
    query_vector: list[float] = await asyncio.to_thread(
        _embed_question,
        embedding_model,
        question,
    )
    collection_name: str = model_collection_name(
        base_collection_name,
        embedding_model_name,
    )
    response = await qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        using=model_vector_name(embedding_model_name),
        query_filter=models.Filter(
            should=[
                models.FieldCondition(
                    key="effective_date_ordinal",
                    range=models.Range(lte=as_of.date().toordinal()),
                ),
                models.IsEmptyCondition(
                    is_empty=models.PayloadField(key="effective_date_ordinal")
                ),
            ]
        ),
        limit=RESULT_LIMIT,
        with_payload=True,
        with_vectors=False,
    )
    results: list[PDFSearchResult] = []
    expanded_sections: set[tuple[str, str]] = set()
    for point in response.points:
        if point.payload is None:
            continue
        source_file: str = str(point.payload["source_file"])
        parent_section_id: str = str(point.payload["parent_section_id"])
        section_key: tuple[str, str] = (source_file, parent_section_id)
        if section_key in expanded_sections:
            continue

        section_records: list[models.Record] = []
        offset: models.ExtendedPointId | None = None
        while True:
            section_page, next_offset = await qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=_section_filter(
                    source_file,
                    parent_section_id,
                    as_of,
                ),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            section_records.extend(section_page)
            if next_offset is None:
                break
            offset = next_offset

        expanded_sections.add(section_key)
        expanded_payload: dict[str, object] = {
            **point.payload,
            "text": _merge_section_text(
                section_records,
                str(point.payload["section_title"]),
            ),
            "section_cite_ids": _section_citations(section_records),
            "score": float(point.score),
            "matched_section_id": str(point.payload["section_id"]),
        }
        results.append(
            PDFSearchResult.model_validate(expanded_payload)
        )
    return results
