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
    """Represent a retrieved policy chunk and its vector similarity.

    Attributes:
        score: Similarity score returned by Qdrant.
    """

    score: float = Field(ge=0)


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


async def retrieve_policy_chunks(
    qdrant_client: AsyncQdrantClient,
    embedding_model: TextEmbedding,
    question: str,
    as_of: datetime,
    base_collection_name: str,
    embedding_model_name: str,
) -> list[PDFSearchResult]:
    """Embed a question and asynchronously retrieve its top policy clauses.

    Args:
        qdrant_client: Connected asynchronous Qdrant client.
        embedding_model: FastEmbed model initialized during application startup.
        question: User query to match against indexed policy chunks.
        as_of: Timestamp used to exclude policies not yet effective.
        base_collection_name: Configured base name for the policy collection.
        embedding_model_name: Name identifying the model used to build the index.

    Returns:
        list[PDFSearchResult]: Up to three results with full provenance payloads.

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
    for point in response.points:
        if point.payload is None:
            continue
        results.append(
            PDFSearchResult.model_validate(
                {
                    **point.payload,
                    "score": float(point.score),
                }
            )
        )
    return results
