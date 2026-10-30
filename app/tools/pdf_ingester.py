"""Index extracted PDF pages in Qdrant with multilingual embeddings."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

from app.tools.pdf_reader import PDFDocument, read_pdf_directory

logger: logging.Logger = logging.getLogger(__name__)
CHUNK_SIZE: int = 1200
CHUNK_OVERLAP: int = 160
SCROLL_PAGE_SIZE: int = 256


class PDFIngestionResult(BaseModel):
    """Report the PDF pages and chunks stored in Qdrant.

    Attributes:
        processed_files: Source PDFs included in the replacement index.
        page_count: Total pages parsed from those PDFs.
        chunk_count: Number of non-empty text chunks vectorized.
        removed_count: Number of old chunks removed from the collection.
        warnings: Warnings from PDF extraction.
    """

    processed_files: int = Field(ge=0)
    page_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    removed_count: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


def _file_sha256(path: Path) -> str:
    """Return a digest of a PDF's bytes.

    Args:
        path: Source PDF to hash.

    Returns:
        str: Lowercase hexadecimal SHA-256 digest.

    Raises:
        OSError: If the PDF file cannot be read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_chunks(text: str) -> list[str]:
    """Split page text into overlapping chunks for embedding.

    Args:
        text: Extracted page text.

    Returns:
        list[str]: Non-empty chunks no longer than ``CHUNK_SIZE`` characters.
    """
    normalized_text: str = " ".join(text.split())
    chunks: list[str] = []
    start: int = 0
    while start < len(normalized_text):
        end: int = min(start + CHUNK_SIZE, len(normalized_text))
        if end < len(normalized_text):
            boundary: int = normalized_text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunk: str = normalized_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized_text):
            break
        start = max(start + 1, end - CHUNK_OVERLAP)
    return chunks


def _existing_points(
    qdrant_client: QdrantClient,
    collection_name: str,
) -> list[models.Record]:
    """Read all existing payloads and IDs from the PDF collection.

    Args:
        qdrant_client: Connected Qdrant async client.
        collection_name: Collection used for the PDF index.

    Returns:
        list[models.Record]: Existing points with payload and ID.

    Raises:
        ApiException: If Qdrant rejects the collection request.
        ResponseHandlingException: If Qdrant cannot be reached.
    """
    collections: models.CollectionsResponse = qdrant_client.get_collections()
    if collection_name not in {
        collection.name for collection in collections.collections
    }:
        return []

    points: list[models.Record] = []
    offset: models.ExtendedPointId | None = None
    while True:
        page, next_offset = qdrant_client.scroll(
            collection_name=collection_name,
            limit=SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(page)
        if next_offset is None:
            return points
        offset = next_offset


def ingest_pdf_directory(
    qdrant_client: QdrantClient,
    data_dir: Path,
    collection_name: str,
    embedding_model: str,
) -> PDFIngestionResult:
    """Read PDFs, embed their page chunks, and refresh the Qdrant index.

    New points are uploaded before obsolete points are deleted. Stable point
    IDs make retries idempotent and cleanup removes chunks for changed or
    deleted source files.

    Args:
        qdrant_client: Connected Qdrant async client.
        data_dir: Directory recursively searched for PDF files.
        collection_name: Dedicated Qdrant collection for PDF content.
        embedding_model: FastEmbed model used to vectorize page chunks.

    Returns:
        PDFIngestionResult: Counts of source files, pages, chunks, and removals.

    Raises:
        FileNotFoundError: If the data directory contains no PDF files.
        OSError: If a source file cannot be read.
        ValueError: If a source PDF is encrypted or the model name is invalid.
        ApiException: If Qdrant rejects a collection or point operation.
        ResponseHandlingException: If Qdrant cannot be reached.
    """
    documents: list[PDFDocument] = read_pdf_directory(data_dir)
    existing_points: list[models.Record] = _existing_points(
        qdrant_client,
        collection_name,
    )

    source_hashes: dict[str, str] = {
        document.source_file: _file_sha256(data_dir / document.source_file)
        for document in documents
    }
    embedding_documents: list[str] = []
    metadata: list[dict[str, str | int]] = []
    point_ids: list[str] = []
    warnings: list[str] = [
        warning for document in documents for warning in document.warnings
    ]
    page_count: int = sum(len(document.pages) for document in documents)

    for document in documents:
        file_hash: str = source_hashes[document.source_file]
        for page in document.pages:
            for chunk_index, chunk in enumerate(_text_chunks(page.text)):
                point_key: str = (
                    f"{document.source_file}:{file_hash}:"
                    f"{page.page_number}:{chunk_index}:{chunk}"
                )
                point_ids.append(str(uuid5(NAMESPACE_URL, point_key)))
                embedding_documents.append(chunk)
                metadata.append(
                    {
                        "source_file": document.source_file,
                        "page_number": page.page_number,
                        "cite_id": page.cite_id,
                        "chunk_index": chunk_index,
                        "sha256": file_hash,
                    }
                )

    if embedding_documents:
        qdrant_client.set_model(embedding_model)
        qdrant_client.add(
            collection_name=collection_name,
            documents=embedding_documents,
            metadata=metadata,
            ids=point_ids,
        )

    # Upload first, then remove old IDs so a failed embedding/upload does not
    # erase the last usable version of the PDF index.
    active_point_ids: set[str] = set(point_ids)
    stale_point_ids: list[models.ExtendedPointId] = [
        point.id
        for point in existing_points
        if str(point.id) not in active_point_ids
    ]
    if stale_point_ids:
        qdrant_client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=stale_point_ids),
            wait=True,
        )

    result: PDFIngestionResult = PDFIngestionResult(
        processed_files=len(documents),
        page_count=page_count,
        chunk_count=len(embedding_documents),
        removed_count=len(stale_point_ids),
        warnings=warnings,
    )
    logger.info(
        "PDF ingestion completed: %d files, %d pages, %d chunks, %d removed",
        result.processed_files,
        result.page_count,
        result.chunk_count,
        result.removed_count,
    )
    return result