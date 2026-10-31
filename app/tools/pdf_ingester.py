"""Index clause-level PDF chunks in Qdrant with multilingual embeddings."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

from app.tools.pdf_reader import PDFDocument, PDFElement, PDFPage, read_pdf_directory

logger: logging.Logger = logging.getLogger(__name__)
CHUNK_SIZE: int = 1200
CHUNK_OVERLAP: int = 160
SCROLL_PAGE_SIZE: int = 256
SECTION_PATTERN: re.Pattern[str] = re.compile(r"^\s*(\d+)\.\s+(?!\d)(.+?)\s*$")
CLAUSE_PATTERN: re.Pattern[str] = re.compile(r"^\s*(\d+(?:\.\d+)+)\.\s*(.*)$")
APPENDIX_PATTERN: re.Pattern[str] = re.compile(
    r"^\s*(Phụ lục\s+[A-Za-z0-9]+)\s*(?:[–—-]\s*(.*))?$",
    re.IGNORECASE,
)
BULLET_PATTERN: re.Pattern[str] = re.compile(r"^\s*([•●▪◦‣*-])\s+(.+?)\s*$")
PAGE_FOOTER_PATTERN: re.Pattern[str] = re.compile(
    r"^\s*Lumen Home\s*[–—-]\s*Sổ tay CSKH v[\d.]+\s*·\s*\d+/\d+\s*$",
    re.IGNORECASE,
)


class PDFPolicyChunk(BaseModel):
    """Represent a citeable clause or table with its original provenance.

    Attributes:
        source_file: Relative path to the original PDF.
        section_id: Numbered clause, section, bullet, or appendix identifier.
        section_title: Parent section title from the handbook.
        page: One-based source PDF page number.
        cite_id: Source filename and clause anchor.
        text: Section-prefixed text, preserving source line breaks and table rows.
        document_version: Handbook version parsed from its header, if present.
        effective_date: ISO-format effective date parsed from its header.

    Raises:
        ValidationError: If a field does not match the model.
    """

    source_file: str
    section_id: str
    section_title: str
    page: int = Field(ge=1)
    cite_id: str
    text: str
    document_version: str | None = None
    effective_date: str | None = None


class PDFIngestionResult(BaseModel):
    """Report the PDF chunks and source files stored in Qdrant.

    Attributes:
        processed_files: Source PDFs included in the replacement index.
        page_count: Total pages parsed from those PDFs.
        chunk_count: Number of non-empty text chunks vectorized.
        removed_count: Number of old chunks removed from the collection.
        collection_name: Model-specific collection receiving the chunks.
        warnings: Warnings from PDF extraction.
    """

    processed_files: int = Field(ge=0)
    page_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    removed_count: int = Field(ge=0)
    collection_name: str
    warnings: list[str] = Field(default_factory=list)


def model_collection_name(base_name: str, embedding_model: str) -> str:
    """Return a Qdrant collection name isolated to one embedding model.

    Args:
        base_name: Configured base collection name.
        embedding_model: Fully qualified FastEmbed model name.

    Returns:
        str: Collection name with a readable model slug and hash suffix.
    """
    model_slug: str = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        embedding_model.rsplit("/", maxsplit=1)[-1],
    ).strip("_")
    model_hash: str = hashlib.sha256(embedding_model.encode("utf-8")).hexdigest()[:10]
    suffix: str = f"_{model_slug}_{model_hash}"
    return f"{base_name[: 255 - len(suffix)]}{suffix}"


def model_vector_name(embedding_model: str) -> str:
    """Return Qdrant's named-vector key for a FastEmbed model.

    Args:
        embedding_model: Fully qualified FastEmbed model name.

    Returns:
        str: Lowercase FastEmbed vector name used by Qdrant.
    """
    model_slug: str = embedding_model.rsplit("/", maxsplit=1)[-1].lower()
    return f"fast-{model_slug}"


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
    """Split long text while preserving line breaks and overlapping context.

    Args:
        text: One clause, table, or unnumbered document's text.

    Returns:
        list[str]: Non-empty chunks no longer than ``CHUNK_SIZE`` characters.
    """
    source_text: str = text.strip()
    chunks: list[str] = []
    start: int = 0
    while start < len(source_text):
        end: int = min(start + CHUNK_SIZE, len(source_text))
        if end < len(source_text):
            boundary: int = max(
                source_text.rfind("\n", start, end),
                source_text.rfind(" ", start, end),
            )
            if boundary > start + CHUNK_SIZE // 2:
                end = boundary
        chunk: str = source_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(source_text):
            break
        start = max(start + 1, end - CHUNK_OVERLAP)
    return chunks


def _policy_chunks(document: PDFDocument) -> list[PDFPolicyChunk]:
    """Split a PDF into section-prefixed clause, bullet, and table chunks.

    Args:
        document: Extracted PDF document and its source metadata.

    Returns:
        list[PDFPolicyChunk]: Ordered chunks with clause citations and page data.
    """
    chunks: list[PDFPolicyChunk] = []
    parent_section_id: str = "document"
    section_id: str = parent_section_id
    section_title: str = Path(document.source_file).stem
    active_clause_id: str | None = None
    active_citation_id: str = section_id
    active_lines: list[str] = []
    active_kind: str = "section"
    bullet_counts: dict[str, int] = {}

    def flush_active() -> None:
        """Append the active text block as size-bounded section chunks.

        Returns:
            None: Active lines are added to the enclosing ``chunks`` list.
        """
        nonlocal active_lines
        if not active_lines:
            return
        prefix: str = section_title
        body: str = "\n".join(active_lines).strip()
        content: str = f"{prefix}\n{body}" if body else prefix
        for text in _text_chunks(content):
            chunks.append(
                PDFPolicyChunk(
                    source_file=document.source_file,
                    section_id=section_id,
                    section_title=section_title,
                    page=current_page.page_number,
                    cite_id=f"{document.source_file}#{active_citation_id}",
                    text=text,
                    document_version=document.document_version,
                    effective_date=(
                        document.effective_date.isoformat()
                        if document.effective_date
                        else None
                    ),
                )
            )
        active_lines = []

    def add_table(element: PDFElement) -> None:
        """Append a table as its own clause-cited chunk.

        Args:
            element: Table element with formatted row text.

        Returns:
            None: Table chunks are added to the enclosing ``chunks`` list.
        """
        table_id: str = active_clause_id or section_id
        citation_id: str = active_clause_id or (
            f"{section_id}.table-{element.table_index or 1}"
        )
        table_prefix: str = section_title
        if active_clause_id:
            table_prefix = f"{table_prefix}\n{active_clause_id}."
        table_text: str = f"{table_prefix}\n{element.text}"
        for text in _text_chunks(table_text):
            chunks.append(
                PDFPolicyChunk(
                    source_file=document.source_file,
                    section_id=table_id,
                    section_title=section_title,
                    page=current_page.page_number,
                    cite_id=f"{document.source_file}#{citation_id}",
                    text=text,
                    document_version=document.document_version,
                    effective_date=(
                        document.effective_date.isoformat()
                        if document.effective_date
                        else None
                    ),
                )
            )

    for current_page in document.pages:
        elements: list[PDFElement] = current_page.elements or [
            PDFElement(kind="text", text=current_page.text)
        ]
        for element in elements:
            if element.kind == "table":
                flush_active()
                add_table(element)
                continue

            if active_kind == "bullet":
                flush_active()
                active_kind = "section"
                section_id = parent_section_id
                active_clause_id = None
                active_citation_id = parent_section_id

            for raw_line in element.text.splitlines():
                line: str = raw_line.strip()
                if not line or PAGE_FOOTER_PATTERN.match(line):
                    continue

                appendix_match: re.Match[str] | None = APPENDIX_PATTERN.match(line)
                if appendix_match:
                    flush_active()
                    appendix_key: str = appendix_match.group(1).split()[-1].lower()
                    parent_section_id = f"appendix-{appendix_key}"
                    section_id = parent_section_id
                    section_title = line
                    active_clause_id = None
                    active_citation_id = section_id
                    active_kind = "section"
                    if appendix_match.group(2):
                        active_lines.append(appendix_match.group(2))
                    continue

                clause_match: re.Match[str] | None = CLAUSE_PATTERN.match(line)
                if clause_match:
                    flush_active()
                    active_clause_id = clause_match.group(1)
                    section_id = active_clause_id
                    active_citation_id = active_clause_id
                    active_kind = "clause"
                    active_lines.append(line)
                    continue

                section_match: re.Match[str] | None = SECTION_PATTERN.match(line)
                if section_match:
                    flush_active()
                    parent_section_id = section_match.group(1)
                    section_id = parent_section_id
                    section_title = f"{section_id}. {section_match.group(2)}"
                    active_clause_id = None
                    active_citation_id = section_id
                    active_kind = "section"
                    continue

                bullet_match: re.Match[str] | None = BULLET_PATTERN.match(line)
                if bullet_match:
                    flush_active()
                    bullet_number: int = (
                        bullet_counts.get(parent_section_id, 0) + 1
                    )
                    bullet_counts[parent_section_id] = bullet_number
                    bullet_id: str = f"{parent_section_id}.bullet-{bullet_number}"
                    active_clause_id = bullet_id
                    section_id = bullet_id
                    active_citation_id = bullet_id
                    active_kind = "bullet"
                    active_lines.append(line)
                    continue

                active_lines.append(line)
        flush_active()

    return chunks


def _existing_points(
    qdrant_client: QdrantClient,
    collection_name: str,
) -> list[models.Record]:
    """Read all existing payloads and IDs from the PDF collection.

    Args:
        qdrant_client: Connected Qdrant synchronous client.
        collection_name: Model-specific collection used for the PDF index.

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
    """Read PDFs, embed semantic chunks, and refresh a model-specific index.

    New points are uploaded before obsolete points are deleted. Stable point
    IDs include the document digest and model name, making retries idempotent
    without colliding when the embedding model changes.

    Args:
        qdrant_client: Connected Qdrant synchronous client.
        data_dir: Directory recursively searched for PDF files.
        collection_name: Base name for the model-specific Qdrant collection.
        embedding_model: FastEmbed model used to vectorize PDF chunks.

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
    resolved_collection: str = model_collection_name(
        collection_name,
        embedding_model,
    )
    existing_points: list[models.Record] = _existing_points(
        qdrant_client,
        resolved_collection,
    )

    source_hashes: dict[str, str] = {
        document.source_file: _file_sha256(data_dir / document.source_file)
        for document in documents
    }
    embedding_documents: list[str] = []
    metadata: list[dict[str, str | int | None]] = []
    point_ids: list[str] = []
    warnings: list[str] = [
        warning for document in documents for warning in document.warnings
    ]
    page_count: int = sum(len(document.pages) for document in documents)

    for document in documents:
        file_hash: str = source_hashes[document.source_file]
        for chunk_index, policy_chunk in enumerate(_policy_chunks(document)):
            text_hash: str = hashlib.sha256(
                policy_chunk.text.encode("utf-8")
            ).hexdigest()
            point_key: str = (
                f"{document.source_file}:{file_hash}:{embedding_model}:"
                f"{policy_chunk.cite_id}:{policy_chunk.page}:{chunk_index}:{text_hash}"
            )
            point_ids.append(str(uuid5(NAMESPACE_URL, point_key)))
            embedding_documents.append(policy_chunk.text)
            metadata.append(
                {
                    **policy_chunk.model_dump(mode="json"),
                    "sha256": file_hash,
                    "chunk_sha256": text_hash,
                    "chunk_index": chunk_index,
                    "embedding_model": embedding_model,
                    "effective_date_ordinal": (
                        document.effective_date.toordinal()
                        if document.effective_date
                        else None
                    ),
                }
            )

    if embedding_documents:
        qdrant_client.set_model(embedding_model)
        qdrant_client.add(
            collection_name=resolved_collection,
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
            collection_name=resolved_collection,
            points_selector=models.PointIdsList(points=stale_point_ids),
            wait=True,
        )

    result: PDFIngestionResult = PDFIngestionResult(
        processed_files=len(documents),
        page_count=page_count,
        chunk_count=len(embedding_documents),
        removed_count=len(stale_point_ids),
        collection_name=resolved_collection,
        warnings=warnings,
    )
    logger.info(
        "PDF ingestion completed: %d files, %d pages, %d chunks, %d removed "
        "into %s",
        result.processed_files,
        result.page_count,
        result.chunk_count,
        result.removed_count,
        result.collection_name,
    )
    return result
