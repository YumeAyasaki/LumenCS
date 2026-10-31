"""Measure whether Vietnamese policy questions retrieve the expected clause."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import NamedTuple

from fastembed import TextEmbedding

from app.tools.pdf_ingester import PDFPolicyChunk, _policy_chunks
from app.tools.pdf_reader import read_pdf


class RetrievalCase(NamedTuple):
    """Describe one Vietnamese question and its expected policy clause."""

    question: str
    expected_section_id: str


CASES: tuple[RetrievalCase, ...] = (
    RetrievalCase(
        "Sổ tay này áp dụng cho sản phẩm nào và bán qua những kênh nào?",
        "1.1",
    ),
    RetrievalCase(
        "Nếu sổ tay mâu thuẫn với bảng tham chiếu thì áp dụng tài liệu nào?",
        "1.3",
    ),
    RetrievalCase(
        "Khách đổi ý nhưng sản phẩm không lỗi thì có được trả hàng không?",
        "2.3",
    ),
    RetrievalCase(
        "Những sản phẩm nào không được đổi trả do khách đổi ý?",
        "2.4",
    ),
    RetrievalCase(
        "Đơn mua từ 01/10/2026 có bao nhiêu ngày kích hoạt bảo hành, "
        "và nếu không kích hoạt thì linh kiện chính bảo hành bao lâu?",
        "3.2",
    ),
    RetrievalCase(
        "Trường hợp nào bị từ chối bảo hành nếu tem hoặc số serial có vấn đề?",
        "4.bullet-1",
    ),
    RetrievalCase(
        "Lịch lắp đặt nội thành là bao lâu sau khi giao hàng?",
        "5.3",
    ),
    RetrievalCase(
        "Khi nào được miễn phí kiểm tra chẩn đoán tại nhà ngoài bảo hành?",
        "6.1",
    ),
    RetrievalCase(
        "Hotline hoạt động vào những giờ nào và ngày nào?",
        "7",
    ),
    RetrievalCase(
        "Khách được bồi thường thế nào nếu kỹ thuật viên trễ hơn hai giờ "
        "mà không báo trước?",
        "8.1",
    ),
)


def _vector_values(model: TextEmbedding, text: str, *, query: bool) -> list[float]:
    """Embed text as a query or document using the model's task-specific path.

    Args:
        model: Loaded multilingual embedding model.
        text: Input text to embed.
        query: Whether the text is a search query rather than a document.

    Returns:
        list[float]: Embedding values.

    Raises:
        StopIteration: If the model does not return an embedding.
    """
    embeddings = model.query_embed(text) if query else model.passage_embed(text)
    return [float(value) for value in next(embeddings)]


def _cosine_similarity(first: list[float], second: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors.

    Args:
        first: First vector.
        second: Second vector with the same dimensionality.

    Returns:
        float: Cosine similarity score.
    """
    dot_product: float = sum(left * right for left, right in zip(first, second))
    first_norm: float = math.sqrt(sum(value * value for value in first))
    second_norm: float = math.sqrt(sum(value * value for value in second))
    return dot_product / (first_norm * second_norm)


def evaluate(model_name: str, pdf_path: Path, limit: int) -> dict[str, object]:
    """Run the Vietnamese clause-retrieval evaluation.

    Args:
        model_name: FastEmbed model used for document and query embeddings.
        pdf_path: Policy handbook PDF path.
        limit: Number of top-ranked chunks checked for each expected clause.

    Returns:
        dict[str, object]: Per-question top results and aggregate recall.

    Raises:
        FileNotFoundError: If the handbook PDF does not exist.
        ValueError: If the model or source PDF cannot be read.
    """
    document = read_pdf(pdf_path)
    chunks: list[PDFPolicyChunk] = _policy_chunks(document)
    model: TextEmbedding = TextEmbedding(model_name=model_name)
    document_vectors: list[list[float]] = [
        _vector_values(model, chunk.text, query=False)
        for chunk in chunks
    ]
    results: list[dict[str, object]] = []
    hits: int = 0
    for case in CASES:
        query_vector: list[float] = _vector_values(
            model,
            case.question,
            query=True,
        )
        ranked: list[tuple[float, PDFPolicyChunk]] = sorted(
            (
                (_cosine_similarity(query_vector, vector), chunk)
                for chunk, vector in zip(chunks, document_vectors)
            ),
            key=lambda result: result[0],
            reverse=True,
        )
        top_results: list[dict[str, object]] = [
            {
                "section_id": chunk.section_id,
                "cite_id": chunk.cite_id,
                "page": chunk.page,
                "score": round(score, 4),
            }
            for score, chunk in ranked[:limit]
        ]
        hit: bool = any(
            chunk.section_id == case.expected_section_id
            for _, chunk in ranked[:limit]
        )
        hits += int(hit)
        results.append(
            {
                "question": case.question,
                "expected_section_id": case.expected_section_id,
                "hit": hit,
                "top_results": top_results,
            }
        )
    return {
        "model": model_name,
        "cases": len(CASES),
        "recall_at_k": round(hits / len(CASES), 3),
        "k": limit,
        "results": results,
    }


def main() -> None:
    """Run and print the retrieval evaluation as formatted JSON.

    Returns:
        None: JSON output is written to standard output.

    Raises:
        SystemExit: If command-line arguments are invalid.
        FileNotFoundError: If the policy handbook is missing.
        ValueError: If the model or PDF cannot be read.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("data/policy_handbook.pdf"),
    )
    parser.add_argument("--k", type=int, default=3)
    arguments: argparse.Namespace = parser.parse_args()
    print(
        json.dumps(
            evaluate(arguments.model, arguments.pdf, arguments.k),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
