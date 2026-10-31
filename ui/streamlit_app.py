"""Streamlit chat interface for the Lumen Home assistant API."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import httpx
import streamlit as st


DEFAULT_API_URL: str = "http://127.0.0.1:5000"
REQUEST_TIMEOUT: httpx.Timeout = httpx.Timeout(
    connect=15.0,
    read=600.0,
    write=30.0,
    pool=15.0,
)


def _api_url() -> str:
    """Return the configured Lumen API URL without a trailing slash.

    Returns:
        str: Base URL for the FastAPI service.
    """
    return os.getenv("LUMEN_API_URL", DEFAULT_API_URL).rstrip("/")


def _stream_answer(
    question: str,
    history: list[dict[str, str]],
) -> tuple[str, list[str], int | None]:
    """Ask the API a question and consume its server-sent events.

    Args:
        question: Current user question.
        history: Earlier user and assistant messages in this chat.

    Returns:
        tuple[str, list[str], int | None]: Answer text, citations, and API
            latency in milliseconds when supplied by the API.

    Raises:
        httpx.HTTPError: If the API cannot be reached or returns an HTTP error.
        ValueError: If an event contains malformed JSON or unexpected data.
    """
    answer_parts: list[str] = []
    citations: list[str] = []
    latency_ms: int | None = None
    event_name: str = ""
    payload: dict[str, Any] = {
        "question": question,
        "asked_at": datetime.now().astimezone().isoformat(),
        "history": history,
    }

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        with client.stream(
            "POST",
            f"{_api_url()}/ask",
            json=payload,
            headers={"Accept": "text/event-stream"},
        ) as response:
            if response.is_error:
                response.read()
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    event_name = ""
                    continue
                if line.startswith("event:"):
                    event_name = line.removeprefix("event:").strip()
                    continue
                if not line.startswith("data:"):
                    continue

                event_data: str = line.removeprefix("data:").strip()
                decoded_event: Any = json.loads(event_data)
                if event_name == "answer":
                    if not isinstance(decoded_event, dict):
                        raise ValueError("The API returned an invalid answer event.")
                    text_chunk: Any = decoded_event.get("text")
                    if not isinstance(text_chunk, str):
                        raise ValueError("The API returned an invalid answer chunk.")
                    answer_parts.append(text_chunk)
                elif event_name == "citations":
                    if not isinstance(decoded_event, list) or not all(
                        isinstance(cite_id, str) for cite_id in decoded_event
                    ):
                        raise ValueError("The API returned invalid citations.")
                    citations = decoded_event
                elif event_name == "done":
                    if not isinstance(decoded_event, dict):
                        raise ValueError("The API returned an invalid completion event.")
                    event_latency: Any = decoded_event.get("latency_ms")
                    if isinstance(event_latency, int):
                        latency_ms = event_latency

    return "".join(answer_parts), citations, latency_ms


def _render_citations(citations: list[str]) -> None:
    """Display source identifiers returned with an assistant answer.

    Args:
        citations: Source IDs supporting the answer.
    """
    if not citations:
        return
    with st.expander("Nguồn tham khảo"):
        for citation in citations:
            st.markdown(f"- `{citation}`")


def main() -> None:
    """Render the chat interface and manage conversation state."""
    st.set_page_config(
        page_title="Lumen Home Assistant",
        page_icon="💬",
        layout="centered",
    )
    st.title("Lumen Home")
    st.caption("Trợ lý chăm sóc khách hàng · Trả lời dựa trên tài liệu đã cung cấp")

    with st.sidebar:
        st.subheader("Cuộc trò chuyện")
        st.caption(f"API: `{_api_url()}`")
        if st.button("Xóa cuộc trò chuyện", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    messages: list[dict[str, Any]] = st.session_state.messages
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                citations_value: Any = message.get("citations", [])
                if isinstance(citations_value, list):
                    _render_citations(citations_value)

    question: str | None = st.chat_input("Bạn muốn tìm hiểu thông tin gì?")
    if question is None:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    previous_history: list[dict[str, str]] = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in st.session_state.messages[:-1]
        if message["role"] in ("user", "assistant")
    ]
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        try:
            with st.spinner("Đang tìm thông tin trong tài liệu..."):
                answer, citations, latency_ms = _stream_answer(
                    question=question,
                    history=previous_history,
                )
            if not answer:
                raise ValueError("API returned an empty answer.")
            answer_placeholder.markdown(answer)
            _render_citations(citations)
            if latency_ms is not None:
                st.caption(f"Thời gian xử lý: {latency_ms} ms")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                }
            )
        except httpx.HTTPStatusError as error:
            detail: str = error.response.text
            try:
                error_payload: Any = json.loads(detail)
            except json.JSONDecodeError:
                error_payload = None
            if isinstance(error_payload, dict):
                error_detail: Any = error_payload.get("detail")
                if isinstance(error_detail, str):
                    detail = error_detail
            st.error(
                f"API trả về lỗi HTTP {error.response.status_code}: {detail}"
            )
        except (httpx.HTTPError, ValueError, json.JSONDecodeError) as error:
            st.error(f"Không thể hoàn tất yêu cầu: {error}")


if __name__ == "__main__":
    main()
