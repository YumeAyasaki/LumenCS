"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path
from typing import Literal, TypeAlias

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr
from sqlalchemy.engine import URL


LLMProvider: TypeAlias = Literal[
    "gemini",
    "openai",
    "openai-compatible",
]


class Settings(BaseModel):
    """Hold connection settings for the application data stores.

    Attributes:
        database_url: SQLAlchemy async URL for PostgreSQL.
        qdrant_url: HTTP URL for the Qdrant instance.
        data_dir: Directory containing source workbooks and PDFs.
        qdrant_collection: Qdrant collection used for policy PDF chunks.
        pdf_embedding_model: Multilingual FastEmbed model used for PDF chunks.
        llm_provider: Provider type for OpenAI-compatible chat completions.
        llm_model: Provider model identifier.
        llm_base_url: OpenAI-compatible API base URL.
        llm_api_key: Optional secret API credential.
        llm_read_timeout_seconds: Maximum wait for provider response data.
        llm_input_cost_per_million: Optional input-token price in USD.
        llm_output_cost_per_million: Optional output-token price in USD.

    Raises:
        ValueError: If a required URL is empty.
    """

    database_url: str
    qdrant_url: str
    data_dir: Path = Path("data")
    qdrant_collection: str = "policy_documents"
    pdf_embedding_model: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    llm_provider: LLMProvider = "gemini"
    llm_model: str = "gemini-3.6-flash"
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    llm_api_key: SecretStr | None = None
    llm_read_timeout_seconds: float = Field(default=180.0, gt=0, le=600)
    llm_input_cost_per_million: Decimal | None = Field(default=None, ge=0)
    llm_output_cost_per_million: Decimal | None = Field(default=None, ge=0)

    @classmethod
    def from_environment(cls) -> Settings:
        """Create settings from process environment variables.

        Returns:
            Settings: PostgreSQL and Qdrant connection settings.

        Raises:
            ValueError: If either connection URL is empty.
        """
        load_dotenv(override=False)
        database_url: str | None = os.getenv("DATABASE_URL")
        if database_url is None:
            database_url = URL.create(
                drivername="postgresql+asyncpg",
                username=os.getenv("POSTGRES_USER", "lumen"),
                password=os.getenv("POSTGRES_PASSWORD", "lumen-local-dev"),
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=int(os.getenv("POSTGRES_PORT", "5432")),
                database=os.getenv("POSTGRES_DB", "lumen"),
            ).render_as_string(hide_password=False)
        qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
        if not database_url.strip():
            raise ValueError("DATABASE_URL must not be empty")
        if not qdrant_url.strip():
            raise ValueError("QDRANT_URL must not be empty")
        llm_provider_value: str = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
        if llm_provider_value not in ("gemini", "openai", "openai-compatible"):
            raise ValueError(
                "LLM_PROVIDER must be gemini, openai, or openai-compatible"
            )
        if llm_provider_value == "gemini":
            llm_provider: LLMProvider = "gemini"
            default_llm_base_url: str = (
                "https://generativelanguage.googleapis.com/v1beta/openai"
            )
            default_llm_model: str = "gemini-3.6-flash"
            provider_key: str | None = os.getenv("GEMINI_KEY")
        elif llm_provider_value == "openai":
            llm_provider = "openai"
            default_llm_base_url = "https://api.openai.com/v1"
            default_llm_model = "gpt-4o-mini"
            provider_key = os.getenv("OPENAI_API_KEY")
        else:
            llm_provider = "openai-compatible"
            default_llm_base_url = ""
            default_llm_model = ""
            provider_key = None
        input_cost: str | None = os.getenv("LLM_INPUT_COST_PER_MILLION_USD")
        output_cost: str | None = os.getenv("LLM_OUTPUT_COST_PER_MILLION_USD")
        return cls(
            database_url=database_url,
            qdrant_url=qdrant_url,
            data_dir=Path(os.getenv("DATA_DIR", "data")),
            qdrant_collection=os.getenv(
                "QDRANT_COLLECTION",
                "policy_documents",
            ),
            pdf_embedding_model=os.getenv(
                "PDF_EMBEDDING_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
            llm_provider=llm_provider,
            llm_model=os.getenv("LLM_MODEL") or default_llm_model,
            llm_base_url=(
                os.getenv("LLM_BASE_URL") or default_llm_base_url
            ).rstrip("/"),
            llm_api_key=(
                SecretStr(os.getenv("LLM_API_KEY") or provider_key or "")
                if (os.getenv("LLM_API_KEY") or provider_key or "").strip()
                else None
            ),
            llm_read_timeout_seconds=float(
                os.getenv("LLM_READ_TIMEOUT_SECONDS", "180")
            ),
            llm_input_cost_per_million=(
                Decimal(input_cost) if input_cost and input_cost.strip() else None
            ),
            llm_output_cost_per_million=(
                Decimal(output_cost)
                if output_cost and output_cost.strip()
                else None
            ),
        )
