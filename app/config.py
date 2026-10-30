"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel
from sqlalchemy.engine import URL


class Settings(BaseModel):
    """Hold connection settings for the application data stores.

    Attributes:
        database_url: SQLAlchemy async URL for PostgreSQL.
        qdrant_url: HTTP URL for the Qdrant instance.
        data_dir: Directory containing source workbooks and PDFs.
        qdrant_collection: Qdrant collection used for policy PDF chunks.
        pdf_embedding_model: Multilingual FastEmbed model used for PDF chunks.

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

    @classmethod
    def from_environment(cls) -> Settings:
        """Create settings from process environment variables.

        Returns:
            Settings: PostgreSQL and Qdrant connection settings.

        Raises:
            ValueError: If either connection URL is empty.
        """
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
        )
