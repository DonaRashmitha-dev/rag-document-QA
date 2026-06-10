# MIT License
# Copyright (c) 2026 RAG-QA Contributors

"""SQLite/PostgreSQL metadata store for document chunk provenance."""

import uuid
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""


class DocumentRecord(Base):
    """ORM model for ingested document metadata."""

    __tablename__ = "documents"

    doc_id = Column(String(36), primary_key=True)
    filename = Column(String(512), nullable=False)
    num_chunks = Column(Integer, nullable=False, default=0)
    file_type = Column(String(16), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)


class ChunkRecord(Base):
    """ORM model for individual chunk provenance."""

    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String(36), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    filename = Column(String(512), nullable=False)
    page = Column(Integer, nullable=True)
    content_hash = Column(String(64), nullable=False, index=True)
    content_preview = Column(Text, nullable=True)


class MetadataStore:
    """CRUD operations for document and chunk metadata."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the metadata store with a SQLAlchemy engine.

        Args:
            settings: Application settings with database_url.
        """
        self.settings = settings or get_settings()
        self.engine = create_engine(
            self.settings.database_url,
            connect_args={"check_same_thread": False}
            if self.settings.database_url.startswith("sqlite")
            else {},
        )
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
        Base.metadata.create_all(self.engine)

    def _session(self) -> Session:
        """Create a new database session.

        Returns:
            SQLAlchemy Session instance.
        """
        return self.SessionLocal()

    @staticmethod
    def generate_doc_id() -> str:
        """Generate a short unique document identifier.

        Returns:
            8-character hex doc_id.
        """
        return uuid.uuid4().hex[:8]

    def insert_document(
        self,
        doc_id: str,
        filename: str,
        num_chunks: int,
        file_type: str,
    ) -> dict[str, Any]:
        """Insert a document metadata record.

        Args:
            doc_id: Unique document identifier.
            filename: Original filename.
            num_chunks: Number of chunks indexed.
            file_type: File extension.

        Returns:
            Dictionary representation of the inserted record.
        """
        now = datetime.now(UTC)
        record = DocumentRecord(
            doc_id=doc_id,
            filename=filename,
            num_chunks=num_chunks,
            file_type=file_type,
            created_at=now,
            updated_at=now,
        )
        with self._session() as session:
            session.add(record)
            session.commit()
            session.refresh(record)

        logger.info("document_metadata_inserted", doc_id=doc_id, num_chunks=num_chunks)
        return self._document_to_dict(record)

    def insert_chunks(
        self,
        doc_id: str,
        filename: str,
        chunks: list[dict[str, Any]],
    ) -> int:
        """Insert chunk provenance records for a document.

        Args:
            doc_id: Parent document identifier.
            filename: Source filename.
            chunks: List of dicts with chunk_index, page, content_hash, content_preview.

        Returns:
            Number of chunk records inserted.
        """
        records = [
            ChunkRecord(
                doc_id=doc_id,
                chunk_index=chunk["chunk_index"],
                filename=filename,
                page=chunk.get("page"),
                content_hash=chunk["content_hash"],
                content_preview=chunk.get("content_preview", "")[:200],
            )
            for chunk in chunks
        ]
        with self._session() as session:
            session.add_all(records)
            session.commit()

        logger.info("chunk_metadata_inserted", doc_id=doc_id, count=len(records))
        return len(records)

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """Retrieve a document record by doc_id.

        Args:
            doc_id: Document identifier.

        Returns:
            Document dict or None if not found.
        """
        with self._session() as session:
            record = session.get(DocumentRecord, doc_id)
            if record is None:
                return None
            return self._document_to_dict(record)

    def list_documents(self) -> list[dict[str, Any]]:
        """List all document records.

        Returns:
            List of document metadata dicts.
        """
        with self._session() as session:
            stmt = select(DocumentRecord).order_by(DocumentRecord.created_at.desc())
            results = session.execute(stmt)
            return [self._document_to_dict(row) for row in results.scalars()]

    def get_chunks_for_doc(self, doc_id: str) -> list[dict[str, Any]]:
        """Retrieve all chunk records for a document.

        Args:
            doc_id: Document identifier.

        Returns:
            List of chunk metadata dicts.
        """
        with self._session() as session:
            stmt = (
                select(ChunkRecord)
                .where(ChunkRecord.doc_id == doc_id)
                .order_by(ChunkRecord.chunk_index)
            )
            results = session.execute(stmt)
            return [self._chunk_to_dict(row) for row in results.scalars()]

    @staticmethod
    def _document_to_dict(record: DocumentRecord) -> dict[str, Any]:
        """Convert a DocumentRecord ORM object to a dict.

        Args:
            record: ORM document record.

        Returns:
            Serializable dictionary.
        """
        return {
            "doc_id": record.doc_id,
            "filename": record.filename,
            "num_chunks": record.num_chunks,
            "file_type": record.file_type,
            "created_at": record.created_at.isoformat() if record.created_at else None,
            "updated_at": record.updated_at.isoformat() if record.updated_at else None,
        }

    @staticmethod
    def _chunk_to_dict(record: ChunkRecord) -> dict[str, Any]:
        """Convert a ChunkRecord ORM object to a dict.

        Args:
            record: ORM chunk record.

        Returns:
            Serializable dictionary.
        """
        return {
            "doc_id": record.doc_id,
            "chunk_index": record.chunk_index,
            "filename": record.filename,
            "page": record.page,
            "content_hash": record.content_hash,
            "content_preview": record.content_preview,
        }


_metadata_store: MetadataStore | None = None


def get_metadata_store(settings: Settings | None = None) -> MetadataStore:
    """Return the global MetadataStore singleton.

    Args:
        settings: Optional settings for first initialization.

    Returns:
        MetadataStore instance.
    """
    global _metadata_store
    if _metadata_store is None:
        _metadata_store = MetadataStore(settings)
    return _metadata_store


def reset_metadata_store() -> None:
    """Reset the global metadata store singleton (used in tests)."""
    global _metadata_store
    _metadata_store = None
