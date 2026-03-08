from datetime import datetime
from typing import List, Dict, Any

from sqlalchemy import String, DateTime, ForeignKey, Integer, Text, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


class Page(Base):
    """Model representing a single page within a document."""

    __tablename__ = "pages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(
        String, ForeignKey("documents.file_id", ondelete="CASCADE"), nullable=False
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)

    # Relationship back to document
    document: Mapped["Document"] = relationship("Document", back_populates="pages")

    __table_args__ = (
        Index("ix_pages_document_id", "document_id"),
        Index("ix_pages_page_number", "page_number"),
    )

    def __repr__(self) -> str:
        return (
            f"<Page(id={self.id}, document_id={self.document_id}, page_number={self.page_number})>"
        )


class Document(Base):
    """Model representing a document with one or multiple pages."""

    __tablename__ = "documents"

    file_id: Mapped[str] = mapped_column(String, primary_key=True)
    file_name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now, nullable=False
    )
    metadata_: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)

    # Relationship to pages - one document has many pages
    pages: Mapped[List["Page"]] = relationship(
        "Page",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        Index("ix_documents_file_id", "file_id", unique=True),
        Index("ix_documents_file_name", "file_name", unique=True),
    )

    def __repr__(self) -> str:
        return f"<Document(file_id={self.file_id}, file_name={self.file_name})>"
