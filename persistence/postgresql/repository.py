from typing import List
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.exc import IntegrityError
from tenacity import retry, stop_after_attempt, wait_exponential

from persistence.base import BaseDocumentRepository
from persistence.postgresql.models import Document, Page
from persistence.postgresql.client import PostgreSQLClient, get_client, DB_NOT_CONNECTED_ERROR
from persistence.schemas import DocumentCreate


class DocumentRepository(BaseDocumentRepository[Document]):
    """Repository for document operations using PostgreSQL."""

    def __init__(self, db_manager: PostgreSQLClient | None = None) -> None:
        """Initialize the repository."""
        self._db_manager = db_manager or get_client()
        if not self._db_manager.is_connected:
            raise RuntimeError(DB_NOT_CONNECTED_ERROR)

    @property
    def _session_factory(self) -> async_sessionmaker[AsyncSession]:
        return self._db_manager.db

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def save(self, document: Document) -> Document:
        """Save a document entity."""
        async with self._session_factory() as session:
            try:
                session.add(document)
                await session.commit()
                await session.refresh(document)
                return document
            except IntegrityError:
                await session.rollback()
                raise

    async def get_by_id(self, file_id: str, session: AsyncSession | None = None) -> Document | None:
        """Retrieve a document by file_id."""
        if session:
            result = await session.execute(select(Document).where(Document.file_id == file_id))
            return result.scalar_one_or_none()

        async with self._session_factory() as sess:
            result = await sess.execute(select(Document).where(Document.file_id == file_id))
            return result.scalar_one_or_none()

    async def get_by_file_name(
        self, file_name: str, session: AsyncSession | None = None
    ) -> Document | None:
        """Retrieve a document by file_name."""
        if session:
            result = await session.execute(select(Document).where(Document.file_name == file_name))
            return result.scalar_one_or_none()

        async with self._session_factory() as sess:
            result = await sess.execute(select(Document).where(Document.file_name == file_name))
            return result.scalar_one_or_none()

    async def list_all(self, skip: int = 0, limit: int = 100) -> List[Document]:
        """List all documents."""
        async with self._session_factory() as session:
            result = await session.execute(select(Document).offset(skip).limit(limit))
            return list(result.scalars().all())

    async def delete(self, file_id: str) -> bool:
        """Delete a document."""
        async with self._session_factory() as session:
            document = await self.get_by_id(file_id, session=session)
            if document:
                await session.delete(document)
                await session.commit()
                return True
            return False

    async def save_from_schema(self, schema: DocumentCreate) -> Document:
        """Consolidated upsert entry point using DocumentCreate schema."""
        pages = [
            Page(page_number=p.page_number, content=p.content, metadata_=p.metadata)
            for p in schema.pages
        ]
        
        async with self._session_factory() as session:
            # Upsert logic
            existing = await self.get_by_id(schema.file_id, session=session)
            if not existing:
                existing = await self.get_by_file_name(schema.file_name, session=session)

            if existing:
                existing.pages = pages
                existing.metadata_ = schema.metadata
                existing.updated_at = datetime.now()
                await session.commit()
                await session.refresh(existing)
                return existing

            document = Document(
                file_id=schema.file_id,
                file_name=schema.file_name,
                pages=pages,
                metadata_=schema.metadata,
            )
            session.add(document)
            await session.commit()
            await session.refresh(document)
            return document


class PageRepository:
    """Repository for page-level operations."""

    def __init__(self, db_manager: PostgreSQLClient | None = None) -> None:
        self._db_manager = db_manager or get_client()
        if not self._db_manager.is_connected:
            raise RuntimeError(DB_NOT_CONNECTED_ERROR)

    @property
    def _session_factory(self) -> async_sessionmaker[AsyncSession]:
        return self._db_manager.db

    async def get_page(
        self, file_id: str, page_number: int, session: AsyncSession | None = None
    ) -> Page | None:
        target_session = session
        if not target_session:
            async with self._session_factory() as sess:
                result = await sess.execute(
                    select(Page).where(Page.document_id == file_id, Page.page_number == page_number)
                )
                return result.scalar_one_or_none()
        
        result = await target_session.execute(
            select(Page).where(Page.document_id == file_id, Page.page_number == page_number)
        )
        return result.scalar_one_or_none()

    async def list_pages(self, file_id: str) -> List[Page]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(Page).where(Page.document_id == file_id).order_by(Page.page_number)
            )
            return list(result.scalars().all())

    async def update_page_content(self, file_id: str, page_number: int, content: str) -> bool:
        async with self._session_factory() as session:
            page = await self.get_page(file_id, page_number, session=session)
            if page:
                page.content = content
                await session.execute(
                    update(Document)
                    .where(Document.file_id == file_id)
                    .values(updated_at=datetime.now())
                )
                await session.commit()
                return True
            return False
