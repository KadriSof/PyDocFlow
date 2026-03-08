from typing import List, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.exc import IntegrityError
from tenacity import retry, stop_after_attempt, wait_exponential

from persistence.base import BaseRepository
from persistence.postgresql.models import Document, Page
from persistence.postgresql.client import PostgreSQLClient, get_client, DB_NOT_CONNECTED_ERROR


class DocumentRepository(BaseRepository[Document]):
    """Repository for document operations."""

    def __init__(self, db_manager: PostgreSQLClient | None = None) -> None:
        """
        Initialize the repository with a database manager.

        Args:
            db_manager: DatabaseManager instance. If None, uses the global instance.

        Raises:
            RuntimeError: If db_manager is not provided and not connected.
        """
        self._db_manager = db_manager or get_client()
        if not self._db_manager.is_connected:
            raise RuntimeError(DB_NOT_CONNECTED_ERROR)

    @property
    def _session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Get the async session factory from the database manager."""
        return self._db_manager.db

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def save(self, document: Document) -> Document:
        """
        Save a document to the database.

        Args:
            document: The document to save.

        Returns:
            The saved document.
        """
        async with self._session_factory() as session:
            try:
                session.add(document)
                await session.commit()
                await session.refresh(document)
                return document
            except IntegrityError:
                await session.rollback()
                # Document already exists, update it
                return await self.save(document)

    async def get_by_id(self, file_id: str, session: AsyncSession | None = None) -> Document | None:
        """
        Retrieve a document by its file_id.

        Args:
            file_id: The unique identifier of the document.
            session: Optional session to use. If None, creates a new session.

        Returns:
            The document if found, None otherwise.
        """
        if session is not None:
            result = await session.execute(select(Document).where(Document.file_id == file_id))
            return result.scalar_one_or_none()

        async with self._session_factory() as sess:
            result = await sess.execute(select(Document).where(Document.file_id == file_id))
            return result.scalar_one_or_none()

    async def get_by_file_name(
        self, file_name: str, session: AsyncSession | None = None
    ) -> Document | None:
        """
        Retrieve a document by its file name.

        Args:
            file_name: The name of the file to retrieve.
            session: Optional session to use. If None, creates a new session.

        Returns:
            The document if found, None otherwise.
        """
        if session is not None:
            result = await session.execute(select(Document).where(Document.file_name == file_name))
            return result.scalar_one_or_none()

        async with self._session_factory() as sess:
            result = await sess.execute(select(Document).where(Document.file_name == file_name))
            return result.scalar_one_or_none()

    async def list_all(self, skip: int = 0, limit: int = 100) -> List[Document]:
        """
        List all documents in the database.

        Args:
            skip: Number of documents to skip (for pagination).
            limit: Maximum number of documents to return.

        Returns:
            List of documents.
        """
        async with self._session_factory() as session:
            result = await session.execute(select(Document).offset(skip).limit(limit))
            return list(result.scalars().all())

    async def delete(self, file_id: str) -> bool:
        """
        Delete a document by its file_id.

        Args:
            file_id: The unique identifier of the document to delete.

        Returns:
            True if deleted successfully, False if document not found.
        """
        async with self._session_factory() as session:
            document = await self.get_by_id(file_id, session=session)
            if document:
                await session.delete(document)
                await session.commit()
                return True
            return False

    async def save_result(
        self,
        file_name: str,
        pages: List[Page],
        metadata: Optional[dict] = None,
    ) -> Document:
        """
        Save a document with its pages to the database.

        Args:
            file_name: The name of the file that was processed.
            pages: List of Page objects representing the document pages.
            metadata: Optional dictionary containing additional metadata.

        Returns:
            Document: The saved document object.
        """
        from datetime import datetime

        async with self._session_factory() as session:
            # Check if document already exists (using same session)
            existing_document = await self.get_by_file_name(file_name, session=session)

            if existing_document:
                # Update existing document
                existing_document.pages = pages
                existing_document.metadata_ = metadata or {}
                existing_document.metadata_["overwrite"] = True
                existing_document.updated_at = datetime.now()
                await session.commit()
                await session.refresh(existing_document)
                return existing_document

            # Create new document
            document = Document(
                file_id=file_name,
                file_name=file_name,
                pages=pages,
                metadata_=metadata or {},
            )

            session.add(document)
            await session.commit()
            await session.refresh(document)
            return document


class PageRepository:
    """Repository for page-level operations within documents."""

    def __init__(self, db_manager: PostgreSQLClient | None = None) -> None:
        """
        Initialize the repository with a database manager.

        Args:
            db_manager: DatabaseManager instance. If None, uses the global instance.

        Raises:
            RuntimeError: If db_manager is not provided and not connected.
        """
        self._db_manager = db_manager or get_client()
        if not self._db_manager.is_connected:
            raise RuntimeError(DB_NOT_CONNECTED_ERROR)

    @property
    def _session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Get the async session factory from the database manager."""
        return self._db_manager.db

    async def get_page(
        self, file_id: str, page_number: int, session: AsyncSession | None = None
    ) -> Page | None:
        """
        Retrieve a specific page from a document.

        Args:
            file_id: The file_id of the document.
            page_number: The page number to retrieve.
            session: Optional session to use. If None, creates a new session.

        Returns:
            The page if found, None otherwise.
        """
        if session is not None:
            result = await session.execute(
                select(Page).where(Page.document_id == file_id, Page.page_number == page_number)
            )
            return result.scalar_one_or_none()

        async with self._session_factory() as sess:
            result = await sess.execute(
                select(Page).where(Page.document_id == file_id, Page.page_number == page_number)
            )
            return result.scalar_one_or_none()

    async def list_pages(self, file_id: str) -> List[Page]:
        """
        List all pages from a document.

        Args:
            file_id: The file_id of the document.

        Returns:
            List of pages in the document.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(Page).where(Page.document_id == file_id).order_by(Page.page_number)
            )
            return list(result.scalars().all())

    async def update_page_content(self, file_id: str, page_number: int, content: str) -> bool:
        """
        Update the content of a specific page.

        Args:
            file_id: The file_id of the document.
            page_number: The page number to update.
            content: The new content for the page.

        Returns:
            True if updated successfully, False if page not found.
        """
        from datetime import datetime

        async with self._session_factory() as session:
            # Get page using same session
            page = await self.get_page(file_id, page_number, session=session)
            if page:
                page.content = content
                # Update parent document's updated_at
                await session.execute(
                    update(Document)
                    .where(Document.file_id == file_id)
                    .values(updated_at=datetime.now())
                )
                await session.commit()
                return True
            return False
