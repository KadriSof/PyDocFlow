from typing import List

from odmantic import AIOEngine
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from persistence.base import BaseDocumentRepository
from persistence.mongodb.models import Document, Page
from persistence.mongodb.client import MongoDBClient, get_client, DB_NOT_CONNECTED_ERROR
from persistence.schemas import DocumentCreate


class DocumentRepository(BaseDocumentRepository[Document]):
    """Repository for document operations."""

    def __init__(self, db_manager: MongoDBClient | None = None) -> None:
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
    def _engine(self) -> AIOEngine:
        """Get the ODMantic engine from the database manager."""
        return self._db_manager.engine

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        reraise=True,
    )
    async def save(self, document: Document) -> Document:
        """
        Save a document to the database.

        Args:
            document: The document to save.

        Returns:
            The saved document.

        Raises:
            odmantic.exceptions.DuplicateKeyError: If a document with the same primary key exists.
        """
        return await self._engine.save(document)

    async def get_by_id(self, file_id: str) -> Document | None:
        """
        Retrieve a document by its file_id.

        Args:
            file_id: The unique identifier of the document.

        Returns:
            The document if found, None otherwise.
        """
        return await self._engine.find_one(Document, Document.file_id == file_id)

    async def get_by_file_name(self, file_name: str) -> Document | None:
        """
        Retrieve a document by its file name.

        Args:
            file_name: The name of the file to retrieve.

        Returns:
            The document if found, None otherwise.
        """
        return await self._engine.find_one(Document, Document.file_name == file_name)

    async def list_all(self, skip: int = 0, limit: int = 100) -> List[Document]:
        """
        List all documents in the database.

        Args:
            skip: Number of documents to skip (for pagination).
            limit: Maximum number of documents to return.

        Returns:
            List of documents.
        """
        return await self._engine.find(Document, skip=skip, limit=limit)

    async def delete(self, file_id: str) -> bool:
        """
        Delete a document by its file_id.

        Args:
            file_id: The unique identifier of the document to delete.

        Returns:
            True if deleted successfully, False if document not found.
        """
        document = await self.get_by_id(file_id)
        if document:
            await self._engine.delete(document)
            return True
        return False

    async def save_from_schema(self, schema: DocumentCreate) -> Document:
        """
        Save a document from a DocumentCreate schema.

        Args:
            schema: The DocumentCreate schema.

        Returns:
            Document: The saved document object.
        """
        from datetime import datetime

        pages = [
            Page(page_number=p.page_number, content=p.content, metadata=p.metadata)
            for p in schema.pages
        ]

        existing_document = await self.get_by_id(schema.file_id)
        if not existing_document:
            existing_document = await self.get_by_file_name(schema.file_name)

        if existing_document:
            existing_document.pages = pages
            existing_document.metadata = schema.metadata
            existing_document.updated_at = datetime.now()
            return await self._engine.save(existing_document)

        document = Document(
            id=schema.file_id,
            file_id=schema.file_id,
            file_name=schema.file_name,
            pages=pages,
            metadata=schema.metadata,
        )

        return await self.save(document)


class PageRepository:
    """Repository for page-level operations within documents."""

    def __init__(self, db_manager: MongoDBClient | None = None) -> None:
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
    def _engine(self) -> AIOEngine:
        """Get the ODMantic engine from the database manager."""
        return self._db_manager.engine

    async def get_page(self, file_id: str, page_number: int) -> Page | None:
        """
        Retrieve a specific page from a document.

        Args:
            file_id: The file_id of the document.
            page_number: The page number to retrieve.

        Returns:
            The page if found, None otherwise.
        """
        document = await self._engine.find_one(Document, Document.file_id == file_id)
        if document and document.pages:
            for page in document.pages:
                if page.page_number == page_number:
                    return page
        return None

    async def list_pages(self, file_id: str) -> List[Page]:
        """
        List all pages from a document.

        Args:
            file_id: The file_id of the document.

        Returns:
            List of pages in the document.
        """
        document = await self._engine.find_one(Document, Document.file_id == file_id)
        return document.pages if document else []

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

        document = await self._engine.find_one(Document, Document.file_id == file_id)
        if not document:
            return False

        for page in document.pages:
            if page.page_number == page_number:
                page.content = content
                document.updated_at = datetime.now()
                await self._engine.save(document)
                return True

        return False
