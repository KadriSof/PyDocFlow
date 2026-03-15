from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from fastapi import UploadFile

from persistence.base import BaseDocumentRepository
from persistence.schemas import DocumentCreate, PageCreate
from services.base import BaseService


class PersistenceService(BaseService):
    """
    Service for managing document persistence tasks.

    Handles data preparation, ID generation, and metadata enrichment
    before saving to the repository.
    """

    def __init__(self, repository: BaseDocumentRepository):
        """
        Initialize the persistence service.

        Args:
            repository: The document repository for persistence.
        """
        super().__init__(name="PersistenceService")
        self.repository = repository

    async def start(self) -> None:
        """Start the persistence service."""
        self._is_running = True

    async def stop(self) -> None:
        """Stop the persistence service."""
        self._is_running = False

    async def health_check(self) -> dict[str, Any]:
        """Check the health status of the persistence service."""
        return {
            "status": "healthy" if self._is_running else "unhealthy",
            "healthy": self._is_running,
            "service": self._name,
            "repository_connected": self.repository is not None,
        }

    async def persist_document(
        self,
        file: UploadFile,
        texts: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Prepare document data and save it to the repository.

        Args:
            file: The uploaded file (used for filename and content type).
            texts: List of extracted text strings per page.
            metadata: Optional additional metadata.

        Returns:
            The saved document object.
        """
        # 1. Prepare Data
        file_name = file.filename or "unknown"
        file_id = str(uuid.uuid4())

        full_metadata = metadata or {}
        full_metadata.update({
            "content_type": file.content_type,
            "persisted_at": datetime.now().isoformat(),
            "original_filename": file_name,
        })

        pages = [
            PageCreate(page_number=i + 1, content=text)
            for i, text in enumerate(texts)
        ]

        schema = DocumentCreate(
            file_id=file_id,
            file_name=file_name,
            pages=pages,
            metadata=full_metadata
        )

        # 2. Persist
        return await self.repository.save_from_schema(schema)
