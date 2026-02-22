from typing import List

from tenacity import retry, stop_after_attempt, wait_exponential

from persistence.models import OCRResult
from persistence.db import DatabaseManager, get_db_manager


class OCRRepository:
    """Repository for OCR result operations."""

    def __init__(self, db_manager: DatabaseManager | None = None) -> None:
        """
        Initialize the repository with a database manager.

        Args:
            db_manager: DatabaseManager instance. If None, uses the global instance.
        """
        self._db_manager = db_manager or get_db_manager()

    @property
    def _engine(self):
        """Get the ODMantic engine from the database manager."""
        return self._db_manager.engine

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def save_result(self, file_name: str, content: str, metadata: dict = None) -> OCRResult:
        """
        Save an OCR result to the database.

        Args:
            file_name: The name of the file that was processed.
            content: The extracted text content from the image.
            metadata: Optional dictionary containing additional metadata.

        Returns:
            OCRResult: The saved OCR result object.
        """
        from datetime import datetime
        from odmantic.exceptions import DuplicateKeyError

        existing_result = await self.get_result(file_name)

        if existing_result:
            existing_result.content = content
            current_metadata = metadata or {}
            current_metadata["overwrite"] = True
            existing_result.metadata = current_metadata
            existing_result.updated_at = datetime.now()
            return await self._engine.save(existing_result)

        ocr_result = OCRResult(file_name=file_name, content=content, metadata=metadata or {})

        try:
            return await self._engine.save(ocr_result)

        except DuplicateKeyError:
            return await self.save_result(file_name, content, metadata)

    async def get_result(self, file_name: str) -> OCRResult | None:
        """
        Retrieve an OCR result by file name.

        Args:
            file_name: The name of the file to retrieve.

        Returns:
            OCRResult: The retrieved OCR result object, or None if not found.
        """
        return await self._engine.find_one(OCRResult, OCRResult.file_name == file_name)

    async def list_results(self) -> List[OCRResult]:
        """
        List all OCR results in the database.

        Returns:
            List[OCRResult]: A list of all OCR result objects.
        """
        return await self._engine.find(OCRResult)