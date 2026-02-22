from abc import ABC, abstractmethod
from typing import Any

from io import BytesIO
from typing import List, Union, BinaryIO

import fitz
import pdf2image
from PIL import Image
from fastapi import UploadFile

from ocr.pytesseract_engine import OCREngine
from services.base import BaseService


class DocumentStrategy(ABC):
    """Abstract base class for document processing strategies."""

    @abstractmethod
    async def prepare(self, file: Union[BytesIO, BinaryIO]) -> List[bytes]:
        """Prepare the document for OCR processing."""
        pass


class ImageStrategy(DocumentStrategy):
    """Strategy for processing image files."""

    async def prepare(self, file: Union[BytesIO, BinaryIO]) -> List[bytes]:
        """Return the image bytes for OCR processing."""
        return [file.read()]


class SearchablePDFStrategy(DocumentStrategy):
    """Strategy for processing PDF files by converting pages to images."""

    async def prepare(self, file: Union[BytesIO, BinaryIO]) -> List[bytes]:
        """Convert PDF pages to images and return their bytes."""
        content = file.read() if hasattr(file, "read") else file
        images = pdf2image.convert_from_bytes(content)
        return [self._image_to_bytes(img) for img in images]

    @staticmethod
    def _image_to_bytes(image: Image.Image) -> bytes:
        """Convert a PIL Image to bytes."""
        with BytesIO() as output:
            image.save(output, format='PNG')
            return output.getvalue()


class NativePDFStrategy(DocumentStrategy):
    """Strategy for processing native PDF files by extracting text directly."""

    async def prepare(self, file: Union[BytesIO, BinaryIO]) -> List[bytes]:
        """Extract text directly from PDF without OCR."""
        content = file.read() if hasattr(file, "read") else file
        doc = fitz.open(stream=content, filetype="pdf")
        texts = []
        for page in doc:
            text = page.get_text()
            if text.strip():
                texts.append(text.encode('utf-8'))

        doc.close()
        return texts


class IngestionService(BaseService):
    """Service for handling document ingestion and preparation."""

    # Supported image MIME types
    IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/gif", "image/bmp", "image/tiff", "image/webp"}

    def __init__(self, ocr_engine: OCREngine):
        super().__init__(name="IngestionService")
        self.ocr_engine = ocr_engine
        self.strategies = {
            'image': ImageStrategy(),
            'searchable_pdf': SearchablePDFStrategy(),
            'native_pdf': NativePDFStrategy()
        }

    async def start(self) -> None:
        """Start the ingestion service."""
        self._is_running = True

    async def stop(self) -> None:
        """Stop the ingestion service."""
        self._is_running = False

    async def health_check(self) -> dict[str, Any]:
        """Check the health status of the ingestion service."""
        return {
            "status": "healthy" if self._is_running else "unhealthy",
            "healthy": self._is_running,
            "service": self._name,
        }

    def set_ocr_language(self, language: str) -> None:
        """Set the OCR language for the engine."""
        self.ocr_engine.set_language(language)

    async def ingest(self, file: UploadFile, force_ocr: bool = False) -> List[str]:
        """
        Ingest a document and prepare it for OCR processing.

        Args:
            file: The uploaded file to process.
            force_ocr: If True, always use OCR even for native PDFs.
                       If False, auto-detect and use native text extraction when possible.

        Returns:
            List of extracted text strings (one per page/image).

        Raises:
            ValueError: If the file type is not supported.
        """
        content = await file.read()
        file_stream = BytesIO(content)

        # Determine file type
        if self.is_image(file):
            file_type = 'image'
        elif self.is_pdf(file):
            # Auto-detect if PDF is scanned or has native text
            if force_ocr or self.is_scanned(file_stream):
                file_type = 'searchable_pdf'
            else:
                file_type = 'native_pdf'
        else:
            raise ValueError(f"Unsupported file type: {file.content_type or file.filename}")

        strategy = self.strategies.get(file_type)

        if not strategy:
            raise ValueError(f"No strategy available for file type: {file_type}")

        # Reset stream position for strategy
        file_stream.seek(0)
        prepared_data = await strategy.prepare(file_stream)

        # For native PDF, text is already extracted - return as-is
        if file_type == 'native_pdf':
            return [text.decode('utf-8') for text in prepared_data]

        # For images (from image or scanned PDF), run OCR
        return self.ocr_engine.ocr_images_batch(images_bytes=prepared_data)

    def is_image(self, file: UploadFile) -> bool:
        """Check if the uploaded file is an image based on MIME type or extension."""
        if file.content_type and file.content_type in self.IMAGE_MIME_TYPES:
            return True

        # Fallback to extension check
        if file.filename:
            ext = file.filename.lower().split('.')[-1]
            image_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
            return ext in image_extensions

        return False

    def is_pdf(self, file: UploadFile) -> bool:
        """Check if the uploaded file is a PDF based on MIME type or extension."""
        if file.content_type == "application/pdf":
            return True

        if file.filename and file.filename.lower().endswith(".pdf"):
            return True

        return False

    def is_scanned(self, file: Union[BytesIO, BinaryIO]) -> bool:
        """
        Determine if a PDF file is scanned (image-based) or has native text.

        Uses text coverage ratio: if less than 10% of the page area contains
        extractable text, the PDF is considered scanned.

        Args:
            file: File-like object containing PDF data.

        Returns:
            True if the PDF appears to be scanned (image-based),
            False if it has native text content.
        """
        # Save current position
        original_position = file.tell() if hasattr(file, "tell") else 0

        try:
            # Read content without modifying the file object
            content = file.read() if hasattr(file, "read") else file

            doc = fitz.open(stream=content, filetype="pdf")

            total_page_area = 0.0
            total_text_area = 0.0

            for page in doc:
                total_page_area += abs(page.rect)
                text_area = 0.0
                for block in page.get_text_blocks():
                    rect = fitz.Rect(block[:4])
                    text_area += abs(rect)
                total_text_area += text_area

            doc.close()

            text_coverage = total_text_area / total_page_area if total_page_area > 0 else 0

            # Reset file position
            if hasattr(file, "seek"):
                file.seek(original_position)

            return text_coverage < 0.1

        except Exception:
            # If we can't analyze, assume it's scanned and use OCR
            # Reset file position on error
            if hasattr(file, "seek"):
                file.seek(original_position)
            return True
