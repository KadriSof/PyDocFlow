import pytest
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

from services.ingestion import (
    DocumentStrategy,
    ImageStrategy,
    SearchablePDFStrategy,
    NativePDFStrategy,
    IngestionService,
)
from services.base import BaseService
from ocr.pytesseract_engine import OCREngine


class TestImageStrategy:
    """Tests for ImageStrategy class."""

    @pytest.mark.asyncio
    async def test_prepare_returns_image_bytes(self):
        """ImageStrategy.prepare() should return list containing image bytes."""
        strategy = ImageStrategy()
        image_data = b"fake_image_bytes"
        file_stream = BytesIO(image_data)

        result = await strategy.prepare(file_stream)

        assert result == [image_data]
        assert isinstance(result, list)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_prepare_empty_image(self):
        """ImageStrategy.prepare() should handle empty image bytes."""
        strategy = ImageStrategy()
        file_stream = BytesIO(b"")

        result = await strategy.prepare(file_stream)

        assert result == [b""]


class TestSearchablePDFStrategy:
    """Tests for SearchablePDFStrategy class."""

    @pytest.mark.asyncio
    @patch("services.ingestion.pdf2image.convert_from_bytes")
    async def test_prepare_converts_pdf_to_images(self, mock_convert):
        """SearchablePDFStrategy.prepare() should convert PDF pages to image bytes."""
        mock_image1 = MagicMock()
        mock_image1.tobytes = MagicMock(return_value=b"image1_data")
        mock_image2 = MagicMock()
        mock_image2.tobytes = MagicMock(return_value=b"image2_data")
        mock_convert.return_value = [mock_image1, mock_image2]

        strategy = SearchablePDFStrategy()
        pdf_data = b"fake_pdf_bytes"
        file_stream = BytesIO(pdf_data)

        result = await strategy.prepare(file_stream)

        mock_convert.assert_called_once_with(pdf_data)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(img_bytes, bytes) for img_bytes in result)

    @pytest.mark.asyncio
    @patch("services.ingestion.pdf2image.convert_from_bytes")
    async def test_prepare_single_page_pdf(self, mock_convert):
        """SearchablePDFStrategy.prepare() should handle single-page PDFs."""
        mock_image = MagicMock()
        mock_image.tobytes = MagicMock(return_value=b"single_page_data")
        mock_convert.return_value = [mock_image]

        strategy = SearchablePDFStrategy()
        file_stream = BytesIO(b"single_page_pdf")

        result = await strategy.prepare(file_stream)

        assert len(result) == 1

    @pytest.mark.asyncio
    @patch("services.ingestion.pdf2image.convert_from_bytes")
    async def test_prepare_empty_pdf(self, mock_convert):
        """SearchablePDFStrategy.prepare() should handle empty PDF (no pages)."""
        mock_convert.return_value = []

        strategy = SearchablePDFStrategy()
        file_stream = BytesIO(b"")

        result = await strategy.prepare(file_stream)

        assert result == []

    def test_image_to_bytes(self):
        """SearchablePDFStrategy._image_to_bytes() should convert PIL Image to PNG bytes."""
        from PIL import Image

        strategy = SearchablePDFStrategy()
        test_image = Image.new("RGB", (100, 100), color="red")

        result = strategy._image_to_bytes(test_image)

        assert isinstance(result, bytes)
        assert len(result) > 0
        assert result.startswith(b"\x89PNG")


class TestNativePDFStrategy:
    """Tests for NativePDFStrategy class."""

    @pytest.mark.asyncio
    async def test_prepare_extracts_text_from_pdf(self):
        """NativePDFStrategy.prepare() should extract text from PDF."""
        # Create a mock fitz document
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Extracted text from page"
        mock_page.rect = MagicMock()
        mock_page.rect.__abs__ = lambda self: 100.0

        mock_doc = MagicMock()
        mock_doc.__iter__ = lambda self: iter([mock_page])
        mock_doc.__enter__ = lambda self: self
        mock_doc.__exit__ = lambda self, *args: None

        with patch("services.ingestion.fitz.open", return_value=mock_doc):
            strategy = NativePDFStrategy()
            file_stream = BytesIO(b"fake_pdf_with_text")

            result = await strategy.prepare(file_stream)

            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0] == b"Extracted text from page"

    @pytest.mark.asyncio
    async def test_prepare_skips_empty_pages(self):
        """NativePDFStrategy.prepare() should skip pages with no text."""
        mock_page_empty = MagicMock()
        mock_page_empty.get_text.return_value = "   "  # Whitespace only

        mock_page_with_text = MagicMock()
        mock_page_with_text.get_text.return_value = "Has text"

        mock_doc = MagicMock()
        mock_doc.__iter__ = lambda self: iter([mock_page_empty, mock_page_with_text])
        mock_doc.__enter__ = lambda self: self
        mock_doc.__exit__ = lambda self, *args: None

        with patch("services.ingestion.fitz.open", return_value=mock_doc):
            strategy = NativePDFStrategy()
            file_stream = BytesIO(b"fake_pdf")

            result = await strategy.prepare(file_stream)

            assert len(result) == 1
            assert result[0] == b"Has text"


class TestIngestionService:
    """Tests for IngestionService class."""

    @pytest.fixture
    def mock_ocr_engine(self):
        """Create a mock OCR engine."""
        engine = MagicMock(spec=OCREngine)
        engine.ocr_images_batch = MagicMock(return_value=["extracted text 1", "extracted text 2"])
        return engine

    @pytest.fixture
    def ingestion_service(self, mock_ocr_engine):
        """Create an IngestionService instance with mock OCR engine."""
        return IngestionService(ocr_engine=mock_ocr_engine)

    def test_inherits_from_base_service(self, ingestion_service):
        """IngestionService should inherit from BaseService."""
        assert isinstance(ingestion_service, BaseService)

    def test_has_service_name(self, ingestion_service):
        """IngestionService should have correct service name."""
        assert ingestion_service.name == "IngestionService"

    def test_is_running_initially_false(self, ingestion_service):
        """IngestionService should not be running by default."""
        assert ingestion_service.is_running is False

    @pytest.mark.asyncio
    async def test_start_sets_running_state(self, ingestion_service):
        """Starting the service should set is_running to True."""
        await ingestion_service.start()
        assert ingestion_service.is_running is True

    @pytest.mark.asyncio
    async def test_stop_clears_running_state(self, ingestion_service):
        """Stopping the service should set is_running to False."""
        await ingestion_service.start()
        await ingestion_service.stop()
        assert ingestion_service.is_running is False

    @pytest.mark.asyncio
    async def test_health_check_returns_status(self, ingestion_service):
        """Health check should return status dictionary."""
        await ingestion_service.start()
        health = await ingestion_service.health_check()

        assert health["status"] == "healthy"
        assert health["healthy"] is True
        assert health["service"] == "IngestionService"

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_when_stopped(self, ingestion_service):
        """Health check should show unhealthy when stopped."""
        health = await ingestion_service.health_check()

        assert health["status"] == "unhealthy"
        assert health["healthy"] is False

    @pytest.mark.asyncio
    async def test_ingest_image_file(self, ingestion_service, mock_ocr_engine):
        """IngestionService.ingest() should process image files correctly."""
        image_data = b"test_image_bytes"
        mock_file = AsyncMock()
        mock_file.filename = "test_image.png"
        mock_file.read = AsyncMock(return_value=image_data)

        result = await ingestion_service.ingest(mock_file)

        mock_file.read.assert_called_once()
        mock_ocr_engine.ocr_images_batch.assert_called_once()
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_ingest_pdf_file(self, ingestion_service, mock_ocr_engine):
        """IngestionService.ingest() should process PDF files correctly (scanned)."""
        pdf_data = b"test_pdf_bytes"
        mock_file = AsyncMock()
        mock_file.filename = "document.pdf"
        mock_file.content_type = "application/pdf"
        mock_file.read = AsyncMock(return_value=pdf_data)

        # Mock is_scanned to return True (scanned PDF)
        with patch.object(ingestion_service, 'is_scanned', return_value=True):
            with patch("services.ingestion.pdf2image.convert_from_bytes") as mock_convert:
                mock_image = MagicMock()
                mock_image.tobytes = MagicMock(return_value=b"page_image")
                mock_convert.return_value = [mock_image]

                result = await ingestion_service.ingest(mock_file)

                mock_file.read.assert_called_once()
                mock_ocr_engine.ocr_images_batch.assert_called_once()
                assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_ingest_pdf_force_ocr(self, ingestion_service, mock_ocr_engine):
        """IngestionService.ingest() should respect force_ocr parameter."""
        pdf_data = b"test_pdf_bytes"
        mock_file = AsyncMock()
        mock_file.filename = "document.pdf"
        mock_file.content_type = "application/pdf"
        mock_file.read = AsyncMock(return_value=pdf_data)

        # force_ocr=True should use OCR even for native PDFs
        with patch("services.ingestion.pdf2image.convert_from_bytes") as mock_convert:
            mock_image = MagicMock()
            mock_image.tobytes = MagicMock(return_value=b"page_image")
            mock_convert.return_value = [mock_image]

            await ingestion_service.ingest(mock_file, force_ocr=True)

            mock_convert.assert_called_once()
            mock_ocr_engine.ocr_images_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_native_pdf(self, ingestion_service):
        """IngestionService.ingest() should extract text from native PDFs."""
        pdf_data = b"test_pdf_bytes"
        mock_file = AsyncMock()
        mock_file.filename = "document.pdf"
        mock_file.content_type = "application/pdf"
        mock_file.read = AsyncMock(return_value=pdf_data)

        # Mock is_scanned to return False (native PDF with text)
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Native PDF text"
        mock_doc = MagicMock()
        mock_doc.__iter__ = lambda self: iter([mock_page])
        mock_doc.__enter__ = lambda self: self
        mock_doc.__exit__ = lambda self, *args: None

        with patch.object(ingestion_service, 'is_scanned', return_value=False):
            with patch("services.ingestion.fitz.open", return_value=mock_doc):
                result = await ingestion_service.ingest(mock_file, force_ocr=False)

                assert isinstance(result, list)
                assert result == ["Native PDF text"]

    @pytest.mark.asyncio
    async def test_ingest_unsupported_file_type(self, ingestion_service):
        """IngestionService.ingest() should raise ValueError for unsupported file types."""
        mock_file = AsyncMock()
        mock_file.filename = "document.txt"
        mock_file.content_type = "text/plain"
        mock_file.read = AsyncMock(return_value=b"content")

        with pytest.raises(ValueError, match="Unsupported file type:"):
            await ingestion_service.ingest(mock_file)

    @pytest.mark.asyncio
    async def test_ingest_pdf_case_insensitive(self, ingestion_service, mock_ocr_engine):
        """IngestionService.ingest() should handle PDF extension case-insensitively."""
        pdf_data = b"test_pdf_bytes"
        mock_file = AsyncMock()
        mock_file.filename = "DOCUMENT.PDF"
        mock_file.read = AsyncMock(return_value=pdf_data)

        with patch("services.ingestion.pdf2image.convert_from_bytes") as mock_convert:
            mock_image = MagicMock()
            mock_image.tobytes = MagicMock(return_value=b"page_image")
            mock_convert.return_value = [mock_image]

            await ingestion_service.ingest(mock_file)

            mock_convert.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_various_image_extensions(self, ingestion_service, mock_ocr_engine):
        """IngestionService.ingest() should handle various image file extensions."""
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"]

        for ext in image_extensions:
            mock_file = AsyncMock()
            mock_file.filename = f"test_image{ext}"
            mock_file.read = AsyncMock(return_value=b"image_data")

            await ingestion_service.ingest(mock_file)
            mock_ocr_engine.ocr_images_batch.assert_called()
            mock_ocr_engine.reset_mock()

    @pytest.mark.asyncio
    async def test_ingest_returns_ocr_results(self, ingestion_service, mock_ocr_engine):
        """IngestionService.ingest() should return OCR results."""
        expected_results = ["text from page 1", "text from page 2"]
        mock_ocr_engine.ocr_images_batch.return_value = expected_results

        mock_file = AsyncMock()
        mock_file.filename = "test.png"
        mock_file.content_type = "image/png"
        mock_file.read = AsyncMock(return_value=b"image")

        result = await ingestion_service.ingest(mock_file)

        assert result == expected_results


class TestIngestionServiceHelpers:
    """Tests for IngestionService helper methods."""

    @pytest.fixture
    def mock_ocr_engine(self):
        """Create a mock OCR engine."""
        return MagicMock(spec=OCREngine)

    @pytest.fixture
    def ingestion_service(self, mock_ocr_engine):
        """Create an IngestionService instance."""
        return IngestionService(ocr_engine=mock_ocr_engine)

    def test_is_image_by_mime_type(self, ingestion_service):
        """is_image() should detect images by MIME type."""
        mock_file = AsyncMock()
        mock_file.content_type = "image/png"
        mock_file.filename = "test.png"

        assert ingestion_service.is_image(mock_file) is True

    def test_is_image_by_extension(self, ingestion_service):
        """is_image() should detect images by extension."""
        mock_file = AsyncMock()
        mock_file.content_type = None
        mock_file.filename = "test.jpg"

        assert ingestion_service.is_image(mock_file) is True

    def test_is_image_unknown_type(self, ingestion_service):
        """is_image() should return False for unknown types."""
        mock_file = AsyncMock()
        mock_file.content_type = "application/unknown"
        mock_file.filename = "test.xyz"

        assert ingestion_service.is_image(mock_file) is False

    def test_is_pdf_by_mime_type(self, ingestion_service):
        """is_pdf() should detect PDF by MIME type."""
        mock_file = AsyncMock()
        mock_file.content_type = "application/pdf"
        mock_file.filename = "test.txt"  # Wrong extension

        assert ingestion_service.is_pdf(mock_file) is True

    def test_is_pdf_by_extension(self, ingestion_service):
        """is_pdf() should detect PDF by extension."""
        mock_file = AsyncMock()
        mock_file.content_type = "application/unknown"
        mock_file.filename = "document.PDF"

        assert ingestion_service.is_pdf(mock_file) is True

    def test_is_pdf_not_pdf(self, ingestion_service):
        """is_pdf() should return False for non-PDF files."""
        mock_file = AsyncMock()
        mock_file.content_type = "text/plain"
        mock_file.filename = "document.txt"

        assert ingestion_service.is_pdf(mock_file) is False
