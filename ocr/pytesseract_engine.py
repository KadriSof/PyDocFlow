"""OCR module using ProcessPoolExecutor for parallel CPU-bound OCR tasks."""
import os
import logging
import pytesseract
import concurrent.futures

from PIL import Image
from io import BytesIO
from contextlib import contextmanager

from typing import List


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(name="PyDocFlow")


class OCREngine:
    """OCR engine using ProcessPoolExecutor for parallel processing."""

    def __init__(self, language: str = "ara", max_workers: int = 4):
        self.language = language
        self.max_workers = max_workers or min(os.cpu_count() or 1, 4)
        pytesseract.get_tesseract_version()

    def set_language(self, language: str) -> None:
        """Set the OCR language."""
        available_languages = pytesseract.get_languages()
        if language not in available_languages:
            logger.warning(f"[OCREngine] Language '{language}' not found in Tesseract. Available languages: {available_languages}")
            logger.warning(f"[OCREngine] Falling back to default language: {self.language}")
            return

        self.language = language
        logger.info(f"[OCREngine] OCR language set to: {self.language}")

    def _preprocess_image(self, image_bytes: bytes) -> BytesIO:
        """Preprocess image bytes and return a BytesIO object."""
        # TODO: Add actual preprocessing steps (e.g., resizing, binarization) if needed

    def _process_image(self, image_bytes: bytes) -> str | None:
        """Process a single image file and return extracted text."""
        try:
            image = Image.open(BytesIO(image_bytes))
            text = pytesseract.image_to_string(image, lang=self.language)
            return text.strip()

        except pytesseract.TesseractError:
            print("Tesseract OCR error occurred.")
            return ""

        except Exception as e:
            print(f"[OCREngine] Error processing image:\n{e}\n---")

    def ocr_images_batch(self, images_bytes: List[bytes]) -> List[str]:
        """Process multiple images in parallel using ProcessPoolExecutor."""
        with self.process_pool_context(max_workers=self.max_workers) as executor:
            return list(executor.map(self._process_image, images_bytes))

    @contextmanager
    def process_pool_context(self, max_workers: int = None):
        """Context manager for ProcessPoolExecutor."""
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        try:
            logger.info(f"[OCREngine] ProcessPoolExecutor started with max_workers={max_workers}")
            yield executor

        finally:
            executor.shutdown(wait=True)
            logger.info("[OCREngine] ProcessPoolExecutor shutdown complete")
