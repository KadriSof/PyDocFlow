from typing import Any

from ocr.pytesseract_engine import OCREngine
from services.base import BaseService
from llm.base import LLMProvider


class ParsingService(BaseService):
    """
    Service for handling OCR parsing and LLM-enhanced text extraction.

    Combines OCR engine for text extraction with LLM provider for
    intelligent content enhancement and analysis.
    """

    def __init__(self, ocr_engine, llm_provider: LLMProvider | None = None):
        """
        Initialize the parsing service.

        Args:
            ocr_engine: OCR engine for text extraction.
            llm_provider: Optional LLM provider for enhanced parsing.
        """
        super().__init__(name="ParsingService")
        self.ocr_engine: OCREngine = ocr_engine
        self.llm_provider: LLMProvider = llm_provider

    @property
    def has_llm(self) -> bool:
        """Check if LLM provider is configured."""
        return self.llm_provider is not None

    async def start(self) -> None:
        """Start the parsing service."""
        self._is_running = True

    async def stop(self) -> None:
        """Stop the parsing service."""
        self._is_running = False

    async def health_check(self) -> dict[str, Any]:
        """Check the health status of the parsing service."""
        health = {
            "status": "healthy" if self.is_running else "unhealthy",
            "healthy": self.is_running,
            "service": self.name,
            "llm_configured": self.has_llm,
        }

        # Include LLM provider health if configured
        if self.llm_provider and hasattr(self.llm_provider, "health_check"):
            llm_health = await self.llm_provider.health_check()
            health["llm_health"] = llm_health

        return health

    async def parse_content(self, document_bytes: bytes) -> str:
        """
        Parse the document bytes and extract text using OCR.

        Args:
            document_bytes: Raw document bytes.

        Returns:
            Extracted text content.
        """
        if not self.is_running:
            raise RuntimeError("Parsing service is not running")

        # ISSUE: "._process_image" is a synchronous method, but we're calling it in an async context.
        return await self.ocr_engine._process_image(document_bytes)

    async def parse_with_llm(
        self,
        document_content: str,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Parse document and enhance with LLM.

        Args:
            document_content: Document text content.
            prompt: Optional prompt for LLM enhancement.
            **kwargs: Additional parameters for LLM.

        Returns:
            LLM-enhanced text content.

        Raises:
            RuntimeError: If LLM provider is not configured.
        """
        if not self.has_llm:
            raise RuntimeError("LLM provider not configured")

        if not self.is_running:
            raise RuntimeError("Parsing service is not running")

        # TODO-1: Remove the OCR call. We have IngestionService for that.
        # ocr_text = await self.ocr_engine.extract_text(document_bytes)

        # Build prompt
        default_prompt = (
            "Extract and summarize the key information from the following document:\n\n"
        )
        full_prompt = f"{prompt or default_prompt}{document_content}"

        # Get LLM enhancement
        response = await self.llm_provider.complete(full_prompt, **kwargs)

        return response.content

    async def enhance_with_llm(
        self,
        text: str,
        instruction: str,
        **kwargs: Any,
    ) -> str:
        """
        Enhance existing text using LLM.

        Args:
            text: Existing text to enhance.
            instruction: Instruction for LLM (e.g., "Summarize", "Translate").
            **kwargs: Additional parameters for LLM.

        Returns:
            Enhanced text.

        Raises:
            RuntimeError: If LLM provider is not configured.
        """
        if not self.has_llm:
            raise RuntimeError("LLM provider not configured")

        prompt = f"{instruction}:\n\n{text}"
        response = await self.llm_provider.complete(prompt, **kwargs)

        return response.content

    async def chat_with_document(
        self,
        document_bytes: bytes,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """
        Chat about a document using LLM.

        Args:
            document_bytes: Raw document bytes.
            messages: Chat conversation history.
            **kwargs: Additional parameters for LLM.

        Returns:
            LLM response.

        Raises:
            RuntimeError: If LLM provider is not configured.
        """
        if not self.has_llm:
            raise RuntimeError("LLM provider not configured")

        # Extract text via OCR
        ocr_text = await self.ocr_engine.extract_text(document_bytes)

        # Add document context to system message
        system_message = {
            "role": "system",
            "content": f"You are analyzing a document. Document content:\n\n{ocr_text}",
        }

        # Prepend system message to conversation
        full_messages = [system_message] + messages

        # Get LLM response
        response = await self.llm_provider.chat(full_messages, **kwargs)

        return response.content
