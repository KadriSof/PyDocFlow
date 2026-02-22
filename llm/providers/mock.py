"""Mock LLM provider for testing."""

from typing import Any

from llm.base import LLMProvider, LLMResponse


class MockProvider(LLMProvider):
    """
    Mock LLM provider for testing purposes.

    Returns predefined responses without making actual API calls.
    Useful for unit tests and development without LLM access.
    """

    def __init__(
        self,
        mock_response: str = "This is a mock response from the LLM.",
        model: str = "mock/model",
    ):
        """
        Initialize the mock provider.

        Args:
            mock_response: Default response text to return.
            model: Model identifier (for consistency with real providers).
        """
        super().__init__(name="mock")
        self._mock_response = mock_response
        self._model = model
        self._call_count = 0
        self._last_prompt: str | None = None

    @property
    def call_count(self) -> int:
        """Return the number of times the provider was called."""
        return self._call_count

    @property
    def last_prompt(self) -> str | None:
        """Return the last prompt received."""
        return self._last_prompt

    def set_mock_response(self, response: str) -> None:
        """Set a new mock response."""
        self._mock_response = response

    async def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """
        Return a mock completion.

        Args:
            prompt: The text prompt (stored for inspection).
            **kwargs: Additional parameters (ignored).

        Returns:
            LLMResponse with mock content.
        """
        self._call_count += 1
        self._last_prompt = prompt

        # Support custom response via kwargs
        content = kwargs.get("mock_response", self._mock_response)

        return LLMResponse(
            content=content,
            model=self._model,
            usage={"prompt_tokens": len(prompt.split()), "completion_tokens": len(content.split())},
            raw_response={"mock": True, "prompt": prompt},
        )

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        """
        Return a mock chat response.

        Args:
            messages: List of message dicts (last user message stored).
            **kwargs: Additional parameters (ignored).

        Returns:
            LLMResponse with mock content.
        """
        self._call_count += 1

        # Get last user message
        user_messages = [m for m in messages if m.get("role") == "user"]
        self._last_prompt = user_messages[-1].get("content", "") if user_messages else ""

        content = kwargs.get("mock_response", self._mock_response)

        return LLMResponse(
            content=content,
            model=self._model,
            usage={"prompt_tokens": sum(len(m.get("content", "").split()) for m in messages), "completion_tokens": len(content.split())},
            raw_response={"mock": True, "messages": messages},
        )

    async def health_check(self) -> dict[str, Any]:
        """
        Mock provider is always healthy.

        Returns:
            Health status dictionary.
        """
        return {
            "status": "healthy",
            "healthy": True,
            "provider": self.name,
            "model": self._model,
            "call_count": self._call_count,
        }

    def reset(self) -> None:
        """Reset call count and last prompt."""
        self._call_count = 0
        self._last_prompt = None

    def __repr__(self) -> str:
        return f"MockProvider(model='{self._model}', calls={self._call_count})"
