"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    model: str
    usage: dict[str, int] | None = None
    raw_response: Any | None = None

    def __str__(self) -> str:
        return self.content

    def __repr__(self) -> str:
        return f"LLMResponse(model='{self.model}', content_length={len(self.content)})"


class LLMProvider(ABC):
    """
    Abstract base class for all LLM provider implementations.

    Defines the interface that all LLM providers must implement.
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        """
        Initialize the LLM provider.

        Args:
            name: Provider name identifier (e.g., 'ollama', 'openai').
            config: Optional configuration dictionary.
        """
        self._name = name
        self._config = config or {}

    @property
    def name(self) -> str:
        """Return the provider name."""
        return self._name

    @property
    def config(self) -> dict[str, Any]:
        """Return the provider configuration."""
        return self._config

    @abstractmethod
    async def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: The text prompt to complete.
            **kwargs: Additional provider-specific parameters.

        Returns:
            LLMResponse containing the generated completion.
        """
        pass

    @abstractmethod
    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        """
        Generate a response for a chat conversation.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            **kwargs: Additional provider-specific parameters.

        Returns:
            LLMResponse containing the generated response.
        """
        pass

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """
        Check the health status of the provider connection.

        Returns:
            Dictionary with health status information.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self._name}')"
