"""LLM module for language model inference and provider management."""

from llm.base import LLMProvider, LLMResponse
from llm.config import LLMConfig
from llm.registry import LLMRegistry, get_llm_provider, create_provider
from llm.errors import (
    LLMError,
    ProviderNotFoundError,
    ProviderConfigurationError,
    LLMCompletionError,
    LLMConnectionError,
)
from llm.providers import OllamaProvider, LiteLLMProvider, MockProvider

__all__ = [
    # Base classes
    "LLMProvider",
    "LLMResponse",
    # Configuration
    "LLMConfig",
    # Registry & Factory
    "LLMRegistry",
    "get_llm_provider",
    "create_provider",
    # Exceptions
    "LLMError",
    "ProviderNotFoundError",
    "ProviderConfigurationError",
    "LLMCompletionError",
    "LLMConnectionError",
    # Providers
    "OllamaProvider",
    "LiteLLMProvider",
    "MockProvider",
]
