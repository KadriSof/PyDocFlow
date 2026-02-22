"""Tests for LLM module."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

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
from llm.providers.ollama import OllamaProvider
from llm.providers.litellm import LiteLLMProvider
from llm.providers.mock import MockProvider


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_create_response(self):
        """LLMResponse can be created with required fields."""
        response = LLMResponse(content="Hello", model="test-model")

        assert response.content == "Hello"
        assert response.model == "test-model"
        assert response.usage is None
        assert response.raw_response is None

    def test_create_response_with_usage(self):
        """LLMResponse can include usage statistics."""
        response = LLMResponse(
            content="Hello",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )

        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 5

    def test_str_returns_content(self):
        """str(LLMResponse) returns content."""
        response = LLMResponse(content="Test content", model="test")
        assert str(response) == "Test content"

    def test_repr(self):
        """repr(LLMResponse) shows model and content length."""
        response = LLMResponse(content="Hello World", model="gpt-4")
        repr_str = repr(response)

        assert "gpt-4" in repr_str
        assert "content_length=11" in repr_str


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_default_config(self):
        """LLMConfig has sensible defaults."""
        # Use _env_file=None to skip loading .env file and get true defaults
        config = LLMConfig(_env_file=None)

        assert config.provider == "ollama"
        assert config.model == "llama3.2"
        assert config.ollama_host == "http://localhost:11434"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048

    def test_is_ollama_property(self):
        """is_ollama returns True for ollama provider."""
        config = LLMConfig(provider="ollama")
        assert config.is_ollama is True

        config = LLMConfig(provider="litellm")
        assert config.is_ollama is False

    def test_is_litellm_property(self):
        """is_litellm returns True for litellm provider."""
        config = LLMConfig(provider="litellm")
        assert config.is_litellm is True

    def test_is_mock_property(self):
        """is_mock returns True for mock provider."""
        config = LLMConfig(provider="mock")
        assert config.is_mock is True

    def test_get_model_for_provider(self):
        """get_model_for_provider returns mapped model."""
        config = LLMConfig(
            model="default-model",
            provider_models={"openai": "gpt-4", "anthropic": "claude-3"},
        )

        assert config.get_model_for_provider("openai") == "gpt-4"
        assert config.get_model_for_provider("unknown") == "default-model"


class TestMockProvider:
    """Tests for MockProvider."""

    @pytest.mark.asyncio
    async def test_complete_returns_mock_response(self):
        """MockProvider.complete returns predefined response."""
        provider = MockProvider(mock_response="Mock text")

        response = await provider.complete("Test prompt")

        assert response.content == "Mock text"
        assert response.model == "mock/model"

    @pytest.mark.asyncio
    async def test_chat_returns_mock_response(self):
        """MockProvider.chat returns predefined response."""
        provider = MockProvider(mock_response="Mock chat response")

        messages = [{"role": "user", "content": "Hello"}]
        response = await provider.chat(messages)

        assert response.content == "Mock chat response"

    @pytest.mark.asyncio
    async def test_complete_tracks_calls(self):
        """MockProvider tracks number of calls."""
        provider = MockProvider()

        await provider.complete("prompt 1")
        await provider.complete("prompt 2")

        assert provider.call_count == 2

    @pytest.mark.asyncio
    async def test_complete_stores_last_prompt(self):
        """MockProvider stores the last prompt."""
        provider = MockProvider()

        await provider.complete("First prompt")
        assert provider.last_prompt == "First prompt"

        await provider.complete("Second prompt")
        assert provider.last_prompt == "Second prompt"

    @pytest.mark.asyncio
    async def test_health_check_always_healthy(self):
        """MockProvider.health_check always returns healthy."""
        provider = MockProvider()

        health = await provider.health_check()

        assert health["healthy"] is True
        assert health["status"] == "healthy"

    def test_set_mock_response(self):
        """set_mock_response changes the response."""
        provider = MockProvider(mock_response="Initial")
        provider.set_mock_response("Updated")

        # Would need async test to verify, but method exists
        assert provider._mock_response == "Updated"

    def test_reset_clears_counters(self):
        """reset clears call count and last prompt."""
        provider = MockProvider()
        provider._call_count = 5
        provider._last_prompt = "test"

        provider.reset()

        assert provider.call_count == 0
        assert provider.last_prompt is None


class TestOllamaProvider:
    """Tests for OllamaProvider."""

    def test_init_with_defaults(self):
        """OllamaProvider initializes with default values."""
        provider = OllamaProvider()

        assert provider.name == "ollama"
        assert provider._host == "http://localhost:11434"
        assert provider._model == "llama3.2"

    def test_init_with_custom_params(self):
        """OllamaProvider accepts custom parameters."""
        provider = OllamaProvider(
            host="http://custom:11434",
            model="mistral",
            temperature=0.5,
        )

        assert provider._host == "http://custom:11434"
        assert provider._model == "mistral"
        assert provider._temperature == 0.5

    @pytest.mark.asyncio
    async def test_health_check_connection_failed(self):
        """OllamaProvider.health_check handles connection errors."""
        provider = OllamaProvider(host="http://invalid-host:9999")

        health = await provider.health_check()

        assert health["healthy"] is False
        assert health["status"] == "unhealthy"


class TestLiteLLMProvider:
    """Tests for LiteLLMProvider."""

    def test_init_requires_model_prefix(self):
        """LiteLLMProvider requires model with provider prefix."""
        with pytest.raises(ProviderConfigurationError) as exc_info:
            LiteLLMProvider(model="gpt-4")  # Missing prefix

        assert "model" in str(exc_info.value)
        assert "provider prefix" in str(exc_info.value)

    def test_init_accepts_valid_model(self):
        """LiteLLMProvider accepts model with prefix."""
        provider = LiteLLMProvider(model="openai/gpt-4")

        assert provider.name == "litellm"
        assert provider._model == "openai/gpt-4"

    def test_init_with_api_key(self):
        """LiteLLMProvider accepts API key."""
        provider = LiteLLMProvider(
            model="openai/gpt-4",
            api_key="test-key",
            api_base="https://custom.api.com",
        )

        assert provider._api_key == "test-key"
        assert provider._api_base == "https://custom.api.com"


class TestLLMRegistry:
    """Tests for LLMRegistry."""

    def test_list_providers_returns_defaults(self):
        """Registry lists default providers."""
        providers = LLMRegistry.list_providers()

        assert "ollama" in providers
        assert "litellm" in providers
        assert "mock" in providers

    def test_is_registered(self):
        """is_registered checks provider availability."""
        assert LLMRegistry.is_registered("ollama") is True
        assert LLMRegistry.is_registered("unknown") is False

    def test_get_ollama_provider(self):
        """Registry creates Ollama provider."""
        config = LLMConfig(provider="ollama", model="test-model")
        provider = LLMRegistry.get("ollama", config)

        assert isinstance(provider, OllamaProvider)
        assert provider._model == "test-model"

    def test_get_litellm_provider(self):
        """Registry creates LiteLLM provider."""
        config = LLMConfig(provider="litellm", model="openai/gpt-4")
        provider = LLMRegistry.get("litellm", config)

        assert isinstance(provider, LiteLLMProvider)
        assert provider._model == "openai/gpt-4"

    def test_get_mock_provider(self):
        """Registry creates Mock provider."""
        config = LLMConfig(provider="mock")
        provider = LLMRegistry.get("mock", config)

        assert isinstance(provider, MockProvider)

    def test_get_unknown_provider_raises(self):
        """Registry raises error for unknown provider."""
        with pytest.raises(ProviderNotFoundError) as exc_info:
            LLMRegistry.get("unknown")

        assert "unknown" in str(exc_info.value)
        assert "ollama" in str(exc_info.value)

    def test_register_custom_provider(self):
        """Registry accepts custom provider registration."""

        class CustomProvider(LLMProvider):
            async def complete(self, prompt: str, **kwargs):
                return LLMResponse(content="", model="")

            async def chat(self, messages: list, **kwargs):
                return LLMResponse(content="", model="")

            async def health_check(self):
                return {"healthy": True}

        LLMRegistry.register("custom", CustomProvider)
        assert LLMRegistry.is_registered("custom")

    def test_get_provider_case_insensitive(self):
        """Provider lookup is case-insensitive."""
        config = LLMConfig(provider="mock")

        provider1 = LLMRegistry.get("MOCK", config)
        provider2 = LLMRegistry.get("mock", config)

        assert type(provider1) == type(provider2)


class TestGetLLMProvider:
    """Tests for get_llm_provider convenience function."""

    def test_get_provider_from_config(self):
        """get_llm_provider uses config provider setting."""
        config = LLMConfig(provider="mock")
        provider = get_llm_provider(config)

        assert isinstance(provider, MockProvider)

    def test_get_provider_defaults_to_env(self):
        """get_llm_provider loads from env when no config."""
        # This will use default (ollama) since no .env
        provider = get_llm_provider()
        assert provider is not None


class TestCreateProvider:
    """Tests for create_provider convenience function."""

    def test_create_ollama_provider(self):
        """create_provider creates Ollama provider."""
        provider = create_provider(
            "ollama",
            host="http://test:11434",
            model="test-model",
        )

        assert isinstance(provider, OllamaProvider)
        assert provider._host == "http://test:11434"

    def test_create_mock_provider(self):
        """create_provider creates Mock provider."""
        provider = create_provider("mock", model="test/mock")

        assert isinstance(provider, MockProvider)


class TestLLMExceptions:
    """Tests for LLM exception classes."""

    def test_llm_error_basic(self):
        """LLMError formats message correctly."""
        error = LLMError("Something went wrong")

        assert "Something went wrong" in str(error)

    def test_llm_error_with_provider(self):
        """LLMError includes provider name."""
        error = LLMError("Failed", provider="ollama")

        assert "[ollama]" in str(error)
        assert "Failed" in str(error)

    def test_provider_not_found_error(self):
        """ProviderNotFoundError includes available providers."""
        error = ProviderNotFoundError(
            provider_name="unknown",
            available_providers=["ollama", "mock"],
        )

        assert "unknown" in str(error)
        assert "ollama" in str(error)
        assert "mock" in str(error)

    def test_provider_configuration_error(self):
        """ProviderConfigurationError includes missing fields."""
        error = ProviderConfigurationError(
            provider_name="litellm",
            missing_fields=["api_key", "model"],
        )

        assert "litellm" in str(error)
        assert "api_key" in str(error)
        assert "model" in str(error)

    def test_llm_completion_error(self):
        """LLMCompletionError wraps original error."""
        original = ValueError("Original error")
        error = LLMCompletionError(
            message="Completion failed",
            provider="ollama",
            original_error=original,
        )

        assert error.original_error is original
        assert "Completion failed" in str(error)

    def test_llm_connection_error(self):
        """LLMConnectionError includes URL."""
        error = LLMConnectionError(
            provider="ollama",
            url="http://localhost:11434",
        )

        assert "ollama" in str(error)
        assert "http://localhost:11434" in str(error)
