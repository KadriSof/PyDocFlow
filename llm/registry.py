"""Registry and factory for LLM providers."""

from typing import Any, Callable, Type

from llm.base import LLMProvider
from llm.config import LLMConfig
from llm.errors import ProviderNotFoundError
from llm.providers.ollama import OllamaProvider
from llm.providers.litellm import LiteLLMProvider
from llm.providers.mock import MockProvider


class LLMRegistry:
    """
    Registry and factory for LLM providers.

    Uses the Factory pattern to dynamically instantiate providers
    based on configuration. Supports runtime registration of custom providers.
    """

    _providers: dict[str, Callable[[LLMConfig], LLMProvider]] = {}
    _initialized: bool = False

    @classmethod
    def _register_defaults(cls) -> None:
        """Register default provider factories."""
        if cls._initialized:
            return

        cls.register("ollama", cls._create_ollama)
        cls.register("litellm", cls._create_litellm)
        cls.register("mock", cls._create_mock)
        cls._initialized = True

    @classmethod
    def register(
        cls,
        name: str,
        factory: Callable[[LLMConfig], LLMProvider] | Type[LLMProvider],
    ) -> None:
        """
        Register a provider factory.

        Args:
            name: Provider name identifier.
            factory: Callable that takes LLMConfig and returns LLMProvider,
                     or a LLMProvider subclass.
        """
        if isinstance(factory, type) and issubclass(factory, LLMProvider):
            # If a class is provided, wrap it in a factory
            cls._providers[name.lower()] = lambda config: factory()
        else:
            cls._providers[name.lower()] = factory

    @classmethod
    def get(cls, name: str, config: LLMConfig | None = None) -> LLMProvider:
        """
        Get a provider instance by name.

        Args:
            name: Provider name identifier.
            config: Optional configuration. Uses default if not provided.

        Returns:
            Configured LLMProvider instance.

        Raises:
            ProviderNotFoundError: If provider is not registered.
        """
        cls._register_defaults()

        name = name.lower()
        if name not in cls._providers:
            raise ProviderNotFoundError(
                provider_name=name,
                available_providers=list(cls._providers.keys()),
            )

        config = config or LLMConfig()
        factory = cls._providers[name]
        return factory(config)

    @classmethod
    def list_providers(cls) -> list[str]:
        """Return list of registered provider names."""
        cls._register_defaults()
        return list(cls._providers.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a provider is registered."""
        cls._register_defaults()
        return name.lower() in cls._providers

    @classmethod
    def _create_ollama(cls, config: LLMConfig) -> OllamaProvider:
        """Factory method for Ollama provider."""
        return OllamaProvider(
            host=config.ollama_host,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
        )

    @classmethod
    def _create_litellm(cls, config: LLMConfig) -> LiteLLMProvider:
        """Factory method for LiteLLM provider."""
        return LiteLLMProvider(
            model=config.model,
            api_key=config.api_key,
            api_base=config.api_base,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
        )

    @classmethod
    def _create_mock(cls, config: LLMConfig) -> MockProvider:
        """Factory method for Mock provider."""
        return MockProvider(model=config.model)


def get_llm_provider(config: LLMConfig | None = None) -> LLMProvider:
    """
    Get an LLM provider instance based on configuration.

    Convenience function that uses LLMRegistry to instantiate
    the provider specified in the configuration.

    Args:
        config: Optional configuration. If not provided, loads from environment.

    Returns:
        Configured LLMProvider instance.

    Example:
        provider = get_llm_provider()  # Uses env vars
        response = await provider.complete("Hello!")
    """
    config = config or LLMConfig()
    return LLMRegistry.get(config.provider, config)


def create_provider(name: str, **kwargs: Any) -> LLMProvider:
    """
    Create a provider with explicit parameters.

    Alternative to get_llm_provider() when you want to specify
    parameters directly instead of via LLMConfig.

    Args:
        name: Provider name ('ollama', 'litellm', 'mock').
        **kwargs: Provider-specific parameters.

    Returns:
        Configured LLMProvider instance.

    Example:
        >>> provider = create_provider('ollama', host='http://localhost:11434')
    """
    # Map common parameter names to LLMConfig field names
    config_kwargs = {"provider": name}

    # Handle provider-specific parameter mapping
    if name.lower() == "ollama":
        config_kwargs["ollama_host"] = kwargs.pop("host", kwargs.pop("ollama_host", "http://localhost:11434"))

    # Pass through remaining kwargs that match LLMConfig fields
    config_kwargs.update(kwargs)

    config = LLMConfig(**config_kwargs)
    return LLMRegistry.get(name, config)
