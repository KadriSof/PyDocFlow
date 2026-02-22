"""Configuration model for LLM providers."""

import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseSettings):
    """
    Configuration for LLM provider selection and settings.

    Reads configuration from environment variables with sensible defaults.
    Ollama is the default provider for local development.
    """

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Provider selection
    provider: str = Field(
        default="ollama",
        description="LLM provider name (ollama, litellm, mock)",
    )

    # Model configuration
    model: str = Field(
        default="llama3.2",
        description="Model identifier (e.g., 'llama3.2', 'gpt-4', 'ollama/llama3.2')",
    )

    # API configuration (for external providers via LiteLLM)
    api_key: str | None = Field(
        default=None,
        description="API key for external providers",
    )

    api_base: str | None = Field(
        default=None,
        description="Base URL for API requests",
    )

    # Ollama-specific configuration
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama server host URL",
    )

    # Generation parameters
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 = deterministic, 2.0 = creative)",
    )

    max_tokens: int = Field(
        default=2048,
        gt=0,
        description="Maximum tokens to generate",
    )

    timeout: int = Field(
        default=120,
        gt=0,
        description="Request timeout in seconds",
    )

    # Optional: Provider-specific model mappings
    # e.g., {"openai": "gpt-4", "anthropic": "claude-3-opus"}
    provider_models: dict[str, str] = Field(
        default_factory=dict,
        description="Model mapping per provider",
    )

    @property
    def is_ollama(self) -> bool:
        """Check if Ollama is the selected provider."""
        return self.provider.lower() == "ollama"

    @property
    def is_litellm(self) -> bool:
        """Check if LiteLLM (external provider) is selected."""
        return self.provider.lower() == "litellm"

    @property
    def is_mock(self) -> bool:
        """Check if mock provider is selected (for testing)."""
        return self.provider.lower() == "mock"

    def get_model_for_provider(self, provider_name: str) -> str:
        """Get the model identifier for a specific provider."""
        return self.provider_models.get(provider_name, self.model)

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create configuration from environment variables."""
        return cls()
