"""Custom exceptions for LLM module."""


class LLMError(Exception):
    """Base exception for all LLM-related errors."""

    def __init__(self, message: str, provider: str | None = None):
        self.message = message
        self.provider = provider
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if self.provider:
            return f"[{self.provider}] {self.message}"
        return self.message


class ProviderNotFoundError(LLMError):
    """Raised when a requested LLM provider is not found in the registry."""

    def __init__(self, provider_name: str, available_providers: list[str] | None = None):
        message = f"Provider '{provider_name}' not found"
        if available_providers:
            message += f". Available: {', '.join(available_providers)}"
        super().__init__(message, provider=provider_name)


class ProviderConfigurationError(LLMError):
    """Raised when provider configuration is invalid or missing required fields."""

    def __init__(self, provider_name: str, missing_fields: list[str] | None = None):
        message = f"Invalid configuration for provider '{provider_name}'"
        if missing_fields:
            message += f". Missing fields: {', '.join(missing_fields)}"
        super().__init__(message, provider=provider_name)


class LLMCompletionError(LLMError):
    """Raised when LLM completion fails."""

    def __init__(self, message: str, provider: str, original_error: Exception | None = None):
        self.original_error = original_error
        super().__init__(f"Completion failed: {message}", provider=provider)


class LLMConnectionError(LLMError):
    """Raised when connection to LLM provider fails."""

    def __init__(self, provider: str, url: str | None = None, original_error: Exception | None = None):
        self.original_error = original_error
        message = f"Connection failed"
        if url:
            message += f" to {url}"
        super().__init__(message, provider=provider)
