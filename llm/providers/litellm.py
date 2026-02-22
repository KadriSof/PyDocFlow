"""LiteLLM provider for external LLM APIs."""

from typing import Any

from llm.base import LLMProvider, LLMResponse
from llm.errors import LLMCompletionError, ProviderConfigurationError

# LiteLLM is imported lazily to avoid import errors if not installed


class LiteLLMProvider(LLMProvider):
    """
    LiteLLM provider for external LLM APIs.

    Supports OpenAI, Anthropic, Google, and other providers via LiteLLM.
    Provider is determined by the model prefix (e.g., 'openai/gpt-4').
    """

    def __init__(
        self,
        model: str = "openai/gpt-4",
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 120,
    ):
        """
        Initialize the LiteLLM provider.

        Args:
            model: Model identifier with provider prefix.
            api_key: API key for the provider.
            api_base: Optional base URL for API requests.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.
        """
        super().__init__(name="litellm")

        if not model or "/" not in model:
            raise ProviderConfigurationError(
                provider_name="litellm",
                missing_fields=["model (must include provider prefix, e.g., 'openai/gpt-4')"],
            )

        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

        # Import litellm lazily
        import litellm
        self._litellm = litellm

        # Configure litellm
        if self._api_key:
            self._litellm.api_key = self._api_key
        if self._api_base:
            self._litellm.api_base = self._api_base

    async def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: The text prompt to complete.
            **kwargs: Additional parameters.

        Returns:
            LLMResponse with generated completion.
        """
        model = kwargs.get("model", self._model)
        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)

        try:
            # Use litellm's acompletion for async
            response = await self._litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self._timeout,
            )

            content = response.choices[0].message.content or ""

            return LLMResponse(
                content=content,
                model=model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                raw_response=response,
            )

        except Exception as e:
            raise LLMCompletionError(
                message=str(e),
                provider=self.name,
                original_error=e,
            ) from e

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        """
        Generate a response for a chat conversation.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            **kwargs: Additional parameters.

        Returns:
            LLMResponse with generated response.
        """
        model = kwargs.get("model", self._model)
        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)

        try:
            response = await self._litellm.acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self._timeout,
            )

            content = response.choices[0].message.content or ""

            return LLMResponse(
                content=content,
                model=model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                raw_response=response,
            )

        except Exception as e:
            raise LLMCompletionError(
                message=str(e),
                provider=self.name,
                original_error=e,
            ) from e

    async def health_check(self) -> dict[str, Any]:
        """
        Check provider health with a simple test request.

        Returns:
            Health status dictionary.
        """
        try:
            # Simple test with minimal tokens
            response = await self._litellm.acompletion(
                model=self._model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
                timeout=10,
            )

            return {
                "status": "healthy",
                "healthy": True,
                "provider": self.name,
                "model": self._model,
                "api_base": self._api_base or "default",
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "healthy": False,
                "provider": self.name,
                "model": self._model,
                "error": str(e),
            }

    def __repr__(self) -> str:
        return f"LiteLLMProvider(model='{self._model}')"
