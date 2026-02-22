"""Ollama LLM provider implementation."""

import httpx
from typing import Any

from llm.base import LLMProvider, LLMResponse
from llm.errors import LLMConnectionError, LLMCompletionError


class OllamaProvider(LLMProvider):
    """
    Ollama provider for local LLM inference.

    Uses Ollama's REST API for completions and chat.
    Default provider for local development.
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama3.2",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 120,
    ):
        """
        Initialize the Ollama provider.

        Args:
            host: Ollama server URL.
            model: Model name to use.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.
        """
        super().__init__(name="ollama")
        self._host = host.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

    async def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: The text prompt to complete.
            **kwargs: Additional parameters (model, temperature, etc.).

        Returns:
            LLMResponse with generated completion.
        """
        model = kwargs.get("model", self._model)
        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._host}/api/generate",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                return LLMResponse(
                    content=data.get("response", ""),
                    model=model,
                    usage={
                        "prompt_tokens": data.get("prompt_eval_count", 0),
                        "completion_tokens": data.get("eval_count", 0),
                    },
                    raw_response=data,
                )

        except httpx.ConnectError as e:
            raise LLMConnectionError(
                provider=self.name,
                url=self._host,
                original_error=e,
            ) from e
        except httpx.HTTPStatusError as e:
            raise LLMCompletionError(
                message=f"HTTP {e.response.status_code}: {e.response.text}",
                provider=self.name,
                original_error=e,
            ) from e
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

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._host}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                message = data.get("message", {})

                return LLMResponse(
                    content=message.get("content", ""),
                    model=model,
                    usage={
                        "prompt_tokens": data.get("prompt_eval_count", 0),
                        "completion_tokens": data.get("eval_count", 0),
                    },
                    raw_response=data,
                )

        except httpx.ConnectError as e:
            raise LLMConnectionError(
                provider=self.name,
                url=self._host,
                original_error=e,
            ) from e
        except httpx.HTTPStatusError as e:
            raise LLMCompletionError(
                message=f"HTTP {e.response.status_code}: {e.response.text}",
                provider=self.name,
                original_error=e,
            ) from e
        except Exception as e:
            raise LLMCompletionError(
                message=str(e),
                provider=self.name,
                original_error=e,
            ) from e

    async def health_check(self) -> dict[str, Any]:
        """
        Check Ollama server health.

        Returns:
            Health status dictionary.
        """
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self._host}/api/tags")
                is_healthy = response.status_code == 200

                if is_healthy:
                    data = response.json()
                    models = data.get("models", [])
                    model_names = [m.get("name", "") for m in models]

                    return {
                        "status": "healthy",
                        "healthy": True,
                        "provider": self.name,
                        "host": self._host,
                        "models": model_names,
                        "model_available": self._model in model_names,
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "healthy": False,
                        "provider": self.name,
                        "error": f"HTTP {response.status_code}",
                    }

        except httpx.ConnectError:
            return {
                "status": "unhealthy",
                "healthy": False,
                "provider": self.name,
                "error": "Cannot connect to Ollama server",
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "healthy": False,
                "provider": self.name,
                "error": str(e),
            }

    def __repr__(self) -> str:
        return f"OllamaProvider(host='{self._host}', model='{self._model}')"
