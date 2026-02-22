from abc import ABC, abstractmethod
from typing import Any


class BaseService(ABC):
    """
    Abstract base class for all service implementations.

    Provides common lifecycle management, health checking,
    and logging infrastructure for service classes.
    """

    def __init__(self, name: str | None = None):
        """
        Initialize the base service.

        Args:
            name: Optional service name for identification.
                  Defaults to the class name if not provided.
        """
        self._name = name or self.__class__.__name__
        self._is_running = False

    @property
    def name(self) -> str:
        """Return the service name."""
        return self._name

    @property
    def is_running(self) -> bool:
        """Return whether the service is currently running."""
        return self._is_running

    @abstractmethod
    async def start(self) -> None:
        """
        Start the service and initialize resources.
        (Should be called before using the service)
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """
        Stop the service and clean up resources.
        (Should be called when shutting down the application)
        """
        pass

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """
        Check the health status of the service.

        Returns:
            A dictionary containing health status information.
            Should include at minimum:
            - 'status': 'healthy' | 'unhealthy' | 'degraded'
            - 'healthy': bool
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self._name}', running={self._is_running})"
