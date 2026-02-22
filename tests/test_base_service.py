import pytest
from services.base import BaseService


class ConcreteService(BaseService):
    """Concrete implementation for testing BaseService."""

    async def start(self) -> None:
        self._is_running = True

    async def stop(self) -> None:
        self._is_running = False

    async def health_check(self) -> dict:
        return {
            "status": "healthy" if self._is_running else "unhealthy",
            "healthy": self._is_running,
        }


class TestBaseService:
    """Tests for BaseService abstract class."""

    def test_cannot_instantiate_abstract_class(self):
        """BaseService cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract methods"):
            BaseService()

    def test_concrete_implementation_instantiates(self):
        """Concrete implementations can be instantiated."""
        service = ConcreteService()
        assert service is not None
        assert isinstance(service, BaseService)

    def test_default_name_is_class_name(self):
        """Service name defaults to class name when not provided."""
        service = ConcreteService()
        assert service.name == "ConcreteService"

    def test_custom_name(self):
        """Service accepts custom name."""
        service = ConcreteService(name="MyCustomService")
        assert service.name == "MyCustomService"

    def test_is_running_initially_false(self):
        """Service is not running by default."""
        service = ConcreteService()
        assert service.is_running is False

    @pytest.mark.asyncio
    async def test_start_changes_running_state(self):
        """Starting the service sets is_running to True."""
        service = ConcreteService()
        assert service.is_running is False

        await service.start()

        assert service.is_running is True

    @pytest.mark.asyncio
    async def test_stop_changes_running_state(self):
        """Stopping the service sets is_running to False."""
        service = ConcreteService()
        await service.start()
        assert service.is_running is True

        await service.stop()

        assert service.is_running is False

    @pytest.mark.asyncio
    async def test_health_check_returns_status(self):
        """Health check returns status dictionary."""
        service = ConcreteService()

        await service.start()
        health = await service.health_check()

        assert "status" in health
        assert "healthy" in health
        assert health["status"] == "healthy"
        assert health["healthy"] is True

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_when_stopped(self):
        """Health check shows unhealthy when service is stopped."""
        service = ConcreteService()

        health = await service.health_check()

        assert health["status"] == "unhealthy"
        assert health["healthy"] is False

    def test_repr_includes_name_and_state(self):
        """String representation includes service name and running state."""
        service = ConcreteService(name="TestService")
        repr_str = repr(service)

        assert "TestService" in repr_str
        assert "running=False" in repr_str

    @pytest.mark.asyncio
    async def test_repr_updates_with_state(self):
        """String representation reflects current running state."""
        service = ConcreteService()

        assert "running=False" in repr(service)

        await service.start()

        assert "running=True" in repr(service)
