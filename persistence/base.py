from abc import ABC, abstractmethod
from typing import Any, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from persistence.schemas import DocumentCreate

T = TypeVar("T")


class BaseClient(ABC):
    """
    Abstract base class for database clients that interact with external services.

    This class defines the interface that all database client implementations
    must follow, ensuring consistency across different database providers
    (e.g., MongoDB, Firestore, PostgreSQL).
    """

    @property
    @abstractmethod
    def client(self) -> Any:
        """
        Return the raw client/connection instance.

        Returns:
            The underlying client object (e.g., AsyncIOMotorClient for MongoDB).

        Raises:
            RuntimeError: If not connected.
        """
        ...

    @property
    @abstractmethod
    def db(self) -> Any:
        """
        Return the database instance.

        Returns:
            The database object (e.g., AsyncIOMotorDatabase for MongoDB).

        Raises:
            RuntimeError: If not connected.
        """
        ...

    @property
    @abstractmethod
    def engine(self) -> Any:
        """
        Return the ODM/engine instance for object-document mapping.

        Returns:
            The ODM engine object (e.g., AIOEngine for ODMantic).

        Raises:
            RuntimeError: If not connected.
        """
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if the client is connected to the database.

        Returns:
            True if connected, False otherwise.
        """
        ...

    @abstractmethod
    async def connect(self) -> None:
        """
        Establish a connection to the database service.

        Raises:
            ConnectionError: If connection fails.
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Close the connection to the database service.
        """
        ...

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """
        Perform a health check on the database connection.

        Returns:
            Dictionary containing health status information.
        """
        ...


class BaseRepository[T](ABC):
    """
    Abstract base class for repositories that manage entity persistence.

    Args:
        Generic[T]: The type of entity this repository manages.
    """

    @abstractmethod
    async def save(self, entity: T) -> T:
        """
        Save an entity to the database.

        Args:
            entity: The entity to save.

        Returns:
            The saved entity (possibly with updated fields like ID).
        """
        ...

    @abstractmethod
    async def get_by_id(self, id: str) -> T | None:
        """
        Retrieve an entity by its ID.

        Args:
            id: The unique identifier of the entity.

        Returns:
            The entity if found, None otherwise.
        """
        ...

    @abstractmethod
    async def list_all(self, skip: int = 0, limit: int = 100) -> list[T]:
        """
        List all entities in the database.

        Args:
            skip: Number of entities to skip (for pagination).
            limit: Maximum number of entities to return.

        Returns:
            List of entities.
        """
        ...

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """
        Delete an entity by its ID.

        Args:
            id: The unique identifier of the entity to delete.

        Returns:
            True if deleted successfully, False if entity not found.
        """
        ...


class BaseDocumentRepository[T](BaseRepository[T], ABC):
    """
    Abstract base class for document repositories.
    """

    @abstractmethod
    async def save_from_schema(self, schema: "DocumentCreate") -> T:
        """
        Save a document from a DocumentCreate schema.

        Args:
            schema: The DocumentCreate schema containing document and page data.

        Returns:
            The saved document object.
        """
        ...
