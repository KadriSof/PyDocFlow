import logging
from typing import Dict

from persistence.base import BaseClient, BaseDocumentRepository
from persistence.postgresql.client import PostgreSQLClient
from persistence.mongodb.client import MongoDBClient
from persistence.postgresql.repository import DocumentRepository as PostgreSQLDocumentRepository
from persistence.mongodb.repository import DocumentRepository as MongoDBDocumentRepository

logger = logging.getLogger(__name__)


class DatabaseFactory:
    """
    Factory class for managing database clients and repositories.

    This class maintains a registry of initialized database clients and
    provides methods to create repositories for specific database types.
    """

    _clients: Dict[str, BaseClient] = {}

    @classmethod
    async def get_client(cls, db_type: str) -> BaseClient:
        """
        Get or initialize a database client for the specified type.

        Args:
            db_type: The type of database ('postgresql' or 'mongodb').

        Returns:
            An initialized instance of BaseClient.

        Raises:
            ValueError: If the database type is not supported.
        """
        db_type = db_type.lower()
        if db_type not in cls._clients:
            logger.info(f"Initializing database client: {db_type}")
            if db_type == "postgresql":
                client = PostgreSQLClient()
            elif db_type == "mongodb":
                client = MongoDBClient()
            else:
                raise ValueError(f"Unsupported database type: {db_type}")

            await client.connect()
            cls._clients[db_type] = client

        return cls._clients[db_type]

    @classmethod
    def get_repository(cls, db_type: str) -> BaseDocumentRepository:
        """
        Get a repository instance for the specified database type.

        Args:
            db_type: The type of database.

        Returns:
            An instance of BaseRepository.

        Raises:
            RuntimeError: If the database client is not initialized.
            ValueError: If the database type is not supported.
        """
        db_type = db_type.lower()
        client = cls._clients.get(db_type)

        if not client:
            raise RuntimeError(
                f"Database client for '{db_type}' is not initialized. Call get_client() first."
            )

        if db_type == "postgresql":
            from persistence.postgresql.client import PostgreSQLClient

            assert isinstance(client, PostgreSQLClient)
            return PostgreSQLDocumentRepository(client)
        elif db_type == "mongodb":
            from persistence.mongodb.client import MongoDBClient

            assert isinstance(client, MongoDBClient)
            return MongoDBDocumentRepository(client)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

    @classmethod
    async def disconnect_all(cls) -> None:
        """Disconnect all active database clients."""
        for db_type, client in cls._clients.items():
            logger.info(f"Disconnecting database client: {db_type}")
            await client.disconnect()
        cls._clients.clear()

    @classmethod
    def get_active_clients(cls) -> Dict[str, BaseClient]:
        """Get all currently active database clients."""
        return cls._clients
