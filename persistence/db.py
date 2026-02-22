import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from odmantic import AIOEngine
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from persistence.settings import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DB_NOT_CONNECTED_ERROR = "Database not connected. Call connect() first."


class DatabaseManager:
    """
    Production-ready database manager class that encapsulates MongoDB connection logic.
    
    Features:
    - Connection pooling for better performance under high load
    - Robust error handling with specific exception handling
    - Automatic retry logic for transient failures
    - Proper resource lifecycle management
    - Index management
    - Context manager support for safe connection handling
    """

    def __init__(
        self,
        settings: Settings | None = None,
        max_pool_size: int = 100,
        min_pool_size: int = 10,
        max_idle_time_ms: int = 300000,
        connect_timeout_ms: int = 5000,
        server_selection_timeout_ms: int = 30000,
    ) -> None:
        """
        Initialize the DatabaseManager.

        Args:
            settings: Settings instance. If None, creates a new one.
            max_pool_size: Maximum number of connections in the pool.
            min_pool_size: Minimum number of connections in the pool.
            max_idle_time_ms: Maximum idle time for a connection before removal.
            connect_timeout_ms: Timeout for establishing a connection.
            server_selection_timeout_ms: Timeout for server selection.
        """
        self.settings = settings or Settings()
        self._client: AsyncIOMotorClient | None = None
        self._db: AsyncIOMotorDatabase | None = None
        self._engine: AIOEngine | None = None

        # Connection pool configuration
        self._max_pool_size = max_pool_size
        self._min_pool_size = min_pool_size
        self._max_idle_time_ms = max_idle_time_ms
        self._connect_timeout_ms = connect_timeout_ms
        self._server_selection_timeout_ms = server_selection_timeout_ms

    @property
    def client(self) -> AsyncIOMotorClient:
        """Get the MongoDB client. Raises RuntimeError if not connected."""
        if self._client is None:
            raise RuntimeError(DB_NOT_CONNECTED_ERROR)

        return self._client

    @property
    def db(self) -> AsyncIOMotorDatabase:
        """Get the database instance. Raises RuntimeError if not connected."""
        if self._db is None:
            raise RuntimeError(DB_NOT_CONNECTED_ERROR)

        return self._db

    @property
    def engine(self) -> AIOEngine:
        """Get the ODMantic engine. Raises RuntimeError if not connected."""
        if self._engine is None:
            raise RuntimeError(DB_NOT_CONNECTED_ERROR)

        return self._engine

    @property
    def is_connected(self) -> bool:
        """Check if the database is connected."""
        return self._client is not None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        reraise=True,
    )
    async def _create_client(self) -> AsyncIOMotorClient:
        """
        Create a MongoDB client with connection pooling.

        Returns:
            AsyncIOMotorClient: Configured MongoDB client.

        Raises:
            ConnectionError: If connection to MongoDB fails.
            TimeoutError: If connection times out.
        """
        try:
            _client = AsyncIOMotorClient(
                self.settings.mongo_uri,
                maxPoolSize=self._max_pool_size,
                minPoolSize=self._min_pool_size,
                maxIdleTimeMS=self._max_idle_time_ms,
                connectTimeoutMS=self._connect_timeout_ms,
                serverSelectionTimeoutMS=self._server_selection_timeout_ms,
            )

            # Test the connection with a ping command
            await _client.admin.command("ping")
            logger.info(
                "Successfully connected to MongoDB with connection pooling "
                f"(maxPoolSize={self._max_pool_size}, minPoolSize={self._min_pool_size})"
            )
            return _client

        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Connection error to MongoDB: {type(e).__name__}: {e}")
            raise

        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {type(e).__name__}: {e}")
            raise ConnectionError(f"Failed to connect to MongoDB: {e}") from e

    async def _create_indexes(self) -> None:
        """
        Create necessary database indexes.

        Raises:
            RuntimeError: If index creation fails.
        """
        try:
            collection = self.db.get_collection("ocr_result")

            # Create a unique index on the "file_name" field
            await collection.create_index("file_name", unique=True)
            logger.info("Created unique index on 'file_name' field.")

            # Create an index on 'created_at' for efficient time-based queries
            await collection.create_index("created_at")
            logger.info("Created index on 'created_at' field.")

        except Exception as e:
            logger.error(f"Failed to create indexes: {type(e).__name__}: {e}")
            raise RuntimeError(f"Index creation failed: {e}") from e

    async def connect(self) -> None:
        """
        Establish connection to the database.

        Raises:
            ConnectionError: If connection fails after all retries.
        """
        if self.is_connected:
            logger.warning("Database already connected.")
            return

        self._client = await self._create_client()
        self._db = self.client.get_database(self.settings.mongo_db)
        self._engine = AIOEngine(client=self.client, database=self.settings.mongo_db)

        await self._create_indexes()
        logger.info(f"Database manager initialized for database: {self.settings.mongo_db}")

    async def disconnect(self) -> None:
        """Close the database connection and cleanup resources."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._db = None
            self._engine = None
            logger.info("Database connection closed.")
        else:
            logger.debug("No active database connection to close.")

    @asynccontextmanager
    async def connection(self) -> AsyncGenerator["DatabaseManager", None]:
        """
        Context manager for database connection.

        Usage:
            async with db_manager.connection() as db:
                # Use db.client, db.db, db.engine
        """
        await self.connect()
        try:
            yield self
        finally:
            await self.disconnect()

    async def health_check(self) -> dict[str, Any]:
        """
        Perform a health check on the database connection.

        Returns:
            dict: Health status information.
        """
        try:
            if not self.is_connected:
                return {"status": "disconnected", "healthy": False}

            # Ping the database
            await self.client.admin.command("ping")

            # Get server stats
            server_info = await self.client.server_info()

            return {
                "status": "connected",
                "healthy": True,
                "mongodb_version": server_info.get("version", "unknown"),
                "database": self.settings.mongo_db,
            }

        except Exception as e:
            logger.error(f"Health check failed: {type(e).__name__}: {e}")
            return {
                "status": "error",
                "healthy": False,
                "error": str(e),
            }


# Global instance for backward compatibility and convenience
_db_manager: DatabaseManager | None = None


def get_db_manager() -> DatabaseManager:
    """Get the global DatabaseManager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


# Backward compatibility functions (deprecated - use DatabaseManager directly)
async def connect_to_db() -> None:
    """Initialize the global database connection."""
    await get_db_manager().connect()


async def close_db_connection() -> None:
    """Close the global database connection."""
    await get_db_manager().disconnect()


# Expose properties for backward compatibility
@property
async def client() -> AsyncIOMotorClient | None:
    """Get the global client (backward compatibility)."""
    db_manager = get_db_manager()
    return db_manager.client if db_manager.is_connected else None


@property
async def db() -> AsyncIOMotorDatabase | None:
    """Get the global database (backward compatibility)."""
    db_manager = get_db_manager()
    return db_manager.db if db_manager.is_connected else None


@property
async def engine() -> AIOEngine | None:
    """Get the global engine (backward compatibility)."""
    db_manager = get_db_manager()
    return db_manager.engine if db_manager.is_connected else None
