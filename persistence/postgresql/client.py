import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
    AsyncEngine,
)
from sqlalchemy.pool import NullPool
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from persistence.base import BaseClient
from persistence.postgresql.settings import Settings

logger = logging.getLogger(__name__)

DB_NOT_CONNECTED_ERROR = "Database not connected. Call connect() first."


class PostgreSQLClient(BaseClient):
    """
    Production-ready PostgreSQL client using SQLAlchemy with asyncpg.

    Features:
    - Async session management for better performance
    - Connection pooling configuration
    - Robust error handling with specific exception handling
    - Automatic retry logic for transient failures
    - Proper resource lifecycle management
    - Context manager support for safe connection handling
    """

    def __init__(
            self,
            settings: Settings | None = None,
            pool_size: int = 10,
            max_overflow: int = 20,
            pool_timeout: int = 30,
            pool_recycle: int = 1800,
            echo: bool = False,
    ) -> None:
        """
        Initialize the PostgreSQL client.

        Args:
            settings: Settings instance. If None, creates a new one.
            pool_size: Number of connections to keep in the pool.
            max_overflow: Max connections that can be opened beyond pool_size.
            pool_timeout: Timeout for getting a connection from the pool.
            pool_recycle: Time in seconds after which a connection is recycled.
            echo: If True, log all SQL statements.
        """
        self.settings = settings or Settings()
        self._engine: AsyncEngine | None = None
        self._async_session_factory: async_sessionmaker[AsyncSession] | None = None
        self._connected = False

        # Connection pool configuration
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._pool_timeout = pool_timeout
        self._pool_recycle = pool_recycle
        self._echo = echo

    @property
    def client(self) -> AsyncEngine:
        """Get the SQLAlchemy async engine. Raises RuntimeError if not connected."""
        if self._engine is None:
            raise RuntimeError(DB_NOT_CONNECTED_ERROR)
        return self._engine

    @property
    def db(self) -> async_sessionmaker[AsyncSession]:
        """Get the async session factory. Raises RuntimeError if not connected."""
        if self._async_session_factory is None:
            raise RuntimeError(DB_NOT_CONNECTED_ERROR)
        return self._async_session_factory

    @property
    def engine(self) -> AsyncEngine:
        """Get the SQLAlchemy async engine. Raises RuntimeError if not connected."""
        if self._engine is None:
            raise RuntimeError(DB_NOT_CONNECTED_ERROR)
        return self._engine

    @property
    def is_connected(self) -> bool:
        """Check if the database is connected."""
        return self._connected

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        reraise=True,
    )
    async def _create_engine(self) -> AsyncEngine:
        """
        Create a SQLAlchemy async engine with connection pooling.

        Returns:
            AsyncEngine: Configured SQLAlchemy engine.

        Raises:
            ConnectionError: If connection to PostgreSQL fails.
            TimeoutError: If connection times out.
        """
        try:
            _engine = create_async_engine(
                self.settings.database_url,
                pool_size=self._pool_size,
                max_overflow=self._max_overflow,
                pool_timeout=self._pool_timeout,
                pool_recycle=self._pool_recycle,
                pool_pre_ping=True,
                echo=self._echo,
            )

            # Test the connection
            async with _engine.begin() as conn:
                await conn.execute("SELECT 1")

            logger.info(
                "Successfully connected to PostgreSQL with connection pooling "
                f"(pool_size={self._pool_size}, max_overflow={self._max_overflow})"
            )
            return _engine

        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Connection error to PostgreSQL: {type(e).__name__}: {e}")
            raise

        except Exception as e:
            logger.error(f"Unexpected error connecting to PostgreSQL: {type(e).__name__}: {e}")
            raise ConnectionError(f"Failed to connect to PostgreSQL: {e}") from e

    async def _create_tables(self) -> None:
        """
        Create database tables.

        Raises:
            RuntimeError: If table creation fails.
        """
        try:
            # Import models to ensure they are registered with SQLAlchemy metadata
            from persistence.postgresql.models import Document, Page

            async with self._engine.begin() as conn:
                await conn.run_sync(Document.metadata.create_all)

            logger.info("Database tables created successfully.")

        except Exception as e:
            logger.error(f"Failed to create tables: {type(e).__name__}: {e}")
            raise RuntimeError(f"Table creation failed: {e}") from e

    async def connect(self) -> None:
        """
        Establish connection to the database.

        Raises:
            ConnectionError: If connection fails after all retries.
        """
        if self.is_connected:
            logger.warning("Database already connected.")
            return

        self._engine = await self._create_engine()
        self._async_session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        await self._create_tables()
        self._connected = True
        logger.info(f"Database manager initialized for database: {self.settings.postgres_db}")

    async def disconnect(self) -> None:
        """Close the database connection and cleanup resources."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._async_session_factory = None
            self._connected = False
            logger.info("Database connection closed.")
        else:
            logger.debug("No active database connection to close.")

    @asynccontextmanager
    async def connection(self) -> AsyncGenerator["PostgreSQLClient", None]:
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

            # Test connection with a simple query
            async with self._engine.begin() as conn:
                result = await conn.execute("SELECT version()")
                version = result.scalar()

            return {
                "status": "connected",
                "healthy": True,
                "postgresql_version": version or "unknown",
                "database": self.settings.postgres_db,
            }

        except Exception as e:
            logger.error(f"Health check failed: {type(e).__name__}: {e}")
            return {
                "status": "error",
                "healthy": False,
                "error": str(e),
            }


# Global instance for backward compatibility and convenience
_db_manager: PostgreSQLClient | None = None


def get_db_manager() -> PostgreSQLClient:
    """Get the global DatabaseManager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = PostgreSQLClient()
    return _db_manager


# Backward compatibility functions (deprecated - use DatabaseManager directly)
async def connect_to_db() -> None:
    """Initialize the global database connection."""
    await get_db_manager().connect()


async def close_db_connection() -> None:
    """Close the global database connection."""
    await get_db_manager().disconnect()


# Expose async functions for backward compatibility
async def client() -> AsyncEngine | None:
    """Get the global client (backward compatibility)."""
    db_manager = get_db_manager()
    return db_manager.client if db_manager.is_connected else None


async def db() -> async_sessionmaker[AsyncSession] | None:
    """Get the global database (backward compatibility)."""
    db_manager = get_db_manager()
    return db_manager.db if db_manager.is_connected else None


async def engine() -> AsyncEngine | None:
    """Get the global engine (backward compatibility)."""
    db_manager = get_db_manager()
    return db_manager.engine if db_manager.is_connected else None
