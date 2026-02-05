"""Database connection and session management."""

from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from modelguard.core.config import Config, get_config
from modelguard.storage.models import Base


class Database:
    """Database connection manager."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize database connection.

        Args:
            config: Configuration object. If None, uses global config.
        """
        self.config = config or get_config()
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None

    @property
    def engine(self) -> Engine:
        """Get or create the database engine."""
        if self._engine is None:
            self._engine = create_engine(
                self.config.database.url,
                echo=self.config.database.echo,
            )
        return self._engine

    @property
    def session_factory(self) -> sessionmaker:
        """Get or create the session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
            )
        return self._session_factory

    def create_tables(self) -> None:
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)

    def drop_tables(self) -> None:
        """Drop all database tables."""
        Base.metadata.drop_all(bind=self.engine)

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions.

        Yields:
            SQLAlchemy session

        Example:
            with db.session() as session:
                session.add(record)
                session.commit()
        """
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_session(self) -> Session:
        """Get a new session (caller responsible for closing)."""
        return self.session_factory()


# Global database instance
_database: Optional[Database] = None


def init_database(config: Optional[Config] = None) -> Database:
    """
    Initialize the global database instance.

    Args:
        config: Configuration object. If None, uses global config.

    Returns:
        Database instance
    """
    global _database
    _database = Database(config)
    _database.create_tables()
    return _database


def get_database() -> Database:
    """
    Get the global database instance.

    Returns:
        Database instance

    Raises:
        RuntimeError: If database not initialized
    """
    global _database
    if _database is None:
        _database = init_database()
    return _database
