"""Embedding cache for efficient feature storage and retrieval."""

import pickle
import sqlite3
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingCache:
    """SQLite-based cache for storing embeddings."""

    def __init__(self, cache_dir: Path, cache_name: str = "embeddings.db"):
        """Initialize the cache.

        Args:
            cache_dir: Directory to store cache database
            cache_name: Name of the cache database file
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = cache_dir / cache_name

        self._init_db()
        logger.info("Initialized embedding cache", path=str(self.db_path))

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    key TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_created_at ON embeddings(created_at)
                """
            )
            conn.commit()

    def get(self, key: str) -> Optional[np.ndarray]:
        """Retrieve embedding from cache.

        Args:
            key: Cache key (typically SHA256 hash)

        Returns:
            Numpy array or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT embedding FROM embeddings WHERE key = ?",
                    (key,),
                )
                row = cursor.fetchone()

                if row:
                    embedding = pickle.loads(row[0])
                    logger.debug("Cache hit", key=key[:16] + "...")
                    return embedding
                else:
                    logger.debug("Cache miss", key=key[:16] + "...")
                    return None

        except Exception as e:
            logger.warning("Failed to retrieve from cache", key=key[:16] + "...", error=str(e))
            return None

    def set(self, key: str, embedding: np.ndarray, metadata: Optional[dict] = None) -> None:
        """Store embedding in cache.

        Args:
            key: Cache key (typically SHA256 hash)
            embedding: Numpy array to store
            metadata: Optional metadata dictionary
        """
        try:
            embedding_blob = pickle.dumps(embedding)
            metadata_str = pickle.dumps(metadata) if metadata else None

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO embeddings (key, embedding, metadata)
                    VALUES (?, ?, ?)
                    """,
                    (key, embedding_blob, metadata_str),
                )
                conn.commit()

            logger.debug("Cached embedding", key=key[:16] + "...", shape=embedding.shape)

        except Exception as e:
            logger.warning("Failed to cache embedding", key=key[:16] + "...", error=str(e))

    def delete(self, key: str) -> None:
        """Delete embedding from cache.

        Args:
            key: Cache key
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM embeddings WHERE key = ?", (key,))
                conn.commit()
            logger.debug("Deleted cached embedding", key=key[:16] + "...")
        except Exception as e:
            logger.warning("Failed to delete from cache", key=key[:16] + "...", error=str(e))

    def clear(self) -> None:
        """Clear all cached embeddings."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM embeddings")
                conn.commit()
            logger.info("Cleared embedding cache")
        except Exception as e:
            logger.error("Failed to clear cache", error=str(e))

    def size(self) -> int:
        """Get number of cached embeddings.

        Returns:
            Number of entries in cache
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
                count = cursor.fetchone()[0]
            return count
        except Exception as e:
            logger.warning("Failed to get cache size", error=str(e))
            return 0

    def cleanup_old(self, days: int = 30) -> int:
        """Remove old cache entries.

        Args:
            days: Remove entries older than this many days

        Returns:
            Number of entries removed
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM embeddings
                    WHERE created_at < datetime('now', '-' || ? || ' days')
                    """,
                    (days,),
                )
                deleted = cursor.rowcount
                conn.commit()

            logger.info("Cleaned up old cache entries", deleted=deleted, days=days)
            return deleted

        except Exception as e:
            logger.error("Failed to cleanup cache", error=str(e))
            return 0
