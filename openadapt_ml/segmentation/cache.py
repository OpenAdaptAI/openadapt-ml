"""Caching utilities for segmentation pipeline.

This module provides caching functionality to avoid re-processing
recordings and to speed up iterative development.
"""

import hashlib
import json
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CacheManager:
    """Manages cached artifacts for the segmentation pipeline.

    Provides a simple file-based cache with optional TTL (time-to-live)
    and size limits.

    Example:
        >>> cache = CacheManager()
        >>> cache.set("key", {"data": "value"})
        >>> data = cache.get("key")
        >>> cache.clear()
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        ttl_hours: Optional[int] = None,
        max_size_mb: Optional[int] = None,
    ):
        """Initialize the cache manager.

        Args:
            cache_dir: Directory for cache files. Defaults to ~/.openadapt/cache/segmentation
            ttl_hours: Time-to-live in hours. None for no expiration.
            max_size_mb: Maximum cache size in MB. None for no limit.
        """
        self.cache_dir = (
            cache_dir or Path.home() / ".openadapt" / "cache" / "segmentation"
        )
        self.ttl_hours = ttl_hours
        self.max_size_mb = max_size_mb
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        """Convert cache key to file path."""
        # Hash long keys
        if len(key) > 100:
            key = hashlib.md5(key.encode()).hexdigest()
        # Sanitize key for filesystem
        safe_key = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
        return self.cache_dir / f"{safe_key}.json"

    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found/expired.
        """
        path = self._key_to_path(key)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())

            # Check TTL
            if self.ttl_hours is not None:
                cached_at = datetime.fromisoformat(data.get("_cached_at", "1970-01-01"))
                if datetime.now() - cached_at > timedelta(hours=self.ttl_hours):
                    path.unlink()
                    return None

            return data.get("value")

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Invalid cache entry for {key}: {e}")
            path.unlink(missing_ok=True)
            return None

    def set(self, key: str, value: Any) -> None:
        """Set a value in the cache.

        Args:
            key: Cache key.
            value: Value to cache (must be JSON serializable).
        """
        # Enforce size limit
        if self.max_size_mb is not None:
            self._enforce_size_limit()

        path = self._key_to_path(key)
        data = {
            "value": value,
            "_cached_at": datetime.now().isoformat(),
        }

        try:
            path.write_text(json.dumps(data))
        except (TypeError, OSError) as e:
            logger.warning(f"Failed to cache {key}: {e}")

    def delete(self, key: str) -> bool:
        """Delete a cache entry.

        Args:
            key: Cache key.

        Returns:
            True if entry was deleted, False if not found.
        """
        path = self._key_to_path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries.

        Args:
            pattern: Optional glob pattern to match keys.

        Returns:
            Number of entries cleared.
        """
        count = 0
        glob_pattern = f"*{pattern}*.json" if pattern else "*.json"

        for path in self.cache_dir.glob(glob_pattern):
            path.unlink()
            count += 1

        return count

    def _enforce_size_limit(self) -> None:
        """Remove oldest entries if cache exceeds size limit."""
        if self.max_size_mb is None:
            return

        # Calculate current size
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.json"))
        max_bytes = self.max_size_mb * 1024 * 1024

        if total_size <= max_bytes:
            return

        # Sort by modification time (oldest first)
        files = sorted(
            self.cache_dir.glob("*.json"),
            key=lambda f: f.stat().st_mtime,
        )

        # Remove oldest until under limit
        for path in files:
            if total_size <= max_bytes:
                break
            total_size -= path.stat().st_size
            path.unlink()
            logger.debug(f"Evicted cache entry: {path.name}")

    def stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with cache stats (count, size, oldest, newest).
        """
        files = list(self.cache_dir.glob("*.json"))
        if not files:
            return {
                "count": 0,
                "size_mb": 0,
                "oldest": None,
                "newest": None,
            }

        mtimes = [f.stat().st_mtime for f in files]
        total_size = sum(f.stat().st_size for f in files)

        return {
            "count": len(files),
            "size_mb": total_size / (1024 * 1024),
            "oldest": datetime.fromtimestamp(min(mtimes)).isoformat(),
            "newest": datetime.fromtimestamp(max(mtimes)).isoformat(),
        }


class RecordingCache:
    """Cache for processed recording artifacts.

    Provides specialized caching for:
    - Frame descriptions (Stage 1)
    - Episode extractions (Stage 2)
    - Embeddings (Stage 3)
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize recording cache.

        Args:
            cache_dir: Base cache directory.
        """
        base_dir = cache_dir or Path.home() / ".openadapt" / "cache" / "segmentation"
        self.descriptions_cache = CacheManager(base_dir / "descriptions")
        self.extractions_cache = CacheManager(base_dir / "extractions")
        self.embeddings_dir = base_dir / "embeddings"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

    def get_description(self, recording_id: str, frame_hash: str) -> Optional[dict]:
        """Get cached frame description."""
        key = f"{recording_id}_{frame_hash}"
        return self.descriptions_cache.get(key)

    def set_description(
        self, recording_id: str, frame_hash: str, description: dict
    ) -> None:
        """Cache frame description."""
        key = f"{recording_id}_{frame_hash}"
        self.descriptions_cache.set(key, description)

    def get_extraction(self, recording_id: str, model: str) -> Optional[dict]:
        """Get cached episode extraction."""
        key = f"{recording_id}_{model}"
        return self.extractions_cache.get(key)

    def set_extraction(self, recording_id: str, model: str, extraction: dict) -> None:
        """Cache episode extraction."""
        key = f"{recording_id}_{model}"
        self.extractions_cache.set(key, extraction)

    def clear_recording(self, recording_id: str) -> int:
        """Clear all cache entries for a recording."""
        count = self.descriptions_cache.clear(recording_id)
        count += self.extractions_cache.clear(recording_id)

        # Clear embeddings
        for path in self.embeddings_dir.glob(f"{recording_id}*"):
            path.unlink()
            count += 1

        return count

    def clear_all(self) -> int:
        """Clear entire cache."""
        count = self.descriptions_cache.clear()
        count += self.extractions_cache.clear()

        if self.embeddings_dir.exists():
            shutil.rmtree(self.embeddings_dir)
            self.embeddings_dir.mkdir()

        return count


# Default cache instance
_default_cache: Optional[RecordingCache] = None


def get_cache() -> RecordingCache:
    """Get the default cache instance."""
    global _default_cache
    if _default_cache is None:
        _default_cache = RecordingCache()
    return _default_cache


def clear_cache(recording_id: Optional[str] = None) -> int:
    """Clear cache entries.

    Args:
        recording_id: If specified, only clear cache for this recording.

    Returns:
        Number of entries cleared.
    """
    cache = get_cache()
    if recording_id:
        return cache.clear_recording(recording_id)
    return cache.clear_all()
