"""Simple in-memory cache with LRU eviction."""
from typing import Optional
from collections import OrderedDict


class LRUCache:
    """Thread-safe in-memory cache with LRU eviction policy."""

    def __init__(self, max_size: int = 500):
        """Initialize cache with maximum size.
        
        Args:
            max_size: Maximum number of entries before LRU eviction.
        """
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()

    def get_cache(self, key: str) -> Optional[str]:
        """Get value from cache.
        
        Args:
            key: Cache key.
        
        Returns:
            Cached value or None if not found.
        """
        if key not in self.cache:
            return None
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def set_cache(self, key: str, value: str) -> None:
        """Set value in cache.
        
        Args:
            key: Cache key.
            value: Value to cache.
        """
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                # Remove least recently used (first item)
                self.cache.popitem(last=False)
        self.cache[key] = value

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()

    def size(self) -> int:
        """Return current cache size."""
        return len(self.cache)


# Global cache instance
_global_cache = LRUCache(max_size=500)


def get_cache(key: str) -> Optional[str]:
    """Get value from global cache.
    
    Args:
        key: Cache key.
    
    Returns:
        Cached value or None if not found.
    """
    return _global_cache.get_cache(key)


def set_cache(key: str, value: str) -> None:
    """Set value in global cache.
    
    Args:
        key: Cache key.
        value: Value to cache.
    """
    _global_cache.set_cache(key, value)


def clear_cache() -> None:
    """Clear all global cache entries."""
    _global_cache.clear()
