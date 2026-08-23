import json
import hashlib
from typing import Any, Optional
from functools import wraps
import time

class EmbeddingCache:
    """Cache for document embeddings to avoid recomputation."""
    
    def __init__(self, cache_dir: str = '.cache/embeddings'):
        self.cache_dir = cache_dir
        self.memory_cache = {}
        self.hits = 0
        self.misses = 0
    
    def _get_key(self, text: str) -> str:
        """Generate cache key from text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def get(self, text: str) -> Optional[list]:
        """Retrieve embedding from cache."""
        key = self._get_key(text)
        if key in self.memory_cache:
            self.hits += 1
            return self.memory_cache[key]
        self.misses += 1
        return None
    
    def set(self, text: str, embedding: list):
        """Store embedding in cache."""
        key = self._get_key(text)
        self.memory_cache[key] = embedding
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.2%}",
            'size': len(self.memory_cache)
        }

def timed_lru_cache(seconds: int = 300, maxsize: int = 128):
    """LRU cache with TTL expiration."""
    def decorator(func):
        cache = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            now = time.time()
            
            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < seconds:
                    return result
            
            result = func(*args, **kwargs)
            cache[key] = (result, now)
            
            # Cleanup old entries
            if len(cache) > maxsize:
                oldest = min(cache.items(), key=lambda x: x[1][1])
                del cache[oldest[0]]
            
            return result
        return wrapper
    return decorator
