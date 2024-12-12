from typing import Any, Optional, Callable
from datetime import datetime, timedelta
import json
from functools import wraps
import asyncio
import hashlib
from redis import asyncio as aioredis

class Cache:
    def __init__(self, redis_url: str):
        self.redis = aioredis.from_url(redis_url)
        self.default_ttl = 300  # 5 minutes

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        value = await self.redis.get(key)
        if value:
            return json.loads(value)
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        await self.redis.set(
            key,
            json.dumps(value),
            ex=ttl or self.default_ttl
        )

    async def delete(self, key: str) -> None:
        """Delete value from cache"""
        await self.redis.delete(key)

    async def clear_pattern(self, pattern: str) -> None:
        """Clear all keys matching pattern"""
        keys = await self.redis.keys(pattern)
        if keys:
            await self.redis.delete(*keys)

def cache_key(*args, **kwargs) -> str:
    """Generate cache key from arguments"""
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
    
    # Create hash of the key parts
    key_hash = hashlib.sha256(
        ":".join(key_parts).encode()
    ).hexdigest()
    
    return f"kpi:{key_hash}"

def cached(ttl: Optional[int] = None):
    """Decorator for caching function results"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Generate cache key
            key = cache_key(func.__name__, *args, **kwargs)
            
            # Try to get from cache
            cached_value = await self.cache.get(key)
            if cached_value is not None:
                return cached_value
            
            # Get fresh value
            value = await func(self, *args, **kwargs)
            
            # Store in cache
            await self.cache.set(key, value, ttl)
            
            return value
        return wrapper
    return decorator

def invalidate_cache(pattern: str):
    """Decorator for invalidating cache after function execution"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            result = await func(self, *args, **kwargs)
            
            # Invalidate cache
            await self.cache.clear_pattern(pattern)
            
            return result
        return wrapper
    return decorator

class CacheManager:
    def __init__(self, cache: Cache):
        self.cache = cache
        self._background_tasks = set()

    async def refresh_cache(self, key: str, func: Callable, ttl: Optional[int] = None):
        """Refresh cache value in background"""
        try:
            value = await func()
            await self.cache.set(key, value, ttl)
        except Exception as e:
            # Log error but don't raise
            print(f"Error refreshing cache: {str(e)}")

    def schedule_refresh(self, key: str, func: Callable, ttl: Optional[int] = None):
        """Schedule cache refresh in background"""
        task = asyncio.create_task(self.refresh_cache(key, func, ttl))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def get_or_compute(self, 
                           key: str,
                           compute_func: Callable,
                           ttl: Optional[int] = None,
                           refresh_after: Optional[int] = None):
        """Get value from cache or compute it"""
        value = await self.cache.get(key)
        
        if value is None:
            # Compute value if not in cache
            value = await compute_func()
            await self.cache.set(key, value, ttl)
        elif refresh_after:
            # Schedule background refresh if needed
            self.schedule_refresh(key, compute_func, ttl)
        
        return value

    async def bulk_get_or_compute(self,
                                keys: list[str],
                                compute_func: Callable,
                                ttl: Optional[int] = None):
        """Get multiple values from cache or compute them"""
        # Get all cached values
        cached_values = await asyncio.gather(
            *(self.cache.get(key) for key in keys)
        )
        
        # Find missing keys
        missing_keys = [
            key for key, value in zip(keys, cached_values)
            if value is None
        ]
        
        if missing_keys:
            # Compute missing values
            computed_values = await compute_func(missing_keys)
            
            # Store computed values in cache
            await asyncio.gather(*(
                self.cache.set(key, value, ttl)
                for key, value in zip(missing_keys, computed_values)
            ))
            
            # Update cached_values with computed values
            for i, key in enumerate(keys):
                if key in missing_keys:
                    cached_values[i] = computed_values[
                        missing_keys.index(key)
                    ]
        
        return cached_values
