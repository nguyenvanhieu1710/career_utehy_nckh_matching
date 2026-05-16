# app/core/redis_config.py

import os
import redis.asyncio as aioredis
from typing import Optional
import logging
import json
from app.core.config import settings

logger = logging.getLogger(__name__)

class RedisManager:
    """Redis connection manager for caching"""
    
    def __init__(self):
        self._client: Optional[aioredis.Redis] = None
        self._is_connected = False

    async def get_client(self) -> aioredis.Redis:
        """Get async Redis client"""
        if self._client is None:
            try:
                self._client = aioredis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    password=settings.REDIS_PASSWORD if settings.REDIS_PASSWORD else None,
                    db=settings.REDIS_DB,
                    decode_responses=True,
                    socket_timeout=1.0,
                    socket_connect_timeout=1.0,
                    retry_on_timeout=False
                )
                
                # Connection will be tested on first command with 1s timeout
                self._is_connected = True
                logger.info(f"✅ Redis client initialized: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
                
            except Exception as e:
                logger.error(f"❌ Redis connection failed: {e}")
                self._is_connected = False
                # Don't raise - allow app to work without Redis
                
        return self._client

    async def health_check(self) -> bool:
        """Check Redis connection health"""
        try:
            if self._client:
                await self._client.ping()
                return True
            return False
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            self._is_connected = False
            return False

    def is_connected(self) -> bool:
        """Check if Redis is connected"""
        return self._is_connected

    async def close(self):
        """Close Redis connection"""
        try:
            if self._client:
                await self._client.aclose()
                self._client = None
            self._is_connected = False
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")

# Global Redis manager
redis_manager = RedisManager()

class CacheService:
    """Service for caching operations"""
    
    @staticmethod
    async def get_jobs_cache() -> Optional[list]:
        """Get cached jobs data"""
        try:
            client = await redis_manager.get_client()
            if not client or not redis_manager.is_connected():
                return None
                
            cached_data = await client.get("jobs_cache")
            if cached_data:
                logger.info("✅ Jobs loaded from Redis cache")
                return json.loads(cached_data)
            return None
        except Exception as e:
            logger.warning(f"Failed to get jobs from cache: {e}")
            return None

    @staticmethod
    async def set_jobs_cache(jobs: list, expire_seconds: int = 3600):
        """Cache jobs data (default 1 hour)"""
        try:
            client = await redis_manager.get_client()
            if not client or not redis_manager.is_connected():
                return False
                
            # Convert jobs to JSON serializable format
            jobs_data = []
            for job in jobs:
                if hasattr(job, 'dict'):
                    jobs_data.append(job.dict())
                else:
                    jobs_data.append(job)
            
            await client.setex(
                "jobs_cache", 
                expire_seconds, 
                json.dumps(jobs_data, default=str)
            )
            logger.info(f"✅ Cached {len(jobs)} jobs for {expire_seconds}s")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache jobs: {e}")
            return False

    @staticmethod
    async def get_match_cache(cv_hash: str) -> Optional[dict]:
        """Get cached match results for CV"""
        try:
            client = await redis_manager.get_client()
            if not client or not redis_manager.is_connected():
                return None
                
            cache_key = f"match:{cv_hash}"
            cached_data = await client.get(cache_key)
            if cached_data:
                logger.info(f"✅ Match results loaded from cache for CV: {cv_hash[:8]}...")
                return json.loads(cached_data)
            return None
        except Exception as e:
            logger.warning(f"Failed to get match cache: {e}")
            return None

    @staticmethod
    async def set_match_cache(cv_hash: str, match_results: dict, expire_seconds: int = 1800):
        """Cache match results (default 30 minutes)"""
        try:
            client = await redis_manager.get_client()
            if not client or not redis_manager.is_connected():
                return False
                
            cache_key = f"match:{cv_hash}"
            await client.setex(
                cache_key,
                expire_seconds,
                json.dumps(match_results, default=str)
            )
            logger.info(f"✅ Cached match results for CV: {cv_hash[:8]}...")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache match results: {e}")
            return False

    @staticmethod
    async def clear_jobs_cache():
        """Clear jobs cache (call when jobs are updated)"""
        try:
            client = await redis_manager.get_client()
            if not client or not redis_manager.is_connected():
                return False
                
            await client.delete("jobs_cache")
            logger.info("✅ Jobs cache cleared")
            return True
        except Exception as e:
            logger.warning(f"Failed to clear jobs cache: {e}")
            return False

    @staticmethod
    def generate_cv_hash(cv_text: str) -> str:
        """Generate hash for CV content for caching"""
        import hashlib
        return hashlib.md5(cv_text.encode()).hexdigest()

# Convenience functions
async def get_redis_client():
    """Get Redis client"""
    return await redis_manager.get_client()

async def redis_health_check() -> bool:
    """Check Redis health"""
    return await redis_manager.health_check()

async def close_redis():
    """Close Redis connection"""
    await redis_manager.close()