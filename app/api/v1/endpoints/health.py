# app/api/v1/endpoints/health.py

from fastapi import APIRouter, HTTPException
from app.core.database import get_database
from app.services.job_service import JobService
from app.core.redis_config import redis_health_check
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/")
async def health_check():
    """
    Comprehensive health check with database connectivity and system status
    """
    health_status = {
        "status": "healthy",
        "service": "Career UTEHY NCKH - CV Job Matching Service",
        "version": "1.0.0",
        "checks": {}
    }
    
    # Check database connectivity
    try:
        db = get_database()
        if db is not None:
            # Try to get jobs count
            jobs_count = await JobService.get_jobs_count()
            health_status["checks"]["database"] = {
                "status": "healthy",
                "jobs_available": jobs_count
            }
        else:
            health_status["checks"]["database"] = {
                "status": "unhealthy",
                "error": "Database connection not available"
            }
            health_status["status"] = "unhealthy"
    except Exception as e:
        health_status["checks"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "unhealthy"
    
    # Check Redis connectivity
    try:
        redis_healthy = await redis_health_check()
        health_status["checks"]["redis"] = {
            "status": "healthy" if redis_healthy else "unhealthy",
            "caching_enabled": redis_healthy
        }
        # Don't mark service as unhealthy if only Redis fails
        if not redis_healthy:
            logger.warning("Redis unavailable - caching disabled but service functional")
    except Exception as e:
        health_status["checks"]["redis"] = {
            "status": "unhealthy", 
            "error": str(e),
            "caching_enabled": False
        }
        logger.warning(f"Redis health check failed: {e}")
    
    # Check embedding model
    try:
        from app.services.ml.embeddings import get_model
        model = get_model()
        if model is not None:
            health_status["checks"]["embedding_model"] = {
                "status": "healthy",
                "model_name": "sentence-transformers/all-MiniLM-L6-v2"
            }
        else:
            health_status["checks"]["embedding_model"] = {
                "status": "unhealthy",
                "error": "Embedding model not loaded"
            }
            health_status["status"] = "unhealthy"
    except Exception as e:
        health_status["checks"]["embedding_model"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "unhealthy"
    
    if health_status["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health_status)
    
    return health_status