# app/api/v1/endpoints/health.py

from fastapi import APIRouter, HTTPException
from app.core.database import get_database
from app.services.job_service import JobService
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/")
async def health_check():
    """
    Basic health check endpoint
    """
    return {
        "status": "healthy",
        "service": "Career UTEHY NCKH - CV Job Matching Service",
        "version": "1.0.0"
    }

@router.get("/detailed")
async def detailed_health_check():
    """
    Detailed health check including database connectivity
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
    
    # Check embedding model
    try:
        from app.embedding import get_model
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