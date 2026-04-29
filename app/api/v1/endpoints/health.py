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
    from app.core.database import mongodb, postgresdb, get_milvus_client
    from app.core.config import settings
    import sqlalchemy as sa
    
    health_status = {
        "status": "healthy",
        "service": "Career UTEHY NCKH - CV Job Matching Service",
        "version": "1.0.0",
        "checks": {}
    }
    
    # 1. Check MongoDB
    try:
        if mongodb.client:
            await mongodb.client.admin.command('ping')
            health_status["checks"]["mongodb"] = {
                "status": "healthy",
                "database": settings.MONGODB_DB_NAME
            }
        else:
            health_status["checks"]["mongodb"] = {"status": "unhealthy", "error": "Not connected"}
            health_status["status"] = "unhealthy"
    except Exception as e:
        health_status["checks"]["mongodb"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "unhealthy"

    # 2. Check PostgreSQL
    try:
        if postgresdb.session_factory:
            async with postgresdb.session_factory() as session:
                await session.execute(sa.text("SELECT 1"))
            health_status["checks"]["postgresql"] = {
                "status": "healthy",
                "database": settings.POSTGRES_DB
            }
        else:
            health_status["checks"]["postgresql"] = {"status": "unhealthy", "error": "Not connected"}
            health_status["status"] = "unhealthy"
    except Exception as e:
        health_status["checks"]["postgresql"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "unhealthy"

    # 3. Check Milvus (Zilliz Cloud)
    try:
        client = get_milvus_client()
        if client:
            # Check connection status
            health_status["checks"]["milvus"] = {
                "status": "healthy",
                "endpoint": settings.MILVUS_URL
            }
        else:
            health_status["checks"]["milvus"] = {"status": "unhealthy", "error": "Client not initialized"}
            health_status["status"] = "unhealthy"
    except Exception as e:
        health_status["checks"]["milvus"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "unhealthy"
    
    # 4. Check Redis
    try:
        redis_healthy = await redis_health_check()
        health_status["checks"]["redis"] = {
            "status": "healthy" if redis_healthy else "unhealthy",
            "caching_enabled": redis_healthy
        }
    except Exception as e:
        health_status["checks"]["redis"] = {"status": "unhealthy", "error": str(e)}
    
    # 5. Check AI Embedding Model
    try:
        from app.services.ml.embeddings import get_model
        model = get_model()
        if model is not None:
            health_status["checks"]["embedding_model"] = {
                "status": "healthy",
                "model_name": settings.EMBEDDING_MODEL_NAME
            }
        else:
            health_status["checks"]["embedding_model"] = {"status": "unhealthy", "error": "Model not loaded"}
            health_status["status"] = "unhealthy"
    except Exception as e:
        health_status["checks"]["embedding_model"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "unhealthy"
    
    if health_status["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health_status)
    
    return health_status