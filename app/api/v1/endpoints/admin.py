# app/api/v1/endpoints/admin.py

from fastapi import APIRouter, HTTPException
from app.services.embedding_service import EmbeddingService
from app.core.redis_config import CacheService
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/precompute-embeddings")
async def precompute_embeddings():
    """
    Pre-compute embeddings for all OPEN jobs
    This endpoint should be called when jobs are updated
    """
    try:
        result = await EmbeddingService.precompute_all_job_embeddings()
        
        if result["success"]:
            return {
                "success": True,
                "total_jobs": result["total_jobs"],
                "updated_jobs": result["updated_jobs"],
                "message": result["message"]
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("error"))
            
    except Exception as e:
        logger.error(f"Failed to pre-compute embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sync-jobs")
async def sync_specific_jobs(data: dict):
    """
    Sync only specific job IDs to Milvus.
    Input: {"job_ids": ["id1", "id2", ...]}
    """
    job_ids = data.get("job_ids", [])
    if not job_ids:
        raise HTTPException(status_code=400, detail="job_ids is required")
        
    try:
        result = await EmbeddingService.sync_specific_jobs(job_ids)
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=500, detail=result.get("error"))
    except Exception as e:
        logger.error(f"Failed to sync specific jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/embeddings-status")
async def get_embeddings_status():
    """
    Check embeddings status for all jobs
    """
    try:
        status = await EmbeddingService.check_embeddings_status()
        
        if status:
            return {
                "success": True,
                **status
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to check embeddings status")
            
    except Exception as e:
        logger.error(f"Failed to check embeddings status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear-cache")
async def clear_cache():
    """
    Clear all caches (jobs cache and match results cache)
    """
    try:
        await CacheService.clear_jobs_cache()
        
        return {
            "success": True,
            "message": "Cache cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))