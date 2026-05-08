# app/api/v1/endpoints/match.py

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import Any, List, Optional
from fastapi.responses import JSONResponse
from app.schemas.cv import CVJsonInput
from app.schemas.match import MatchResponse, MatchRequest
from app.services.cv_service import CVService
from app.services.matching_service import MatchingService
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/cv-file", response_model=MatchResponse)
async def match_cv_file(
    file: UploadFile = File(..., description="CV file (PDF format)"),
    top_k: int = Query(default=5, ge=1, le=50, description="Number of top matches to return")
):
    """
    Match uploaded CV file to jobs
    
    - **file**: CV file in PDF format
    - **top_k**: Number of top matches to return (1-50)
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400, 
                detail="Only PDF files are supported"
            )
        
        # Process CV file
        cv_data = await CVService.process_cv_file(file)
        
        # Prepare match parameters
        match_params = MatchRequest(top_k=top_k)
        
        # Perform matching
        result = await MatchingService.match_cv_to_jobs(cv_data, match_params)
        
        return result
        
    except ValueError as e:
        logger.error(f"CV processing error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Matching error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during matching")

@router.post("/cv-json", response_model=MatchResponse)
async def match_cv_json(
    cv_data: CVJsonInput,
    top_k: int = Query(default=5, ge=1, le=50, description="Number of top matches to return"),
    job_ids: Optional[List[str]] = Query(None, description="Specific job IDs to match against")
):
    """
    Match CV data (JSON format) to jobs
    
    - **cv_data**: CV information in JSON format
    - **top_k**: Number of top matches to return (1-50)
    - **job_ids**: Optional list of job IDs to filter the search
    """
    try:
        # Process CV JSON
        parsed_cv = CVService.process_cv_json(cv_data)
        
        # Prepare match parameters
        match_params = MatchRequest(top_k=top_k, job_ids=job_ids)
        
        # Perform matching
        result = await MatchingService.match_cv_to_jobs(parsed_cv, match_params)
        
        return result
        
    except ValueError as e:
        logger.error(f"CV processing error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Matching error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during matching")