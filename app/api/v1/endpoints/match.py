# app/api/v1/endpoints/match.py

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
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
    top_k: int = Query(default=5, ge=1, le=50, description="Number of top matches to return"),
    min_score: float = Query(default=0.0, ge=0.0, le=1.0, description="Minimum match score threshold"),
    filter_location: str = Query(default=None, description="Filter by location"),
    filter_skills: str = Query(default=None, description="Filter by skills (comma-separated)")
):
    """
    Match uploaded CV file to jobs
    
    - **file**: CV file in PDF format
    - **top_k**: Number of top matches to return (1-50)
    - **min_score**: Minimum match score threshold (0.0-1.0)
    - **filter_location**: Filter jobs by location
    - **filter_skills**: Filter jobs by skills (comma-separated)
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
        filter_skills_list = None
        if filter_skills:
            filter_skills_list = [skill.strip() for skill in filter_skills.split(",")]
        
        match_params = MatchRequest(
            top_k=top_k,
            min_score=min_score,
            filter_location=filter_location,
            filter_skills=filter_skills_list
        )
        
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
    min_score: float = Query(default=0.0, ge=0.0, le=1.0, description="Minimum match score threshold"),
    filter_location: str = Query(default=None, description="Filter by location"),
    filter_skills: str = Query(default=None, description="Filter by skills (comma-separated)")
):
    """
    Match CV data (JSON format) to jobs
    
    - **cv_data**: CV information in JSON format
    - **top_k**: Number of top matches to return (1-50)
    - **min_score**: Minimum match score threshold (0.0-1.0)
    - **filter_location**: Filter jobs by location
    - **filter_skills**: Filter jobs by skills (comma-separated)
    """
    try:
        # Process CV JSON
        parsed_cv = CVService.process_cv_json(cv_data)
        
        # Prepare match parameters
        filter_skills_list = None
        if filter_skills:
            filter_skills_list = [skill.strip() for skill in filter_skills.split(",")]
        
        match_params = MatchRequest(
            top_k=top_k,
            min_score=min_score,
            filter_location=filter_location,
            filter_skills=filter_skills_list
        )
        
        # Perform matching
        result = await MatchingService.match_cv_to_jobs(parsed_cv, match_params)
        
        return result
        
    except ValueError as e:
        logger.error(f"CV processing error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Matching error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during matching")