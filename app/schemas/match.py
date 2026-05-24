# app/schemas/match.py

from pydantic import BaseModel, Field
from typing import List, Optional

class MatchResult(BaseModel):
    """Single job match result with recommendations"""
    job_id: str = Field(..., description="Job ID")
    job_title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    compatibility_score: float = Field(..., description="Compatibility percentage (0-100)", ge=0, le=100)
    matched_skills: List[str] = Field(default_factory=list, description="Skills you already have")
    missing_skills: List[str] = Field(default_factory=list, description="Skills to improve/learn")
    skill_improvement_suggestions: List[str] = Field(default_factory=list, description="Specific recommendations")
    match_explanation: str = Field(..., description="Why this job matches and recommendations")
    location: Optional[str] = Field(None, description="Job location")
    salary: Optional[str] = Field(None, description="Salary range")
    url_source: Optional[str] = Field(None, description="Link to original job post")
    image_url: Optional[str] = Field(None, description="Image or logo URL for the job/company")
    scores: dict = Field(default_factory=dict, description="Detailed sub-scores for the match")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_123",
                "job_title": "Backend Engineer",
                "company": "Tech Corp",
                "compatibility_score": 87.5,
                "matched_skills": ["Python", "FastAPI", "Docker"],
                "missing_skills": ["Kubernetes", "AWS"],
                "skill_improvement_suggestions": [
                    "Get AWS certification and practice with free tier",
                    "Learn Kubernetes through online labs and minikube",
                    "Focus on API design and system architecture"
                ],
                "match_explanation": "Good match (87.5% compatibility). You have 3 relevant skills. Consider developing 2 additional skills to improve your chances. Good fit, consider applying.",
                "location": "Hanoi, Vietnam",
                "salary": "$1000-$2000"
            }
        }

class MatchResponse(BaseModel):
    """Response for CV-Job matching"""
    success: bool = Field(..., description="Whether matching was successful")
    matches: List[MatchResult] = Field(default_factory=list, description="List of matched jobs")
    total_jobs_analyzed: int = Field(..., description="Total number of jobs analyzed")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    message: Optional[str] = Field(None, description="Additional message or error")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "matches": [
                    {
                        "job_id": "job_123",
                        "job_title": "Backend Engineer",
                        "company": "Tech Corp",
                        "compatibility_score": 87.5,
                        "matched_skills": ["Python", "FastAPI"],
                        "missing_skills": ["Kubernetes"],
                        "skill_improvement_suggestions": ["Learn Kubernetes through online labs"],
                        "match_explanation": "Good match (87.5% compatibility). Good fit, consider applying.",
                        "location": "Hanoi",
                        "salary": "$1000-$2000"
                    }
                ],
                "total_jobs_analyzed": 150,
                "processing_time_ms": 234.5,
                "message": "Successfully matched CV to jobs"
            }
        }

class MatchRequest(BaseModel):
    """Request parameters for matching"""
    top_k: int = Field(default=5, description="Number of top matches to return", ge=1, le=50)
    job_ids: Optional[List[str]] = Field(None, description="Specific job IDs to match against")
