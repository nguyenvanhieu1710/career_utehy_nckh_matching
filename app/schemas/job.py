# app/schemas/job.py

from pydantic import BaseModel, Field
from typing import Optional, List

class JobSchema(BaseModel):
    """Schema for Job data from MongoDB"""
    id: str = Field(..., description="Job ID")
    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    description: Optional[str] = Field(None, description="Job description")
    skills: List[str] = Field(default_factory=list, description="Required skills")
    location: Optional[str] = Field(None, description="Job location")
    requirements: List[str] = Field(default_factory=list, description="Job requirements")
    salary: Optional[str] = Field(None, description="Salary range")
    status: Optional[str] = Field(None, description="Job status")
    embedding: Optional[List[float]] = Field(None, description="Pre-computed embedding vector")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "job_123",
                "title": "Backend Engineer",
                "company": "Tech Corp",
                "description": "We are looking for a talented backend engineer...",
                "skills": ["Python", "FastAPI", "Docker"],
                "location": "Hanoi, Vietnam",
                "requirements": ["3+ years experience", "Strong Python skills"],
                "salary": "$1000-$2000",
                "status": "OPEN",
                "embedding": None
            }
        }
