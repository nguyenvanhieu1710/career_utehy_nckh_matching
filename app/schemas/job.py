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
    location_city: Optional[str] = Field(None, description="City for location matching")
    requirements: List[str] = Field(default_factory=list, description="Job requirements")
    exp_min: float = Field(0.0, description="Minimum years of experience")
    exp_max: float = Field(999.0, description="Maximum years of experience")
    salary: Optional[str] = Field(None, description="Salary range")
    status: Optional[str] = Field(None, description="Job status")
    embedding: Optional[List[float]] = Field(None, description="Legacy embedding")
    title_vec: Optional[List[float]] = Field(None, description="Title embedding")
    tech_vec: Optional[List[float]] = Field(None, description="Tech stack embedding")
    mota_vec: Optional[List[float]] = Field(None, description="Description embedding")
    
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
