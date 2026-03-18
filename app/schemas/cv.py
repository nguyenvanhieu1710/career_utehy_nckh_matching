# app/schemas/cv.py

from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List

class CVJsonInput(BaseModel):
    """Schema for CV input in JSON format"""
    name: str = Field(..., description="Full name of the candidate")
    email: Optional[EmailStr] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    skills: Optional[List[str]] = Field(default_factory=list, description="List of skills")
    experience: Optional[str] = Field(None, description="Work experience description")
    education: Optional[str] = Field(None, description="Education background")
    summary: Optional[str] = Field(None, description="Professional summary")
    certifications: Optional[List[str]] = Field(default_factory=list, description="Certifications")
    languages: Optional[List[str]] = Field(default_factory=list, description="Languages spoken")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Nguyen Van A",
                "email": "nguyenvana@example.com",
                "phone": "+84 123 456 789",
                "skills": ["Python", "FastAPI", "Docker", "MongoDB"],
                "experience": "3 years as Backend Developer at Tech Corp. Built RESTful APIs and microservices.",
                "education": "Bachelor of Computer Science, HUST, 2020",
                "summary": "Passionate backend developer with expertise in Python and cloud technologies",
                "certifications": ["AWS Certified Developer"],
                "languages": ["Vietnamese", "English"]
            }
        }

class CVParsedData(BaseModel):
    """Internal representation of parsed CV data"""
    raw_text: str
    name: Optional[str] = None
    email: Optional[str] = None
    skills: List[str] = []
    experience: Optional[str] = None
    education: Optional[str] = None
