# app/models/job.py

from sqlalchemy import Column, String, Text, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from .base import BaseModel
from sqlalchemy.dialects.postgresql import UUID

class CompanyModel(BaseModel):
    __tablename__ = 'companies'
    
    name = Column(String(200), nullable=False)
    slug = Column(String(255), unique=True)
    logo_url = Column(String(255))
    website = Column(String(255))
    address = Column(String(255))
    description = Column(Text)
    industry = Column(String(100))
    sub_industries = Column(JSON)
    size = Column(String(50))
    locations = Column(JSON)
    email = Column(String(100))
    support_email = Column(String(100))
    phone = Column(String(20))
    
    # Relationships
    jobs = relationship('JobModel', back_populates='company')

class JobModel(BaseModel):
    __tablename__ = 'jobs'
    
    title = Column(String(200), nullable=False)
    slug = Column(String(255), unique=True)
    company_id = Column(UUID(as_uuid=True), ForeignKey('companies.id'), nullable=False)
    location = Column(String(150))
    other_locations = Column(JSON)
    work_arrangement = Column(String(50))
    job_type = Column(String(20))
    salary_display = Column(String(100))
    salary_min = Column(Integer)
    salary_max = Column(Integer)
    skills = Column(JSON)
    requirements = Column(Text)
    description = Column(Text)
    benefits = Column(Text)
    job_level = Column(String(100))
    years_of_experience = Column(Integer)
    status = Column(String(20))
    source_id = Column(UUID(as_uuid=True), ForeignKey('data_sources.id'), nullable=True)
    url_source = Column(String(255))
    posted_at = Column(DateTime)
    expired_at = Column(DateTime)
    
    # Relationships
    company = relationship('CompanyModel', back_populates='jobs')
