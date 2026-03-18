# app/services/job_service.py

from typing import List, Optional
from app.core.database import get_database
from app.schemas.job import JobSchema
import logging

logger = logging.getLogger(__name__)

class JobService:
    """Service for fetching jobs from MongoDB"""
    
    @staticmethod
    async def fetch_all_jobs() -> List[JobSchema]:
        """
        Fetch all OPEN jobs from MongoDB companies collection
        Returns list of JobSchema objects
        """
        try:
            db = get_database()
            companies_collection = db["companies"]
            
            # Get all companies with jobs
            companies = await companies_collection.find({}).to_list(length=None)
            
            job_list = []
            
            for company in companies:
                company_name = company.get("name", "Unknown Company")
                jobs = company.get("jobs", [])
                
                for job in jobs:
                    # Only include OPEN jobs
                    if job.get("status") != "OPEN":
                        continue
                    
                    # Parse skills and requirements
                    skills = job.get("skills", [])
                    if isinstance(skills, str):
                        skills = [s.strip() for s in skills.split(",") if s.strip()]
                    elif not isinstance(skills, list):
                        skills = []
                    
                    requirements = job.get("requirements", [])
                    if isinstance(requirements, str):
                        requirements = [r.strip() for r in requirements.split(",") if r.strip()]
                    elif not isinstance(requirements, list):
                        requirements = []
                    
                    job_schema = JobSchema(
                        id=job.get("id", ""),
                        title=job.get("title", ""),
                        company=company_name,
                        description=job.get("description", ""),
                        skills=skills,
                        location=job.get("location", ""),
                        requirements=requirements,
                        salary=job.get("salary", ""),
                        status=job.get("status", "")
                    )
                    
                    job_list.append(job_schema)
            
            logger.info(f"Fetched {len(job_list)} OPEN jobs from MongoDB")
            return job_list
            
        except Exception as e:
            logger.error(f"Failed to fetch jobs from MongoDB: {str(e)}")
            return []
    
    @staticmethod
    async def get_job_by_id(job_id: str) -> Optional[JobSchema]:
        """Get a specific job by ID"""
        try:
            jobs = await JobService.fetch_all_jobs()
            for job in jobs:
                if job.id == job_id:
                    return job
            return None
        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {str(e)}")
            return None
    
    @staticmethod
    async def get_jobs_count() -> int:
        """Get total count of OPEN jobs"""
        try:
            jobs = await JobService.fetch_all_jobs()
            return len(jobs)
        except Exception as e:
            logger.error(f"Failed to get jobs count: {str(e)}")
            return 0