# app/services/job_service.py

from app.core.config import settings
from typing import List, Optional
from app.core.database import get_database, get_postgres_session
from app.schemas.job import JobSchema
from app.models.job import JobModel, CompanyModel
from sqlalchemy.future import select
from sqlalchemy.orm import joinedload
from app.core.redis_config import CacheService
import logging

logger = logging.getLogger(__name__)

class JobService:
    """Service for fetching jobs from MongoDB with Redis caching"""
    
    @staticmethod
    async def fetch_all_jobs() -> List[JobSchema]:
        """
        Fetch all OPEN jobs from MongoDB with Redis caching
        Returns list of JobSchema objects
        """
        try:
            # Try to get from cache first
            cached_jobs = await CacheService.get_jobs_cache()
            if cached_jobs:
                # Convert cached data back to JobSchema objects
                return [JobSchema(**job_data) for job_data in cached_jobs]
            
            # Cache miss - fetch from both DBs
            logger.info("Cache miss - fetching jobs from MongoDB & PostgreSQL")
            mongo_jobs = await JobService._fetch_jobs_from_mongo()
            pg_jobs = await JobService._fetch_jobs_from_postgres()
            
            all_jobs = mongo_jobs + pg_jobs
            
            # Cache the results in background to not block the response
            if all_jobs:
                import asyncio
                asyncio.create_task(CacheService.set_jobs_cache(all_jobs))
            
            return all_jobs
            
        except Exception as e:
            logger.error(f"Failed to fetch jobs: {str(e)}")
            return []

    @staticmethod
    async def fetch_jobs_by_ids(job_ids: List[str]) -> List[JobSchema]:
        """
        Fetch specific jobs by their IDs from both DBs.
        Optimized for incremental sync.
        """
        if not job_ids:
            return []
            
        try:
            # Bypass cache for ID-specific lookups to ensure maximum speed and reliability
            # Fetching specific IDs from PostgreSQL is extremely fast (<10ms)
            session_factory = await get_postgres_session()
            found_jobs = []
            remaining_ids = set(job_ids)
                
            # 2. Fetch missing from PG (New jobs are usually here)
            session_factory = await get_postgres_session()
            async with session_factory as session:
                query = select(JobModel).where(JobModel.id.in_(remaining_ids)).options(joinedload(JobModel.company))
                result = await session.execute(query)
                pg_jobs = result.scalars().all()
                
                for job in pg_jobs:
                    found_jobs.append(JobSchema(
                        id=str(job.id),
                        title=job.title,
                        company=job.company.name if job.company else "Unknown Company",
                        description=job.description,
                        skills=job.skills or [],
                        location=job.location,
                        location_city=job.location,
                        requirements=[job.requirements] if job.requirements else [],
                        exp_min=float(job.years_of_experience or 0),
                        exp_max=float(job.years_of_experience or 99),
                        salary=job.salary_display,
                        status=job.status,
                        url_source=job.url_source
                    ))
                    if str(job.id) in remaining_ids:
                        remaining_ids.remove(str(job.id))
            
            # 3. If still missing, check Mongo (Optional/Secondary)
            if remaining_ids:
                # Add mongo logic if needed, but usually new jobs are in PG
                pass
                
            return found_jobs
        except Exception as e:
            logger.error(f"Failed to fetch jobs by IDs: {str(e)}")
            return []
    
    @staticmethod
    async def _fetch_jobs_from_mongo() -> List[JobSchema]:
        """
        Fetch jobs from MongoDB (internal method)
        """
        try:
            db = get_database()
            companies_collection = db[settings.MONGODB_COLLECTION_COMPANIES]
            
            # Get all companies with jobs
            companies = await companies_collection.find({}).to_list(length=None)
            
            job_list = []
            
            for company in companies:
                company_name = company.get("name", "Unknown Company")
                jobs = company.get("jobs", [])
                
                # 4. Apply Heuristic Scoring
                # Use weights from configuration
                weights = {
                    "title": settings.WEIGHT_TITLE,
                    "tech": settings.WEIGHT_TECH,
                    "mota": settings.WEIGHT_MOTA,
                    "loc": settings.WEIGHT_LOCATION,
                    "exp": settings.WEIGHT_EXPERIENCE
                }
                
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
                        status=job.get("status", ""),
                        url_source=job.get("url_source", ""),
                        embedding=job.get("embedding")  # Include pre-computed embedding
                    )
                    
                    job_list.append(job_schema)
            
            logger.info(f"Fetched {len(job_list)} OPEN jobs from MongoDB")
            return job_list
            
        except Exception as e:
            logger.error(f"Failed to fetch jobs from MongoDB: {str(e)}")
            return []

    @staticmethod
    async def _fetch_jobs_from_postgres() -> List[JobSchema]:
        """
        Fetch jobs from PostgreSQL (internal method)
        """
        try:
            session_factory = await get_postgres_session()
            async with session_factory as session:
                query = select(JobModel).options(joinedload(JobModel.company))
                result = await session.execute(query)
                jobs = result.scalars().all()
                
                job_list = []
                for job in jobs:
                    job_list.append(JobSchema(
                        id=str(job.id),
                        title=job.title,
                        company=job.company.name if job.company else "Unknown Company",
                        description=job.description,
                        skills=job.skills or [],
                        location=job.location,
                        location_city=job.location, # Fallback to location
                        requirements=[job.requirements] if job.requirements else [],
                        exp_min=float(job.years_of_experience or 0),
                        exp_max=float(job.years_of_experience or 99),
                        salary=job.salary_display,
                        status=job.status,
                        url_source=job.url_source
                    ))
                
                logger.info(f"Fetched {len(job_list)} OPEN jobs from PostgreSQL")
                return job_list
        except Exception as e:
            logger.error(f"Failed to fetch jobs from PostgreSQL: {str(e)}")
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