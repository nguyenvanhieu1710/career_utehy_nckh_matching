# app/services/job_service.py

from app.core.config import settings
from typing import List, Optional
from app.core.database import get_postgres_session
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
            
            # Cache miss - fetch from PostgreSQL only (MongoDB disabled)
            logger.info("Cache miss - fetching jobs from PostgreSQL only")
            pg_jobs = await JobService._fetch_jobs_from_postgres()
            
            all_jobs = pg_jobs
            
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
                        url_source=job.url_source,
                        image_url=job.company.logo_url if job.company else None
                    ))
                    if str(job.id) in remaining_ids:
                        remaining_ids.remove(str(job.id))
            
            # 3. If still missing, check Mongo (Optional/Secondary)
            if remaining_ids:
                # Fallback: try to retrieve job metadata from Milvus (if jobs were synced there)
                try:
                    from app.core.database import get_milvus_client
                    from app.core.config import settings

                    milvus_client = get_milvus_client()
                    if milvus_client:
                        for rid in list(remaining_ids):
                            try:
                                res = milvus_client.query(
                                    collection_name=settings.MILVUS_COLLECTION,
                                    filter=f"job_id == '{rid}'",
                                    output_fields=["job_id", "job_title", "company_name", "location_city", "skills", "exp_min", "title_vec", "tech_vec", "mota_vec"]
                                )
                                if res:
                                    row = res[0]
                                    skills_field = row.get("skills", "")
                                    if isinstance(skills_field, str):
                                        skills_list = [s.strip() for s in skills_field.split(",") if s.strip()]
                                    else:
                                        skills_list = skills_field or []

                                    found_jobs.append(JobSchema(
                                        id=str(row.get("job_id")),
                                        title=row.get("job_title", ""),
                                        company=row.get("company_name", "Unknown Company"),
                                        description=None,
                                        skills=skills_list,
                                        location=row.get("location_city"),
                                        location_city=row.get("location_city"),
                                        requirements=[],
                                        exp_min=float(row.get("exp_min") or 0),
                                        exp_max=99.0,
                                        salary=None,
                                        status="OPEN",
                                        url_source=None,
                                        image_url=None,
                                        title_vec=row.get("title_vec"),
                                        tech_vec=row.get("tech_vec"),
                                        mota_vec=row.get("mota_vec")
                                    ))
                                    if str(row.get("job_id")) in remaining_ids:
                                        remaining_ids.remove(str(row.get("job_id")))
                            except Exception:
                                continue
                except Exception as e:
                    logger.warning(f"Milvus fallback failed: {e}")
                
            return found_jobs
        except Exception as e:
            logger.error(f"Failed to fetch jobs by IDs: {str(e)}")
            return []
    
    @staticmethod
    async def _fetch_jobs_from_mongo() -> List[JobSchema]:
        """
        Fetch jobs from MongoDB (internal method)
        """
        logger.warning("_fetch_jobs_from_mongo() called, but MongoDB is disabled. Returning empty job list.")
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
                        url_source=job.url_source,
                        image_url=job.company.logo_url if job.company else None
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