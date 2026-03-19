# app/services/embedding_service.py

import numpy as np
from typing import List, Dict, Optional
from app.services.ml.embeddings import embed_text, get_model
from app.core.database import get_database
from app.core.redis_config import CacheService
import logging
import json

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for managing job embeddings"""
    
    @staticmethod
    async def precompute_all_job_embeddings():
        """
        Pre-compute embeddings for all OPEN jobs and store in MongoDB
        This should be run once or when jobs are updated
        """
        try:
            db = get_database()
            companies_collection = db["companies"]
            
            # Get all companies
            companies = await companies_collection.find({}).to_list(length=None)
            
            total_jobs = 0
            updated_jobs = 0
            
            logger.info("Starting job embeddings pre-computation...")
            
            for company in companies:
                company_id = company.get("_id")
                company_name = company.get("name", "Unknown")
                jobs = company.get("jobs", [])
                
                updated_jobs_in_company = []
                
                for job in jobs:
                    # Only process OPEN jobs
                    if job.get("status") != "OPEN":
                        updated_jobs_in_company.append(job)
                        continue
                    
                    total_jobs += 1
                    
                    # Check if embedding already exists
                    if "embedding" in job and job["embedding"]:
                        logger.debug(f"Job {job.get('id')} already has embedding")
                        updated_jobs_in_company.append(job)
                        continue
                    
                    # Generate job text for embedding
                    job_text = EmbeddingService._generate_job_text(job, company_name)
                    
                    # Compute embedding
                    try:
                        embedding = embed_text(job_text)
                        
                        # Convert numpy array to list for MongoDB storage
                        if isinstance(embedding, np.ndarray):
                            embedding = embedding.tolist()
                        
                        # Add embedding to job
                        job["embedding"] = embedding
                        job["embedding_version"] = "v1"  # Track embedding version
                        
                        updated_jobs += 1
                        logger.info(f"✅ Computed embedding for job: {job.get('title')} at {company_name}")
                        
                    except Exception as e:
                        logger.error(f"Failed to compute embedding for job {job.get('id')}: {e}")
                    
                    updated_jobs_in_company.append(job)
                
                # Update company with new job embeddings
                if updated_jobs_in_company:
                    await companies_collection.update_one(
                        {"_id": company_id},
                        {"$set": {"jobs": updated_jobs_in_company}}
                    )
            
            logger.info(f"✅ Pre-computed embeddings for {updated_jobs}/{total_jobs} jobs")
            
            # Clear jobs cache to force reload with embeddings
            await CacheService.clear_jobs_cache()
            
            return {
                "success": True,
                "total_jobs": total_jobs,
                "updated_jobs": updated_jobs,
                "message": f"Pre-computed embeddings for {updated_jobs} jobs"
            }
            
        except Exception as e:
            logger.error(f"Failed to pre-compute embeddings: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def _generate_job_text(job: dict, company_name: str) -> str:
        """Generate text representation of job for embedding"""
        parts = [
            job.get("title", ""),
            company_name,
            job.get("description", ""),
        ]
        
        # Add skills
        skills = job.get("skills", [])
        if isinstance(skills, list):
            parts.append(", ".join(skills))
        elif isinstance(skills, str):
            parts.append(skills)
        
        # Add requirements
        requirements = job.get("requirements", [])
        if isinstance(requirements, list):
            parts.append(", ".join(requirements))
        elif isinstance(requirements, str):
            parts.append(requirements)
        
        # Add location
        if job.get("location"):
            parts.append(job.get("location"))
        
        return "\n".join(part for part in parts if part)
    
    @staticmethod
    async def get_job_embedding(job_id: str) -> Optional[List[float]]:
        """Get pre-computed embedding for a specific job"""
        try:
            db = get_database()
            companies_collection = db["companies"]
            
            # Find job with embedding
            company = await companies_collection.find_one(
                {"jobs.id": job_id},
                {"jobs.$": 1}
            )
            
            if company and "jobs" in company and len(company["jobs"]) > 0:
                job = company["jobs"][0]
                return job.get("embedding")
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get job embedding: {e}")
            return None
    
    @staticmethod
    async def check_embeddings_status():
        """Check how many jobs have pre-computed embeddings"""
        try:
            db = get_database()
            companies_collection = db["companies"]
            
            companies = await companies_collection.find({}).to_list(length=None)
            
            total_jobs = 0
            jobs_with_embeddings = 0
            
            for company in companies:
                jobs = company.get("jobs", [])
                for job in jobs:
                    if job.get("status") == "OPEN":
                        total_jobs += 1
                        if "embedding" in job and job["embedding"]:
                            jobs_with_embeddings += 1
            
            percentage = (jobs_with_embeddings / total_jobs * 100) if total_jobs > 0 else 0
            
            return {
                "total_jobs": total_jobs,
                "jobs_with_embeddings": jobs_with_embeddings,
                "jobs_without_embeddings": total_jobs - jobs_with_embeddings,
                "percentage": round(percentage, 2)
            }
            
        except Exception as e:
            logger.error(f"Failed to check embeddings status: {e}")
            return None