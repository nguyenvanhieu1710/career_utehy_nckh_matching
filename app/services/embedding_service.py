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
    def _ensure_collection_exists(client):
        """Ensure Milvus collection exists with correct schema for BGE-M3 (1024 dims)"""
        from app.core.config import settings
        from pymilvus import DataType
        
        if client.has_collection(collection_name=settings.MILVUS_COLLECTION):
            return True
            
        logger.info(f"Creating collection {settings.MILVUS_COLLECTION} on Milvus Cloud...")
        
        schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
        
        # Add fields
        schema.add_field(field_name="job_id", datatype=DataType.VARCHAR, max_length=100, is_primary=True)
        schema.add_field(field_name="job_title", datatype=DataType.VARCHAR, max_length=500)
        schema.add_field(field_name="company_name", datatype=DataType.VARCHAR, max_length=500)
        schema.add_field(field_name="location_city", datatype=DataType.VARCHAR, max_length=200)
        schema.add_field(field_name="exp_min", datatype=DataType.FLOAT)
        schema.add_field(field_name="skills", datatype=DataType.VARCHAR, max_length=2000) # Added skills
        
        # Vector fields (1024 dimensions for bge-m3)
        dim = 1024 
        schema.add_field(field_name="title_vec", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="tech_vec", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="mota_vec", datatype=DataType.FLOAT_VECTOR, dim=dim)
        
        # Index params
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="title_vec", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(field_name="tech_vec", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(field_name="mota_vec", index_type="AUTOINDEX", metric_type="COSINE")
        
        client.create_collection(
            collection_name=settings.MILVUS_COLLECTION,
            schema=schema,
            index_params=index_params
        )
        logger.info(f"✅ Collection {settings.MILVUS_COLLECTION} created successfully")
        return True

    @staticmethod
    async def precompute_all_job_embeddings():
        """
        Pre-compute embeddings for all jobs and store in Milvus Cloud
        """
        try:
            from app.core.config import settings
            from app.services.job_service import JobService
            from app.core.database import get_milvus_client
            
            # 1. Fetch all jobs
            all_jobs = await JobService.fetch_all_jobs()
            total_jobs = len(all_jobs)
            
            if total_jobs == 0:
                return {"success": True, "total_jobs": 0, "updated_jobs": 0, "message": "No jobs found to process"}

            # 2. Initialize and ensure collection
            milvus_client = get_milvus_client()
            EmbeddingService._ensure_collection_exists(milvus_client)
            
            # 3. Process jobs
            updated_jobs = 0
            logger.info(f"Starting pre-computation for {total_jobs} jobs...")
            
            for job in all_jobs:
                try:
                    # Generate text and embed separately for Multi-Vector Search
                    title_text = job.title or ""
                    tech_text = ", ".join(job.skills) if isinstance(job.skills, list) else str(job.skills or "")
                    mota_text = f"{job.description}\n{', '.join(job.requirements)}" if hasattr(job, 'requirements') else str(job.description or "")

                    title_vector = embed_text(title_text).tolist()
                    tech_vector = embed_text(tech_text).tolist()
                    mota_vector = embed_text(mota_text).tolist()
                    
                    # Upsert into Milvus
                    milvus_client.upsert(
                        collection_name=settings.MILVUS_COLLECTION,
                        data=[{
                            "job_id": job.id,
                            "job_title": job.title,
                            "company_name": job.company,
                            "location_city": job.location,
                            "skills": tech_text,
                            "title_vec": title_vector,
                            "tech_vec": tech_vector,
                            "mota_vec": mota_vector,
                            "exp_min": job.exp_min
                        }]
                    )
                    updated_jobs += 1
                except Exception as je:
                    logger.error(f"Error processing job {job.id}: {je}")

            # 4. Clear cache
            await CacheService.clear_jobs_cache()
            
            return {
                "success": True,
                "total_jobs": total_jobs,
                "updated_jobs": updated_jobs,
                "message": f"Successfully synced {updated_jobs} jobs to Milvus Cloud"
            }
        except Exception as e:
            logger.error(f"Failed to pre-compute embeddings: {e}")
            return {"success": False, "error": str(e)}
    
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
        """Check how many jobs have pre-computed embeddings in both DBs and Milvus"""
        try:
            from app.core.config import settings
            from app.services.job_service import JobService
            from app.core.database import get_milvus_client
            
            # Fetch all jobs from both sources
            all_jobs = await JobService.fetch_all_jobs()
            total_jobs = len(all_jobs)
            
            # Check Milvus for actual stored vectors
            jobs_in_milvus = 0
            try:
                milvus_client = get_milvus_client()
                if milvus_client.has_collection(settings.MILVUS_COLLECTION):
                    milvus_res = milvus_client.query(
                        collection_name=settings.MILVUS_COLLECTION,
                        filter="",
                        output_fields=["count(*)"]
                    )
                    jobs_in_milvus = milvus_res[0]["count(*)"] if milvus_res else 0
            except Exception as me:
                logger.warning(f"Milvus collection not ready: {me}")
            
            percentage = (jobs_in_milvus / total_jobs * 100) if total_jobs > 0 else 0
            
            return {
                "total_jobs": total_jobs,
                "jobs_with_embeddings": jobs_in_milvus,
                "jobs_without_embeddings": max(0, total_jobs - jobs_in_milvus),
                "percentage": round(percentage, 2),
                "milvus_sync": "synced" if total_jobs > 0 and jobs_in_milvus >= total_jobs else "pending"
            }
            
        except Exception as e:
            logger.error(f"Failed to check embeddings status: {e}")
            return None