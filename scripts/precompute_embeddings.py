#!/usr/bin/env python3
"""
Script to pre-compute embeddings for all jobs
Run this once or when jobs are updated
"""

import asyncio
from app.core.database import connect_to_mongo, close_mongo_connection
from app.services.embedding_service import EmbeddingService
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

async def main():
    """Pre-compute embeddings for all jobs"""
    print("🚀 Starting Job Embeddings Pre-computation")
    print("=" * 50)
    
    try:
        # Connect to MongoDB
        print("📡 Connecting to MongoDB...")
        await connect_to_mongo()
        print("✅ Connected to MongoDB")
        
        # Check current status
        print("\n📊 Checking current embeddings status...")
        status = await EmbeddingService.check_embeddings_status()
        
        if status:
            print(f"   Total OPEN jobs: {status['total_jobs']}")
            print(f"   Jobs with embeddings: {status['jobs_with_embeddings']}")
            print(f"   Jobs without embeddings: {status['jobs_without_embeddings']}")
            print(f"   Coverage: {status['percentage']}%")
        
        # Pre-compute embeddings
        print("\n🔄 Pre-computing embeddings...")
        print("   This may take a few minutes depending on number of jobs...")
        
        result = await EmbeddingService.precompute_all_job_embeddings()
        
        if result["success"]:
            print(f"\n✅ Success!")
            print(f"   Total jobs processed: {result['total_jobs']}")
            print(f"   Embeddings computed: {result['updated_jobs']}")
            print(f"   Message: {result['message']}")
        else:
            print(f"\n❌ Failed: {result.get('error')}")
        
        # Check final status
        print("\n📊 Final embeddings status...")
        final_status = await EmbeddingService.check_embeddings_status()
        
        if final_status:
            print(f"   Total OPEN jobs: {final_status['total_jobs']}")
            print(f"   Jobs with embeddings: {final_status['jobs_with_embeddings']}")
            print(f"   Coverage: {final_status['percentage']}%")
        
        # Close connection
        await close_mongo_connection()
        
        print("\n" + "=" * 50)
        print("🎉 Pre-computation completed!")
        
        if final_status and final_status['percentage'] == 100:
            print("✅ All jobs now have pre-computed embeddings")
            print("🚀 Matching performance will be significantly faster!")
        
    except Exception as e:
        logger.error(f"Pre-computation failed: {e}")
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())