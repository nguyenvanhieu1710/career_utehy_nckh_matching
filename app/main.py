import os
import warnings
import logging

# Suppress warnings from third-party libraries
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="pypdf")

# Set logging levels for noisy libraries
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.api.v1.router import api_router
from app.core.config import settings
# MongoDB is disabled in this deployment to save resources
# from app.core.database import connect_to_mongo, close_mongo_connection
from app.core.redis_config import redis_manager
import logging

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Career UTEHY NCKH - CV Job Matching Service...")
    
    # Connect to MongoDB (disabled)
    logger.warning("MongoDB startup skipped because the matching service is configured to use PostgreSQL only.")
    
    # Connect to PostgreSQL
    try:
        from app.core.database import connect_to_postgres
        await connect_to_postgres()
        logger.info("PostgreSQL connection established")
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {str(e)}")

    # Connect to Milvus
    try:
        from app.core.database import connect_to_milvus
        connect_to_milvus()
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {str(e)}")
    
    # Connect to Redis
    try:
        await redis_manager.get_client()
        if redis_manager.is_connected():
            logger.info("Redis connection established")
        else:
            logger.warning("Redis connection failed - caching disabled")
    except Exception as e:
        logger.warning(f"Redis connection failed: {str(e)} - caching disabled")
    
    # Load embedding model
    try:
        from app.services.ml.embeddings import get_model
        model = get_model()
        logger.info("Embedding model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {str(e)}")
    
    logger.info("Startup complete")
    
    # Log Swagger UI URL
    base_url = "http://127.0.0.1" if settings.APP_HOST == "0.0.0.0" else f"http://{settings.APP_HOST}"
    logger.info(f"Swagger UI is available at: {base_url}:{settings.APP_PORT}/docs")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    logger.warning("MongoDB shutdown skipped because it was disabled at startup.")
    await redis_manager.close()

# Create FastAPI app
app = FastAPI(
    title="Career UTEHY NCKH - CV Job Matching Service",
    description="AI-powered CV to Job matching service for UTEHY Career Platform",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Career UTEHY NCKH - CV Job Matching Service",
        "version": "1.0.0",
        "service": "career_utehy_nckh_matching",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.APP_RELOAD
    )
