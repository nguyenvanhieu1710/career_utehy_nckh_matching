# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.api.v1.router import api_router
from app.config import settings
from app.core.database import connect_to_mongo, close_mongo_connection
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
    
    # Connect to MongoDB
    try:
        await connect_to_mongo()
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        # Continue without database for now
    
    # Load embedding model
    try:
        from app.embedding import get_model
        model = get_model()
        logger.info("Embedding model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {str(e)}")
    
    logger.info("Startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    await close_mongo_connection()

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

# Legacy endpoint for backward compatibility
@app.post("/match")
async def legacy_match_endpoint():
    return {
        "message": "This endpoint is deprecated. Please use /api/v1/match/cv-file or /api/v1/match/cv-json",
        "new_endpoints": {
            "cv_file": "/api/v1/match/cv-file",
            "cv_json": "/api/v1/match/cv-json"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_DEBUG
    )
