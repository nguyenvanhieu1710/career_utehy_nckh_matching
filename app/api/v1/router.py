# app/api/v1/router.py

from fastapi import APIRouter
from app.api.v1.endpoints import match, health

api_router = APIRouter()

# Include only essential endpoint routers
api_router.include_router(match.router, prefix="/match", tags=["CV-Job Matching"])
api_router.include_router(health.router, prefix="/health", tags=["Health Check"])