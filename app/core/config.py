# app/core/config.py

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Application settings"""
    
    # App Settings
    APP_HOST: str = os.getenv("APP_HOST")
    APP_PORT: int = int(os.getenv("APP_PORT"))
    APP_RELOAD: bool = os.getenv("APP_RELOAD", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL")
    
    # CORS Settings
    ALLOWED_ORIGINS: list = os.getenv("ALLOWED_ORIGINS", "").split(",")
    
    # MongoDB Settings
    MONGODB_URL: str = os.getenv("MONGODB_URL")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME")
    MONGODB_COLLECTION_COMPANIES: str = os.getenv("MONGODB_COLLECTION_COMPANIES")
    
    # PostgreSQL Settings
    POSTGRES_USER: str = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD")
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB")
    
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    # Milvus Settings
    MILVUS_URL: str = os.getenv("MILVUS_URL")
    MILVUS_TOKEN: str = os.getenv("MILVUS_TOKEN")
    MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION")
    
    # Redis Settings
    REDIS_HOST: str = os.getenv("REDIS_HOST")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD")
    REDIS_USERNAME: str = os.getenv("REDIS_USERNAME")
    REDIS_DB: int = int(os.getenv("REDIS_DB"))
    
    # Matching Weights
    WEIGHT_TITLE: float = float(os.getenv("WEIGHT_TITLE"))
    WEIGHT_TECH: float = float(os.getenv("WEIGHT_TECH"))
    WEIGHT_MOTA: float = float(os.getenv("WEIGHT_MOTA"))
    WEIGHT_LOCATION: float = float(os.getenv("WEIGHT_LOCATION"))
    WEIGHT_EXPERIENCE: float = float(os.getenv("WEIGHT_EXPERIENCE"))
    
    # ML Model Settings
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME")
    HF_TOKEN: str = os.getenv("HF_TOKEN")
    
    # File Upload Settings
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR")
    CV_TEXT_LOG_DIR: str = os.getenv("CV_TEXT_LOG_DIR")
    MAX_UPLOAD_SIZE: int = int(os.getenv("MAX_UPLOAD_SIZE"))

    # Cache Settings
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "false").lower() == "true"
    JOBS_CACHE_TTL: int = int(os.getenv("JOBS_CACHE_TTL"))
    MATCH_CACHE_TTL: int = int(os.getenv("MATCH_CACHE_TTL"))

settings = Settings()

# Export CV_TEXT_LOG_DIR for backward compatibility
CV_TEXT_LOG_DIR = settings.CV_TEXT_LOG_DIR
