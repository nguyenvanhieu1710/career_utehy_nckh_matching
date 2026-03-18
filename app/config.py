# app/config.py

import os
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    # MongoDB Configuration
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "career_db")
    
    # Embedding Model
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    
    # File Storage
    JD_FOLDER: str = os.getenv("JD_FOLDER", "data/jds")
    CV_TEXT_LOG_DIR: str = os.getenv("CV_TEXT_LOG_DIR", "data/log-cv-to-text")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "data/cvs")
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_DEBUG: bool = os.getenv("API_DEBUG", "false").lower() == "true"
    
    # CORS Origins
    ALLOWED_ORIGINS: List[str] = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
    
    # Optional: Hugging Face Token for higher rate limits
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()

# Backward compatibility
EMBEDDING_MODEL_NAME = settings.EMBEDDING_MODEL_NAME
JD_FOLDER = settings.JD_FOLDER
CV_TEXT_LOG_DIR = settings.CV_TEXT_LOG_DIR
UPLOAD_DIR = settings.UPLOAD_DIR
HF_TOKEN = settings.HF_TOKEN
API_HOST = settings.API_HOST
API_PORT = settings.API_PORT
API_DEBUG = settings.API_DEBUG
