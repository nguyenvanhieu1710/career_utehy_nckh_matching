# app/core/database.py

from motor.motor_asyncio import AsyncIOMotorClient
from pymilvus import MilvusClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()

class MongoDB:
    client: AsyncIOMotorClient = None
    database = None

class MilvusDB:
    client: MilvusClient = None

class PostgresDB:
    engine = None
    session_factory = None

mongodb = MongoDB()
milvusdb = MilvusDB()
postgresdb = PostgresDB()

async def connect_to_mongo():
    """Create MongoDB connection"""
    try:
        mongodb.client = AsyncIOMotorClient(settings.MONGODB_URL)
        mongodb.database = mongodb.client[settings.MONGODB_DB_NAME]
        
        # Test connection
        await mongodb.client.admin.command('ping')
        logger.info(f"Connected to MongoDB: {settings.MONGODB_DB_NAME}")
        
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        raise

async def connect_to_postgres():
    """Create PostgreSQL connection"""
    try:
        postgresdb.engine = create_async_engine(settings.SQLALCHEMY_DATABASE_URI, echo=False)
        postgresdb.session_factory = sessionmaker(
            postgresdb.engine, class_=AsyncSession, expire_on_commit=False
        )
        logger.info(f"Connected to PostgreSQL: {settings.POSTGRES_DB}")
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {str(e)}")

def connect_to_milvus():
    """Create Milvus connection"""
    try:
        milvusdb.client = MilvusClient(
            uri=settings.MILVUS_URL,
            token=settings.MILVUS_TOKEN
        )
        logger.info(f"Connected to Milvus: {settings.MILVUS_URL}")
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {str(e)}")
        # Don't raise here, allow app to run without Milvus if needed (though matching will fail)

async def close_mongo_connection():
    """Close MongoDB connection"""
    if mongodb.client:
        mongodb.client.close()
        logger.info("Disconnected from MongoDB")

async def close_postgres_connection():
    """Close PostgreSQL connection"""
    if postgresdb.engine:
        await postgresdb.engine.dispose()
        logger.info("Disconnected from PostgreSQL")

def get_database():
    """Get MongoDB instance"""
    return mongodb.database

async def get_postgres_session():
    """Get PostgreSQL session factory"""
    return postgresdb.session_factory()

def get_milvus_client():
    """Get Milvus client instance"""
    if milvusdb.client is None:
        connect_to_milvus()
    return milvusdb.client