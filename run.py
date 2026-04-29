import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from app.core.config import settings

if __name__ == "__main__":    
    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.APP_RELOAD
    )
