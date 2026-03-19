# app/core/logging.py

import logging
import os
from app.core.config import CV_TEXT_LOG_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Console logging only
    ]
)

# Get logger
logger = logging.getLogger(__name__)
