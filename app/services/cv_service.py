# app/services/cv_service.py

import os
import shutil
from typing import Optional
from fastapi import UploadFile
from app.cv_parser import parse_cv
from app.schemas.cv import CVJsonInput, CVParsedData
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class CVService:
    """Service for processing CV data"""
    
    @staticmethod
    async def process_cv_file(file: UploadFile) -> CVParsedData:
        """Process uploaded CV file (PDF)"""
        try:
            # Ensure upload directory exists
            os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
            
            # Save uploaded file
            file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Parse CV text
            cv_text = parse_cv(file_path)
            
            # Clean up uploaded file (optional)
            try:
                os.remove(file_path)
            except:
                pass
            
            # Create parsed data object
            parsed_data = CVParsedData(
                raw_text=cv_text,
                name=CVService._extract_name_from_text(cv_text),
                email=CVService._extract_email_from_text(cv_text),
                skills=CVService._extract_skills_from_text(cv_text)
            )
            
            logger.info(f"Successfully processed CV file: {file.filename}")
            return parsed_data
            
        except Exception as e:
            logger.error(f"Failed to process CV file {file.filename}: {str(e)}")
            raise ValueError(f"Failed to process CV file: {str(e)}")
    
    @staticmethod
    def process_cv_json(cv_data: CVJsonInput) -> CVParsedData:
        """Process CV data from JSON input"""
        try:
            # Combine all text fields for matching
            text_parts = []
            
            if cv_data.name:
                text_parts.append(f"Name: {cv_data.name}")
            
            if cv_data.summary:
                text_parts.append(f"Summary: {cv_data.summary}")
            
            if cv_data.experience:
                text_parts.append(f"Experience: {cv_data.experience}")
            
            if cv_data.education:
                text_parts.append(f"Education: {cv_data.education}")
            
            if cv_data.skills:
                text_parts.append(f"Skills: {', '.join(cv_data.skills)}")
            
            if cv_data.certifications:
                text_parts.append(f"Certifications: {', '.join(cv_data.certifications)}")
            
            if cv_data.languages:
                text_parts.append(f"Languages: {', '.join(cv_data.languages)}")
            
            raw_text = "\n".join(text_parts)
            
            # Create parsed data object
            parsed_data = CVParsedData(
                raw_text=raw_text,
                name=cv_data.name,
                email=cv_data.email,
                skills=cv_data.skills or [],
                experience=cv_data.experience,
                education=cv_data.education
            )
            
            logger.info(f"Successfully processed CV JSON for: {cv_data.name}")
            return parsed_data
            
        except Exception as e:
            logger.error(f"Failed to process CV JSON: {str(e)}")
            raise ValueError(f"Failed to process CV JSON: {str(e)}")
    
    @staticmethod
    def _extract_name_from_text(text: str) -> Optional[str]:
        """Extract name from CV text (basic implementation)"""
        try:
            lines = text.split('\n')
            for line in lines[:5]:  # Check first 5 lines
                line = line.strip()
                if line and len(line.split()) <= 4 and len(line) > 5:
                    # Likely a name if it's short and at the top
                    return line
            return None
        except:
            return None
    
    @staticmethod
    def _extract_email_from_text(text: str) -> Optional[str]:
        """Extract email from CV text"""
        import re
        try:
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            matches = re.findall(email_pattern, text)
            return matches[0] if matches else None
        except:
            return None
    
    @staticmethod
    def _extract_skills_from_text(text: str) -> list:
        """Extract skills from CV text using existing skill extractor"""
        try:
            from app.skill_extractor import extract_skills
            return extract_skills(text)
        except:
            return []