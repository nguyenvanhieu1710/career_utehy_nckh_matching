# app/cv_parser.py

import os
import uuid
from datetime import datetime
from pypdf import PdfReader
from app.core.config import settings
from app.core.logging import logger

def extract_with_pypdf(file_path: str) -> str:
    """Extract text using PyPDF (existing method)"""
    try:
        reader = PdfReader(file_path)
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text.strip():  # Only add non-empty pages
                text.append(page_text)
        
        if not text:
            logger.warning("PyPDF could not extract any text from PDF")
            return ""
            
        extracted_text = "\n".join(text)
        logger.info(f"PyPDF extracted {len(extracted_text)} characters")
        return extracted_text
        
    except Exception as e:
        logger.error(f"PyPDF extraction failed: {str(e)}")
        return ""

def extract_with_ocr(file_path: str) -> str:
    """Extract text using OCR (fallback method)"""
    try:
        from pdf2image import convert_from_path
        import pytesseract
        
        logger.info("Starting OCR extraction for PDF")
        images = convert_from_path(file_path, dpi=200, fmt='jpeg')
        text_parts = []
        
        for i, image in enumerate(images):
            try:
                text = pytesseract.image_to_string(image, lang='eng')
                text_parts.append(text)
                logger.debug(f"OCR processed page {i+1}/{len(images)}")
            except Exception as e:
                logger.warning(f"OCR failed on page {i+1}: {str(e)}")
                continue
        
        extracted_text = "\n".join(text_parts)
        logger.info(f"OCR extracted {len(extracted_text)} characters from {len(images)} pages")
        return extracted_text
        
    except ImportError as e:
        logger.error(f"OCR dependencies not installed: {str(e)}")
        logger.error("Please install: pip install pytesseract pdf2image Pillow")
        return ""
    except Exception as e:
        logger.error(f"OCR extraction failed: {str(e)}")
        return ""

def parse_pdf(file_path: str) -> str:
    """Parse PDF with fallback to OCR"""
    try:
        # Try PyPDF first (fast and accurate for text-based PDFs)
        extracted_text = extract_with_pypdf(file_path)
        
        # Check if extracted text is meaningful (not just spaced characters)
        is_meaningful = is_text_meaningful(extracted_text)
        
        # If PyPDF failed or extracted garbage text, try OCR
        if not extracted_text.strip() or not is_meaningful:
            logger.warning("PyPDF extraction insufficient or garbage, trying OCR...")
            extracted_text = extract_with_ocr(file_path)
        
        if not extracted_text.strip():
            raise ValueError("No text could be extracted from PDF - possibly corrupted or unsupported format")
            
        # Log the final extracted text
        log_extracted_text(file_path, extracted_text)
        
        return extracted_text
        
    except Exception as e:
        error_text = f"Error parsing PDF: {str(e)}"
        logger.error(error_text)
        log_extracted_text(file_path, error_text)
        raise ValueError(error_text)


def is_text_meaningful(text: str) -> bool:
    """Check if extracted text is meaningful or garbage OCR/spaced text"""
    if not text or len(text.strip()) < 50:
        return False
    
    # Check for excessive spacing between characters (OCR garbage)
    words = text.split()
    if not words:
        return False
    
    # Calculate average word length
    avg_word_length = sum(len(word) for word in words) / len(words)
    
    # If average word length is very short (< 2), likely spaced garbage
    if avg_word_length < 2:
        return False
    
    # Check for common patterns of spaced text
    spaced_chars = len([c for c in text if c == ' '])
    total_chars = len(text.replace('\n', '').replace('\r', ''))
    
    # If spaces are more than 30% of characters, likely garbage
    if total_chars > 0 and spaced_chars / total_chars > 0.3:
        return False
    
    # Check for actual words (not single characters)
    meaningful_words = [word for word in words if len(word) > 1]
    if len(meaningful_words) < len(words) * 0.5:  # Less than 50% meaningful words
        return False
    
    return True


def log_extracted_text(original_file_path: str, text: str):
    """Save extracted text to log file"""
    os.makedirs(settings.CV_TEXT_LOG_DIR, exist_ok=True)
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    original_name = os.path.basename(original_file_path).replace('.pdf', '')
    log_filename = f"{timestamp}_{unique_id}_{original_name}.txt"
    
    log_path = os.path.join(settings.CV_TEXT_LOG_DIR, log_filename)
    
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Original file: {original_file_path}\n")
        f.write(f"Extraction time: {datetime.now().isoformat()}\n")
        f.write(f"Text length: {len(text)} characters\n")
        f.write("-" * 50 + "\n")
        f.write(text)


def parse_cv(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        return parse_pdf(file_path)
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError("Unsupported CV format")
