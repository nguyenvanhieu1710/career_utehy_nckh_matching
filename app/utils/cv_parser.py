# app/utils/cv_parser.py
"""
CV Parser - Xử lý toàn bộ pipeline đọc và phân tích CV:
1. Trích xuất text từ PDF (PyPDF → OCR fallback)
2. Tách text thành các trường có cấu trúc (title, location, tech, mota, ...)
"""

import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from pypdf import PdfReader
from app.core.config import settings
from app.core.logging import logger

# ============================================================
# PART 1: EXTRACT TEXT FROM PDF
# ============================================================

def extract_with_pypdf(file_path: str) -> str:
    """Extract text using PyPDF"""
    try:
        reader = PdfReader(file_path)
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text.strip():
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
    """Extract text from PDF using OCR (fallback)"""
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
        return ""
    except Exception as e:
        logger.error(f"OCR extraction failed: {str(e)}")
        return ""


def is_text_meaningful(text: str) -> bool:
    """Check if text is meaningful or not"""
    if not text or len(text.strip()) < 50:
        return False
    words = text.split()
    if not words:
        return False
    avg_word_length = sum(len(word) for word in words) / len(words)
    if avg_word_length < 2:
        return False
    spaced_chars = len([c for c in text if c == ' '])
    total_chars = len(text.replace('\n', '').replace('\r', ''))
    if total_chars > 0 and spaced_chars / total_chars > 0.3:
        return False
    meaningful_words = [word for word in words if len(word) > 1]
    if len(meaningful_words) < len(words) * 0.5:
        return False
    return True


def log_extracted_text(original_file_path: str, text: str):
    """Save extracted text to log file"""
    os.makedirs(settings.CV_TEXT_LOG_DIR, exist_ok=True)
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


def parse_pdf(file_path: str) -> str:
    """Parse PDF với fallback sang OCR nếu cần"""
    try:
        extracted_text = extract_with_pypdf(file_path)
        if not extracted_text.strip() or not is_text_meaningful(extracted_text):
            logger.warning("PyPDF extraction insufficient, trying OCR...")
            extracted_text = extract_with_ocr(file_path)
        if not extracted_text.strip():
            raise ValueError("No text could be extracted from PDF")
        log_extracted_text(file_path, extracted_text)
        return extracted_text
    except Exception as e:
        error_text = f"Error parsing PDF: {str(e)}"
        logger.error(error_text)
        log_extracted_text(file_path, error_text)
        raise ValueError(error_text)


def parse_cv(file_path: str) -> str:
    """Entry point: Parse CV từ file path, trả về raw text"""
    if file_path.endswith(".pdf"):
        return parse_pdf(file_path)
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError("Unsupported CV format")


# ============================================================
# PART 2: SPLIT TEXT INTO STRUCTURED FIELDS
# ============================================================

SECTION_PATTERNS = {
    "skills":     r"^\s*(KỸ NĂNG|SKILLS?|TECHNICAL SKILLS?|CÔNG NGHỆ|TECHNOLOGIES?|TOOLS?)\s*$",
    "experience": r"^\s*(KINH NGHIỆM|WORK EXPERIENCE|EXPERIENCE|PROFESSIONAL EXPERIENCE)\s*$",
    "objective":  r"^\s*(MỤC TIÊU|OBJECTIVE|SUMMARY|PROFILE|ABOUT ME)\s*$",
    "education":  r"^\s*(HỌC VẤN|EDUCATION|ACADEMIC)\s*$",
    "certif":     r"^\s*(CHỨNG CHỈ|CERTIFICATIONS?|CERTIFICATES?)\s*$",
    "awards":     r"^\s*(GIẢI THƯỞNG|AWARDS?|HONORS?|ACHIEVEMENTS?)\s*$",
}

LOCATION_MAPPING = {
    "hồ chí minh": "tp.hcm", "ho chi minh": "tp.hcm",
    "tphcm": "tp.hcm", "tp hcm": "tp.hcm", "tp.hcm": "tp.hcm",
    "sài gòn": "tp.hcm", "sai gon": "tp.hcm", "hcm": "tp.hcm",
    "hà nội": "hà nội", "ha noi": "hà nội", "hanoi": "hà nội",
    "đà nẵng": "đà nẵng", "da nang": "đà nẵng", "danang": "đà nẵng",
    "hải dương": "hải dương", "hai duong": "hải dương",
    "hưng yên": "hưng yên", "hung yen": "hưng yên",
    "hải phòng": "hải phòng", "hai phong": "hải phòng",
    "cần thơ": "cần thơ", "can tho": "cần thơ",
    "bình dương": "bình dương", "binh duong": "bình dương",
    "đồng nai": "đồng nai", "dong nai": "đồng nai",
    "bắc ninh": "bắc ninh", "bac ninh": "bắc ninh",
}

KNOWN_CITIES = list(LOCATION_MAPPING.keys())


@dataclass
class CVSections:
    """Result of splitting CV into structured fields"""
    title: str = ""
    location: str = ""
    tech: str = ""
    mota: str = ""
    years_of_experience: float = 0.0
    raw_sections: dict = field(default_factory=dict)


def _normalize_location(text: str) -> str:
    return LOCATION_MAPPING.get(text.lower().strip(), text.lower().strip())


def _split_into_sections(text: str) -> dict:
    """Extracts sections from raw CV text based on predefined patterns"""
    sections = {k: [] for k in SECTION_PATTERNS}
    sections["header"] = []
    sections["other"] = []
    current = "header"

    for line in text.splitlines():
        detected = None
        for key, pattern in SECTION_PATTERNS.items():
            if re.match(pattern, line, re.IGNORECASE):
                detected = key
                break
        if detected:
            current = detected
        else:
            sections[current].append(line)

    return {k: "\n".join(v).strip() for k, v in sections.items()}


def _extract_title(header_text: str) -> str:
    """Extract title from header section of CV"""
    phone_re = re.compile(r'\d{9,11}')
    email_re = re.compile(r'@')
    url_re = re.compile(r'http|www\.', re.IGNORECASE)
    year_re = re.compile(r'\b\d{4}\b')

    lines = [l.strip() for l in header_text.splitlines() if l.strip()]
    for line in lines[1:6]:
        if phone_re.search(line): continue
        if email_re.search(line): continue
        if url_re.search(line): continue
        if year_re.search(line): continue
        if len(line) < 60 and not line.startswith("•"):
            return line
    return ""


def _extract_location(header_text: str) -> str:
    """Extract and normalize city from CV header"""
    text_lower = header_text.lower()
    for city_raw in sorted(KNOWN_CITIES, key=len, reverse=True):
        if city_raw in text_lower:
            return _normalize_location(city_raw)
    return ""


def _calculate_years_of_experience(experience_text: str) -> float:
    """Calculate years of experience from time periods in WORK EXPERIENCE"""
    now = datetime.now()
    period_re = re.compile(
        r'(\d{1,2})[/\-](\d{4})\s*[-–]\s*(\d{1,2})[/\-](\d{4})|'
        r'(\d{1,2})[/\-](\d{4})\s*[-–]\s*(current|present|now|nay|hiện tại)',
        re.IGNORECASE
    )
    total_months = 0
    for m in period_re.finditer(experience_text):
        g = m.groups()
        if g[0] and g[1] and g[2] and g[3]:
            start = int(g[1]) * 12 + int(g[0])
            end = int(g[3]) * 12 + int(g[2])
            total_months += max(0, end - start)
        elif g[4] and g[5]:
            start = int(g[5]) * 12 + int(g[4])
            end = now.year * 12 + now.month
            total_months += max(0, end - start)
    return round(total_months / 12, 1)


def parse_cv_sections(raw_text: str) -> CVSections:
    """
    Receive raw text from CV, return CVSections with structured fields.
    This is the main function to use in matching_service.
    """
    raw_sections = _split_into_sections(raw_text)

    title = _extract_title(raw_sections.get("header", ""))
    location = _extract_location(raw_sections.get("header", ""))
    tech = raw_sections.get("skills", "")
    mota = "\n".join(filter(None, [
        raw_sections.get("objective", ""),
        raw_sections.get("experience", ""),
        raw_sections.get("certif", ""),
    ]))
    years_exp = _calculate_years_of_experience(raw_sections.get("experience", ""))

    return CVSections(
        title=title,
        location=location,
        tech=tech,
        mota=mota,
        years_of_experience=years_exp,
        raw_sections=raw_sections,
    )
