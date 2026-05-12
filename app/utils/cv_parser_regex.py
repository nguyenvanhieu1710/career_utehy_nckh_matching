# app/utils/cv_parser_regex.py
"""
Strategy: Parse CV sections bằng Regex.
Tự trả đủ toàn bộ CVSections — không phụ thuộc strategy nào khác.
Được dùng làm fallback khi LLM strategy fail.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime

from app.core.logging import logger


# ============================================================
# SECTION PATTERNS & LOCATION MAPPING
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


# ============================================================
# CVSections DATACLASS — định nghĩa tại đây, import bởi các file khác
# ============================================================

@dataclass
class CVSections:
    """Kết quả parse CV thành các trường có cấu trúc"""
    title: str = ""
    location: str = ""
    tech: str = ""
    mota: str = ""
    years_of_experience: float = 0.0
    raw_sections: dict = field(default_factory=dict)


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _normalize_location(text: str) -> str:
    return LOCATION_MAPPING.get(text.lower().strip(), text.lower().strip())


def _split_into_sections(text: str) -> dict:
    """Tách raw text thành dict các section dựa trên heading patterns"""
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
    """Trích xuất job title từ phần header của CV"""
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
    """Trích xuất và chuẩn hóa tên thành phố từ header CV"""
    text_lower = header_text.lower()
    for city_raw in sorted(KNOWN_CITIES, key=len, reverse=True):
        if city_raw in text_lower:
            return _normalize_location(city_raw)
    return ""


def _calculate_years_of_experience(experience_text: str) -> float:
    """Tính số năm kinh nghiệm từ các khoảng thời gian trong phần WORK EXPERIENCE"""
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


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def parse_sections(raw_text: str) -> CVSections:
    """
    Entry point của Regex strategy.
    Trả CVSections đầy đủ 100% fields — tự đủ sống, không phụ thuộc strategy khác.
    """
    logger.debug("Regex strategy: parsing CV sections")
    raw_sections = _split_into_sections(raw_text)

    return CVSections(
        title=_extract_title(raw_sections.get("header", "")),
        location=_extract_location(raw_sections.get("header", "")),
        tech=raw_sections.get("skills", ""),
        mota="\n".join(filter(None, [
            raw_sections.get("objective", ""),
            raw_sections.get("experience", ""),
            raw_sections.get("certif", ""),
        ])),
        years_of_experience=_calculate_years_of_experience(
            raw_sections.get("experience", "")
        ),
        raw_sections=raw_sections,
    )
