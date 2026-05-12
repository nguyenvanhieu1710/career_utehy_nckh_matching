# app/utils/cv_parser_llm.py
"""
Strategy: Parse CV sections bằng Gemini LLM.
Tự trả đủ toàn bộ CVSections — không phụ thuộc strategy nào khác.
Được dùng làm primary strategy, chất lượng cao hơn regex.
"""

import json
import asyncio
import time
from google import genai
from google.genai import types

from app.core.config import settings
from app.core.logging import logger
from app.utils.cv_parser_regex import CVSections  # Dùng chung dataclass


# ============================================================
# LLM CLIENT (Lazy-init)
# ============================================================

_client = None

def get_client():
    """Get or create Gemini client (lazy-init)"""
    global _client
    if _client is None:
        key = settings.GEMINI_API_KEY
        if not key:
            logger.warning("GEMINI_API_KEY is missing. Gemini API will fail.")
            return None
        try:
            _client = genai.Client(api_key=key)
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {str(e)}")
            return None
    return _client


# ============================================================
# PROMPT & JSON SCHEMA
# ============================================================

EXTRACT_PROMPT = """Bạn là chuyên gia phân tích CV tuyển dụng. Hãy đọc CV text dưới đây và trích xuất thông tin theo đúng JSON format.

QUY TẮC TRÍCH XUẤT:
- "title": Vị trí/chức danh ứng viên đang ứng tuyển hoặc đang làm. VD: "Backend Developer", "Fullstack Engineer". Để "" nếu không rõ.
- "location": Thành phố ứng viên muốn làm việc (ưu tiên) hoặc đang sinh sống. VD: "Hà Nội", "TP.HCM", "Hưng Yên". Để "" nếu không rõ.
- "tech": Liệt kê TẤT CẢ kỹ năng, ngôn ngữ lập trình, framework, tools, công nghệ mà ứng viên biết. Phân tách bằng dấu phẩy. VD: "Python, FastAPI, Docker, PostgreSQL, Redis".
- "mota": Tóm tắt toàn bộ kinh nghiệm làm việc, dự án đã thực hiện, mô tả công việc. Gộp tất cả vào 1 đoạn text liên tục.
- "years_of_experience": Tổng số năm kinh nghiệm làm việc thực tế (số thực). QUAN TRỌNG: Phải tính cả thời gian thực tập (Internship/Student projects). Nếu làm 6 tháng ghi 0.5, 3 tháng ghi 0.25. Chỉ để 0.0 nếu tuyệt đối chưa từng đi làm hay thực tập.
- "sections": Object chứa từng section đã tách được từ CV. Mỗi key là tên section, value là nội dung text.

CHỈ trả về JSON, không giải thích gì thêm.

CV TEXT:
---
{cv_text}
---"""

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "title":               {"type": "string"},
        "location":            {"type": "string"},
        "tech":                {"type": "string"},
        "mota":                {"type": "string"},
        "years_of_experience": {"type": "number"},
        "sections": {
            "type": "object",
            "properties": {
                "skills":     {"type": "string"},
                "experience": {"type": "string"},
                "objective":  {"type": "string"},
                "education":  {"type": "string"},
                "certif":     {"type": "string"},
                "awards":     {"type": "string"},
                "header":     {"type": "string"},
                "other":      {"type": "string"},
            }
        }
    },
    "required": ["title", "location", "tech", "mota", "years_of_experience"]
}

# Giới hạn độ dài text gửi cho LLM để tránh vượt context window
MAX_CV_TEXT_LENGTH = 4000


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

async def parse_sections(raw_text: str) -> CVSections | None:
    """
    Entry point của LLM strategy.
    Trả CVSections đầy đủ 100% fields nếu thành công, None nếu fail.
    Orchestrator dùng None để biết cần chuyển sang fallback.
    Cơ chế: Thử lại tối đa 3 lần nếu gặp lỗi quá tải (503, 429).
    """
    client = get_client()
    if not client:
        return None

    max_retries = 3
    base_delay = 2  # giây

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=EXTRACT_PROMPT.format(cv_text=raw_text[:MAX_CV_TEXT_LENGTH]),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=JSON_SCHEMA,
                    temperature=0.1,
                ),
            )

            if not response.text:
                logger.warning("Gemini returned empty response")
                return None

            data = json.loads(response.text)
            result = CVSections(
                title=data.get("title", ""),
                location=data.get("location", ""),
                tech=data.get("tech", ""),
                mota=data.get("mota", ""),
                years_of_experience=float(data.get("years_of_experience", 0.0)),
                raw_sections=data.get("sections", {}),
            )
            logger.info(
                f"LLM strategy OK — title='{result.title}', "
                f"location='{result.location}', exp={result.years_of_experience}yr"
            )
            return result

        except Exception as e:
            err_msg = str(e)
            # Kiểm tra xem có phải lỗi quá tải (503) hoặc giới hạn request (429) không
            is_retryable = "503" in err_msg or "429" in err_msg or "UNAVAILABLE" in err_msg or "Resource has been exhausted" in err_msg
            
            if is_retryable and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # 2, 4, 8 giây
                logger.warning(f"Gemini busy (attempt {attempt+1}/{max_retries}). Retrying in {delay}s... Error: {err_msg}")
                await asyncio.sleep(delay)
            else:
                logger.error(f"LLM strategy failed after {attempt+1} attempts: {err_msg}")
                return None
    
    return None
