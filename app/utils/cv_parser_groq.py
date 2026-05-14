# app/utils/cv_parser_groq.py
"""
Strategy: Parse CV sections bằng Groq LLM (High speed fallback).
"""

import json
import asyncio
from groq import Groq

from app.core.config import settings
from app.core.logging import logger
from app.utils.cv_parser_regex import CVSections

_client = None

def get_client():
    global _client
    if _client is None:
        key = settings.GROQ_API_KEY
        if not key:
            logger.warning("GROQ_API_KEY is missing.")
            return None
        try:
            _client = Groq(api_key=key)
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {str(e)}")
            return None
    return _client

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

MAX_CV_TEXT_LENGTH = 4000

async def parse_sections(raw_text: str) -> CVSections | None:
    client = get_client()
    if not client:
        return None

    try:
        # Groq is synchronous, wrap in run_in_executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are a CV analyst. Return only JSON."},
                    {"role": "user", "content": EXTRACT_PROMPT.format(cv_text=raw_text[:MAX_CV_TEXT_LENGTH])}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
        )

        content = response.choices[0].message.content
        if not content:
            return None

        data = json.loads(content)
        result = CVSections(
            title=data.get("title", ""),
            location=data.get("location", ""),
            tech=data.get("tech", ""),
            mota=data.get("mota", ""),
            years_of_experience=float(data.get("years_of_experience", 0.0)),
            raw_sections=data.get("sections", {}),
        )
        logger.info(f"Groq strategy OK — title='{result.title}', exp={result.years_of_experience}yr")
        return result

    except Exception as e:
        logger.error(f"Groq strategy failed: {str(e)}")
        return None
