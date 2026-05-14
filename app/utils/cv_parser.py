# app/utils/cv_parser.py
"""
CV Parser Orchestrator — CHỈ điều phối, không chứa logic parsing.

Pipeline:
  1. Check Redis cache → hit thì trả luôn
  2. Thử LLM strategy (primary) → thành công thì cache + trả về
  3. LLM fail → Regex strategy (fallback) → trả về

Mỗi strategy TỰ ĐỦ SỐNG — đều trả CVSections đầy đủ 100% fields.
Muốn thêm/bỏ strategy → chỉ sửa file này, không đụng logic parsing.
"""

import hashlib
import json
from dataclasses import asdict

from app.core.logging import logger
from app.core.redis_config import get_redis_client
from app.utils.cv_parser_regex import CVSections
from app.utils.cv_parser_regex import parse_sections as _regex_parse
from app.utils import cv_parser_llm
from app.utils import cv_parser_groq
from app.utils.cv_text_extractor import extract_cv_text  # Re-export để backward compat

CACHE_PREFIX = "cv_parse:"
CACHE_TTL = 3600  # 1 giờ


# ============================================================
# PUBLIC ENTRY POINTS
# ============================================================

async def parse_cv_sections(raw_text: str) -> CVSections:
    """
    Orchestrator chính: nhận raw text, trả CVSections đầy đủ.
    Được gọi bởi matching_service.py.
    """
    # === BƯỚC 1: Cache check ===
    cached = await _get_cached(raw_text)
    if cached:
        logger.info("CV parse cache HIT")
        return cached

    # === BƯỚC 2: Primary — LLM strategy (Groq) ===
    result = await cv_parser_groq.parse_sections(raw_text)

    if result is not None:
        await _set_cache(raw_text, result)
        return result

    # === BƯỚC 3: Fallback — Gemini strategy ===
    logger.warning("Groq strategy failed → fallback to Gemini strategy")
    result = await cv_parser_llm.parse_sections(raw_text)

    if result is not None:
        await _set_cache(raw_text, result)
        return result

    # === BƯỚC 4: Final Fallback — Regex strategy ===
    logger.warning("All LLM strategies failed → fallback to regex strategy")
    return _regex_parse(raw_text)


# Backward compatibility — cv_service.py vẫn dùng parse_cv(file_path)
parse_cv = extract_cv_text


# ============================================================
# CACHE HELPERS (internal)
# ============================================================

def _cache_key(raw_text: str) -> str:
    return CACHE_PREFIX + hashlib.md5(raw_text.encode()).hexdigest()


async def _get_cached(raw_text: str) -> CVSections | None:
    try:
        redis = await get_redis_client()
        data = await redis.get(_cache_key(raw_text))
        if data:
            return CVSections(**json.loads(data))
    except Exception as e:
        logger.warning(f"Cache read failed: {e}")
    return None


async def _set_cache(raw_text: str, result: CVSections):
    try:
        redis = await get_redis_client()
        await redis.set(
            _cache_key(raw_text),
            json.dumps(asdict(result), ensure_ascii=False),
            ex=CACHE_TTL
        )
    except Exception as e:
        logger.warning(f"Cache write failed: {e}")
