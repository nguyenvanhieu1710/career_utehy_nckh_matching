# app/services/matching_service.py

import time
from typing import List
from app.schemas.cv import CVParsedData
from app.schemas.job import JobSchema
from app.schemas.match import MatchResult, MatchResponse, MatchRequest
from app.services.job_service import JobService
from app.services.ml.similarity import analyze_fit
from app.services.ml.embeddings import embed_text
from app.utils.skill_extractor import extract_skills
from app.utils.cv_parser import parse_cv_sections
from app.utils.text_cleaner import clean_text
from app.core.database import get_milvus_client
from app.core.config import settings
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class MatchingService:
    """Service for CV-Job matching"""
    
    @staticmethod
    async def match_cv_to_jobs(
        cv_data: CVParsedData, 
        match_params: MatchRequest
    ) -> MatchResponse:
        """
        Match CV to jobs using Milvus Multi-Vector search and Heuristic Scoring
        """
        start_time = time.time()
        
        try:
            import asyncio

            # 1. Extract structured fields from CV
            # Only bypass LLM parsing if CV is truly structured online profile data (has experience/education details)
            # PDF CV uploads parsed by cv_service do NOT have experience or education populated in python objects,
            # meaning they MUST be parsed via LLM to extract rich semantic fields.
            is_truly_structured = (
                hasattr(cv_data, 'skills') and cv_data.skills and len(cv_data.skills) > 0
            ) and (
                (hasattr(cv_data, 'experience') and cv_data.experience and len(cv_data.experience) > 10) or
                (hasattr(cv_data, 'education') and cv_data.education and len(cv_data.education) > 10)
            )
            
            if is_truly_structured:
                class CVSectionsMock:
                    title = getattr(cv_data, 'name', '') or ""
                    tech = ", ".join(cv_data.skills) if cv_data.skills else ""
                    mota = getattr(cv_data, 'experience', "") or ""
                    location = ""
                    years_of_experience = 0.0
                cv_sections = CVSectionsMock()
                logger.info("Bypassed LLM parsing for truly structured CV data.")
            else:
                logger.info("Parsing CV via LLM to ensure high quality analysis.")
                cv_sections = await parse_cv_sections(cv_data.raw_text)
            # logger.info(f"CV parsed - Title: '{cv_sections.title}', Location: '{cv_sections.location}', Exp: {cv_sections.years_of_experience}yr")

            # 2. Create 3 separate vectors from CV parts corresponding to JD
            # Title fallback priority: title → mota (infer role from experience) → tech (infer role from skills)
            # Avoid embedding raw_text[:200] which is typically just contact info (phone, email, address)
            title_text = clean_text(
                cv_sections.title
                or cv_sections.mota
                or cv_sections.tech
                or cv_data.raw_text
            )
            tech_text = clean_text(cv_sections.tech or cv_data.raw_text)
            # Mota fallback: combine mota + tech for richer semantic context when mota is sparse
            mota_text = clean_text(
                cv_sections.mota
                or (cv_sections.tech + " " + cv_data.raw_text[:500] if cv_sections.tech else cv_data.raw_text)
            )
            logger.info(f"CV vectors — title_src={'title' if cv_sections.title else 'mota' if cv_sections.mota else 'tech'}, "
                        f"mota_len={len(cv_sections.mota)}, tech_len={len(cv_sections.tech)}")

            # Batch embedding: Pass all 3 texts in a single list to leverage model batching
            vectors = embed_text([title_text, tech_text, mota_text])
            title_vector = vectors[0].tolist()
            tech_vector = vectors[1].tolist()
            mota_vector = vectors[2].tolist()

            # 3. Search Milvus with correct vector for correct field
            milvus_client = get_milvus_client()
            # Increased to 15x to ensure title_vec candidates (e.g. Fullstack jobs)
            # are included in the pool alongside high-tech-score jobs
            search_top_k = match_params.top_k * 15

            def search_field(field, vector):
                # Add ID filter if provided
                filter_expr = None
                if match_params.job_ids:
                    filter_expr = f'job_id in {match_params.job_ids}'
                    
                return milvus_client.search(
                    collection_name=settings.MILVUS_COLLECTION,
                    data=[vector],
                    anns_field=field,
                    limit=search_top_k,
                    filter=filter_expr,  # Apply the ID filter here
                    search_params={"metric_type": "COSINE", "params": {}},
                    output_fields=["job_id", "job_title", "company_name", "location_city", "exp_min", "exp_max", "skills"]
                )[0]

            loop = asyncio.get_event_loop()
            title_hits, tech_hits, mota_hits = await asyncio.gather(
                loop.run_in_executor(None, search_field, "title_vec", title_vector),
                loop.run_in_executor(None, search_field, "tech_vec", tech_vector),
                loop.run_in_executor(None, search_field, "mota_vec", mota_vector)
            )
            
            # 3. Merge and Rank Candidates
            candidates = {}
            for hits, weight_key in [(title_hits, "sim_title"), (tech_hits, "sim_tech"), (mota_hits, "sim_mota")]:
                for hit in hits:
                    jid = hit["entity"]["job_id"]
                    if jid not in candidates:
                        candidates[jid] = {
                            "job_id": jid,
                            "job_title": hit["entity"].get("job_title", ""),
                            "company_name": hit["entity"].get("company_name", ""),
                            "location_city": hit["entity"].get("location_city", ""),
                            "exp_min": float(hit["entity"].get("exp_min", 0.0)),
                            "skills": hit["entity"].get("skills", ""),
                            "sim_title": 0.0,
                            "sim_tech": 0.0,
                            "sim_mota": 0.0,
                        }
                    candidates[jid][weight_key] = max(candidates[jid][weight_key], float(hit["distance"]))

            if not candidates:
                return MatchResponse(success=True, matches=[], total_jobs_analyzed=0, processing_time_ms=0, message="No matching jobs found")

            # 4. Weights — khai báo sớm để dùng cho cả pre-sort và heuristic scoring
            weights = {
                "title": settings.WEIGHT_TITLE,
                "tech": settings.WEIGHT_TECH,
                "mota": settings.WEIGHT_MOTA,
                "loc": settings.WEIGHT_LOCATION,
                "exp": settings.WEIGHT_EXPERIENCE
            }

            # 3.5. Fetch job details only for candidates from DB
            # Pre-sort by semantic scores to limit DB fetch to top candidates only,
            # avoiding fetching all 400+ jobs when we only need top_k results.
            PRE_FETCH_LIMIT = match_params.top_k * 5  # fetch 5x top_k, enough buffer for re-ranking
            pre_sorted_ids = sorted(
                candidates.keys(),
                key=lambda jid: (
                    candidates[jid]["sim_tech"] * weights["tech"] +
                    candidates[jid]["sim_mota"] * weights["mota"] +
                    candidates[jid]["sim_title"] * weights["title"]
                ),
                reverse=True
            )[:PRE_FETCH_LIMIT]

            job_ids_to_fetch = pre_sorted_ids
            matched_jobs = await JobService.fetch_jobs_by_ids(job_ids_to_fetch)
            job_lookup = {str(j.id): j for j in matched_jobs}

            match_results = []
            user_city = cv_sections.location
            user_exp = cv_sections.years_of_experience
            # Direct split from SKILLS section (more accurate than N-gram for short lists)
            cv_skills = MatchingService._extract_skills_from_tech(cv_sections.tech or "")
            # logger.info(f"CV skills extracted: {cv_skills}")

            for jid, c in candidates.items():
                # Normalize job skills before intersection
                job_skills_str = c.get("skills", "")
                job_skills = set([
                    MatchingService._normalize_skill_name(s)
                    for s in job_skills_str.split(",") if s.strip()
                ])

                matched_skills = list(cv_skills.intersection(job_skills))
                missing_skills = list(job_skills.difference(cv_skills))
                
                # Use reliable exp_min from PostgreSQL if available
                full_job = job_lookup.get(str(jid))
                job_exp_min = full_job.exp_min if full_job else c["exp_min"]
                
                loc_score = MatchingService._get_location_score(user_city, c["location_city"])
                exp_score = MatchingService._experience_match_score(user_exp, job_exp_min)

                # Dynamically redistribute weights when location data is missing.
                # If user_city is unknown, the location weight is redistributed to tech
                # to avoid injecting a constant 0.5 score that flattens all results.
                has_location = bool(MatchingService._normalize_city(user_city))
                if has_location:
                    w_title = weights["title"]
                    w_tech  = weights["tech"]
                    w_mota  = weights["mota"]
                    w_loc   = weights["loc"]
                    w_exp   = weights["exp"]
                else:
                    # Redistribute loc weight: 70% → tech, 30% → mota
                    w_loc   = 0.0
                    w_tech  = weights["tech"]  + weights["loc"] * 0.7
                    w_mota  = weights["mota"]  + weights["loc"] * 0.3
                    w_title = weights["title"]
                    w_exp   = weights["exp"]

                final_score = (
                    w_title * c["sim_title"] +
                    w_tech  * c["sim_tech"] +
                    w_mota  * c["sim_mota"] +
                    w_loc   * loc_score +
                    w_exp   * exp_score
                )
                
                final_score = max(0.0, min(1.0, final_score))
                
                if final_score * 100 < 0: 
                    continue
                
                # Generate suggestions if skills are missing
                suggestions = MatchingService._generate_skill_suggestions(missing_skills, c["job_title"])
                
                # Retrieve salary and url_source from job_lookup
                full_job = job_lookup.get(str(jid))
                job_salary = full_job.salary if full_job else None
                job_url = full_job.url_source if full_job else None
                job_image = full_job.image_url if full_job else None

                explanation = MatchingService._generate_explanation(
                    score=final_score * 100,
                    matched_skills=matched_skills,
                    missing_skills=missing_skills,
                    user_city=user_city,
                    job_city=c["location_city"],
                    loc_score=loc_score,
                    exp_score=exp_score,
                    user_exp=user_exp,
                    job_exp_min=job_exp_min
                )

                match_results.append(MatchResult(
                    job_id=str(jid),
                    job_title=c["job_title"],
                    company=c["company_name"],
                    compatibility_score=round(final_score * 100, 1),
                    matched_skills=matched_skills,
                    missing_skills=missing_skills,
                    skill_improvement_suggestions=suggestions,
                    location=c["location_city"],
                    match_explanation=explanation,
                    salary=job_salary,
                    url_source=job_url,
                    image_url=job_image,
                    scores={
                        "sim_title": round(c["sim_title"] * 100),
                        "sim_tech": round(c["sim_tech"] * 100),
                        "sim_mota": round(c["sim_mota"] * 100),
                        "loc_score": round(loc_score * 100),
                        "exp_score": round(exp_score * 100)
                    }
                ))

            # Sort and Limit
            match_results.sort(key=lambda x: x.compatibility_score, reverse=True)
            top_matches = match_results[:match_params.top_k]
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            return MatchResponse(
                success=True,
                matches=top_matches,
                total_jobs_analyzed=len(candidates),
                processing_time_ms=round(processing_time, 2),
                message=f"Successfully matched CV using Multi-Vector search"
            )
            
        except Exception as e:
            logger.error(f"Matching failed: {str(e)}")
            return MatchResponse(success=False, matches=[], total_jobs_analyzed=0, processing_time_ms=0, message=f"Error: {str(e)}")
    
    @staticmethod
    def _apply_filters(jobs: List[JobSchema]) -> List[JobSchema]:
        """Apply basic filters to job list"""
        return jobs
    
    @staticmethod
    def _job_to_text(job: JobSchema) -> str:
        """Convert job to text for analysis"""
        text_parts = [
            job.title,
            job.company,
            job.description or "",
            ", ".join(job.skills),
            job.location or "",
            ", ".join(job.requirements)
        ]
        
        return "\n".join(part for part in text_parts if part.strip())
    
    @staticmethod
    def _generate_explanation(
        score: float,
        matched_skills: List[str],
        missing_skills: List[str],
        user_city: str,
        job_city: str,
        loc_score: float,
        exp_score: float,
        user_exp: float,
        job_exp_min: float
    ) -> str:
        """Generate high-quality, personalized, human-readable Vietnamese match explanation"""
        # Assess match level
        if score >= 80:
            level = "Xuất sắc"
            recommend = "Hồ sơ của bạn cực kỳ khớp với các yêu cầu cốt lõi của công việc này. Đây là cơ hội tuyệt vời để bạn ứng tuyển ngay!"
        elif score >= 65:
            level = "Tốt"
            recommend = "Hồ sơ của bạn đáp ứng rất tốt đa số các tiêu chí tuyển dụng. Bạn nên tự tin nộp hồ sơ."
        elif score >= 50:
            level = "Khá"
            recommend = "Hồ sơ của bạn phù hợp một phần lớn với công việc. Hãy bổ sung thêm một số kỹ năng còn thiếu để tự tin ứng tuyển."
        else:
            level = "Trung bình"
            recommend = "Độ tương thích ở mức trung bình. Bạn nên cân nhắc học hỏi thêm các công nghệ cốt lõi trước khi ứng tuyển."

        explanation = f"Mức độ tương thích: **{level}** ({score:.1f}%).\n\n"
        
        # Skill match info
        if matched_skills:
            skills_shown = matched_skills[:5]
            explanation += f"• **Điểm mạnh:** Bạn sở hữu các kỹ năng rất phù hợp bao gồm: *{', '.join(skills_shown)}*.\n"
        
        # Location info
        if loc_score >= 0.9:
            explanation += f"• **Địa điểm:** Hoàn toàn thuận lợi, cả bạn và công ty đều ở cùng khu vực hoặc lân cận (*{job_city}*).\n"
        elif loc_score >= 0.8:
            explanation += f"• **Địa điểm:** Rất thuận lợi, khoảng cách di chuyển giữa *{user_city}* và *{job_city}* rất tối ưu.\n"
        elif loc_score <= 0.3:
            explanation += f"• **Địa điểm:** Khác biệt địa lý lớn (bạn ở *{user_city}*, công ty ở *{job_city}*). Nên cân nhắc hình thức làm việc từ xa (Remote) nếu có.\n"

        def _format_years(years):
            try:
                years = float(years)
            except Exception:
                return None
            if years <= 0:
                return None
            if years < 1:
                months = round(years * 12)
                return f"{months} tháng"
            if years.is_integer():
                return f"{int(years)} năm"
            return f"{years:.1f} năm"

        def _format_exp_value(value):
            if value is None:
                return None
            try:
                value = float(value)
            except Exception:
                return str(value)
            return "không yêu cầu" if value == 0 else _format_years(value)

        user_exp_label = _format_exp_value(user_exp) or f"{user_exp} năm"
        job_exp_min_label = _format_exp_value(job_exp_min) or f"{job_exp_min} năm"

        def _build_experience_phrase(user_label, job_label):
            if user_label == "không yêu cầu" and job_label == "không yêu cầu":
                return "Công việc này không yêu cầu kinh nghiệm."

            user_phrase = (
                "Bạn chưa có kinh nghiệm" if user_label == "không yêu cầu" else f"Bạn có {user_label}"
            )
            job_phrase = (
                "công việc không yêu cầu kinh nghiệm" if job_label == "không yêu cầu" else f"công việc yêu cầu {job_label}"
            )
            return f"{user_phrase}, {job_phrase}."

        # Experience info
        if exp_score >= 1.0:
            experience_phrase = _build_experience_phrase(user_exp_label, job_exp_min_label)
            explanation += f"• **Kinh nghiệm:** Bạn hoàn toàn đáp ứng hoặc vượt mức kinh nghiệm tối thiểu yêu cầu ({experience_phrase})\n"
        elif exp_score >= 0.7:
            explanation += f"• **Kinh nghiệm:** Bạn hơi thiếu một chút kinh nghiệm nhưng hoàn toàn có thể bù đắp bằng nền tảng kỹ năng sẵn có.\n"
        else:
            job_requirement = "không yêu cầu kinh nghiệm" if job_exp_min_label == "không yêu cầu" else f"{job_exp_min_label}"
            explanation += f"• **Kinh nghiệm:** Yêu cầu tối thiểu là {job_requirement} kinh nghiệm, bạn cần chứng minh thêm năng lực thực tế để bù đắp khoảng trống này.\n"

        explanation += f"\n👉 **Lời khuyên:** {recommend}"
        return explanation
    @staticmethod
    def _experience_match_score(candidate_years, exp_min, under_decay=0.15, min_score=0.2):
        """Calculate score based on experience requirements with decay"""
        try:
            candidate_years = float(candidate_years)
            exp_min = float(exp_min)
            if candidate_years >= exp_min:
                return 1.0
            distance = exp_min - candidate_years
            score = 1.0 - under_decay * distance
            return max(min_score, score)
        except:
            return 0.5

    @staticmethod
    def _normalize_skill_name(skill: str) -> str:
        """Normalize skill name for comparison: lowercase, remove dots and special chars.
        Examples: React.js -> reactjs, Node.js -> nodejs, RESTful API -> restful api
        """
        import re
        s = skill.lower().strip()
        s = re.sub(r'\.js$', '', s)      # react.js -> react
        s = re.sub(r'\.net$', 'net', s)  # .net -> net
        s = re.sub(r'[^\w\s\+#\-]', '', s) # Keep C++, C#, hyphens, remove other special chars
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    @staticmethod
    def _extract_skills_from_tech(tech_text: str) -> set:
        """Extract skills from text by simple split to avoid regex errors"""
        if not tech_text:
            return set()
        # Just split by common delimiters
        import re
        parts = re.split(r'[,\n\r•|/]', tech_text)
        return set([MatchingService._normalize_skill_name(p) for p in parts if p.strip()])

    @staticmethod
    def _normalize_city(loc: str) -> str:
        """Normalize location string to canonical city name for matrix lookup"""
        t = str(loc or "").lower().strip()
        # Map common variations
        city_map = {
            "tp.hcm": "tp.hcm", "tp hcm": "tp.hcm", "tphcm": "tp.hcm",
            "hồ chí minh": "tp.hcm", "ho chi minh": "tp.hcm",
            "sài gòn": "tp.hcm", "sai gon": "tp.hcm", "hcm": "tp.hcm",
            "hà nội": "hà nội", "ha noi": "hà nội", "hanoi": "hà nội",
            "đà nẵng": "đà nẵng", "da nang": "đà nẵng",
            "hải dương": "hải dương", "hai duong": "hải dương",
            "hưng yên": "hưng yên", "hung yen": "hưng yên",
            "hải phòng": "hải phòng", "hai phong": "hải phòng",
            "bình dương": "bình dương", "binh duong": "bình dương",
            "đồng nai": "đồng nai", "dong nai": "đồng nai",
            "cần thơ": "cần thơ", "can tho": "cần thơ",
        }
        # Direct match
        if t in city_map:
            return city_map[t]
        # Check if any known city name is contained in the string (e.g. "Bình Thạnh, Hồ Chí Minh")
        for key, val in city_map.items():
            if key in t:
                return val
        return t

    @staticmethod
    def _get_location_score(user_loc, job_loc):
        """Calculate location score using distance matrix"""
        matrix = {
            "hưng yên": {"hưng yên": 1.00, "hải dương": 0.88, "hà nội": 0.92, "tp.hcm": 0.20, "đà nẵng": 0.45, "hải phòng": 0.80},
            "hà nội":   {"hà nội": 1.00, "hưng yên": 0.92, "hải dương": 0.90, "tp.hcm": 0.18, "đà nẵng": 0.50, "hải phòng": 0.85},
            "hải dương": {"hải dương": 1.00, "hưng yên": 0.88, "hà nội": 0.90, "tp.hcm": 0.20, "đà nẵng": 0.45, "hải phòng": 0.82},
            "tp.hcm":   {"tp.hcm": 1.00, "hưng yên": 0.20, "hà nội": 0.18, "đà nẵng": 0.55, "bình dương": 0.90, "đồng nai": 0.88},
            "đà nẵng":  {"đà nẵng": 1.00, "hưng yên": 0.45, "hà nội": 0.50, "tp.hcm": 0.55},
            "hải phòng": {"hải phòng": 1.00, "hà nội": 0.85, "hưng yên": 0.80, "hải dương": 0.82},
            "bình dương": {"bình dương": 1.00, "tp.hcm": 0.90, "đồng nai": 0.80},
            "đồng nai": {"đồng nai": 1.00, "tp.hcm": 0.88, "bình dương": 0.80},
        }
        u = MatchingService._normalize_city(user_loc)
        j = MatchingService._normalize_city(job_loc)
        if not u or not j:
            return 0.5
        if u == j:
            return 1.0
        return matrix.get(u, {}).get(j, 0.5)

    @staticmethod
    def _generate_skill_suggestions(missing_skills: List[str], job_title: str) -> List[str]:
        """Generate specific Vietnamese skill improvement suggestions based on missing skills and job type"""
        suggestions = []
        job_title_lower = job_title.lower()
        
        for skill in missing_skills[:5]:  # Limit to top 5 missing skills
            skill_lower = skill.lower().strip()
            
            # Programming languages
            if skill_lower in ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'golang', 'ruby', 'rust']:
                suggestions.append(f"Học ngôn ngữ {skill} thông qua các khóa học trực tuyến (Coursera, Udemy) hoặc luyện giải thuật trên HackerRank/LeetCode.")
            
            # Frontend/Backend Frameworks
            elif skill_lower in ['react', 'vue', 'angular', 'django', 'flask', 'fastapi', 'spring boot', 'laravel', 'nest.js', 'next.js', 'express']:
                suggestions.append(f"Xây dựng dự án thực tế sử dụng framework {skill} để làm phong phú portfolio của bạn.")
            
            # Cloud & DevOps
            elif skill_lower in ['aws', 'azure', 'docker', 'kubernetes', 'ci/cd', 'git', 'jenkins']:
                suggestions.append(f"Tìm hiểu công nghệ {skill} qua tài liệu chính hãng (documentation) và thực hành trên môi trường đám mây miễn phí.")
            
            # Databases
            elif skill_lower in ['mongodb', 'postgresql', 'mysql', 'redis', 'oracle', 'sql server']:
                suggestions.append(f"Thực hành thiết kế cơ sở dữ liệu và tối ưu hóa các câu truy vấn phức tạp với {skill}.")
            
            # AI & Data Science
            elif skill_lower in ['machine learning', 'data science', 'deep learning', 'nlp', 'pytorch', 'tensorflow', 'scikit-learn']:
                suggestions.append(f"Tham gia các khóa học chuyên sâu về {skill} và thử sức với các bài toán thực tế trên Kaggle.")
            
            # Mobile Development
            elif skill_lower in ['flutter', 'react native', 'ios', 'android', 'swift', 'kotlin']:
                suggestions.append(f"Thực hành viết các ứng dụng di động nhỏ bằng {skill} và phát hành lên Google Play hoặc App Store.")
            
            # Default suggestion
            else:
                suggestions.append(f"Nghiên cứu nâng cao kỹ năng {skill} thông qua các bài viết kỹ thuật, tutorial và thực hành trực tiếp.")
        
        # Add job-specific advice
        if 'backend' in job_title_lower or 'api' in job_title_lower:
            suggestions.append("Nâng cao hiểu biết về API design (REST, GraphQL), tối ưu hóa database và các kiến thức hệ thống phân tán.")
        elif 'frontend' in job_title_lower or 'ui' in job_title_lower:
            suggestions.append("Luyện tập xây dựng giao diện responsive, tương thích đa thiết bị và nâng cao tư duy UI/UX của người dùng.")
        elif 'fullstack' in job_title_lower:
            suggestions.append("Cân bằng cả kỹ năng frontend và backend, đồng thời làm quen với quy trình deploy/DevOps hoàn chỉnh.")
        elif 'data' in job_title_lower:
            suggestions.append("Tập trung vào phân tích dữ liệu, trực quan hóa thông tin và các phương pháp thống kê nâng cao.")
        
        return suggestions[:6]  # Return max 6 suggestions
