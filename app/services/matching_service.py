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
            # 0. Fetch all jobs from cache to enrich data (like salary)
            all_jobs = await JobService.fetch_all_jobs()
            job_lookup = {str(j.id): j for j in all_jobs}

            # 1. Extract structured fields from CV
            cv_sections = parse_cv_sections(cv_data.raw_text)
            # logger.info(f"CV parsed - Title: '{cv_sections.title}', Location: '{cv_sections.location}', Exp: {cv_sections.years_of_experience}yr")

            # 2. Create 3 separate vectors from CV parts corresponding to JD
            title_text = clean_text(cv_sections.title or cv_data.raw_text[:200])
            tech_text = clean_text(cv_sections.tech or cv_data.raw_text)
            mota_text = clean_text(cv_sections.mota or cv_data.raw_text)

            title_vector = embed_text(title_text).tolist()
            tech_vector = embed_text(tech_text).tolist()
            mota_vector = embed_text(mota_text).tolist()

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

            title_hits = search_field("title_vec", title_vector)
            tech_hits = search_field("tech_vec", tech_vector)
            mota_hits = search_field("mota_vec", mota_vector)
            
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

            # 4. Apply Heuristic Scoring
            # Use weights from configuration
            weights = {
                "title": settings.WEIGHT_TITLE,
                "tech": settings.WEIGHT_TECH,
                "mota": settings.WEIGHT_MOTA,
                "loc": settings.WEIGHT_LOCATION,
                "exp": settings.WEIGHT_EXPERIENCE
            }
            
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
                
                final_score = (
                    weights["title"] * c["sim_title"] +
                    weights["tech"] * c["sim_tech"] +
                    weights["mota"] * c["sim_mota"] +
                    weights["loc"] * loc_score +
                    weights["exp"] * exp_score
                )
                
                if final_score * 100 < 0: 
                    continue
                
                # Generate suggestions if skills are missing
                suggestions = MatchingService._generate_skill_suggestions(missing_skills, c["job_title"])
                
                # Retrieve salary and url_source from job_lookup
                full_job = job_lookup.get(str(jid))
                job_salary = full_job.salary if full_job else None
                job_url = full_job.url_source if full_job else None

                match_results.append(MatchResult(
                    job_id=str(jid),
                    job_title=c["job_title"],
                    company=c["company_name"],
                    compatibility_score=round(final_score * 100, 1),
                    matched_skills=matched_skills,
                    missing_skills=missing_skills,
                    skill_improvement_suggestions=suggestions,
                    location=c["location_city"],
                    match_explanation=f"Match score based on: Title ({c['sim_title']:.2f}), Tech ({c['sim_tech']:.2f}), Location ({loc_score:.2f})",
                    salary=job_salary,
                    url_source=job_url,
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
    def _generate_explanation(fit_analysis: dict) -> str:
        """Generate human-readable explanation for match with recommendations"""
        score = fit_analysis["fit_score"]
        matched_skills = len(fit_analysis["cv_skills"])
        missing_skills = len(fit_analysis["missing_skills"])
        
        # Match level assessment
        if score >= 0.8:
            level = "Excellent"
            recommendation = "Highly recommended to apply!"
        elif score >= 0.6:
            level = "Good"
            recommendation = "Good fit, consider applying."
        elif score >= 0.4:
            level = "Fair"
            recommendation = "Moderate fit, may need skill improvement."
        else:
            level = "Poor"
            recommendation = "Low compatibility, significant skill gaps."
        
        # Build explanation with actionable insights
        explanation = f"{level} match ({score*100:.1f}% compatibility). "
        
        if matched_skills > 0:
            explanation += f"You have {matched_skills} relevant skills. "
        
        if missing_skills > 0:
            explanation += f"Consider developing {missing_skills} additional skills to improve your chances. "
        else:
            explanation += "You meet all technical requirements! "
        
        explanation += recommendation
        
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
        s = re.sub(r'\.', '', s)        # React.js -> reactjs
        s = re.sub(r'[^\w\s]', '', s)   # Remove remaining special chars
        s = ' '.join(s.split())          # Collapse extra spaces
        return s

    @staticmethod
    def _extract_skills_from_tech(tech_text: str) -> set:
        """Extract skills directly from CV SKILLS section using split approach.
        More accurate than N-gram for structured skill lists (comma/newline separated).
        """
        import re
        if not tech_text:
            return set()
        normalized = set()
        # Split by comma, newline, bullet points, pipes
        raw_items = re.split(r'[,\n\r•|/]', tech_text)
        for item in raw_items:
            item = item.strip()
            if item and 1 < len(item) <= 50:
                normalized.add(MatchingService._normalize_skill_name(item))
        return normalized

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
        """Generate specific skill improvement suggestions based on missing skills and job type"""
        suggestions = []
        
        # Job type specific recommendations
        job_title_lower = job_title.lower()
        
        for skill in missing_skills[:5]:  # Limit to top 5 missing skills
            skill_lower = skill.lower()
            
            # Programming languages
            if skill_lower in ['python', 'java', 'javascript', 'typescript']:
                suggestions.append(f"Learn {skill} through online courses (Coursera, Udemy) or practice on HackerRank")
            
            # Frameworks
            elif skill_lower in ['react', 'vue', 'angular', 'django', 'flask', 'fastapi']:
                suggestions.append(f"Build projects using {skill} framework and create a portfolio")
            
            # Cloud & DevOps
            elif skill_lower in ['aws', 'azure', 'docker', 'kubernetes']:
                suggestions.append(f"Get {skill} certification and practice with free tier/tutorials")
            
            # Databases
            elif skill_lower in ['mongodb', 'postgresql', 'mysql']:
                suggestions.append(f"Practice {skill} database design and queries through online labs")
            
            # General skills
            elif skill_lower in ['machine learning', 'data science']:
                suggestions.append(f"Take {skill} courses and work on Kaggle competitions")
            
            # Default suggestion
            else:
                suggestions.append(f"Develop {skill} skills through online resources and hands-on projects")
        
        # Add job-specific advice
        if 'backend' in job_title_lower or 'api' in job_title_lower:
            suggestions.append("Focus on API design, database optimization, and system architecture")
        elif 'frontend' in job_title_lower or 'ui' in job_title_lower:
            suggestions.append("Build responsive web applications and improve UI/UX design skills")
        elif 'fullstack' in job_title_lower:
            suggestions.append("Balance both frontend and backend skills, learn modern development workflows")
        elif 'data' in job_title_lower:
            suggestions.append("Practice data analysis, visualization, and statistical modeling")
        
        return suggestions[:6]  # Return max 6 suggestions
