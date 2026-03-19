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
import logging

logger = logging.getLogger(__name__)

class MatchingService:
    """Service for CV-Job matching"""
    
    @staticmethod
    async def match_cv_to_jobs(
        cv_data: CVParsedData, 
        match_params: MatchRequest
    ) -> MatchResponse:
        """
        Match CV to jobs with filtering and scoring
        """
        start_time = time.time()
        
        try:
            # Fetch all jobs from MongoDB
            all_jobs = await JobService.fetch_all_jobs()
            
            if not all_jobs:
                return MatchResponse(
                    success=False,
                    matches=[],
                    total_jobs_analyzed=0,
                    processing_time_ms=0,
                    message="No jobs available for matching"
                )
            
            # Apply filters
            filtered_jobs = MatchingService._apply_filters(all_jobs, match_params)
            
            # Calculate matches
            matches = []
            
            for job in filtered_jobs:
                try:
                    # Use pre-computed embedding if available, otherwise compute on-the-fly
                    job_embedding = job.embedding if hasattr(job, 'embedding') and job.embedding else None
                    
                    if job_embedding:
                        # Fast path: use pre-computed embedding
                        from sklearn.metrics.pairwise import cosine_similarity
                        import numpy as np
                        
                        cv_embedding = embed_text(cv_data.raw_text)
                        similarity_score = cosine_similarity(
                            [cv_embedding], 
                            [np.array(job_embedding)]
                        )[0][0]
                        
                        # Extract skills for analysis
                        cv_skills = extract_skills(cv_data.raw_text)
                        jd_skills = extract_skills(MatchingService._job_to_text(job))
                        missing_skills = list(set(jd_skills) - set(cv_skills))
                        
                        fit_analysis = {
                            "fit_score": float(similarity_score),
                            "cv_skills": cv_skills,
                            "missing_skills": missing_skills
                        }
                    else:
                        # Slow path: compute embedding on-the-fly
                        fit_analysis = analyze_fit(cv_data.raw_text, 
                                                 MatchingService._job_to_text(job))
                    
                    # Skip if score is below threshold
                    if fit_analysis["fit_score"] < match_params.min_score:
                        continue
                    
                    # Generate skill improvement suggestions
                    suggestions = MatchingService._generate_skill_suggestions(
                        fit_analysis["missing_skills"], 
                        job.title
                    )
                    
                    # Create match result with enhanced recommendations
                    match_result = MatchResult(
                        job_id=job.id,
                        job_title=job.title,
                        company=job.company,
                        compatibility_score=round(fit_analysis["fit_score"] * 100, 1),
                        matched_skills=fit_analysis["cv_skills"],
                        missing_skills=fit_analysis["missing_skills"],
                        skill_improvement_suggestions=suggestions,
                        match_explanation=MatchingService._generate_explanation(fit_analysis),
                        location=job.location,
                        salary=job.salary
                    )
                    
                    matches.append(match_result)
                    
                except Exception as e:
                    logger.warning(f"Failed to analyze job {job.id}: {str(e)}")
                    continue
            
            # Sort by score and limit results
            matches.sort(key=lambda x: x.compatibility_score, reverse=True)
            top_matches = matches[:match_params.top_k]
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            return MatchResponse(
                success=True,
                matches=top_matches,
                total_jobs_analyzed=len(filtered_jobs),
                processing_time_ms=round(processing_time, 2),
                message=f"Successfully matched CV to {len(top_matches)} jobs"
            )
            
        except Exception as e:
            logger.error(f"Matching failed: {str(e)}")
            processing_time = (time.time() - start_time) * 1000
            
            return MatchResponse(
                success=False,
                matches=[],
                total_jobs_analyzed=0,
                processing_time_ms=round(processing_time, 2),
                message=f"Matching failed: {str(e)}"
            )
    
    @staticmethod
    def _apply_filters(jobs: List[JobSchema], params: MatchRequest) -> List[JobSchema]:
        """Apply filters to job list"""
        filtered_jobs = jobs
        
        # Filter by location
        if params.filter_location:
            filtered_jobs = [
                job for job in filtered_jobs 
                if params.filter_location.lower() in (job.location or "").lower()
            ]
        
        # Filter by skills
        if params.filter_skills:
            filtered_jobs = [
                job for job in filtered_jobs
                if any(skill.lower() in [s.lower() for s in job.skills] 
                      for skill in params.filter_skills)
            ]
        
        return filtered_jobs
    
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