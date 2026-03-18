# app/matcher.py

import os
from sklearn.metrics.pairwise import cosine_similarity
from app.embedding import embed_text
from app.config import JD_FOLDER
from app.logger import logger
from app.fit_analyzer import analyze_fit

def load_jds():
    jds = {}
    try:
        for file in os.listdir(JD_FOLDER):
            if file.endswith(".txt"):
                path = os.path.join(JD_FOLDER, file)
                with open(path, "r", encoding="utf-8") as f:
                    jds[file.replace(".txt", "")] = f.read()
        logger.info(f"Loaded {len(jds)} job descriptions from {JD_FOLDER}")
        return jds
    except Exception as e:
        logger.error(f"Error loading job descriptions: {str(e)}")
        return {}


def match_cv_to_jds(cv_text: str, top_k: int = 3):
    logger.info(f"Starting CV matching - CV text length: {len(cv_text)} characters")
    
    jds = load_jds()
    if not jds:
        logger.error("No job descriptions available for matching")
        return []

    results = []

    for job_name, jd_text in jds.items():
        # Use fit analyzer for detailed analysis
        fit_analysis = analyze_fit(cv_text, jd_text)
        
        results.append({
            "job": job_name,
            "score": fit_analysis["fit_score"],
            "cv_skills": fit_analysis["cv_skills"],
            "missing_skills": fit_analysis["missing_skills"],
            "fit_reason": f"Similarity score: {fit_analysis['fit_score']}. Found {len(fit_analysis['cv_skills'])} matching skills. Missing {len(fit_analysis['missing_skills'])} required skills."
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    top_results = results[:top_k]
    
    logger.info(f"Top {len(top_results)} matches: {[(r['job'], r['score']) for r in top_results]}")
    return top_results
