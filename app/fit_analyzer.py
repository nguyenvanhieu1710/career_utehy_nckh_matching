# app/fit_analyzer.py

from app.skill_extractor import extract_skills
from app.embedding import embed_text
from sklearn.metrics.pairwise import cosine_similarity

def analyze_fit(cv_text: str, jd_text: str):
    cv_embedding = embed_text(cv_text)
    jd_embedding = embed_text(jd_text)

    similarity_score = cosine_similarity(
        [cv_embedding], [jd_embedding]
    )[0][0]

    cv_skills = extract_skills(cv_text)
    jd_skills = extract_skills(jd_text)

    missing_skills = list(set(jd_skills) - set(cv_skills))

    return {
        "fit_score": round(float(similarity_score), 4),
        "cv_skills": cv_skills,
        "missing_skills": missing_skills
    }
