import re
from typing import List, Set

# Basic stop words (Vietnamese & English) to filter out non-important words
STOP_WORDS = {
    "và", "của", "là", "các", "cho", "trong", "để", "với", "tại", "như", "này", "được", "những", "một",
    "the", "and", "of", "to", "in", "for", "with", "on", "at", "by", "from", "up", "about", "into", "over", "after",
    "có", "khi", "ra", "vào", "lại", "thì", "mà", "nên", "hơn", "còn", "nhiều", "ít", "đã", "đang", "sẽ",
    "must", "have", "been", "was", "were", "is", "are", "do", "does", "did", "can", "could", "should", "would",
    "kinh", "nghiệm", "kỹ", "năng", "yêu", "cầu", "công", "việc", "trách", "nhiệm", "quyền", "lợi",
    "experience", "skill", "skills", "job", "work", "required", "requirements", "knowledge", "ability"
}

def clean_text(text: str) -> str:
    """Basic text cleaning for keyphrase extraction"""
    text = text.lower()
    # Remove special characters but keep some punctuation for splitting
    text = re.sub(r'[^a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s\-/]', ' ', text)
    return text

def extract_skills(text: str) -> List[str]:
    """
    Domain-agnostic keyphrase extraction.
    Identifies potential skills/concepts without a hardcoded dictionary.
    """
    cleaned_text = clean_text(text)
    words = cleaned_text.split()
    
    # Identify potential technical/specialized terms
    # Strategy: Look for capitalized words (original text) or technical symbols
    # Since we lowercased, let's use an N-gram approach with frequency
    
    candidates = []
    # Simple strategy: collect sequences of 1-3 words that don't start/end with stop words
    for i in range(len(words)):
        for n in range(1, 4): # 1 to 3-grams
            if i + n <= len(words):
                ngram = words[i:i+n]
                # Filter out ngrams starting or ending with stop words
                if ngram[0] not in STOP_WORDS and ngram[-1] not in STOP_WORDS:
                    # Filter out short or numeric-only terms
                    phrase = " ".join(ngram)
                    if len(phrase) > 2 and not phrase.isdigit():
                        candidates.append(phrase)
    
    # Score candidates by frequency (basic implementation)
    counts = {}
    for c in candidates:
        counts[c] = counts.get(c, 0) + 1
    
    # Sort by frequency and return top unique keyphrases
    sorted_keyphrases = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    # Return top keyphrases (limited to avoid noise)
    return [kp[0] for kp in sorted_keyphrases if kp[1] >= 1][:20]

def match_skills_semantically(cv_skills: List[str], job_skills: List[str]) -> tuple[List[str], List[str]]:
    """
    Optional: Use semantic similarity to match skills instead of exact strings.
    For now, we'll use a basic intersection but can be expanded with embeddings.
    """
    cv_set = set(cv_skills)
    job_set = set(job_skills)
    
    matched = list(cv_set.intersection(job_set))
    missing = list(job_set.difference(cv_set))
    
    return matched, missing
