# app/services/ml/embeddings.py

import os
from sentence_transformers import SentenceTransformer
from app.core.config import settings

_model = None

def get_model():
    """Get or initialize the sentence transformer model"""
    global _model
    if _model is None:
        # Use HF_TOKEN if available for authenticated requests
        if settings.HF_TOKEN:
            os.environ["HF_TOKEN"] = settings.HF_TOKEN
        _model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    return _model


def embed_text(text: str | list[str]):
    """Generate normalized embedding vector(s) for text or list of texts"""
    model = get_model()
    # Truncate long text to improve speed and stay within model limits (max 512-1024 tokens)
    if isinstance(text, str):
        text = text[:1500] # Limit to ~300-400 words for speed
    elif isinstance(text, list):
        text = [t[:1500] for t in text]
        
    return model.encode(text, normalize_embeddings=True)