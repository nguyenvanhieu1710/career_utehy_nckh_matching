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


def embed_text(text: str):
    """Generate embedding vector for text"""
    model = get_model()
    return model.encode(text)