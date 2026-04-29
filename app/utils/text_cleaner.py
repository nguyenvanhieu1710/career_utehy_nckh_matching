# app/utils/text_cleaner.py

import re
import pandas as pd
from bs4 import BeautifulSoup
from typing import Any

def clean_text(text: Any) -> str:
    """
    Clean text by removing HTML tags and normalizing whitespace.
    """
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()
