"""Text cleaning shared by ETL, training and the Streamlit app."""
import re
 
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
 
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
 
 
def clean_text(text: str) -> str:
    """Lowercase, mask URLs, drop punctuation and English stop words."""
    text = str(text).lower()
    text = URL_RE.sub(" urltoken ", text)
    text = NON_ALNUM_RE.sub(" ", text)
    tokens = [t for t in text.split() if t not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)
