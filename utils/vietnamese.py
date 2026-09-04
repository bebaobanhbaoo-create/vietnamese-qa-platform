import re
from typing import List

def normalize_vietnamese(text: str) -> str:
    """Normalize Vietnamese text for better matching."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def remove_punctuation(text: str) -> str:
    """Remove punctuation while keeping Vietnamese characters."""
    return re.sub(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', '', text, flags=re.IGNORECASE)

def sentence_tokenize(text: str) -> List[str]:
    """Split Vietnamese text into sentences."""
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]
