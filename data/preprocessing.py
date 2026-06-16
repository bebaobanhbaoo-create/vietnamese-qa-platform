import re

def preprocess_text(text):
    return text.strip().lower()

def clean_text(text):
    return re.sub(r'[^\w\s]', '', text)

def normalize_vietnamese(text):
    return text.strip()

def tokenize(text):
    return text.split()
