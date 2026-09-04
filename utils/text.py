import re

def normalize_text(text):
    return text.strip()

def word_tokenize(text):
    return text.split()

def sentence_split(text):
    return re.split(r'[.!?]', text)
