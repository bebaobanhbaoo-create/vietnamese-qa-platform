def validate_question(q):
    return q and len(q) >= 5

def validate_document(doc):
    return 'title' in doc and 'content' in doc
