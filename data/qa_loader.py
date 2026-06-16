import json

def load_qa_pairs(path):
    with open(path) as f:
        return json.load(f)

def save_qa_pairs(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False)
