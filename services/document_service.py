class DocumentService:
    def __init__(self):
        self.documents = []

    def add(self, doc):
        self.documents.append(doc)
        return len(self.documents) - 1

    def get(self, doc_id):
        return self.documents[doc_id]

    def search(self, query):
        return [d for d in self.documents if query.lower() in d.lower()]
