import faiss
import numpy as np

class VectorIndex:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)

    def add(self, vectors):
        self.index.add(np.array(vectors).astype('float32'))

    def search(self, query, k=5):
        D, I = self.index.search(np.array([query]).astype('float32'), k)
        return I[0], D[0]

    def save(self, path):
        faiss.write_index(self.index, path)

    def load(self, path):
        self.index = faiss.read_index(path)
