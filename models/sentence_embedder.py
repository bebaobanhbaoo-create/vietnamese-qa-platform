from sentence_transformers import SentenceTransformer

class SentenceEmbedder:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(texts)

    def encode_batch(self, texts, batch_size=32):
        return self.model.encode(texts, batch_size=batch_size)
