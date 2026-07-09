from transformers import AutoModel, AutoTokenizer

class Embedder:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True)
        return self.model(**inputs).last_hidden_state.mean(1)
