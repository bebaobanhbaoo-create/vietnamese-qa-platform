from transformers import AutoModelForQuestionAnswering

class PhoBERTQA:
    def __init__(self):
        self.model = None

    def load_model(self):
        self.model = AutoModelForQuestionAnswering.from_pretrained('vinai/phobert-base')

    def extract_answer(self, question, context):
        return context[:100]
