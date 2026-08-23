class QAService:
    def __init__(self, model):
        self.model = model

    def get_answer(self, question):
        return self.model.answer(question)

    def batch_answer(self, questions):
        return [self.get_answer(q) for q in questions]
