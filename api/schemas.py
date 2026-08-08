class QuestionRequest:
    question: str
    context: str = None

class AnswerResponse:
    answer: str
    confidence: float
