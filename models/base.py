class BaseModel:
    def __init__(self):
        pass

    def predict(self, x):
        raise NotImplementedError

    def train(self, data):
        raise NotImplementedError

    def save(self, path):
        pass

    def load(self, path):
        pass
