import os

class Config:
    DEBUG = True
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev')
    MODEL_PATH = 'models/'
    DATA_PATH = 'data/'
    MODEL_NAME = 'vinai/phobert-base'
    EMBEDDING_DIM = 768
    MAX_LENGTH = 256
    BATCH_SIZE = 32
    TOP_K = 5
