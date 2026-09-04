import logging

def setup_logging():
    logging.basicConfig(level='INFO')

def get_logger(name):
    return logging.getLogger(name)
