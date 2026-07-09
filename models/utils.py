import torch

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed=42):
    torch.manual_seed(seed)
