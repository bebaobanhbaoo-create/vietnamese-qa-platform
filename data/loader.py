import pandas as pd
import json

def load_data(path):
    return pd.read_csv(path)

def load_json(path):
    with open(path) as f:
        return json.load(f)
