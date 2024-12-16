import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    print(data.head())
    return data

load_data("./full-intent-data.csv")