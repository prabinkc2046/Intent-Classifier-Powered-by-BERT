from sklearn.preprocessing import LabelEncoder
from load_data import load_data
data = load_data("./full-intent-data.csv")

def encode_label(data, column_label="intent"):
    encoder = LabelEncoder()
    data[column_label] = encoder.fit_transform(data[column_label])
    print(encoder.classes_)
    return data, encoder

encode_label(data)
