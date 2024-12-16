import pandas as pd

# Load the dataset
df = pd.read_csv("./cleanedData.csv", header=None, names=["text", "intent"])

# Print the dataframe
print(df)

# Check for issues
print(df.isnull().sum())  # Ensure no missing values
