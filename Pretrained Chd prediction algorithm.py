import pandas as pd

# Load the dataset
file_path = r'C:\Users\HP\CHd prediction llm\MGH_PredictionDataSet.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
data.head(), data.info(), data.describe()
