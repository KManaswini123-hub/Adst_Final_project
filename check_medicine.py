import pandas as pd

data = pd.read_csv("Medicine_Details.csv")

print("Columns:")
print(data.columns.tolist())

print("\nFirst 5 rows:")
print(data.head())