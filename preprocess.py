import pandas as pd
from scipy.stats import zscore
import numpy as np
from pprint import pprint 

# Read the file
df = pd.read_csv("data/airbnbDataset.csv")

# Check for NaN
print(df.isna().mean())

print("Calculating z-score\n")
# Calculating Z-scores of the dataset
z_scores = np.abs(zscore(df.select_dtypes(include=np.number)))
outliers = (z_scores > 3)

# Counting the total number of rows removed
total_rows_removed = outliers.any(axis=1).sum()
print(f"Total rows removed: {total_rows_removed}")

# Removing rows with outliers
data_clean = df[~outliers.any(axis=1)]

# Counting the number of rows removed per column
rows_removed_per_column = outliers.sum(axis=0)
print("Rows removed per column:\n")
pprint(dict(rows_removed_per_column))

print("Data after removing outliers:\n")
print(data_clean)

# Dropping columns
columns_to_drop = ['Restraunt Index', 'Normalised Restraunt Index']

# Drop the specified columns
data_clean = data_clean.drop(columns=columns_to_drop)

# Print the DataFrame after dropping columns
print(data_clean)

# Checking unique values for encoding
columns_to_check = ['City', 'Day', 'Room Type']

for column in columns_to_check:
    unique_values = data_clean[column].unique()
    print(f"Unique values in {column}:\n{unique_values}\n")

# Use one-hot encoding as none of the columns are ordinal
data_clean = pd.get_dummies(data_clean, columns=['City', 'Day', 'Room Type'], prefix=['City', 'Day', 'Room_Type'])
print(data_clean)

