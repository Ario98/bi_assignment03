import pandas as pd
import numpy as np
from scipy.stats import zscore
from pprint import pprint

df = pd.read_csv("data/airbnbDataset.csv")

missing_values = df.isnull().sum()
print("Missing Values\n",missing_values)

data_clean = pd.get_dummies(df, columns=['Day', 'City', 'Room Type'])

print("Calculating z-score\n")
# Calculating Z-scores of the dataset
z_scores = np.abs(zscore(data_clean.select_dtypes(include=np.number)))
outliers = (z_scores > 3).any(axis=1)
outliers_summary = outliers.value_counts()
print(outliers_summary)
# Identifying specific columns with outliers
outlier_columns = data_clean.select_dtypes(include=np.number).columns
outlier_indices = np.where(z_scores > 3)[1]

# Counting outliers in each column
outlier_counts_per_column = {}
for idx in outlier_indices:
    col_name = outlier_columns[idx]
    if col_name in outlier_counts_per_column:
        outlier_counts_per_column[col_name] += 1
    else:
        outlier_counts_per_column[col_name] = 1

print("Outliers per column\n")
pprint(outlier_counts_per_column)
