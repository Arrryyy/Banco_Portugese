# feature_engineering.py

import pandas as pd
import os

# Paths
input_path = "/Users/arrryyy/Desktop/bank/data/bank-full.csv"
output_path = "/Users/arrryyy/Desktop/bank/data/processed/bank_full_encoded.csv"

# Load the original dataset
df = pd.read_csv(input_path, sep=';')

# Keep only selected features and target
features = [
    'age', 'job', 'marital', 'education', 'default', 
    'balance', 'housing', 'loan'
]
df = df[features + ['y']]

# Encode categorical features
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Save the encoded dataset
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_encoded.to_csv(output_path, index=False)

print(f"Processed dataset saved to {output_path}")