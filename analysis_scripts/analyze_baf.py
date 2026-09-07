import pandas as pd
import numpy as np

print('=== BAF DATASET - FULL ANALYSIS ===')
base_path = r'Datasets\BAF\Base.csv'

df = pd.read_csv(base_path)
print(f'Shape: {df.shape}')
print(f'\nDtypes:')
print(df.dtypes.to_string())
print(f'\nMissing values:')
print(df.isnull().sum().to_string())
print(f'\nDuplicate rows: {df.duplicated().sum()}')
print(f'\nBasic stats:')
print(df.describe().to_string())

fraud_col = 'fraud_bool'
print(f'\nFraud distribution:')
print(df[fraud_col].value_counts().to_string())
fraud_count = df[fraud_col].sum()
legit_count = len(df) - fraud_count
print(f'Fraud rate: {df[fraud_col].mean():.4%}')
print(f'Fraud: {fraud_count}, Legit: {legit_count}')
print(f'Class ratio (legit:fraud): {legit_count/fraud_count:.1f}:1')

print(f'\nMemory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB')

print(f'\nColumn-by-column detail:')
for col in df.columns:
    uv = df[col].nunique()
    miss = df[col].isnull().sum()
    sample = df[col].dropna().unique()[:5].tolist()
    print(f'  {col}: n_unique={uv}, missing={miss}, sample={sample}')

# Temporal analysis - 'month' column
print(f'\nMonth distribution:')
print(df['month'].value_counts().sort_index().to_string())
print(f'\nFraud rate by month:')
print(df.groupby('month')[fraud_col].mean().to_string())
print(f'\nTransaction count by month:')
print(df.groupby('month').size().to_string())

# Categorical analysis
cat_cols = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
for col in cat_cols:
    if col in df.columns:
        print(f'\n{col} distribution:')
        print(df[col].value_counts().to_string())
        print(f'Fraud rate by {col}:')
        print(df.groupby(col)[fraud_col].mean().sort_values(ascending=False).to_string())

# Amount-related
print(f'\nAmount-related columns (intended_balcon_amount):')
print(df['intended_balcon_amount'].describe().to_string())
print(f'Negative values in intended_balcon_amount: {(df["intended_balcon_amount"] < 0).sum()}')

# Velocity features analysis
vel_cols = ['velocity_6h', 'velocity_24h', 'velocity_4w']
for col in vel_cols:
    print(f'\n{col} stats:')
    print(df.groupby(fraud_col)[col].describe().to_string())

# Variants - check headers only
import os
variants = ['Variant I.csv', 'Variant II.csv', 'Variant III.csv', 'Variant IV.csv', 'Variant V.csv']
print(f'\n=== BAF VARIANT HEADERS ===')
for v in variants:
    vpath = f'Datasets/BAF/{v}'
    vdf = pd.read_csv(vpath, nrows=5)
    print(f'\n{v}: shape columns={len(vdf.columns)}, columns={vdf.columns.tolist()}')
    # Check if same columns as base
    if vdf.columns.tolist() == df.columns.tolist():
        print(f'  -> SAME columns as Base')
    else:
        diff = set(vdf.columns) ^ set(df.columns)
        print(f'  -> DIFFERENT columns: {diff}')
    print(f'  First row: {vdf.iloc[0].tolist()}')
