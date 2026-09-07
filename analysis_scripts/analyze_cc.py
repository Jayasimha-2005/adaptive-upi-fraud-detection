import pandas as pd
import numpy as np

print('=== CREDIT CARD FRAUD 10K - FULL ANALYSIS ===')
path = r'Datasets\Credit card Fraud detection\credit_card_fraud_10k.csv'
df = pd.read_csv(path)
print(f'Shape: {df.shape}')
print(f'File size: ~360KB')
print(f'\nDtypes:')
print(df.dtypes.to_string())
print(f'\nMissing values:')
print(df.isnull().sum().to_string())
print(f'\nDuplicate rows: {df.duplicated().sum()}')
print(f'\nBasic stats:')
print(df.describe().to_string())
fraud_col = 'is_fraud'
print(f'\nFraud distribution:')
print(df[fraud_col].value_counts().to_string())
print(f'Fraud rate: {df[fraud_col].mean():.4%}')
fraud_count = df[fraud_col].sum()
legit_count = len(df) - fraud_count
print(f'Fraud count: {fraud_count}')
print(f'Legit count: {legit_count}')
print(f'Class ratio (legit:fraud): {legit_count/fraud_count:.1f}:1')
print(f'\nMemory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB')
print(f'\nColumn unique values:')
for col in df.columns:
    uv = df[col].unique()
    sample = uv[:5].tolist()
    print(f'  {col}: n_unique={df[col].nunique()}, sample={sample}')

# Check for negative/zero amounts
print(f'\nAmount stats:')
print(f'  Negative amounts: {(df["amount"] < 0).sum()}')
print(f'  Zero amounts: {(df["amount"] == 0).sum()}')
print(f'  Min: {df["amount"].min()}, Max: {df["amount"].max()}')

# Amount by fraud
print(f'\nAmount by fraud label:')
print(df.groupby(fraud_col)['amount'].describe().to_string())

# Categorical analysis
print(f'\nMerchant category distribution:')
print(df['merchant_category'].value_counts().to_string())
print(f'\nFraud rate by merchant category:')
print(df.groupby('merchant_category')[fraud_col].mean().sort_values(ascending=False).to_string())

# Check transaction_id
print(f'\ntransaction_id unique: {df["transaction_id"].nunique()}, total rows: {len(df)}')
print(f'Duplicate transaction_ids: {df["transaction_id"].duplicated().sum()}')
