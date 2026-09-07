import pandas as pd
import numpy as np

print('=== PAYSIM DATASET - FULL ANALYSIS ===')
path = r'Datasets\Paysim\paysim dataset.csv'

# Read in chunks to check structure without blowing memory
chunk_size = 100000
chunks_meta = []
row_count = 0
fraud_count = 0
flagged_count = 0
dtypes_sample = None
headers = None

for i, chunk in enumerate(pd.read_csv(path, chunksize=chunk_size)):
    if i == 0:
        headers = chunk.columns.tolist()
        dtypes_sample = chunk.dtypes
        print(f'Columns: {headers}')
        print(f'Dtypes:\n{dtypes_sample.to_string()}')
        print(f'\nFirst 3 rows:')
        print(chunk.head(3).to_string())
        
    row_count += len(chunk)
    fraud_count += chunk['isFraud'].sum()
    flagged_count += chunk['isFlaggedFraud'].sum()
    
    if i == 0:
        # Deep analysis on first chunk
        print(f'\n--- First chunk analysis ---')
        print(f'Missing values:\n{chunk.isnull().sum().to_string()}')
        print(f'Duplicate rows in first chunk: {chunk.duplicated().sum()}')
        print(f'Amount stats:\n{chunk["amount"].describe().to_string()}')
        print(f'Transaction types:\n{chunk["type"].value_counts().to_string()}')
        print(f'Fraud in chunk: {chunk["isFraud"].sum()} ({chunk["isFraud"].mean():.4%})')
        print(f'\nSample nameOrig: {chunk["nameOrig"].head(5).tolist()}')
        print(f'Sample nameDest: {chunk["nameDest"].head(5).tolist()}')
        
    if i >= 9:  # Sample first 1M rows for speed
        break

print(f'\n--- Totals (first {row_count:,} rows) ---')
print(f'Total rows inspected: {row_count:,}')
print(f'Fraud count: {fraud_count:,}')
print(f'Fraud rate: {fraud_count/row_count:.4%}')
print(f'Flagged fraud: {flagged_count:,}')

# Count total rows
import subprocess
result = subprocess.run(['python', '-c', 
    "f=open(r'Datasets/Paysim/paysim dataset.csv'); print(sum(1 for _ in f)-1)"],
    capture_output=True, text=True)
print(f'Total file rows (approx): checking via full scan...')

# Use wc-like approach
with open(path, 'rb') as f:
    total = sum(1 for _ in f) - 1  # subtract header
print(f'Total rows in file: {total:,}')
