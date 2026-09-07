import pandas as pd
import numpy as np

print('=== PAYSIM FULL DATASET ANALYSIS ===')
path = r'Datasets\Paysim\paysim dataset.csv'

# Full scan for totals
print('Scanning full dataset...')
total_rows = 0
total_fraud = 0
total_flagged = 0
amount_sum = 0
amount_sq_sum = 0
amount_min = float('inf')
amount_max = float('-inf')
type_counts = {}
step_min = float('inf')
step_max = float('-inf')
name_orig_set_sample = set()
name_dest_set_sample = set()
fraud_by_type = {}
amount_fraud = []
amount_legit_sample = []
balance_issues = 0

for chunk in pd.read_csv(path, chunksize=100000):
    total_rows += len(chunk)
    total_fraud += chunk['isFraud'].sum()
    total_flagged += chunk['isFlaggedFraud'].sum()
    
    chunk_min = chunk['amount'].min()
    chunk_max = chunk['amount'].max()
    if chunk_min < amount_min: amount_min = chunk_min
    if chunk_max > amount_max: amount_max = chunk_max
    
    amount_sum += chunk['amount'].sum()
    amount_sq_sum += (chunk['amount']**2).sum()
    
    step_min = min(step_min, chunk['step'].min())
    step_max = max(step_max, chunk['step'].max())
    
    for t, cnt in chunk['type'].value_counts().items():
        type_counts[t] = type_counts.get(t, 0) + cnt
    
    # Fraud by type
    for t in chunk['type'].unique():
        sub = chunk[chunk['type']==t]
        if t not in fraud_by_type:
            fraud_by_type[t] = {'fraud': 0, 'total': 0}
        fraud_by_type[t]['fraud'] += sub['isFraud'].sum()
        fraud_by_type[t]['total'] += len(sub)
    
    # Sample names
    if len(name_orig_set_sample) < 20:
        name_orig_set_sample.update(chunk['nameOrig'].head(20).tolist())
    if len(name_dest_set_sample) < 20:
        name_dest_set_sample.update(chunk['nameDest'].head(20).tolist())
    
    # Balance check: newbalanceOrig = oldbalanceOrg - amount (for non-cash-in)
    mask = chunk['type'].isin(['PAYMENT','TRANSFER','DEBIT'])
    expected = chunk.loc[mask, 'oldbalanceOrg'] - chunk.loc[mask, 'amount']
    actual = chunk.loc[mask, 'newbalanceOrig']
    mismatch = (abs(expected - actual) > 0.01).sum()
    balance_issues += mismatch

n = total_rows
mean = amount_sum / n
std = ((amount_sq_sum / n) - mean**2)**0.5

print(f'Total rows: {total_rows:,}')
print(f'Total fraud: {total_fraud:,}')
print(f'Fraud rate: {total_fraud/total_rows:.4%}')
print(f'Flagged fraud: {total_flagged:,}')
print(f'Step range: {step_min} to {step_max} (hours of simulation)')
print(f'Time span: {step_max} hours = ~{step_max/24:.1f} days = ~{step_max/24/30:.1f} months')
print(f'Amount: min={amount_min:.2f}, max={amount_max:.2f}, mean={mean:.2f}, std={std:.2f}')

print(f'\nTransaction type distribution:')
for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f'  {t}: {c:,} ({c/total_rows:.2%})')

print(f'\nFraud by transaction type:')
for t, d in fraud_by_type.items():
    rate = d["fraud"]/d["total"] if d["total"] > 0 else 0
    print(f'  {t}: fraud={d["fraud"]:,}/{d["total"]:,} ({rate:.4%})')

print(f'\nBalance mismatch count (non-cash-in): {balance_issues:,}')

print(f'\nSample nameOrig prefixes: C=Customer, M=Merchant')
orig_c = sum(1 for n in name_orig_set_sample if str(n).startswith('C'))
orig_m = sum(1 for n in name_orig_set_sample if str(n).startswith('M'))
dest_c = sum(1 for n in name_dest_set_sample if str(n).startswith('C'))
dest_m = sum(1 for n in name_dest_set_sample if str(n).startswith('M'))
print(f'  nameOrig sample: C={orig_c}, M={orig_m}')
print(f'  nameDest sample: C={dest_c}, M={dest_m}')

# Now analyze sequence suitability
print('\n=== SEQUENCE SUITABILITY ANALYSIS ===')
# Sample 500k rows
df_sample = pd.read_csv(path, nrows=500000)
orig_counts = df_sample['nameOrig'].value_counts()
print(f'Unique originators in 500k rows: {len(orig_counts):,}')
print(f'Transactions per originator (500k sample):')
print(f'  Mean: {orig_counts.mean():.2f}')
print(f'  Median: {orig_counts.median():.1f}')
print(f'  Max: {orig_counts.max()}')
print(f'  Min: {orig_counts.min()}')
print(f'  % with only 1 transaction: {(orig_counts==1).sum()/len(orig_counts):.2%}')
print(f'  % with 2+ transactions: {(orig_counts>=2).sum()/len(orig_counts):.2%}')
print(f'  % with 5+ transactions: {(orig_counts>=5).sum()/len(orig_counts):.2%}')
print(f'  % with 10+ transactions: {(orig_counts>=10).sum()/len(orig_counts):.2%}')
