"""
IEEE-CIS Forensic Analysis Script
Adaptive Financial Fraud Detection Project
==========================================
Performs a complete, memory-safe forensic analysis of:
  - train_transaction.csv  (394 columns, ~590K rows, 683 MB)
  - train_identity.csv     (41 columns, ~144K rows, 26.5 MB)
  - test_transaction.csv   (613 MB)
  - test_identity.csv      (25.8 MB)
  - sample_submission.csv  (6.1 MB)

Uses chunked reading where files > 100 MB.
Writes all output to: dataset_analysis/ieee_cis/
Does NOT modify raw files.
Does NOT load full 683 MB into memory simultaneously.
"""

import os
import csv
import json
import math
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = r'Datasets\IEEE CIS-20260829T103704Z-1-001\IEEE CIS'
TRN_TX   = os.path.join(BASE_DIR, 'train_transaction.csv')
TRN_ID   = os.path.join(BASE_DIR, 'train_identity.csv')
TST_TX   = os.path.join(BASE_DIR, 'test_transaction.csv')
TST_ID   = os.path.join(BASE_DIR, 'test_identity.csv')
SUB      = os.path.join(BASE_DIR, 'sample_submission.csv')

OUT_DIR  = r'dataset_analysis\ieee_cis'
os.makedirs(OUT_DIR, exist_ok=True)

CHUNK = 100_000   # rows per chunk

print("=" * 70)
print("  IEEE-CIS FORENSIC ANALYSIS")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — FILE INVENTORY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[0] FILE INVENTORY")
file_records = []
for label, path in [('train_transaction', TRN_TX), ('train_identity', TRN_ID),
                     ('test_transaction', TST_TX),  ('test_identity', TST_ID),
                     ('sample_submission', SUB)]:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1e6
        # count lines without loading into memory
        with open(path, 'rb') as f:
            n_lines = sum(1 for _ in f)
        rows = n_lines - 1   # subtract header
        file_records.append({'file': label, 'path': path,
                              'size_mb': round(size_mb, 2), 'rows': rows,
                              'exists': True})
        print(f"  {label}: {size_mb:.1f} MB, {rows:,} rows — OK")
    else:
        file_records.append({'file': label, 'path': path, 'size_mb': 0,
                              'rows': 0, 'exists': False})
        print(f"  {label}: NOT FOUND")

with open(f'{OUT_DIR}/file_inventory.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['file','path','size_mb','rows','exists'])
    w.writeheader(); w.writerows(file_records)
print("  -> file_inventory.csv saved")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TRAIN_TRANSACTION: CHUNKED FULL SCAN
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1] TRAIN_TRANSACTION — CHUNKED FULL SCAN")

# Pass 1: column names + dtypes from header chunk
hdr = pd.read_csv(TRN_TX, nrows=5)
tx_cols    = hdr.columns.tolist()
n_tx_cols  = len(tx_cols)
print(f"  Columns: {n_tx_cols}")

# Identify column groups
V_COLS  = [c for c in tx_cols if c.startswith('V')]
C_COLS  = [c for c in tx_cols if c.startswith('C') and c != 'card1']
D_COLS  = [c for c in tx_cols if c.startswith('D')]
M_COLS  = [c for c in tx_cols if c.startswith('M')]
CARD    = [c for c in tx_cols if c.startswith('card')]
ADDR    = [c for c in tx_cols if c.startswith('addr')]
DIST    = [c for c in tx_cols if c.startswith('dist')]
EMAIL   = [c for c in tx_cols if 'email' in c.lower()]
KNOWN   = ['TransactionID','isFraud','TransactionDT','TransactionAmt',
           'ProductCD'] + CARD + ADDR + DIST + EMAIL + C_COLS + D_COLS + M_COLS

print(f"  V-cols: {len(V_COLS)} | C-cols: {len(C_COLS)} | D-cols: {len(D_COLS)} | "
      f"M-cols: {len(M_COLS)} | card: {len(CARD)} | addr: {len(ADDR)} | "
      f"dist: {len(DIST)} | email: {len(EMAIL)}")

# Accumulators
total_rows   = 0
fraud_count  = 0
tx_id_min, tx_id_max = float('inf'), float('-inf')
dt_min, dt_max       = float('inf'), float('-inf')
amt_min, amt_max     = float('inf'), float('-inf')
amt_sum  = 0.0
amt_sq   = 0.0

# Missing counts
miss_counts = defaultdict(int)

# Unique value tracking (for low-card columns only — to avoid memory blow-up)
LOW_CARD_COLS = ['isFraud','ProductCD'] + CARD + ADDR + EMAIL + M_COLS
uniq_vals = {c: Counter() for c in LOW_CARD_COLS}

# For entity analysis — accumulate card1 -> tx counts
card1_counts    = Counter()
card1_fraud     = Counter()
card12_counts   = Counter()  # card1+card2 composite
card12_fraud    = Counter()
addr1_counts    = Counter()
card1addr1_counts = Counter()
card1addr1_fraud  = Counter()

# TransactionDT tracking (for temporal analysis)
dt_by_fraud     = defaultdict(list)   # sample — keep 1-in-100 for histogram
fraud_dt_vals   = []
legit_dt_vals   = []

# C-column sums for fraud vs legit
c_fraud_sum     = defaultdict(float)
c_legit_sum     = defaultdict(float)
c_fraud_n       = defaultdict(int)
c_legit_n       = defaultdict(int)

# D1 (timedelta) distribution
d1_fraud_vals   = []
d1_legit_vals   = []

# Duplicate ID check
all_tx_ids      = []

chunk_n = 0
print("  Scanning chunks...")
for chunk in pd.read_csv(TRN_TX, chunksize=CHUNK, low_memory=False):
    chunk_n += 1
    n = len(chunk)
    total_rows += n
    fraud_this = int(chunk['isFraud'].sum())
    fraud_count += fraud_this

    # TransactionID
    chunk_ids = chunk['TransactionID'].dropna()
    if len(chunk_ids):
        tx_id_min = min(tx_id_min, int(chunk_ids.min()))
        tx_id_max = max(tx_id_max, int(chunk_ids.max()))
    all_tx_ids.extend(chunk_ids.tolist())

    # TransactionDT
    dt_vals = chunk['TransactionDT'].dropna()
    if len(dt_vals):
        dt_min = min(dt_min, float(dt_vals.min()))
        dt_max = max(dt_max, float(dt_vals.max()))

    # Amount
    amt = chunk['TransactionAmt'].dropna()
    if len(amt):
        amt_min = min(amt_min, float(amt.min()))
        amt_max = max(amt_max, float(amt.max()))
        amt_sum += float(amt.sum())
        amt_sq  += float((amt**2).sum())

    # Missing counts
    for c in tx_cols:
        miss_counts[c] += int(chunk[c].isnull().sum())

    # Low-cardinality unique values
    for c in LOW_CARD_COLS:
        if c in chunk.columns:
            uniq_vals[c].update(chunk[c].dropna().astype(str).tolist())

    # Entity analysis — card1
    chunk['card1'] = chunk['card1'].astype(str)
    card1_counts.update(chunk['card1'].tolist())
    fraud_mask = chunk['isFraud'] == 1
    for cid in chunk.loc[fraud_mask, 'card1'].tolist():
        card1_fraud[cid] += 1

    # card1 + card2 composite
    chunk['card12'] = chunk['card1'].astype(str) + '_' + chunk['card2'].astype(str)
    card12_counts.update(chunk['card12'].tolist())
    for cid in chunk.loc[fraud_mask, 'card12'].tolist():
        card12_fraud[cid] += 1

    # addr1
    chunk['addr1_s'] = chunk['addr1'].astype(str)
    addr1_counts.update(chunk['addr1_s'].tolist())

    # card1 + addr1
    chunk['c1a1'] = chunk['card1'].astype(str) + '_' + chunk['addr1'].astype(str)
    card1addr1_counts.update(chunk['c1a1'].tolist())
    for cid in chunk.loc[fraud_mask, 'c1a1'].tolist():
        card1addr1_fraud[cid] += 1

    # TransactionDT sample for temporal plots
    if chunk_n % 2 == 0:  # sample every 2nd chunk
        for idx, row in chunk[['TransactionDT','isFraud']].iterrows():
            if not pd.isna(row['TransactionDT']):
                if row['isFraud'] == 1:
                    fraud_dt_vals.append(float(row['TransactionDT']))
                elif len(legit_dt_vals) < 50000:
                    legit_dt_vals.append(float(row['TransactionDT']))

    # D1 sample
    if chunk_n <= 10:
        d1 = chunk[['D1','isFraud']].dropna(subset=['D1'])
        d1_fraud_vals.extend(d1[d1['isFraud']==1]['D1'].tolist())
        d1_legit_vals.extend(d1[d1['isFraud']==0]['D1'].sample(
            min(500, len(d1[d1['isFraud']==0])), random_state=42).tolist())

    # C-column stats by fraud label
    for cc in C_COLS:
        if cc in chunk.columns:
            fr = chunk.loc[fraud_mask, cc].dropna()
            lg = chunk.loc[~fraud_mask, cc].dropna()
            c_fraud_sum[cc] += float(fr.sum())
            c_fraud_n[cc]   += len(fr)
            c_legit_sum[cc] += float(lg.sum())
            c_legit_n[cc]   += len(lg)

    if chunk_n % 2 == 0:
        print(f"    ...chunk {chunk_n}: {total_rows:,} rows, {fraud_count:,} fraud so far")

print(f"  DONE. Total rows: {total_rows:,} | Fraud: {fraud_count:,}")
amt_mean = amt_sum / total_rows
amt_std  = math.sqrt(max(0, amt_sq/total_rows - amt_mean**2))

# Duplicate TransactionID check
tx_id_dupes = len(all_tx_ids) - len(set(all_tx_ids))
print(f"  TransactionID duplicates: {tx_id_dupes}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TRAIN_IDENTITY: FULL LOAD (small enough)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2] TRAIN_IDENTITY — FULL LOAD")
df_id = pd.read_csv(TRN_ID, low_memory=False)
id_rows = len(df_id)
id_cols = df_id.columns.tolist()
print(f"  Rows: {id_rows:,} | Columns: {len(id_cols)}")

id_miss  = {c: int(df_id[c].isnull().sum()) for c in id_cols}
id_uniq  = {c: df_id[c].nunique()           for c in id_cols}
id_dtype = {c: str(df_id[c].dtype)          for c in id_cols}

# Save identity column profile
id_profile = []
for c in id_cols:
    miss_pct = round(id_miss[c] / id_rows * 100, 2)
    n_uniq   = id_uniq[c]
    dtype    = id_dtype[c]
    ex_vals  = df_id[c].dropna().unique()[:5].tolist()
    id_profile.append({'column': c, 'dtype': dtype, 'missing_pct': miss_pct,
                        'n_unique': n_uniq, 'example_values': str(ex_vals)})

with open(f'{OUT_DIR}/identity_column_profile.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['column','dtype','missing_pct',
                                       'n_unique','example_values'])
    w.writeheader(); w.writerows(id_profile)
print("  -> identity_column_profile.csv saved")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TRANSACTIONID LINKAGE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3] TRANSACTIONID LINKAGE")
train_ids_set = set(all_tx_ids)
id_tx_ids     = set(df_id['TransactionID'].dropna().astype(int).tolist())

overlap       = len(train_ids_set & id_tx_ids)
only_in_tx    = len(train_ids_set - id_tx_ids)
only_in_id    = len(id_tx_ids - train_ids_set)

pct_tx_with_id = overlap / total_rows * 100 if total_rows > 0 else 0

print(f"  train_transaction unique IDs: {len(train_ids_set):,}")
print(f"  train_identity unique IDs:    {len(id_tx_ids):,}")
print(f"  Overlap (both files):         {overlap:,} ({pct_tx_with_id:.1f}% of transactions)")
print(f"  Only in transaction:          {only_in_tx:,}")
print(f"  Only in identity:             {only_in_id:,}")

# Check for duplicates in identity
id_dup_ids = df_id['TransactionID'].duplicated().sum()
print(f"  Duplicate TransactionIDs in identity: {id_dup_ids}")

linkage = {
    'train_tx_unique_ids': len(train_ids_set),
    'train_id_unique_ids': len(id_tx_ids),
    'overlap': overlap,
    'pct_tx_with_identity': round(pct_tx_with_id, 2),
    'only_in_transaction': only_in_tx,
    'only_in_identity': only_in_id,
    'join_type': 'left join (not all transactions have identity)',
    'id_duplicate_tx_ids': int(id_dup_ids)
}
with open(f'{OUT_DIR}/linkage_analysis.json', 'w') as f:
    json.dump(linkage, f, indent=2)
print("  -> linkage_analysis.json saved")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TEMPORAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4] TEMPORAL ANALYSIS (TransactionDT)")
# TransactionDT is seconds since a reference point (start of study period)
# Min, max, span
dt_span_sec  = dt_max - dt_min
dt_span_days = dt_span_sec / 86400.0
dt_span_months = dt_span_days / 30.44

print(f"  TransactionDT min:   {dt_min:,.0f}")
print(f"  TransactionDT max:   {dt_max:,.0f}")
print(f"  Span (seconds):      {dt_span_sec:,.0f}")
print(f"  Span (days):         {dt_span_days:.1f}")
print(f"  Span (months):       {dt_span_months:.1f}")
print(f"  Likely unit:         SECONDS (not a Unix timestamp — relative epoch)")

# Transactions per day (approximate)
tx_per_day = total_rows / max(dt_span_days, 1)
print(f"  Avg transactions/day: {tx_per_day:,.0f}")

# Fraud DT distribution — bin into 50 windows
if fraud_dt_vals and legit_dt_vals:
    dt_range = dt_max - dt_min
    n_bins = 50
    bin_size = dt_range / n_bins

    fraud_bins  = [0]*n_bins
    legit_bins  = [0]*n_bins
    for v in fraud_dt_vals:
        b = min(int((v - dt_min) / bin_size), n_bins-1)
        fraud_bins[b] += 1
    for v in legit_dt_vals:
        b = min(int((v - dt_min) / bin_size), n_bins-1)
        legit_bins[b] += 1

    temporal_bins = []
    for i in range(n_bins):
        day_start = dt_min + i * bin_size
        temporal_bins.append({
            'bin': i,
            'dt_start': round(day_start, 0),
            'day_start': round(day_start / 86400.0, 2),
            'fraud_sample_count': fraud_bins[i],
            'legit_sample_count': legit_bins[i],
        })

    with open(f'{OUT_DIR}/temporal_distribution.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['bin','dt_start','day_start',
                                           'fraud_sample_count','legit_sample_count'])
        w.writeheader(); w.writerows(temporal_bins)
    print("  -> temporal_distribution.csv saved")

# Test vs train DT range — check test file
print("\n  Checking test_transaction.csv DT range (first 50K rows)...")
test_dt_sample = pd.read_csv(TST_TX, usecols=['TransactionID','TransactionDT'],
                              nrows=50000)
test_dt_min_s  = float(test_dt_sample['TransactionDT'].min())
test_dt_max_s  = float(test_dt_sample['TransactionDT'].max())
print(f"  Test DT sample min: {test_dt_min_s:,.0f} | max: {test_dt_max_s:,.0f}")
print(f"  Test DT sample min day: {test_dt_min_s/86400:.1f} | max day: {test_dt_max_s/86400:.1f}")

temporal_summary = {
    'train_dt_min': dt_min, 'train_dt_max': dt_max,
    'train_dt_span_seconds': dt_span_sec,
    'train_dt_span_days': round(dt_span_days, 2),
    'train_dt_span_months': round(dt_span_months, 2),
    'dt_unit': 'seconds (relative, not Unix timestamp)',
    'test_dt_min_sample': test_dt_min_s,
    'test_dt_max_sample': test_dt_max_s,
    'train_avg_tx_per_day': round(tx_per_day, 1),
    'total_train_rows': total_rows,
    'total_fraud': fraud_count,
    'fraud_rate_pct': round(fraud_count / total_rows * 100, 4)
}
with open(f'{OUT_DIR}/temporal_summary.json', 'w') as f:
    json.dump(temporal_summary, f, indent=2)
print("  -> temporal_summary.json saved")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MISSING VALUES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5] MISSING VALUES ANALYSIS")
miss_profile = []
for c in tx_cols:
    miss_n   = miss_counts[c]
    miss_pct = round(miss_n / total_rows * 100, 4)
    miss_profile.append({'column': c, 'missing_n': miss_n, 'missing_pct': miss_pct})

miss_profile.sort(key=lambda x: -x['missing_pct'])

with open(f'{OUT_DIR}/missingness_analysis.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['column','missing_n','missing_pct'])
    w.writeheader(); w.writerows(miss_profile)
print("  -> missingness_analysis.csv saved")

# How many columns with >80% missing?
high_miss = [r for r in miss_profile if r['missing_pct'] > 80]
mid_miss  = [r for r in miss_profile if 20 < r['missing_pct'] <= 80]
low_miss  = [r for r in miss_profile if 0 < r['missing_pct'] <= 20]
zero_miss = [r for r in miss_profile if r['missing_pct'] == 0]

print(f"  Columns with 0% missing:      {len(zero_miss)}")
print(f"  Columns with 0-20% missing:   {len(low_miss)}")
print(f"  Columns with 20-80% missing:  {len(mid_miss)}")
print(f"  Columns with >80% missing:    {len(high_miss)}")
print(f"  Worst missing columns (top 10):")
for r in miss_profile[:10]:
    print(f"    {r['column']}: {r['missing_pct']:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — ENTITY ANALYSIS (SEQUENCE SUITABILITY)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[6] ENTITY ANALYSIS — SEQUENCE SUITABILITY")

def entity_stats(counter, name, fraud_counter=None):
    counts = list(counter.values())
    if not counts:
        return {}
    counts_arr = np.array(counts)
    n_entities  = len(counts)
    total_tx    = sum(counts)
    mean_tx     = float(np.mean(counts_arr))
    median_tx   = float(np.median(counts_arr))
    max_tx      = int(np.max(counts_arr))
    p90         = float(np.percentile(counts_arr, 90))
    p95         = float(np.percentile(counts_arr, 95))
    p99         = float(np.percentile(counts_arr, 99))
    pct_1       = round(np.sum(counts_arr == 1) / n_entities * 100, 2)
    pct_2plus   = round(np.sum(counts_arr >= 2) / n_entities * 100, 2)
    pct_5plus   = round(np.sum(counts_arr >= 5) / n_entities * 100, 2)
    pct_10plus  = round(np.sum(counts_arr >= 10) / n_entities * 100, 2)
    pct_20plus  = round(np.sum(counts_arr >= 20) / n_entities * 100, 2)

    # Fraud-linked entities
    fraud_entities = 0
    if fraud_counter:
        fraud_entities = len(fraud_counter)

    quality = "EXCELLENT" if pct_5plus > 30 and median_tx >= 3 else \
              "GOOD"      if pct_5plus > 15 and median_tx >= 2 else \
              "WEAK"      if pct_2plus > 10 else "POOR"

    return {
        'entity_key': name,
        'unique_entities': n_entities,
        'total_transactions': total_tx,
        'mean_txns': round(mean_tx, 2),
        'median_txns': round(median_tx, 2),
        'max_txns': max_tx,
        'p90_txns': round(p90, 1),
        'p95_txns': round(p95, 1),
        'p99_txns': round(p99, 1),
        'pct_single_txn': pct_1,
        'pct_2plus': pct_2plus,
        'pct_5plus': pct_5plus,
        'pct_10plus': pct_10plus,
        'pct_20plus': pct_20plus,
        'fraud_linked_entities': fraud_entities,
        'sequence_quality': quality,
    }

entity_results = []
for counter, name, fraud_c in [
    (card1_counts,    'card1',        card1_fraud),
    (card12_counts,   'card1+card2',  card12_fraud),
    (addr1_counts,    'addr1',        None),
    (card1addr1_counts,'card1+addr1', card1addr1_fraud),
]:
    stats = entity_stats(counter, name, fraud_c)
    entity_results.append(stats)
    print(f"\n  [{name}]")
    print(f"    Unique entities:  {stats['unique_entities']:,}")
    print(f"    Mean txns:        {stats['mean_txns']:.2f}")
    print(f"    Median:           {stats['median_txns']:.2f}")
    print(f"    P95:              {stats['p95_txns']:.1f}")
    print(f"    Max:              {stats['max_txns']}")
    print(f"    % single-txn:     {stats['pct_single_txn']:.2f}%")
    print(f"    % >= 5 txns:      {stats['pct_5plus']:.2f}%")
    print(f"    % >= 10 txns:     {stats['pct_10plus']:.2f}%")
    print(f"    % >= 20 txns:     {stats['pct_20plus']:.2f}%")
    print(f"    Fraud-linked:     {stats['fraud_linked_entities']:,}")
    print(f"    Sequence quality: {stats['sequence_quality']}")

with open(f'{OUT_DIR}/entity_analysis.csv', 'w', newline='') as f:
    fieldnames = list(entity_results[0].keys())
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader(); w.writerows(entity_results)
print("\n  -> entity_analysis.csv saved")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — SEQUENCE CONSTRUCTION FEASIBILITY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[7] SEQUENCE CONSTRUCTION FEASIBILITY")

# Best entity key determined from above (card1 is primary candidate)
# Load just TransactionID, TransactionDT, isFraud, card1 for sequence analysis
print("  Loading card1/DT/fraud for sequence analysis...")
seq_cols = ['TransactionID','TransactionDT','isFraud','card1','card2','addr1']
seq_chunks = []
for chunk in pd.read_csv(TRN_TX, usecols=seq_cols, chunksize=CHUNK, low_memory=False):
    seq_chunks.append(chunk)
df_seq = pd.concat(seq_chunks, ignore_index=True)
df_seq['card1'] = df_seq['card1'].astype(str)
df_seq = df_seq.sort_values('TransactionDT').reset_index(drop=True)
print(f"  Loaded {len(df_seq):,} rows for sequence analysis")

# Group by card1 and compute sequence lengths
seq_lengths_by_card1 = df_seq.groupby('card1').size()

# Sequence count at various min-length thresholds
seq_analysis = []
for min_len in [2, 3, 5, 10, 20, 50]:
    eligible = (seq_lengths_by_card1 >= min_len)
    n_entities   = int(eligible.sum())
    # Number of valid sequences = for each entity, (entity_tx_count - (min_len - 1))
    # i.e., how many starting points exist for a window of size min_len
    eligible_lens = seq_lengths_by_card1[eligible]
    n_sequences  = int((eligible_lens - (min_len - 1)).sum())

    # Fraud sequences: entity has at least 1 fraud and len >= min_len
    fraud_per_entity = df_seq.groupby('card1')['isFraud'].sum()
    fraud_eligible   = fraud_per_entity[fraud_per_entity > 0].index
    fraud_entities_seq = set(seq_lengths_by_card1[seq_lengths_by_card1 >= min_len].index) \
                         & set(fraud_eligible)
    n_fraud_entities = len(fraud_entities_seq)
    # Fraud sequence count (approximate: sequences that contain at least one fraud tx)
    fraud_entity_lens = seq_lengths_by_card1[list(fraud_entities_seq)]
    n_fraud_seqs = int((fraud_entity_lens - (min_len - 1)).sum()) if len(fraud_entity_lens) > 0 else 0

    seq_analysis.append({
        'min_sequence_length': min_len,
        'eligible_entities': n_entities,
        'total_sequences': n_sequences,
        'fraud_linked_entities': n_fraud_entities,
        'approx_fraud_sequences': n_fraud_seqs,
        'viable_for_lstm': 'YES' if n_fraud_seqs > 5000 else 'MARGINAL' if n_fraud_seqs > 1000 else 'NO'
    })
    print(f"  seq_len>={min_len}: {n_entities:,} entities, {n_sequences:,} seqs, "
          f"~{n_fraud_seqs:,} fraud seqs -> {seq_analysis[-1]['viable_for_lstm']}")

with open(f'{OUT_DIR}/sequence_analysis.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['min_sequence_length','eligible_entities',
                                       'total_sequences','fraud_linked_entities',
                                       'approx_fraud_sequences','viable_for_lstm'])
    w.writeheader(); w.writerows(seq_analysis)
print("  -> sequence_analysis.csv saved")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — FRAUD DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[8] FRAUD DISTRIBUTION")
fraud_rate = fraud_count / total_rows * 100
class_ratio = (total_rows - fraud_count) / fraud_count if fraud_count > 0 else float('inf')
print(f"  Total rows:      {total_rows:,}")
print(f"  Fraud:           {fraud_count:,} ({fraud_rate:.4f}%)")
print(f"  Legitimate:      {total_rows - fraud_count:,}")
print(f"  Class ratio:     {class_ratio:.1f}:1")

# Fraud by ProductCD
print("\n  Fraud by ProductCD:")
for val, cnt in sorted(uniq_vals['ProductCD'].items()):
    print(f"    ProductCD={val}: {cnt:,} transactions")

# Fraud by card4 (card network)
print("\n  card4 distribution:")
for val, cnt in uniq_vals['card4'].most_common(10):
    print(f"    card4={val}: {cnt:,}")

# Fraud by card6 (credit/debit)
print("\n  card6 distribution:")
for val, cnt in uniq_vals['card6'].most_common(10):
    print(f"    card6={val}: {cnt:,}")

fraud_dist = {
    'total_rows': total_rows,
    'fraud_count': fraud_count,
    'legit_count': total_rows - fraud_count,
    'fraud_rate_pct': round(fraud_rate, 4),
    'class_ratio': round(class_ratio, 1),
    'amt_min': round(amt_min, 4),
    'amt_max': round(amt_max, 4),
    'amt_mean': round(amt_mean, 4),
    'amt_std': round(amt_std, 4),
    'tx_id_min': tx_id_min,
    'tx_id_max': tx_id_max,
    'tx_id_unique': len(set(all_tx_ids)),
    'tx_id_duplicates': tx_id_dupes,
}
with open(f'{OUT_DIR}/fraud_distribution.json', 'w') as f:
    json.dump(fraud_dist, f, indent=2)
print("  -> fraud_distribution.json saved")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — MISSINGNESS AS SIGNAL (sample-based)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[9] MISSINGNESS AS SIGNAL (first 200K rows)")
miss_signal_cols = ['D2','D3','D4','D5','D6','D7','D8','D9','D10','D11',
                    'D12','D13','D14','D15','dist2','card2','card3','card5',
                    'addr1','addr2','P_emaildomain','R_emaildomain']

miss_sig_df = pd.read_csv(TRN_TX, usecols=['isFraud'] + miss_signal_cols,
                           nrows=200000, low_memory=False)
miss_signal_rows = []
for c in miss_signal_cols:
    if c not in miss_sig_df.columns:
        continue
    is_missing = miss_sig_df[c].isnull()
    fraud_col  = miss_sig_df['isFraud']
    p_fraud_present = float(fraud_col[~is_missing].mean()) if (~is_missing).sum() > 0 else float('nan')
    p_fraud_missing = float(fraud_col[is_missing].mean())  if is_missing.sum()  > 0 else float('nan')
    miss_signal_rows.append({
        'column': c,
        'pct_missing': round(is_missing.mean() * 100, 2),
        'p_fraud_when_present': round(p_fraud_present * 100, 4) if not math.isnan(p_fraud_present) else 'N/A',
        'p_fraud_when_missing': round(p_fraud_missing * 100, 4) if not math.isnan(p_fraud_missing) else 'N/A',
        'missingness_predictive': 'YES' if (not math.isnan(p_fraud_present) and
                                             not math.isnan(p_fraud_missing) and
                                             abs(p_fraud_present - p_fraud_missing) > 1.0)
                                       else 'WEAK'
    })

with open(f'{OUT_DIR}/missingness_signal.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['column','pct_missing',
                                       'p_fraud_when_present',
                                       'p_fraud_when_missing',
                                       'missingness_predictive'])
    w.writeheader(); w.writerows(miss_signal_rows)
print("  -> missingness_signal.csv saved")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — CONCEPT DRIFT (temporal windows)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[10] CONCEPT DRIFT — TEMPORAL WINDOW ANALYSIS")
# Divide train data into 6 temporal windows by TransactionDT
window_size_sec = dt_span_sec / 6.0

drift_records = []
print("  Loading TransactionDT+isFraud+TransactionAmt+card1 for drift analysis...")
drift_chunks = []
for chunk in pd.read_csv(TRN_TX,
                          usecols=['TransactionDT','isFraud','TransactionAmt','card1'],
                          chunksize=CHUNK, low_memory=False):
    drift_chunks.append(chunk)
df_drift = pd.concat(drift_chunks, ignore_index=True)

for w_idx in range(6):
    w_start = dt_min + w_idx * window_size_sec
    w_end   = dt_min + (w_idx + 1) * window_size_sec
    mask    = (df_drift['TransactionDT'] >= w_start) & (df_drift['TransactionDT'] < w_end)
    sub     = df_drift[mask]
    if len(sub) == 0:
        continue
    fraud_r = float(sub['isFraud'].mean() * 100)
    amt_m   = float(sub['TransactionAmt'].mean())
    n_uniq_cards = sub['card1'].nunique()
    drift_records.append({
        'window': w_idx + 1,
        'dt_start': round(w_start, 0),
        'dt_end':   round(w_end, 0),
        'day_start': round(w_start / 86400, 1),
        'day_end':   round(w_end   / 86400, 1),
        'n_transactions': len(sub),
        'fraud_count':    int(sub['isFraud'].sum()),
        'fraud_rate_pct': round(fraud_r, 4),
        'mean_amount':    round(amt_m, 2),
        'unique_cards':   n_uniq_cards,
    })
    print(f"  Window {w_idx+1} (days {round(w_start/86400,0):.0f}–{round(w_end/86400,0):.0f}): "
          f"{len(sub):,} txns, fraud={fraud_r:.3f}%, amt_mean={amt_m:.2f}")

with open(f'{OUT_DIR}/drift_analysis.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['window','dt_start','dt_end','day_start','day_end',
                                       'n_transactions','fraud_count','fraud_rate_pct',
                                       'mean_amount','unique_cards'])
    w.writeheader(); w.writerows(drift_records)
print("  -> drift_analysis.csv saved")
del df_drift

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — TRANSACTION COLUMN PROFILE (sample-based for V-cols)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[11] TRANSACTION COLUMN PROFILE (sample 200K rows for V-column stats)")
sample_df = pd.read_csv(TRN_TX, nrows=200000, low_memory=False)

tx_profile = []
for c in tx_cols:
    miss_pct = round(miss_counts[c] / total_rows * 100, 4)
    if c in LOW_CARD_COLS:
        n_uniq  = len(uniq_vals[c])
        ex_vals = list(uniq_vals[c].keys())[:5]
    else:
        n_uniq  = sample_df[c].nunique() if c in sample_df.columns else -1
        ex_vals = sample_df[c].dropna().unique()[:5].tolist() if c in sample_df.columns else []

    dtype = str(sample_df[c].dtype) if c in sample_df.columns else 'unknown'

    # Column group
    if c == 'isFraud':         grp = 'TARGET'
    elif c == 'TransactionID': grp = 'TRANSACTION_ID'
    elif c == 'TransactionDT': grp = 'TEMPORAL'
    elif c == 'TransactionAmt':grp = 'AMOUNT'
    elif c == 'ProductCD':     grp = 'PRODUCT'
    elif c.startswith('card'): grp = 'CARD'
    elif c.startswith('addr'): grp = 'ADDRESS'
    elif c.startswith('dist'): grp = 'DISTANCE'
    elif 'email' in c.lower(): grp = 'EMAIL'
    elif c.startswith('C'):    grp = 'COUNTING'
    elif c.startswith('D'):    grp = 'TIMEDELTA'
    elif c.startswith('M'):    grp = 'MATCH_FLAG'
    elif c.startswith('V'):    grp = 'ANONYMIZED_V'
    else:                      grp = 'OTHER'

    tx_profile.append({'column': c, 'group': grp, 'dtype': dtype,
                        'missing_pct': miss_pct, 'n_unique_sample': n_uniq,
                        'example_values': str(ex_vals)})

with open(f'{OUT_DIR}/transaction_column_profile.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['column','group','dtype','missing_pct',
                                       'n_unique_sample','example_values'])
    w.writeheader(); w.writerows(tx_profile)
print("  -> transaction_column_profile.csv saved")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — C-COLUMN ANALYSIS (counting features)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[12] C-COLUMN ANALYSIS (counting behavioral features)")
c_col_records = []
for cc in C_COLS:
    fraud_mean = c_fraud_sum[cc] / c_fraud_n[cc] if c_fraud_n[cc] > 0 else float('nan')
    legit_mean = c_legit_sum[cc] / c_legit_n[cc] if c_legit_n[cc] > 0 else float('nan')
    miss_pct   = round(miss_counts[cc] / total_rows * 100, 4)
    c_col_records.append({
        'column': cc,
        'missing_pct': miss_pct,
        'fraud_mean': round(fraud_mean, 4) if not math.isnan(fraud_mean) else 'N/A',
        'legit_mean': round(legit_mean, 4) if not math.isnan(legit_mean) else 'N/A',
        'interpretation': 'Count-type behavioral feature (exact semantics anonymized)',
    })

with open(f'{OUT_DIR}/c_column_analysis.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['column','missing_pct','fraud_mean','legit_mean','interpretation'])
    w.writeheader(); w.writerows(c_col_records)
print("  -> c_column_analysis.csv saved")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — D-COLUMN (timedelta) ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[13] D-COLUMN ANALYSIS (timedelta features)")
print(f"  D1 (likely days since account creation or last transaction)")
if d1_fraud_vals:
    d1_f = np.array(d1_fraud_vals)
    d1_l = np.array(d1_legit_vals)
    print(f"  D1 fraud mean: {np.mean(d1_f):.2f} | median: {np.median(d1_f):.2f}")
    print(f"  D1 legit mean: {np.mean(d1_l):.2f} | median: {np.median(d1_l):.2f}")

d_col_records = []
for dc in D_COLS:
    miss_pct = round(miss_counts[dc] / total_rows * 100, 4)
    d_col_records.append({
        'column': dc,
        'missing_pct': miss_pct,
        'interpretation': 'Timedelta feature (time since some reference event per entity)',
    })
with open(f'{OUT_DIR}/d_column_analysis.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['column','missing_pct','interpretation'])
    w.writeheader(); w.writerows(d_col_records)
print("  -> d_column_analysis.csv saved")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — TEST FILE HEADER CHECK
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[14] TEST FILE VALIDATION")
test_tx_hdr = pd.read_csv(TST_TX, nrows=3, low_memory=False)
test_id_hdr = pd.read_csv(TST_ID, nrows=3, low_memory=False)
sub_df      = pd.read_csv(SUB)

print(f"  test_transaction columns: {len(test_tx_hdr.columns)} (should be 393 — no isFraud)")
print(f"  test_identity columns:    {len(test_id_hdr.columns)}")
print(f"  sample_submission rows:   {len(sub_df):,}")
print(f"  sample_submission cols:   {sub_df.columns.tolist()}")
has_is_fraud_in_test = 'isFraud' in test_tx_hdr.columns
print(f"  isFraud in test_transaction: {has_is_fraud_in_test}")
print(f"  sample_submission head:\n{sub_df.head()}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 15 — WRITE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[15] WRITING FINAL SUMMARY")

summary = {
    'dataset': 'IEEE-CIS Fraud Detection (Kaggle)',
    'source': 'Vesta Corporation / Kaggle Competition',
    'type': 'Real-world-derived, anonymized, historical e-commerce transactions',
    'train_transaction_rows': total_rows,
    'train_transaction_cols': n_tx_cols,
    'train_identity_rows': id_rows,
    'train_identity_cols': len(id_cols),
    'test_transaction_rows': file_records[2]['rows'],
    'test_transaction_cols': len(test_tx_hdr.columns),
    'test_has_fraud_label': has_is_fraud_in_test,
    'fraud_count': fraud_count,
    'legit_count': total_rows - fraud_count,
    'fraud_rate_pct': round(fraud_count / total_rows * 100, 4),
    'class_ratio': round((total_rows - fraud_count) / fraud_count, 1),
    'tx_id_min': tx_id_min,
    'tx_id_max': tx_id_max,
    'tx_id_duplicates': tx_id_dupes,
    'dt_min': dt_min,
    'dt_max': dt_max,
    'dt_span_days': round(dt_span_days, 1),
    'dt_span_months': round(dt_span_months, 1),
    'pct_transactions_with_identity': round(pct_tx_with_id, 2),
    'v_columns': len(V_COLS),
    'c_columns': len(C_COLS),
    'd_columns': len(D_COLS),
    'm_columns': len(M_COLS),
    'card_columns': len(CARD),
    'addr_columns': len(ADDR),
    'amt_min': round(amt_min, 4),
    'amt_max': round(amt_max, 4),
    'amt_mean': round(amt_mean, 4),
    'amt_std': round(amt_std, 4),
}
with open(f'{OUT_DIR}/dataset_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("  -> dataset_summary.json saved")

print("\n" + "=" * 70)
print("  IEEE-CIS FORENSIC ANALYSIS COMPLETE")
print(f"  Output directory: {OUT_DIR}")
print("=" * 70)
