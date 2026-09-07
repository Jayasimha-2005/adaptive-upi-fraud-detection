# IEEE-CIS Data Validation Report

**Status:** PASS

## File Summary

| File | Path | Size (MB) |
|------|------|-----------|
| train_transaction | `train_transaction.csv` | 683.35 |
| train_identity | `train_identity.csv` | 26.53 |

## train_transaction.csv

- **Rows:** 590,540
- **Columns:** 394
- **Duplicate TransactionIDs:** 0
- **Fraud cases:** 20,663 (3.499%)
- **Legitimate cases:** 569,877
- **Class ratio (legit:fraud):** 27.6:1
- **TransactionDT range:** 86400.0 → 15811131.0 sec
- **Temporal span:** 182.0 days
- **Columns with 0% missing:** 20
- **Columns with 0–20% missing:** 162
- **Columns with 20–80% missing:** 157
- **Columns with >80% missing:** 55

## train_identity.csv

- **Rows:** 144,233
- **Columns:** 41
- **Duplicate TransactionIDs:** 0

## TransactionID Linkage

- **Transactions with identity record:** 144,233 (24.42%)
- **Transactions without identity:** 446,307
- **Join type:** LEFT JOIN (transaction is primary)