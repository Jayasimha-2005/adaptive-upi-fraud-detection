# Phase 1 — Data Split Report

## Join Summary

- Transaction rows: 590,540
- Identity rows:    144,233
- Joined rows:      590,540
- With identity:    140,810 (23.8%)
- Without identity: 449,730

## Temporal Split

Splitting strategy: **chronological** (70% / 15% / 15% by TransactionDT).
No random shuffling. No temporal overlap.

| Split | Rows | Fraud | Legit | Fraud % | Day Start | Day End |
|-------|------|-------|-------|---------|-----------|---------|
| Train | 434,176 | 15,252 | 418,924 | 3.513% | 1.0 | 127.0 |
| Validation | 77,822 | 2,637 | 75,185 | 3.389% | 127.0 | 155.0 |
| Test | 78,542 | 2,774 | 75,768 | 3.532% | 155.0 | 183.0 |

- No TransactionID overlap between splits: confirmed.
- Train DT max: day 127.0
- Val DT min:   day 127.0
- Test DT min:  day 155.0