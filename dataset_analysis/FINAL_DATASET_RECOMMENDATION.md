# FINAL DATASET RECOMMENDATION
# Adaptive UPI Fraud Detection System

Generated: 2026-08-29

## PRIMARY TRAINING DATASET
USE: BAF Base.csv

Why:
- 1,000,000 rows with 32 behavioral features
- Zero missing values, zero duplicates
- 8 months of data showing fraud drift (1.13% to 1.47%)
- Fraud label: 11,029 fraud / 988,971 legit (1.10%)
- Velocity, device, session behavioral features included
- device_fraud_count = CONSTANT ZERO - remove

Limitations:
- No user/session ID for entity-level sequences
- Anonymized categoricals (AA/AB/BA/BC)
- Represents bank fraud, not UPI payment fraud
- intended_balcon_amount: 74.25% negative (sentinel values)

## STREAMING / KAFKA DATASET
USE: PaySim paysim dataset.csv

Why:
- 6,362,620 rows with hourly step (1-743)
- Can replay events chronologically by step
- Fraud only in TRANSFER (0.77%) and CASH_OUT (0.18%)
- Has nameOrig, nameDest for entity tracking

Limitations:
- 99.99% customers have 1 transaction only - no entity history
- isFlaggedFraud catches 16/8213 fraud cases - discard
- 1,687,138 balance mismatches
- REMOVE: newbalanceOrig, newbalanceDest, isFlaggedFraud

## CONCEPT DRIFT DATASET
USE: BAF Base.csv + Variant files

Fraud rate by month:
Month 0: 1.13%, Month 1: 0.94%, Month 2: 0.87%
Month 3: 0.92%, Month 4: 1.14%, Month 5: 1.18%
Month 6: 1.34%, Month 7: 1.47%

Variants I, II, IV: same 32 columns
Variants III, V: +2 unknown columns (x1, x2) - do not use as primary

## BACKGROUND EDA ONLY
USE: Credit Card Fraud 10K (limited to EDA)

Why limited:
- Only 10,000 rows, 151 fraud cases
- No user ID, no date, only transaction_hour
- velocity_last_24h artificially capped at 9
- Unknown provenance - possibly toy dataset

## EXPERIMENT STRUCTURE
BAF months 0-4 -> Training
BAF month 5 -> Validation  
BAF months 6-7 -> Testing
BAF Variants I/II/IV -> Robustness
PaySim -> Kafka streaming replay

## NEXT IMPLEMENTATION STEP
1. Feature engineering on BAF Base.csv
2. Temporal train/val/test split (months 0-4 / 5 / 6-7)
3. Class balancing strategy (weights or undersample)
4. Baseline XGBoost/LightGBM on tabular BAF
5. Drift experiment: train months 0-4, eval on 5, 6, 7 separately
6. Prepare PaySim streaming replay pipeline (step-ordered events)
