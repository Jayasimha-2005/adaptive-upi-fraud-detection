import csv, os

OUT = 'dataset_analysis'

# =============================================
# 1. DATASET INVENTORY CSV
# =============================================
inventory = [
    ['Dataset', 'Directory', 'File', 'Format', 'Size_MB', 'Rows', 'Columns', 'Approx_Memory_MB', 'Status'],
    ['BAF-Base', 'Datasets/BAF', 'Base.csv', 'CSV', '203.6', '1000000', '32', '480.7', 'Analyzed'],
    ['BAF-Variant-I', 'Datasets/BAF', 'Variant I.csv', 'CSV', '203.6', '1000000', '32', '480.7', 'Headers Verified'],
    ['BAF-Variant-II', 'Datasets/BAF', 'Variant II.csv', 'CSV', '203.6', '1000000', '32', '480.7', 'Headers Verified'],
    ['BAF-Variant-III', 'Datasets/BAF', 'Variant III.csv', 'CSV', '240.4', '1000000', '34', '510.0', 'Headers Verified (2 extra cols)'],
    ['BAF-Variant-IV', 'Datasets/BAF', 'Variant IV.csv', 'CSV', '203.6', '1000000', '32', '480.7', 'Headers Verified'],
    ['BAF-Variant-V', 'Datasets/BAF', 'Variant V.csv', 'CSV', '240.5', '1000000', '34', '510.0', 'Headers Verified (2 extra cols)'],
    ['PaySim', 'Datasets/Paysim', 'paysim dataset.csv', 'CSV', '493.5', '6362620', '11', '546.5', 'Fully Analyzed'],
    ['CreditCard10K', 'Datasets/Credit card Fraud detection', 'credit_card_fraud_10k.csv', 'CSV', '0.36', '10000', '10', '1.28', 'Fully Analyzed'],
]

with open(f'{OUT}/dataset_inventory.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(inventory)

print("dataset_inventory.csv saved.")

# =============================================
# 2. DATASET SUMMARY CSV
# =============================================
summary = [
    ['Dataset', 'Rows', 'Columns', 'Fraud_Count', 'Legit_Count', 'Fraud_Rate_Pct', 'Class_Ratio',
     'Missing_Values', 'Duplicate_Rows', 'Has_Timestamp', 'Has_Entity_ID', 'Has_Transaction_ID',
     'Temporal_Span', 'Data_Type', 'Primary_Use'],
    ['BAF-Base', '1000000', '32', '11029', '988971', '1.10', '89.7:1',
     'None', '0', 'Partial (month only, 0-7)', 'No', 'No',
     '8 months (month 0-7)', 'Synthetic (bank account fraud)', 'Primary Training'],
    ['PaySim', '6362620', '11', '8213', '6354407', '0.13', '773.8:1',
     'None', 'Unknown', 'Yes (step = hour, 1-743)', 'Yes (nameOrig, nameDest)', 'No',
     '743 hours (~31 days)', 'Simulated (mobile money)', 'Streaming/Secondary'],
    ['CreditCard10K', '10000', '10', '151', '9849', '1.51', '65.2:1',
     'None', '0', 'Partial (hour 0-23)', 'No', 'Yes (transaction_id)',
     'Cannot be determined', 'Synthetic (credit card)', 'Benchmark/Discard'],
]

with open(f'{OUT}/dataset_summary.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(summary)

print("dataset_summary.csv saved.")

# =============================================
# 3. COLUMN DICTIONARY CSV
# =============================================
col_dict = [
    ['Dataset', 'Column', 'Data_Type', 'Unique_Values', 'Missing_Pct', 'Example_Values', 'Description', 'Role', 'Leakage_Risk']
]

# BAF columns
baf_cols = [
    ['BAF', 'fraud_bool', 'int64', '2', '0%', '[0, 1]', 'Fraud label: 1=fraud, 0=legitimate', 'Target', 'None'],
    ['BAF', 'income', 'float64', '9', '0%', '[0.1, 0.2, 0.3, ..., 0.9]', 'Customer income (normalized, 9 bands)', 'Numerical feature', 'None'],
    ['BAF', 'name_email_similarity', 'float64', '~999K', '0%', '[0.0001, ..., 0.9999]', 'Similarity between name and email (0-1)', 'Numerical feature', 'None'],
    ['BAF', 'prev_address_months_count', 'int64', '374', '0%', '[-1, 0, ..., 383]', 'Months at previous address (-1=unknown)', 'Numerical feature', 'None'],
    ['BAF', 'current_address_months_count', 'int64', '423', '0%', '[-1, 0, ..., 428]', 'Months at current address (-1=unknown)', 'Numerical feature', 'None'],
    ['BAF', 'customer_age', 'int64', '9', '0%', '[10, 20, 30, 40, 50, 60, 70, 80, 90]', 'Age in decade bands', 'Numerical feature', 'None'],
    ['BAF', 'days_since_request', 'float64', '~989K', '0%', '[0.0067, 0.010, ...]', 'Days since application request was made', 'Numerical feature', 'None'],
    ['BAF', 'intended_balcon_amount', 'float64', '~995K', '0%', '[-15.5, ..., 112.9]', 'Intended balance transfer amount (74% negative = sentinel values)', 'Numerical feature', 'None'],
    ['BAF', 'payment_type', 'str', '5', '0%', '[AA, AB, AC, AD, AE]', 'Anonymized payment type categories', 'Categorical feature', 'None'],
    ['BAF', 'zip_count_4w', 'int64', '6306', '0%', '[1, ..., 6700]', 'Number of applications from same zip in 4 weeks', 'Numerical feature', 'None'],
    ['BAF', 'velocity_6h', 'float64', '~999K', '0%', '[-170.6, ..., 16715.6]', 'Application velocity in last 6 hours', 'Numerical feature', 'None'],
    ['BAF', 'velocity_24h', 'float64', '~999K', '0%', '[1300.3, ..., 9506.9]', 'Application velocity in last 24 hours', 'Numerical feature', 'None'],
    ['BAF', 'velocity_4w', 'float64', '~999K', '0%', '[2825.7, ..., 6994.8]', 'Application velocity in last 4 weeks', 'Numerical feature', 'None'],
    ['BAF', 'bank_branch_count_8w', 'int64', '2326', '0%', '[0, ..., 2385]', 'Bank branch count in 8 weeks', 'Numerical feature', 'None'],
    ['BAF', 'date_of_birth_distinct_emails_4w', 'int64', '40', '0%', '[0, ..., 39]', 'Distinct emails with same DOB in 4 weeks', 'Numerical feature', 'None'],
    ['BAF', 'employment_status', 'str', '7', '0%', '[CA, CB, CC, CD, CE, CF, CG]', 'Anonymized employment status', 'Categorical feature', 'None'],
    ['BAF', 'credit_risk_score', 'int64', '551', '0%', '[-170, ..., 389]', 'Credit risk score (higher = better)', 'Numerical feature', 'None'],
    ['BAF', 'email_is_free', 'int64', '2', '0%', '[0, 1]', 'Whether email domain is free provider', 'Binary feature', 'None'],
    ['BAF', 'housing_status', 'str', '7', '0%', '[BA, BB, BC, BD, BE, BF, BG]', 'Anonymized housing status', 'Categorical feature', 'None'],
    ['BAF', 'phone_home_valid', 'int64', '2', '0%', '[0, 1]', 'Whether home phone is valid', 'Binary feature', 'None'],
    ['BAF', 'phone_mobile_valid', 'int64', '2', '0%', '[0, 1]', 'Whether mobile phone is valid', 'Binary feature', 'None'],
    ['BAF', 'bank_months_count', 'int64', '33', '0%', '[-1, 0, ..., 32]', 'Months with current bank (-1=unknown)', 'Numerical feature', 'None'],
    ['BAF', 'has_other_cards', 'int64', '2', '0%', '[0, 1]', 'Whether applicant has other cards', 'Binary feature', 'None'],
    ['BAF', 'proposed_credit_limit', 'float64', '12', '0%', '[190, 200, 500, 1000, 1500, 2100]', 'Proposed credit limit (12 discrete values)', 'Numerical feature', 'None'],
    ['BAF', 'foreign_request', 'int64', '2', '0%', '[0, 1]', 'Whether application is from foreign IP', 'Binary feature', 'None'],
    ['BAF', 'source', 'str', '2', '0%', '[INTERNET, TELEAPP]', 'Application channel', 'Categorical feature', 'None'],
    ['BAF', 'session_length_in_minutes', 'float64', '~995K', '0%', '[-1, 0.001, ..., 85.9]', 'Session duration (-1=TELEAPP)', 'Numerical feature', 'None'],
    ['BAF', 'device_os', 'str', '5', '0%', '[linux, other, windows, x11, macintosh]', 'Operating system of device', 'Categorical feature', 'None'],
    ['BAF', 'keep_alive_session', 'int64', '2', '0%', '[0, 1]', 'Whether session kept alive', 'Binary feature', 'None'],
    ['BAF', 'device_distinct_emails_8w', 'int64', '4', '0%', '[-1, 0, 1, 2]', 'Distinct emails from device in 8w (-1=unknown)', 'Numerical feature', 'None'],
    ['BAF', 'device_fraud_count', 'int64', '1', '0%', '[0]', 'CONSTANT ZERO - prior fraud from device. Uninformative.', 'Derived feature', 'Potential leakage if nonzero'],
    ['BAF', 'month', 'int64', '8', '0%', '[0, 1, 2, 3, 4, 5, 6, 7]', 'Month index (0-7, only temporal info available)', 'Timestamp', 'None'],
]
col_dict.extend(baf_cols)

# PaySim columns
paysim_cols = [
    ['PaySim', 'step', 'int64', '743', '0%', '[1, 2, ..., 743]', 'Hour of simulation (1-743, ~1 month)', 'Timestamp', 'None'],
    ['PaySim', 'type', 'str', '5', '0%', '[PAYMENT, TRANSFER, CASH_OUT, CASH_IN, DEBIT]', 'Transaction type', 'Categorical feature', 'None'],
    ['PaySim', 'amount', 'float64', 'Very high', '0%', '[0.00, ..., 92445516.64]', 'Transaction amount', 'Numerical feature', 'None'],
    ['PaySim', 'nameOrig', 'str', 'Very high', '0%', '[C1231006815, ...]', 'Originator ID (C=customer, M=merchant)', 'Identifier', 'None'],
    ['PaySim', 'oldbalanceOrg', 'float64', 'Very high', '0%', '[0, ..., large]', 'Balance of originator before transaction', 'Numerical feature', 'None'],
    ['PaySim', 'newbalanceOrig', 'float64', 'Very high', '0%', '[0, ..., large]', 'Balance of originator after transaction', 'Numerical feature', 'Potential leakage'],
    ['PaySim', 'nameDest', 'str', 'Very high', '0%', '[C553264065, M1979787155, ...]', 'Destination ID (C=customer, M=merchant)', 'Identifier', 'None'],
    ['PaySim', 'oldbalanceDest', 'float64', 'Very high', '0%', '[0, ..., large]', 'Balance of destination before transaction', 'Numerical feature', 'None'],
    ['PaySim', 'newbalanceDest', 'float64', 'Very high', '0%', '[0, ..., large]', 'Balance of destination after transaction', 'Numerical feature', 'Potential leakage'],
    ['PaySim', 'isFraud', 'int64', '2', '0%', '[0, 1]', 'Fraud label: 1=fraud, 0=legitimate', 'Target', 'None'],
    ['PaySim', 'isFlaggedFraud', 'int64', '2', '0%', '[0, 1]', 'System flag for large TRANSFER fraud (only 16 flagged vs 8213 actual) - unreliable', 'Administrative/meta column', 'High leakage risk'],
]
col_dict.extend(paysim_cols)

# Credit Card columns
cc_cols = [
    ['CreditCard', 'transaction_id', 'int64', '10000', '0%', '[1, 2, ..., 10000]', 'Sequential integer ID, no user linkage', 'Identifier', 'None'],
    ['CreditCard', 'amount', 'float64', '8788', '0%', '[0.0, ..., 1471.04]', 'Transaction amount (1 zero amount present)', 'Numerical feature', 'None'],
    ['CreditCard', 'transaction_hour', 'int64', '24', '0%', '[0, 1, ..., 23]', 'Hour of transaction (0-23)', 'Timestamp', 'None'],
    ['CreditCard', 'merchant_category', 'str', '5', '0%', '[Electronics, Travel, Grocery, Food, Clothing]', 'Merchant category (5 categories)', 'Categorical feature', 'None'],
    ['CreditCard', 'foreign_transaction', 'int64', '2', '0%', '[0, 1]', 'Whether transaction is foreign', 'Binary feature', 'None'],
    ['CreditCard', 'location_mismatch', 'int64', '2', '0%', '[0, 1]', 'Whether location mismatches cardholder address', 'Binary feature', 'None'],
    ['CreditCard', 'device_trust_score', 'int64', '75', '0%', '[25, ..., 99]', 'Device trust score (25-99)', 'Numerical feature', 'None'],
    ['CreditCard', 'velocity_last_24h', 'int64', '10', '0%', '[0, 1, ..., 9]', 'Number of transactions in last 24h (0-9)', 'Numerical feature', 'None'],
    ['CreditCard', 'cardholder_age', 'int64', '52', '0%', '[18, ..., 69]', 'Cardholder age in years', 'Numerical feature', 'None'],
    ['CreditCard', 'is_fraud', 'int64', '2', '0%', '[0, 1]', 'Fraud label: 1=fraud, 0=legitimate', 'Target', 'None'],
]
col_dict.extend(cc_cols)

with open(f'{OUT}/column_dictionary.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(col_dict)

print("column_dictionary.csv saved.")

# =============================================
# 4. FRAUD LABEL ANALYSIS CSV
# =============================================
fraud_analysis = [
    ['Dataset', 'Target_Column', 'Label_Type', 'Positive_Class', 'Negative_Class',
     'Fraud_Count', 'Legit_Count', 'Total_Rows', 'Fraud_Pct', 'Class_Ratio',
     'Unknown_Labels', 'Label_Encoding', 'Label_Trustworthy', 'Notes'],
    ['BAF-Base', 'fraud_bool', 'Binary', '1', '0',
     '11029', '988971', '1000000', '1.10%', '89.7:1',
     '0', 'Binary integer (0/1)', 'Yes',
     'Fraud rate increases from month 0 (1.13%) to month 7 (1.47%), showing drift'],
    ['PaySim', 'isFraud', 'Binary', '1', '0',
     '8213', '6354407', '6362620', '0.13%', '773.8:1',
     '0', 'Binary integer (0/1)', 'Yes (simulated)',
     'Fraud ONLY in TRANSFER (0.77%) and CASH_OUT (0.18%) types. isFlaggedFraud catches only 16 of 8213 fraud cases.'],
    ['CreditCard10K', 'is_fraud', 'Binary', '1', '0',
     '151', '9849', '10000', '1.51%', '65.2:1',
     '0', 'Binary integer (0/1)', 'Unknown',
     'Dataset appears synthetic. Label generation method unknown. No source documentation in repository.'],
]

with open(f'{OUT}/fraud_label_analysis.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(fraud_analysis)

print("fraud_label_analysis.csv saved.")

# =============================================
# 5. TEMPORAL ANALYSIS CSV
# =============================================
temporal = [
    ['Dataset', 'Timestamp_Column', 'Earliest', 'Latest', 'Time_Span', 'Granularity',
     'Missing_Timestamps', 'Timezone', 'Chronologically_Ordered', 'Temporal_Drift_Detected',
     'Streaming_Suitable', 'LSTM_Temporal_Support'],
    ['BAF-Base', 'month', '0 (month 0)', '7 (month 7)', '8 months', 'Monthly (coarse)',
     '0', 'Unknown', 'No (not transaction-level)', 'Yes - fraud rate rises over time',
     'No - too coarse for streaming', 'Weak - only 8 time buckets'],
    ['PaySim', 'step', '1 (hour 1)', '743 (hour 743)', '~31 days (1 month)', 'Hourly',
     '0', 'Unknown (simulated)', 'Yes (by step)', 'Cannot determine - single month simulation',
     'Yes - can replay by step', 'Moderate - but 99.99% of users have only 1 transaction'],
    ['CreditCard10K', 'transaction_hour', '0 (midnight)', '23 (11pm)', 'No date info - hour only', 'Sub-daily (hour)',
     '0', 'Unknown', 'Unknown - no date', 'Cannot determine',
     'No - no date/timestamp', 'Not suitable - no date, no sequence'],
]

with open(f'{OUT}/temporal_analysis.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(temporal)

print("temporal_analysis.csv saved.")

# =============================================
# 6. DATA QUALITY REPORT CSV
# =============================================
quality = [
    ['Dataset', 'Missing_Count', 'Missing_Pct', 'Duplicate_Rows', 'Duplicate_IDs',
     'Negative_Amounts', 'Zero_Amounts', 'Invalid_Values', 'Constant_Columns',
     'Balance_Mismatches', 'Suspicious_Columns', 'Overall_Quality'],
    ['BAF-Base', '0', '0%', '0', 'N/A (no ID column)', 'N/A', 'N/A',
     'prev_address_months_count: -1 as sentinel; bank_months_count: -1 as sentinel; velocity_6h can be negative (-170.6); intended_balcon_amount: 74.25% negative (sentinel values)',
     'device_fraud_count: ALL ZEROS (1M rows) - uninformative constant column',
     'N/A', 'device_fraud_count (constant); intended_balcon_amount (mixed sentinel/real)',
     'HIGH - complete, no missing, minimal issues'],
    ['PaySim', '0', '0%', 'Not fully checked', 'Not verified',
     'None detected in sample', '0 (min=0.00)', 
     'Balance accounting: 1,687,138 mismatches where newbalanceOrig != oldbalanceOrg - amount',
     'isFlaggedFraud: only catches 16/8213 fraud cases (0.19% recall)',
     '1,687,138', 'isFlaggedFraud (near-useless fraud flag); balance mismatches',
     'MODERATE - complete data but significant balance accounting inconsistencies'],
    ['CreditCard10K', '0', '0%', '0', '0 (all unique)', '0', '1 (transaction 1 has amount=0.0)',
     'velocity_last_24h capped at 9 (suspicious for real data); all values seem artificially bounded',
     'None', 'N/A',
     'velocity_last_24h (suspiciously capped); no date column; no user linkage',
     'LOW - likely toy/synthetic dataset with engineered feature distributions'],
]

with open(f'{OUT}/data_quality_report.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(quality)

print("data_quality_report.csv saved.")

# =============================================
# 7. LEAKAGE ANALYSIS CSV
# =============================================
leakage = [
    ['Dataset', 'Feature', 'Why_Suspicious', 'Leakage_Risk', 'Recommendation'],
    ['BAF', 'device_fraud_count', 'ALL ZEROS - if this were populated with post-event fraud counts from device, it would be leakage. Currently uninformative.', 'Confirmed leakage if nonzero in future; currently Safe (constant)', 'Remove from training features - zero variance, uninformative'],
    ['BAF', 'velocity_6h / velocity_24h / velocity_4w', 'These may incorporate real-time signals. If computed on the same batch including the fraud event, they are leakage.', 'Potential leakage', 'Use with caution. Ensure these are computed on data BEFORE the current transaction.'],
    ['BAF', 'date_of_birth_distinct_emails_4w', 'If computed post-approval, this aggregates fraud signals. If computed pre-application, safe.', 'Potential leakage', 'Requires source documentation to determine computation timing.'],
    ['PaySim', 'isFlaggedFraud', 'This is a SYSTEM FLAG that theoretically flags fraud AFTER detection. Using it as a feature would be direct leakage.', 'Confirmed leakage', 'Remove completely from feature set. Only use isFraud as target.'],
    ['PaySim', 'newbalanceOrig', 'Balance AFTER transaction. Computed simultaneously with fraud decision. In real-time, this is not available before fraud scoring.', 'High leakage risk', 'Remove from feature set for real-time inference. Use oldbalanceOrg only.'],
    ['PaySim', 'newbalanceDest', 'Balance of destination AFTER transaction. Same issue as newbalanceOrig.', 'High leakage risk', 'Remove from feature set for real-time inference.'],
    ['CreditCard', 'velocity_last_24h', 'If this count includes the CURRENT transaction, it leaks information about transaction frequency at fraud time.', 'Potential leakage', 'Must verify: does count include current transaction? If yes, subtract 1.'],
    ['CreditCard', 'location_mismatch', 'Location mismatch requires knowing cardholder location at time of transaction. Verify this is available in real-time.', 'Low risk (if computed in real-time)', 'Safe if computed from device GPS vs. billing address. Verify computation method.'],
]

with open(f'{OUT}/leakage_analysis.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(leakage)

print("leakage_analysis.csv saved.")

# =============================================
# 8. SEQUENCE SUITABILITY CSV
# =============================================
sequence = [
    ['Dataset', 'Entity_Column', 'Unique_Entities', 'Avg_Txn_Per_Entity', 'Median_Txn_Per_Entity',
     'Max_Txn_Per_Entity', 'Pct_Single_Txn', 'Timestamp_Quality', 'Sequence_Possible',
     'LSTM_GRU_Suitability', 'Reasoning'],
    ['BAF-Base', 'None (no entity ID)', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
     'Month only (0-7)', 'No',
     'Not suitable',
     'No customer/session/device ID exists. Each row is a standalone application. Cannot form sequences. Month granularity too coarse.'],
    ['PaySim', 'nameOrig (customer)', '~6.35M unique', '~1.00', '1.0', '2',
     '99.99%', 'Hourly step (good)',
     'Theoretically yes but practically no',
     'Weak',
     '99.99% of customers have exactly 1 transaction. Cannot build meaningful LSTM sequences from single-transaction customers. The step column provides good temporal ordering but entity history is absent.'],
    ['CreditCard10K', 'None (transaction_id only)', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
     'Hour only (no date)', 'No',
     'Not suitable',
     'transaction_id is sequential integer, not a user identifier. No user linking possible. Hour-only timestamp with no date cannot support temporal sequences.'],
]

with open(f'{OUT}/sequence_suitability.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(sequence)

print("sequence_suitability.csv saved.")

# =============================================
# 9. STREAMING SUITABILITY CSV
# =============================================
streaming = [
    ['Dataset', 'Timestamp_Available', 'Transaction_Ordering', 'Unique_Tx_ID', 'Event_Frequency',
     'Entity_Info', 'Feature_At_Event_Time', 'Replay_Possible', 'Streaming_Rating', 'Notes'],
    ['BAF-Base', 'Month only', 'By month (coarse)', 'No', '~125K/month avg',
     'No entity ID', 'Most features available at event time', 'Partial (monthly batch only)',
     'Poor',
     'Monthly granularity unsuitable for event streaming. Can only replay in monthly batches. No fine-grained event ordering.'],
    ['PaySim', 'Step (hour)', 'By step (hourly)', 'No (no tx ID)', '~8,563 events/hour avg',
     'nameOrig, nameDest', 'oldbalanceOrg/Dest available; newbalance is post-event',
     'Yes - replay by step order',
     'Good',
     'Can replay events in step order. High volume. Missing transaction UUID must be generated. newbalanceOrig/Dest must be excluded (post-event).'],
    ['CreditCard10K', 'Hour only (no date)', 'Unknown (no date)', 'Yes (transaction_id)', '~417/hour avg theoretical',
     'None', 'All features appear pre-event', 'No (no date for ordering)',
     'Poor',
     'No date makes chronological replay impossible. 10K rows too small for meaningful streaming experiments.'],
]

with open(f'{OUT}/streaming_suitability.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(streaming)

print("streaming_suitability.csv saved.")

# =============================================
# 10. DATASET COMPARISON CSV
# =============================================
comparison = [
    ['Criterion', 'BAF-Base', 'PaySim', 'CreditCard10K'],
    ['Rows', '1,000,000', '6,362,620', '10,000'],
    ['Columns', '32', '11', '10'],
    ['Fraud Count', '11,029', '8,213', '151'],
    ['Fraud Rate', '1.10%', '0.13%', '1.51%'],
    ['Class Ratio', '89.7:1', '773.8:1', '65.2:1'],
    ['Timestamp Quality', 'MODERATE - Month only (0-7)', 'EXCELLENT - Hourly step (1-743)', 'POOR - Hour only, no date'],
    ['Entity History', 'POOR - None', 'MODERATE - nameOrig (but 99.99% single-tx)', 'POOR - None'],
    ['LSTM/GRU Suitability', 'POOR - Not suitable', 'WEAK - No entity history', 'POOR - Not suitable'],
    ['Concept Drift', 'GOOD - Fraud rate change over 8 months', 'POOR - 1-month simulation', 'POOR - No time context'],
    ['Streaming/Kafka', 'POOR - Monthly batches', 'GOOD - Hourly step, 6M rows', 'POOR - No date'],
    ['UPI Relevance', 'POOR - Bank account fraud, not UPI', 'POOR - Mobile money, not UPI', 'POOR - Credit card fraud, not UPI'],
    ['Feature Richness', 'EXCELLENT - 32 cols, behavioral features', 'MODERATE - 11 cols, balance-based', 'MODERATE - 10 cols, basic features'],
    ['Behavioral Features', 'EXCELLENT - Velocity, device, session', 'MODERATE - Balance deltas, type', 'MODERATE - Velocity (1 col), trust score'],
    ['Data Quality', 'HIGH - No missing, no duplicates', 'MODERATE - Balance inconsistencies', 'MODERATE - Appears synthetic/toy'],
    ['Data Provenance', 'Synthetic (IEEE competition)', 'Simulated (Mobile money sim)', 'Unknown (possibly toy)'],
    ['Data Volume Score (1-10)', '8', '9', '2'],
    ['Fraud Label Quality (1-10)', '9', '7', '5'],
    ['Temporal Info (1-10)', '5', '8', '1'],
    ['LSTM/GRU (1-10)', '3', '5', '1'],
    ['Feature Richness (1-10)', '8', '5', '4'],
    ['Behavioral Features (1-10)', '7', '5', '3'],
    ['UPI Relevance (1-10)', '3', '3', '2'],
    ['Class Balance (1-10)', '6', '3', '6'],
    ['Concept Drift (1-10)', '7', '5', '1'],
    ['Streaming (1-10)', '4', '7', '2'],
    ['Reproducibility (1-10)', '9', '8', '4'],
    ['OVERALL SCORE (1-10)', '6.4', '5.5', '2.7'],
]

with open(f'{OUT}/dataset_comparison.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(comparison)


print("dataset_comparison.csv saved.")

print("\nAll CSV files saved to:", OUT)
