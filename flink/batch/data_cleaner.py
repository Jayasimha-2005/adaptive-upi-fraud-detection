import pandas as pd

def clean_transactions(df):
    out = df.copy()
    if "TransactionDT" in out:
        out["event_time_seconds"] = pd.to_numeric(
            out["TransactionDT"], errors="coerce"
        ).fillna(0)
    if "TransactionAmt" in out:
        out["TransactionAmt"] = pd.to_numeric(
            out["TransactionAmt"], errors="coerce"
        ).fillna(0.0)
    if "isFraud" in out:
        out["isFraud"] = pd.to_numeric(
            out["isFraud"], errors="coerce"
        ).fillna(0).astype("int8")
    return out
