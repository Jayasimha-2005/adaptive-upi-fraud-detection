import math

def build_historical_features(df):
    cols = [
        c for c in [
            "TransactionID", "TransactionAmt", "TransactionDT",
            "event_time_seconds", "ProductCD", "card1", "addr1", "isFraud"
        ] if c in df.columns
    ]
    out = df[cols].copy()
    if "TransactionAmt" in out:
        out["log_transaction_amount"] = (
            out["TransactionAmt"].clip(lower=0) + 1
        ).map(math.log1p)
    return out
