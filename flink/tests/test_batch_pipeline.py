import pandas as pd
from flink.batch.data_cleaner import clean_transactions
from flink.batch.dataset_joiner import join_transaction_identity

def test_clean_transactions():
    df = pd.DataFrame({
        "TransactionID": [1, 2],
        "TransactionAmt": ["10.5", None],
        "isFraud": [0, 1],
        "TransactionDT": [100, 200]
    })
    out = clean_transactions(df)
    assert out["TransactionAmt"].tolist() == [10.5, 0.0]
    assert out["isFraud"].tolist() == [0, 1]

def test_join_identity():
    tx = pd.DataFrame({"TransactionID": [1, 2]})
    identity = pd.DataFrame({"TransactionID": [1], "DeviceType": ["mobile"]})
    out = join_transaction_identity(tx, identity)
    assert len(out) == 2
