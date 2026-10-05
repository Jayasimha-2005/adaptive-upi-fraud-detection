def join_transaction_identity(transactions, identity):
    if identity is None:
        return transactions.copy()
    if "TransactionID" not in transactions or "TransactionID" not in identity:
        return transactions.copy()
    return transactions.merge(identity, on="TransactionID", how="left")
