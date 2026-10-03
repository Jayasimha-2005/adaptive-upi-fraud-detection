def amount_bucket(amount):
    amount = float(amount)
    if amount < 100:
        return "LOW"
    if amount < 1000:
        return "MEDIUM"
    if amount < 10000:
        return "HIGH"
    return "VERY_HIGH"
