def aggregate_transactions(events):
    amounts = [float(e.get("amount", 0)) for e in events]
    total = sum(amounts)
    return {
        "transaction_count": len(amounts),
        "total_amount": total,
        "average_amount": total / len(amounts) if amounts else 0.0
    }
