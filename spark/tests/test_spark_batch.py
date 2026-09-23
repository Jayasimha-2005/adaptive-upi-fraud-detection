import pytest


def test_batch_module_imports():
    pytest.importorskip("pyspark")
    from spark.batch.historical_pipeline import clean_transactions
    assert callable(clean_transactions)
