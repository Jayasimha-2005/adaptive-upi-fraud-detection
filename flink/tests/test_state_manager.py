from flink.streaming.state_manager import user_total_state, user_count_state

def test_state():
    assert user_total_state().name == "user-total-amount"
    assert user_count_state().name == "user-transaction-count"
