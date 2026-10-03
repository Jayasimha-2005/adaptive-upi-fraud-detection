from spark.experiments.event_time_experiment import EVENTS, accepted_events


def test_out_of_order_sequence():
    assert [e.minute for e in EVENTS] == [1, 4, 2, 3]


def test_watermark_tolerance_changes_result():
    assert len(accepted_events(1)) <= len(accepted_events(5))
