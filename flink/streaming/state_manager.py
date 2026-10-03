from pyflink.common import Types
from pyflink.datastream.state import ValueStateDescriptor

def user_total_state():
    return ValueStateDescriptor("user-total-amount", Types.DOUBLE())

def user_count_state():
    return ValueStateDescriptor("user-transaction-count", Types.LONG())
