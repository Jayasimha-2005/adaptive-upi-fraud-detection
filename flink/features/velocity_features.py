from collections import deque

class UserVelocityState:
    def __init__(self):
        self.events = deque()

    def add(self, timestamp_ms, amount):
        self.events.append((timestamp_ms, float(amount)))

    def remove_older_than(self, cutoff_ms):
        while self.events and self.events[0][0] < cutoff_ms:
            self.events.popleft()

    @property
    def count(self):
        return len(self.events)

    @property
    def total_amount(self):
        return sum(x[1] for x in self.events)
