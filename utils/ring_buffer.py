from collections import deque

class RingBuffer:
    def __init__(self, capacity: int):
        self.data = deque(maxlen=capacity)

    def push(self, x):
        self.data.append(x)

    def ready(self, need: int) -> bool:
        return len(self.data) >= need

    def last(self, k: int):
        # returns a list copy of the last k items
        if k > len(self.data):
            k = len(self.data)
        return list(self.data)[-k:]
