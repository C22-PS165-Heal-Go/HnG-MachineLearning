from collections import deque
import random

class HistoryBuffer(object):

    def __init__(self, capacity_size):
        self.capacity_size = capacity_size
        self.buffer = deque()

    def __len__(self):
        return len(self.buffer)

    def push(self, item):
        if len(self.buffer) < self.capacity_size:
            self.buffer.append(item)
        else:
            self.buffer.popleft()
            self.buffer.append(item)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def to_list(self):
        return list(self.buffer)

    def clear(self):
        self.buffer.clear()