from collections import namedtuple, deque
import random
import heapq

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory():
    def __init__(self, size):
        self.memory = deque(maxlen=size)
        self.max_size = size
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class PrioritizedReplayMemory():
    def __init__(self, size):
        self.memory = []
        self.max_size = size
        self.id = 0
    def push(self, q_val, *args):
        if len(self.memory) < self.max_size:
            heapq.heappush(self.memory, (q_val, self.id, Transition(*args)))
        else:
            heapq.heappushpop(self.memory, (q_val, self.id, Transition(*args)))
        self.id += 1
    def sample(self, batch_size):
        sampled = random.sample(self.memory, batch_size)
        return [x[2] for x in sampled]
    def __len__(self):
        return len(self.memory)


