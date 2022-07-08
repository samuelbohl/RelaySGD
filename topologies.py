import networkx as nx
import random
import math

__all__ = [
    "Topology",
    "ChainTopology",
    "BinaryTreeTopology",
    "DoubleBinaryTreeTopology",
]

class Topology():
    def __init__(self, num_workers):
        self.num_workers = num_workers
    
    def neighbors(self, worker):
        return []

class ChainTopology(Topology):
    def __init__(self, num_workers):
        super().__init__(num_workers=num_workers)
    
    def neighbors(self, worker):
        if self.num_workers == 1:
            return []
        elif worker == 0:
            return [1]
        elif worker == self.num_workers - 1:
            return [worker - 1]
        else:
            return [worker - 1, worker + 1]

class BinaryTreeTopology(Topology):
    def __init__(self, num_workers):
        super().__init__(num_workers=num_workers)

    def neighbors(self, worker):
        if self.num_workers == 1:
            return []
        elif worker == 0:
            return [1]
        else:
            parent = worker // 2
            children = [worker * 2, worker * 2 + 1]
            children = [c for c in children if c < self.num_workers]
            return [parent, *children]

class DoubleBinaryTreeTopology(BinaryTreeTopology):
    def __init__(self, num_workers):
        super().__init__(num_workers=num_workers)

    def neighbors(self, worker):
        neighbors = super().neighbors(worker)
        reverse_neighbors = self.reverse_neighbors(worker)
        return neighbors, reverse_neighbors

    def reverse_neighbors(self, worker):
        if worker == self.num_workers - 1:
            return [worker - 1]
        else:
            worker = self.num_workers - 1 - worker
            parent = worker // 2
            children = [worker * 2, worker * 2 + 1]
            children = [self.num_workers - 1 - c for c in children if c < self.num_workers]
            parent = self.num_workers - 1 - parent
            return [parent, *children]