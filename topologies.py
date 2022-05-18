import networkx as nx
import random
import math

__all__ = [
    "Topology",
    "ChainTopology",
    "BinaryTreeTopology",
    "DoubleBinaryTreeTopology",
    "RandomBinaryTreeTopology",
    "RandomDoubleBinaryTreeTopology"
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
    
class RandomBinaryTreeTopology(Topology):
    def __init__(self, num_workers, random_seed=0):
        super().__init__(num_workers=num_workers)
        self.random_seed = random_seed
        #fix random seed to get the same graph on all nodes
        random.seed(random_seed)

        self.height = int(math.log2(self.num_workers))
        self.G = nx.empty_graph(self.num_workers)
        
        # init tree
        self.build_tree()

    def neighbors(self, worker):
        return list(self.G.neighbors(worker))

    def build_tree(self, remapping=True):
        # create a list of indices and shuffle
        rand_indices = list(range(self.num_workers))
        random.shuffle(rand_indices)

        # create a random mapping using the shuffled list 
        mapping = dict()
        for ind in range(self.num_workers):
            mapping[ind] = rand_indices[ind]

        # generate balanced binary tree
        self.G = nx.balanced_tree(2, self.height)

        # remove redundant nodes
        self.G.remove_nodes_from(list(range(self.num_workers, 2 ** (self.height + 1))))

        # return mapping to perform is outside of this function
        if not remapping:
            return mapping

        # rename graph according to random mapping
        self.G = nx.relabel_nodes(self.G, mapping)

class RandomDoubleBinaryTreeTopology(RandomBinaryTreeTopology):
    def __init__(self, num_workers, random_seed=0):
        super().__init__(num_workers=num_workers, random_seed=random_seed)
        self.G_r = nx.empty_graph(self.num_workers)
        self.build_tree()

    def neighbors(self, worker):
        return list(self.G.neighbors(worker)), list(self.G_r.neighbors(worker))
    
    def build_tree(self):
        # build first tree and get random mapping (not applied)
        rand_mapping = super().build_tree(remapping=False)

        # build reverse tree
        reverse_mapping = dict([(i, self.num_workers - i - 1) for i in range(self.num_workers)])
        self.G_r = self.G.copy()
        self.G_r = nx.relabel_nodes(self.G_r, reverse_mapping)
        
        # now we apply the same random mapping from earlier to both trees
        self.G = nx.relabel_nodes(self.G, rand_mapping)
        self.G_r = nx.relabel_nodes(self.G_r, rand_mapping)