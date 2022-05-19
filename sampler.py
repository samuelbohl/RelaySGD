import math
import numpy as np
from torchvision import datasets
from torch.utils.data import Sampler, Dataset
from typing import Iterator, List
import random

__all__ = [
    "DistributedHeterogeneousSampler"
]

class DistributedHeterogeneousSampler(Sampler):
    r"""Sampler that restricts data loading to a subset of the dataset,
    which is non iid dirichlet distributed.
    Args:
        dataset: Dataset used for sampling.
        num_workers (int): Number of workers participating in
            distributed training. Default ist 1.
        rank (int): Rank of the current process within :attr:`num_workers`.
        alpha (float): Alpha parameter of the non iid Dirichlet distribution.
        seed (int): random seed used to in the dirichlet distribution. 
        This number should be identical across all processes in the distributed group. 
        Default: ``0``.
    """

    def __init__(self, dataset: Dataset, num_workers: int = 1,
                 rank: int = None, alpha: float = 1.0,
                 seed: int = 0) -> None:
        self.dataset = dataset
        self.num_workers = num_workers
        if rank is None:
            raise RuntimeError("Rank not specified")
        self.rank = rank
        self.alpha = alpha
        self.seed = seed
        self.indices = self.distribute_data_dirichlet(targets=self.dataset.targets, non_iid_alpha=self.alpha, n_workers=self.num_workers, seed=self.seed)
        
        self.equalize_size()

        self.total_size = sum([len(x) for x in self.indices])
        self.num_samples = len(self.indices[self.rank])
        
    def __iter__(self) -> Iterator:
        return iter(self.indices[self.rank])
    
    def __len__(self) -> int:
        return self.num_samples

    def distribute_data_dirichlet(self, targets, non_iid_alpha, n_workers, seed=0, num_auxiliary_workers=10) -> List:
        """Code from RelaySGD, Vogels et al. (non_iid_dirichlet.py)"""
        random_state = np.random.RandomState(seed=seed)
        
        num_indices = len(targets)
        num_classes = len(np.unique(targets))

        indices2targets = np.array(list(enumerate(targets)))
        random_state.shuffle(indices2targets)

        # partition indices.
        from_index = 0
        splitted_targets = []
        num_splits = math.ceil(n_workers / num_auxiliary_workers)
        split_n_workers = [
            num_auxiliary_workers
            if idx < num_splits - 1
            else n_workers - num_auxiliary_workers * (num_splits - 1)
            for idx in range(num_splits)
        ]
        split_ratios = [_n_workers / n_workers for _n_workers in split_n_workers]
        for idx, ratio in enumerate(split_ratios):
            to_index = from_index + int(num_auxiliary_workers / n_workers * num_indices)
            splitted_targets.append(
                indices2targets[
                    from_index : (num_indices if idx == num_splits - 1 else to_index)
                ]
            )
            from_index = to_index

        idx_batch = []
        for _targets in splitted_targets:
            # rebuild _targets.
            _targets = np.array(_targets)
            _targets_size = len(_targets)

            # use auxi_workers for this subset targets.
            _n_workers = min(num_auxiliary_workers, n_workers)
            n_workers = n_workers - num_auxiliary_workers

            # get the corresponding idx_batch.
            min_size = 0
            while min_size < int(0.50 * _targets_size / _n_workers):
                _idx_batch = [[] for _ in range(_n_workers)]
                for _class in range(num_classes):
                    # get the corresponding indices in the original 'targets' list.
                    idx_class = np.where(_targets[:, 1] == _class)[0]
                    idx_class = _targets[idx_class, 0]

                    # sampling.
                    try:
                        proportions = random_state.dirichlet(
                            np.repeat(non_iid_alpha, _n_workers)
                        )
                        # balance
                        proportions = np.array(
                            [
                                p * (len(idx_j) < _targets_size / _n_workers)
                                for p, idx_j in zip(proportions, _idx_batch)
                            ]
                        )
                        proportions = proportions / proportions.sum()
                        proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[
                            :-1
                        ]
                        _idx_batch = [
                            idx_j + idx.tolist()
                            for idx_j, idx in zip(
                                _idx_batch, np.split(idx_class, proportions)
                            )
                        ]
                        sizes = [len(idx_j) for idx_j in _idx_batch]
                        min_size = min([_size for _size in sizes])
                    except ZeroDivisionError:
                        pass
            idx_batch += _idx_batch
        return idx_batch

    def equalize_size(self, method="even"):
        if method == "cutoff_min":
            random.seed(self.seed)
            min_len = 9999999
            for idx in range(len(self.indices)):
                if len(self.indices[idx]) < min_len:
                    min_len = len(self.indices[idx])

            for idx in range(len(self.indices)):
                random.shuffle(self.indices[idx])
                self.indices[idx] = self.indices[idx][:min_len]
        elif method == "even":
            # flatten list
            flatten_list = sum(self.indices, [])
            # split evenly
            even_list = list()
            step_size = len(self.dataset) // self.num_workers
            for i in range(0, len(self.dataset), step_size):
                even_list.append(flatten_list[i:i+step_size])
            self.indices = even_list
            # shuffle the split data separately for each worker 
            for idx in range(len(self.indices)):
                random.shuffle(self.indices[idx])
        else:
            raise NotImplementedError
       