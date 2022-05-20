#!/usr/bin/env python3
from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.tensor import BaguaTensor
from bagua.torch_api.data_parallel.bagua_distributed import BaguaDistributedDataParallel
from bagua.torch_api.algorithms import Algorithm, AlgorithmImpl
from bagua.torch_api.communication import BaguaProcessGroup
from typing import List
import torch
import bagua.torch_api as bagua
from torch.optim import Optimizer
import sys

__all__ = [
    "AllreduceAlgorithm"
]


class AllreduceAlgorithmImpl(AlgorithmImpl):
    def __init__(
        self,
        process_group: BaguaProcessGroup,
        hierarchical: bool = True,
        communication_interval: int = 1,
        optimizer: Optimizer = None
    ):
        """
        Implementation of the decantrlized model allreduce algorithm.

        Args:
            process_group (BaguaProcessGroup): The process group to work on.
            hierarchical (bool): Enable hierarchical communication.
            communication_interval (int): Number of iterations between two communication steps.
            optimizer (Optimizer): A torch Optimizer initialized with model parameters.
        """
        super(AllreduceAlgorithmImpl, self).__init__(process_group)
        self.hierarchical = hierarchical
        self.communication_interval = communication_interval
        self.x_buffered = 0
        self.optimizer = optimizer
        self.param_size = 0
        for layer in optimizer.param_groups[0]['params']:
            self.param_size += layer.numel()
        self.m_send = torch.zeros(self.param_size, dtype=torch.float32).cuda()


    def _should_communicate(self, bagua_ddp: BaguaDistributedDataParallel) -> bool:
        cur_step = bagua_ddp.bagua_train_step_counter - 1
        return cur_step % self.communication_interval == 0

    def init_tensors(self, bagua_ddp: BaguaDistributedDataParallel) -> List[BaguaTensor]:
        parameters = bagua_ddp.bagua_build_params()
        self.tensors = [
            param.ensure_bagua_tensor(name, bagua_ddp.bagua_module_name)
            for name, param in parameters.__reversed__()
        ]
        return self.tensors

    def tensors_to_buckets(
        self, tensors: List[List[BaguaTensor]], do_flatten: bool
    ) -> List[BaguaBucket]:
        all_tensors = []
        for idx, bucket in enumerate(tensors):
            all_tensors.extend(bucket)

        bagua_bucket = BaguaBucket(all_tensors, flatten=do_flatten, name=str(0))

        return [bagua_bucket]

    def init_forward_pre_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook(input):
            return

        return hook

    def init_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook(parameter_name, parameter):
            return

        return hook

    def init_post_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook():
            return

        return hook
    
    def init_post_optimizer_step_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook(optimizer: torch.optim.Optimizer):
            if not self._should_communicate(bagua_ddp):
                return

            def pack(tensors):
                """Packs a list of tensors into one buffer for sending to other workers"""
                buffer = torch.cat([t.view(-1) for t in tensors])  # copies
                shapes = [tensor.shape for tensor in tensors]
                return buffer, shapes

            def unpack(buffer, shapes):
                """Provides pointers to tensors of original `shapes` in a flat-packed buffer."""
                idx = 0
                entries = []
                for tensor_shape in shapes:
                    end = idx + tensor_shape.numel()
                    entries.append(buffer[idx:end].view(size=tensor_shape))
                    idx = end

                return entries
            

            # init X_i^(t + 1/2)
            x_i = [layer for layer in optimizer.param_groups[0]['params']]
            x_i_buffered, shapes = pack(x_i)
            self.m_send.copy_(x_i_buffered)

            bagua.allreduce_inplace(self.m_send, op=bagua.ReduceOp.AVG)

            # unpack x_buffered
            x_i_2 = unpack(self.m_send, shapes)

            # overwrite current weights
            for idx, layer in enumerate(optimizer.param_groups[0]['params']):
                layer.data.copy_(x_i_2[idx])

        return hook

    def _init_states(self, bucket: BaguaBucket):
        weight_tensor = bucket.flattened_tensor()
        bucket._peer_weight = weight_tensor.ensure_bagua_tensor("peer_weight")

    def init_operations(
        self,
        bagua_ddp: BaguaDistributedDataParallel,
        bucket: BaguaBucket,
    ):
        self._init_states(bucket)
        torch.cuda.synchronize()
        bucket.clear_ops()


class AllreduceAlgorithm(Algorithm):
    def __init__(
        self,
        hierarchical: bool = True,
        communication_interval: int = 1,
        optimizer: Optimizer = None
    ):
        """
        Create an instance of the decentralized model allreduce algorithm.

        Args:
            hierarchical (bool): Enable hierarchical communication.
            communication_interval (int): Number of iterations between two communication steps.
            optimizer (Optimizer): A torch Optimizer initialized with model parameters.
        """
        self.hierarchical = hierarchical
        self.communication_interval = communication_interval
        self.optimizer = optimizer

    def reify(self, process_group: BaguaProcessGroup) -> AllreduceAlgorithmImpl:
        return AllreduceAlgorithmImpl(
            process_group,
            hierarchical=self.hierarchical,
            communication_interval=self.communication_interval,
            optimizer=self.optimizer
        )
