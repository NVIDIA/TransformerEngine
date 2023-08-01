from __future__ import annotations
from dataclasses import dataclass
from ..common_back.generic_environment import DistributedGroup, ExecutionEnv
import torch.distributed
from torch.distributed import ProcessGroup


class PytorchDistributedGroup(DistributedGroup):
    tp_group: ProcessGroup
    __size: int
    __rank: int

    def __init__(self, tp_group: ProcessGroup):
        self.tp_group = tp_group
        self.__size = torch.distributed.get_world_size(self.tp_group)
        self.__rank = torch.distributed.get_rank(self.tp_group)

    def size(self) -> int:
        return self.__size

    def rank(self) -> int:
        return self.__rank


@dataclass
class PytorchExecutionEnv(ExecutionEnv):
    fp8: bool
    training: bool
    distributed_group: PytorchDistributedGroup | None
