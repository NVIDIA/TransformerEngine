from __future__ import annotations
from dataclasses import dataclass
from ..common_back.generic_environment import DistributedGroup, ExecutionEnv
from ...distributed import dist_group_type, get_distributed_world_size


class PytorchDistributedGroup(DistributedGroup):
    tp_group: dist_group_type
    __size: int

    def __init__(self, tp_group: dist_group_type):
        self.tp_group = tp_group
        self.__size = get_distributed_world_size(self.tp_group)

    def size(self) -> int:
        return self.__size


@dataclass
class PytorchExecutionEnv(ExecutionEnv):
    fp8: bool
    training: bool
    distributed_group: PytorchDistributedGroup | None
