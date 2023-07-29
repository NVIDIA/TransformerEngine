from abc import ABC, abstractmethod
from typing import TypeVar

DistributedExecutionEnvType = TypeVar(
    "DistributedExecutionEnvType", bound="ExecutionEnv"
)


class DistributedGroup(ABC):
    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError()


class ExecutionEnv(ABC):
    fp8: bool
    training: bool
    distributed_group: DistributedGroup | None
