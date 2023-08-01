from abc import ABC, abstractmethod
from typing import TypeVar

DistributedExecutionEnvType = TypeVar(
    "DistributedExecutionEnvType", bound="ExecutionEnv"
)


class DistributedGroup(ABC):
    @abstractmethod
    def size(self) -> int:
        """
        Returns the number of parallel workers in the group.
        """
        raise NotImplementedError()

    @abstractmethod
    def rank(self) -> int:
        """
        Returns the index, in the range [0, size), of the current worker in the group.
        """
        raise NotImplementedError()


class ExecutionEnv(ABC):
    fp8: bool
    training: bool
    distributed_group: DistributedGroup | None
