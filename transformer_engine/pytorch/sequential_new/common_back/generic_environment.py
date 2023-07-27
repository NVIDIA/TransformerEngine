from typing import TypeVar, Protocol

DistributedExecutionEnvType = TypeVar(
    "DistributedExecutionEnvType", bound="ExecutionEnv"
)


class DistributedGroup(Protocol):
    def size(self) -> int:
        raise NotImplementedError()


class ExecutionEnv(Protocol):
    fp8: bool
    training: bool
    distributed_group: DistributedGroup | None
