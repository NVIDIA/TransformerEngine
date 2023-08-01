from dataclasses import dataclass
from torch.distributed import ProcessGroup  # type: ignore


@dataclass
class Environment:
    fp8: bool
    training: bool
    distributed_group: ProcessGroup | None


CUR_ENV_STACK: list[Environment] = [Environment(False, False, None)]


def get_current_environment() -> Environment:
    return CUR_ENV_STACK[-1]
