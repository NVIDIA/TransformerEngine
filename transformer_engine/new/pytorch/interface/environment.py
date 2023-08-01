from contextlib import contextmanager
from torch.distributed import ProcessGroup  # type: ignore
from ..implementation.environment import Environment, CUR_ENV_STACK


@contextmanager
def environment(
    fp8: bool = False,
    training: bool = False,
    distributed_group: ProcessGroup | None = None,
):
    CUR_ENV_STACK.append(Environment(fp8, training, distributed_group))
    yield
    CUR_ENV_STACK.pop()


__all__ = ["environment"]
