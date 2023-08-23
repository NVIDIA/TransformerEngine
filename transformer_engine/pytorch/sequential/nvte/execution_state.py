from typing import Literal
from contextlib import contextmanager
import torch
from ..persistent import Persistent
from ..meta import PersistentFP8Meta

FP8Meta = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

pass_: Literal["forward", "backward", "inference"]
meta_tensor_provider: Persistent[FP8Meta]


@contextmanager
def set_execution_state(
    pass__: Literal["forward", "backward", "inference"],
    meta_tensor_provider_: Persistent[FP8Meta],
):
    global meta_tensor_provider
    global pass_

    meta_tensor_provider = meta_tensor_provider_
    pass_ = pass__
    try:
        yield
    finally:
        meta_tensor_provider = PersistentFP8Meta()
        meta_tensor_provider.next_iteration()
        pass_ = "inference"
