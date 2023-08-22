from typing import Literal
import torch
from ..persistent import Persistent

FP8Meta = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

pass_: Literal["forward", "backward", "inference"] = None  # type: ignore
meta_tensor_provider: Persistent[FP8Meta] = None  # type: ignore


def set_execution_state(
    pass__: Literal["forward", "backward", "inference"],
    meta_tensor_provider_: Persistent[FP8Meta],
):
    global meta_tensor_provider
    meta_tensor_provider = meta_tensor_provider_
    global pass_
    pass_ = pass__
