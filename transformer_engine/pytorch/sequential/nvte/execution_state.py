from __future__ import annotations
from typing import Literal
import torch
from ..utils import contextmanager
from ..persistent import Persistent
from ..metatensors import PersistentFP8Meta
from .cpp_extensions import DType

FP8Meta = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _default_meta_tensor_provider():
    meta_tensor_provider = PersistentFP8Meta()
    meta_tensor_provider.next_iteration()
    return meta_tensor_provider


pass_: Literal["forward", "backward", "inference"] = "inference"
meta_tensor_provider: Persistent[DType, FP8Meta] = _default_meta_tensor_provider()


@contextmanager
def set_execution_state(
    pass__: Literal["forward", "backward", "inference"],
    meta_tensor_provider_: Persistent[DType, FP8Meta],
):
    global meta_tensor_provider
    global pass_

    meta_tensor_provider = meta_tensor_provider_
    pass_ = pass__
    try:
        yield
    finally:
        meta_tensor_provider = _default_meta_tensor_provider()
        pass_ = "inference"
