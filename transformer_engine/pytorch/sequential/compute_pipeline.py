import torch.nn as nn
from typing import Any, Callable

COMPUTE_PIPELINE_SERIALIZE_EXT: dict[type, Callable] = {}


class ComputePipeline:
    def __init__(self, *modules: nn.Module) -> None:
        for module in modules:
            serialize_func = getattr(module, "compute_pipeline_serialize", None)
            if serialize_func is None or not callable(serialize_func):
                if type(module) in COMPUTE_PIPELINE_SERIALIZE_EXT:
                    serialize_func = COMPUTE_PIPELINE_SERIALIZE_EXT[type(module)]
                else:
                    raise TypeError(
                        f"A module of type {type(module)} was used with ComputePipeline, but it does not have a compute_pipeline_serialize method. You have to either add this method or register a custom serializer using the COMPUTE_PIPELINE_SERIALIZE_EXT dictionary."
                    )
            serialized = serialize_func(module)

    def __call__(self, x: Any) -> Any:
        ...


__all__ = ["ComputePipeline"]
