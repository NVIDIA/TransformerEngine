import torch.nn as nn
from typing import Any
from .ops import OpGraph
from . import serializers
from .custom_serializer_holder import COMPUTE_PIPELINE_CUSTOM_SERIALIZERS


class ComputePipeline:
    def __init__(self, *modules: nn.Module) -> None:
        # Construct forward graph
        for module in modules:
            serialize_func = getattr(module, "compute_pipeline_serialize", None)
            if serialize_func is None or not callable(serialize_func):
                if type(module) in COMPUTE_PIPELINE_CUSTOM_SERIALIZERS:
                    serialize_func = COMPUTE_PIPELINE_CUSTOM_SERIALIZERS[type(module)]
                else:
                    raise TypeError(
                        f"A module of type {type(module)} was used with ComputePipeline, but it does not have a compute_pipeline_serialize method. You have to either add this method or register a custom serializer using the COMPUTE_PIPELINE_CUSTOM_SERIALIZERS dictionary."
                    )
            serialized: OpGraph = serialize_func(module)
            if not hasattr(self, "_graph"):
                self._graph = serialized
            else:
                self._graph = OpGraph.combine_graphs(self._graph, serialized)
        assert len(self._graph.out_nodes) == 1
        self._in_grad_node = self._graph.in_()

    def backward(self):
        self._graph.create_backward_graph_(self._graph.out_nodes[0], self._in_grad_node)

    def __call__(self, x: Any) -> Any:
        ...


__all__ = ["ComputePipeline", "COMPUTE_PIPELINE_CUSTOM_SERIALIZERS"]
