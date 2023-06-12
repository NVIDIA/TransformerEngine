from .. import COMPUTE_PIPELINE_CUSTOM_SERIALIZERS
from ...module import Linear
from ..ops import OpGraph


def _serializer(module: Linear):
    graph = OpGraph()
    in_ = graph.in_()
    weights = graph.param_(module.in_features * module.out_features)
    if module.use_bias:
        bias = graph.param_(module.out_features)

    y = graph.mul_(in_, weights)
    if module.use_bias:
        y = graph.add_(y, bias)
    graph.out_(y)
    return graph


COMPUTE_PIPELINE_CUSTOM_SERIALIZERS[Linear] = _serializer
