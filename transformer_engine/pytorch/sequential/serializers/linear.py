from ..custom_serializer_holder import COMPUTE_PIPELINE_CUSTOM_SERIALIZERS
from ...module import Linear
from ..ops import OpGraph


def _serializer(module: Linear):
    module_name: str = getattr(module, "_compute_pipeline_name")
    graph = OpGraph()

    in_ = graph.in_()
    weights = graph.param_(
        module.in_features * module.out_features, f"{module_name}.weight"
    )
    y = graph.bmm_(in_, weights)

    if module.use_bias:
        bias = graph.param_(module.out_features, f"{module_name}.bias")
        y = graph.add_(y, bias)

    graph.out_(y)
    return graph


COMPUTE_PIPELINE_CUSTOM_SERIALIZERS[Linear] = _serializer
