from .. import COMPUTE_PIPELINE_CUSTOM_SERIALIZERS
from ...module import LayerNormLinear, LayerNorm, Linear
from ..ops import OpGraph


def _serializer(module: LayerNormLinear):
    layernorm_impostor = object()
    layernorm_impostor.weight = object()
    layernorm_impostor.weight.shape = (module.in_features,)  # pylint: disable=no-member
    layernorm_impostor.eps = module.eps
    layernorm_impostor.zero_centered_gamma = module.zero_centered_gamma
    layernorm_graph = COMPUTE_PIPELINE_CUSTOM_SERIALIZERS[LayerNorm](layernorm_impostor)

    linear_impostor = object()
    linear_impostor.in_features = module.in_features
    linear_impostor.out_features = module.out_features
    linear_impostor.use_bias = module.use_bias
    linear_graph = COMPUTE_PIPELINE_CUSTOM_SERIALIZERS[Linear](linear_impostor)

    graph = OpGraph.combine_graphs(layernorm_graph, linear_graph)
    return graph


COMPUTE_PIPELINE_CUSTOM_SERIALIZERS[LayerNormLinear] = _serializer
