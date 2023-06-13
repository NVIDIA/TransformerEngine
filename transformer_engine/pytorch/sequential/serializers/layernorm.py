from ..custom_serializer_holder import COMPUTE_PIPELINE_CUSTOM_SERIALIZERS
from ...module import LayerNorm
from ..ops import OpGraph


def _serializer(module: LayerNorm):
    graph = OpGraph()
    in_ = graph.in_()
    hidden_size = module.weight.shape[0]
    gamma = graph.param_(hidden_size)
    beta = graph.param_(hidden_size)
    out_ = graph.f_layernorm_(in_, gamma, beta, module.eps, module.zero_centered_gamma)
    graph.out_(out_)
    return graph


COMPUTE_PIPELINE_CUSTOM_SERIALIZERS[LayerNorm] = _serializer
