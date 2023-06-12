from .. import COMPUTE_PIPELINE_CUSTOM_SERIALIZERS
from ...module import LayerNormMLP, LayerNorm, Linear
from ..ops import OpGraph


def _gelu_graph():
    graph = OpGraph()
    in_ = graph.in_()
    out_ = graph.f_gelu_(in_)
    graph.out_(out_)
    return graph


def _serializer(module: LayerNormMLP):
    hidden_size = module.layer_norm_weight.shape[0]
    ffn_hidden_size = module.size_per_partition * module.tp_size

    layernorm_impostor = object()
    layernorm_impostor.weight = object()
    layernorm_impostor.weight.shape = (hidden_size,)  # pylint: disable=no-member
    layernorm_impostor.eps = module.eps
    layernorm_impostor.zero_centered_gamma = module.zero_centered_gamma
    layernorm_graph = COMPUTE_PIPELINE_CUSTOM_SERIALIZERS[LayerNorm](layernorm_impostor)

    linear1_impostor = object()
    linear1_impostor.in_features = hidden_size
    linear1_impostor.out_features = ffn_hidden_size
    linear1_impostor.use_bias = module.use_bias
    linear1_graph = COMPUTE_PIPELINE_CUSTOM_SERIALIZERS[Linear](linear1_impostor)

    gelu_graph = _gelu_graph()

    linear2_impostor = object()
    linear2_impostor.in_features = ffn_hidden_size
    linear2_impostor.out_features = hidden_size
    linear2_impostor.use_bias = module.use_bias
    linear2_graph = COMPUTE_PIPELINE_CUSTOM_SERIALIZERS[Linear](linear2_impostor)

    graph = OpGraph.combine_graphs(layernorm_graph, linear1_graph)
    graph = OpGraph.combine_graphs(graph, gelu_graph)
    graph = OpGraph.combine_graphs(graph, linear2_graph)
    return graph


COMPUTE_PIPELINE_CUSTOM_SERIALIZERS[LayerNormMLP] = _serializer
