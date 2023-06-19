from ..custom_serializer_holder import COMPUTE_PIPELINE_CUSTOM_SERIALIZERS
from ...module import LayerNormLinear, LayerNorm, Linear
from ..ops import OpGraph
from types import SimpleNamespace


def _serializer(module: LayerNormLinear):
    module_name: str = getattr(module, "_compute_pipeline_name")

    layernorm_impostor = SimpleNamespace()
    layernorm_impostor.weight = SimpleNamespace()
    layernorm_impostor.weight.shape = (module.in_features,)
    layernorm_impostor.eps = module.eps
    layernorm_impostor.zero_centered_gamma = module.zero_centered_gamma
    layernorm_impostor._compute_pipeline_name = f"{module_name}_ln"
    layernorm_graph = COMPUTE_PIPELINE_CUSTOM_SERIALIZERS[LayerNorm](layernorm_impostor)

    linear_impostor = SimpleNamespace()
    linear_impostor.in_features = module.in_features
    linear_impostor.out_features = module.out_features
    linear_impostor.use_bias = module.use_bias
    linear_impostor._compute_pipeline_name = f"{module_name}_fc"
    linear_graph = COMPUTE_PIPELINE_CUSTOM_SERIALIZERS[Linear](linear_impostor)

    graph = OpGraph.combine_graphs(layernorm_graph, linear_graph)
    return graph


COMPUTE_PIPELINE_CUSTOM_SERIALIZERS[LayerNormLinear] = _serializer
