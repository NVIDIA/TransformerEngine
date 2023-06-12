from .. import COMPUTE_PIPELINE_CUSTOM_SERIALIZERS
from ...module import LayerNormLinear


def _serializer(module: LayerNormLinear):
    ...


COMPUTE_PIPELINE_CUSTOM_SERIALIZERS[LayerNormLinear] = _serializer
