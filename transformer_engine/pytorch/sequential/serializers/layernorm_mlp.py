from .. import COMPUTE_PIPELINE_CUSTOM_SERIALIZERS
from ...module import LayerNormMLP


def _serializer(module: LayerNormMLP):
    ...


COMPUTE_PIPELINE_CUSTOM_SERIALIZERS[LayerNormMLP] = _serializer
