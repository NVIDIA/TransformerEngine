from .. import COMPUTE_PIPELINE_CUSTOM_SERIALIZERS
from ...module import LayerNorm


def _serializer(module: LayerNorm):
    ...


COMPUTE_PIPELINE_CUSTOM_SERIALIZERS[LayerNorm] = _serializer
