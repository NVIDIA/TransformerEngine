from .. import COMPUTE_PIPELINE_CUSTOM_SERIALIZERS
from ...module import Linear


def _serializer(module: Linear):
    ...


COMPUTE_PIPELINE_CUSTOM_SERIALIZERS[Linear] = _serializer
