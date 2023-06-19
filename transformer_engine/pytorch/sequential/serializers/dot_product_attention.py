from transformer_engine.pytorch.sequential.ops.base import Op
from ..custom_serializer_holder import COMPUTE_PIPELINE_CUSTOM_SERIALIZERS
from ...attention import DotProductAttention
from ..ops import OpGraph


def _serializer(module: DotProductAttention) -> OpGraph:
    ...


COMPUTE_PIPELINE_CUSTOM_SERIALIZERS[DotProductAttention] = _serializer
