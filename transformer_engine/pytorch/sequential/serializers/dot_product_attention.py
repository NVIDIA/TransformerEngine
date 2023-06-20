from typing import ContextManager
from ..custom_serializer_holder import COMPUTE_PIPELINE_CUSTOM_SERIALIZERS
from ...attention import DotProductAttention
from ..ops import OpGraph


def _serializer(module: DotProductAttention):
    module_name: str = getattr(module, "_compute_pipeline_name")
    attn_mask_type = module.attn_mask_type
    layer_number = module.unfused_attention.layer_number
    dropout = module.unfused_attention.attention_dropout
    rng_ctx: ContextManager[None] = module.unfused_attention.attention_dropout_ctx  # type: ignore[assignment]

    # Notation
    # b: batch size
    # np: number of heads
    # hn: hidden size
    # sk: number of keys (and values)
    # sq: number of queries

    op_graph = OpGraph()
    # We assume that the input has the right shape
    q = op_graph.in_()  # [sq, b, np, hn]
    k = op_graph.in_()  # [sk, b, np, hn]
    v = op_graph.in_()  # [sk, b, np, hn]

    q = op_graph.view_(q, [1, 2, 0, 3])  # [b, np, sq, hn]
    k = op_graph.view_(k, [1, 2, 3, 0])  # [b, np, hn, sk]
    scores = op_graph.bmm_(q, k)  # [b, np, sq, sk]
    # TODO: causal masking, softmax, dropout
    v = op_graph.view_(v, [1, 2, 0, 3])  # [b, np, sk, hn]
    o = op_graph.bmm_(scores, v)  # [b, np, sq, hn]
    o = op_graph.view_(o, [2, 0, 1, 3])  # [sq, b, np, hn]

    op_graph.out_(o)

    return op_graph


COMPUTE_PIPELINE_CUSTOM_SERIALIZERS[DotProductAttention] = _serializer
