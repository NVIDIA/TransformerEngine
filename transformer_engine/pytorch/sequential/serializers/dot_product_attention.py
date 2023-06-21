from math import sqrt
from typing import ContextManager, Literal
from ..custom_serializer_holder import COMPUTE_PIPELINE_CUSTOM_SERIALIZERS
from ...attention import DotProductAttention
from ..ops import OpGraph


def _serializer(module: DotProductAttention):
    attn_mask_type: Literal["causal", "padding"] = module.attn_mask_type  # type: ignore[assignment]
    hidden_size = module.hidden_size_per_attention_head
    scale = sqrt(hidden_size) * (module.unfused_attention.layer_number or 1.0)
    dropout_p = module.unfused_attention.attention_dropout.p
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
    v = op_graph.view_(v, [1, 2, 0, 3])  # [b, np, sk, hn]

    scores = op_graph.bmm_(q, k)  # [b, np, sq, sk]
    scores = op_graph.scale_(scores, scale)  # [b, np, sq, sk]

    if attn_mask_type == "causal":
        scores = op_graph.f_causal_mask_(scores)  # [b, np, sq, sk]
    elif attn_mask_type == "padding":
        raise NotImplementedError("Padding mask is not implemented yet")

    scores = op_graph.f_softmax_(scores)  # [b, np, sq, sk]
    scores = op_graph.f_dropout_(scores, dropout_p, rng_ctx)  # [b, np, sq, sk]

    o = op_graph.bmm_(scores, v)  # [b, np, sq, hn]
    o = op_graph.view_(o, [2, 0, 1, 3])  # [sq, b, np, hn]

    op_graph.out_(o)

    return op_graph


COMPUTE_PIPELINE_CUSTOM_SERIALIZERS[DotProductAttention] = _serializer
