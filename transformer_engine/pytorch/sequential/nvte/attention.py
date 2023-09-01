from __future__ import annotations
from . import cpp_extensions as _nvte
from .empty import empty


def dot_product_attention(
    QKV: _nvte.Tensor, cu_seqlens: _nvte.Tensor, attn_scale: float, dropout: float
):
    S = empty((), _nvte.DType.Float8E4M3)
    token_count = QKV.shape[0]
    assert QKV.shape[1] % 3 == 0
    token_dim = QKV.shape[1] // 3

    _nvte.fused_attn_fwd_qkvpacked(
        QKV,
        empty(),
        S,
    )
