# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Compat shim: the MLA RoPE kernels moved to
``transformer_engine.pytorch.models.deepseek_v3.mla_rope``."""

import torch

from transformer_engine.pytorch.models.deepseek_v3.mla_rope import (  # noqa: F401
    HAVE_TRITON,
    apply_mla_rope_kv,
    apply_mla_rope_q,
    build_rope_tables,
)

HEAD_DIM_ROPE = 64
HEAD_DIM_NOPE = 128
HEAD_DIM_V = 128
ROTARY_BASE = 10000


def apply_mla_rope(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_pos_emb: torch.Tensor,
    head_dim_nope: int = HEAD_DIM_NOPE,
    head_dim_rope: int = HEAD_DIM_ROPE,
    head_dim_v: int = HEAD_DIM_V,
    base: int = ROTARY_BASE,
    cos_table: torch.Tensor | None = None,
    sin_table: torch.Tensor | None = None,
):
    if cos_table is None or sin_table is None:
        cos_table, sin_table = build_rope_tables(
            q.shape[0], head_dim_rope, base=base, device=q.device
        )
    q = apply_mla_rope_q(q, cos_table, sin_table, head_dim_nope, head_dim_rope)
    k, v = apply_mla_rope_kv(
        kv, k_pos_emb, cos_table, sin_table, head_dim_nope, head_dim_rope, head_dim_v
    )
    return q, k, v
