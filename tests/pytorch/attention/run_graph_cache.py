# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Worker for test_attention.py::test_fused_attn_graph_cache.

Runs a fixed sequence of support queries and executions against the FusedAttention graph cache and
collects level-2 diagnostics with NVTE_FUSED_ATTN_CACHE_DEBUG=2 on, for different phases.
  query    the first support query for a config -- the miss that creates its graphs
  requery  the identical query again -- should hit the cache
  exec     forward and backward of that config -- should hit the cache and build plans before executing
  rescale  the same execution with only softmax_scale changed -- should hit the cache and reuse the plans,
           since attn_scale is normalized out of the cache key
  reshape  a query differing in max_seqlen -- should miss and recreate the graphs
"""

import os
import pathlib
import sys

import torch

_current_file = pathlib.Path(__file__).resolve()
sys.path = [str(_current_file.parent.parent)] + sys.path

from transformer_engine.pytorch import DotProductAttention
from transformer_engine.pytorch.attention.dot_product_attention import _attention_backends
from utils import ModelConfig, get_available_attention_backends

DTYPE = torch.bfloat16
QKV_FORMAT = "bshd"
QKV_LAYOUT = "bshd_bshd_bshd"
DETERMINISTIC = (
    not bool(int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")))
    or torch.are_deterministic_algorithms_enabled()
)


def mark_phase(name: str) -> None:
    """Delimit the cache events of one phase from the next one's."""
    sys.stderr.write(f"[CACHE-TEST] phase={name}\n")
    sys.stderr.flush()


def query(config: ModelConfig) -> bool:
    """Run one backend support query and report whether FusedAttention supports
    the configuration. If so, it populates the cache without executing anything."""
    available_backends, _, fused_attn_backends = get_available_attention_backends(
        config, qkv_dtype=DTYPE, qkv_layout=QKV_LAYOUT, deterministic=DETERMINISTIC
    )
    _, fused_attn_supported, _ = available_backends
    return fused_attn_supported and len(fused_attn_backends) > 0


def execute(config: ModelConfig, softmax_scale: float) -> None:
    """Run a forward and backward pass of `config` on the FusedAttention backend."""
    block = DotProductAttention(
        config.num_heads,
        config.head_dim_qk,
        attention_dropout=config.dropout_p,
        qkv_format=QKV_FORMAT,
        attn_mask_type=config.attn_mask_type,
        softmax_scale=softmax_scale,
        layer_number=1,
        attention_type=config.attn_type,
    ).to(dtype=DTYPE, device="cuda")
    shape = (config.batch_size, config.max_seqlen_q, config.num_heads, config.head_dim_qk)
    q, k, v = [torch.randn(shape, dtype=DTYPE, device="cuda", requires_grad=True) for _ in range(3)]
    out = block(q, k, v, core_attention_bias_type=config.attn_bias_type)
    out.backward(torch.randn_like(out))
    torch.cuda.synchronize()


def main() -> int:
    torch.manual_seed(1234)
    config = ModelConfig(2, 512, 8, 64)
    reshaped = ModelConfig(2, 256, 8, 64)

    mark_phase("query")
    fused_available = query(config)
    print(f"[CACHE-TEST] fused={int(fused_available)}", flush=True)
    if not fused_available:
        return 0

    mark_phase("requery")
    query(config)

    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "1"
    os.environ["NVTE_UNFUSED_ATTN"] = "0"
    _attention_backends["backend_selection_requires_update"] = True

    mark_phase("exec")
    execute(config, softmax_scale=0.125)

    mark_phase("rescale")
    execute(config, softmax_scale=0.25)

    mark_phase("reshape")
    query(reshaped)

    mark_phase("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
