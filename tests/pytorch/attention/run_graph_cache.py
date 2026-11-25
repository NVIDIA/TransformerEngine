# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Worker for test_attention.py::test_fused_attn_graph_cache.

Runs a fixed sequence of support queries and executions against the FusedAttention graph cache and
collects level-2 diagnostics with NVTE_FUSED_ATTN_CACHE_DEBUG=2 for different phases:
  query    the first support query for a config -- the miss that creates its fwd/bwd graphs
  requery  the identical query again -- hits the cache
  exec     forward and backward of that config -- hits the cache and builds plans before execution
  rescale  the same execution with only softmax_scale changed -- should hit the cache and reuse the plans,
           since attn_scale is normalized out of the cache key
  reshape  a query differing in max_seqlen -- should miss and recreate the graphs
  thd_warm / thd_vary
           THD forwards over varying numbers of packed sequences within one batch-size bucket -- the
           first builds the forward graph and the rest should hit it, whichever way cu_seqlens reach cuDNN
"""

import os
import pathlib
import sys

import torch

_current_file = pathlib.Path(__file__).resolve()
sys.path = [str(_current_file.parent.parent)] + sys.path

from transformer_engine.pytorch import DotProductAttention
from transformer_engine.pytorch.attention.dot_product_attention import _attention_backends
from transformer_engine.pytorch.utils import get_cudnn_version
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


def thd_graphs_bucket_the_batch() -> bool:
    """Whether the THD graphs are built at a bucketed batch size (ragged graphs, cuDNN >= 9.6 on
    sm90+ except sm120)."""
    major, minor = torch.cuda.get_device_capability()
    return get_cudnn_version() >= (9, 6, 0) and major * 10 + minor >= 90 and major != 12


def execute_thd(num_seqs: int) -> None:
    """Run a THD forward pass over `num_seqs` packed sequences of a fixed total token count."""
    num_tokens, num_heads, head_dim = 1024, 8, 64
    block = DotProductAttention(
        num_heads,
        head_dim,
        attention_dropout=0.0,
        qkv_format="thd",
        attn_mask_type="padding_causal",
        layer_number=1,
    ).to(dtype=DTYPE, device="cuda")
    bounds = torch.linspace(0, num_tokens, num_seqs + 1).round().to(torch.int32).cuda()
    max_seqlen = int((bounds[1:] - bounds[:-1]).max())
    shape = (num_tokens, num_heads, head_dim)
    q, k, v = [torch.randn(shape, dtype=DTYPE, device="cuda") for _ in range(3)]
    with torch.no_grad():
        block(
            q,
            k,
            v,
            cu_seqlens_q=bounds,
            cu_seqlens_kv=bounds,
            cu_seqlens_q_padded=bounds,
            cu_seqlens_kv_padded=bounds,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
        )
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

    if thd_graphs_bucket_the_batch():
        print("[CACHE-TEST] thd=1", flush=True)
        mark_phase("thd_warm")
        execute_thd(2)

        mark_phase("thd_vary")
        for num_seqs in (3, 9, 17, 31):
            execute_thd(num_seqs)

    mark_phase("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
