# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Worker for test_attention.py::test_fused_attn_graph_cache.

Runs a fixed sequence of support queries and executions against the cuDNN graph cache and
marks each phase boundary on stderr, so that the parent can attribute the
[FUSED-ATTN-CACHE] lines NVTE_FUSED_ATTN_CACHE_DEBUG=2 emits to the phase that produced
them.

This runs as its own process because the cache is process-wide and its counters only
accumulate: inside the pytest process, the graphs every earlier test built would be mixed
into the counts, and the cache would already be warm for whatever this test asked about.

The phases, in order, and what each one is for:
  query    the first support query for a config -- the miss that builds its graphs
  requery  the identical query again -- must be answered from the cache
  exec     forward and backward of that config -- must reuse the graphs the query built,
           and is where the plan build the query deferred happens
  rescale  the same execution with only softmax_scale changed -- must still reuse them,
           since attn_scale is normalized out of the cache key
  reshape  a query differing in max_seqlen -- must build again, once per pass

Prints ``[CACHE-TEST] fused=1`` (or 0) on stdout so the parent can skip rather than fail on
a GPU or cuDNN version with no fused-attention backend for the config.
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


def mark_phase(name: str) -> None:
    """Delimit the cache events of one phase from the next one's.

    Written to stderr, which is where the diagnostics go, so that the marker keeps its
    place in the stream instead of racing them on a second file descriptor.
    """
    sys.stderr.write(f"[CACHE-TEST] phase={name}\n")
    sys.stderr.flush()


def query(config: ModelConfig) -> bool:
    """Run one backend support query, as the test suite does, and report whether cuDNN
    took the configuration. This is the call that populates the cache without executing
    anything."""
    available_backends, _, fused_attn_backends = get_available_attention_backends(
        config, qkv_dtype=DTYPE, qkv_layout=QKV_LAYOUT
    )
    _, fused_attn_supported, _ = available_backends
    return fused_attn_supported and len(fused_attn_backends) > 0


def execute(config: ModelConfig, softmax_scale: float) -> None:
    """Run a forward and backward pass of `config` on the fused backend."""
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
    # The counters are incremented from the launching thread, but the graphs are not
    # necessarily done with; synchronize so that nothing lands in the next phase.
    torch.cuda.synchronize()


def main() -> int:
    torch.manual_seed(1234)
    # No mask, no bias, no dropout: the simplest configuration cuDNN supports, so that the
    # counts this produces are about the cache rather than about which graph got built.
    config = ModelConfig(2, 512, 8, 64)
    reshaped = ModelConfig(2, 256, 8, 64)

    mark_phase("query")
    fused_available = query(config)
    print(f"[CACHE-TEST] fused={int(fused_available)}", flush=True)
    if not fused_available:
        return 0

    mark_phase("requery")
    query(config)

    # get_available_attention_backends() enables every backend so it can report on all of
    # them; the execution phases have to land on the fused one for their cache events to
    # exist at all, so leave it the only one available.
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
