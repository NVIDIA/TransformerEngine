# DSv4 attention API

`DSv4Attention` is an experimental API in TE that composes cuDNN compression,
top-k indexing, and sparse attention kernels with Megatron's DSA RoPE kernels
through reusable building blocks such as `Compressor` and `Indexer`. cuDNN
supplies compression and sparse attention backward kernels
and separate indexer-loss score/backward primitives; top-k selection itself is
discrete. The layer maps hidden states to hidden states with the same shape.
`Compressor` and `Indexer` can also be composed into other attention layers;
`dsa_cudnn` exposes the underlying operations for prepared tensors.

## Module hierarchy

```text
DSv4Attention
├── TE query/local-KV projections + norms
├── Compressor
│   ├── TE KV/gate projections + learned position bias
│   ├── dsa_cudnn.compress -> cuDNN compressor
│   └── TE RMSNorm
├── Indexer (CSA only)
│   ├── TE index-query/head-weight projections
│   ├── own Compressor (index keys)
│   ├── dsa_rope.apply_rotary (query + index keys)
│   └── dsa_cudnn.select_indices -> cuDNN score + top-k
├── dsa_rope embeddings + rotary application
├── dsa_cudnn.sparse_attention -> cuDNN sparse attention
└── TE grouped output projection
```

## Inside DSv4Attention

The following sketch expands the cuDNN calls underneath Compressor, Indexer, and attention.
Query and local KV are already projected, normalized, and rotated; tensors are contiguous CUDA BF16, with FP32 position bias/sink and CUDA INT32 sequence prefixes.

```python
from transformer_engine.pytorch.attention.sparse_attention import (
    dsa_cudnn,
)

pooled = dsa_cudnn.compress(
    kv, gates, position_bias.float(), cu, cu_comp,
    ratio=ratio, overlap=is_csa, total_comp=total_comp,
)
compressed_kv = ...  # Apply RMSNorm and RoPE to pooled rows.

indices = None
if is_csa:
    index_pooled = dsa_cudnn.compress(
        index_kv, index_gates, index_position_bias.float(), cu, cu_comp,
        ratio=ratio, overlap=True, total_comp=total_comp,
    )
    index_key = model.finish_index_key(index_pooled)
    indices = dsa_cudnn.select_indices(
        index_query, index_key, index_weights, cu, cu_comp,
        top_k=top_k, ratio=ratio, max_seqlen=max_seqlen,
        max_compressed_seqlen=max_compressed_seqlen,
        scale=(index_head_dim * index_n_heads) ** -0.5,
    )

output = dsa_cudnn.sparse_attention(
    query, local_kv, compressed_kv, sink.float(), cu, cu_comp,
    window_size=window_size, ratio=ratio, indices=indices,
    max_compressed_seqlen=max_compressed_seqlen if indices is None else None,
)
output = ...  # Apply inverse RoPE and grouped output projection.
```

`dsa_rope` supplies token/compressed-position embeddings and rotary application.
A caller may override the default embeddings with a `rope` module returning
token and compressed `(cos, sin)` pairs.

## Usage

```python
import torch
from transformer_engine.pytorch.attention.sparse_attention import DSv4Attention

attention = DSv4Attention(
    hidden_size=4096, q_lora_rank=1536,
    layer_type="compressed_sparse_attention", head_dim=512, rope_head_dim=64,
    sliding_window=128, compression_ratio=4, o_groups=8, o_lora_rank=1024,
    index_n_heads=32, index_topk=32, max_seqlen=2048,
    params_dtype=torch.bfloat16,
)
output = attention(hidden_states)  # [S, B, 4096] -> [S, B, 4096]
```

HCA uses non-overlapping compression and omits the Indexer:

```python
hca = DSv4Attention(
    hidden_size=4096, q_lora_rank=1536,
    layer_type="heavily_compressed_attention", head_dim=512, rope_head_dim=64,
    sliding_window=128, compression_ratio=128, o_groups=8, o_lora_rank=1024,
    max_seqlen=2048, params_dtype=torch.bfloat16,
)
output = hca(hidden_states)  # [S, B, 4096] -> [S, B, 4096]
```

Supported input tensor shapes are "batch-first" (shape `BSD`), "sequence-first" (shape `SBD`),
and packed (shape `TD`). Packed calls require contiguous CUDA INT32
`cu_seqlens_q [B+1]` (from zero to `T`) and `max_seqlen_q`:

```python
# Construct attention with qkv_format="thd".
output = attention(hidden_states, cu_seqlens_q=cu, max_seqlen_q=max_length)
```

## Note on support

The current layer supports full-sequence BF16 forward/backward on SM100,
with 64 heads of width 512 and at least one complete compression window
per sequence. Sequences start at position zero; cache/decode, padding,
TP/SP/CP, FP8, and sliding-only layers are unsupported.

Compression and attention support autograd; top-k selection is discrete.
CSA's `return_indexer_context=True` returns `(output, context)` for a
single-sequence, caller-owned indexer loss. Without that objective, the
indexer is not trained by the attention output alone. Pre-attention
normalization, residuals, and loss calculation remain model-owned.
