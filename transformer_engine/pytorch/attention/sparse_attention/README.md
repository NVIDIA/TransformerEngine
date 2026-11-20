# DSv4 attention API

`DSv4Attention` is an experimental API in TE that composes cuDNN compression,
top-k indexing, and sparse attention kernels with Triton RoPE through reusable
building blocks such as `Compressor` and `Indexer`. DSv4 uses `rotary_fwd_q_kernel`
and `rotary_bwd_q_kernel` for both Q and KV rotation. MLA's `apply_mla_rope_q`
uses the same forward and backward kernels for Q.

cuDNN supplies compression and sparse attention forward and backward kernels
and separate indexer-loss score/backward primitives; top-k selection itself is
discrete. The layer maps hidden states to hidden states with the same shape.
`Compressor` and `Indexer` can also be composed into other attention layers;
`dsa_cudnn` exposes the underlying operations for prepared tensors.

## Module hierarchy

```text
DSv4Attention
├── TE query/local-KV projections + norms
├── Compressor (HCA or CSA)
│   ├── TE KV/gate projections + learned additive position embedding (APE)
│   ├── dsa_cudnn.compress -> cuDNN compressor
│   └── TE RMSNorm
├── Indexer (CSA only)
│   ├── TE index-query/head-weight projections
│   ├── own Compressor (index keys)
│   ├── rope.apply_dsa_rope (query + index keys)
│   └── dsa_cudnn.select_indices -> cuDNN score + top-k
├── rope embeddings + rotary application
├── dsa_cudnn.sparse_attention -> cuDNN sparse attention
└── TE grouped output projection
```

## Inside DSv4Attention

The following sketch expands the cuDNN calls underneath Compressor, Indexer, and attention.
Query and local KV are already projected, normalized, and rotated; tensors are contiguous CUDA, with FP32 APE/sink and CUDA INT32 sequence prefixes.

```python
import transformer_engine.pytorch.attention.sparse_attention.dsa_cudnn as dsa_cudnn

compressed_kv = dsa_cudnn.compress(
    kv, gates, ape.float(), cu, cu_comp,
    ratio=ratio, overlap=is_csa, total_comp=total_comp,
)
compressed_kv = ...  # Apply RMSNorm and RoPE to compressed KV.

indices = None
if is_csa:
    index_key = dsa_cudnn.compress(
        index_kv, index_gates, index_ape.float(), cu, cu_comp,
        ratio=ratio, overlap=True, total_comp=total_comp,
    )
    index_key = ...  # Apply RMSNorm and RoPE to compressed index keys.
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

Batched inputs contain B sequences of the same length S. `DSv4Attention` accepts
`[S,B,hidden_size]` (sbhd) or `[B,S,hidden_size]` (bshd), and packs tokens in
batch-major order before these calls. Packed `[T,hidden_size]` (thd) inputs
concatenate sequences, with cumulative token counts identifying their boundaries.
Standalone `Compressor` and `Indexer` accept only batched `[B,S,D]` or packed
`thd` `[T,D]` input; callers must transpose sequence-major inputs first. Hidden
states have no explicit head axis; prepared queries have shape `[T,H,D]`.
Attention uses the original query
`[T,64,D]` and concatenated local/compressed KV `[T+Tc,D]`; there is no
compressed query. `T` is the total query/local KV token count.
`_prepare_sparse_kv_indices` creates a list of KV token indices for each query token,
combining causal local KV token indices with compressed KV token indices offset
by `T`; invalid entries are `-1`. CSA supplies global compressed KV token indices
from `select_indices`; HCA passes `indices=None` and includes all eligible
compressed tokens. `_prepare_sparse_kv_indices` documents a small example of these
index spaces. Selection indices address only packed compressed KV before the
local KV offset is applied.

`transformer_engine.pytorch.attention.rope` provides the rotary utilities.
`rope.DSv4RotaryEmbedding` caches token/compressed-position embeddings.
`rope.apply_dsa_rope` applies them.
A caller may override the default embeddings with a `rope` module returning
token and compressed `(cos, sin)` pairs.

## Usage

CSA uses overlapping compression (see `Compressor`) and an Indexer to select
compressed KV entries.

```python
import torch
from transformer_engine.pytorch.attention.sparse_attention import DSv4Attention

csa = DSv4Attention(
    hidden_size=4096, q_lora_rank=1536,
    layer_type="compressed_sparse_attention", head_dim=512, rope_head_dim=64,
    sliding_window=128, compression_ratio=4, o_groups=8, o_lora_rank=1024,
    index_n_heads=32, index_topk=32, max_seqlen=2048,
    params_dtype=torch.bfloat16,
)
output = csa(hidden_states)  # [S, B, 4096] -> [S, B, 4096]
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

`hidden_states` has shape `[S,B,hidden_size]` for `qkv_format="sbhd"` (default),
`[B,S,hidden_size]` for `"bshd"`, or `[T,hidden_size]` for `"thd"`.
Packed calls require contiguous CUDA INT32
`cu_seqlens_q [B+1]` (from zero to `T`) and `max_seqlen_q`:

```python
packed_csa = DSv4Attention(
    hidden_size=4096, q_lora_rank=1536,
    layer_type="compressed_sparse_attention", head_dim=512, rope_head_dim=64,
    sliding_window=128, compression_ratio=4, o_groups=8, o_lora_rank=1024,
    index_n_heads=32, index_topk=32, max_seqlen=2048,
    params_dtype=torch.bfloat16, qkv_format="thd",
)
output = packed_csa(hidden_states, cu_seqlens_q=cu, max_seqlen_q=max_length)
```

## Note on support

The layer admits SM100, SM103, and SM107 GPUs for full-sequence
forward/backward and requires BF16 activations. FP16 is unsupported by the
compression and indexer training dependencies. The operations require cuDNN
Frontend 1.29 or newer with its CSA/DSA APIs and are delivered as CuTe kernels.
Hardware validation includes prior SM100 runs and SM103/SM107 runs with
cuDNN Frontend 1.30.0, Cutlass DSL 4.8.0, CUDA 13.4, and cuDNN 9.27.

The layer uses 64 heads of width 512 and requires at least one complete
compression window per sequence. Sequences start at position zero;
cache/decode, physical inter-sequence padding, TP/SP/CP, and FP8 are unsupported.
CSA and HCA include sliding-window attention; standalone sliding-window-only
layers should use `DotProductAttention` with `window_size` instead.

Compression and attention support autograd; top-k selection is discrete.
CSA's `return_indexer_context=True` returns `(output, context)` for a
single-sequence, caller-owned indexer loss. Without that objective, the
indexer is not trained by the attention output alone. Pre-attention
normalization, residuals, and loss calculation remain model-owned.

For B=1, the caller can train the indexer with `dense_indexer_loss_scores` and
`dense_indexer_loss_backward`, exported alongside the three sparse-attention classes:

```python
from transformer_engine.pytorch import dense_indexer_loss_scores, dense_indexer_loss_backward

output, context = csa(hidden_states, return_indexer_context=True)
qi, ki, wi = (context[name].unsqueeze(0)
              for name in ("index_q", "index_k", "index_w"))
qa, kc = (context[name].unsqueeze(0)
          for name in ("attn_q", "compressed_kv"))
# Caller computes detached FP32 full_lse [1,T,64] over the local window,
# ALL eligible compressed keys, and sink, using context's teacher tensors.
# The selected sparse-attention LSE excludes sink and is not this denominator.
full_lse = ...
index_scale = (qi.shape[2] * qi.shape[3]) ** -0.5
index_scores, teacher_scores = dense_indexer_loss_scores(
    qi, ki, wi, qa, kc, full_lse,
    ratio=csa.compression_ratio, index_scale=index_scale,
    attention_scale=csa.head_dim ** -0.5,
)
index_grads = dense_indexer_loss_backward(
    qi, ki, wi, index_scores, teacher_scores,
    torch.ones((), device=qi.device),
    ratio=csa.compression_ratio, index_scale=index_scale,
    loss_coeff=loss_coefficient,
)
torch.autograd.backward((qi, ki, wi), index_grads)
```

These helpers return score buffers and explicit gradients, not an autograd
loss. The model owns its loss coefficient and propagates its main objective
through `output` separately. Auxiliary gradients reach the live indexer
projections and compressor; they do not reach the shared hidden/query
projections or detached attention teacher. The dense-reference example in
`tests/pytorch/test_dsv4_attention.py` shows full teacher-LSE construction.

Shared-input projections in `DSv4Attention` are fused: `q_a_kv_proj.weight`
stores query-down then local-KV rows; `compressor.fused_proj.weight` stores
KV then gate rows; the CSA `indexer.compressor.fused_proj.weight` additionally
stores head-weight rows last. These default checkpoint keys and shapes are
preserved. Separate-weight model imports must concatenate those rows explicitly.
