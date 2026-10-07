# DSv4 model integration

For stateless, unpadded BF16 CSA/HCA layers, `DSv4HybridAttention` owns the
DSv4 projections, norms, RoPE, compressors, indexer, sink, and grouped output
projection. Its default input and output are both `[sequence, batch, hidden_size]`:

```python
import torch
from transformer_engine.pytorch.attention.sparse_attention import DSv4HybridAttention

attention = DSv4HybridAttention(
    hidden_size=4096, q_lora_rank=1536,
    layer_type="compressed_sparse_attention", head_dim=512, rope_head_dim=64,
    sliding_window=128, compression_ratio=4, o_groups=8, o_lora_rank=1024,
    index_n_heads=32, index_topk=32, max_seqlen=2048,
    params_dtype=torch.bfloat16,
)
output = attention(hidden_states)  # [S, B, 4096] -> [S, B, 4096]
```

```text
SBD ── TE query/local-KV projections + norms ───────────────┐
  ├── TE compressor projections/norm ── cuDNN compression ──┼── cuDNN attention ── TE grouped output ── SBD
  └── (CSA) TE index projections/norm ── cuDNN compression ── cuDNN selection ──┘
```

The high-level layer uses 64 heads of width 512 and defaults to plain interleaved
partial RoPE with `rope_theta`; callers may inject a `rope` module returning token
and compressed `(cos, sin)` pairs.
Use `input_format="bsd"` for batch-major callers. The SBD boundary conversion
is a view when batch size is one; larger batches require a repack for the current
cuDNN packed-row contract.
It supports full sequences starting at position zero, equal unpadded lengths within
a batch, and at least one complete compression window. Sliding-only layers,
cache/decode, padding, TP/CP, and FP8 remain outside this first pass. The
indexer auxiliary loss and its attachment remain caller-owned; without one, the
language-model loss does not train `indexer.q_proj` or the
index-key/weight slices of `indexer.compressor.fused_proj`: top-k selection has no gradient.

`dsa_rope.py` caches FP32 token frequencies (prebuilt when `max_seqlen` is
provided) and shares window-start slices with the CSA `_Indexer`. The indexer
owns its projections, index-key compressor, and block selection; neither helper
changes the cuDNN call contracts.

The default projection layout groups operations that read the same input: one
TE `Linear` for query-down plus local KV, one for compressor KV plus gates, and
in CSA one for index-compressor KV plus gates plus per-head weights. Each weight
is contiguous; output slices feed the existing norms and cuDNN calls. The
current wrapper validates 64 attention heads. `_fuse_projections=False` retains
separate Linear calls as a comparison path.

The cuDNN calls used by this layer are:

| Stage | HCA | CSA |
| --- | --- | --- |
| Compressor forward/backward | Once | Twice: attention KV and index keys |
| Fused indexer score + top-k | — | Once |
| Sparse attention forward/backward | Once | Once |

For CSA, `return_indexer_context=True` returns `(SBD_output, context)` by default; HCA rejects
it. The live BF16 index Q/K/W have shapes `[B*S,H,128]`, `[B*S/ratio,128]`, and
`[B*S,H]`; selected IDs are `[B*S,topk]` global compressed-row IDs. Detached
attention Q/local KV/compressed KV/sink have shapes `[B*S,64,512]`,
`[B*S,512]`, `[B*S/ratio,512]`, and `[64]`. The opt-in context and dense
helpers currently support B1 BF16; the helpers take BSHD views plus a
caller-supplied FP32 full LSE `[1,S,64]` (local
window, all eligible compressed keys, and sink). Score helpers are not autograd
operations; the caller owns the mean-reduced scalar KL, coefficient, and loss
attachment. Per-token reduction is untested.

The lower-level `dsa_cudnn` calls expose compression and selection when
a model owns its own projections, RMSNorm, RoPE, sink, and output projection.
`DSv4Attention` owns the parameter-free attention call:

```text
model projected KV + gates ── dsa_cudnn.compress ── model norm/RoPE ── compressed KV
model index projections ───── dsa_cudnn.compress ── model norm/RoPE ── index key
model index query + key + weights ───────────── dsa_cudnn.select_blocks ── IDs (CSA)
model query + local KV + compressed KV + IDs ─ dsv4_attention.DSv4Attention ── head output
```

The model calls the same stages for HCA, omitting index compression and selection.
It passes `indices=None` and `max_compressed_seqlen` to attention. CSA calls
`compress` twice because its attention memory and index keys use separate
projections. `select_blocks` returns global IDs into the packed **compressed**
rows; the attention core maps them into its combined local/compressed KV space.

```python
from transformer_engine.pytorch.attention.sparse_attention import (
    dsa_cudnn,
    dsv4_attention,
)

# Model code supplies BF16 projected tensors and applies its own norm/RoPE.
pooled = dsa_cudnn.compress(
    kv, gates, position_bias.float(), cu, cu_comp,
    ratio=ratio, overlap=is_csa, total_comp=total_comp,
)
compressed_kv = model.finish_compressed(pooled)

indices = None
if is_csa:
    index_pooled = dsa_cudnn.compress(
        index_kv, index_gates, index_position_bias.float(), cu, cu_comp,
        ratio=ratio, overlap=True, total_comp=total_comp,
    )
    index_key = model.finish_index_key(index_pooled)
    indices = dsa_cudnn.select_blocks(
        index_query, index_key, index_weights, cu, cu_comp,
        top_k=top_k, ratio=ratio, max_seqlen=max_seqlen,
        max_compressed_seqlen=max_compressed_seqlen,
        scale=(index_head_dim * index_n_heads) ** -0.5,
    )

output = dsv4_attention.DSv4Attention(window_size=window_size, ratio=ratio)(
    query, local_kv, compressed_kv, sink.float(), cu, cu_comp,
    indices=indices,
    max_compressed_seqlen=max_compressed_seqlen if indices is None else None,
)
output = model.finish_output(output)
```

This is a wiring sketch: `model.finish_*` are model-owned transforms, not TE APIs.
The model must apply its own normalization and RoPE to `query`, `local_kv`,
`compressed_kv`, `index_query`, and `index_key` before their TE calls. `cu` and
`cu_comp` are CUDA INT32 packed-sequence prefixes; `total_comp` is the valid
number of compressed rows. All Q/KV tensors passed to these stages are
contiguous CUDA BF16. For CSA, pass the raw BF16 `index_weights` projection and
put both positive index scaling factors in `scale` to preserve the FP32 score
scale without an extra BF16 weight rounding. The cuDNN compressor pools with
FP32 intermediates and one BF16 output cast, so keys can differ slightly from
an eager model that rounds its softmax weights or products to BF16.

The current core supports 64 query heads and head width 512 or 576 on SM100;
the selector supports 32 or 64 index heads of width 128. It handles
full-sequence prefill starting at position zero, with exact packed row counts
in the prefix arrays. A model using segment-local compressed IDs must convert
them to the global packed IDs expected here. Cache/decode, padded-capacity
attention rows, context parallelism, and the separate indexer auxiliary loss
are outside this API.
