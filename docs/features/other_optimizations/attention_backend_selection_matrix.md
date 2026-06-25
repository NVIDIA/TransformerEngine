# Attention Backend Selection ASCII Outline Draft

This Markdown file is the editable companion to
`attention_backend_selection.rst`. It starts from the shape in
`attention_docs/misc/ideas.md`: first an ASCII outline, then compact support
tables at the leaves where the outline stops.

Scope:

- PyTorch `DotProductAttention` is the first target.
- JAX is left as a later pass.
- Exact cuDNN patch-level gates and FlashAttention package-version gates stay
  in `transformer_engine/pytorch/attention/dot_product_attention/utils.py` and
  `transformer_engine/common/fused_attn/fused_attn.cpp`.

Legend:

| Term | Meaning |
| --- | --- |
| Supported | Intended to work when package, hardware, shape, and runtime gates pass. |
| Limited | Some common forms work, but narrower shape, layout, architecture, package, or cuDNN gates still apply. |
| No | The selector removes this backend family for this feature class. |

## Outline

```text
CP docs
    PyTorch/JAX
        -> Table 1: documentation scope
    Load balance strategy
        -> Table 2: PyTorch CP load-balance scope
    Fused Attention
        A2A
            F16
                BSHD/SBHD
                    -> Table 3: A2A F16 BSHD/SBHD
                THD
                    -> Table 4: A2A F16 THD
                    pad_between_seqs=T
                        -> Table 5: A2A F16 THD pad_between_seqs=T
            FP8 DelayedS
                BSHD/SBHD
                    fp8_dpa/fp8_mha
                        -> Table 6: A2A FP8 DelayedS BSHD/SBHD
                THD
                    -> Table 7: A2A FP8 DelayedS THD
            FP8 CurrentS
                -> Table 8: A2A FP8 CurrentS
            MXFP8
                -> Table 9: A2A MXFP8
        P2P
            -> Table 10: P2P Fused Attention
        A2A+P2P
            -> Table 11: A2A+P2P Fused Attention
        AG
            -> Table 12: AG Fused Attention
    Flash Attention
        -> Table 13: Flash Attention
    Unfused Attention
        -> Table 14: Unfused Attention
```

## Tables

### Table 1: Documentation Scope

| Area | Scope | Notes |
| --- | --- | --- |
| PyTorch | Supported | Prototype target for this draft. |
| JAX | Later | Add separately instead of inferring from PyTorch behavior. |

### Table 2: PyTorch CP Load-Balance Scope

| Item | Support | Notes |
| --- | --- | --- |
| DCP load balance | Supported | Document this first for PyTorch. |
| CP without KV cache | Supported | Backend support is resolved below. |
| CP with KV cache | No | The selector removes all backend families. |
| CP with Unfused attention | No | Context parallelism removes Unfused. |
| CP with FA4 | No | Current selector removes FA4 for CP. |

### Table 3: A2A F16 BSHD/SBHD

| Feature | Support | Notes |
| --- | --- | --- |
| MHA/MQA/GQA | Supported | Requires even `num_heads` and `num_gqa_groups` for A2A. |
| MLA | Limited | Selected unequal QK/V head dimensions can remain. |
| SWA | Limited | CP SWA keeps only selected communication and mask forms. |
| Mask types | Limited | No bottom-right CP masks, causal cross-attention, or arbitrary masks. |
| Bias | Limited | No bias and selected post-scale bias can remain; pre-scale bias is removed. |
| Sink softmax | Limited | A2A is the CP mode that can keep non-vanilla softmax. |
| `return_max_logit` | Limited | Non-FP8 only, and cuDNN must accept the shape. |
| Determinism | Limited | Trainable bias, architecture, or cuDNN gates can remove FusedAttention. |
| BR/TL | Limited | Top-left self-attention can remain; bottom-right and causal cross-attention are removed under CP. |

### Table 4: A2A F16 THD

| Feature | Support | Notes |
| --- | --- | --- |
| MHA/MQA/GQA | Limited | THD CP depends on cuDNN shape support. |
| MLA | Limited | Selected shapes only. |
| SWA | Limited | CP SWA constraints still apply. |
| Mask types | Limited | Padding-style masks only; bottom-right CP masks are removed. |
| Bias | No | THD CP removes attention bias. |
| Sink softmax | Limited | A2A can keep selected non-vanilla softmax paths. |
| `return_max_logit` | Limited | Non-FP8 only, and cuDNN must accept the shape. |
| Determinism | Limited | Depends on shape, bias, architecture, and cuDNN gates. |

### Table 5: A2A F16 THD `pad_between_seqs=True`

| Backend family | Support | Notes |
| --- | --- | --- |
| Fused F16 | Limited | Depends on cuDNN THD support. |
| FlashAttention | Limited | FA2 and FA4 are removed; FA3 can remain if other gates pass. |
| Unfused | No | Removed for `pad_between_seqs=True` in THD. |

### Table 6: A2A FP8 DelayedS BSHD/SBHD

| Feature | Support | Notes |
| --- | --- | --- |
| `fp8_dpa=True` | Limited | Architecture and cuDNN gates still apply. |
| `fp8_mha=True` | Limited | KV cache with FP8 MHA is removed. |
| MHA/MQA/GQA | Limited | Broadly plausible when FP8 gates pass. |
| MLA | Limited | Selected shapes only. |
| SWA | Limited | CP SWA constraints still apply. |
| Mask types | Limited | No bottom-right CP masks or arbitrary masks. |
| Bias | No | Fused FP8 requires no bias. |
| Sink softmax | Limited | Requires newer cuDNN-supported paths. |
| `return_max_logit` | No | Removed for FP8. |
| Determinism | Limited | Depends on architecture and cuDNN gates. |

### Table 7: A2A FP8 DelayedS THD

| Feature | Support | Notes |
| --- | --- | --- |
| Fused FP8 with CP THD | No | The selector removes FusedAttention for FP8 CP with THD format. |
| FlashAttention fallback | Limited | Only if FlashAttention CP constraints pass. |
| Unfused fallback | No | Context parallelism removes Unfused. |

### Table 8: A2A FP8 CurrentS

| Feature | Support | Notes |
| --- | --- | --- |
| Dense layouts | Limited | Broadly `sm100+` and selected cuDNN-supported shapes. |
| THD | No | Not the target path for FP8 CurrentS CP. |
| Bias | No | Fused FP8 requires no bias. |
| Sink softmax | Limited | Requires newer cuDNN-supported paths. |
| `return_max_logit` | No | Removed for FP8. |
| Determinism | Limited | Depends on architecture and cuDNN gates. |

### Table 9: A2A MXFP8

| Feature | Support | Notes |
| --- | --- | --- |
| Dense layouts | Limited | Broadly `sm100+` and selected cuDNN-supported shapes. |
| THD | No | FusedAttention MXFP8 removes THD. |
| `fp8_mha=True` | No | FusedAttention MXFP8 excludes this mode. |
| Bias | No | Fused FP8 requires no bias. |
| Sink softmax | Limited | Requires newer cuDNN-supported paths. |
| `return_max_logit` | No | Removed for FP8. |
| Determinism | Limited | Depends on architecture and cuDNN gates. |

### Table 10: P2P Fused Attention

| Precision / layout | Support | Notes |
| --- | --- | --- |
| F16 BSHD/SBHD | Limited | Broadest CP mode for FusedAttention; selected post-scale bias can remain. |
| F16 THD | Limited | Padding-style masks only; no THD attention bias. |
| F16 THD `pad_between_seqs=True` | Limited | cuDNN-dependent. |
| FP8 DelayedS BSHD/SBHD | Limited | No bias, no bottom-right CP masks, no `return_max_logit`. |
| FP8 DelayedS THD | No | Fused FP8 CP removes THD. |
| FP8 CurrentS | Limited | Broadly `sm100+`, no bias, no bottom-right CP masks. |
| MXFP8 | Limited | Broadly `sm100+`, no THD, no `fp8_mha=True`, no bias. |
| Sink softmax | No | CP non-vanilla softmax keeps A2A Fused paths, not P2P. |
| SWA | Limited | Removed for P2P combinations that violate CP SWA constraints. |

### Table 11: A2A+P2P Fused Attention

| Precision / layout | Support | Notes |
| --- | --- | --- |
| F16 BSHD/SBHD | Limited | Requires even `num_heads` and `num_gqa_groups`. |
| F16 THD | No | THD is removed for FusedAttention in this CP mode. |
| FP8 DelayedS BSHD/SBHD | Limited | Even head counts, no bias, no bottom-right CP masks. |
| FP8 DelayedS THD | No | Fused FP8 CP removes THD. |
| FP8 CurrentS | Limited | Broadly `sm100+`, even head counts, dense layouts only. |
| MXFP8 | Limited | Broadly `sm100+`, even head counts, no THD, no `fp8_mha=True`. |
| Sink softmax | No | CP non-vanilla softmax keeps A2A Fused paths, not A2A+P2P. |
| SWA | Limited | Removed for A2A+P2P combinations that violate CP SWA constraints. |

### Table 12: AG Fused Attention

AG means `cp_comm_type="all_gather"`.

| Precision / layout | Support | Notes |
| --- | --- | --- |
| F16 BSHD/SBHD | Limited | No causal cross-attention, no bottom-right CP masks, no arbitrary masks. |
| F16 THD | No | THD is removed for FusedAttention with AG. |
| FP8 DelayedS BSHD/SBHD | Limited | No bias, no bottom-right CP masks, no `return_max_logit`. |
| FP8 DelayedS THD | No | Fused FP8 CP removes THD. |
| FP8 CurrentS | Limited | Broadly `sm100+`, no bias, dense layouts only. |
| MXFP8 | Limited | Broadly `sm100+`, no THD, no `fp8_mha=True`, no bias. |
| Sink softmax | No | CP non-vanilla softmax keeps A2A Fused paths, not AG. |
| SWA | Limited | Narrower than no-CP; exact shape remains cuDNN-owned. |

### Table 13: Flash Attention

| Branch | Support | Notes |
| --- | --- | --- |
| FA2 no CP | Supported | `sm80+`, FP16/BF16, no MLA, no FP8 DPA. |
| FA3 no CP | Limited | `sm90`, FP16/BF16 and selected FP8 inference paths. |
| FA4 no CP | Limited | `sm80+`, FP16/BF16, no KV cache. |
| FA2/FA3 CP | Limited | No FP8 CP, no bottom-right CP masks, no causal cross-attention. |
| FA4 CP | No | Removed for CP. |
| THD without `pad_between_seqs` | Limited | Package and shape gates apply. |
| THD with `pad_between_seqs=True` | Limited | FA2 and FA4 are removed; FA3 can remain. |
| Arbitrary mask | No | FlashAttention is removed. |
| Explicit pre/post-scale bias | No | FlashAttention is removed. |
| ALiBi | Limited | FA2 only, with package and alignment limits. |
| Dropout | Limited | FA2 can remain; FA3 and FA4 are removed for nonzero dropout. |

### Table 14: Unfused Attention

| Branch | Support | Notes |
| --- | --- | --- |
| No CP FP32/FP16/BF16 | Supported | Fallback path. |
| CP | No | Context parallelism removes Unfused. |
| Arbitrary mask | Supported | Non-FP8 attention only. |
| FP8 DPA | Limited | Requires emulation or ONNX export mode. |
| KV cache | Supported | Non-FP8 paths. |
| `return_max_logit` | Supported | Non-FP8 paths. |

## When To Check The Code

Use this outline to decide whether a backend family is plausible. Check the
runtime selector for:

- exact cuDNN version or known-bad cuDNN versions,
- exact FlashAttention package version or installed FlashAttention family,
- Blackwell, `sm120`, or architecture-specific workarounds,
- head dimensions above 128, MLA, or mixed Q/K/V dimensions,
- paged KV cache, THD padding-between-sequences, or split Q/KV layouts,
- deterministic training with bias, FP8, CUDA graphs, or non-vanilla softmax.
