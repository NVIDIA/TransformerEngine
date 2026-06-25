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
            F16
                BSHD/SBHD
                    -> Table 10: P2P F16 BSHD/SBHD
                THD
                    -> Table 11: P2P F16 THD
                    pad_between_seqs=T
                        -> Table 12: P2P F16 THD pad_between_seqs=T
            FP8 DelayedS
                BSHD/SBHD
                    fp8_dpa/fp8_mha
                        -> Table 13: P2P FP8 DelayedS BSHD/SBHD
                THD
                    -> Table 14: P2P FP8 DelayedS THD
            FP8 CurrentS
                -> Table 15: P2P FP8 CurrentS
            MXFP8
                -> Table 16: P2P MXFP8
        A2A+P2P
            F16
                BSHD/SBHD
                    -> Table 17: A2A+P2P F16 BSHD/SBHD
                THD
                    -> Table 18: A2A+P2P F16 THD
                    pad_between_seqs=T
                        -> Table 19: A2A+P2P F16 THD pad_between_seqs=T
            FP8 DelayedS
                BSHD/SBHD
                    fp8_dpa/fp8_mha
                        -> Table 20: A2A+P2P FP8 DelayedS BSHD/SBHD
                THD
                    -> Table 21: A2A+P2P FP8 DelayedS THD
            FP8 CurrentS
                -> Table 22: A2A+P2P FP8 CurrentS
            MXFP8
                -> Table 23: A2A+P2P MXFP8
        AG
            F16
                BSHD/SBHD
                    -> Table 24: AG F16 BSHD/SBHD
                THD
                    -> Table 25: AG F16 THD
                    pad_between_seqs=T
                        -> Table 26: AG F16 THD pad_between_seqs=T
            FP8 DelayedS
                BSHD/SBHD
                    fp8_dpa/fp8_mha
                        -> Table 27: AG FP8 DelayedS BSHD/SBHD
                THD
                    -> Table 28: AG FP8 DelayedS THD
            FP8 CurrentS
                -> Table 29: AG FP8 CurrentS
            MXFP8
                -> Table 30: AG MXFP8
    Flash Attention
        -> Table 31: Flash Attention
    Unfused Attention
        -> Table 32: Unfused Attention
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
| Bias | No | FusedAttention CP keeps explicit bias only for `p2p`; A2A uses no bias. |
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

### Table 10: P2P F16 BSHD/SBHD

| Feature | Support | Notes |
| --- | --- | --- |
| MHA/MQA/GQA | Supported | `p2p` is the broadest FusedAttention CP communication mode. |
| MLA | Limited | Selected unequal QK/V head dimensions can remain. |
| SWA | No | FusedAttention CP removes SWA for `p2p`. |
| Mask types | Limited | No bottom-right CP masks, causal cross-attention, or arbitrary masks. |
| Bias | Limited | `p2p` can keep selected post-scale bias; pre-scale bias is removed. |
| Sink softmax | No | CP non-vanilla softmax keeps A2A Fused paths, not `p2p`. |
| `return_max_logit` | Limited | Non-FP8 only, and cuDNN must accept the shape. |
| Determinism | Limited | Depends on shape, bias, architecture, and cuDNN gates. |
| BR/TL | Limited | Top-left self-attention can remain; bottom-right and causal cross-attention are removed under CP. |

### Table 11: P2P F16 THD

| Feature | Support | Notes |
| --- | --- | --- |
| MHA/MQA/GQA | Limited | THD CP depends on cuDNN shape support. |
| MLA | Limited | Selected shapes only. |
| SWA | No | FusedAttention CP removes SWA for `p2p`. |
| Mask types | Limited | Padding-style masks only; bottom-right CP masks are removed. |
| Bias | No | THD CP removes attention bias. |
| Sink softmax | No | CP non-vanilla softmax keeps A2A Fused paths, not `p2p`. |
| `return_max_logit` | Limited | Non-FP8 only, and cuDNN must accept the shape. |
| Determinism | Limited | Depends on shape, architecture, and cuDNN gates. |

### Table 12: P2P F16 THD `pad_between_seqs=True`

| Backend family | Support | Notes |
| --- | --- | --- |
| Fused F16 | Limited | Depends on cuDNN THD support. |
| FlashAttention | Limited | FA2 and FA4 are removed; FA3 can remain if other gates pass. |
| Unfused | No | Context parallelism and `pad_between_seqs=True` remove Unfused. |

### Table 13: P2P FP8 DelayedS BSHD/SBHD

| Feature | Support | Notes |
| --- | --- | --- |
| `fp8_dpa=True` | Limited | Architecture and cuDNN gates still apply. |
| `fp8_mha=True` | Limited | KV cache with FP8 MHA is removed. |
| MHA/MQA/GQA | Limited | Broadly plausible when FP8 gates pass. |
| MLA | Limited | Selected shapes only. |
| SWA | No | FusedAttention CP removes SWA for `p2p`. |
| Mask types | Limited | No bottom-right CP masks or arbitrary masks. |
| Bias | No | Fused FP8 requires no bias. |
| Sink softmax | No | CP non-vanilla softmax keeps A2A Fused paths, not `p2p`. |
| `return_max_logit` | No | Removed for FP8. |
| Determinism | Limited | Depends on architecture and cuDNN gates. |

### Table 14: P2P FP8 DelayedS THD

| Feature | Support | Notes |
| --- | --- | --- |
| Fused FP8 with CP THD | No | The selector removes FusedAttention for FP8 CP with THD format. |
| FlashAttention fallback | Limited | Only if FlashAttention CP constraints pass. |
| Unfused fallback | No | Context parallelism removes Unfused. |

### Table 15: P2P FP8 CurrentS

| Feature | Support | Notes |
| --- | --- | --- |
| Dense layouts | Limited | Broadly `sm100+` and selected cuDNN-supported shapes. |
| THD | No | Not the target path for FP8 CurrentS CP. |
| Bias | No | Fused FP8 requires no bias. |
| Sink softmax | No | CP non-vanilla softmax keeps A2A Fused paths, not `p2p`. |
| `return_max_logit` | No | Removed for FP8. |
| Determinism | Limited | Depends on architecture and cuDNN gates. |

### Table 16: P2P MXFP8

| Feature | Support | Notes |
| --- | --- | --- |
| Dense layouts | Limited | Broadly `sm100+` and selected cuDNN-supported shapes. |
| THD | No | FusedAttention MXFP8 removes THD. |
| `fp8_mha=True` | No | FusedAttention MXFP8 excludes this mode. |
| Bias | No | Fused FP8 requires no bias. |
| Sink softmax | No | CP non-vanilla softmax keeps A2A Fused paths, not `p2p`. |
| `return_max_logit` | No | Removed for FP8. |
| Determinism | Limited | Depends on architecture and cuDNN gates. |

### Table 17: A2A+P2P F16 BSHD/SBHD

| Feature | Support | Notes |
| --- | --- | --- |
| MHA/MQA/GQA | Limited | Requires even `num_heads` and `num_gqa_groups`. |
| MLA | Limited | Selected unequal QK/V head dimensions can remain. |
| SWA | No | FusedAttention CP removes SWA for `a2a+p2p`. |
| Mask types | Limited | No bottom-right CP masks, causal cross-attention, or arbitrary masks. |
| Bias | No | FusedAttention CP keeps explicit bias only for `p2p`. |
| Sink softmax | No | CP non-vanilla softmax keeps A2A Fused paths, not `a2a+p2p`. |
| `return_max_logit` | Limited | Non-FP8 only, and cuDNN must accept the shape. |
| Determinism | Limited | Depends on shape, architecture, and cuDNN gates. |

### Table 18: A2A+P2P F16 THD

| Feature | Support | Notes |
| --- | --- | --- |
| Fused F16 with CP THD | No | The selector removes FusedAttention for THD with `a2a+p2p`. |
| FlashAttention fallback | Limited | Only if FlashAttention CP constraints pass. |
| Unfused fallback | No | Context parallelism removes Unfused. |

### Table 19: A2A+P2P F16 THD `pad_between_seqs=True`

| Backend family | Support | Notes |
| --- | --- | --- |
| Fused F16 | No | THD is removed for FusedAttention with `a2a+p2p`. |
| FlashAttention | Limited | FA2 and FA4 are removed; FA3 can remain if other gates pass. |
| Unfused | No | Context parallelism removes Unfused. |

### Table 20: A2A+P2P FP8 DelayedS BSHD/SBHD

| Feature | Support | Notes |
| --- | --- | --- |
| `fp8_dpa=True` | Limited | Requires even head counts plus architecture and cuDNN gates. |
| `fp8_mha=True` | Limited | KV cache with FP8 MHA is removed. |
| MHA/MQA/GQA | Limited | Even `num_heads` and `num_gqa_groups` are required. |
| MLA | Limited | Selected shapes only. |
| SWA | No | FusedAttention CP removes SWA for `a2a+p2p`. |
| Mask types | Limited | No bottom-right CP masks or arbitrary masks. |
| Bias | No | Fused FP8 requires no bias. |
| Sink softmax | No | CP non-vanilla softmax keeps A2A Fused paths, not `a2a+p2p`. |
| `return_max_logit` | No | Removed for FP8. |
| Determinism | Limited | Depends on architecture and cuDNN gates. |

### Table 21: A2A+P2P FP8 DelayedS THD

| Feature | Support | Notes |
| --- | --- | --- |
| Fused FP8 with CP THD | No | The selector removes FusedAttention for FP8 CP with THD format. |
| FlashAttention fallback | Limited | Only if FlashAttention CP constraints pass. |
| Unfused fallback | No | Context parallelism removes Unfused. |

### Table 22: A2A+P2P FP8 CurrentS

| Feature | Support | Notes |
| --- | --- | --- |
| Dense layouts | Limited | Broadly `sm100+`, even head counts, and selected cuDNN-supported shapes. |
| THD | No | Not the target path for FP8 CurrentS CP. |
| Bias | No | Fused FP8 requires no bias. |
| Sink softmax | No | CP non-vanilla softmax keeps A2A Fused paths, not `a2a+p2p`. |
| `return_max_logit` | No | Removed for FP8. |
| Determinism | Limited | Depends on architecture and cuDNN gates. |

### Table 23: A2A+P2P MXFP8

| Feature | Support | Notes |
| --- | --- | --- |
| Dense layouts | Limited | Broadly `sm100+`, even head counts, and selected cuDNN-supported shapes. |
| THD | No | FusedAttention MXFP8 removes THD. |
| `fp8_mha=True` | No | FusedAttention MXFP8 excludes this mode. |
| Bias | No | Fused FP8 requires no bias. |
| Sink softmax | No | CP non-vanilla softmax keeps A2A Fused paths, not `a2a+p2p`. |
| `return_max_logit` | No | Removed for FP8. |
| Determinism | Limited | Depends on architecture and cuDNN gates. |

### Table 24: AG F16 BSHD/SBHD

AG means `cp_comm_type="all_gather"`.

| Feature | Support | Notes |
| --- | --- | --- |
| MHA/MQA/GQA | Limited | AG does not overlap communication with compute. |
| MLA | Limited | Selected unequal QK/V head dimensions can remain. |
| SWA | Limited | Exact support remains cuDNN-owned. |
| Mask types | Limited | No bottom-right CP masks, causal cross-attention, or arbitrary masks. |
| Bias | No | FusedAttention CP keeps explicit bias only for `p2p`. |
| Sink softmax | No | CP non-vanilla softmax keeps A2A Fused paths, not AG. |
| `return_max_logit` | Limited | Non-FP8 only, and cuDNN must accept the shape. |
| Determinism | Limited | Depends on shape, architecture, and cuDNN gates. |

### Table 25: AG F16 THD

| Feature | Support | Notes |
| --- | --- | --- |
| Fused F16 with CP THD | No | The selector removes FusedAttention for THD with AG. |
| FlashAttention fallback | Limited | Only if FlashAttention CP constraints pass. |
| Unfused fallback | No | Context parallelism removes Unfused. |

### Table 26: AG F16 THD `pad_between_seqs=True`

| Backend family | Support | Notes |
| --- | --- | --- |
| Fused F16 | No | THD is removed for FusedAttention with AG. |
| FlashAttention | Limited | FA2 and FA4 are removed; FA3 can remain if other gates pass. |
| Unfused | No | Context parallelism removes Unfused. |

### Table 27: AG FP8 DelayedS BSHD/SBHD

| Feature | Support | Notes |
| --- | --- | --- |
| `fp8_dpa=True` | Limited | Architecture and cuDNN gates still apply. |
| `fp8_mha=True` | Limited | KV cache with FP8 MHA is removed. |
| MHA/MQA/GQA | Limited | Broadly plausible when FP8 gates pass. |
| MLA | Limited | Selected shapes only. |
| SWA | Limited | Exact support remains cuDNN-owned. |
| Mask types | Limited | No bottom-right CP masks or arbitrary masks. |
| Bias | No | Fused FP8 requires no bias. |
| Sink softmax | No | CP non-vanilla softmax keeps A2A Fused paths, not AG. |
| `return_max_logit` | No | Removed for FP8. |
| Determinism | Limited | Depends on architecture and cuDNN gates. |

### Table 28: AG FP8 DelayedS THD

| Feature | Support | Notes |
| --- | --- | --- |
| Fused FP8 with CP THD | No | The selector removes FusedAttention for FP8 CP with THD format. |
| FlashAttention fallback | Limited | Only if FlashAttention CP constraints pass. |
| Unfused fallback | No | Context parallelism removes Unfused. |

### Table 29: AG FP8 CurrentS

| Feature | Support | Notes |
| --- | --- | --- |
| Dense layouts | Limited | Broadly `sm100+` and selected cuDNN-supported shapes. |
| THD | No | Not the target path for FP8 CurrentS CP. |
| Bias | No | Fused FP8 requires no bias. |
| Sink softmax | No | CP non-vanilla softmax keeps A2A Fused paths, not AG. |
| `return_max_logit` | No | Removed for FP8. |
| Determinism | Limited | Depends on architecture and cuDNN gates. |

### Table 30: AG MXFP8

| Feature | Support | Notes |
| --- | --- | --- |
| Dense layouts | Limited | Broadly `sm100+` and selected cuDNN-supported shapes. |
| THD | No | FusedAttention MXFP8 removes THD. |
| `fp8_mha=True` | No | FusedAttention MXFP8 excludes this mode. |
| Bias | No | Fused FP8 requires no bias. |
| Sink softmax | No | CP non-vanilla softmax keeps A2A Fused paths, not AG. |
| `return_max_logit` | No | Removed for FP8. |
| Determinism | Limited | Depends on architecture and cuDNN gates. |

### Table 31: Flash Attention

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

### Table 32: Unfused Attention

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
