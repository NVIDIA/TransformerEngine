# Attention Backend Selection Matrix Draft

This Markdown file is the editable support-matrix companion to
`attention_backend_selection.rst`. The linked RST page is the Sphinx document;
this draft keeps the same broad support information in a resolver format that
matches the CP/Fused/Flash outline in `attention_docs/misc/ideas.md`.

Scope:

- PyTorch `DotProductAttention`.
- JAX attention backend selection is out of scope for this draft.
- Exact cuDNN patch-level gates and FlashAttention package-version gates remain
  in the runtime selector:
  `transformer_engine/pytorch/attention/dot_product_attention/utils.py` and
  `transformer_engine/common/fused_attn/fused_attn.cpp`.

Legend:

| Term | Meaning |
| --- | --- |
| Supported | The backend family is intended to support this row when package, hardware, shape, and runtime gates pass. |
| Limited | Some common forms are supported, but narrower shape, layout, architecture, package, or cuDNN gates still apply. |
| No | The selector removes this backend family for the feature class. |
| N/A | The row does not apply to that backend family. |

Backend names:

| Name | Meaning |
| --- | --- |
| FA2 | FlashAttention 2 |
| FA3 | FlashAttention 3 |
| FA4 | FlashAttention 4 |
| Fused F16 | cuDNN FusedAttention for FP16/BF16 inputs |
| Fused FP8 | cuDNN FusedAttention for FP8 DPA |
| Unfused | `UnfusedDotProductAttention` |

## Resolver Order

Read the tables in this order:

1. Pick the execution family: no CP, CP, or KV cache.
2. If CP is enabled, apply the CP communication resolver first.
3. If FusedAttention remains, apply the FusedAttention resolver for
   communication mode, precision, and layout.
4. If FlashAttention remains, apply the FlashAttention resolver.
5. Apply the cross-cutting modifier resolver for masks, bias, SWA, sink
   softmax, `return_max_logit`, determinism, and `num_splits`.

Final priority still matters. If FlashAttention and FusedAttention both remain,
Transformer Engine prefers FusedAttention on Hopper and newer GPUs and
FlashAttention on older supported GPUs. Unfused is the fallback when accelerated
backends do not support the requested configuration.

## Top-Level Backend Matrix

| Scenario | FlashAttention | Fused F16 | Fused FP8 | Unfused | Notes |
| --- | --- | --- | --- | --- | --- |
| FP16/BF16 dense self-attention | Supported | Supported | N/A | Supported | Common `bshd`/`sbhd` path with standard masks. |
| FP32 attention | No | No | N/A | Supported | Accelerated backends require FP16/BF16 or FP8 inputs. |
| FP8 DPA | Limited | N/A | Supported | Limited | FlashAttention is limited to selected FA3 inference paths; Unfused FP8 requires emulation or ONNX export mode. |
| GQA or MQA | Supported | Supported | Supported | Supported | Some backends require compatible head-group divisibility. |
| MLA or unequal QK/V head dimensions | Limited | Limited | Limited | Supported | FA2 is removed; FA3, FA4, and cuDNN accept selected shapes. |
| THD packed variable-length input | Limited | Supported | Limited | Limited | Padding between packed sequences narrows FlashAttention and Unfused support. |
| Arbitrary attention mask | No | No | No | Supported | Arbitrary masks route to Unfused for non-FP8 attention. |
| KV cache | Limited | Limited | No | Supported | KV cache cannot be combined with CP; FA4 is removed for KV cache. |
| `score_mod` callback | No | Limited | No | No | Requires a narrow cuDNN F16/BF16 path. |

## CP Resolver

Context parallelism removes Unfused attention. It also removes FA4 and narrows
FlashAttention and FusedAttention before package or cuDNN gates run.

| CP communication | FlashAttention | Fused F16 | Fused FP8 | Unfused | Notes |
| --- | --- | --- | --- | --- | --- |
| No CP | Supported | Supported | Supported | Supported | Use the non-CP Fused and Flash resolvers below. |
| `p2p` | Limited | Limited | Limited | No | Broadest Fused CP mode; selected post-scale bias can remain. |
| `all_gather` / AG | Limited | Limited | Limited | No | THD FusedAttention is removed; THD CP routes through FlashAttention if available. |
| `a2a` | Limited | Limited | Limited | No | FusedAttention requires even `num_heads` and `num_gqa_groups`. |
| `a2a+p2p` | Limited | Limited | Limited | No | FusedAttention requires even head counts and removes THD. |

| CP feature | FlashAttention | Fused F16 | Fused FP8 |
| --- | --- | --- | --- |
| `bshd`/`sbhd` FP16/BF16 | Supported with CP mask and bias limits. | Supported with CP mask, bias, and communication limits. | N/A |
| `thd` FP16/BF16 | Supported with CP mask and bias limits. | No for AG and `a2a+p2p`; limited otherwise. | N/A |
| FP8 DPA | No | N/A | Limited to non-THD and no-bias paths. |
| Bottom-right causal masks | No | No | No |
| Causal cross-attention | No | No | No |
| Explicit bias | Limited to selected post-scale bias. | Limited to selected post-scale bias; `p2p` only when bias is present. | No |
| Sliding window attention | Limited | No for `p2p` and `a2a+p2p`; limited otherwise. | Limited |
| Non-vanilla softmax | No | Limited to `a2a`. | Limited to `a2a` and newer cuDNN paths. |

## FusedAttention Resolver Matrix

This section follows the shape sketched in `misc/ideas.md`: CP communication
mode first, then precision/recipe, then layout and feature support.

### FusedAttention by CP Communication and Precision

| CP communication | F16 `bshd`/`sbhd` | F16 `thd` | FP8 delayed `bshd`/`sbhd` | FP8 delayed `thd` | FP8 current scaling | MXFP8 | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No CP | Supported | Supported | Supported | Limited | Supported on `sm100+` | Supported on `sm100+` | Exact shape support comes from cuDNN. |
| `p2p` | Limited | Limited | Limited | No | Limited | Limited | Broadest CP mode for FusedAttention. |
| `a2a` | Limited | Limited | Limited | No | Limited | Limited | Requires even `num_heads` and `num_gqa_groups`. |
| `a2a+p2p` | Limited | No | Limited | No | Limited | Limited | THD is removed for FusedAttention in this mode. |
| AG / `all_gather` | Limited | No | Limited | No | Limited | Limited | THD is removed for FusedAttention in this mode. |

### A2A Resolver

| Precision / layout | MHA/GQA/MQA | MLA | SWA | Masks | Bias | Sink softmax | `return_max_logit` | Determinism | Support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F16 `bshd`/`sbhd` | Supported | Limited | Limited | No bottom-right CP masks or causal cross-attention. | No bias or selected post-scale bias. | Limited | Limited | Limited | Limited |
| F16 `thd` | Supported | Limited | Limited | Padding-style masks only; no bottom-right CP masks. | No bias for THD CP. | Limited | Limited | Limited | Limited |
| FP8 delayed scaling | Limited | Limited | Limited | No bottom-right CP masks. | No bias. | Limited | No | Limited | Limited |
| FP8 current scaling | Limited on `sm100+` | Limited | Limited | No bottom-right CP masks. | No bias. | Limited | No | Limited | Limited |
| MXFP8 | Limited on `sm100+` | Limited | Limited | No bottom-right CP masks. | No bias. | Limited | No | Limited | Limited |

### P2P Resolver

| Precision / layout | MHA/GQA/MQA | MLA | SWA | Masks | Bias | Sink softmax | `return_max_logit` | Determinism | Support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F16 `bshd`/`sbhd` | Supported | Limited | No with CP SWA restrictions. | No bottom-right CP masks or causal cross-attention. | No bias or selected post-scale bias. | No | Limited | Limited | Limited |
| F16 `thd` | Supported | Limited | No with CP SWA restrictions. | Padding-style masks only; no bottom-right CP masks. | No bias. | No | Limited | Limited | Limited |
| FP8 delayed scaling | Limited | Limited | No with CP SWA restrictions. | No bottom-right CP masks. | No bias. | No | No | Limited | Limited |
| FP8 current scaling | Limited on `sm100+` | Limited | No with CP SWA restrictions. | No bottom-right CP masks. | No bias. | No | No | Limited | Limited |
| MXFP8 | Limited on `sm100+` | Limited | No with CP SWA restrictions. | No bottom-right CP masks. | No bias. | No | No | Limited | Limited |

### A2A+P2P Resolver

| Precision / layout | MHA/GQA/MQA | MLA | SWA | Masks | Bias | Sink softmax | `return_max_logit` | Determinism | Support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F16 `bshd`/`sbhd` | Supported with even head counts. | Limited | No with CP SWA restrictions. | No bottom-right CP masks or causal cross-attention. | No bias unless CP mode permits the selected bias path. | No | Limited | Limited | Limited |
| F16 `thd` | No | No | No | No | No | No | No | No | No |
| FP8 delayed scaling | Limited with even head counts. | Limited | No with CP SWA restrictions. | No bottom-right CP masks. | No bias. | No | No | Limited | Limited |
| FP8 current scaling | Limited on `sm100+` with even head counts. | Limited | No with CP SWA restrictions. | No bottom-right CP masks. | No bias. | No | No | Limited | Limited |
| MXFP8 | Limited on `sm100+` with even head counts. | Limited | No with CP SWA restrictions. | No bottom-right CP masks. | No bias. | No | No | Limited | Limited |

### AG / All-Gather Resolver

| Precision / layout | MHA/GQA/MQA | MLA | SWA | Masks | Bias | Sink softmax | `return_max_logit` | Determinism | Support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F16 `bshd`/`sbhd` | Supported | Limited | Limited | No bottom-right CP masks or causal cross-attention. | No bias unless CP mode permits the selected bias path. | No | Limited | Limited | Limited |
| F16 `thd` | No | No | No | No | No | No | No | No | No |
| FP8 delayed scaling | Limited | Limited | Limited | No bottom-right CP masks. | No bias. | No | No | Limited | Limited |
| FP8 current scaling | Limited on `sm100+` | Limited | Limited | No bottom-right CP masks. | No bias. | No | No | Limited | Limited |
| MXFP8 | Limited on `sm100+` | Limited | Limited | No bottom-right CP masks. | No bias. | No | No | Limited | Limited |

### Non-CP FusedAttention Resolver

| Precision / layout | MHA/GQA/MQA | MLA | SWA | Masks | Bias | Sink softmax | `return_max_logit` | Determinism | Support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F16 `bshd`/`sbhd` | Supported | Limited | Supported with Fused gates. | Standard masks, including selected bottom-right masks; no arbitrary mask. | No bias, selected post-scale bias, selected ALiBi conversion. | Supported when cuDNN accepts the shape. | Limited | Limited | Supported |
| F16 `thd` | Supported | Limited | Limited | Padding-style masks and selected bottom-right padding masks; no arbitrary mask. | No bias or selected post-scale bias depending on shape. | Limited | Limited | Limited | Supported |
| FP8 delayed scaling | Supported | Limited | Limited | Standard no-mask, causal, padding, and selected bottom-right forms. | No bias. | Limited | No | Limited | Supported |
| FP8 current scaling | Supported on `sm100+` | Limited | Limited | Standard no-mask, causal, padding, and selected bottom-right forms. | No bias. | Limited | No | Limited | Supported on `sm100+` |
| MXFP8 | Supported on `sm100+` | Limited | Limited | Standard supported FP8 masks; THD is excluded. | No bias. | Limited | No | Limited | Supported on `sm100+` |

## FlashAttention Resolver

| Backend | Architecture | Precision | CP | KV cache | Broad exclusions |
| --- | --- | --- | --- | --- | --- |
| FA2 | `sm80+` | FP16/BF16 | Limited | Limited | No FP8 DPA, MLA, arbitrary mask, or THD padding-between-sequences. |
| FA3 | `sm90` | FP16/BF16; limited FP8 inference | Limited | Limited | No dropout, ALiBi, explicit bias, arbitrary mask, or FP8 training. |
| FA4 | `sm80+` | FP16/BF16 | No | No | No FP8 DPA, THD padding-between-sequences, dropout, bias, or ALiBi. |

## Mask And Modifier Resolver

| Feature | FlashAttention | FusedAttention | Unfused |
| --- | --- | --- | --- |
| `no_mask` and padding masks | Supported | Supported | Supported |
| Top-left causal cross-attention | No | Supported | Supported |
| Bottom-right causal masks | Supported outside CP | Supported outside CP | Supported |
| Arbitrary mask | No | No | Supported |
| ALiBi | FA2 only, with limits. | Limited through post-scale bias conversion. | Supported |
| Explicit pre/post-scale bias | No | Limited post-scale bias only. | Supported |
| Dropout | FA2 only. | Supported except for selected sliding-window paths. | Supported |
| `return_max_logit=True` | No | Supported for non-FP8 paths when cuDNN accepts the shape. | Supported for non-FP8 paths. |
| `num_splits != 1` | Limited to FA3 or FA4 SplitKV on supported hardware. | No | No |

## When To Check The Code

The tables above answer whether a feature class is intended to be supported by
a backend family. Check the runtime selector when a configuration depends on:

- exact cuDNN version or known-bad cuDNN versions,
- exact FlashAttention package version or installed FlashAttention family,
- Blackwell, `sm120`, or architecture-specific workarounds,
- head dimensions above 128, MLA, or mixed Q/K/V dimensions,
- paged KV cache, THD padding-between-sequences, or split Q/KV layouts,
- deterministic training with bias, FP8, CUDA graphs, or non-vanilla softmax.
