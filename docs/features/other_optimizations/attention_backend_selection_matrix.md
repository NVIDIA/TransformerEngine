# Attention Backend Selection Outline Draft

This Markdown file is the editable companion to
`attention_backend_selection.rst`. It intentionally starts from the outline in
`attention_docs/misc/ideas.md` and fills in broad support notes under that
shape, instead of starting with wide support tables.

Scope:

- PyTorch `DotProductAttention` first.
- JAX is a later pass.
- Exact cuDNN patch-level gates and FlashAttention package-version gates stay
  in `transformer_engine/pytorch/attention/dot_product_attention/utils.py` and
  `transformer_engine/common/fused_attn/fused_attn.cpp`.
- Terms used below:
  - Supported: intended to work when package, hardware, shape, and runtime gates
    pass.
  - Limited: some common forms work, but narrower shape, layout, architecture,
    package, or cuDNN gates still apply.
  - No: the selector removes this backend family for this feature class.

## CP Docs

### PyTorch / JAX

- PyTorch is the prototype target for this draft.
- JAX attention backend selection should be added later as a separate pass, not
  inferred from PyTorch behavior.

### Load Balance Strategy

- PyTorch CP documentation should cover DCP first because that is the supported
  load-balance strategy in this scope.
- The backend support notes below assume PyTorch `DotProductAttention` and its
  current `cp_comm_type` values: `p2p`, `all_gather`, `a2a`, and `a2a+p2p`.
- Context parallelism removes `UnfusedDotProductAttention`.
- Context parallelism also removes FA4 and narrows FA2, FA3, Fused F16, and
  Fused FP8 before package or cuDNN gates run.
- CP plus KV cache is not supported.

## Fused Attention

General FusedAttention notes:

- Fused F16 covers FP16/BF16 inputs.
- Fused FP8 covers FP8 DPA recipes, not F16/BF16 attention.
- FusedAttention does not support arbitrary masks.
- FusedAttention CP does not support bottom-right causal masks or causal
  cross-attention.
- Fused FP8 requires no attention bias.
- Exact shape support is owned by cuDNN and should be checked in the fused
  backend probe.

### A2A

#### F16

##### BSHD/SBHD

- MHA / MQA / GQA: Supported, with even `num_heads` and `num_gqa_groups`
  required by A2A.
- MLA: Limited. Selected unequal QK/V head-dimension shapes can remain when the
  layout and cuDNN gates accept them.
- SWA: Limited. CP SWA keeps only selected communication and mask combinations.
- Mask types:
  - `no_mask`, `padding`, `causal`, and `padding_causal`: Limited.
  - `causal_bottom_right` and `padding_causal_bottom_right`: No under CP.
  - `arbitrary`: No.
- Bias:
  - `no_bias`: Supported.
  - selected `post_scale_bias`: Limited.
  - `pre_scale_bias`: No.
  - ALiBi: Limited through post-scale bias conversion when the shape is accepted.
- Sink softmax (`off-by-one` / `learnable`): Limited to FusedAttention CP paths
  that allow A2A.
- `return_max_logit`: Limited for non-FP8 when cuDNN accepts the shape.
- Determinism: Limited. Training with trainable bias, older cuDNN, or selected
  architecture paths can remove FusedAttention.
- BR/TL:
  - top-left causal self-attention: Limited.
  - top-left causal cross-attention: No under CP.
  - bottom-right causal: No under CP.

##### THD

- MHA / MQA / GQA: Limited.
- MLA: Limited.
- SWA: Limited.
- Mask types: padding-style masks only; bottom-right CP masks are removed.
- Bias: no bias for THD CP.
- Sink softmax: Limited.
- `return_max_logit`: Limited for non-FP8 when cuDNN accepts the shape.
- Determinism: Limited.
- `pad_between_seqs=True`: Limited. It removes FlashAttention 2/4 and Unfused;
  FusedAttention depends on cuDNN THD support.

#### FP8 DelayedS

##### BSHD/SBHD

- `fp8_dpa=True`: Supported in broad terms when architecture and cuDNN gates
  pass.
- `fp8_mha=True`: Limited; KV cache with FP8 MHA is removed, and MXFP8 has its
  own `fp8_mha` exclusion.
- MHA / MQA / GQA: Limited.
- MLA: Limited.
- SWA: Limited.
- Mask types: no bottom-right CP masks; arbitrary masks are not supported.
- Bias: no bias only.
- Sink softmax: Limited to newer cuDNN paths.
- `return_max_logit`: No for FP8.
- Determinism: Limited.

##### THD

- Broad support: No for Fused FP8 CP. The selector removes FusedAttention for
  FP8 CP with THD format.
- Use FlashAttention candidates instead if the rest of the CP constraints allow
  them.

#### FP8 CurrentS

- Broad support: Limited to `sm100+` and selected cuDNN-supported shapes.
- Layout: primarily `bshd`/`sbhd`/selected dense layouts; THD is not the target
  path for CP.
- Bias: no bias only.
- Sink softmax: Limited to newer cuDNN paths.
- `return_max_logit`: No.
- Determinism: Limited.

#### MXFP8

- Broad support: Limited to `sm100+` and selected cuDNN-supported shapes.
- `fp8_mha=True`: No for FusedAttention MXFP8.
- THD: No for FusedAttention MXFP8.
- Bias: no bias only.
- Sink softmax: Limited to newer cuDNN paths.
- `return_max_logit`: No.
- Determinism: Limited.

### P2P

#### F16

##### BSHD/SBHD

- MHA / MQA / GQA: Supported in broad terms.
- MLA: Limited.
- SWA: No for CP SWA combinations that the selector removes for `p2p`.
- Mask types:
  - `no_mask`, `padding`, `causal`, and `padding_causal`: Limited.
  - causal cross-attention: No under CP.
  - bottom-right causal masks: No under CP.
  - arbitrary masks: No.
- Bias:
  - no bias: Supported.
  - selected post-scale bias: Limited and most natural under `p2p`.
  - pre-scale bias: No.
- Sink softmax: No for `p2p` CP because non-vanilla softmax in CP keeps only
  A2A FusedAttention paths.
- `return_max_logit`: Limited for non-FP8 when cuDNN accepts the shape.
- Determinism: Limited.
- BR/TL: top-left self-attention is the practical CP causal path; bottom-right
  and causal cross-attention are removed.

##### THD

- MHA / MQA / GQA: Limited.
- MLA: Limited.
- SWA: No for CP SWA combinations removed for `p2p`.
- Mask types: padding-style masks only; no bottom-right CP masks.
- Bias: no bias for THD CP.
- Sink softmax: No for `p2p` CP.
- `return_max_logit`: Limited for non-FP8 when cuDNN accepts the shape.
- Determinism: Limited.
- `pad_between_seqs=True`: Limited and cuDNN-dependent for FusedAttention.

#### FP8 DelayedS

##### BSHD/SBHD

- `fp8_dpa=True`: Limited.
- `fp8_mha=True`: Limited; KV cache with FP8 MHA is removed.
- MHA / MQA / GQA: Limited.
- MLA: Limited.
- SWA: No for CP SWA combinations removed for `p2p`.
- Masks: no bottom-right CP masks; no arbitrary masks.
- Bias: no bias only.
- Sink softmax: No for `p2p` CP.
- `return_max_logit`: No.
- Determinism: Limited.

##### THD

- Broad support: No for Fused FP8 CP with THD.

#### FP8 CurrentS

- Broad support: Limited to `sm100+`.
- CP `p2p`: Limited, no bias, no bottom-right masks, no `return_max_logit`.

#### MXFP8

- Broad support: Limited to `sm100+`.
- CP `p2p`: Limited, no THD, no `fp8_mha=True`, no bias, no
  `return_max_logit`.

### A2A+P2P

#### F16

##### BSHD/SBHD

- MHA / MQA / GQA: Limited; even `num_heads` and `num_gqa_groups` are required.
- MLA: Limited.
- SWA: No for CP SWA combinations removed for `a2a+p2p`.
- Mask types: no causal cross-attention, no bottom-right CP masks, no arbitrary
  masks.
- Bias: Limited; attention bias is narrower than `p2p`.
- Sink softmax: No for `a2a+p2p` CP.
- `return_max_logit`: Limited for non-FP8 when cuDNN accepts the shape.
- Determinism: Limited.

##### THD

- Broad support: No for FusedAttention with `a2a+p2p`.
- THD CP with `a2a+p2p` should be considered FlashAttention-only if the
  FlashAttention constraints pass.

#### FP8 DelayedS

##### BSHD/SBHD

- Broad support: Limited with even head counts.
- Bias: no bias only.
- Sink softmax: No for `a2a+p2p` CP.
- `return_max_logit`: No.
- Determinism: Limited.

##### THD

- Broad support: No for Fused FP8 CP with THD.

#### FP8 CurrentS

- Broad support: Limited to `sm100+`, even head counts, no bias, and selected
  dense layouts.

#### MXFP8

- Broad support: Limited to `sm100+`, even head counts, no THD,
  `fp8_mha=False`, and no bias.

### AG

AG means `cp_comm_type="all_gather"` in the current PyTorch selector.

#### F16

##### BSHD/SBHD

- MHA / MQA / GQA: Limited.
- MLA: Limited.
- SWA: Limited.
- Mask types: no causal cross-attention, no bottom-right CP masks, no arbitrary
  masks.
- Bias: Limited and narrower than `p2p`.
- Sink softmax: No for AG CP.
- `return_max_logit`: Limited for non-FP8 when cuDNN accepts the shape.
- Determinism: Limited.

##### THD

- Broad support: No for FusedAttention with AG.
- THD AG should be considered FlashAttention-only if the FlashAttention
  constraints pass.

#### FP8 DelayedS

##### BSHD/SBHD

- Broad support: Limited.
- Bias: no bias only.
- Sink softmax: No for AG CP.
- `return_max_logit`: No.
- Determinism: Limited.

##### THD

- Broad support: No for Fused FP8 CP with THD.

#### FP8 CurrentS

- Broad support: Limited to `sm100+`, no bias, and selected dense layouts.

#### MXFP8

- Broad support: Limited to `sm100+`, no THD, `fp8_mha=False`, and no bias.

## Flash Attention

General FlashAttention notes:

- FA2: `sm80+`, FP16/BF16, no MLA, no FP8 DPA.
- FA3: `sm90`, FP16/BF16 and selected FP8 inference paths.
- FA4: `sm80+`, FP16/BF16, but no CP or KV cache in the current selector.
- FlashAttention does not support arbitrary masks.
- FlashAttention CP does not support bottom-right causal masks or causal
  cross-attention.
- FlashAttention with explicit pre/post-scale bias is removed. FA2 can support
  selected ALiBi cases.

### No CP

- F16/BF16 dense self-attention: Supported by FA2/FA3/FA4 when installed and
  shape gates pass.
- THD:
  - without padding between packed sequences: Limited.
  - with `pad_between_seqs=True`: FA2 and FA4 are removed; FA3 can remain.
- MLA:
  - FA2: No.
  - FA3 / FA4: Limited to selected head shapes.
- SWA: Limited. FA2 requires a sufficiently new package; FA3/FA4 have narrower
  dropout/bias behavior.
- KV cache:
  - FA2: Limited.
  - FA3: Limited.
  - FA4: No.
- Dropout:
  - FA2: Supported.
  - FA3 / FA4: No when dropout is nonzero.

### CP

- FA4: No.
- FA2 / FA3: Limited.
- F16/BF16 `bshd`/`sbhd`: Limited by CP mask and bias constraints.
- F16/BF16 `thd`: Limited; no THD attention bias.
- FP8 DPA: No for CP.
- Bottom-right causal masks: No.
- Causal cross-attention: No.
- Explicit bias: No except selected post-scale-bias CP forms in the high-level
  selector constraints; THD bias is removed.
- Sink softmax: No for FlashAttention.

## Unfused Attention

- No CP: Supported for FP32, FP16, and BF16 fallback paths.
- CP: No.
- Arbitrary mask: Supported for non-FP8 attention.
- FP8 DPA: Limited to emulation or ONNX export mode.
- KV cache: Supported for non-FP8 paths.
- `return_max_logit`: Supported for non-FP8 paths.

## When To Check The Code

Use this outline to decide whether a backend family is plausible. Check the
runtime selector for:

- exact cuDNN version or known-bad cuDNN versions,
- exact FlashAttention package version or installed FlashAttention family,
- Blackwell, `sm120`, or architecture-specific workarounds,
- head dimensions above 128, MLA, or mixed Q/K/V dimensions,
- paged KV cache, THD padding-between-sequences, or split Q/KV layouts,
- deterministic training with bias, FP8, CUDA graphs, or non-vanilla softmax.
