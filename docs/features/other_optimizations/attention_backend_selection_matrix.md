# Attention Backend Selection Matrix Draft

This Markdown file is for rough iteration and is not linked from the Sphinx
toctree. It explores a decision-table presentation for PyTorch
`DotProductAttention` backend support.

Source target: `get_attention_backend()` and `nvte_get_fused_attn_backend()` on
GitHub `main` commit `815bf369`, audited on 2026-06-04.

## What Changed in This Iteration

This iteration applies the layout recommendation from
`attention_backend_selection_layout_experiment.md`:

1. The main routing table uses configuration axes as columns.
2. Backend families appear as candidate values in one cell.
3. Each row points to the resolver tables that still need to be applied.
4. Backend-family columns are reserved for smaller resolver tables where
   side-by-side comparison is useful.

The goal is to make the matrix read like a route:

`hardware -> precision -> execution -> layout/pattern -> candidates -> resolver`

## How to Read This Draft

The routing table answers: which backend families are still plausible after
matching the largest configuration axes?

Candidate backends are not final selections. A candidate means the routing row
has not ruled out that backend family. After matching a row:

1. Apply global overrides.
2. Apply the row's follow-up resolver tables.
3. Apply cross-cutting modifiers for bias, dropout, softmax, determinism, head
   geometry, `num_splits`, and `return_max_logit`.
4. Apply final backend priority.

Final priority still matters:

- On Hopper and newer GPUs, if FlashAttention and FusedAttention both survive,
  TE prefers FusedAttention.
- Otherwise, TE prefers FlashAttention, then FusedAttention, then unfused.
- `NVTE_FLASH_ATTN=0`, `NVTE_FUSED_ATTN=0`, and `NVTE_UNFUSED_ATTN=0` remove
  those backend families before the rest of the checks.

Backend names used below:

| Name | Meaning |
| --- | --- |
| FA2 | FlashAttention 2 |
| FA3 | FlashAttention 3 |
| FA4 | FlashAttention 4 |
| Fused F16 | cuDNN FusedAttention `NVTE_F16_arbitrary_seqlen` for FP16/BF16 |
| Fused FP8 | cuDNN FusedAttention `NVTE_FP8` for FP8 DPA |
| Unfused | `UnfusedDotProductAttention` |

Resolver tables:

| Resolver | Use it when |
| --- | --- |
| [FlashAttention support resolver](#flashattention-support-resolver) | A routing row leaves FA2, FA3, or FA4 as candidates. |
| [Fused F16 cuDNN support resolver](#fused-f16-cudnn-support-resolver) | A routing row leaves Fused F16 as a candidate. |
| [Fused FP8 cuDNN support resolver](#fused-fp8-cudnn-support-resolver) | A routing row leaves Fused FP8 as a candidate. |
| [Cross-cutting modifier tables](#cross-cutting-modifier-tables) | Bias, ALiBi, dropout, non-vanilla softmax, determinism, head-geometry, `num_splits`, or `return_max_logit` modifiers are present. |

## Global Overrides

Apply these before using a routing row.

| Override / modifier | Effect |
| --- | --- |
| `NVTE_FLASH_ATTN=0` | Removes FA2, FA3, and FA4 candidates. |
| `NVTE_FUSED_ATTN=0` | Removes Fused F16 and Fused FP8 candidates. |
| `NVTE_UNFUSED_ATTN=0` | Removes the Unfused candidate. |
| Missing FlashAttention package | Removes the matching FA candidate; FA2 also needs a supported version range. |
| `attn_mask_type="arbitrary"` | Removes FA2, FA3, FA4, Fused F16, and Fused FP8; Unfused remains for non-FP8. |
| FP32 QKV | Removes FA2, FA3, FA4, Fused F16, and Fused FP8; Unfused remains. |
| `return_max_logit=True` | Removes FlashAttention and Fused FP8; Fused F16 and Unfused remain only if other gates pass. |
| `num_splits != 1` | Removes FA2, Fused F16, Fused FP8, and Unfused; FA3 remains, and FA4 remains only for SplitKV on `sm100+`. |
| KV cache plus context parallelism | Removes every backend family. |
| FP8 on `sm120` | Removes FlashAttention and Fused FP8; only FP8 emulation or ONNX-export Unfused paths can remain. |

## Main Routing Table

Rows are intentionally broad. The `Candidate backends` cell says what survives
this routing row before resolver tables and modifiers are applied.

| Hardware | Precision / recipe | Execution | Layout / pattern | Head geometry | Candidate backends | Excluded here | Follow-up |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `sm < 80` | FP16/BF16 or FP32 | No CP, no KV cache | Any supported non-FP8 path | Any | Unfused | FA2, FA3, FA4, Fused F16, Fused FP8 | Modifiers |
| `sm < 80` | Any | Context parallelism | Any | Any | None | All backend families | CP requires accelerated backends that are not available on `sm < 80` |
| `sm < 80` | FP8 DPA | Any | Any | Any | Unfused only for FP8 emulation or ONNX export | FA2, FA3, FA4, Fused F16, Fused FP8 | FP8 recipe notes |
| `sm80` / `sm89` | FP16/BF16 | No CP, no KV cache | BSHD/SBHD self-attention with `no_mask`, `padding`, `causal`, or `padding_causal`; no bias | MHA/GQA/MQA, dimensions normally multiple of 8 | FA2; FA4 if installed; Fused F16; Unfused | FA3; Fused FP8 | Flash resolver; Fused F16 resolver; modifiers |
| `sm80` / `sm89` | FP16/BF16 | No CP, no KV cache | THD padding-style masks without padding between packed sequences | MHA/GQA/MQA | FA2; FA4 if installed; Fused F16; Unfused | FA3; Fused FP8 | Flash THD row; Fused F16 THD rows; modifiers |
| `sm80` / `sm89` | FP16/BF16 | No CP, no KV cache | THD with padding between packed sequences | MHA/GQA/MQA | Fused F16 if cuDNN accepts THD; Unfused unless KV cache is present | FA2; FA3; FA4; Fused FP8 | Fused F16 THD rows; modifiers |
| `sm80` / `sm89` | FP16/BF16 | No CP, no KV cache | Top-left causal cross-attention with unequal Q/KV sequence lengths | Any supported non-MLA or MLA shape | Fused F16; Unfused | FA2; FA3; FA4; Fused FP8 | Fused F16 mask rows; modifiers |
| `sm80` / `sm89` | FP16/BF16 | No CP, no KV cache | Bottom-right causal cross-attention or sliding window attention | Any supported non-MLA or MLA shape | FA2; FA4 if installed; Fused F16; Unfused | FA3; Fused FP8 | Flash mask/SWA rows; Fused F16 mask/SWA rows; modifiers |
| `sm80` / `sm89` | FP16/BF16 | Context parallelism | BSHD/SBHD, no bias, no bottom-right causal mask, not causal cross-attention | MHA/GQA/MQA | FA2; Fused F16 | FA3; FA4; Fused FP8; Unfused | Flash CP row; Fused F16 CP gates; modifiers |
| `sm80` / `sm89` | FP16/BF16 | Context parallelism | THD with `cp_comm_type="all_gather"` or `"a2a+p2p"` | MHA/GQA/MQA | FA2 | FA3; FA4; Fused F16 on `815bf369`; Fused FP8; Unfused | Flash CP row; THD CP notes |
| `sm80` / `sm89` | FP16/BF16 | KV cache | Paged or non-paged decode path | MHA/GQA/MQA | FA2; Fused F16 on `sm80`; Unfused | FA3; FA4; Fused F16 on `sm89`; Fused FP8 | Flash KV row; Fused F16 KV rows; modifiers |
| `sm80` / `sm89` | FP8 DPA | Any | Any | Any | Unfused only for FP8 emulation or ONNX export | FA2, FA3, FA4, Fused F16, Fused FP8 | FP8 recipe notes |
| `sm90` | FP16/BF16 | No CP, no KV cache | BSHD/SBHD self-attention with `no_mask`, `padding`, `causal`, or `padding_causal`; no bias | MHA/GQA/MQA, dimensions normally multiple of 8 | FA2; FA3 if installed; FA4 if installed and FA3 absent; Fused F16; Unfused | Fused FP8 | Flash resolver; Fused F16 resolver; Hopper priority |
| `sm90` | FP16/BF16 | No CP, no KV cache | THD padding-style masks without padding between packed sequences | MHA/GQA/MQA | FA2; FA3 if installed; FA4 if installed and FA3 absent; Fused F16; Unfused | Fused FP8 | Flash THD row; Fused F16 THD rows; Hopper priority |
| `sm90` | FP16/BF16 | No CP, no KV cache | THD with padding between packed sequences | MHA/GQA/MQA | FA3 if installed; Fused F16 if cuDNN accepts THD; Unfused unless KV cache is present | FA2; FA4; Fused FP8 | Flash THD row; Fused F16 THD rows |
| `sm90` | FP16/BF16 | No CP, no KV cache | Top-left causal cross-attention with unequal Q/KV sequence lengths | Any supported non-MLA or MLA shape | Fused F16; Unfused | FA2; FA3; FA4; Fused FP8 | Fused F16 mask rows; modifiers |
| `sm90` | FP16/BF16 | No CP, no KV cache | Bottom-right causal cross-attention or sliding window attention | Any supported non-MLA or MLA shape | FA2; FA3 if installed; FA4 if installed and FA3 absent; Fused F16; Unfused | Fused FP8 | Flash mask/SWA rows; Fused F16 mask/SWA rows; Hopper priority |
| `sm90` | FP16/BF16 | No CP, no KV cache | MLA with unequal QK/V dimensions | Separate Q/K/V layout for Fused F16; selected MLA shapes for FA3/FA4 | FA3 if installed; FA4 if installed and FA3 absent; Fused F16; Unfused | FA2; Fused FP8 unless FP8 recipe is active | Flash head-dim rows; Fused F16 head/layout rows |
| `sm90` | FP16/BF16 | Context parallelism | BSHD/SBHD, no bias, no bottom-right causal mask, not causal cross-attention | MHA/GQA/MQA | FA2; FA3 if installed; Fused F16 | FA4; Fused FP8; Unfused | Flash CP row; Fused F16 CP gates; modifiers |
| `sm90` | FP16/BF16 | Context parallelism | THD with `cp_comm_type="all_gather"` or `"a2a+p2p"` | MHA/GQA/MQA | FA2; FA3 if installed | FA4; Fused F16 on `815bf369`; Fused FP8; Unfused | Flash CP row; THD CP notes |
| `sm90` | FP16/BF16 | KV cache | Paged or non-paged decode path | MHA/GQA/MQA | FA2; FA3 if installed; Fused F16; Unfused | FA4; Fused FP8 for FP8 KV cache | Flash KV row; Fused F16 KV rows; modifiers |
| `sm90` | FP8 delayed scaling | No CP, no KV cache | BSHD/SBHD, no bias, allowed mask | Dimensions accepted by cuDNN FP8 branch, commonly `head_dim_qk=head_dim_v=128` | Fused FP8; Unfused only for FP8 emulation or ONNX export | FA2; FA4; Fused F16; FA3 for training | Fused FP8 resolver |
| `sm90` | FP8 delayed scaling | Context parallelism | Non-THD, no bias | Dimensions accepted by cuDNN FP8 branch | Fused FP8 if CP and cuDNN gates pass | FA2; FA3; FA4; Fused F16; Unfused | Fused FP8 CP row |
| `sm90` | FP8 current scaling or MXFP8 | Any | Any | Any | Unfused only for FP8 emulation or ONNX export | FA2, FA3, FA4, Fused F16, Fused FP8 | Current scaling and MXFP8 require `sm100+` |
| `sm100+` except `sm120` | FP16/BF16 | No CP, no KV cache | BSHD/SBHD or THD, common self-attention masks, no bias | MHA/GQA/MQA | FA2 with Blackwell version gate; FA4 if installed; Fused F16; Unfused | FA3; Fused FP8 | Flash resolver; Fused F16 resolver; Hopper+ priority |
| `sm100+` except `sm120` | FP16/BF16 | No CP, no KV cache | Top-left causal cross-attention with unequal Q/KV sequence lengths | Any supported non-MLA or MLA shape | Fused F16; Unfused | FA2; FA3; FA4; Fused FP8 | Fused F16 mask rows; modifiers |
| `sm100+` except `sm120` | FP16/BF16 | No CP, no KV cache | Bottom-right causal cross-attention or sliding window attention | Any supported non-MLA or MLA shape | FA2; FA4 if installed; Fused F16; Unfused | FA3; Fused FP8 | Flash mask/SWA rows; Fused F16 mask/SWA rows; Hopper+ priority |
| `sm100+` except `sm120` | FP16/BF16 | No CP, no KV cache | MLA | Selected FA4 shapes or separate Q/K/V layout for Fused F16 | FA4 if installed; Fused F16; Unfused | FA2; FA3; Fused FP8 unless FP8 recipe is active | Flash head-dim rows; Fused F16 head/layout rows |
| `sm100+` except `sm120` | FP16/BF16 | Context parallelism | No bottom-right causal mask, not causal cross-attention, no unsupported bias | MHA/GQA/MQA | FA2; Fused F16 | FA3; FA4; Fused FP8; Unfused | Flash CP row; Fused F16 CP gates |
| `sm100+` except `sm120` | FP16/BF16 | KV cache | Paged or non-paged decode path | MHA/GQA/MQA | FA2; Fused F16; Unfused | FA3; FA4; Fused FP8 for FP8 KV cache | Flash KV row; Fused F16 KV rows |
| `sm100+` except `sm120` | FP8 delayed scaling | No CP, no KV cache | Non-THD unless cuDNN FP8 gates say otherwise, no bias | Dimensions accepted by cuDNN FP8 branch | Fused FP8; Unfused only for FP8 emulation or ONNX export | FA2; FA3; FA4; Fused F16 | Fused FP8 resolver |
| `sm100+` except `sm120` | FP8 current scaling | No CP, no KV cache | BSHD/SBHD/BHSD, no bias, allowed mask | Dimensions accepted by cuDNN FP8 current-scaling branch | Fused FP8; Unfused only for FP8 emulation or ONNX export | FA2; FA3; FA4; Fused F16 | Fused FP8 current-scaling row |
| `sm100+` except `sm120` | MXFP8 | No CP, no KV cache | Non-THD, no bias | Dimensions accepted by cuDNN FP8 branch | Fused FP8; Unfused only for FP8 emulation or ONNX export | FA2; FA3; FA4; Fused F16 | Fused FP8 MXFP8 row |
| `sm100+` except `sm120` | FP8 block scaling or NVFP4 | Any | Any | Any | Unfused only for FP8 emulation or ONNX export | FA2, FA3, FA4, Fused F16, Fused FP8 | FP8 recipe notes |
| `sm120` | FP16/BF16 | No CP, no KV cache | Common non-FP8 attention paths | MHA/GQA/MQA and supported dimensions | FA2; FA4 if installed; Fused F16; Unfused | FA3; Fused FP8 | Flash resolver; Fused F16 `sm120` row; modifiers |
| `sm120` | FP16/BF16 | KV cache | Decode path | MHA/GQA/MQA | Unfused | FA2; FA3; FA4; Fused F16; Fused FP8 | KV-cache notes; modifiers |
| `sm120` | FP16/BF16 | Context parallelism | No bottom-right causal mask, not causal cross-attention | MHA/GQA/MQA | FA2; Fused F16 if CP and `sm120` gates pass | FA3; FA4; Fused FP8; Unfused | Flash CP row; Fused F16 CP and `sm120` rows |
| `sm120` | FP8 DPA | Any | Any | Any | Unfused only for FP8 emulation or ONNX export | FA2, FA3, FA4, Fused F16, Fused FP8 | FP8 on `sm120` is disabled in Python selector |

## Backend Resolver Tables

<a id="flashattention-support-resolver"></a>

### FlashAttention Support Resolver

Use this table after a routing row leaves FA2, FA3, or FA4 as a candidate.
Backend columns are useful here because readers often compare Flash variants
directly.

| Gate | FA2 | FA3 | FA4 |
| --- | --- | --- | --- |
| Installation and architecture | Requires supported `flash-attn` 2 and `sm80+`; Blackwell requires `flash-attn >= 2.7.3`. | Requires FA3 installed and `sm90`. | Requires FA4 installed and `sm80+`; on `sm90`, TE disables FA4 when FA3 is installed. |
| Dtype | FP16/BF16. FP8 DPA is disabled. | FP16/BF16; selected FP8 inference paths exist, but FP8 training and FP8 CP are disabled. | FP16/BF16. FP8 DPA is disabled. |
| THD with padding between packed sequences | No. | Yes. | No. |
| KV cache | Supported with FA2 gates: paged requires `flash-attn >= 2.5` and page size multiple of 256; non-paged requires effective `max_seqlen_kv` divisible by 256. Disabled on `sm120`. | Supported; FP8 KV cache only for THD Q format. | Disabled for KV cache. |
| Context parallelism | Candidate only for non-FP8, no bottom-right causal mask, no causal cross-attention, limited bias, and no THD bias. | Same family restrictions as FA2. | Disabled for context parallelism. |
| Masks | `arbitrary` disables FA2. Top-left causal cross-attention with unequal sequence lengths disables FA2. Bottom-right causal masks are supported. | Same mask-family restrictions as FA2. | Same mask-family restrictions as FA2, plus FA4-specific validator/workaround gates. |
| Bias and ALiBi | No explicit bias except ALiBi. ALiBi requires `flash-attn >= 2.4` and bottom-right alignment for cross-attention. | ALiBi and explicit bias are disabled. | ALiBi and explicit bias are disabled. |
| Dropout | Supported. | Nonzero dropout disables FA3. | Nonzero dropout disables FA4. |
| Head dimensions | Requires `head_dim_qk == head_dim_v`, divisible by 8, `<= 256`; `> 192` is restricted to selected architectures. MLA is disabled. | Requires `head_dim_qk <= 256`, `num_heads % num_gqa_groups == 0`, and selected MLA shapes. | Defers first-pass validation to FA4. TE also disables selected SM100 `head_dim=256` cross-attention and selected SM100 training MLA shapes. |
| Determinism | Requires `flash-attn >= 2.4.1`. | Deterministic training with `max(head_dim_qk, head_dim_v) >= 256` is disabled. | Candidate if FA4 accepts the shape. |

<a id="fused-f16-cudnn-support-resolver"></a>

### Fused F16 cuDNN Support Resolver

Use this table after a routing row leaves Fused F16 as a candidate. All
applicable rows must pass for cuDNN to return `NVTE_F16_arbitrary_seqlen`.
The rows below are scoped gates: match the input branch, apply the minimum
cuDNN version and constraints, then continue to the next gate. This narrows a
Fused F16 candidate instead of only listing what each cuDNN version introduced.

#### F16 Always-On Gates

| Axis | Requirement before version rows matter |
| --- | --- |
| Dtype | Q and KV dtype must be FP16 or BF16. |
| Head dimensions | `head_dim_qk` and `head_dim_v` must be multiples of 8. Dimensions `<= 128` are the base case; wider dimensions need version rows below. |
| Base layouts | BSHD, SBHD, and BHSD are base candidates. THD, split Q/KV groups, and paged KV need version rows below. |
| Ragged offsets | THD cases that require 64-bit ragged offsets need cuDNN 9.5+. |
| Bad cuDNN versions | cuDNN `< 8.9.0`, 9.10.0, and 9.10.1 return no F16 fused backend. |
| Python pre-filters | `pre_scale_bias`, unsupported CP combinations, `num_splits != 1`, and FP32 QKV are removed before the cuDNN call. |

#### F16 Platform Gate

| Input branch | Minimum cuDNN | Narrowed scope if gate passes | Reject if |
| --- | --- | --- | --- |
| `sm < 80` | N/A | None | F16 fused requires `sm80+`. |
| `sm80` or `sm90` | 8.9.0 | Continue to sequence, head, layout/mask, and modifier gates. | cuDNN `< 8.9.0` or known-bad 9.10.0/9.10.1. |
| `sm89` or other `80 <= sm < 100` | 8.9.3 | Continue to sequence, head, layout/mask, and modifier gates. | cuDNN `< 8.9.3` or known-bad 9.10.0/9.10.1. |
| `sm100+` except `sm120` | 9.7.0 | Continue to Blackwell-specific shape and determinism gates. | cuDNN `< 9.7.0` or known-bad 9.10.0/9.10.1. |
| `sm120` | 9.18.1 | Continue only if non-deterministic training or inference and layout is not `t3hd`/`th3d`. | cuDNN `< 9.18.1`, deterministic training, or `t3hd`/`th3d`. |

#### F16 Sequence and Head Gate

| Input branch | Minimum cuDNN | Narrowed scope if gate passes | Reject if |
| --- | --- | --- | --- |
| General sequence lengths before cuDNN 9.0 | 8.9.0 | `max_seqlen_q` and `max_seqlen_kv` are both divisible by 64. | cuDNN `< 9.0.0` and either max sequence length is not divisible by 64. |
| General sequence lengths on cuDNN 9.0+ | 9.0.0 | General divisibility gate is removed. | Bottom-right mask-specific rows below may still require divisibility before 9.7. |
| MHA, `num_heads == num_gqa_groups` | 8.9.0 | Continue to layout/mask and head-dimension gates. | Other gates fail. |
| GQA/MQA, `num_heads != num_gqa_groups` | 8.9.7 | Continue to layout/mask and head-dimension gates. | cuDNN `< 8.9.7`. |
| Head dims multiple of 8 and both `<= 128` | 8.9.0 | Base head-dimension scope passes. | Either dimension is not a multiple of 8. |
| Hopper fprop, head dims `<= 256` | 9.1.0 | Wider-dimension Hopper inference/fprop passes. | Training path with cuDNN `< 9.5.0`. |
| Hopper bprop, head dims `<= 256` | 9.5.0 | Wider-dimension Hopper training passes unless MLA exclusion applies. | cuDNN `< 9.5.0`. |
| Blackwell fprop, wider dims, non-paged `sq > 1` | 9.9.0 | Blackwell fprop wider-dim path passes. | Paged or `sq=1` branch before 9.10.2. |
| Any arch fprop, wider dims, paged or selected `sq=1`/`sq>1` cases | 9.10.2 | Broad fprop wider-dim path passes. | cuDNN 9.10.0/9.10.1 or branch outside the paged/`sq` conditions. |
| Blackwell bprop `(head_dim_qk, head_dim_v) = (192, 128)` | 9.11.0 | Blackwell training special case passes. | Other wider Blackwell bprop shape without support. |
| Hopper training MLA with both dims `>=128`, unequal dims, not `(192, 128)` | N/A | None | Explicit cuDNN 9.11+ exclusion. |

#### F16 Layout and Mask Gate

| Input branch | Minimum cuDNN | Narrowed scope if gate passes | Reject if |
| --- | --- | --- | --- |
| BSHD/SBHD with `causal` | 8.9.0 | Base causal mask path passes. | Other gates fail. |
| BSHD/SBHD with `no_mask`, `padding`, or `padding_causal` | 8.9.6 | Common dense layout masks pass. | `post_scale_bias` with `padding` or `padding_causal`. |
| THD with `padding` or `padding_causal`, MHA | 9.1.0 | THD padding-style MHA path passes. | `sm < 90`, non-MHA before 9.6, or unsupported ragged offset. |
| THD with `padding` or `padding_causal`, GQA/MQA | 9.6.0 | THD padding-style GQA/MQA path passes. | `sm < 90` or unsupported ragged offset. |
| Split Q/KV layout combinations | 9.7.0 | Split layout group can continue through later gates. | cuDNN `< 9.7.0`. |
| Paged KV layout with `padding` or `padding_causal` | 9.5.0 | Paged KV padding-style path passes. | cuDNN `< 9.5.0`. |
| `causal_bottom_right`, BSHD/SBHD | 9.3.0 | Bottom-right causal passes with `sq <= skv`, 64-divisible lengths, no bias, zero dropout. | cuDNN `< 9.7.0` and bottom-right length constraints fail. |
| `padding_causal_bottom_right`, BSHD/SBHD/THD | 9.6.0 | Padding bottom-right path passes with `sq <= skv`, no bias, zero dropout. | cuDNN `< 9.7.0` and bottom-right length constraints fail. |
| Bottom-right masks on split, paged, or broader layout groups | 9.7.0 | Bottom-right path passes with `sq <= skv`; 64-divisibility relaxed. | Bias/dropout constraints fail. |
| `arbitrary` mask | N/A | None | cuDNN F16 fused backend does not support arbitrary masks. |

#### F16 Modifier Gate

| Input branch | Minimum cuDNN | Narrowed scope if gate passes | Reject if |
| --- | --- | --- | --- |
| `no_bias` | 8.9.0 | Bias gate passes. | Other gates fail. |
| ALiBi | 8.9.6 | Converts to non-trainable post-scale bias on `sm90+` for selected non-padding masks. | `sm < 90` or padding-style excluded masks. |
| `post_scale_bias` on `sm90+` | 8.9.6 | Post-scale bias can pass if mask is not excluded. | `padding` or `padding_causal`; Python `111s` trainable dbias exclusion. |
| `post_scale_bias` on `sm80+` | 9.0.0 | Post-scale bias can pass if mask is not excluded. | `padding` or `padding_causal`; Python `111s` trainable dbias exclusion. |
| Sliding window, top-left BSHD/SBHD | 9.2.0 | SWA passes with no bias, zero dropout, `sq <= skv`. | THD/broader masks before 9.6. |
| Sliding window, BSHD/SBHD/THD broader masks | 9.6.0 | SWA passes with no bias, zero dropout, `sq <= skv`. | cuDNN 9.14.0 with non-causal SWA and `max_seqlen_kv > 1024`. |
| Non-vanilla softmax | 9.13.1 | cuDNN supports off-by-one and learnable softmax. | THD with cuDNN `< 9.18.0` due to Python pre-filter. |
| CUDA graph training, BSHD/SBHD, non-padding mask, `max_seqlen_kv % 128 != 0` | 9.15.1 | Graph-captured training path passes. | cuDNN `<= 9.15.0`. |
| Deterministic F16 training | 8.9.5 plus `sm90+` in Python | Passes only with no trainable attention bias; Blackwell deterministic training requires 9.18.1+, zero dropout, no bias. | `sm120` deterministic training; trainable bias gradients; older cuDNN. |

<a id="fused-fp8-cudnn-support-resolver"></a>

### Fused FP8 cuDNN Support Resolver

Use this table after a routing row leaves Fused FP8 as a candidate. Python
recipe filters run before the cuDNN FP8 sub-backend check. Unlike F16, FP8 is
best represented as a small set of mutually narrowing recipe and cuDNN path
rows.

#### FP8 Always-On Gates

| Axis | Requirement before version rows matter |
| --- | --- |
| Dtype | Q and KV dtype passed to cuDNN must be FP8. |
| Architecture | `sm90+` is required, but Python disables FP8 attention on `sm120`. |
| Bias and outputs | Bias must be `no_bias`, `return_max_logit=False`, and 64-bit ragged offsets must not be required. |
| Bad cuDNN versions | cuDNN 9.10.0 is excluded for FP8 SDPA. |
| Python FP8 recipe filters | Block scaling and NVFP4 recipes remove FusedAttention before cuDNN. |
| Python feature filters | FP8 KV cache removes FusedAttention; FP8 CP removes FusedAttention for THD or any bias. |

#### FP8 Recipe Gate

| Input branch | Minimum cuDNN before cuDNN path | Narrowed scope if gate passes | Reject if |
| --- | --- | --- | --- |
| Delayed scaling, `fp8_dpa=True` | 9.2.1 | Continue to FP8 cuDNN path gate. | cuDNN `< 9.2.1`; deterministic `sm100+` with cuDNN `< 9.18.0`. |
| Current scaling | 9.14.0 | Continue only on `sm100+`. | `sm < 100`, cuDNN `< 9.14.0`, or deterministic current scaling with cuDNN `< 9.18.0`. |
| MXFP8 | 9.21.0 | Continue only on `sm100+`, `fp8_mha=False`, and non-THD QKV format. | `sm < 100`, cuDNN `< 9.21.0`, `fp8_mha=True`, or THD. |
| Block scaling or NVFP4 | N/A | None | Python removes FusedAttention before cuDNN. |

#### FP8 cuDNN Path Gate

| Input branch | Minimum cuDNN | Narrowed scope if gate passes | Reject if |
| --- | --- | --- | --- |
| Early `sm90` delayed-scaling path | 9.2.1 | BSHD/SBHD, vanilla softmax, `no_mask` or `causal`, both sequence lengths divisible by 128, `head_dim_qk=head_dim_v=128`. | Padding masks, BHSD, non-vanilla softmax, non-128 head dims, or non-divisible sequence lengths. |
| General FP8 path before 9.21 | 9.7.0 | BSHD/SBHD, vanilla softmax, `no_mask`, `causal`, `padding`, or `padding_causal`; head dims multiples of 16. On `sm90`, inference allows dims `<=256` and training requires both dims 128. On `sm100+`, dims must be `<=128`. | BHSD, non-vanilla softmax, bottom-right mask, unsupported dimensions. |
| Current-scaling path | 9.14.0 | Same cuDNN path as 9.7 row, but Python now allows current scaling on `sm100+`. | Same exclusions as 9.7 row plus current-scaling recipe gate failures. |
| Broad `sm100+` FP8 path | 9.21.0 | BSHD/SBHD/BHSD; vanilla, off-by-one, or learnable softmax; `no_mask`, `causal`, or `causal_bottom_right`; `head_dim_qk <= 192`, `head_dim_v <= 128`, dimensions multiple of 16. | THD, padding masks in the 9.21-only broad path, bias, `sm120`, unsupported dimensions. |
| FP8 sliding window | 9.21.0 | SWA can pass only on `sm100+` after the broad path and Python SWA gates. | `sm < 100`, cuDNN `< 9.21.0`, bias, or unsupported dropout/mask combination. |
| FP8 deterministic training | 9.19.0 | Deterministic training can pass after recipe and cuDNN path gates. | Training with deterministic mode and cuDNN `< 9.19.0`. |

## Cross-Cutting Modifier Tables

These modifiers should be applied after selecting the closest hardware,
precision, execution, and layout branch.

### Bias and ALiBi

| Bias branch | FA2 | FA3 | FA4 | Fused F16 | Fused FP8 | Unfused |
| --- | --- | --- | --- | --- | --- | --- |
| No actual bias tensor, `core_attention_bias_shape=None` | Candidate; apply [Flash resolver](#flashattention-support-resolver). | Candidate; apply [Flash resolver](#flashattention-support-resolver). | Candidate; apply [Flash resolver](#flashattention-support-resolver). | Candidate; apply [F16 resolver](#fused-f16-cudnn-support-resolver). | Candidate; apply [FP8 resolver](#fused-fp8-cudnn-support-resolver). | Yes. |
| ALiBi | Candidate; apply [Flash resolver](#flashattention-support-resolver) ALiBi row. | No. | No. | Candidate after conversion to non-trainable post-scale bias; apply [F16 resolver](#fused-f16-cudnn-support-resolver) bias rows. | No. | Yes. |
| `pre_scale_bias` | No. | No. | No. | No. | No. | Yes. |
| Non-trainable `post_scale_bias` | No. | No. | No. | Candidate; apply [F16 resolver](#fused-f16-cudnn-support-resolver) bias rows. | No. | Yes. |
| Trainable `post_scale_bias` | No. | No. | No. | Only supported dbias shapes. Trainable `111s` disables FusedAttention; deterministic bias grad disables FusedAttention. | No. | Yes. |

### Dropout, Softmax, Determinism

| Modifier | FA2 | FA3 | FA4 | Fused F16 | Fused FP8 | Unfused |
| --- | --- | --- | --- | --- | --- | --- |
| Nonzero dropout | Candidate; apply [Flash resolver](#flashattention-support-resolver). | No. | No. | Candidate; apply [F16 resolver](#fused-f16-cudnn-support-resolver) dropout/mask/SWA rows. | Candidate; apply [FP8 resolver](#fused-fp8-cudnn-support-resolver). | Yes. |
| `softmax_type != "vanilla"` | No. | No. | No. | Candidate; apply [F16 resolver](#fused-f16-cudnn-support-resolver) softmax row. | Candidate; apply [FP8 resolver](#fused-fp8-cudnn-support-resolver) softmax row. | Yes for non-FP8; no for FP8. |
| Deterministic mode, no bias, vanilla softmax | Candidate; apply [Flash resolver](#flashattention-support-resolver) determinism row. | Candidate; apply [Flash resolver](#flashattention-support-resolver) determinism row. | Candidate; apply [Flash resolver](#flashattention-support-resolver). | Candidate; apply [F16 resolver](#fused-f16-cudnn-support-resolver) determinism row. | Candidate; apply [FP8 resolver](#fused-fp8-cudnn-support-resolver) determinism rows. | Yes. |
| Deterministic trainable bias | No for explicit bias. | No for explicit bias. | No for explicit bias. | No when deterministic training needs bias gradients. | No. | Yes. |

### Head Geometry

| Geometry | FA2 | FA3 | FA4 | Fused F16 | Fused FP8 | Unfused |
| --- | --- | --- | --- | --- | --- | --- |
| MHA, `num_heads == num_gqa_groups` | Candidate; apply [Flash resolver](#flashattention-support-resolver). | Candidate; apply [Flash resolver](#flashattention-support-resolver). | Candidate; apply [Flash resolver](#flashattention-support-resolver). | Candidate; apply [F16 resolver](#fused-f16-cudnn-support-resolver) GQA/MQA row. | Candidate; apply [FP8 resolver](#fused-fp8-cudnn-support-resolver). | Yes. |
| GQA/MQA | Candidate; apply [Flash resolver](#flashattention-support-resolver). | Candidate; apply [Flash resolver](#flashattention-support-resolver). | Candidate; apply [Flash resolver](#flashattention-support-resolver). | Candidate; apply [F16 resolver](#fused-f16-cudnn-support-resolver) GQA/MQA row. | Candidate; apply [FP8 resolver](#fused-fp8-cudnn-support-resolver). | Yes. |
| MLA, `head_dim_qk != head_dim_v` | No. | Candidate; apply [Flash resolver](#flashattention-support-resolver) head-dimension row. | Candidate; apply [Flash resolver](#flashattention-support-resolver) head-dimension row. | Candidate only with separate Q/K/V layout; apply [F16 resolver](#fused-f16-cudnn-support-resolver) head-dimension rows. | Candidate only with no bias; apply [FP8 resolver](#fused-fp8-cudnn-support-resolver). | Yes. |
| Head dimensions multiple of 8 and `<= 128` | Candidate; apply [Flash resolver](#flashattention-support-resolver). | Candidate; apply [Flash resolver](#flashattention-support-resolver). | Candidate; apply [Flash resolver](#flashattention-support-resolver). | Candidate; apply [F16 resolver](#fused-f16-cudnn-support-resolver) head-dimension rows. | Candidate; apply [FP8 resolver](#fused-fp8-cudnn-support-resolver) head-dimension rows. | Yes. |
| Wider head dimensions up to 256 | Candidate; apply [Flash resolver](#flashattention-support-resolver) head-dimension row. | Candidate; apply [Flash resolver](#flashattention-support-resolver) head-dimension row. | Candidate; apply [Flash resolver](#flashattention-support-resolver) head-dimension row. | Candidate; apply [F16 resolver](#fused-f16-cudnn-support-resolver) head-dimension rows. | Candidate; apply [FP8 resolver](#fused-fp8-cudnn-support-resolver) head-dimension rows. | Yes. |
| `head_dim_qk > 256` | No. | No. | Candidate only if FA4 validates it; apply [Flash resolver](#flashattention-support-resolver). | Candidate only for limited cuDNN branches; apply [F16 resolver](#fused-f16-cudnn-support-resolver). | Candidate only for limited cuDNN FP8 branches; apply [FP8 resolver](#fused-fp8-cudnn-support-resolver). | Yes. |

## Probe Results Used While Drafting

These selector-only checks were run on H100 (`sm90`), CUDA 13.2, cuDNN 9.21.0,
FlashAttention 2.7.4.post1 installed, and no FA3 or FA4 installed:

| Configuration | Selected backend | Candidate availability before priority |
| --- | --- | --- |
| BF16 causal BSHD MHA, no bias | Fused `NVTE_F16_arbitrary_seqlen` | `[1, 1, 1]` |
| Same with `NVTE_FUSED_ATTN=0` | FlashAttention 2 | `[1, 0, 1]` |
| GQA, 16 query heads and 4 K/V groups | Fused `NVTE_F16_arbitrary_seqlen` | `[1, 1, 1]` |
| MQA, 16 query heads and 1 K/V group | Fused `NVTE_F16_arbitrary_seqlen` | `[1, 1, 1]` |
| MLA with `head_dim_qk=64`, `head_dim_v=128` | Fused `NVTE_F16_arbitrary_seqlen` | `[False, 1, 1]` |
| `arbitrary` mask | Unfused | `[False, False, 1]` |
| Deterministic trainable `post_scale_bias` | Unfused | `[False, False, 1]` |
| THD context parallelism with `cp_comm_type="all_gather"` | FlashAttention 2 | `[1, False, False]` |
| Top-left causal cross-attention | Fused `NVTE_F16_arbitrary_seqlen` | `[False, 1, 1]` |
| `causal_bottom_right` cross-attention | Fused `NVTE_F16_arbitrary_seqlen` | `[1, 1, 1]` |
| FP8 delayed scaling, `head_dim_qk=head_dim_v=128` | Fused `NVTE_FP8` | `[False, 1, False]` |
| FP8 current scaling, `head_dim_qk=head_dim_v=128` on `sm90` | No backend | `[False, False, False]` |
