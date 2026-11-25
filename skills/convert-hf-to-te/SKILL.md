---
name: convert-hf-to-te
description: >-
  Adapt a Hugging Face PyTorch model for training with compatible Transformer Engine modules.
  Replace a decoder layer, sublayer or individual operation and map its checkpoint weights.
  Choose fusions that preserve the requested workload's behavior.
license: Apache-2.0
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
compatibility: >-
  Writing the implementation needs no GPU. Running it needs transformer_engine[pytorch],
  transformers and an NVIDIA GPU. FP8 support depends on the recipe, GPU and backend.
  The training script may have other dependencies (the Llama tutorial also needs accelerate,
  datasets and peft). Newer models may need more, such as flash-linear-attention and
  causal-conv1d for Qwen3.5.
metadata:
  version: "0.4.0"
  author: NVIDIA Transformer Engine contributors
  domain: model-training
  tags:
    - transformer-engine
    - hugging-face
    - pytorch
    - fine-tuning
    - mixture-of-experts
---

# Convert a Hugging Face model to Transformer Engine

Use the project's requested entry point or add a model-family module such as `te_llama.py`.
Replace compatible building blocks of the Hugging Face (HF) model with Transformer Engine (TE)
modules and load the original checkpoint into them. The building block is the whole transformer
layer, as for Llama and Gemma, or else the attention
(MHA, linear attention) and MLP or MoE blocks inside it. A partial conversion can also replace
a single operation inside an HF block. For a review or plan, stop after the requested feasibility,
boundary or weight-mapping decision in Steps 1–3. For a local implementation, Step 4 ends at the
first BF16 forward comparison. Use Step 5 for requested training checks, integration or performance
measurements at the scope the user specified.
In this skill `te` means `transformer_engine.pytorch`.

In a plan or implementation handoff, lead with the chosen replacement boundary and weight map.

Not for JAX or inference serving (vLLM, TensorRT-LLM). For Megatron-LM or another PyTorch
framework, the TE implementation can serve as a reference for which TE modules to use and how
the weights map.

## 1. Check the architecture and feasibility

Identify the training path, representative inputs and target precision from the request and project.
Read the model's `config.json`, checkpoint keys and decoder `forward` at the requested revision
or installed version. For each distinct layer type, record its modules, which tensors they consume
or share, and their checkpoint keys. Use this inventory to identify candidate TE modules.
Consult examples for the relevant boundary and verify API choices against the target TE release.
For source-only work, use those sources to resolve API and layout questions before setting up a
runtime.

| HF component | Candidate TE module |
|---|---|
| The whole transformer block | `te.TransformerLayer` |
| Self-attention with QKV and output projections | `te.MultiheadAttention`. Set `input_layernorm=True` only when it replaces HF's input norm, so the norm runs once. |
| Scaled dot-product attention on projected Q/K/V | `te.DotProductAttention` |
| Input norm followed by one linear projection, which can pack compatible projections of the same normalized input | `te.LayerNormLinear` |
| A norm and a dense MLP | `te.LayerNormMLP` |
| Grouped linear projections across MoE experts | `te.ops.GroupedLinear`. The surrounding [MoE block](#2-choose-the-granularity) supplies activation and routing. |
| Dispatch and combine across expert-parallel ranks | `te.ops.MoeDispatch` and `te.ops.MoeCombine` in the experimental operation fuser. Check compatibility with the router outputs and backend. |
| A single projection or norm | `te.Linear`, `te.RMSNorm`, `te.LayerNorm` |

The table is not exhaustive. Newer architectures may need other TE modules, such as the
experimental Gated DeltaNet core (`te.GatedDeltaNetAttention`). Check that the installed TE
exports the candidate and its backend supports the target inputs and workload. Keep unsupported
components in HF.

## 2. Choose the granularity

Start with the largest region that preserves HF's computation and the interface its caller uses.
The level describes which part of the HF model you replace.

**Level 1: the whole transformer block.** Try one `te.TransformerLayer` for the HF decoder block.
Match the attention and MLP widths, query and KV head counts, normalization and residual order,
activation, mask, and rotary positions for the requested inputs. A causal HF decoder with only
self-attention uses TE's self-attention topology with a causal mask. In TE 2.18, `layer_type="decoder"`
adds cross-attention. When the block matches, substitute it at the HF decoder entry point and preserve
the caller's arguments, layout and return value. Map the norm, attention and MLP weights and compare
the whole block's BF16 output to HF in Step 4.
[Llama wrapper](https://github.com/NVIDIA/TransformerEngine/blob/db1aff638b070cb9d08a0dc91629ff1457ead413/docs/examples/te_llama/te_llama.py#L40-L100) ·
[Gemma constructor](https://github.com/NVIDIA/TransformerEngine/blob/db1aff638b070cb9d08a0dc91629ff1457ead413/docs/examples/te_gemma/te_gemma.py#L143-L184) ·
[TE layer topology](https://github.com/NVIDIA/TransformerEngine/blob/27486e03cfc1fa41f6932dcecdc47c71c47eac3e/transformer_engine/pytorch/transformer.py#L475-L484)

**Level 2: attention and MLP separately.** If the whole layer does not fit, consider each
sublayer independently. Experts and linear attention are two examples from a non-exhaustive list.

- Attention: use `te.MultiheadAttention` when it implements the target attention block.
- MLP: `te.LayerNormMLP` for a dense MLP. For mixture-of-experts (MoE), compose a model-specific
  block around the compatible operations. For example,
  [`te_mixtral.py`](https://github.com/NVIDIA/TransformerEngine/blob/db1aff638b070cb9d08a0dc91629ff1457ead413/docs/examples/te_mixtral/te_mixtral.py)
  defines its own `TEMixtralSparseMoeBlock` to route tokens, run the expert FFN and combine outputs.

**Level 3: an operation inside an HF sublayer.** For example, replace a projection inside HF
attention or the expert GEMMs inside an HF MoE block. Choose this boundary when you can preserve
its inputs, outputs and shared state and have a workload benefit to test. If TE cannot reproduce
the attention behavior, keep HF's attention computation, masking, positions and cache together.
An independent MLP can still benefit. Keep HF wherever correctness or measured performance favors it.

### Choose what to fuse

Choose fusions before mapping weights. Preserve operation order and all consumers of a shared
tensor. For example, TE can combine:

- a norm with the QKV projection: `te.LayerNormLinear`, or `te.MultiheadAttention` with
  `input_layernorm=True`;
- a norm with the MLP: `te.LayerNormMLP`.

Look up other fusions in the [PyTorch API](https://docs.nvidia.com/deeplearning/transformer-engine/api/pytorch.html)
and [operation fuser guide](https://docs.nvidia.com/deeplearning/transformer-engine/examples/op_fuser/op_fuser.html).
Check the installed release for support at the target shapes, dtype and hardware.

## 3. Map the weights

Use the Step 1 inventory and Step 2 boundaries to map each required HF key or slice to a TE
parameter and slice. Record axis, order and intended cast where the layout changes. Preserve tied
weights. For expert-parallel MoE, map each global expert ID to its owning rank and local expert
parameter. When loading quantized weights into BF16, use the checkpoint's quantization metadata
to recover the reference values.

Plan how the loader will track copies across all shards. Require it to fail on any missing
required source or destination slice. For packed weights, reconstruct the named HF tensors after loading and compare after the
intended cast to catch equal-shaped swaps. Three examples illustrate why the layout matters:

- [Llama gate and up](https://github.com/NVIDIA/TransformerEngine/blob/db1aff638b070cb9d08a0dc91629ff1457ead413/docs/examples/te_llama/te_llama.py#L208-L218) fill separate FC1 halves and can arrive in different shards.
- [Gemma QKV](https://github.com/NVIDIA/TransformerEngine/blob/db1aff638b070cb9d08a0dc91629ff1457ead413/docs/examples/te_gemma/te_gemma_loading_weights.py#L154-L187) uses different mappings for separate and interleaved projections.
- [Mixtral mappings](https://github.com/NVIDIA/TransformerEngine/blob/db1aff638b070cb9d08a0dc91629ff1457ead413/docs/examples/te_mixtral/hf_to_te_weights.py#L46-L104) pack Q heads by KV group and interleave expert gate/up rows for a fused kernel.

## 4. Implement

### Design patterns

- **Substitute at construction** when the replacement satisfies the HF caller. Keep the arguments,
  return values and state used by the workload so the surrounding HF model can call it directly.
  For example, Llama substitutes its decoder class while HF constructs the causal LM. [Llama constructor](https://github.com/NVIDIA/TransformerEngine/blob/db1aff638b070cb9d08a0dc91629ff1457ead413/docs/examples/te_llama/te_llama.py#L87-L100)
- **Compose around model-specific behavior** when only part of a block matches. Keep routing or
  attention state with the code that owns it and attach TE at the boundary chosen in Step 2.
  For example, Mixtral dispatches routed tokens to its expert FFN, then combines its outputs. [Mixtral MoE forward](https://github.com/NVIDIA/TransformerEngine/blob/db1aff638b070cb9d08a0dc91629ff1457ead413/docs/examples/te_mixtral/te_mixtral.py#L443-L465)
- **Load after construction** so the destination layout exists before applying the Step 3 mapping.
  Load retained HF parameters by their original mapping and verify the converted weights.
  For example, Gemma constructs the model before its standard loader maps the decoder weights
  and copies the retained embeddings. [Gemma loader](https://github.com/NVIDIA/TransformerEngine/blob/db1aff638b070cb9d08a0dc91629ff1457ead413/docs/examples/te_gemma/te_gemma_loading_weights.py#L49-L105)
- **Check each replacement's forward output first.** After copying weights, run the original HF
  computation and its TE replacement in BF16 on the same representative inputs. Match relevant
  masks, positions, state and stochastic behavior. Choose tolerances for the quantities being
  compared before running the checks. Compare outputs at the same boundary and in the same
  layout. For example, for a fused norm and QKV projection, run the HF norm and all three HF projections,
  then arrange their outputs in the TE head order. A replacement constructed inside the model
  can be checked directly using captured layer inputs.

### When replacement outputs differ

Choose checks that exercise the HF module being compared. These model-specific examples are
non-exhaustive.

- If attention outputs differ, compare the positions each path uses and which keys each query
  can see. Use an input that exercises the target behavior, such as a boundary in packed data.
- A norm that multiplies by `1 + weight` (Gemma, Qwen3.5) needs `zero_centered_gamma=True`.
- Head sizes: pass the attention's real query/key and value head sizes as `kv_channels`.
  `config.head_dim` is not always that; in GLM-5 it is only the RoPE part of each head.
  For multi-head latent attention, check HF's effective softmax scale and which slice of
  Q and K receives rotary embeddings before substituting the attention core.
- Match attention dropout and projection biases to HF. Check the installed module's defaults.
- HF's `post_init()` overwrites TE weights; load the checkpoint before training.

## 5. Validation

Once the replacement outputs match, compare integrated HF and TE
outputs on the same BF16 inputs and with the same stochastic controls. For training, match
labels and loss reduction, then compare loss and a gradient or optimizer update on a replaced
parameter in each changed path. Use the same parameter layout and exercise a routed expert for
MoE. If the full model does not fit, start with a deterministic check for each changed layer type
and report the layers it covers. When you supply a comparison, put its source entry point, exact
command, what it checks and execution status together in the handoff. Report weight mapping
coverage and any blockers. When execution is outside scope or unavailable, mark numerical results
as unverified. Use FP32 to diagnose discrepancies.
If the workload calls for lower-precision execution, repeat the relevant checks with that recipe
and its own tolerances.

For requested FP8 training, verify the BF16 port first and choose a TE recipe supported by the
installed release, GPU and operation shapes. If the trainer manages TE precision, configure its
TE backend. In a manual loop, wrap the model forward in `with te.autocast(enabled=True, recipe=fp8_recipe):`
and call `loss.backward()` after leaving the context. Keep the baseline parameter and loss dtypes.
Compare loss and a gradient or update for a replaced weight against the matched BF16 run using
recipe-specific tolerances. Report the FP8 check as pending if the target is unavailable.
[TE training example](https://github.com/NVIDIA/TransformerEngine/blob/27486e03cfc1fa41f6932dcecdc47c71c47eac3e/README.rst#L75-L98)

Before recommending a speedup, warm up and synchronize end-to-end runs of optimized HF and TE.
Compare them first in BF16 with matched effective tokens, batching and GPU topology. Measure a
requested precision change separately. Report the HF/TE versions, workload, measured result,
regions kept or replaced and why, and mapping and parity evidence. Identify unmeasured results.
For training-step timing, see the [Llama warmup and timed loop](https://github.com/NVIDIA/TransformerEngine/blob/db1aff638b070cb9d08a0dc91629ff1457ead413/docs/examples/te_llama/utils.py#L196-L231)
and the [Mixtral per-step CUDA events](https://github.com/NVIDIA/TransformerEngine/blob/db1aff638b070cb9d08a0dc91629ff1457ead413/docs/examples/te_mixtral/utils.py#L340-L382).

## Troubleshooting case-study: Qwen3.5

These illustrative troubleshooting notes show how checkpoint, expert and attention issues can
surface when instructions in this skill are followed over successive iterations for one example
model - Qwen3.5.

**Weights are missing or left over (Qwen3.5).** Take the names from the checkpoint files, not
from `model.state_dict()`: a checkpoint may stack all experts of a layer in one tensor, and the
layer prefix may be deeper (`model.language_model.layers.N.`). Skip what HF skips
(`_keys_to_ignore_on_load_unexpected`). A model made of sub-models needs every sub-config's
`dtype` set, or HF builds parts of it in float32.

**MoE experts give wrong results or fail (Qwen3.5).** Check the grouped-GEMM backend's supported
cuBLAS versions and expert group sizes. Check activation ordering and where
routing probabilities are applied. If using MXFP8, follow that backend's row-alignment and
activation-layout requirements and keep the router and LM head outside the expert FP8 recipe.

**Gated DeltaNet rejects an input layout (Qwen3.5).** Check the installed core's requirements for
Q/K/V layout and head grouping and for the gate tensors' dtypes. Retain the HF core if the
available backend cannot implement the target behavior.

## Related skills

If the model is a protein or biology model, use the
[BioNeMo Recipes acceleration skill](https://github.com/NVIDIA-BioNeMo/bionemo-recipes/blob/19ee43214fec23e87916efd468eaa62df47dba3e/skills/bionemo-recipes-acceleration/SKILL.md)
instead.
