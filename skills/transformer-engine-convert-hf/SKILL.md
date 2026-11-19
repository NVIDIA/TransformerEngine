---
name: transformer-engine-convert-hf
description: >-
  Use when asked to convert, port or accelerate a Hugging Face PyTorch model (Llama, Gemma,
  Mistral, Mixtral) with Transformer Engine (TE), or to write a TE fine-tuning recipe. Produces a
  te_llama.py-style module built on fused TE blocks (LayerNorm+QKV, LayerNorm+MLP,
  TransformerLayer, GroupedLinear MoE experts) with the checkpoint weight mapping, for BF16, FP8
  or MXFP8. Not for Megatron-LM, JAX or inference serving.
license: Apache-2.0
compatibility: >-
  Python >= 3.10. Writing a recipe needs no GPU. Running one needs transformer_engine[pytorch],
  transformers and an NVIDIA GPU (FP8 needs Hopper or newer; MXFP8 and NVFP4 need Blackwell).
metadata:
  version: "0.1.0"
  author: NVIDIA Transformer Engine contributors
  domain: model-training
  tags:
    - transformer-engine
    - hugging-face
    - pytorch
    - fine-tuning
    - fp8
    - mxfp8
    - mixture-of-experts
---

# Convert a Hugging Face model to a Transformer Engine recipe

## Purpose

A *recipe* is one Python module, named after the model family (`te_llama.py`, `te_gemma.py`,
`te_mixtral.py`), that rebuilds a Hugging Face (HF) model on Transformer Engine building blocks
and loads the original checkpoint into it. The Transformer Engine repository publishes three of
them as tutorials; this skill produces the same kind of file for the model the user names.

In this skill `te` always means `transformer_engine.pytorch` (`import transformer_engine.pytorch
as te`).

TE's speed comes from *fused* blocks, not from swapping `nn.Linear` for `te.Linear` one at a
time. A fused block runs the normalization, the projection and the quantization as one unit, so
the goal is to move as much of each layer as possible into fused TE modules while keeping the
math of the original model unchanged.

## When to Use This Skill

- "I want a recipe for Llama" / "convert Gemma to Transformer Engine" / "port Mixtral to TE".
- "I want a fine-tuning recipe for this HF model with TE", "accelerate my HF fine-tune with FP8".
- "How do I fuse layer norm and QKV?", "replace the MoE block with GroupedLinear".

Do not use it for Megatron-LM or NeMo launch configuration, JAX/Flax models, inference serving
(vLLM, TensorRT-LLM), or installing and building TE. Those have their own documentation.

## Examples

**Dense decoder, fine-tuning recipe**

> "I want a fine-tuning recipe for Llama with TE. The checkpoint is in `/models/Llama-2-7b-hf`."

Output: `te_llama.py` with `TELlamaDecoderLayer` (a `te.TransformerLayer` with RMSNorm, SwiGLU
and grouped-query attention), `TELlamaForCausalLM.from_pretrained_local`, `replace_params`, a
`select_recipe` helper, and a docstring example that runs a training step under `te.autocast`.
The report says the static check passed and, without a GPU, that numerical parity was not run.

**Dense decoder with family-specific arguments**

> "Convert Gemma to Transformer Engine."

Output: `te_gemma.py` whose layer adds `activation="geglu"`, `zero_centered_gamma=True` and
`kv_channels=config.head_dim` to the same structure.

**Mixture of experts**

> "Convert Mixtral to Transformer Engine."

Output: `te_mixtral.py` whose layer composes `te.MultiheadAttention` with a fused input
norm, a TE norm, and an MoE block that sorts tokens with `moe_permute` and runs all experts
through `GroupedLinear`, with the router kept out of quantization.

## Prerequisites

- **To write a recipe:** Python 3.10 or newer. The bundled scripts use only the standard
  library and read local files; they need no GPU, no network and no TE installation.
- **To run a recipe:** `transformer_engine[pytorch]`, `torch`, `transformers` and an NVIDIA GPU.
  Requirements for each TE release are in the Transformer Engine installation guide.
- **Model files:** the model's `config.json`, and for loading weights a local checkpoint
  directory. Access to gated models is the user's; do not work around it.

## What to deliver

Write the recipe in the user's working directory, named `te_` plus the family name, unless they
name another path. One file, with these parts:

| Part | Purpose |
|---|---|
| Module docstring | Model family, techniques applied, how to load and run it |
| TE layer class | The fused replacement for the HF layer, e.g. `TELlamaDecoderLayer` |
| Model class | For example `TELlamaForCausalLM`, with a loader that reads the HF checkpoint |
| `replace_params(hf_state_dict, te_state_dict, config)` | The HF key to TE key weight mapping |
| Precision hook | Only for fine-tuning requests: a `select_recipe` helper and a docstring example of a training step under `te.autocast` |

Then report to the user as described in step 7. A recipe that was written but never executed
must be described that way.

## Instructions

### 1. Read the architecture

```bash
python scripts/inspect_hf_config.py path/to/config.json
```

Pass the `config.json` (or the directory containing it) of the checkpoint the user will load.
The script prints the facts the recipe depends on (normalization, activation, attention heads
and GQA groups, RoPE base, MoE settings) together with the matching TE constructor arguments and
a suggested pattern.

A config cannot show the order of normalization and residual connections. Confirm it by reading
the layer's `forward` in the installed `transformers` modeling source for that family.

When the user names a Hub model and there is no local copy, ask for the path of the checkpoint
they will load, or for its `config.json`. If that is not available either, do not reconstruct a
config from memory to feed the script: use the family notes in the references, read every size
from `config` at run time instead of hard-coding it, and list in the report which facts were
assumed rather than read.

### 2. Choose the deepest conversion that keeps the math identical

| Layer structure | Recipe pattern | Read |
|---|---|---|
| Pre-norm dense layer: `x + attn(norm(x))`, then `x + mlp(norm(x))`, with a norm and activation TE supports | Subclass `te.TransformerLayer` | [dense-decoder.md](references/dense-decoder.md) |
| Mixture-of-experts layer (router plus experts) | Compose `te.MultiheadAttention(input_layernorm=True)`, a TE norm, and grouped experts | [moe-decoder.md](references/moe-decoder.md) |
| Anything else (extra norms, gated residuals, custom attention) | Fuse the sublayers that match (`MultiheadAttention`, `LayerNormMLP`, `LayerNormLinear`) and keep the rest native | "Partial conversion" in [dense-decoder.md](references/dense-decoder.md) |

Go as deep as equivalence allows. A whole-layer replacement fuses norm+QKV and norm+MLP at once;
sublayer fusion still gets one of them; replacing individual `nn.Linear` modules gets neither and
is the last resort. If a TE module cannot express the model's computation, leave that part native
instead of changing the model to fit.

### 3. Write the recipe

Start from the code in the reference for the chosen pattern and change only what the model
requires: the HF classes being wrapped, the constructor arguments from step 1, and the forward
arguments the HF model passes to its layers. Keep the naming of the published tutorials (a
`TE...DecoderLayer` class, `replace_decoder`, a `TE...ForCausalLM` class, `replace_params`) so the
result reads like them.

### 4. Map the weights

Follow [weight-mapping.md](references/weight-mapping.md). Every HF parameter of a replaced layer
needs a destination, and every TE parameter needs a source. The fused tensors have specific
layouts (QKV interleaved per GQA group, gate and up halves of `fc1`), so copy the layout code
from the reference instead of concatenating by intuition.

### 5. Check the recipe

```bash
python scripts/check_recipe.py te_llama.py
```

Pass every file the recipe consists of. This static check needs no GPU. It fails when the
recipe has no fused TE block, no weight mapping, or a mapping that never writes the fused
parameters. It also prints the TE constructor arguments it found; compare them with the output
of step 1, or with the family notes when step 1 could not run. Fix every failure and read every
warning.

When a GPU with TE is available, also run the numerical parity check in
[validation.md](references/validation.md): same weights, same input, HF logits against TE logits
in BF16 with quantization off.

### 6. Add fine-tuning and precision hooks when asked

For a fine-tuning recipe, follow
[finetuning-and-precision.md](references/finetuning-and-precision.md): pick the recipe the GPU
supports, wrap the forward pass in `te.autocast`, and keep the layers that must stay in
high precision (MoE router, LM head) out of it. Leave precision alone when the user only asked
for a conversion.

### 7. Report

State the file written, the techniques applied per block, anything left native with the reason,
the checks that ran with their results, and the checks that could not run here (for example "no
GPU: parity not run").

## Techniques by block

| Block | Technique | TE API |
|---|---|---|
| Attention | Input norm and Q/K/V projection as one fused op | `layernorm_qkv` inside `MultiheadAttention` / `TransformerLayer` |
| Attention | Single fused QKV parameter | `fuse_qkv_params=True` |
| Attention | Grouped-query attention | `num_gqa_groups=config.num_key_value_heads` |
| Attention | Rotary embeddings | `RotaryPositionEmbedding`, passed as `rotary_pos_emb` |
| Attention | Packed variable-length batches | `attn_input_format="thd"` with `cu_seqlens_*` |
| Dense MLP | Norm, both projections and the activation as one fused op | `layernorm_mlp` (`LayerNormMLP`) |
| MoE | Token sort by expert | `moe_permute`, `moe_unpermute` |
| MoE | All experts in one grouped GEMM | `ops.GroupedLinear` with per-expert split sizes |
| MoE | GEMM + SwiGLU + routing-weight scaling in one kernel | `ops.Sequential(GroupedLinear, ScaledSwiGLU, GroupedLinear)` under MXFP8 |
| Whole model | Low-precision GEMMs | `te.autocast` with an FP8, MXFP8 or NVFP4 recipe |

## Rules that prevent silent errors

- **RoPE base.** `RotaryPositionEmbedding` defaults to base 10000. Pass
  `rotary_base=config.rope_theta`, or the model runs with wrong positions and still produces
  plausible text. A config with `rope_scaling` needs the HF frequencies copied instead.
- **Masks.** HF hands each layer an additive float mask; TE masks are boolean with `True`
  meaning *masked out*. Do not forward the HF mask. For a causal model let TE build the causal
  mask, which is exact for unpadded and right-padded batches. Left-padded batches need a padding
  mask or packed (`thd`) input; the dense reference explains both.
- **Layer return type.** The HF model loop either uses the layer result directly or indexes it
  as a tuple, depending on the `transformers` release. Return what the installed loop expects.
- **KV cache.** TE layers ignore HF's `past_key_values`. Set `config.use_cache = False`.
- **`strict=False` hides mistakes.** The loaders use it because replaced layers have new key
  names. Prove that every TE layer parameter was written (the NaN sentinel in the dense
  reference) instead of trusting a load that did not raise.
- **Layer numbers start at 1** in TE (`layer_number=layer_idx + 1`).
- **Convert before** building the optimizer, wrapping with Accelerate or PEFT, or compiling,
  because fused modules own new parameters.
- **Check what the installed TE supports** before using grouped, expert-parallel or
  block-scaled features; they depend on the TE version and GPU generation.

## Available Scripts

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/inspect_hf_config.py` | Read a model config and print the architecture facts, the suggested recipe pattern and the TE constructor arguments as JSON | `source`: a `config.json` file or the directory containing it |
| `scripts/check_recipe.py` | Statically check a generated recipe for fused blocks, a weight mapping and mapped fused parameters; print the TE constructor arguments found | `files`: one or more recipe files, including the file that defines the weight mapping |

Run them with bash as shown in the instructions, or with `run_script()` where the harness provides
it:

```python
run_script("scripts/inspect_hf_config.py", args=["path/to/config.json"])
run_script("scripts/check_recipe.py", args=["te_llama.py"])
```

Both scripts only read the files given to them, print JSON to standard output and exit non-zero
on failure.

## Limitations

- **Unexecuted recipes.** Without a GPU the recipe is checked structurally only. The numerical
  parity check has to pass before the recipe is trusted for training.
- **Layer shapes.** Whole-layer replacement covers pre-norm layers with two norms. Layers with
  extra norms, gated residuals or attention variants get partial conversion.
- **Family coverage.** The config script knows the structure of a few families (Llama, Mistral,
  Gemma, Mixtral, Qwen2, Qwen3, Gemma 2). For others it reports the structure as unconfirmed.
- **Static check.** `check_recipe.py` cannot tell a correct weight layout from a wrong one.
- **Version-dependent features.** The fused grouped MLP and expert parallelism depend on the TE
  release, GPU generation and library versions. They are opt-in, not the default.
- **Generation.** The converted layers ignore HF's KV cache, so `generate()` with caching does
  not work on the default recipe. Generation through TE's own KV cache is shown in the published
  Gemma tutorial.
- **Padding.** The default recipe is exact for unpadded and right-padded batches only.
- **Scope.** PyTorch only.

## Troubleshooting

| Error / Symptom | Cause | Solution |
|---|---|---|
| Parity error near 1.0, or fluent but wrong text | Wrong fused layout or constructor argument | Check in order: QKV layout, gate/up order in `fc1`, RoPE base, `zero_centered_gamma`, `kv_channels`, mask type. Compare one layer at a time |
| `TE parameters not fully loaded from the checkpoint` | The mapping skipped a key or half of a fused tensor | Print the HF `state_dict()` keys for one layer and compare with the mapping table |
| Shape error while copying Q/K/V | The config sets an explicit `head_dim` | Pass it as `kv_channels` and use it as the head size in the QKV layout code |
| HF model loop fails on the layer output | Tuple versus tensor return type | Return what the installed `transformers` loop expects |
| `check_recipe.py` reports the mapping is imported | The mapping lives in another file | Pass that file to the script as well |
| No speedup from the fused grouped MLP | The fusion did not fire | Verify recipe type, GPU generation, feature sizes, interleaving and padding in the MoE reference |
| FP8 or MXFP8 unavailable | GPU or TE build does not support it | Call `te.is_fp8_available(return_reason=True)` or `te.is_mxfp8_available(return_reason=True)` and report the reason |

## Bundled references

- `references/dense-decoder.md`: `te.TransformerLayer` recipe, family notes, partial conversion.
- `references/moe-decoder.md`: MoE layer, grouped experts, fused grouped MLP, expert parallelism.
- `references/weight-mapping.md`: key tables and layout code for fused parameters.
- `references/finetuning-and-precision.md`: recipes, `autocast`, what to keep in high precision.
- `references/validation.md`: static, structural and numerical checks, and what to report.
