---
name: transformer-engine-convert-hf
description: >-
  Produce a TE based drop-in module replacement for the corresponding HF Transformer Layer module
  for the model similar to the one in docs/examples/te_llama/te_llama.py for llama and
  docs/examples/te_gemma/te_gemma.py for gemma. The TE based drop-in replacement should use most
  appropriate fused implementation as necessary such as fusing layer norm into QKV.
license: Apache-2.0
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
compatibility: >-
  Writing the implementation needs no GPU. Running it needs transformer_engine[pytorch],
  transformers and an NVIDIA GPU (FP8 needs Hopper or newer; MXFP8 needs Blackwell), plus what
  the training script uses (the Llama tutorial also needs accelerate, datasets and peft). Newer
  models may need more, such as flash-linear-attention and causal-conv1d for Qwen3.5.
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

The result is a TE implementation of the model: one Python module, named after the model family
(`te_llama.py`, `te_mixtral.py`), that replaces the basic building blocks of the Hugging Face
(HF) model with Transformer Engine (TE) modules and loads the original checkpoint into it. The
building block is the whole transformer layer, as for Llama and Gemma, or else the attention
(MHA, linear attention) and MLP or MoE blocks inside it. Steps 1 to 3 are decisions; step 4
writes the code. In this skill `te` means `transformer_engine.pytorch`.

Not for JAX or inference serving (vLLM, TensorRT-LLM). For Megatron-LM or another PyTorch
framework, the TE implementation can serve as a reference for which TE modules to use and how
the weights map.

## 1. Check the architecture and feasibility

Read the model's `config.json`, the weight names in its checkpoint and the decoder layer's
`forward` in the installed `transformers`, and match each component to a TE module:

| HF component | TE module |
|---|---|
| The whole transformer block | `te.TransformerLayer` |
| Attention: norm, QKV projection, core attention, output projection | `te.MultiheadAttention` |
| Core attention | `te.DotProductAttention` |
| A norm and the projections that read it | `te.LayerNormLinear` |
| A norm and a dense MLP | `te.LayerNormMLP` |
| MoE experts | `GroupedLinear` in `transformer_engine.pytorch.ops` |
| A single projection or norm | `te.Linear`, `te.RMSNorm`, `te.LayerNorm` |

The table is not exhaustive. Newer architectures may need other TE modules, some of them
experimental, such as `te.GatedDeltaNetAttention` for Gated DeltaNet; check what the installed
TE exports. A component with no TE module stays HF code.

If the model is a protein or biology model, use the
[BioNeMo Recipes acceleration skill](https://github.com/NVIDIA-BioNeMo/bionemo-recipes/blob/19ee43214fec23e87916efd468eaa62df47dba3e/skills/bionemo-recipes-acceleration/SKILL.md)
instead.

## 2. Choose the granularity

Try level 1 first, then level 2, and go no deeper.

**Level 1: the whole transformer block.** When every component maps to TE, subclass
`te.TransformerLayer`, as
[`te_llama.py`](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/te_llama/te_llama.py)
(`class TELlamaDecoderLayer(te.TransformerLayer)`) and
[`te_gemma.py`](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/te_gemma/te_gemma.py)
(`class TEGemmaDecoderLayer(te.TransformerLayer)`) do.

**Level 2: attention and MLP separately.** When the block has something `te.TransformerLayer`
cannot express, such as experts or linear attention, replace the attention and the MLP each with
TE modules:

- Attention: `te.MultiheadAttention` when it fits; otherwise `te.LayerNormLinear` for the norm
  and input projections, a TE core attention, `te.Linear` for the output projection, and HF code
  for what lies between.
- MLP: `te.LayerNormMLP` for a dense MLP. For mixture-of-experts (MoE), write an MoE layer
  around grouped GEMMs, as
  [`te_mixtral.py`](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/te_mixtral/te_mixtral.py)
  does in `class TEMixtralSparseMoeBlock(nn.Module)`.

A block that neither level covers stays HF code.

## 3. Map the weights

Every checkpoint weight of a replaced block needs a destination in the TE modules, and every TE
parameter a source; make the loader fail on a weight left over on either side. Follow
`replace_params(hf_state_dict, te_state_dict, config)` in `te_llama.py` and
`replace_params_bf16` in `docs/examples/te_mixtral/hf_to_te_weights.py`.

## 4. Implement

### Choose what to fuse

Choose fusions that keep the math identical and ideally improve performance, for example:

- a norm with the QKV projection: `te.LayerNormLinear`, or `te.MultiheadAttention` with
  `input_layernorm=True`;
- a norm with the MLP: `te.LayerNormMLP`.

### Design patterns

Follow the three existing TE implementations: `te_llama.py`, `te_gemma.py` and
`docs/examples/te_mixtral/`. They show how to swap HF's decoder layer class while HF builds the
model (`replace_decoder(te_decoder_cls)`), load the checkpoint
(`from_pretrained_local(cls, pretrained_model_name_or_path, *args, config, **kwargs)`), map the
weights, and keep the layer interface the installed `transformers` model loop expects.

## Troubleshooting

**How do I check the TE implementation?** Run HF in BF16, HF in float32 and the TE
implementation in BF16 on the same real text, a few layers at a time if the model does not fit
on one GPU. The TE implementation is right when it is no further from the float32 output than
HF's own BF16 run is; a swapped weight puts it many times further, and top-1 agreement alone can
miss one. For fine-tuning, also run one training step under `te.autocast` with an FP8 recipe:
the loss should be close to the BF16 loss and the gradients finite. Time it against HF before
claiming a speedup. Report the granularity, what stayed HF code and which checks ran.

**The outputs differ from HF.** Look for a config value TE did not get:

- RoPE base: `RotaryPositionEmbedding` defaults to 10000; pass `rope_theta` as `rotary_base`.
- A norm that multiplies by `1 + weight` (Gemma, Qwen3.5) needs `zero_centered_gamma=True`.
- Head sizes: pass the attention's real query/key and value head sizes as `kv_channels`.
  `config.head_dim` is not always that; in GLM-5 it is only the RoPE part of each head.
- Sliding window: HF lets a token see itself and `sliding_window - 1` earlier tokens; pass
  `window_size=(sliding_window - 1, 0)`.
- `te.MultiheadAttention` defaults to `attention_dropout=0.1` and `bias=True`.
- TE builds the causal mask; do not pass HF's additive mask.
- HF's `post_init()` overwrites TE weights; load the checkpoint before training.

**Weights are missing or left over (Qwen3.5).** Take the names from the checkpoint files, not
from `model.state_dict()`: a checkpoint may stack all experts of a layer in one tensor, and the
layer prefix may be deeper (`model.language_model.layers.N.`). Skip what HF skips
(`_keys_to_ignore_on_load_unexpected`). A model made of sub-models needs every sub-config's
`dtype` set, or HF builds parts of it in float32.

**MoE experts give wrong results or fail (Qwen3.5).**

- cuBLAS 13.7.0 to 13.8.0 returns wrong grouped GEMM results; use 13.8.1 or newer.
- One `GroupedLinear` holds at most 64 experts; split more into blocks of 64.
- Under MXFP8, pad each expert's rows to a multiple of 128 (`te.moe_permute_and_pad_with_probs`)
  and apply the routing weights in `ScaledSwiGLU` between the two `GroupedLinear`s.
- Keep the router and the LM head out of quantization.

**`te.GatedDeltaNetAttention` fails with "buffer for 'v' must have a stride-1 innermost dim"
(Qwen3.5).** Pass Q, K and V contiguous in BF16, with Q and K repeated to the number of value
heads, and `g` and `beta` in float32.
