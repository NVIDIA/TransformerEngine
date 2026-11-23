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
  Writing a recipe needs no GPU. Running one needs transformer_engine[pytorch], transformers and
  an NVIDIA GPU (FP8 needs Hopper or newer; MXFP8 needs Blackwell).
metadata:
  version: "0.3.0"
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

The result is one Python module, named after the model family (`te_llama.py`, `te_mixtral.py`),
that replaces the Hugging Face (HF) decoder layer with a Transformer Engine (TE) one and loads
the original checkpoint into it. The TE tutorials already hold this code: follow the one that
matches the model rather than writing from scratch. In this skill `te` means
`transformer_engine.pytorch`.

Not for Megatron-LM, JAX, or inference serving (vLLM, TensorRT-LLM).

## 1. Match the architecture

Read the model's `config.json`, the weight names in its checkpoint
(`model.safetensors.index.json`) and the decoder layer's `forward` in the installed
`transformers`. Then pick the reference:

| The model is | Follow |
|---|---|
| A dense decoder language model (Llama, Mistral, Qwen, Gemma) | [`docs/examples/te_llama/te_llama.py`](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/te_llama/te_llama.py); for Gemma, [`docs/examples/te_gemma/te_gemma.py`](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/te_gemma/te_gemma.py) |
| A mixture-of-experts (MoE) language model (Mixtral, Qwen MoE) | [`docs/examples/te_mixtral/te_mixtral.py`](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/te_mixtral/te_mixtral.py) and its [`hf_to_te_weights.py`](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/te_mixtral/hf_to_te_weights.py) |
| A genomics or protein language model (for example ESM-2) | The [BioNeMo Recipes acceleration skill](https://github.com/NVIDIA-BioNeMo/bionemo-recipes/blob/19ee43214fec23e87916efd468eaa62df47dba3e/skills/bionemo-recipes-acceleration/SKILL.md); use it instead of this one |

A layer neither tutorial covers (attention with extra norms or an output gate, or linear
attention such as Gated DeltaNet): fuse the layer's input norm with every projection that reads
it into one `te.LayerNormLinear`, run the core on `te.DotProductAttention` or
`te.GatedDeltaNetAttention` where it matches, and keep HF's code for the rest. A model whose
config lists `layer_types` gets one layer class that builds the mixer each entry names.

## 2. Write the module

Reuse the parts of the reference and keep their names.

From `te_llama.py` (dense):

- `replace_decoder(te_decoder_cls)`: makes HF build its model with the TE layer class.
- `class TELlamaDecoderLayer(te.TransformerLayer)`, with
  `__init__(self, config, *args, **kwargs)`: the TE layer, every size read from `config`.
- `TELlamaForCausalLM.from_pretrained_local(cls, pretrained_model_name_or_path, *args, config,
  **kwargs)`: builds the model and loads the checkpoint shards.
- `replace_params(hf_state_dict, te_state_dict, config)`: copies each HF weight into its TE
  parameter.

From `te_mixtral.py` and `hf_to_te_weights.py` (MoE):

- `class TEMixtralDecoderLayer(nn.Module)`, with
  `__init__(self, config, layer_idx, dispatcher=None)`: `te.MultiheadAttention` with the input
  norm fused into QKV, then a `te.RMSNorm` feeding the MoE block.
- `class TEMixtralSparseMoeBlock(nn.Module)`, with `__init__(self, config, dispatcher=None)`:
  router plus experts in `GroupedLinear`.
- `replace_params_bf16(hf_state_dict, te_state_dict, config) -> set[str]`: the weight mapping,
  including stacked expert tensors.

Take weight names from the checkpoint files, not from `model.state_dict()`: experts may be
stored stacked as `[num_experts, out, in]`, and the layer prefix may be deeper
(`model.language_model.layers.N.`). Every checkpoint weight needs a destination and every TE
weight a source; make the loader raise on a leftover instead of relying on `strict=False`.

## 3. Check it against HF

On the real checkpoint, a few layers at a time if the model does not fit on one GPU:

- Run HF in BF16, HF in float32 and the TE module in BF16 on the same real text. The TE module
  is right when it is no further from the float32 output than HF's own BF16 run is; a swapped
  weight puts it many times further. Top-1 agreement alone can miss a swap.
- For fine-tuning, also run one forward and backward pass under `te.autocast` with
  `MXFP8BlockScaling` (Blackwell) or `DelayedScaling` (Hopper): the loss should be close to the
  BF16 loss and the gradients finite.
- Before claiming a speedup, time the TE module against HF, with HF's optional fast kernels
  installed, at the token counts the user trains with.

Without a GPU, say that the module was not run.

## Rules that prevent silent errors

- **RoPE base.** `RotaryPositionEmbedding` defaults to base 10000; pass the model's
  `rope_theta` as `rotary_base`.
- **Norms.** A norm that multiplies by `1 + weight` (Gemma, Qwen3.5) needs
  `zero_centered_gamma=True`.
- **Head size.** When the config sets `head_dim`, pass it as `kv_channels`.
- **Defaults.** `te.MultiheadAttention` defaults to `attention_dropout=0.1` and `bias=True`; set
  both from the config.
- **Masks.** Do not pass HF's additive mask to a TE attention block. TE builds the causal mask,
  and its masks are boolean with `True` meaning masked out.
- **KV cache.** TE layers ignore HF's cache; set `config.use_cache = False`.
- **Many experts.** One `GroupedLinear` holds at most 64 experts; split more into blocks. Under
  MXFP8, pad each expert's rows to a multiple of 128 (`te.moe_permute_and_pad_with_probs`) and
  put `ScaledSwiGLU`, not `SwiGLU`, between the two `GroupedLinear`s.
- **cuBLAS 13.7.0 to 13.8.0** returns wrong grouped GEMM results; use 13.8.1 or newer before
  trusting an MoE check.
- **Initial values.** HF's `post_init()` overwrites TE weights; fill the model from a
  checkpoint before training it.

## Report

Tell the user which reference the module follows, what stayed HF code and why, and which checks
ran or could not.
