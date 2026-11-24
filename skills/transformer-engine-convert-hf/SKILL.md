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

The result is one Python module, named after the model family (`te_llama.py`, `te_mixtral.py`),
that replaces the Hugging Face (HF) decoder layer with Transformer Engine (TE) modules and loads
the original checkpoint into it. Steps 1 to 3 are decisions; step 4 writes the code. In this
skill `te` means `transformer_engine.pytorch`.

Not for Megatron-LM, JAX, or inference serving (vLLM, TensorRT-LLM).

## 1. Check the architecture and feasibility

Read the model's `config.json`, the weight names in its checkpoint
(`model.safetensors.index.json`) and the decoder layer's `forward` in the installed
`transformers`. List the components of one layer and what TE offers for each:

| HF component | TE module |
|---|---|
| The whole pre-norm block: `x + attn(norm(x))`, then `x + mlp(norm(x))` | `te.TransformerLayer` |
| Norm, QKV, softmax attention and output projection | `te.MultiheadAttention(input_layernorm=True)` |
| The softmax attention core alone | `te.DotProductAttention` |
| A Gated DeltaNet (linear attention) core | `te.GatedDeltaNetAttention` (experimental) |
| A norm and every projection that reads it | `te.LayerNormLinear` |
| A norm and a dense MLP | `te.LayerNormMLP` |
| A single projection; a norm | `te.Linear`; `te.RMSNorm`, `te.LayerNorm` |
| MoE experts | `GroupedLinear` and `ScaledSwiGLU` in a `te.ops.Sequential`, with `te.moe_permute_and_pad_with_probs` and `te.moe_unpermute` around them |
| Rotary position embedding | `RotaryPositionEmbedding` in `transformer_engine.pytorch.attention` |

For each component also note what the TE module must reproduce: the norm type and epsilon, and
whether it multiplies by `weight` or by `1 + weight`; the activation and whether the MLP is
gated; heads, key/value heads, head size, RoPE base, RoPE on part of the head, Q/K norms and
output gates; biases; the number of experts, top-k, routing function and any shared expert.

A component with no TE module (a causal convolution, a gated output norm, a custom router) stays
HF code. A genomics or protein language model (for example ESM-2) is covered by the
[BioNeMo Recipes acceleration skill](https://github.com/NVIDIA-BioNeMo/bionemo-recipes/blob/19ee43214fec23e87916efd468eaa62df47dba3e/skills/bionemo-recipes-acceleration/SKILL.md):
use it instead of this one.

## 2. Choose the granularity

**Level 1: the whole transformer block.** When every component maps to TE and the layer is the
standard pre-norm block, subclass `te.TransformerLayer`. It fuses the input norm into QKV and
the second norm into the MLP. Llama, Mistral and Gemma convert this way; follow
[`docs/examples/te_llama/te_llama.py`](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/te_llama/te_llama.py)
(`class TELlamaDecoderLayer(te.TransformerLayer)`) or
[`docs/examples/te_gemma/te_gemma.py`](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/te_gemma/te_gemma.py)
(`class TEGemmaDecoderLayer(te.TransformerLayer)`).

**Level 2: attention and MLP separately.** When the block has something `te.TransformerLayer`
cannot express (experts, linear attention, an output gate, a norm feeding both a router and the
experts, alternating layer types), compose the layer from one TE module per sublayer:

- Attention: `te.MultiheadAttention` when it fits. Otherwise `te.LayerNormLinear` for the norm
  and all input projections, `te.DotProductAttention` or `te.GatedDeltaNetAttention` for the
  core, `te.Linear` for the output projection, and HF code for what lies between.
- MLP: `te.LayerNormMLP` for a dense MLP. For mixture-of-experts (MoE), write the MoE layer: a
  router, the experts in grouped GEMMs (`GroupedLinear`) and the token permutation, as
  [`docs/examples/te_mixtral/te_mixtral.py`](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/te_mixtral/te_mixtral.py)
  does in `class TEMixtralSparseMoeBlock(nn.Module)`.

Mixtral converts this way, and so do Qwen3-Next and Qwen3.5, which mix Gated DeltaNet and gated
attention layers with 512-expert MoE blocks. DeepSeek-V4 is only partly covered by TE: write its
MoE layer around grouped GEMMs and keep what TE lacks as HF code. A model whose config lists
`layer_types` gets one layer class that builds the sublayers each entry names.

Swapping single `nn.Linear` modules for `te.Linear` fuses nothing; it is the last resort.

## 3. Map the weights

Every checkpoint weight of a replaced layer needs a destination in the TE modules, and every TE
parameter a source. [references/design-patterns.md](references/design-patterns.md) has the code
for each case below.

- Take the names from the checkpoint files, not from `model.state_dict()`: a checkpoint may
  stack all experts of a layer in one `[num_experts, out, in]` tensor, and the layer prefix may
  be deeper (`model.language_model.layers.N.`). Skip what HF itself skips: the class's
  `_keys_to_ignore_on_load_unexpected` and old per-layer `rotary_emb.inv_freq` buffers.
- Norms and output projections copy one to one under a TE name.
- Fused QKV (`fuse_qkv_params=True`) interleaves Q, K and V per key/value head group.
- A fused block's weight is the HF projections stacked row-wise in the order of the split in
  `forward` (`te.LayerNormLinear`), or gate rows then up rows (`te.LayerNormMLP` `fc1`).
- Expert `i` goes to `weight{i % 64}` of `GroupedLinear` block `i // 64`.
- Make the loader raise on a checkpoint weight with no destination, a TE weight nothing wrote,
  or a wrong dtype; loading with `strict=False` hides all three.

## 4. Implement

### Choose what to fuse

A fusion keeps the math when the fused operations run in the same order on the same input. TE
fuses:

- a norm with every projection that reads it: QKV in `te.MultiheadAttention(input_layernorm=True)`,
  or any set of projections in one `te.LayerNormLinear` (Q with its output gate, K and V; or the
  Q/K/V, Z, B and A projections of a Gated DeltaNet layer);
- a norm with a dense MLP in `te.LayerNormMLP`, gate and up in one GEMM;
- Q, K and V in one GEMM with `fuse_qkv_params=True`;
- the experts of a block in one grouped GEMM per projection (`GroupedLinear`), with gate and up
  as one weight per expert and the routing weight applied inside `ScaledSwiGLU`, which is exact
  when the down projection has no bias.

A norm whose output feeds two sublayers that cannot share one TE module, such as Mixtral's second
norm feeding the router and the experts, stays a separate `te.RMSNorm`. Keep the router and the
LM head out of quantization.

An MoE layer is only faster than HF's when the GPU is kept busy: Python loops over expert blocks,
`.item()` calls and per-expert weight quantization under FP8 stall it at small token counts. On
Qwen3.5 on GB300 a TE MoE layer built this way was 1.4× faster than HF at 64K tokens per GPU and
0.7× at 4K.

### Design patterns

[references/design-patterns.md](references/design-patterns.md) has the code; the tutorials hold
complete versions.

- **Swap the HF layer class.** A context manager replaces `modeling_<model>.<Model>DecoderLayer`
  while HF builds the model, so embeddings, final norm and LM head stay HF's with their weight
  names: `replace_decoder(te_decoder_cls)` in `te_llama.py`. Build the HF class named in the
  checkpoint's `architectures`.
- **Subclass the HF sublayer** to keep its `forward` arithmetic when only its projections and
  core change; delete the replaced projections.
- **Load the checkpoint** shard by shard:
  `TELlamaForCausalLM.from_pretrained_local(cls, pretrained_model_name_or_path, *args, config, **kwargs)`
  builds the model under a BF16 default dtype, maps the replaced layers with
  `replace_params(hf_state_dict, te_state_dict, config)` and loads the rest by name. A model made
  of sub-models (`config.text_config`, `config.vision_config`) needs every sub-config's `dtype`
  set too.
- **Match the HF interfaces.** The layer's `forward` signature and return type (tensor or tuple)
  are what the installed `transformers` model loop expects. TE layers ignore HF's mask and KV
  cache; set `config.use_cache = False`.

### Settings that must match the config

- **RoPE base.** `RotaryPositionEmbedding` defaults to 10000; pass the model's `rope_theta` as
  `rotary_base`.
- **Norms.** A norm that multiplies by `1 + weight` (Gemma, Qwen3.5) needs
  `zero_centered_gamma=True`.
- **Head size.** When the config sets `head_dim`, pass it as `kv_channels`.
- **Defaults.** `te.MultiheadAttention` defaults to `attention_dropout=0.1` and `bias=True`; set
  both from the config.
- **Masks.** Do not pass HF's additive mask to TE attention. TE builds the causal mask, and its
  masks are boolean with `True` meaning masked out.
- **Experts.** One `GroupedLinear` holds at most 64 experts. Under MXFP8, pad each expert's rows
  to a multiple of 128 and put `ScaledSwiGLU`, not `SwiGLU`, between the two `GroupedLinear`s.
- **cuBLAS 13.7.0 to 13.8.0** returns wrong grouped GEMM results; use 13.8.1 or newer.
- **Initial values.** HF's `post_init()` overwrites TE weights; fill the model from a checkpoint
  before training it.

## Check and report

On the real checkpoint, a few layers at a time if the model does not fit on one GPU:

- Run HF in BF16, HF in float32 and the TE module in BF16 on the same real text. The TE module
  is right when it is no further from the float32 output than HF's own BF16 run is; a swapped
  weight puts it many times further. Top-1 agreement alone can miss a swap.
- For fine-tuning, run one forward and backward pass under `te.autocast` with
  `MXFP8BlockScaling` (Blackwell) or `DelayedScaling` (Hopper): the loss should be close to the
  BF16 loss and the gradients finite.
- Before claiming a speedup, time the TE module against HF, with HF's optional fast kernels
  installed, at the token counts the user trains with.

Tell the user the granularity chosen, the reference followed, what stayed HF code and why, and
which checks ran or could not. Without a GPU, say that the module was not run.
