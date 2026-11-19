# Dense layers: the `te.TransformerLayer` recipe

`te` is `transformer_engine.pytorch` throughout this skill.

Use this pattern when every layer of the model computes

```text
h = x + attention(norm1(x))
y = h + mlp(norm2(h))
```

with a norm and an MLP activation that TE supports. Llama, Mistral and Gemma (v1) decoders have
this shape. `te.TransformerLayer` then replaces the whole HF layer and gives two fused
modules per layer:

| TE submodule | Fuses | Replaces in HF |
|---|---|---|
| `self_attention.layernorm_qkv` (`LayerNormLinear`) | `norm1` + Q, K and V projections in one GEMM | `input_layernorm`, `self_attn.{q,k,v}_proj` |
| `self_attention.proj` (`Linear`) | Attention output projection | `self_attn.o_proj` |
| `layernorm_mlp` (`LayerNormMLP`) | `norm2` + `fc1` + activation + `fc2` | `post_attention_layernorm`, `mlp.*` |

## Contents

- [Is the layer eligible?](#is-the-layer-eligible)
- [Constructor arguments from the HF config](#constructor-arguments-from-the-hf-config)
- [Recipe code](#recipe-code)
- [Family notes](#family-notes)
- [Partial conversion](#partial-conversion)
- [Published examples](#published-examples)

## Is the layer eligible?

Read the HF layer's `forward` and check each line of this list. One mismatch means the whole
layer cannot be replaced; go to [Partial conversion](#partial-conversion) for that sublayer.

- Norm placement is pre-norm on both sublayers, and there are exactly two norms per layer.
- The norm is `LayerNorm` or `RMSNorm`. A norm that multiplies by `(1 + weight)` needs
  `zero_centered_gamma=True`.
- The MLP is `fc2(act(fc1(x)))`, or gated as `fc2(act(gate(x)) * up(x))`, with one of TE's
  activations: `gelu`, `geglu`, `glu`, `qgelu`, `qgeglu`, `relu`, `reglu`, `srelu`, `sreglu`,
  `silu`, `swiglu`, `clamped_swiglu`.
- Attention is scaled dot-product self-attention with no extra terms. Query/key normalization,
  logit soft-capping and sliding windows exist in TE but as options you must set and verify.
- Biases are all on or all off. `TransformerLayer(bias=...)` is one switch for the QKV, output
  and MLP projections; a model with bias on some of them only is not an exact match.
- Residual connections are plain additions (no gates, no scaling, no layer-scale parameters).

## Constructor arguments from the HF config

`scripts/inspect_hf_config.py` prints these for a given config.

| `TransformerLayer` argument | HF config source |
|---|---|
| `hidden_size` | `hidden_size` |
| `ffn_hidden_size` | `intermediate_size` |
| `num_attention_heads` | `num_attention_heads` |
| `num_gqa_groups` | `num_key_value_heads` |
| `kv_channels` | `head_dim`, when the config sets one that differs from `hidden_size / num_attention_heads` |
| `layernorm_epsilon` | `rms_norm_eps` or `layer_norm_eps` |
| `normalization` | `"RMSNorm"` or `"LayerNorm"`, from the norm class used in the layer |
| `activation` | Gated SiLU → `"swiglu"`; gated GELU → `"geglu"`; plain GELU → `"gelu"`; plain SiLU → `"silu"` |
| `bias` | `False` when the projections have no bias |
| `zero_centered_gamma` | `True` when the HF norm computes `x * (1 + weight)` |
| `hidden_dropout`, `attention_dropout` | `0` for fine-tuning recipes that follow the published examples |
| `attn_input_format` | `"bshd"` for HF's batch-first tensors; `"thd"` for packed batches |
| `layer_number` | `layer_idx + 1`; HF layers receive a zero-based index, TE counts from 1 |
| `fuse_qkv_params` | See below |

**`fuse_qkv_params`.** The norm and the QKV GEMM are fused either way. The flag decides how the
weight is stored:

- `True`: one parameter, `layernorm_qkv.weight`, interleaved per GQA group. No concatenation on
  each forward pass, and it is what `te_gemma.py` and `te_mixtral.py` use. Needs the layout code
  in `references/weight-mapping.md`.
- `False`: three parameters, `layernorm_qkv.{query,key,value}_weight`, each a direct copy of the
  HF projection. This is what `te_llama.py` uses and is the simplest mapping.

Prefer `True`; use `False` when a direct-copy mapping matters more than the saved concatenation.

## Recipe code

This is the structure of the published `te_llama.py`, with four additions: the RoPE base is read
from the config, the layer number is passed, the loader restores the default dtype, and every TE
layer parameter is proven to have been loaded. Adapt the HF imports, the class names and the
constructor arguments to the model; keep the rest. `te` is `transformer_engine.pytorch`
throughout this skill.

```python
import gc
import os
import re
from contextlib import contextmanager

import torch
import transformer_engine.pytorch as te
import transformers
from transformer_engine.pytorch.attention import RotaryPositionEmbedding
from transformers.modeling_utils import _add_variant, load_state_dict
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM
from transformers.utils import WEIGHTS_INDEX_NAME
from transformers.utils.hub import get_checkpoint_shard_files


@contextmanager
def replace_decoder(te_decoder_cls):
    """Make HF build the model with the TE layer class instead of its own."""
    modeling = transformers.models.llama.modeling_llama
    original = modeling.LlamaDecoderLayer
    modeling.LlamaDecoderLayer = te_decoder_cls
    try:
        yield
    finally:
        modeling.LlamaDecoderLayer = original


def _build_rope_emb(config):
    """TE rotary embeddings with the same base as the HF model."""
    if getattr(config, "rope_scaling", None) is not None:
        # Scaled RoPE is not a plain base change. Copy inv_freq from the HF rotary
        # embedding module instead (see the note under this code).
        raise NotImplementedError("rope_scaling is set; copy inv_freq from the HF model")
    te_rope = RotaryPositionEmbedding(
        config.hidden_size // config.num_attention_heads, rotary_base=config.rope_theta
    )
    return te_rope(max_seq_len=config.max_position_embeddings).cuda()


class TELlamaDecoderLayer(te.TransformerLayer):
    """TE TransformerLayer with the constructor and forward signature HF expects."""

    def __init__(self, config, layer_idx, *args, **kwargs):
        super().__init__(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_gqa_groups=config.num_key_value_heads,
            bias=False,
            layernorm_epsilon=config.rms_norm_eps,
            hidden_dropout=0,
            attention_dropout=0,
            fuse_qkv_params=True,
            normalization="RMSNorm",
            activation="swiglu",
            attn_input_format="bshd",
            layer_number=layer_idx + 1,  # TE counts layers from 1
        )
        self.te_rope_emb = _build_rope_emb(config)

    def forward(self, hidden_states, *args, **kwargs):
        # HF passes its additive mask, position ids and caches; TE takes none of them.
        # The layer was built with the default "causal" mask type, so TE builds the
        # causal mask itself and the HF mask is deliberately not forwarded.
        if isinstance(hidden_states, tuple):  # older transformers pass the previous tuple
            hidden_states = hidden_states[0]
        return super().forward(hidden_states, rotary_pos_emb=self.te_rope_emb)


class TELlamaForCausalLM:
    """LlamaForCausalLM built with TELlamaDecoderLayer in place of LlamaDecoderLayer."""

    def __new__(cls, config: LlamaConfig):
        with replace_decoder(te_decoder_cls=TELlamaDecoderLayer):
            return LlamaForCausalLM(config)

    @classmethod
    def from_pretrained_local(cls, model_dir, *, config, torch_dtype=torch.bfloat16):
        """Build the TE model and load a local sharded HF checkpoint into it.

        The TE layers are created on the GPU; call ``.cuda()`` on the result to move
        the embeddings, final norm and LM head there as well.
        """
        # HF and TE modules pick up the default dtype at construction.
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch_dtype)
        try:
            model = cls(config)
        finally:
            torch.set_default_dtype(previous_dtype)

        for index_name in ("model.safetensors.index.json", WEIGHTS_INDEX_NAME):
            archive_file = os.path.join(model_dir, _add_variant(index_name, None))
            if os.path.isfile(archive_file):
                break
        else:
            raise FileNotFoundError(f"No sharded checkpoint index found in {model_dir}")
        shard_files, _ = get_checkpoint_shard_files(model_dir, archive_file)

        # Mark every TE layer parameter as "not loaded yet". Any element that is still NaN
        # after all shards are read was never written by replace_params.
        te_state_dict = model.state_dict()
        te_layer_keys = [
            key
            for key, value in te_state_dict.items()
            if key.startswith("model.layers.")
            and not key.endswith("_extra_state")
            and value.is_floating_point()
        ]
        with torch.no_grad():
            for key in te_layer_keys:
                te_state_dict[key].fill_(float("nan"))

        for shard_file in shard_files:
            state_dict = load_state_dict(shard_file)
            # TE layers: explicit mapping. Everything else (embeddings, final norm,
            # LM head) has unchanged names and loads through load_state_dict.
            replace_params(state_dict, te_state_dict, config)
            model.load_state_dict(state_dict, strict=False)
            del state_dict
            gc.collect()

        unloaded = [key for key in te_layer_keys if torch.isnan(te_state_dict[key]).any()]
        if unloaded:
            raise RuntimeError(f"TE parameters not fully loaded from the checkpoint: {unloaded}")
        return model
```

`replace_params` for this layer is in `references/weight-mapping.md`. The NaN sentinel
matters because `strict=False` would otherwise let a mistyped key, or a fused tensor that
received only its gate half or only its query rows, stay at random initialization without any
error.

Notes on the parts that depend on the installed `transformers` release:

- **Layer return value.** The model loop either assigns the layer result directly
  (`hidden_states = decoder_layer(...)`) or indexes it (`layer_outputs[0]`). Read the installed
  loop and return a tensor for the first form and `(hidden_states,)` for the second. For Llama
  the published tutorial returns a tensor for `transformers` 4.57 and newer. If `transformers` is
  not installed, return a tensor and say in the report that this was assumed.
- **Masks.** With TE's causal mask and no padding mask, the result is exact for unpadded and
  for right-padded batches: a real token never attends to padding that comes after it, and the
  loss ignores padded positions. Left-padded batches are not handled, because real tokens would
  attend to the padding in front of them. Use right padding for training, pack the batch with
  `attn_input_format="thd"`, or switch to `"padding_causal"` with a boolean mask of shape
  `[batch, 1, 1, seq]` in which `True` means masked out. TE derives sequence lengths from that
  mask by counting, so it describes padding at the end.
- **RoPE.** `config.rope_theta` is read without a fallback on purpose; a missing attribute
  should stop the recipe, not default to 10000. When the config stores RoPE settings elsewhere,
  or `rope_scaling` is set, copy the frequencies from the HF rotary embedding module of the same
  family into `te_rope.inv_freq` before calling it, as the published Mixtral tutorial does.
- **KV cache.** The TE layer ignores HF's `past_key_values`. Set `config.use_cache = False`, as
  the published tutorials do, and evaluate with full-sequence forward passes.

Things the HF model still owns after this conversion: embeddings, the final norm, the LM head,
the model-level forward loop, the loss and the tokenizer.

## Family notes

These come from the published TE examples and the HF configs. Confirm against the installed
`transformers` source, because model code changes between releases.

| Family | HF layer class | Norm | Activation | Bias | Other arguments | Notes |
|---|---|---|---|---|---|---|
| Llama 2 / 3 | `LlamaDecoderLayer` | `RMSNorm` | `swiglu` | `False` | `num_gqa_groups=num_key_value_heads` | Llama 3 uses `rope_theta=500000`; later checkpoints also set `rope_scaling` |
| Mistral 7B | `MistralDecoderLayer` | `RMSNorm` | `swiglu` | `False` | `num_gqa_groups=num_key_value_heads` | Same layer as Llama; early checkpoints set `sliding_window` |
| Gemma (v1) | `GemmaDecoderLayer` | `RMSNorm` | `geglu` | `False` | `num_gqa_groups=num_key_value_heads`, `zero_centered_gamma=True`, `kv_channels=config.head_dim` | HF Gemma norm multiplies by `(1 + weight)`; `head_dim` is set explicitly |

Each HF class lives in `transformers.models.<family>.modeling_<family>` and is the name to patch
in `replace_decoder`.

Models that look similar but do not pass the eligibility list as a whole layer: architectures
with query/key normalization, with bias on the QKV projections only, with four norms per layer,
or with logit soft-capping. Treat those with partial conversion unless you verify that the
matching `TransformerLayer` option reproduces the HF computation exactly.

## Partial conversion

When the layer as a whole does not match, fuse the sublayers that do and keep the HF layer's
own `forward` for everything else.

| HF computation | TE module | Condition |
|---|---|---|
| `attn(norm(x))` | `te.MultiheadAttention(input_layernorm=True, normalization=..., fuse_qkv_params=True)` | The norm output feeds only the Q/K/V projections |
| `mlp(norm(x))`, plain or gated | `te.LayerNormMLP(hidden_size, ffn_hidden_size, normalization=..., activation=...)` | The norm output feeds only the MLP |
| `linear(norm(x))` | `te.LayerNormLinear(in_features, out_features, normalization=...)` | The norm output feeds only that projection |
| A projection with no norm in front | `te.Linear` | Last resort; no fusion |

Before removing a norm, find every consumer of its output. If the normalized tensor is also used
by a router, a second branch or a returned hidden state, fusing it into one consumer changes the
others; `LayerNormLinear(return_layernorm_output=True)` returns the normalized tensor as well
for that case.

Replace the fused modules on the HF layer and set the old norm to `torch.nn.Identity()` so it
runs exactly once. The weight keys are the same as in the whole-layer case without the
`self_attention.` or `layernorm_mlp.` prefixes that `TransformerLayer` adds.

## Published examples

- Llama: <https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/te_llama/te_llama.py>
- Gemma (adds KV cache generation and CUDA graphs):
  <https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/te_gemma/te_gemma.py>
