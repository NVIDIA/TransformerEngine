# Mapping Hugging Face weights onto TE modules

`te` is `transformer_engine.pytorch` throughout this skill.

A replaced layer has new parameter names and, for the fused parameters, new layouts. Everything
outside the replaced layers (embeddings, final norm, LM head) keeps its name and loads through
`load_state_dict(strict=False)`.

All keys below are relative to a layer prefix such as `model.layers.<n>.`.

## Contents

- [Dense layer (`TransformerLayer`)](#dense-layer-transformerlayer)
- [Fused QKV layout](#fused-qkv-layout)
- [`replace_params` for a dense layer](#replace_params-for-a-dense-layer)
- [MoE layer](#moe-layer)
- [Checkpoint shards, dtypes and ties](#checkpoint-shards-dtypes-and-ties)
- [Exporting back to Hugging Face](#exporting-back-to-hugging-face)

## Dense layer (`TransformerLayer`)

| HF key | TE key | Transform |
|---|---|---|
| `input_layernorm.weight` | `self_attention.layernorm_qkv.layer_norm_weight` | Copy |
| `self_attn.{q,k,v}_proj.weight` | `self_attention.layernorm_qkv.weight` | Interleave per GQA group (`fuse_qkv_params=True`) |
| `self_attn.{q,k,v}_proj.weight` | `self_attention.layernorm_qkv.{query,key,value}_weight` | Copy (`fuse_qkv_params=False`) |
| `self_attn.o_proj.weight` | `self_attention.proj.weight` | Copy |
| `post_attention_layernorm.weight` | `layernorm_mlp.layer_norm_weight` | Copy |
| `mlp.gate_proj.weight` | `layernorm_mlp.fc1_weight[:intermediate_size]` | Copy into the first half |
| `mlp.up_proj.weight` | `layernorm_mlp.fc1_weight[intermediate_size:]` | Copy into the second half |
| `mlp.down_proj.weight` | `layernorm_mlp.fc2_weight` | Copy |

For an ungated MLP, `fc1_weight` is a plain copy of the single up projection.

With `bias=True` the bias keys follow the same pattern: `layernorm_qkv.bias` (or
`{query,key,value}_bias`), `proj.bias`, `layernorm_mlp.fc1_bias`, `layernorm_mlp.fc2_bias`. With
`normalization="LayerNorm"` each norm also has `layer_norm_bias`.

Norm weights are a plain copy in both conventions: a model whose norm multiplies by
`(1 + weight)` stores the same tensor that TE expects with `zero_centered_gamma=True`.

HF key names differ between families (`fc1`/`fc2`, `c_fc`/`c_proj`, `dense`, ...). Read them
from the model's own `state_dict()` keys, not from this table.

## Fused QKV layout

With `fuse_qkv_params=True` and the default `qkv_weight_interleaved=True`, the fused weight has
`(num_heads + 2 * num_kv_heads) * head_size` rows. Rows are grouped per key/value head `g`:

```text
[ Q heads of group 0 | K_0 | V_0 | Q heads of group 1 | K_1 | V_1 | ... ]
```

Each group holds `num_heads / num_kv_heads` query heads followed by its one key head and one
value head, each `head_size` rows. Without GQA this reduces to `[q0 k0 v0 q1 k1 v1 ...]`.

```python
def _copy_qkv_to_fused(fused_qkv, proj_weight, kind, config):
    """Copy one HF projection ("q", "k" or "v") into TE's fused, interleaved QKV weight."""
    num_heads = config.num_attention_heads
    num_groups = config.num_key_value_heads
    heads_per_group = num_heads // num_groups
    head_size = getattr(config, "head_dim", None) or config.hidden_size // num_heads
    stride = heads_per_group + 2

    fused = fused_qkv.view(num_heads + 2 * num_groups, head_size, config.hidden_size)
    src = proj_weight.to(device=fused.device, dtype=fused.dtype)
    with torch.no_grad():
        if kind == "q":
            q = src.view(num_heads, head_size, config.hidden_size)
            for g in range(num_groups):
                fused[stride * g : stride * g + heads_per_group].copy_(
                    q[g * heads_per_group : (g + 1) * heads_per_group]
                )
        elif kind == "k":
            k = src.view(num_groups, head_size, config.hidden_size)
            for g in range(num_groups):
                fused[stride * g + heads_per_group].copy_(k[g])
        elif kind == "v":
            v = src.view(num_groups, head_size, config.hidden_size)
            for g in range(num_groups):
                fused[stride * g + heads_per_group + 1].copy_(v[g])
        else:
            raise ValueError(f"Unsupported projection kind: {kind}")
```

The function copies one projection at a time on purpose: in a sharded checkpoint the three
projections of a layer can sit in different files.

## `replace_params` for a dense layer

```python
def replace_params(hf_state_dict, te_state_dict, config):
    """Copy the parameters of the replaced layers from an HF state dict (or one shard of it)."""
    layer_prefixes = set()
    for key in hf_state_dict:
        match = re.match(r"model\.layers\.\d+\.", key)
        if match is not None:
            layer_prefixes.add(match.group())

    def copy(prefix, hf_name, te_name, rows=slice(None)):
        # A shard holds only some of a layer's tensors, so missing keys are expected here.
        if prefix + hf_name not in hf_state_dict:
            return
        dst = te_state_dict[prefix + te_name]
        src = hf_state_dict[prefix + hf_name]
        with torch.no_grad():
            dst[rows].copy_(src.to(device=dst.device, dtype=dst.dtype))

    ffn = config.intermediate_size
    for prefix in layer_prefixes:
        copy(prefix, "input_layernorm.weight", "self_attention.layernorm_qkv.layer_norm_weight")
        copy(prefix, "self_attn.o_proj.weight", "self_attention.proj.weight")
        copy(prefix, "post_attention_layernorm.weight", "layernorm_mlp.layer_norm_weight")
        copy(prefix, "mlp.gate_proj.weight", "layernorm_mlp.fc1_weight", slice(None, ffn))
        copy(prefix, "mlp.up_proj.weight", "layernorm_mlp.fc1_weight", slice(ffn, None))
        copy(prefix, "mlp.down_proj.weight", "layernorm_mlp.fc2_weight")

        fused_key = prefix + "self_attention.layernorm_qkv.weight"
        for kind in ("q", "k", "v"):
            hf_key = prefix + f"self_attn.{kind}_proj.weight"
            if hf_key not in hf_state_dict:
                continue
            if fused_key in te_state_dict:  # fuse_qkv_params=True
                _copy_qkv_to_fused(te_state_dict[fused_key], hf_state_dict[hf_key], kind, config)
            else:  # fuse_qkv_params=False
                name = {"q": "query", "k": "key", "v": "value"}[kind]
                te_name = f"self_attention.layernorm_qkv.{name}_weight"
                copy(prefix, f"self_attn.{kind}_proj.weight", te_name)
    return layer_prefixes
```

`te_state_dict` must come from `model.state_dict()` on the TE model so that its tensors alias
the live parameters; copying into them updates the model in place.

## MoE layer

Attention and norms, for the layer in `references/moe-decoder.md`:

| HF key | TE key | Transform |
|---|---|---|
| `input_layernorm.weight` | `self_attention.layernorm_qkv.layer_norm_weight` | Copy |
| `self_attn.{q,k,v}_proj.weight` | `self_attention.layernorm_qkv.weight` | Interleave per GQA group |
| `self_attn.o_proj.weight` | `self_attention.proj.weight` | Copy |
| `post_attention_layernorm.weight` | `post_attention_layernorm.weight` | Copy (standalone TE norm) |

Router and experts. Mixtral checkpoints exist in two formats, so look for both:

| HF key | TE key | Transform |
|---|---|---|
| `mlp.gate.weight` or `block_sparse_moe.gate.weight` | `mlp.gate.weight` | Copy |
| `mlp.experts.gate_up_proj`, shape `[E, 2I, H]` (packed format) | `mlp.experts_gate_up.weight{i}` | Slice expert `i` |
| `mlp.experts.down_proj`, shape `[E, H, I]` (packed format) | `mlp.experts_down.weight{i}` | Slice expert `i` |
| `block_sparse_moe.experts.{i}.w1.weight` (gate, legacy format) | `mlp.experts_gate_up.weight{i}[:I]` | Copy into the first half |
| `block_sparse_moe.experts.{i}.w3.weight` (up, legacy format) | `mlp.experts_gate_up.weight{i}[I:]` | Copy into the second half |
| `block_sparse_moe.experts.{i}.w2.weight` (down, legacy format) | `mlp.experts_down.weight{i}` | Copy |

`E` is the number of experts, `I` the expert intermediate size, `H` the hidden size. In the
legacy format `w1` is the gate, `w3` the up projection and `w2` the down projection; the numbering
is not the execution order.

Two transforms apply on top of this table when the corresponding feature is on:

- **Fused MXFP8 grouped MLP.** The fused kernel reads `fc1` output as alternating blocks of 32
  gate rows and 32 up rows. Convert each expert's `[gate; up]` weight with
  `te.interleave_glu_tensor(weight, 32)` before copying (and
  `te.deinterleave_glu_tensor` to go back).
- **Expert parallelism.** Rank `r` of an expert-parallel group of size `ep` owns the experts
  from `r * E / ep` up to, but not including, `(r + 1) * E / ep`. Load only that slice, indexed
  locally from zero.

## Checkpoint shards, dtypes and ties

- Load shard by shard and copy what each shard contains; a layer's tensors can be split across
  files. Use the NaN sentinel in `references/dense-decoder.md` to prove full coverage.
- Copy with `copy_` under `torch.no_grad()` and move the source to the destination's device and
  dtype. Do not reassign `.data`, which can silently change dtype or break parameter identity.
- Build the TE model in the dtype the checkpoint will run in (the examples set the default dtype
  to `torch.bfloat16` during construction).
- Buffers that are deliberately FP32, such as RoPE inverse frequencies, should stay FP32.
- Weight tying between the input embedding and the LM head lives outside the replaced layers
  and is preserved as long as the HF model class still owns both.

## Exporting back to Hugging Face

TE state dicts use the TE key names and may include `_extra_state` entries with quantization
metadata, so `save_pretrained` on a converted model does not produce a checkpoint that plain
`from_pretrained` can load. To export, apply the inverse of each row above into a fresh HF model
(split the fused QKV rows back into three projections, split `fc1_weight` into gate and up) and
save that model. Offer this only when the user needs an HF-format checkpoint.
