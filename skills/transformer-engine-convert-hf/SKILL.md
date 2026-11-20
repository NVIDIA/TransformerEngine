---
name: transformer-engine-convert-hf
description: >-
  Use when asked to convert, port or accelerate a Hugging Face PyTorch model (Llama, Gemma,
  Mistral, Mixtral) with Transformer Engine (TE), or to write a TE fine-tuning recipe. Produces a
  te_llama.py-style module built on fused TE blocks (LayerNorm+QKV, LayerNorm+MLP,
  TransformerLayer, GroupedLinear MoE experts) with the checkpoint weight mapping, for BF16, FP8
  or MXFP8. Not for Megatron-LM, JAX or inference serving.
license: Apache-2.0
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
compatibility: >-
  Writing a recipe needs no GPU. Running one needs transformer_engine[pytorch], transformers and
  an NVIDIA GPU (FP8 needs Hopper or newer; MXFP8 needs Blackwell).
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
    - mixture-of-experts
---

# Convert a Hugging Face model to a Transformer Engine recipe

## Purpose

A *recipe* is one Python module, named after the model family (`te_llama.py`, `te_gemma.py`,
`te_mixtral.py`), that rebuilds a Hugging Face (HF) model on Transformer Engine (TE) blocks and
loads the original checkpoint into it. TE publishes three recipes as tutorials; this skill
produces the same kind of file for the model the user names.

TE's speed comes from *fused* blocks, which run the normalization, the projection and the
quantization as one unit. Swapping `nn.Linear` for `te.Linear` one at a time fuses nothing, so
move as much of each layer as possible into fused blocks without changing the model's math.

In this skill `te` means `transformer_engine.pytorch`.

## When to Use This Skill

- "I want a recipe for Llama", "convert Gemma to Transformer Engine", "port Mixtral to TE".
- "I want a fine-tuning recipe for this model with TE", "speed up my fine-tune with FP8".
- "How do I fuse layer norm and QKV?"

Do not use it for Megatron-LM or NeMo configuration, JAX models, inference serving (vLLM,
TensorRT-LLM), or installing TE.

## Examples

> "I want a fine-tuning recipe for Llama 2 7B with TE."

Output: `te_llama.py` with a `TELlamaDecoderLayer` based on `te.TransformerLayer`, a loader,
`replace_params`, and a training step under `te.autocast`.

> "Convert Mixtral 8x7B to TE."

Output: `te_mixtral.py` with `te.MultiheadAttention` (input norm fused into QKV) and the
experts run through `GroupedLinear`.

## Prerequisites

- Writing a recipe needs only the model's `config.json` and its layer code in `transformers`.
- Running one needs `transformer_engine[pytorch]`, `transformers`, an NVIDIA GPU and a local
  checkpoint. Access to gated models is the user's.

## Instructions

### 1. Read the architecture

Read the model's `config.json` and the layer's `forward` in the installed `transformers` source.
The recipe depends on:

- the norm: LayerNorm or RMSNorm, its epsilon, and whether it runs before each sublayer;
- the MLP: its activation, and whether it is gated (`down(act(gate(x)) * up(x))`);
- attention: heads, key/value heads, head size, RoPE base;
- which projections have a bias;
- for mixture-of-experts (MoE) models: the number of experts and the top-k.

If neither is at hand, ask for the local checkpoint path. Failing that, start from the closest
published recipe, read every size from `config` at run time, and say in the report what was
assumed. Do not make up config values.

### 2. Choose the deepest conversion that keeps the math identical

| Layer | Recipe | What gets fused |
|---|---|---|
| Pre-norm dense layer: `x + attn(norm(x))`, then `x + mlp(norm(x))` | Subclass `te.TransformerLayer` | Norm + QKV (`layernorm_qkv`) and norm + MLP (`layernorm_mlp`) |
| MoE layer: router plus experts | `te.MultiheadAttention(input_layernorm=True)`, a TE norm, `GroupedLinear` experts | Norm + QKV; all experts in one grouped GEMM |
| Anything else | Swap the sublayers that match for `te.MultiheadAttention`, `te.LayerNormMLP` or `te.LayerNormLinear` | Whatever matches |

Replacing single `nn.Linear` modules fuses nothing and is the last resort. If TE cannot express
part of the model, leave that part native instead of changing the model.

Published recipes to start from:

| Model | Layer | Recipe |
|---|---|---|
| Llama, Mistral | RMSNorm, SwiGLU, grouped-query attention | <https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/te_llama/te_llama.py> |
| Gemma | RMSNorm with `zero_centered_gamma=True`, GeGLU, `kv_channels=config.head_dim` | <https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/te_gemma/te_gemma.py> |
| Mixtral | MoE with SwiGLU experts | <https://github.com/NVIDIA/TransformerEngine/tree/main/docs/examples/te_mixtral> |

### 3. Write the recipe

Name the file `te_` plus the model family and keep the names of the published recipes. A dense
recipe has four parts: a context manager that swaps the HF layer class, the TE layer, the model
class with a checkpoint loader, and `replace_params` (step 4).

```python
@contextmanager
def replace_decoder(te_decoder_cls):
    """Make HF build the model with the TE layer class in place of its own."""
    modeling = transformers.models.llama.modeling_llama
    original = modeling.LlamaDecoderLayer
    modeling.LlamaDecoderLayer = te_decoder_cls
    try:
        yield
    finally:
        modeling.LlamaDecoderLayer = original


class TELlamaDecoderLayer(te.TransformerLayer):
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
            fuse_qkv_params=False,
            normalization="RMSNorm",
            activation="swiglu",
            attn_input_format="bshd",
            layer_number=layer_idx + 1,  # TE counts layers from 1
        )
        rope = RotaryPositionEmbedding(
            config.hidden_size // config.num_attention_heads, rotary_base=config.rope_theta
        )
        self.te_rope_emb = rope(max_seq_len=config.max_position_embeddings).cuda()

    def forward(self, hidden_states, *args, **kwargs):
        # HF's mask, position ids and caches are not forwarded; TE builds the causal mask.
        return super().forward(hidden_states, rotary_pos_emb=self.te_rope_emb)


class TELlamaForCausalLM:
    def __new__(cls, config):
        with replace_decoder(TELlamaDecoderLayer):
            return LlamaForCausalLM(config)

    @classmethod
    def from_pretrained_local(cls, model_dir, *, config, torch_dtype=torch.bfloat16):
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch_dtype)  # HF and TE modules use it at construction
        try:
            model = cls(config)
        finally:
            torch.set_default_dtype(previous)
        te_state_dict = model.state_dict()
        shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        if not shards:
            raise FileNotFoundError(f"No .safetensors files in {model_dir}")
        for shard in shards:
            state_dict = safetensors.torch.load_file(shard)
            replace_params(state_dict, te_state_dict, config)  # the replaced layers
            model.load_state_dict(state_dict, strict=False)  # embeddings, final norm, LM head
        return model
```

Imports are left out; `RotaryPositionEmbedding` is in `transformer_engine.pytorch.attention`.

For an MoE layer, compose the layer yourself: the attention sublayer is
`te.MultiheadAttention(input_layernorm=True, normalization="RMSNorm", fuse_qkv_params=True,
num_gqa_groups=..., qkv_format="bshd")`, followed by a `te.RMSNorm` and the MoE block below.
`TransformerLayer` does not fit because the second norm feeds both the router and the experts.

```python
# __init__: the router stays out of quantization; gate and up share one weight per expert.
with te.quantized_model_init(enabled=False):
    self.gate = te.Linear(hidden, num_experts, bias=False)
self.experts_gate_up = GroupedLinear(
    num_groups=num_experts, in_features=hidden, out_features=2 * ffn, bias=False, device="cuda"
)
self.experts_down = GroupedLinear(
    num_groups=num_experts, in_features=ffn, out_features=hidden, bias=False, device="cuda"
)
self.experts_ffn = Sequential(self.experts_gate_up, SwiGLU(), self.experts_down)

# forward: tokens is [N, hidden]
with te.autocast(enabled=False):
    probs = torch.softmax(self.gate(tokens), dim=-1, dtype=torch.float32)
weights, experts = torch.topk(probs, top_k, dim=-1)
weights = weights / weights.sum(dim=-1, keepdim=True)
split_sizes = torch.bincount(experts.reshape(-1), minlength=num_experts).int()
permuted, row_id_map = te.moe_permute(  # sort tokens by expert
    tokens, experts.to(torch.int32), num_out_tokens=experts.numel(), map_type="index"
)
expert_out = self.experts_ffn(permuted, split_sizes, split_sizes)  # one grouped GEMM each
out = te.moe_unpermute(  # restore token order and apply the routing weights
    expert_out, row_id_map, merging_probs=weights, restore_shape=tokens.shape, map_type="index"
)
```

`GroupedLinear`, `Sequential` and `SwiGLU` come from `transformer_engine.pytorch.ops`. Expert
parallelism and the fused MXFP8 grouped MLP are opt-in extras shown in the Mixtral recipe.

### 4. Map the weights

Every HF parameter of a replaced layer needs a destination and every TE parameter a source.
Keys are relative to `model.layers.N.`:

```python
def replace_params(hf_state_dict, te_state_dict, config):
    """Copy the replaced layers' weights from an HF state dict (or one shard of it)."""
    ffn = config.intermediate_size
    mapping = {  # HF name: (TE name, rows)
        "input_layernorm.weight": ("self_attention.layernorm_qkv.layer_norm_weight", slice(None)),
        "self_attn.q_proj.weight": ("self_attention.layernorm_qkv.query_weight", slice(None)),
        "self_attn.k_proj.weight": ("self_attention.layernorm_qkv.key_weight", slice(None)),
        "self_attn.v_proj.weight": ("self_attention.layernorm_qkv.value_weight", slice(None)),
        "self_attn.o_proj.weight": ("self_attention.proj.weight", slice(None)),
        "post_attention_layernorm.weight": ("layernorm_mlp.layer_norm_weight", slice(None)),
        "mlp.gate_proj.weight": ("layernorm_mlp.fc1_weight", slice(None, ffn)),
        "mlp.up_proj.weight": ("layernorm_mlp.fc1_weight", slice(ffn, None)),
        "mlp.down_proj.weight": ("layernorm_mlp.fc2_weight", slice(None)),
    }
    for hf_key, value in hf_state_dict.items():
        match = re.match(r"(model\.layers\.\d+\.)(.+)", hf_key)
        if match:
            te_name, rows = mapping[match.group(2)]  # KeyError: an HF weight with no destination
            target = te_state_dict[match.group(1) + te_name][rows]
            with torch.no_grad():
                target.copy_(value.to(device=target.device, dtype=target.dtype))
```

- **Fused QKV.** With `fuse_qkv_params=True` the three weights become one,
  `layernorm_qkv.weight`, interleaved per key/value head: that group's query heads, then its key
  head, then its value head, each `head_dim` rows. It saves a concatenation per step and is what
  the Gemma and Mixtral recipes use.
- **MoE.** The router gate copies to `mlp.gate.weight`. Each expert has two weights: gate and up
  are the two halves of `experts_gate_up.weight{i}` and down is `experts_down.weight{i}`. Older
  Mixtral checkpoints name them `w1` (gate), `w3` (up) and `w2` (down).

### 5. Check the weight mapping

Where the recipe can run (a GPU with TE, and the local checkpoint), run:

```bash
python scripts/check_weight_mapping.py te_llama.py /path/to/checkpoint
```

The script loads the recipe's model and looks for every checkpoint weight inside the TE
weights, row by row, without using the recipe's mapping. It fails on a weight that is missing,
partial, overwritten, in the wrong layer or changed, and on a TE parameter that does not come
from the checkpoint. Fix every failure.

It cannot tell two same-shaped weights apart, so read the table it prints and confirm the
order: gate before up in `fc1_weight`, and query, key and value in their own slots. Without a
GPU, report that this check was not run.

### 6. Add the precision hook (fine-tuning requests only)

Verify the model in BF16 first, build the optimizer after the conversion, then turn on a recipe:

```python
from transformer_engine.common import recipe

if te.is_mxfp8_available():  # Blackwell
    fp8_recipe = recipe.MXFP8BlockScaling()
elif te.is_fp8_available():  # Hopper and newer
    fp8_recipe = recipe.DelayedScaling()
else:
    fp8_recipe = None  # stay in BF16

with te.autocast(enabled=fp8_recipe is not None, recipe=fp8_recipe):
    loss = model(input_ids=input_ids, labels=labels).loss
loss.backward()  # outside autocast
```

Keep the MoE router and the LM head in high precision.

### 7. Check and report

Confirm that the recipe uses a fused block and that the constructor arguments match the config.

With a GPU, once the weight mapping check passes, load the checkpoint into the HF and TE models
and compare logits in BF16 under `te.autocast(enabled=False)`. Expect a small relative error
and near-total top-1 agreement; an error near 1.0 means a wrong order or constructor argument.

Tell the user which techniques were applied, what stayed native and why, and which checks ran
or could not. Describe a recipe that was never executed as such.

## Rules that prevent silent errors

- **RoPE base.** `RotaryPositionEmbedding` defaults to base 10000; pass
  `rotary_base=config.rope_theta`. With `rope_scaling`, copy `inv_freq` from the HF model.
- **Masks.** Do not forward HF's additive mask; TE masks are boolean with `True` meaning masked
  out. TE's causal mask is exact for unpadded and right-padded batches. Left-padded batches need
  a padding mask or packed (`thd`) input.
- **Layer return type.** Return a tensor or a tuple, whichever the installed `transformers`
  model loop expects.
- **KV cache.** TE layers ignore HF's `past_key_values`; set `config.use_cache = False`.
- **Order.** Convert before building the optimizer, wrapping with Accelerate or PEFT, or
  compiling.

## Available Scripts

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/check_weight_mapping.py` | Check that every checkpoint weight is in the recipe's TE model exactly, and that every TE weight comes from the checkpoint | `recipe`, `model_dir`, `--model-class`, `--ignore` |

Run it as in step 5, or with
`run_script("scripts/check_weight_mapping.py", args=[recipe, model_dir])`.

## Limitations

- A recipe written without a GPU is unverified until the logits comparison passes.
- Whole-layer replacement covers pre-norm layers with two norms; other layers get partial
  conversion.
- `generate()` with HF's KV cache does not work on the default recipe; the Gemma recipe shows
  generation with TE's own cache.
- PyTorch only.

## Troubleshooting

| Error / Symptom | Cause | Solution |
|---|---|---|
| `check_weight_mapping.py` fails | A weight is missing, overwritten, in the wrong layer or changed | Compare the HF `state_dict()` keys of one layer with the mapping |
| Logits differ widely | Wrong order or constructor argument | Check gate/up and Q/K/V order, RoPE base, `zero_centered_gamma`, `kv_channels` |
| Shape error copying Q/K/V | The config sets `head_dim` | Pass it as `kv_channels` |
| FP8 or MXFP8 unavailable | GPU or TE build lacks it | Report `te.is_fp8_available(return_reason=True)` |

