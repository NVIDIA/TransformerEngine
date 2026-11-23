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
  version: "0.2.0"
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
- "Convert this Hugging Face model to TE", for a model no tutorial covers.
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

> "Convert Qwen3.5-35B-A3B to TE."

Output: `te_qwen3_5_moe.py`. The input norm of each mixer is fused with the projections that
read it (`te.LayerNormLinear`), the mixer cores run on TE cores where the comparison passes and
stay HF code otherwise, and the experts run through `GroupedLinear`.

## Prerequisites

- Writing a recipe needs only the model's `config.json`, the weight names of its checkpoint and
  its layer code in `transformers`.
- Running one needs `transformer_engine[pytorch]`, `transformers`, an NVIDIA GPU and a local
  checkpoint. Access to gated models is the user's.

## Instructions

### 1. Read the architecture

Read the model's `config.json`, the weight names in the checkpoint and the layer's `forward` in
the installed `transformers` source. The recipe depends on:

- the layer types: one for all layers, or a list (`layer_types`) that mixes attention with
  other mixers;
- the norm: LayerNorm or RMSNorm, its epsilon, whether it runs before each sublayer, and
  whether it multiplies by `weight` or by `1 + weight`;
- the MLP: its activation, and whether it is gated (`down(act(gate(x)) * up(x))`);
- attention: heads, key/value heads, head size, RoPE base, and anything beyond plain softmax
  attention (norms on Q and K, an output gate, RoPE on part of the head, 3-D positions);
- which projections have a bias;
- for mixture-of-experts (MoE) models: the number of experts, the top-k, how the routing
  weights are computed, and any shared expert that every token goes through.

Take the weight names from the checkpoint files (`model.safetensors.index.json`, or the shard
headers), not from `model.state_dict()`. The recipe's loader reads the files directly, and
`transformers` may rename, stack or split weights when it loads and saves: a checkpoint can hold
all experts of a layer in one `[num_experts, out, in]` tensor while `save_pretrained` writes one
entry per expert. Note the prefix of the decoder layers (`model.layers.N.`, or deeper, such as
`model.language_model.layers.N.`), which config holds the sizes (`config`, or a nested
`config.text_config`), and the entries HF itself skips: the class's
`_keys_to_ignore_on_load_unexpected`, and buffers of old checkpoints such as the per-layer
`rotary_emb.inv_freq` of Llama 2, which HF drops for every model.

If these are not at hand, ask for the local checkpoint path. Failing that, start from the closest
published recipe, read every size from `config` at run time, and say in the report what was
assumed. Do not make up config values.

### 2. Choose the deepest conversion that keeps the math identical

| Layer | Recipe | What gets fused |
|---|---|---|
| Pre-norm dense layer: `x + attn(norm(x))`, then `x + mlp(norm(x))` | Subclass `te.TransformerLayer` | Norm + QKV (`layernorm_qkv`) and norm + MLP (`layernorm_mlp`) |
| MoE layer: router plus experts | `te.MultiheadAttention(input_layernorm=True)`, a TE norm, `GroupedLinear` experts | Norm + QKV; the experts in grouped GEMMs |
| A mixer those two cannot express: attention with extra norms or gates, linear attention, state space | Replace its input norm and *all* the projections that read it with one `te.LayerNormLinear`, its output projection with `te.Linear`, and its core with a TE core if one matches (step 3); otherwise the core stays native | Norm + every input projection |
| Anything else | Swap the sublayers that match for `te.MultiheadAttention`, `te.LayerNormMLP` or `te.LayerNormLinear` | Whatever matches |

Choose per layer type: a model that mixes layer types gets one composed layer class that builds
the mixer its `layer_types` entry names. Replacing single `nn.Linear` modules fuses nothing and
is the last resort. If TE cannot express part of the model, leave that part native instead of
changing the model.

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
        # transformers 5 keeps the RoPE base in config.rope_parameters
        rope_theta = getattr(config, "rope_theta", None) or config.rope_parameters["rope_theta"]
        # The head size is config.head_dim where the config has one (Gemma)
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        rope = RotaryPositionEmbedding(head_dim, rotary_base=rope_theta)
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
        torch.set_default_dtype(torch_dtype)  # TE modules use it at construction
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
Keep the signature of `from_pretrained_local`: the scripts call it. Build the HF class named in
the checkpoint's `architectures`, so the weight names outside the decoder layers stay unchanged.
A model made of sub-models (`config.text_config`, `config.vision_config`) builds each of them in
its config's `dtype`, not the default dtype: set `config.dtype` and every sub-config's `dtype`
to `torch_dtype` before building, or the HF parts and the TE layers get different dtypes.

**A mixer `te.TransformerLayer` and `te.MultiheadAttention` cannot express.** Subclass the HF
mixer, delete its input projections and reuse its `forward` arithmetic. One fused block computes
the norm and every projection; its weight is the HF projections stacked row-wise, in the order
of the split:

```python
# __init__: sizes holds the output size of each HF projection, in a fixed order
self.norm_proj = te.LayerNormLinear(
    config.hidden_size, sum(sizes), eps=config.rms_norm_eps, bias=False,
    normalization="RMSNorm", zero_centered_gamma=True,  # True when HF multiplies by 1 + weight
)
self.out_proj = te.Linear(inner_size, config.hidden_size, bias=False)

# forward: the layer no longer applies the input norm itself
query, key, value = torch.split(self.norm_proj(hidden_states), sizes, dim=-1)
```

Then look for a TE core for what is left between the projections, and keep it only if step 7
passes with it; otherwise keep HF's code for the core:

- softmax attention, after the model's own Q/K norms and RoPE:
  `te.DotProductAttention(num_attention_heads, kv_channels, num_gqa_groups=..., qkv_format="bshd",
  attn_mask_type="causal")(query, key, value)`;
- a gated delta rule (linear attention): the experimental
  `te.GatedDeltaNetAttention(num_value_heads, (key_dim, value_dim), qkv_format="bshd")(query,
  key, value, g=g, beta=beta, use_qk_l2norm_in_kernel=...)`, with Q, K and V in BF16, `g` and
  `beta` in float32, and Q and K repeated to the number of value heads. Related cores are
  exported next to it.

**An MoE layer.** Compose the layer yourself: the attention sublayer is
`te.MultiheadAttention(input_layernorm=True, normalization="RMSNorm", fuse_qkv_params=True,
num_gqa_groups=..., qkv_format="bshd", attn_mask_type="causal", bias=False,
attention_dropout=config.attention_dropout, layernorm_epsilon=config.rms_norm_eps)` (or the
composed mixer above), called with `rotary_pos_emb=` as in the dense layer and followed by a
`te.RMSNorm` and the MoE block below. Pass the dropout and the bias: `te.MultiheadAttention`
defaults to `attention_dropout=0.1` and `bias=True`, and the checks, which run in eval mode,
cannot see a wrong dropout. `TransformerLayer` does not fit because the second norm feeds both
the router and the experts.

```python
MAX_GROUPS = 64  # TE's grouped GEMM takes at most 64 weights per call

# __init__: the router stays out of quantization; gate and up share one weight per expert.
with te.quantized_model_init(enabled=False):
    self.gate = te.Linear(hidden, num_experts, bias=False)
blocks = [min(MAX_GROUPS, num_experts - i) for i in range(0, num_experts, MAX_GROUPS)]
self.experts_gate_up = torch.nn.ModuleList(
    GroupedLinear(num_groups=n, in_features=hidden, out_features=2 * ffn, bias=False, device="cuda")
    for n in blocks
)
self.experts_down = torch.nn.ModuleList(
    GroupedLinear(num_groups=n, in_features=ffn, out_features=hidden, bias=False, device="cuda")
    for n in blocks
)
# A plain list: registering the Sequentials would list every expert weight twice.
object.__setattr__(self, "experts_ffn", [
    Sequential(gate_up, ScaledSwiGLU(), down)
    for gate_up, down in zip(self.experts_gate_up, self.experts_down)
])

# forward: tokens is [N, hidden]
with te.autocast(enabled=False):
    probs = torch.softmax(self.gate(tokens), dim=-1, dtype=torch.float32)
weights, experts = torch.topk(probs, top_k, dim=-1)
weights = weights / weights.sum(dim=-1, keepdim=True)
routing_map = torch.zeros_like(probs, dtype=torch.int32).scatter_(1, experts, 1)
routing_weights = torch.zeros_like(probs).scatter_(1, experts, weights)
# Quantized grouped GEMMs need each expert's row count to be a multiple of 128.
align = 128 if FP8GlobalStateManager.is_fp8_enabled() else 1
permuted, permuted_weights, row_id_map, pad_offsets, split_sizes = (
    te.moe_permute_and_pad_with_probs(  # sort tokens by expert, pad each expert's rows
        tokens, routing_weights, routing_map, routing_map.sum(0), align
    )
)
expert_out, start = [], 0
for ffn, sizes in zip(self.experts_ffn, torch.split(split_sizes.int(), MAX_GROUPS)):
    rows = int(sizes.sum())
    if rows:  # a block none of whose experts got a token is skipped
        chunk = slice(start, start + rows)
        expert_out.append(ffn(permuted[chunk], sizes, permuted_weights[chunk], sizes))
    start += rows
out = te.moe_unpermute(  # drop the padding, restore token order, sum each token's experts
    torch.cat(expert_out), row_id_map, restore_shape=tokens.shape, map_type="mask",
    pad_offsets=pad_offsets,
)
```

`GroupedLinear`, `Sequential`, `ScaledSwiGLU`, `SwiGLU` and `BasicLinear` come from
`transformer_engine.pytorch.ops`, `FP8GlobalStateManager` from
`transformer_engine.pytorch.quantization`. Three constraints shaped this block:

- A `GroupedLinear` holds at most 64 experts, so more experts are split into blocks.
- `ScaledSwiGLU` multiplies by the routing weight inside the activation, which equals HF's
  weighting of the expert output when the down projection has no bias. With plain `SwiGLU`
  between two `GroupedLinear` the backward pass fails under MXFP8 and FP8 current scaling.
- Under a quantized recipe the rows of each expert are padded to a multiple of 128; MXFP8
  rejects anything else.

A shared expert that every token goes through is a dense gated MLP on the same normalized
input: `Sequential(BasicLinear(hidden, 2 * ffn), SwiGLU(), BasicLinear(ffn, hidden))`, with its
gate (if any) built like the router. Expert parallelism and the fused MXFP8 grouped MLP are
opt-in extras shown in the Mixtral recipe.

### 4. Map the weights

Every HF parameter of a replaced layer needs a destination and every TE parameter a source.
Keys are relative to the layer prefix found in step 1, here `model.layers.N.`:

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
        if match and not hf_key.endswith("rotary_emb.inv_freq"):  # an old buffer HF drops too
            te_name, rows = mapping[match.group(2)]  # KeyError: an HF weight with no destination
            target = te_state_dict[match.group(1) + te_name][rows]
            with torch.no_grad():
                target.copy_(value.to(device=target.device, dtype=target.dtype))
```

- **Fused QKV.** With `fuse_qkv_params=True` the three weights become one,
  `layernorm_qkv.weight`, interleaved per key/value head: that group's query heads, then its key
  head, then its value head, each `head_dim` rows. It saves a concatenation per step and is what
  the Gemma and Mixtral recipes use. With `q`, `k`, `v` the HF weights:

  ```python
  per_group = num_heads // num_kv_heads
  view = te_state_dict[prefix + "self_attention.layernorm_qkv.weight"].view(
      num_kv_heads, per_group + 2, head_dim, -1
  )
  view[:, :per_group] = q.view(num_kv_heads, per_group, head_dim, -1)
  view[:, per_group] = k.view(num_kv_heads, head_dim, -1)
  view[:, per_group + 1] = v.view(num_kv_heads, head_dim, -1)
  ```

  Write through a view or assign with `tensor[rows] = value`. `tensor[rows].copy_(value)` with
  a list or tensor of row indices copies into a temporary and changes nothing.
- **Stacked projections.** Each HF projection is a row range of the `te.LayerNormLinear` weight,
  in the order of the split in `forward`; the layer's input norm weight goes to its
  `layer_norm_weight`.
- **MoE.** The router gate copies to `mlp.gate.weight`. Each expert has two weights: gate and up
  are the two halves of its gate-up weight and down is its down weight; expert `i` is
  `weight{i % 64}` of block `i // 64`. A stacked `[num_experts, out, in]` tensor is one weight
  per expert along its first dimension. Mixtral checkpoints name the three
  `block_sparse_moe.experts.E.w1` (gate), `w3` (up) and `w2` (down), and the router
  `block_sparse_moe.gate`.
- **No silent leftovers.** `strict=False` hides a checkpoint weight that went nowhere and a TE
  weight nothing wrote; step 5 catches both. For a model with entries the tutorials do not have
  (extra heads, stacked experts), also make the loader raise on them, apart from the entries HF
  skips. Map the layout of the checkpoint you were given and say in the report which layout
  that is.

### 5. Check the weight mapping

Where the recipe can run (a GPU with TE, and the local checkpoint), run:

```bash
python scripts/check_weight_mapping.py te_llama.py /path/to/checkpoint
```

The script loads the recipe's model and looks for every checkpoint weight inside the TE
weights, row by row, without using the recipe's mapping. It fails on a weight that is missing,
partial, overwritten, in the wrong layer, reordered or changed, and on a TE parameter that does
not come from the checkpoint. Fix every failure.

It cannot tell two same-shaped weights apart: gate and up, key and value or two norms stored in
each other's place still pass. Read the table it prints, and treat a pass as necessary, not
sufficient: step 7 settles the order. Without a GPU, report that this check was not run.

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

With a GPU, compare the TE model with the HF model on the checkpoint:

```bash
python scripts/compare_logits.py te_llama.py /path/to/checkpoint --train-step auto
```

It runs both models on the same input in BF16 and also runs HF in float32. The TE model passes
when it is no further from the float32 result than HF's own BF16 rounding puts HF; a wrong
order or constructor argument puts it tens of times further. Judge by that ratio: top-1
agreement can stay high on a few layers even with a wrong order.
`--train-step auto` also runs one forward and backward pass under `te.autocast` with the recipe
step 6 selects. A recipe that matches in BF16 can still fail there, so run it whenever the
recipe has an MoE block or is meant for fine-tuning. `--precision fp32` compares the two models
in float32 instead, a much tighter test for a recipe whose blocks all support float32.

**A model that does not fit on the GPU** is checked a few layers at a time; the hidden states
of one slice are the input of the next, so the last slice gives the logits of the whole model.
The comparison needs room for the HF model in float32, twice its BF16 size, so a model whose
TE form fits may still need slices:

```bash
python scripts/compare_logits.py te_big.py /path/to/checkpoint \
    --layers-per-slice 4 --check-weights --train-step auto
```

`--check-weights` runs step 5 on every slice, which replaces the whole-model run of
`check_weight_mapping.py`, and the train step runs on every slice too. The weight check costs
about half a minute per 13 GB of weights; leave it out of a first pass if time is short and run
it afterwards. To iterate quickly on real weights, cut a few layers that cover every layer type
out once with `scripts/make_subcheckpoint.py` and run both scripts on that directory. A slice
needs GPU memory for about 2.5 times its BF16 size: the HF model in float32, or the train
step's gradients. The script prints the peak it used. Each run keeps its slices and hidden
states in a directory of its own; pass `--work-dir` to keep them, which `--slices` needs to
resume.

Tell the user which techniques were applied, what stayed native and why, and which checks ran
or could not. Describe a recipe that was never executed as such.

## Rules that prevent silent errors

- **RoPE base.** `RotaryPositionEmbedding` defaults to base 10000; pass the model's base as
  `rotary_base`: `config.rope_theta`, or `config.rope_parameters["rope_theta"]` in
  transformers 5. With `rope_scaling`, copy `inv_freq` from the HF model. A recipe that keeps
  HF's rotary module needs none of this.
- **Masks.** Do not forward HF's additive mask to a TE attention block; TE masks are boolean
  with `True` meaning masked out. TE's causal mask is exact for unpadded and right-padded
  batches. Left-padded batches need a padding mask or packed (`thd`) input. A mixer that stays
  native keeps HF's mask. A mask HF applies between the norm and bias-free projections can be
  applied to the output of the fused block instead.
- **Layer return type.** Return a tensor or a tuple, whichever the installed `transformers`
  model loop expects.
- **KV cache.** TE layers ignore HF's `past_key_values`; set `config.use_cache = False`.
- **Order.** Convert before building the optimizer, wrapping with Accelerate or PEFT, or
  compiling.
- **Grouped GEMM library bug.** cuBLAS 13.7.0 to 13.8.0 intermittently returns wrong
  grouped-GEMM results, different from run to run, so `GroupedLinear` experts come out wrong or
  NaN. TE warns once ("has a grouped GEMM bug that may cause data corruption"). Do not train or
  trust a comparison with that warning on screen: use cuBLAS 13.8.1 or newer, for example
  `pip install --no-deps --target DIR "nvidia-cublas>=13.8.1"` with `DIR/nvidia/cu13/lib` first in
  `LD_LIBRARY_PATH`. Make an MoE recipe refuse to build on those versions instead of returning
  garbage: with `import transformer_engine_torch as tex` (after `transformer_engine.pytorch`),
  raise when `130700 <= tex.get_cublasLt_version() < 130801`.
- **Real weights.** A small random model exercises the code, not the mapping of the real
  checkpoint or its expert count. Run steps 5 and 7 on the real checkpoint, or on slices of it.
- **Initial values.** HF's `post_init()` re-initializes modules by class name and overwrites TE
  weights (a `te.LayerNormLinear` weight becomes all ones). A recipe is meant to be filled from
  a checkpoint; do not train one built from a config alone.

## Available Scripts

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/check_weight_mapping.py` | Check that every checkpoint weight is in the recipe's TE model exactly and in order, and that every TE weight comes from the checkpoint | `recipe`, `model_dir`, `--model-class`, `--ignore`, `--allow-reorder` |
| `scripts/compare_logits.py` | Compare the TE model with the HF model, whole or in slices of layers, against HF's own rounding noise; optionally with a low-precision train step and the weight check on every slice | `recipe`, `model_dir`, `--layers-per-slice`, `--slices`, `--work-dir`, `--train-step`, `--check-weights`, `--ignore`, `--precision`, `--no-reference`, `--batch`, `--seq`, `--model-class` |
| `scripts/make_subcheckpoint.py` | Cut a range of decoder layers out of a checkpoint into a directory both models can load, without the entries HF skips | `model_dir`, `out_dir`, `--layers`, `--prefix`, `--drop`, `--keep-all` |

Run them as in steps 5 and 7, or with
`run_script("scripts/check_weight_mapping.py", args=[recipe, model_dir])`.

## Limitations

- A recipe written without a GPU is unverified until the logits comparison passes.
- Whole-layer replacement covers pre-norm layers with two norms; other layers get partial
  conversion.
- `generate()` with HF's KV cache does not work on the default recipe; the Gemma recipe shows
  generation with TE's own cache.
- The checks compare text-only inputs; other input towers keep their HF code and weights.
- A recipe runs on one GPU. A model larger than that is checked in slices here; training it
  needs the parallelism of a framework such as Megatron-LM.
- HF records per-layer outputs by class. With the layer class replaced,
  `output_hidden_states=True` returns nothing unless the TE layer subclasses the HF layer; with
  the router replaced, `output_router_logits=True` fails and HF's MoE load-balancing loss is
  gone. Add that loss in the recipe if the fine-tune needs it.
- PyTorch only.

## Troubleshooting

| Error / Symptom | Cause | Solution |
|---|---|---|
| `check_weight_mapping.py` fails | A weight is missing, overwritten, reordered, in the wrong layer or changed | Compare the checkpoint's weight names for one layer with the mapping |
| Logits differ widely | Wrong order or constructor argument | Check gate/up and Q/K/V order, RoPE base, `zero_centered_gamma`, `kv_channels` |
| Logits differ in some layers only or from run to run, or are NaN | cuBLAS grouped GEMM bug | See "Grouped GEMM library bug" |
| Shape error copying Q/K/V | The config sets `head_dim` | Pass it as `kv_channels` |
| `Grouped GEMM: A_list supports up to 64 tensors per kernel` | More than 64 experts in one `GroupedLinear` | Split the experts into blocks of at most 64 |
| `First logical dimension of a grouped tensor must be divisible by 128` | Unpadded expert rows under MXFP8 | Pad with `te.moe_permute_and_pad_with_probs` |
| `'MXFP8TensorStorage' object has no attribute 'reshape'` in backward | Plain `SwiGLU` between two `GroupedLinear` under a quantized recipe | Use `ScaledSwiGLU` with the routing weights |
| `KeyError` in `replace_params` | A checkpoint weight has no destination, or the checkpoint uses another layout | Read the names from the checkpoint files and map that layout |
| Out of memory loading the model | The model does not fit on the GPU | Check it in slices (step 7) |
| FP8 or MXFP8 unavailable | GPU or TE build lacks it | Report `te.is_fp8_available(return_reason=True)` |
