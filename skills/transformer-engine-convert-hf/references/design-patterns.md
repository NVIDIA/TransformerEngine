# Design patterns and weight mapping

Code for the patterns `SKILL.md` names. `te` is `transformer_engine.pytorch`; every size comes
from `config`. Complete, tested versions are in the tutorials:
`docs/examples/te_llama/te_llama.py`, `docs/examples/te_gemma/te_gemma.py` and
`docs/examples/te_mixtral/`.

## Swap the HF layer class

HF builds its model with the TE layer class in place of its own, so everything outside the
decoder layers keeps HF's code and weight names.

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


class TELlamaForCausalLM:
    def __new__(cls, config):
        with replace_decoder(TELlamaDecoderLayer):
            return LlamaForCausalLM(config)
```

## Level 1: the whole block

```python
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
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        rope = RotaryPositionEmbedding(head_dim, rotary_base=rope_theta)
        self.te_rope_emb = rope(max_seq_len=config.max_position_embeddings).cuda()

    def forward(self, hidden_states, *args, **kwargs):
        # HF's mask, position ids and caches are not forwarded; TE builds the causal mask.
        return super().forward(hidden_states, rotary_pos_emb=self.te_rope_emb)
```

For Gemma: `normalization="RMSNorm"` with `zero_centered_gamma=True`, `activation="geglu"` and
`kv_channels=config.head_dim`.

## Level 2: attention from a fused projection and a TE core

Subclass the HF attention to keep its arithmetic (Q/K norms, partial RoPE, output gate), delete
its projections, and fuse the input norm with all of them. Example: Qwen3.5's gated attention,
whose `q_proj` holds the query and the output gate of every head.

```python
class TEGatedAttention(modeling.Qwen3_5MoeAttention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        del self.q_proj, self.k_proj, self.v_proj, self.o_proj
        heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
        # Output sizes of q_proj (query and gate), k_proj and v_proj, in this order
        self.proj_sizes = [2 * heads * self.head_dim, kv_heads * self.head_dim, kv_heads * self.head_dim]
        self.norm_proj = te.LayerNormLinear(
            config.hidden_size, sum(self.proj_sizes), eps=config.rms_norm_eps, bias=False,
            normalization="RMSNorm", zero_centered_gamma=True,  # HF multiplies by 1 + weight
        )
        self.o_proj = te.Linear(heads * self.head_dim, config.hidden_size, bias=False)
        self.core = te.DotProductAttention(
            heads, self.head_dim, num_gqa_groups=kv_heads, qkv_format="bshd",
            attn_mask_type="causal", attention_dropout=config.attention_dropout,
        )

    def forward(self, hidden_states, position_embeddings, attention_mask=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        # The layer no longer applies its input norm: norm_proj does.
        query_gate, key, value = torch.split(self.norm_proj(hidden_states), self.proj_sizes, dim=-1)
        query, gate = torch.chunk(query_gate.reshape(*input_shape, -1, 2 * self.head_dim), 2, dim=-1)
        query = self.q_norm(query.reshape(hidden_shape))
        key = self.k_norm(key.reshape(hidden_shape))
        cos, sin = position_embeddings
        query, key = modeling.apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=2)
        out = self.core(query, key, value.reshape(hidden_shape))  # TE builds the causal mask
        return self.o_proj(out.reshape(*input_shape, -1) * torch.sigmoid(gate.reshape(*input_shape, -1)))
```

A Gated DeltaNet layer follows the same pattern: one `te.LayerNormLinear` for its Q/K/V, Z, B
and A projections, HF's causal convolution and decay arithmetic, then

```python
self.core = te.GatedDeltaNetAttention(num_v_heads, (head_k_dim, head_v_dim), qkv_format="bshd")
out = self.core(query, key, value, g=g, beta=beta.float(), use_qk_l2norm_in_kernel=True)
```

with Q and K repeated to the number of value heads (`repeat_interleave`), Q/K/V contiguous in
BF16, and `g` and `beta` in float32.

## Level 2: an MoE layer around grouped GEMMs

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
permuted, permuted_weights, row_id_map, pad_offsets, split_sizes = te.moe_permute_and_pad_with_probs(
    tokens, routing_weights, routing_map, routing_map.sum(0), align
)
expert_out, start = [], 0
for ffn, sizes in zip(self.experts_ffn, torch.split(split_sizes.int(), MAX_GROUPS)):
    rows = int(sizes.sum())
    if rows:  # a block none of whose experts got a token is skipped
        chunk = slice(start, start + rows)
        expert_out.append(ffn(permuted[chunk], sizes, permuted_weights[chunk], sizes))
    start += rows
out = te.moe_unpermute(
    torch.cat(expert_out), row_id_map, restore_shape=tokens.shape, map_type="mask",
    pad_offsets=pad_offsets,
)
```

`GroupedLinear`, `Sequential`, `ScaledSwiGLU`, `SwiGLU` and `BasicLinear` come from
`transformer_engine.pytorch.ops`, `FP8GlobalStateManager` from
`transformer_engine.pytorch.quantization`. A shared expert is
`Sequential(BasicLinear(hidden, 2 * ffn), SwiGLU(), BasicLinear(ffn, hidden))` on the same
normalized input. With one block (64 experts or fewer) the loop runs once; with more, each
`int(sizes.sum())` waits for the GPU, which costs time at small token counts.

Refuse to build on the cuBLAS versions whose grouped GEMM is wrong:

```python
import transformer_engine_torch as tex  # after transformer_engine.pytorch

if 130700 <= tex.get_cublasLt_version() < 130801:
    raise RuntimeError("cuBLAS 13.7.0 to 13.8.0 corrupts grouped GEMMs; use 13.8.1 or newer.")
```

## Map the weights

A table from HF name to TE name and row range, relative to the layer prefix. For the level 1
Llama layer (`fuse_qkv_params=False`):

```python
def replace_params(hf_state_dict, te_state_dict, config):
    """Copy the replaced layers' weights from an HF state dict (or one shard of it)."""
    ffn = config.intermediate_size
    all_rows = slice(None)
    mapping = {  # HF name: (TE name, rows)
        "input_layernorm.weight": ("self_attention.layernorm_qkv.layer_norm_weight", all_rows),
        "self_attn.q_proj.weight": ("self_attention.layernorm_qkv.query_weight", all_rows),
        "self_attn.k_proj.weight": ("self_attention.layernorm_qkv.key_weight", all_rows),
        "self_attn.v_proj.weight": ("self_attention.layernorm_qkv.value_weight", all_rows),
        "self_attn.o_proj.weight": ("self_attention.proj.weight", all_rows),
        "post_attention_layernorm.weight": ("layernorm_mlp.layer_norm_weight", all_rows),
        "mlp.gate_proj.weight": ("layernorm_mlp.fc1_weight", slice(None, ffn)),  # gate rows first
        "mlp.up_proj.weight": ("layernorm_mlp.fc1_weight", slice(ffn, None)),
        "mlp.down_proj.weight": ("layernorm_mlp.fc2_weight", all_rows),
    }
    written = set()
    for hf_key, value in hf_state_dict.items():
        match = re.match(r"(model\.layers\.\d+\.)(.+)", hf_key)
        if not match or hf_key.endswith("rotary_emb.inv_freq"):  # an old buffer HF drops too
            continue
        te_name, rows = mapping[match.group(2)]  # KeyError: an HF weight with no destination
        target = te_state_dict[match.group(1) + te_name][rows]
        if target.shape != value.shape:
            raise ValueError(f"{hf_key}: expected {tuple(target.shape)}, got {tuple(value.shape)}")
        with torch.no_grad():
            target.copy_(value.to(device=target.device, dtype=target.dtype))
        written.add(match.group(1) + te_name)
    return written
```

**Fused QKV** (`fuse_qkv_params=True`): one weight `layernorm_qkv.weight`, interleaved per
key/value head: that group's query heads, then its key head, then its value head, each
`head_dim` rows. Write through a view:

```python
per_group = num_heads // num_kv_heads
view = te_state_dict[prefix + "self_attention.layernorm_qkv.weight"].view(
    num_kv_heads, per_group + 2, head_dim, -1
)
view[:, :per_group] = q.view(num_kv_heads, per_group, head_dim, -1)
view[:, per_group] = k.view(num_kv_heads, head_dim, -1)
view[:, per_group + 1] = v.view(num_kv_heads, head_dim, -1)
```

`tensor[rows].copy_(value)` with a list or tensor of row indices copies into a temporary and
changes nothing; use a view, a slice, or `tensor[rows] = value`.

**Stacked projections** (`te.LayerNormLinear`): each HF projection is a row range, in the order
of the split in `forward`; the layer's input norm goes to `layer_norm_weight`.

```python
def stacked(name, sizes):
    """Row range of each projection inside the fused weight `name`."""
    ends = [sum(sizes[: i + 1]) for i in range(len(sizes))]
    return [(f"{name}.weight", slice(end - size, end)) for size, end in zip(sizes, ends)]

q_rows, k_rows, v_rows = stacked("self_attn.norm_proj", proj_sizes)
mapping = {
    "input_layernorm.weight": ("self_attn.norm_proj.layer_norm_weight", slice(None)),
    "self_attn.q_proj.weight": q_rows,
    "self_attn.k_proj.weight": k_rows,
    "self_attn.v_proj.weight": v_rows,
}
```

**Experts.** Expert `i` is `weight{i % 64}` of block `i // 64`. A stacked checkpoint tensor holds
one expert per index of its first dimension, and its gate-up weight keeps HF's row order (gate,
then up), the order `SwiGLU` splits in:

```python
experts = {
    "mlp.experts.gate_up_proj": "mlp.experts_gate_up.{block}.weight{index}",
    "mlp.experts.down_proj": "mlp.experts_down.{block}.weight{index}",
}
for i in range(num_experts):
    te_key = prefix + experts[name].format(block=i // MAX_GROUPS, index=i % MAX_GROUPS)
    te_state_dict[te_key].copy_(value[i])
```

Mixtral's older per-expert names are `block_sparse_moe.experts.E.w1` (gate), `w3` (up) and `w2`
(down), and its router is `block_sparse_moe.gate`.

## Load the checkpoint and raise on leftovers

```python
@classmethod
def from_pretrained_local(cls, model_dir, *, config, torch_dtype=torch.bfloat16):
    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No .safetensors files in {model_dir}")
    for sub_config in (config, *(getattr(config, k) for k in ("text_config", "vision_config") if hasattr(config, k))):
        sub_config.dtype = torch_dtype  # HF builds each sub-model in its own config's dtype
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch_dtype)  # TE modules use it at construction
    try:
        model = cls(config)
    finally:
        torch.set_default_dtype(previous)

    te_state_dict = model.state_dict()
    skipped_by_hf = [re.compile(p) for p in model._keys_to_ignore_on_load_unexpected or []]
    written = set()
    for shard in shards:
        with safetensors.safe_open(shard, framework="pt") as handle:
            state_dict = {
                name: handle.get_tensor(name) for name in handle.keys()
                if not any(p.search(name) for p in skipped_by_hf)
            }
        written |= replace_params(state_dict, te_state_dict, config)  # the replaced layers
        rest = {k: v for k, v in state_dict.items() if not LAYER_KEY.match(k)}
        unknown = sorted(set(rest) - set(te_state_dict))
        if unknown:
            raise KeyError(f"Checkpoint weights with no destination: {unknown[:10]}")
        model.load_state_dict(rest, strict=False)  # embeddings, final norm, LM head
        written |= set(rest)

    unwritten = sorted(n for n, _ in model.named_parameters() if n not in written)
    if unwritten:
        raise KeyError(f"{len(unwritten)} weights are not in the checkpoint, such as {unwritten[:10]}")
    return model
```

`LAYER_KEY` is the regular expression of the decoder layer prefix, for example
`re.compile(r"(model\.layers\.\d+\.)(.+)")`.
