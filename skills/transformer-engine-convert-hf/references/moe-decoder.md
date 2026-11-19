# Mixture-of-experts layers

`te` is `transformer_engine.pytorch` throughout this skill.

A mixture-of-experts (MoE) layer keeps the attention sublayer of a dense layer but replaces the
MLP with a router and a set of expert MLPs, of which each token uses the top `k`. Mixtral is the
reference case: 8 experts, top 2.

`te.TransformerLayer` cannot host this layer, because it always builds a single
`LayerNormMLP`. `LayerNormMLP` does not fit the experts either: the post-attention norm output
feeds both the router and the experts, and the tokens are re-ordered between the norm and the
expert GEMMs. So the layer is composed from three pieces:

```text
h = x + MultiheadAttention(x)          # input norm + QKV fused inside
y = h + MoE(norm(h))                   # standalone TE norm, then router + grouped experts
```

## Contents

- [What the HF implementation costs](#what-the-hf-implementation-costs)
- [Technique ladder](#technique-ladder)
- [Layer](#layer)
- [MoE block: single rank, BF16](#moe-block-single-rank-bf16)
- [Hosting the layer in the HF model](#hosting-the-layer-in-the-hf-model)
- [Fused grouped MLP under MXFP8](#fused-grouped-mlp-under-mxfp8)
- [Expert parallelism](#expert-parallelism)
- [Published example](#published-example)

## What the HF implementation costs

HF loops over the experts in Python. For each expert it gathers that expert's tokens with
`torch.where`, runs three small `nn.Linear` GEMMs and scatters the result back with
`index_add_`. Each GEMM sees only a fraction of the tokens, so the layer is dominated by kernel
launches and memory movement instead of tensor-core work.

## Technique ladder

Apply these in order. Each rung is correct without the ones below it.

| Rung | Technique | TE API | Requires |
|---|---|---|---|
| 1 | Fused input norm + QKV, GQA, TE attention | `MultiheadAttention(input_layernorm=True, fuse_qkv_params=True, num_gqa_groups=...)` | Any TE GPU |
| 2 | Router kept in high precision | `te.Linear` built under `quantized_model_init(enabled=False)`, called under `autocast(enabled=False)` | — |
| 3 | One sort instead of a per-expert gather/scatter | `moe_permute`, `moe_unpermute` | — |
| 4 | All experts in one grouped GEMM per projection; gate and up share a weight | `ops.GroupedLinear` with split sizes | — |
| 5 | Low-precision expert GEMMs | `autocast` with an FP8 or MXFP8 recipe | Hopper or newer; MXFP8 needs Blackwell |
| 6 | GEMM + SwiGLU + routing-weight scaling in one kernel | `ops.Sequential(GroupedLinear, ScaledSwiGLU(glu_interleave_size=32), GroupedLinear)` | MXFP8 or NVFP4 recipe on Blackwell, recent `nvidia-cudnn-frontend` |
| 7 | Experts spread over several GPUs | Expert-parallel dispatch | Multi-GPU launch |

Default to rungs 1 to 4, which need one GPU and no special hardware. Add 5 for a fine-tuning
recipe on a GPU that supports it. Add 6 and 7 only when the user asks for them or the model does
not fit otherwise, and say that they were not exercised if you could not run them.

## Layer

```python
import torch
import torch.nn as nn
import transformer_engine.pytorch as te
from transformer_engine.pytorch.attention import RotaryPositionEmbedding
from transformer_engine.pytorch.ops import GroupedLinear, Sequential, SwiGLU


class TEMixtralDecoderLayer(nn.Module):
    """Mixtral decoder layer: TE attention with fused input norm, TE norm, grouped-expert MoE."""

    def __init__(self, config, layer_idx, *args, **kwargs):
        super().__init__()
        dtype = getattr(config, "dtype", None) or torch.bfloat16
        self.self_attention = te.MultiheadAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_gqa_groups=config.num_key_value_heads,
            bias=False,
            layernorm_epsilon=config.rms_norm_eps,
            attention_dropout=0,
            fuse_qkv_params=True,
            qkv_weight_interleaved=True,
            normalization="RMSNorm",
            input_layernorm=True,  # fuses HF's input_layernorm into the QKV projection
            qkv_format="bshd",
            attn_mask_type="causal",
            layer_number=layer_idx + 1,
            params_dtype=dtype,
        )
        self.post_attention_layernorm = te.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )
        self.mlp = TEMixtralSparseMoeBlock(config)

        # No fallback on purpose: a config without rope_theta should stop the recipe.
        te_rope = RotaryPositionEmbedding(
            config.hidden_size // config.num_attention_heads, rotary_base=config.rope_theta
        )
        self.te_rope_emb = te_rope(max_seq_len=config.max_position_embeddings).cuda()

    def forward(self, hidden_states, *args, **kwargs):
        # HF's additive mask, position ids and caches are not forwarded: TE builds the
        # causal mask itself (exact for unpadded and right-padded batches).
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        attn_output = self.self_attention(hidden_states, rotary_pos_emb=self.te_rope_emb)
        hidden_states = hidden_states + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states
```

Note the argument names: the layout argument is `qkv_format` on `MultiheadAttention` and
`attn_input_format` on `TransformerLayer`. The notes on the layer return value, masks, RoPE and
the KV cache in `references/dense-decoder.md` apply to this layer as well.

## MoE block: single rank, BF16

```python
class TEMixtralSparseMoeBlock(nn.Module):
    """Router, token permutation, and all experts through grouped GEMMs."""

    def __init__(self, config):
        super().__init__()
        dtype = getattr(config, "dtype", None) or torch.bfloat16
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # The router decides which experts run. Quantization noise here changes the
        # routing itself, so the gate is created and called outside quantization.
        with te.quantized_model_init(enabled=False):
            self.gate = te.Linear(
                config.hidden_size, self.num_experts, bias=False, params_dtype=dtype
            )

        # Gate and up projections share one [2 * intermediate, hidden] weight per expert:
        # two grouped GEMMs per layer instead of three GEMMs per expert.
        self.experts_gate_up = GroupedLinear(
            num_groups=self.num_experts,
            in_features=config.hidden_size,
            out_features=2 * config.intermediate_size,
            bias=False,
            dtype=dtype,
            device="cuda",
        )
        self.experts_down = GroupedLinear(
            num_groups=self.num_experts,
            in_features=config.intermediate_size,
            out_features=config.hidden_size,
            bias=False,
            dtype=dtype,
            device="cuda",
        )
        # Sequential lets TE's operation fuser combine the three ops. object.__setattr__
        # keeps the wrapper out of the module tree, so each expert weight is registered once,
        # under experts_gate_up / experts_down.
        object.__setattr__(
            self, "_experts_ffn", Sequential(self.experts_gate_up, SwiGLU(), self.experts_down)
        )

    def forward(self, hidden_states):
        shape = hidden_states.shape
        tokens = hidden_states.reshape(-1, self.hidden_size)

        with te.autocast(enabled=False):
            router_logits = self.gate(tokens)  # [N, E]
        probs = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(probs, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # Sort the (token, expert) pairs by expert: one contiguous chunk per expert.
        split_sizes = torch.bincount(
            selected_experts.reshape(-1), minlength=self.num_experts
        ).int()
        permuted, row_id_map = te.moe_permute(
            tokens,
            selected_experts.to(torch.int32),
            num_out_tokens=selected_experts.numel(),
            map_type="index",
        )

        # Each GroupedLinear takes the per-expert token counts as its extra input.
        expert_out = self._experts_ffn(permuted, split_sizes, split_sizes)

        # Restore token order and sum each token's top-k outputs with its routing weights.
        out = te.moe_unpermute(
            expert_out,
            row_id_map,
            merging_probs=routing_weights,
            restore_shape=tokens.shape,
            map_type="index",
        )
        return out.reshape(shape)
```

Preserve what the HF block does beyond routing. Mixtral multiplies the hidden states by uniform
jitter noise during training when `router_jitter_noise > 0`, and adds a load-balancing loss when
`router_aux_loss_coef > 0`. TE has a fused kernel for that loss,
`transformer_engine.pytorch.router.fused_moe_aux_loss(probs, tokens_per_expert,
total_num_tokens, num_experts, topk, coeff)`. For routers that score with sigmoid, use grouped
top-k or an expert bias, see `fused_topk_with_score_function` in the same module.

The expert weights are named `experts_gate_up.weight0..weight{E-1}` and
`experts_down.weight0..weight{E-1}`; the mapping from HF keys is in
`references/weight-mapping.md`.

## Hosting the layer in the HF model

Two options, in increasing effort:

1. **Patch the HF layer class**, exactly as the dense recipe does with `replace_decoder`, so
   `MixtralForCausalLM` builds itself with `TEMixtralDecoderLayer`. This is the least code and
   keeps HF's embeddings, final norm, LM head and generation. It works when the HF model loop
   only needs the hidden states back from each layer; check what the installed loop does with
   router logits and layer outputs.
2. **Own the model classes**, as the published `te_mixtral.py` does with `TEMixtralModel` and
   `TEMixtralForCausalLM`. This is needed for packed variable-length batches (`thd`), TE's KV
   cache, per-layer precision and expert parallelism, because those change what the model-level
   forward passes to each layer.

Start with option 1 unless the user asked for one of the features that need option 2.

## Fused grouped MLP under MXFP8

On Blackwell GPUs with an MXFP8 (or NVFP4) recipe, TE's operation fuser can replace the three
ops with one fused grouped-MLP kernel that also applies the routing weights and requantizes the
input of the second GEMM. The fusion is silent: if a condition is not met the ops run unfused
and nothing is reported, so verify that it fired.

What changes relative to the BF16 block:

- The activation is `ScaledSwiGLU(glu_interleave_size=32)` and it receives the per-token
  routing probabilities as an extra input:
  `self._experts_ffn(permuted, split_sizes, permuted_probs, split_sizes)`.
- Because the activation already applied the routing weights, call
  `te.moe_unpermute(..., merging_probs=None, map_type="mask", pad_offsets=pad_offsets)`.
- Tokens are permuted with `te.moe_permute_and_pad_with_probs(tokens, routing_probs,
  routing_map, tokens_per_expert, align_size)`, which pads each expert's chunk to the alignment
  the kernel needs and returns
  `(permuted, permuted_probs, row_id_map, pad_offsets, padded_tokens_per_expert)`. `routing_map`
  and `routing_probs` are dense `[N, E]` tensors built with `scatter_`.
- Gate and up rows of each expert's first weight are stored in alternating blocks of 32
  (`te.interleave_glu_tensor`).
- `GroupedLinear` is created with `split_alignment` matching the padding.
- Feature sizes must be multiples of 64.

The alignment values and the supported GPU and library versions have changed between TE
releases. Read them from the installed version (`transformer_engine/pytorch/ops/fused/` and the
published `te_mixtral_mxfp8.py` and `te_moe_dispatch.py`) instead of hard-coding numbers from
memory.

## Expert parallelism

With expert parallelism (EP) each rank owns `E / ep_size` experts and tokens travel to the rank
that owns their expert. This changes the launch (`torchrun`, one process group per data-parallel
replica), the weight loading (each rank loads its own expert slice) and the dispatch (permute,
exchange, re-sort by local expert; then the reverse). The published example implements it with a
`TokenDispatcher` protocol around NCCL `all_to_all_single`; newer TE versions also ship
`transformer_engine.pytorch.ep` and fusible `MoeDispatch` / `MoeCombine` ops.

Treat EP as opt-in. A single-rank recipe is the right default, and a model that does not fit on
one GPU is the reason to add EP.

## Published example

<https://github.com/NVIDIA/TransformerEngine/tree/main/docs/examples/te_mixtral>: `te_mixtral.py`
(BF16, loop and grouped modes, EP), `te_mixtral_mxfp8.py` and `te_moe_dispatch.py` (fused MXFP8
path), `hf_to_te_weights.py` (weight mapping for both checkpoint formats), and the tutorial
notebook with measured step times for each rung.
