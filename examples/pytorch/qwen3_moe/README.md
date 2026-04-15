# Qwen3 MoE with TransformerEngine

Single-GPU implementation of [Qwen3 MoE](https://huggingface.co/Qwen/Qwen3-235B-A22B)
using TransformerEngine for FP8 training and fused kernels.

## Architecture

Same architecture as HuggingFace `Qwen3MoeForCausalLM`, with standard PyTorch
modules replaced by TE equivalents:

| HuggingFace | TransformerEngine |
|---|---|
| `self_attn` (Q/K/V proj + attention + O proj) | `te.MultiheadAttention` (fused LN + QKV + QK-norm + RoPE + attn + O) |
| `post_attention_layernorm` | `te.RMSNorm` |
| Expert MLP (SwiGLU) | `te_ops.Sequential(GroupedLinear, SwiGLU, GroupedLinear)` |
| `model.norm` | `te.RMSNorm` |
| `lm_head` | `te.Linear` |
| RoPE | `te.RotaryPositionEmbedding` |

MoE token routing uses `te.moe_permute_with_probs` / `te.moe_unpermute` for
permutation and expert computation uses `te_ops.GroupedLinear` for fused batched
GEMMs.

## Files

| File | Description |
|---|---|
| `config.py` | `Qwen3MoeConfig` dataclass (defaults match HuggingFace) |
| `model.py` | Full model: `Qwen3MoeRouter`, `Qwen3MoeBlock`, `Qwen3MoeDecoderLayer`, `Qwen3MoeModel`, `Qwen3MoeForCausalLM` |
| `test_vs_hf.py` | Numerical comparison against HuggingFace (forward logits + backward gradients) |

## Running the comparison test

```bash
pip install transformers
python test_vs_hf.py [--seed 42]
```

This builds a small model (2 layers, 8 experts, hidden_size=256), copies weights
from HuggingFace into the TE model, and checks:

1. **Forward**: softmax(logits) match at `atol=1e-5`.
2. **Backward**: all parameter gradients match at `atol=1e-2`.
