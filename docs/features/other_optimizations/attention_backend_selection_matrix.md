# Attention Backend Selection Matrix Draft

This Markdown file is the editable matrix companion to
`attention_backend_selection.rst`. It starts with the CP documentation outline
from `attention_docs/misc/ideas.md`, then uses a hardware-by-feature matrix at
the selected leaf.

Scope:

- PyTorch `DotProductAttention` is the first target.
- JAX is left as a later pass.
- The populated leaf below is only `Fused Attention -> P2P -> F16 ->
  BSHD/SBHD`.
- Exact cuDNN patch-level gates and FlashAttention package-version gates stay
  in `transformer_engine/pytorch/attention/dot_product_attention/utils.py` and
  `transformer_engine/common/fused_attn/fused_attn.cpp`.

## Outline

```text
CP docs
    PyTorch/JAX
        PyTorch: target for this draft
        JAX: later pass

    Load balance strategy
        DCP: supported scope for PyTorch
        CP + KV cache: no backend
        CP + Unfused: no
        CP + FA4: no

    Fused Attention
        A2A
            [not expanded]

        P2P
            F16
                BSHD/SBHD
                    -> Table 1: Hardware x feature support

                THD
                    [not expanded]

            FP8 DelayedS
                [not expanded]

            FP8 CurrentS
                [not expanded]

            MXFP8
                [not expanded]

        A2A+P2P
            [not expanded]

        AG
            [not expanded]

    Flash Attention
        [not expanded]

    Unfused Attention
        [not expanded]
```

## Fused Attention / P2P / F16 / BSHD-SBHD

### Table 1: Hardware x Feature Support

| Hardware | MHA/MQA/GQA | MLA | SWA | Standard masks | Bottom-right masks | Bias | Sink softmax | `return_max_logit` | Determinism |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `sm < 80` | Not supported: FusedAttention requires `sm80+`. | Not supported. | Not supported. | Not supported. | Not supported. | Not supported. | Not supported. | Not supported. | Not supported. |
| `sm80/sm89` | Supported when cuDNN accepts the shape. | Selected separate Q/K/V layout and head dimensions only. | Not supported with `p2p`; use `a2a` or `all_gather`. | `no_mask`, `padding`, `causal`, `padding_causal`; no causal cross-attention. | Not supported under CP. | `no_bias` or selected `post_scale_bias`; no `pre_scale_bias`. | Not supported with `p2p`; use `a2a`. | Supported only for non-FP8 when cuDNN accepts the shape. | Depends on cuDNN, training mode, and bias. |
| `sm90` | Supported when cuDNN accepts the shape. | Selected separate Q/K/V layout and head dimensions only. | Not supported with `p2p`; use `a2a` or `all_gather`. | `no_mask`, `padding`, `causal`, `padding_causal`; no causal cross-attention. | Not supported under CP. | `no_bias` or selected `post_scale_bias`; no `pre_scale_bias`. | Not supported with `p2p`; use `a2a`. | Supported only for non-FP8 when cuDNN accepts the shape. | Better supported than pre-Hopper, but trainable bias can still remove it. |
| `sm100+` | Supported when cuDNN accepts the shape. | Selected separate Q/K/V layout and head dimensions only. | Not supported with `p2p`; use `a2a` or `all_gather`. | `no_mask`, `padding`, `causal`, `padding_causal`; no causal cross-attention. | Not supported under CP. | `no_bias` or selected `post_scale_bias`; no `pre_scale_bias`. | Not supported with `p2p`; use `a2a`. | Supported only for non-FP8 when cuDNN accepts the shape. | Depends on cuDNN; deterministic training has stricter gates. |
| `sm120` | Supported only on newer cuDNN-accepted paths. | Narrower than generic `sm100+`; training MLA has extra exclusions. | Not supported with `p2p`; use `a2a` or `all_gather`. | `no_mask`, `padding`, `causal`, `padding_causal`; no causal cross-attention. | Not supported under CP. | `no_bias` or selected `post_scale_bias`; no `pre_scale_bias`. | Not supported with `p2p`; use `a2a`. | Supported only for non-FP8 when cuDNN accepts the shape. | Deterministic training is not supported. |

### Where To Check Exact Behavior

For exact support, users should refer to:

- Python selector:
  `transformer_engine/pytorch/attention/dot_product_attention/utils.py`
- cuDNN FusedAttention probe:
  `transformer_engine/common/fused_attn/fused_attn.cpp`

Those files contain the precise cuDNN version gates, shape gates, and
hardware-specific exclusions.
