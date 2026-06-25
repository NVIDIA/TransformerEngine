# Attention Backend Selection Matrix Draft

This Markdown file is the editable matrix companion to
`attention_backend_selection.rst`. It starts with the CP documentation outline
from `attention_docs/misc/ideas.md`, then uses a feature-by-hardware monospace
ASCII matrix at the selected leaf.

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
                    +------------------+---------+---------------------------+---------------------------+---------------------------+---------------------------+
                    | Feature          | sm < 80 | sm80/sm89                 | sm90                      | sm100+                    | sm120                     |
                    +------------------+---------+---------------------------+---------------------------+---------------------------+---------------------------+
                    | MHA/MQA/GQA      | No      | Yes; cuDNN shape gate     | Yes; cuDNN shape gate     | Yes; cuDNN shape gate     | Yes; newer cuDNN paths    |
                    | MLA              | No      | Q/K/V separate dims only  | Q/K/V separate dims only  | Q/K/V separate dims only  | Narrower; train MLA gates |
                    | SWA              | No      | No p2p; use a2a/all_gath. | No p2p; use a2a/all_gath. | No p2p; use a2a/all_gath. | No p2p; use a2a/all_gath. |
                    | Standard masks   | No      | no,pad,causal,pad_causal  | no,pad,causal,pad_causal  | no,pad,causal,pad_causal  | no,pad,causal,pad_causal  |
                    | Bottom-right     | No      | No under CP               | No under CP               | No under CP               | No under CP               |
                    | Bias             | No      | no_bias/post_scale only   | no_bias/post_scale only   | no_bias/post_scale only   | no_bias/post_scale only   |
                    | Sink softmax     | No      | No p2p; use a2a           | No p2p; use a2a           | No p2p; use a2a           | No p2p; use a2a           |
                    | return_max_logit | No      | non-FP8; cuDNN shape gate | non-FP8; cuDNN shape gate | non-FP8; cuDNN shape gate | non-FP8; cuDNN shape gate |
                    | Determinism      | No      | cuDNN/train/bias gates    | better; bias may disable  | stricter cuDNN gates      | No deterministic training |
                    +------------------+---------+---------------------------+---------------------------+---------------------------+---------------------------+
                    Notes:
                        - "all_gath." means the all_gather communication path.
                        - Standard masks are no_mask, padding, causal, and padding_causal.
                        - Causal cross-attention and bottom-right masks are excluded under CP here.
                        - For exact cuDNN versions, shape gates, and exclusions, refer to:
                            transformer_engine/pytorch/attention/dot_product_attention/utils.py
                            transformer_engine/common/fused_attn/fused_attn.cpp

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

### Where To Check Exact Behavior

For exact support, users should refer to:

- Python selector:
  `transformer_engine/pytorch/attention/dot_product_attention/utils.py`
- cuDNN FusedAttention probe:
  `transformer_engine/common/fused_attn/fused_attn.cpp`

Those files contain the precise cuDNN version gates, shape gates, and
hardware-specific exclusions.
