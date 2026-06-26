# Attention Backend Selection Matrix Draft

This Markdown file is the editable matrix companion to
`attention_backend_selection.rst`. It starts with the CP documentation outline
from `attention_docs/misc/ideas.md`, then uses a feature-by-hardware monospace
ASCII matrix at the selected leaf.

Scope:

- PyTorch `DotProductAttention` is the first target.
- JAX is left as a later pass.
- The populated leaves below are under `Fused Attention -> P2P`.

Support cells are intentionally broad. For exact cuDNN versions, shape gates,
and exclusions, refer to:

- `transformer_engine/pytorch/attention/dot_product_attention/utils.py`
- `transformer_engine/common/fused_attn/fused_attn.cpp`

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
                    +------------------+-----------------------------------------------+--------------------------------------------+
                    | Feature          | sm80/sm89/sm90                                | sm100+                                     |
                    +------------------+-----------------------------------------------+--------------------------------------------+
                    | MHA/MQA/GQA      | Yes; cuDNN shape gates                        | Yes; Blackwell cuDNN gates differ          |
                    | MLA              | Selected separate Q/K/V dims                   | Selected; narrower Blackwell/SM120 gates   |
                    | SWA              | No with p2p; use a2a or all_gather             | No with p2p; use a2a or all_gather         |
                    | Masks            | no_mask, padding, causal, padding_causal;      | Same CP mask surface; Blackwell gates      |
                    |                  | no causal_bottom_right or                      | differ in backend probe                    |
                    |                  | padding_causal_bottom_right                    |                                            |
                    | Bias             | no_bias; selected post_scale_bias              | Same broad surface; Blackwell gates differ |
                    | Sink softmax     | No with p2p; use a2a                           | No with p2p; use a2a                       |
                    | return_max_logit | Yes for non-FP8 when cuDNN accepts shape       | Same broad surface; Blackwell gates differ |
                    | Determinism      | Yes through selector gates                     | Stricter Blackwell/SM120 training gates    |
                    +------------------+-----------------------------------------------+--------------------------------------------+

                THD
                    sm80/sm89: no FusedAttention THD backend for this leaf.

                    +-------------------+-------------------------------------------------+--------------------------------------------+
                    | Feature           | sm90                                            | sm100+                                     |
                    +-------------------+-------------------------------------------------+--------------------------------------------+
                    | Required metadata | cu_seqlens_q, cu_seqlens_kv,                    | Same; SM120 adds cuDNN/layout gates        |
                    |                   | cu_seqlens_q_padded, cu_seqlens_kv_padded       |                                            |
                    | MHA/MQA/GQA       | MHA and selected MQA/GQA through cuDNN gates    | Same broad surface; Blackwell gates differ |
                    | MLA               | Selected separate Q/K/V dims; training gates    | Selected; Blackwell/SM120 gates differ     |
                    | pad_between_seqs  | Supported through padded cu_seqlens metadata    | Same broad surface; Blackwell gates differ |
                    | SWA               | No with p2p; use a2a                            | No with p2p; use a2a                       |
                    | Masks             | padding; padding_causal when                    | Same broad surface; Blackwell gates differ |
                    |                   | max_seqlen_q == max_seqlen_kv; no               |                                            |
                    |                   | causal_bottom_right or                          |                                            |
                    |                   | padding_causal_bottom_right                     |                                            |
                    | Bias              | no_bias only                                    | no_bias only                               |
                    | Sink softmax      | No with p2p; use a2a                            | No with p2p; use a2a                       |
                    | return_max_logit  | Yes for non-FP8 when cuDNN accepts shape        | Same broad surface; Blackwell gates differ |
                    | Determinism       | Yes through selector gates                      | Stricter Blackwell/SM120 training gates    |
                    +-------------------+-------------------------------------------------+--------------------------------------------+

            FP8 DelayedS
                BSHD/SBHD
                    sm80/sm89: no FP8 FusedAttention backend for this leaf.

                    +------------------+----------------------------------------+----------------------------------------+------------------------+
                    | Feature          | sm90                                   | sm100+ not sm120                       | sm120                  |
                    +------------------+----------------------------------------+----------------------------------------+------------------------+
                    | Recipe           | DelayedScaling + fp8_dpa               | Same; Blackwell gates differ           | No FP8 backend         |
                    | MHA/MQA/GQA      | Yes; selected cuDNN shapes             | Yes; Blackwell cuDNN gates differ      | No FP8 backend         |
                    | MLA              | Selected Q/K/V dims; train gates       | Selected; Blackwell train gates differ | No FP8 backend         |
                    | SWA              | No with p2p; use a2a or all_gather     | No with p2p; use a2a or all_gather     | No FP8 backend         |
                    | Masks            | no_mask, padding, causal,              | Same CP mask surface; Blackwell gates  | No FP8 backend         |
                    |                  | padding_causal; no causal_bottom_right | differ in backend probe                |                        |
                    |                  | or padding_causal_bottom_right         |                                        |                        |
                    | Bias             | no_bias only                           | no_bias only                           | No FP8 backend         |
                    | Sink softmax     | No with p2p; use a2a                   | No with p2p; use a2a                   | No FP8 backend         |
                    | return_max_logit | No with FP8                            | No with FP8                            | No FP8 backend         |
                    | Determinism      | Yes through FP8 training gates         | Yes through Blackwell FP8 gates        | No FP8 backend         |
                    +------------------+----------------------------------------+----------------------------------------+------------------------+

                THD
                    No backend: CP disables FP8 FusedAttention for qkv_format=thd,
                    and FP8 CP does not fall back to FlashAttention or Unfused.

            FP8 CurrentS
                BSHD/SBHD
                    sm80/sm89/sm90: no FP8 current-scaling FusedAttention backend
                    for this leaf.

                    +------------------+----------------------------------------+------------------------+
                    | Feature          | sm100+ not sm120                       | sm120                  |
                    +------------------+----------------------------------------+------------------------+
                    | Recipe           | Float8CurrentScaling + fp8_dpa         | No FP8 backend         |
                    | MHA/MQA/GQA      | Yes; Blackwell cuDNN gates             | No FP8 backend         |
                    | MLA              | Selected; Blackwell train gates differ | No FP8 backend         |
                    | SWA              | No with p2p; use a2a or all_gather     | No FP8 backend         |
                    | Masks            | no_mask, padding, causal,              | No FP8 backend         |
                    |                  | padding_causal; no causal_bottom_right |                        |
                    |                  | or padding_causal_bottom_right         |                        |
                    | Bias             | no_bias only                           | No FP8 backend         |
                    | Sink softmax     | No with p2p; use a2a                   | No FP8 backend         |
                    | return_max_logit | No with FP8                            | No FP8 backend         |
                    | Determinism      | Yes through current-scaling gates      | No FP8 backend         |
                    +------------------+----------------------------------------+------------------------+

                THD
                    No backend: CP disables FP8 FusedAttention for qkv_format=thd,
                    and FP8 CP does not fall back to FlashAttention or Unfused.

            MXFP8
                BSHD/SBHD
                    sm80/sm89/sm90: no MXFP8 FusedAttention backend for this leaf.

                    +------------------+----------------------------------------+------------------------+
                    | Feature          | sm100+ not sm120                       | sm120                  |
                    +------------------+----------------------------------------+------------------------+
                    | Recipe           | MXFP8BlockScaling + fp8_dpa;           | No FP8 backend         |
                    |                  | fp8_mha must be false                  |                        |
                    | MHA/MQA/GQA      | Yes; Blackwell cuDNN gates             | No FP8 backend         |
                    | MLA              | Selected; Blackwell train gates differ | No FP8 backend         |
                    | SWA              | No with p2p; use a2a or all_gather     | No FP8 backend         |
                    | Masks            | no_mask, padding, causal,              | No FP8 backend         |
                    |                  | padding_causal; no causal_bottom_right |                        |
                    |                  | or padding_causal_bottom_right         |                        |
                    | Bias             | no_bias only                           | No FP8 backend         |
                    | Sink softmax     | No with p2p; use a2a                   | No FP8 backend         |
                    | return_max_logit | No with FP8                            | No FP8 backend         |
                    | Determinism      | Yes through MXFP8/Blackwell gates      | No FP8 backend         |
                    +------------------+----------------------------------------+------------------------+

                THD
                    No backend: MXFP8 FusedAttention excludes qkv_format=thd,
                    and FP8 CP does not fall back to FlashAttention or Unfused.

        A2A+P2P
            [not expanded]

        AG
            [not expanded]

    Flash Attention
        [not expanded]

    Unfused Attention
        [not expanded]
```
