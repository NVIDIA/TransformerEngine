#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Report the architecture facts a Transformer Engine recipe depends on.

Reads a local Hugging Face ``config.json`` only: nothing is downloaded and neither a
GPU nor Transformer Engine is needed. The output is a JSON document with the facts found
in the config, the suggested recipe pattern, and the matching TE constructor arguments.

A config does not record the order of normalization and residual connections, nor
whether the MLP is gated. For families this script does not know, those entries are
reported as unconfirmed and have to be read from the model's ``forward``.

Usage:
    python inspect_hf_config.py path/to/config.json
    python inspect_hf_config.py path/to/model_directory
"""

import argparse
import json
import os
import sys

# Families whose layer structure is known. "gated" says whether the MLP computes
# down(act(gate(x)) * up(x)); "pattern" is the reference to follow in this skill.
FAMILIES = {
    "llama": {
        "pattern": "dense-decoder",
        "gated": True,
        "layer_class": "LlamaDecoderLayer",
        "example": "docs/examples/te_llama/te_llama.py",
    },
    "mistral": {
        "pattern": "dense-decoder",
        "gated": True,
        "layer_class": "MistralDecoderLayer",
        "example": "docs/examples/te_llama/te_llama.py",
    },
    "gemma": {
        "pattern": "dense-decoder",
        "gated": True,
        "layer_class": "GemmaDecoderLayer",
        "zero_centered_gamma": True,
        "activation_default": "gelu_pytorch_tanh",
        "example": "docs/examples/te_gemma/te_gemma.py",
    },
    "mixtral": {
        "pattern": "moe-decoder",
        "gated": True,
        "layer_class": "MixtralDecoderLayer",
        "example": "docs/examples/te_mixtral/te_mixtral.py",
    },
    "qwen2": {
        "pattern": "partial",
        "gated": True,
        "layer_class": "Qwen2DecoderLayer",
        "quirks": [
            "Q/K/V projections have a bias while the output and MLP projections do not; "
            "TransformerLayer(bias=...) is a single switch, so fuse norm+MLP and keep "
            "attention native unless an exact match is verified."
        ],
    },
    "qwen3": {
        "pattern": "partial",
        "gated": True,
        "layer_class": "Qwen3DecoderLayer",
        "quirks": [
            "Queries and keys are normalized per head before RoPE; fuse norm+MLP and "
            "verify TE's qk_norm_type / qk_norm_before_rope against the HF forward "
            "before replacing attention."
        ],
    },
    "gemma2": {
        "pattern": "partial",
        "gated": True,
        "layer_class": "Gemma2DecoderLayer",
        "zero_centered_gamma": True,
        "activation_default": "gelu_pytorch_tanh",
        "quirks": [
            "Each layer has four norms (before and after both sublayers), so the layer "
            "is not the two-norm pre-norm block TransformerLayer implements."
        ],
    },
}

# HF activation name -> (TE name when gated, TE name when not gated, note).
ACTIVATIONS = {
    "silu": ("swiglu", "silu", None),
    "swish": ("swiglu", "silu", None),
    "gelu_pytorch_tanh": ("geglu", "gelu", None),
    "gelu_new": ("geglu", "gelu", None),
    "gelu": (
        "geglu",
        "gelu",
        "TE GELU uses the tanh approximation; expect small differences from exact GELU.",
    ),
    "quick_gelu": ("qgeglu", "qgelu", None),
    "relu": ("reglu", "relu", None),
}

MOE_COUNT_KEYS = ("num_local_experts", "num_experts", "n_routed_experts", "moe_num_experts")
MOE_TOPK_KEYS = ("num_experts_per_tok", "num_experts_per_token", "moe_topk", "top_k_experts")


def load_config(source):
    """Return the config as a dict from a config.json file or the directory holding it."""
    if os.path.isdir(source):
        source = os.path.join(source, "config.json")
    if not os.path.isfile(source):
        raise SystemExit(
            f"'{source}' is not a config.json file or a directory containing one. "
            "This script reads local files only."
        )
    with open(source, encoding="utf-8") as handle:
        return json.load(handle)


def first_present(config, keys):
    """Return (key, value) for the first key with a non-null value, else (None, None)."""
    for key in keys:
        if config.get(key) is not None:
            return key, config[key]
    return None, None


def analyze(config):
    """Derive recipe-relevant facts, a suggested pattern and TE arguments from a config."""
    warnings = []
    model_type = config.get("model_type")
    family = FAMILIES.get(model_type, {})
    if not family:
        warnings.append(
            f"Unknown model_type {model_type!r}: norm placement, MLP gating and bias layout "
            "are unconfirmed. Read the layer's forward in the transformers source."
        )

    hidden_size = config.get("hidden_size")
    num_heads = config.get("num_attention_heads")
    num_kv_heads = config.get("num_key_value_heads") or num_heads
    head_dim = config.get("head_dim")
    derived_head_dim = hidden_size // num_heads if hidden_size and num_heads else None

    eps_key, eps = first_present(
        config, ("rms_norm_eps", "layer_norm_eps", "layer_norm_epsilon", "norm_epsilon")
    )
    norm_type = None
    if eps_key == "rms_norm_eps":
        norm_type = "RMSNorm"
    elif eps_key is not None:
        norm_type = "LayerNorm"
    else:
        warnings.append("No normalization epsilon in the config; read the norm class from source.")

    if "activation_default" in family:
        # These families ignore the legacy hidden_act entry when hidden_activation is unset.
        hf_act = config.get("hidden_activation") or family["activation_default"]
    else:
        _, hf_act = first_present(
            config, ("hidden_activation", "hidden_act", "activation_function")
        )
    gated = family.get("gated")
    te_act = None
    if hf_act in ACTIVATIONS:
        gated_name, plain_name, note = ACTIVATIONS[hf_act]
        te_act = plain_name if gated is False else gated_name
        if note:
            warnings.append(note)
        if gated is None:
            te_act = f"{gated_name} if the MLP is gated, else {plain_name}"
    elif hf_act is not None:
        warnings.append(f"No TE activation is mapped for hidden_act={hf_act!r}.")

    count_key, num_experts = first_present(config, MOE_COUNT_KEYS)
    _, top_k = first_present(config, MOE_TOPK_KEYS)
    is_moe = bool(num_experts and num_experts > 1)
    moe = None
    if is_moe:
        moe = {
            "num_experts": num_experts,
            "num_experts_key": count_key,
            "top_k": top_k,
            "router_jitter_noise": config.get("router_jitter_noise"),
            "router_aux_loss_coef": config.get("router_aux_loss_coef"),
            "shared_experts": config.get("n_shared_experts"),
        }
        if moe["shared_experts"]:
            warnings.append("Shared experts run for every token, outside the routing.")

    quirks = list(family.get("quirks", []))
    if config.get("attn_logit_softcapping") is not None:
        quirks.append(
            "Attention logits are soft-capped; set `softcap` on the TE attention and verify, "
            "or keep attention native."
        )
    if config.get("sliding_window") is not None:
        quirks.append(
            "The config sets sliding_window; the published recipes use full causal attention. "
            "Set `window_size` on the TE attention and verify, or state the difference."
        )
    if config.get("is_encoder_decoder"):
        quirks.append("Encoder-decoder model: encoder and decoder layers need separate handling.")
    if config.get("rope_scaling") is not None:
        quirks.append(
            "rope_scaling is set: rotary_base alone does not reproduce the frequencies. "
            "Copy inv_freq from the HF rotary embedding module."
        )

    if is_moe:
        pattern = "moe-decoder"
    elif family:
        pattern = family["pattern"]
    elif config.get("is_encoder_decoder"):
        pattern = "partial"
    elif hidden_size and num_heads and norm_type:
        pattern = "dense-decoder (unconfirmed)"
    else:
        pattern = "unknown"

    if "attention_bias" in config or "mlp_bias" in config:
        bias = bool(config.get("attention_bias", False) or config.get("mlp_bias", False))
    elif family:
        bias = False
    else:
        bias = "unconfirmed: check whether the projections have a bias"

    te_arguments = {
        "hidden_size": hidden_size,
        "num_attention_heads": num_heads,
        "layernorm_epsilon": eps,
        "normalization": norm_type,
        "bias": bias,
        "fuse_qkv_params": True,
    }
    if config.get("num_key_value_heads") is not None:
        te_arguments["num_gqa_groups"] = num_kv_heads
    if head_dim and head_dim != derived_head_dim:
        te_arguments["kv_channels"] = head_dim
    if family.get("zero_centered_gamma"):
        te_arguments["zero_centered_gamma"] = True
    if not is_moe:
        te_arguments["ffn_hidden_size"] = config.get("intermediate_size")
        te_arguments["activation"] = te_act
    if bool(config.get("attention_bias", False)) != bool(config.get("mlp_bias", False)):
        warnings.append(
            "attention_bias and mlp_bias differ; TransformerLayer(bias=...) cannot express that."
        )

    rope_theta = config.get("rope_theta")
    rope_theta_note = None
    if rope_theta is None and family:
        rope_theta_note = "not in config; HF and TE both default to base 10000"
    elif rope_theta is None:
        rope_theta_note = "not in config; check which position encoding the model uses"

    return {
        "model_type": model_type,
        "architectures": config.get("architectures"),
        "pattern": pattern,
        "reference": {
            "dense-decoder": "references/dense-decoder.md",
            "dense-decoder (unconfirmed)": "references/dense-decoder.md",
            "moe-decoder": "references/moe-decoder.md",
            "partial": "references/dense-decoder.md#partial-conversion",
        }.get(pattern),
        "recipe_file": f"te_{model_type}.py" if model_type else None,
        "te_module": {
            "dense-decoder": "TransformerLayer",
            "dense-decoder (unconfirmed)": "TransformerLayer, if the layer is eligible",
            "moe-decoder": "MultiheadAttention + grouped experts",
            "partial": "sublayer fusion: LayerNormMLP, MultiheadAttention, LayerNormLinear",
        }.get(pattern),
        "te_arguments": te_arguments,
        "facts": {
            "num_hidden_layers": config.get("num_hidden_layers"),
            "intermediate_size": config.get("intermediate_size"),
            "num_key_value_heads": num_kv_heads,
            "head_dim": head_dim or derived_head_dim,
            "norm_epsilon_key": eps_key,
            "hf_activation": hf_act,
            "mlp_gated": gated,
            "rope_theta": rope_theta,
            "rope_scaling": config.get("rope_scaling"),
            "rope_theta_note": rope_theta_note,
            "max_position_embeddings": config.get("max_position_embeddings"),
            "vocab_size": config.get("vocab_size"),
            "tie_word_embeddings": config.get("tie_word_embeddings"),
        },
        "moe": moe,
        "hf_source": {
            "modeling_module": (
                f"transformers.models.{model_type}.modeling_{model_type}" if model_type else None
            ),
            "layer_class": family.get("layer_class"),
        },
        "published_example": family.get("example"),
        "quirks": quirks,
        "warnings": warnings,
    }


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", help="config.json file, or the directory containing it")
    args = parser.parse_args()
    json.dump(analyze(load_config(args.source)), sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
