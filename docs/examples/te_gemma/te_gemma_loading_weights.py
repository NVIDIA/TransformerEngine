# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import re
import gc
import torch

from typing import List

from transformer_engine.pytorch.quantization import quantized_model_init

from transformers.modeling_utils import load_state_dict
from transformers.utils.hub import get_checkpoint_shard_files

"""
    This file contains logic of mapping the HuggingFace GemmaModel parameters
    with TransformerEngine TransformerLayer. When we have initialized Transformer models
    both with HF and with TE, we can copy parameters from the first to the second.
"""


def _load_weights_for_fp8_model(vanilla_model, hyperparams):
    """
    Loads weights and FP8 metadata from a calibrated weights file.

    The weights are in BF16 precision, but the state dict also contains
    fp8 metadata computed by the calibration procedure.
    """

    fp8_metadata_sd = torch.load(hyperparams.fp8_model_weights_filename)

    # A hack to remove the extra state from the fp8_metadata_sd
    # that contains the extra state from the core_attention module.
    fp8_metadata_sd = {
        k: v for k, v in fp8_metadata_sd.items() if "core_attention._extra_state" not in k
    }
    vanilla_model.load_state_dict(
        fp8_metadata_sd,
        strict=False,
        # Because some parameters have multiple pointers to the same weight
        # vanilla_model._model_context_phase.model and
        # vanilla_model._model_generation_phase.model we need to load the
        # weights in a non-strict manner.
    )


def _load_weights_for_standard_model(vanilla_model, config):
    """
    Loads weights from the HuggingFace checkpoint.
    """

    archive_file = os.path.join(config.weights_cache_dir, "model.safetensors.index.json")
    resolved_archive_file, _ = get_checkpoint_shard_files(config.weights_cache_dir, archive_file)
    total_dict = {}
    for shard_file in resolved_archive_file:
        state_dict = load_state_dict(shard_file)
        total_dict.update(state_dict)

    replace_params(
        total_dict,
        vanilla_model.state_dict(),
        config,
        qkv_fused_and_interleaved=config.fuse_qkv_params,
    )
    # Copy remaining parameters like embedding.
    vanilla_model.load_state_dict(total_dict, strict=False)

    # Force mem release. Taken from huggingface code.
    del total_dict
    gc.collect()


def load_te_model(cls, config):
    """
    Loads the TE model with proper weights.
    """

    # Force the dtype to bfloat16 while loading the model.
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    """
    Custom method adapted from `from_pretrained` method in HuggingFace
    Transformers repo:
    https://github.com/huggingface/transformers/blob/f497f564bb76697edab09184a252fc1b1a326d1e/src/transformers/modeling_utils.py#L2579
    """
    config.use_cache = False  # To make TransformerLayer compatible with GemmaModel

    # Loading model with FP8 only weights needs both the following context managers.
    # 1. quantized_model_init(config.quantized_model_init) to tell TE to use FP8 only weights.
    # 2. torch.no_grad() during TE modules' initilization so that they respect
    #    the `quantized_model_init` context manager.
    with torch.no_grad(), quantized_model_init(config.quantized_model_init):
        # Just create a model with random weights.
        vanilla_model = cls(config).cuda()

    # Copy proper weights into the model. If loading weights with FP8 metadata,
    # then the source weights are basically the same as the weights in the model.
    # If not, then we need to load the weights from the HuggingFace checkpoint
    # and do mapping of the weight names from HF to the TE model.
    if config.fp8_model_weights_filename is not None:
        _load_weights_for_fp8_model(vanilla_model, config)
    else:
        _load_weights_for_standard_model(vanilla_model, config)

    # Restore the original dtype.
    torch.set_default_dtype(old_dtype)
    return vanilla_model


def _get_all_layer_prefixes_to_update(hf_state_dict):
    """
    There are many parameters in hf_state_dict, whose name start with "model.layers.[number]."
    This function extracts all strings like "model.layers.[number]."
    that are starting strings of keys in hf_state_dict.
    """
    all_layer_prefixes = set()
    for param_key in hf_state_dict.keys():
        layer_prefix_pat = "model.layers.\d+."
        m = re.match(layer_prefix_pat, param_key)
        if m is not None:
            all_layer_prefixes.add(m.group())
    return all_layer_prefixes


def replace_params(hf_state_dict, te_state_dict, config, qkv_fused_and_interleaved=False):
    """
    Replaces params from TE TransformerLayer state_dict with corresponding parameters
    from HuggingFace GemmaModel state_dict.
    """
    all_layer_prefixes: List[str] = _get_all_layer_prefixes_to_update(hf_state_dict)

    for layer_prefix in all_layer_prefixes:

        def copy_from_ht_to_te(te_name, hf_name, start=None, end=None):
            te_state_dict[layer_prefix + te_name].data[start:end].copy_(
                hf_state_dict[layer_prefix + hf_name]
            )

        copy_from_ht_to_te(
            "self_attention.layernorm_qkv.layer_norm_weight", "input_layernorm.weight"
        )
        copy_from_ht_to_te("self_attention.proj.weight", "self_attn.o_proj.weight")
        copy_from_ht_to_te("layernorm_mlp.layer_norm_weight", "post_attention_layernorm.weight")
        copy_from_ht_to_te("layernorm_mlp.fc2_weight", "mlp.down_proj.weight")
        copy_from_ht_to_te(
            "layernorm_mlp.fc1_weight", "mlp.gate_proj.weight", end=config.intermediate_size
        )
        copy_from_ht_to_te(
            "layernorm_mlp.fc1_weight", "mlp.up_proj.weight", start=config.intermediate_size
        )

        if qkv_fused_and_interleaved:
            """
            When qkv_fused_and_interleaved=True, key, query and value layers are on one tensor
            in TE TransformerLayer. Moreover they are interleaved within each head.
            Let q_i, k_i and v_i be query, key and value layers for i-th head respectively.
            Then TE stores weight tensor in the form:
            [q1 k1 v1 q2 k2 v2 ...]
            This is done to maximally optimize performance time.
            """
            te_qkv_layer = te_state_dict[layer_prefix + "self_attention.layernorm_qkv.weight"]

            def copy_interleave(hf_name, idx):
                src = hf_state_dict[layer_prefix + hf_name]
                for head_nr in range(config.num_attention_heads):
                    dst_offset = head_nr * config.head_dim * 3
                    dst_slice = slice(
                        dst_offset + idx * config.head_dim, dst_offset + (idx + 1) * config.head_dim
                    )
                    src_slice = slice(
                        head_nr * config.head_dim, head_nr * config.head_dim + config.head_dim
                    )
                    te_qkv_layer[dst_slice, :] = src[src_slice, :]

            copy_interleave("self_attn.q_proj.weight", 0)
            copy_interleave("self_attn.k_proj.weight", 1)
            copy_interleave("self_attn.v_proj.weight", 2)
        else:
            copy_from_ht_to_te(
                "self_attention.layernorm_qkv.query_weight", "self_attn.q_proj.weight"
            )
            copy_from_ht_to_te("self_attention.layernorm_qkv.key_weight", "self_attn.k_proj.weight")
            copy_from_ht_to_te(
                "self_attention.layernorm_qkv.value_weight", "self_attn.v_proj.weight"
            )

    return all_layer_prefixes
