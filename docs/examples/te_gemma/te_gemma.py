# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import re
import gc
from contextlib import contextmanager

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from transformers.generation import *
from transformers.generation.utils import *

import torch
from torch import nn

import transformer_engine as te
from transformer_engine.pytorch.attention import RotaryPositionEmbedding
from transformer_engine.pytorch.fp8 import fp8_model_init

import transformers
from transformers.models.gemma.modeling_gemma import GemmaModel, GemmaForCausalLM, GemmaRMSNorm, GemmaConfig
from transformers.modeling_utils import _add_variant, load_state_dict, _load_state_dict_into_model
from transformers.utils import WEIGHTS_INDEX_NAME
from transformers.utils.hub import get_checkpoint_shard_files

@contextmanager
def replace_decoder(te_decoder_cls):
    """
    Replace `GemmaDecoderLayer` with custom `TEGemmaDecoderLayer`.
    """
    original_gemma_decoder_cls = transformers.models.gemma.modeling_gemma.GemmaDecoderLayer
    transformers.models.gemma.modeling_gemma.GemmaDecoderLayer = te_decoder_cls
    try:
        yield
    finally:
        transformers.models.gemma.modeling_gemma.GemmaDecoderLayer = original_gemma_decoder_cls


class TEGemmaDecoderLayer(te.pytorch.TransformerLayer):
    """
    Wrapper class over TE's `TransformerLayer`. This makes the wrapper very
    similar to HF's `GemmaDecoderLayer` and easier to replace it in the code.

    Args:
        config: GemmaConfig
        args: positional args (for compatibility with `GemmaDecoderLayer`)
        kwargs: keyword args (for compatibility with `GemmaDecoderLayer`)
    """
    def __init__(self, config, *args, **kwargs):
        super().__init__(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            bias=False,
            layernorm_epsilon=config.rms_norm_eps,
            hidden_dropout=0,
            attention_dropout=0,
            fuse_qkv_params=False,
            normalization="RMSNorm",
            activation="geglu",
            attn_input_format="bshd",
            num_gqa_groups=config.num_key_value_heads,
            attention_hidden_size=4096
        )
        te_rope = RotaryPositionEmbedding(256)
        self.te_rope_emb = te_rope(max_seq_len=config.max_position_embeddings).cuda()

    def forward(self,
                hidden_states,
                *args,
                attention_mask,
                **kwargs):
        """
        Custom forward to make sure we only pass relevant arguments to the
        forward pass of the `TransformerLayer`. Also, make sure the output
        format matches the output of the HF's `GemmaDecoderLayer`.
        """
        return (super().forward(hidden_states, attention_mask=attention_mask, rotary_pos_emb=self.te_rope_emb),)


class TEGemmaForCausalLM:
    """
    Causal LM created with `GemmaModel`. The underlying `GemmaDecoderLayer`
    class is monkey-patched with `TEGemmaDecoderLayer` class before
    initializing the causal LM with `GemmaForCausalLM`.

    Args:
        config: GemmaConfig
    """

    def __new__(cls, config: GemmaConfig):
        with replace_decoder(te_decoder_cls=TEGemmaDecoderLayer):
            gemma_for_causal_lm = GemmaForCausalLM(config)
        return gemma_for_causal_lm

    @classmethod
    def from_pretrained_local(cls, pretrained_model_name_or_path, *args, config, **kwargs):
        """
        Custom method adapted from `from_pretrained` method in HuggingFace
        Transformers repo: https://github.com/huggingface/transformers/blob/f497f564bb76697edab09184a252fc1b1a326d1e/src/transformers/modeling_utils.py#L2579
        """
        vanilla_model = cls(config)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        subfolder = ""
        variant = None
        if os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, _add_variant("model.safetensors.index.json", variant))
            ):
                # Load from a sharded PyTorch checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant("model.safetensors.index.json", variant)
                )
                is_sharded = True

        resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
                pretrained_model_name_or_path,
                archive_file,
        )

        # If the checkpoint is not sharded, it's a trivial sharding case
        if not is_sharded:
            assert not isinstance(resolved_archive_file, list)
            resolved_archive_file = [resolved_archive_file]

        for shard_file in resolved_archive_file:
            state_dict = load_state_dict(shard_file)
            replace_params(state_dict, vanilla_model.state_dict())
            _load_state_dict_into_model(vanilla_model, state_dict, start_prefix="")

            # Force mem release. Taken from huggingface code
            del state_dict
            gc.collect()

        return vanilla_model


def replace_params(hf_state_dict, te_state_dict):
    # collect all layer prefixes to update
    all_layer_prefixes = set()
    for param_key in hf_state_dict.keys():
        layer_prefix_pat = 'model.layers.\d+.'
        m = re.match(layer_prefix_pat, param_key)
        if m is not None:
            all_layer_prefixes.add(m.group())

    GATE_PROJ_SIZE=24576
    
    for layer_prefix in all_layer_prefixes:
        # When loading weights into models with less number of layers, skip the
        # copy if the corresponding layer doesn't exist in HF model
        if layer_prefix + 'input_layernorm.weight' in hf_state_dict:
            te_state_dict[layer_prefix + 'self_attention.layernorm_qkv.layer_norm_weight'].copy_(1 + hf_state_dict[layer_prefix + 'input_layernorm.weight'])
            
        if layer_prefix + 'self_attn.q_proj.weight' in hf_state_dict:
            te_state_dict[layer_prefix + 'self_attention.layernorm_qkv.query_weight'].copy_(hf_state_dict[layer_prefix + 'self_attn.q_proj.weight'])

        if layer_prefix + 'self_attn.k_proj.weight' in hf_state_dict:
            te_state_dict[layer_prefix + 'self_attention.layernorm_qkv.key_weight'].copy_(hf_state_dict[layer_prefix + 'self_attn.k_proj.weight'])

        if layer_prefix + 'self_attn.v_proj.weight' in hf_state_dict:
            te_state_dict[layer_prefix + 'self_attention.layernorm_qkv.value_weight'].copy_(hf_state_dict[layer_prefix + 'self_attn.v_proj.weight'])

        if layer_prefix + 'self_attn.o_proj.weight' in hf_state_dict:
            te_state_dict[layer_prefix + 'self_attention.proj.weight'].copy_(hf_state_dict[layer_prefix + 'self_attn.o_proj.weight'])

        if layer_prefix + 'post_attention_layernorm.weight' in hf_state_dict:
            te_state_dict[layer_prefix + 'layernorm_mlp.layer_norm_weight'].data[:] = hf_state_dict[layer_prefix + 'post_attention_layernorm.weight'].data[:] + 1
        
        if layer_prefix + 'mlp.gate_proj.weight' in hf_state_dict:
            te_state_dict[layer_prefix + 'layernorm_mlp.fc1_weight'].data[:GATE_PROJ_SIZE] = hf_state_dict[layer_prefix + 'mlp.gate_proj.weight'].data[:]

        if layer_prefix + 'mlp.up_proj.weight' in hf_state_dict:
            te_state_dict[layer_prefix + 'layernorm_mlp.fc1_weight'].data[GATE_PROJ_SIZE:] = hf_state_dict[layer_prefix + 'mlp.up_proj.weight'].data[:]

        if layer_prefix + 'mlp.down_proj.weight' in hf_state_dict:
            te_state_dict[layer_prefix + 'layernorm_mlp.fc2_weight'].copy_(hf_state_dict[layer_prefix + 'mlp.down_proj.weight'].data[:])

    return all_layer_prefixes