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
    def __init__(self, config, layer_idx, *args, **kwargs):
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
            attention_hidden_size=4096,
            layer_number=(layer_idx+1)
        )
        te_rope = RotaryPositionEmbedding(256)
        self.te_rope_emb = te_rope(max_seq_len=config.max_position_embeddings).cuda()

    def forward(self,
                hidden_states,
                *args,
                attention_mask,
                inference_params,
                self_attn_mask_type='causal',
                **kwargs):
        """
        Custom forward to make sure we only pass relevant arguments to the
        forward pass of the `TransformerLayer`. Also, make sure the output
        format matches the output of the HF's `GemmaDecoderLayer`.
        """
        return (super().forward(hidden_states, attention_mask=attention_mask, rotary_pos_emb=self.te_rope_emb, inference_params=inference_params, self_attn_mask_type=self_attn_mask_type),)


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
            # trzeba wstawis layer number do tego czegos w jakis sposob
            gemma_for_causal_lm = GemmaForCausalLM(config)

        gemma_for_causal_lm.generate = TEGemmaForCausalLM.generate.__get__(gemma_for_causal_lm, GemmaForCausalLM)

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
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        max_new_tokens = 0,
        **kwargs,
    ):
        num_heads = self.model.config.num_attention_heads
        batch_size, seq_len = input_ids.shape
        max_seq_len = seq_len + max_new_tokens
        generation_config, _ = self._prepare_generation_config(generation_config, **kwargs)
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

        # inference_params object is a cache, where keys and values of previous tokens are stored
        inference_params = te.pytorch.InferenceParams(
            max_batch_size=batch_size, 
            max_sequence_length=seq_len+max_new_tokens+1) 

        # mask has shape [batch_size, num_heads, 1, max_seq_len] and contains False 
        # when coressponding token is padding and True otherwise.
        pad_attention_mask = input_ids.ne(generation_config.pad_token_id)
        mask = torch.ones((batch_size, num_heads, 1, max_seq_len), dtype=torch.bool).cuda()
        mask[..., :seq_len] = mask[..., :seq_len] & pad_attention_mask.unsqueeze(1).unsqueeze(2).expand(-1, num_heads, -1, -1)

        hidden_states = self.model.embed_tokens(input_ids)
        output_tokens = []
        for i in range(max_new_tokens):
            normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
            hidden_states = hidden_states * normalizer
            for decoder_layer in self.model.layers:
                hidden_states = decoder_layer(
                            hidden_states,
                            # In the case of arbiutrary mask, the meaning of True and False is switched, so negation is needed.
                            attention_mask=pad_attention_mask if i == 0 else ~mask[..., :seq_len],
                            self_attn_mask_type="padding_causal" if i == 0 else "arbitrary",
                            inference_params=inference_params
                        )[0]

            # inference_params.sequence_len_offset should contain position of the current token in the sequence.
            inference_params.sequence_len_offset += hidden_states.shape[1]

            hidden_states = self.model.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            logits = logits.float()
            logits = logits[:, -1, :]
            next_tokens = torch.argmax(logits, dim=-1)

            # Sequences, which are finished should contain padding - taken from huggingface transformers.
            next_tokens = next_tokens * unfinished_sequences + generation_config.pad_token_id * (1 - unfinished_sequences)
            output_tokens.append(next_tokens)

            unfinished_sequences = unfinished_sequences & ~(next_tokens == generation_config.eos_token_id)

            hidden_states = self.model.embed_tokens(next_tokens).unsqueeze(1)
            seq_len += 1

        result = torch.cat((input_ids, torch.stack(output_tokens).permute([1, 0])), dim=1)
        return result


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