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
import transformer_engine as te
from transformer_engine.pytorch.attention import InferenceParams, RotaryPositionEmbedding
from transformer_engine.pytorch.fp8 import fp8_model_init
from transformer_engine.common.recipe import Format, DelayedScaling

import transformers
from transformers.models.gemma.modeling_gemma import GemmaForCausalLM, GemmaConfig
from transformers.modeling_utils import _add_variant, load_state_dict, _load_state_dict_into_model
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
            fuse_qkv_params=config.fuse_qkv_params,
            normalization="RMSNorm",
            activation="geglu",
            attn_input_format=config.qkv_format,
            num_gqa_groups=config.num_key_value_heads,
            attention_hidden_size=4096,
            layer_number=(layer_idx+1),
            zero_centered_gamma=True
        )
        te_rope = RotaryPositionEmbedding(256)
        self.te_rope_emb = te_rope(max_seq_len=config.max_position_embeddings).cuda()

    def forward(self,
                hidden_states,
                attention_mask,
                inference_params=None,
                self_attn_mask_type='causal'):
        """
        Custom forward to make sure we only pass relevant arguments to the
        forward pass of the `TransformerLayer`. Also, make sure the output
        format matches the output of the HF's `GemmaDecoderLayer`.
        """
        return (super().forward(
            hidden_states, 
            attention_mask=attention_mask, 
            rotary_pos_emb=self.te_rope_emb, 
            inference_params=inference_params, 
            self_attn_mask_type=self_attn_mask_type
            ),)

class StaticGemma(torch.nn.Module):
    def __init__(self, model, inference_params, dtype, mask, lm_head):
        super().__init__()
        self.model = model
        self.inference_params = inference_params
        self.normalizer = torch.tensor(self.model.config.hidden_size**0.5, dtype=dtype)
        self.mask = mask
        self.lm_head = lm_head
    
    def forward(self, hidden_states):

        hidden_states.data[:] = hidden_states.data[:] * self.normalizer
        for decoder_layer in self.model.layers:
            hidden_states.copy_(decoder_layer(
                hidden_states,
                attention_mask=None,
                self_attn_mask_type=self.mask,
                inference_params=self.inference_params
            )[0])

        hidden_states.copy_(self.model.norm(hidden_states))
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        return logits


class GemmaGenerator(torch.nn.Module):
    def __init__(self, model, lm_head, inference_params, dtype, generation_config):
        super().__init__()
        self.model = model
        self.inference_params = inference_params
        self.normalizer = torch.tensor(self.model.config.hidden_size**0.5, dtype=dtype) 
        self.generation_config = generation_config
        self.lm_head = lm_head
        self.gemma_layers = StaticGemma(model, inference_params, dtype, 'padding', lm_head)

    def forward(self, hidden_states, unfinished_sequences):
        logits = self.gemma_layers(hidden_states)
        logits = logits[:, -1, :]
        next_tokens = torch.argmax(logits, dim=1)

        self.inference_params.seq_len.copy_(self.inference_params.seq_len + 1)

        # Sequences, which are finished should contain padding - taken from huggingface transformers.
        next_tokens = next_tokens * unfinished_sequences + self.generation_config.pad_token_id * (1 - unfinished_sequences)
        unfinished_sequences.copy_(unfinished_sequences & ~(next_tokens == self.generation_config.eos_token_id))
        hidden_states.copy_(self.model.embed_tokens(next_tokens).unsqueeze(1))

        return next_tokens

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

        gemma_for_causal_lm.generate = TEGemmaForCausalLM.generate.__get__(gemma_for_causal_lm, GemmaForCausalLM)

        return gemma_for_causal_lm

    @classmethod
    def from_pretrained_local(cls, pretrained_model_name_or_path, *args, config, fp8_init=False, qkv_format="bshd", **kwargs):
        """
        Custom method adapted from `from_pretrained` method in HuggingFace
        Transformers repo: 
        https://github.com/huggingface/transformers/blob/f497f564bb76697edab09184a252fc1b1a326d1e/src/transformers/modeling_utils.py#L2579
        """
        config.qkv_format = qkv_format
        with fp8_model_init(fp8_init):
            vanilla_model = cls(config)
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

        resolved_archive_file, _ = get_checkpoint_shard_files(
                pretrained_model_name_or_path,
                archive_file,
        )

        # If the checkpoint is not sharded, it's a trivial sharding case
        if not is_sharded:
            assert not isinstance(resolved_archive_file, list)
            resolved_archive_file = [resolved_archive_file]

        total_dict = {}
        for shard_file in resolved_archive_file:
            state_dict = load_state_dict(shard_file)
            total_dict = total_dict | state_dict
        replace_params(total_dict, vanilla_model.state_dict(), config, qkv_fused_and_interleaved=config.fuse_qkv_params)
        _load_state_dict_into_model(vanilla_model, total_dict, start_prefix="") # Copy parameters like embedding.

        # Force mem release. Taken from huggingface code
        del total_dict
        gc.collect()
        return vanilla_model
    
    @staticmethod
    def _padding_to_end(inputs, lengths):
        """
        Gets the tensor with sequence padded from the beginning and
        return tensor padded from its end.

        Parameters
        ----------
        inputs : Tensor, tensor with shape [b, s] containing token numbers. 
                 It's padded from the beggining.
        lengths: Tensor, tensor with shape [s] with lengths of the sequences.

        """
        max_seq_len = torch.max(lengths)
        batch_size, max_seq_len = inputs.shape
        new_input_ids = inputs.clone()
        for i in range(batch_size):
            new_input_ids[i,:lengths[i]] = inputs[i, (max_seq_len-lengths[i]):max_seq_len]
            new_input_ids[i,lengths[i]:] = inputs[i, 0:(max_seq_len-lengths[i])]
        inputs.copy_(new_input_ids)
    
    def _generate_context_phase(
            self,
            gemma_layers,
            input_ids,
            inference_params,
            pad_token_id,
            eos_token_id,
            unfinished_sequences
    ):
        hidden_states = self.model.embed_tokens(input_ids)
        logits = gemma_layers(hidden_states)
        logits = logits[torch.arange(logits.size(0)), inference_params.incoming_seq_len - 1, :]
        next_tokens = torch.argmax(logits, dim=1)

        # Sequences, which are finished should contain padding - taken from huggingface transformers.
        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        unfinished_sequences = unfinished_sequences & ~(next_tokens == eos_token_id)
        hidden_states = self.model.embed_tokens(next_tokens).unsqueeze(1)
        return hidden_states, [next_tokens]

    
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        max_new_tokens: int = 0,
        use_cuda_graphs: bool = False,
        **kwargs,
    ): 
        batch_size, max_input_sequence_len = input_ids.shape
        generation_config, _ = self._prepare_generation_config(generation_config, **kwargs)
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

        # InferenceParams is a cache, where keys and values of previous tokens are stored.
        inference_params = InferenceParams(
            max_batch_size=batch_size, 
            max_sequence_length=max_input_sequence_len + max_new_tokens
        )

        # lengths is a tensor of shape [s] representing lengths of sequences.
        lengths = torch.sum(input_ids.ne(generation_config.pad_token_id), dim=-1).squeeze()
        inference_params.seq_len = torch.zeros_like(lengths).to(torch.int32).clone().cuda()
        inference_params.incoming_seq_len = lengths.to(torch.int32).clone().cuda()
        inference_params.max_incoming_seq_len = input_ids.shape[1]
        
        TEGemmaForCausalLM._padding_to_end(input_ids, lengths)

        context_phase_layers = StaticGemma(self.model, inference_params, torch.float32, 'padding_causal', self.lm_head)
        
        hidden_states, output_tokens = TEGemmaForCausalLM._generate_context_phase(
            self,
            context_phase_layers,
            input_ids,
            inference_params,
            generation_config.pad_token_id,
            generation_config.eos_token_id,
            unfinished_sequences
        )

        inference_params.seq_len.copy_(inference_params.incoming_seq_len)
        inference_params.incoming_seq_len.copy_(torch.ones_like(inference_params.incoming_seq_len))
        inference_params.max_incoming_seq_len = 1

        generator = GemmaGenerator(
            lm_head=self.lm_head,
            model=self.model, 
            inference_params=inference_params, 
            generation_config=generation_config, 
            dtype=hidden_states.dtype,
        )

        args = (hidden_states, unfinished_sequences)

        saved_args = [arg.clone() for arg in args] # Warmup iterations of graph will change the arguments, we want to revert that.
        if use_cuda_graphs:
            fp8_format = Format.HYBRID
            fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=32, amax_compute_algo="max")
            graphed_generator = te.pytorch.make_graphed_callables(
                generator, 
                args, 
                fp8_enabled=True, 
                fp8_recipe=fp8_recipe, 
                allow_unused_input=True,
                num_warmup_iters=10
            )
            
        for i in range(len(saved_args)):
            args[i].copy_(saved_args[i])
        inference_params.seq_len.copy_(lengths.to(torch.int32))

        for i in range(max_new_tokens):
            next_tokens = graphed_generator(*args) if use_cuda_graphs else generator(*args)
            output_tokens.append(next_tokens.clone())

        result = torch.cat((input_ids, torch.stack(output_tokens).permute([1, 0])), dim=1)
        return result

def _get_all_layer_prefixes_to_update(hf_state_dict):
    """
        There are many parameters in hf_state_dict, whose name start with model.layers.[number].
        This function extracts all strings like "model.layers.[number]." that are starting strings of keys in hf_state_dict.
    """
    all_layer_prefixes = set()
    for param_key in hf_state_dict.keys():
        layer_prefix_pat = 'model.layers.\d+.'
        m = re.match(layer_prefix_pat, param_key)
        if m is not None:
            all_layer_prefixes.add(m.group())
    return all_layer_prefixes

def replace_params(hf_state_dict, te_state_dict, config, qkv_fused_and_interleaved=False):
    """
    Replaces params from TE TransformerLayer state_dict with corresponding parameters 
    from HuggingFace GemmaModel state_dict.
    """
    all_layer_prefixes : List[str] = _get_all_layer_prefixes_to_update(hf_state_dict)
    
    for layer_prefix in all_layer_prefixes:
        def copy_from_ht_to_te(te_name, hf_name, start=None, end=None):
            te_state_dict[layer_prefix + te_name].data[start:end].copy_(hf_state_dict[layer_prefix + hf_name])

        copy_from_ht_to_te('self_attention.layernorm_qkv.layer_norm_weight', 'input_layernorm.weight')
        copy_from_ht_to_te('self_attention.proj.weight', 'self_attn.o_proj.weight')
        copy_from_ht_to_te('layernorm_mlp.layer_norm_weight', 'post_attention_layernorm.weight')
        copy_from_ht_to_te('layernorm_mlp.fc2_weight', 'mlp.down_proj.weight')
        copy_from_ht_to_te('layernorm_mlp.fc1_weight', 'mlp.gate_proj.weight', end=config.intermediate_size)
        copy_from_ht_to_te('layernorm_mlp.fc1_weight', 'mlp.up_proj.weight', start=config.intermediate_size)

        if qkv_fused_and_interleaved:
            """
                When qkv_fused_and_interleaved=True, key, query and value layers are on one tensor
                in TE TransformerLayer. Moreover they are interleaved within each head. 
                Let q_i, k_i and v_i be query, key and value layers for i-th head respectively.
                Then TE stores weight tensor in the form:
                [q1 k1 v1 q2 k2 v2 ...]
                This is done to maximally optimize performance time.
            """
            te_qkv_layer = te_state_dict[layer_prefix + 'self_attention.layernorm_qkv.weight']
            def copy_interleave(hf_name, idx):
                src = hf_state_dict[layer_prefix + hf_name] 
                for head_nr in range(config.num_attention_heads):
                    dst_offset = head_nr * config.head_dim * 3
                    te_qkv_layer[(dst_offset + idx * config.head_dim):(dst_offset + (idx + 1) * config.head_dim), :] = \
                        src[(head_nr * config.head_dim):(head_nr * config.head_dim + config.head_dim), :]
            copy_interleave('self_attn.q_proj.weight', 0)
            copy_interleave('self_attn.k_proj.weight', 1)
            copy_interleave('self_attn.v_proj.weight', 2)
        else:
            copy_from_ht_to_te('self_attention.layernorm_qkv.query_weight', 'self_attn.q_proj.weight')
            copy_from_ht_to_te('self_attention.layernorm_qkv.key_weight', 'self_attn.k_proj.weight')
            copy_from_ht_to_te('self_attention.layernorm_qkv.value_weight', 'self_attn.v_proj.weight')

    return all_layer_prefixes