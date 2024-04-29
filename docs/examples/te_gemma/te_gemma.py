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
from torch.utils.cpp_extension import load



import transformer_engine as te
from transformer_engine.pytorch.attention import InferenceParams, RotaryPositionEmbedding
from transformer_engine.pytorch.fp8 import fp8_model_init
from transformer_engine.common.recipe import Format, DelayedScaling

import transformers
from transformers.models.gemma.modeling_gemma import GemmaModel, GemmaForCausalLM, GemmaRMSNorm, GemmaConfig
from transformers.modeling_utils import _add_variant, load_state_dict, _load_state_dict_into_model
from transformers.utils import WEIGHTS_INDEX_NAME
from transformers.utils.hub import get_checkpoint_shard_files

cuda = load(
    name='attention_copy',
    sources=['attention_copy.cu'],
    verbose=True
)


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
                inference_params=None,
                self_attn_mask_type='causal',
                **kwargs):
        """
        Custom forward to make sure we only pass relevant arguments to the
        forward pass of the `TransformerLayer`. Also, make sure the output
        format matches the output of the HF's `GemmaDecoderLayer`.
        """
        return (super().forward(hidden_states, attention_mask=attention_mask, rotary_pos_emb=self.te_rope_emb, inference_params=inference_params, self_attn_mask_type=self_attn_mask_type),)

class TeGraphed(torch.nn.Module):
    def __init__(self, model, lm_head, inference_params, normalizer, generation_config, thd=True):
        super().__init__()
        self.model = model
        self.inference_params = inference_params
        self.inference_params.thd = thd
        self.thd=thd
        self.normalizer = normalizer 
        self.generation_config = generation_config
        self.lm_head = lm_head

        
        self.attn_mask =  torch.ones([inference_params.max_batch_size, inference_params.max_sequence_length]).to(dtype=torch.bool)

    def forward(self,
                hidden_states, 
                unfinished_sequences, 
                seq_len, 
                vl_space, 
                kl_space, 
                ql_space,
                seqlens_q, 
                cu_seqlens_q, 
                cu_seqlens_kv, 
                seq_offsets_q, 
                seq_offsets_k, 
                seq_offsets_v, 
                position_embedding_matrix,
                k_pos_emb,
                q_pos_emb,
                *args
                ):
        hidden_states.data[:] = hidden_states.data[:] * self.normalizer
        inference_params = InferenceParams(self.inference_params.max_batch_size, self.inference_params.max_sequence_length)
        inference_params.thd = self.thd
        inference_params.seq_len = seq_len
        inference_params.value_layer = vl_space
        inference_params.key_layer = kl_space
        inference_params.query_layer = ql_space
        inference_params.seqlens_q = seqlens_q
        inference_params.cu_seqlens_q = cu_seqlens_q
        inference_params.cu_seqlens_kv = cu_seqlens_kv
        inference_params.seq_offsets_q = seq_offsets_q
        inference_params.seq_offsets_k = seq_offsets_k
        inference_params.seq_offsets_v = seq_offsets_v
        inference_params.position_embedding_matrix = position_embedding_matrix
        inference_params.k_pos_emb = k_pos_emb
        inference_params.q_pos_emb = q_pos_emb

        assert len(args) == 28 * 2


        for i in range(0, len(args), 2):
            inference_params.key_value_memory_dict[i // 2 + 1] = (args[i], args[i + 1])

        for decoder_layer in self.model.layers:
            hidden_states.copy_(decoder_layer(
                        hidden_states,
                        inference_params=inference_params,
                        self_attn_mask_type='padding',
                        attention_mask=None
                    )[0])
            

        seq_len.copy_(seq_len + 1)

        hidden_states.copy_(self.model.norm(hidden_states))
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        logits = logits[:, -1, :]
        next_tokens = torch.argmax(logits, dim=-1)

        # Sequences, which are finished should contain padding - taken from huggingface transformers.
        next_tokens = next_tokens * unfinished_sequences + self.generation_config.pad_token_id * (1 - unfinished_sequences)

        unfinished_sequences.copy_(unfinished_sequences & ~(next_tokens == self.generation_config.eos_token_id))
        
        hidden_states.copy_(self.model.embed_tokens(next_tokens).unsqueeze(1))
        return next_tokens, logits

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
    def from_pretrained_local(cls, pretrained_model_name_or_path, *args, config, fp8_init=False, **kwargs):
        """
        Custom method adapted from `from_pretrained` method in HuggingFace
        Transformers repo: https://github.com/huggingface/transformers/blob/f497f564bb76697edab09184a252fc1b1a326d1e/src/transformers/modeling_utils.py#L2579
        """
        
        with fp8_model_init(fp8_init):
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
            replace_params(state_dict, vanilla_model.state_dict(), config, fp8_init=config.fuse_qkv_params)
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
        use_cuda_graphs = False,
        **kwargs,
    ):
        
        batch_size, seq_len = input_ids.shape
        generation_config, _ = self._prepare_generation_config(generation_config, **kwargs)
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

        # inference_params object is a cache, where keys and values of previous tokens are stored
        inference_params = te.pytorch.InferenceParams(
            max_batch_size=batch_size, 
            max_sequence_length=max(128, input_ids.shape[1] + max_new_tokens)
        )

        # mask has shape [batch_size, num_heads, 1, max_seq_len] and contains False 
        # when coressponding token is padding and True otherwise.
        pad_attention_mask = input_ids.ne(generation_config.pad_token_id).unsqueeze(1).unsqueeze(2)
        
        #############################################################################################
        #                                        Encode part                                        #
        #############################################################################################



        hidden_states = self.model.embed_tokens(input_ids)
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        
        output_tokens = []
        hidden_states = hidden_states * normalizer
        for decoder_layer in self.model.layers:
            hidden_states = decoder_layer(
                hidden_states,
                # In the case of arbiutrary mask, the meaning of True and False is switched, so negation is needed.
                attention_mask=pad_attention_mask,
                self_attn_mask_type="padding_causal",
                inference_params=inference_params
            )[0]

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
        lengths = torch.sum(pad_attention_mask, dim=-1).squeeze()

        
        def process(x):
            """
            Args:
                x: Tensor with shape [s, b, h, d], where s is sequence length, b is batch size, h is number of heads, and d is hidden dimension.
                l: List of integers representing the actual lengths of each sequence in the batch before padding.

            Returns:
                torch.Tensor: Tensor with switched contents such that padded zeros are moved to the end of the sequence.
            """
            s1, b, h, d = x.shape
            s = torch.max(lengths)
            new_x = torch.zeros_like(x)
            
            for i in range(b):
                seq_length = lengths[i]
                
                # Check if the sequence length is not the full length of the sequence dimension
                if seq_length < s:
                    # Place the original data to the end part of the new tensor
                    new_x[:seq_length, i, :, :] = x[s - seq_length:s, i, :, :]
                    # Place the padding at the beginning of the new tensor
                    new_x[seq_length:, i, :, :] = 0
                else:
                    # If seq_length is the full length, just copy the entire sequence as is
                    new_x[:, i, :, :] = x[:, i, :, :]

            return new_x.permute((1, 0, 2, 3)).contiguous().cuda()

        inference_params.seq_len = lengths.to(torch.int32)
        seq_len_offset = torch.max(lengths).item() 


        seqlens_q = torch.zeros((batch_size), dtype=torch.int32).cuda()
        cu_seqlens_q = torch.zeros((batch_size + 1), dtype=torch.int32).cuda()
        cu_seqlens_kv = torch.zeros((batch_size + 1), dtype=torch.int32).cuda()
        seq_offsets_q = torch.zeros((batch_size + 1), dtype=torch.int32).cuda()
        seq_offsets_k = torch.zeros((batch_size + 1), dtype=torch.int32).cuda()
        seq_offsets_v = torch.zeros((batch_size + 1), dtype=torch.int32).cuda()




        
        for k, v in inference_params.key_value_memory_dict.items():
            key_layer = process(v[0])
            value_layer = process(v[1])
            inference_params.key_value_memory_dict[k] = (key_layer, value_layer)

        #############################################################################################
        #                                      Generate part                                        #
        #############################################################################################
        print("generate part")


        graphed_generator = TeGraphed(
            lm_head=self.lm_head,
            model=self.model, 
            inference_params=inference_params, 
            normalizer=normalizer, 
            generation_config=generation_config, 
            thd=True
        )

        tensor_pointers = [(kc, vc) for kc, vc in inference_params.key_value_memory_dict.values()]
        tensor_pointers = [element for tuple_ in tensor_pointers for element in tuple_]

        copy_hidden = hidden_states.clone()
        copy_unfinished_sequences = unfinished_sequences.clone()
        copy_tensor_pointers = [t.clone() for t in tensor_pointers]
        copy_seq_len = inference_params.seq_len.clone()

        vl_space = torch.zeros((batch_size, 1, 16, 256)).to(torch.bfloat16).cuda()
        kl_space = torch.zeros((batch_size, 1, 16, 256)).to(torch.bfloat16).cuda()
        ql_space = torch.zeros((batch_size, 1, 16, 256)).to(torch.bfloat16).cuda()
        q_pos_emb = torch.zeros((batch_size, 1, 1, 256)).to(torch.float32).cuda()
        k_pos_emb = torch.zeros((batch_size, 1, 1, 256)).to(torch.float32).cuda()


        te_rope = RotaryPositionEmbedding(256)
        position_embedding_matrix = te_rope(8192).to(torch.float32).cuda()
        
        
        graphed_layers = None
        if use_cuda_graphs:
            fp8_format = Format.HYBRID
            fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=32, amax_compute_algo="max")

            print("recording...")
            graphed_layers = te.pytorch.make_graphed_callables(
                graphed_generator, 
                (
                    hidden_states, 
                    unfinished_sequences, 
                    inference_params.seq_len, 
                    vl_space, 
                    kl_space, 
                    ql_space,
                    seqlens_q, 
                    cu_seqlens_q, 
                    cu_seqlens_kv, 
                    seq_offsets_q, 
                    seq_offsets_k, 
                    seq_offsets_v, 
                    position_embedding_matrix,
                    k_pos_emb,
                    q_pos_emb,
                    *tensor_pointers
                ), 
                fp8_enabled=True, 
                fp8_recipe=fp8_recipe, 
                allow_unused_input=True
                )
            print("recorded...")
        hidden_states.data[:] = copy_hidden
        unfinished_sequences.data[:] = copy_unfinished_sequences
        inference_params.seq_len.data[:] = copy_seq_len


        i = 0
        for t in tensor_pointers:
            t.data[:] = copy_tensor_pointers[i]
            i = i + 1

        for i in range(max_new_tokens):
            next_tokens, logits = graphed_layers(
                    hidden_states, 
                    unfinished_sequences, 
                    inference_params.seq_len, 
                    vl_space, 
                    kl_space, 
                    ql_space,
                    seqlens_q, 
                    cu_seqlens_q, 
                    cu_seqlens_kv, 
                    seq_offsets_q, 
                    seq_offsets_k, 
                    seq_offsets_v, 
                    position_embedding_matrix,
                    k_pos_emb,
                    q_pos_emb,
                    *tensor_pointers
                ) if use_cuda_graphs else graphed_generator(
                    hidden_states, 
                    unfinished_sequences, 
                    inference_params.seq_len, 
                    vl_space, 
                    kl_space, 
                    ql_space,
                    seqlens_q, 
                    cu_seqlens_q, 
                    cu_seqlens_kv, 
                    seq_offsets_q, 
                    seq_offsets_k, 
                    seq_offsets_v, 
                    position_embedding_matrix,
                    k_pos_emb,
                    q_pos_emb,
                    *tensor_pointers
                )
            output_tokens.append(next_tokens.clone())
            seq_len_offset += 1

        result = torch.cat((input_ids, torch.stack(output_tokens).permute([1, 0])), dim=1)
        return result


def replace_params(hf_state_dict, te_state_dict, config, fp8_init=False):
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
        
        if fp8_init:
            dst = te_state_dict[layer_prefix + 'self_attention.layernorm_qkv.weight']

            if layer_prefix + 'self_attn.q_proj.weight' in hf_state_dict:
                q =  hf_state_dict[layer_prefix + 'self_attn.q_proj.weight'] 
                for head_nr in range(config.num_attention_heads):
                    dst_offset = head_nr * config.head_dim * 3
                    # copy query
                    dst[dst_offset:(dst_offset + config.head_dim), :] = \
                        q[(head_nr * config.head_dim):(head_nr * config.head_dim + config.head_dim), :]

            if layer_prefix + 'self_attn.k_proj.weight' in hf_state_dict:
                k = hf_state_dict[layer_prefix + 'self_attn.k_proj.weight']
                for head_nr in range(config.num_attention_heads):
                    dst_offset = head_nr * config.head_dim * 3
                    # copy query
                    dst[( dst_offset + config.head_dim):(dst_offset + 2 * config.head_dim), :] = \
                        k[(head_nr * config.head_dim):(head_nr * config.head_dim + config.head_dim), :]
            
            if layer_prefix + 'self_attn.v_proj.weight' in hf_state_dict:
                v = hf_state_dict[layer_prefix + 'self_attn.v_proj.weight']
                for head_nr in range(config.num_attention_heads):
                    dst_offset = head_nr * config.head_dim * 3
                    dst[(dst_offset + 2 * config.head_dim):(dst_offset + 3 * config.head_dim), :] = \
                        v[(head_nr * config.head_dim):(head_nr * config.head_dim + config.head_dim), :]
        else:
        
            if layer_prefix + 'self_attn.q_proj.weight' in hf_state_dict:
                te_state_dict[layer_prefix + 'self_attention.layernorm_qkv.query_weight'].copy_(hf_state_dict[layer_prefix + 'self_attn.q_proj.weight'])

            if layer_prefix + 'self_attn.k_proj.weight' in hf_state_dict:
                te_state_dict[layer_prefix + 'self_attention.layernorm_qkv.key_weight'].copy_(hf_state_dict[layer_prefix + 'self_attn.k_proj.weight'])


            if layer_prefix + 'self_attn.v_proj.weight' in hf_state_dict:
                te_state_dict[layer_prefix + 'self_attention.layernorm_qkv.value_weight'].copy_(hf_state_dict[layer_prefix + 'self_attn.v_proj.weight'])

        if layer_prefix + 'self_attn.o_proj.weight' in hf_state_dict:
            te_state_dict[layer_prefix + 'self_attention.proj.weight'].copy_(hf_state_dict[layer_prefix + 'self_attn.o_proj.weight'])

        if layer_prefix + 'post_attention_layernorm.weight' in hf_state_dict:
            te_state_dict[layer_prefix + 'layernorm_mlp.layer_norm_weight'].copy_(1 + hf_state_dict[layer_prefix + 'post_attention_layernorm.weight'])
        
        if layer_prefix + 'mlp.gate_proj.weight' in hf_state_dict:
            te_state_dict[layer_prefix + 'layernorm_mlp.fc1_weight'].data[:GATE_PROJ_SIZE].copy_(hf_state_dict[layer_prefix + 'mlp.gate_proj.weight'])

        if layer_prefix + 'mlp.up_proj.weight' in hf_state_dict:
            te_state_dict[layer_prefix + 'layernorm_mlp.fc1_weight'].data[GATE_PROJ_SIZE:].copy_(hf_state_dict[layer_prefix + 'mlp.up_proj.weight'])

        if layer_prefix + 'mlp.down_proj.weight' in hf_state_dict:
            te_state_dict[layer_prefix + 'layernorm_mlp.fc2_weight'].copy_(hf_state_dict[layer_prefix + 'mlp.down_proj.weight'].data[:])



    return all_layer_prefixes