# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from contextlib import contextmanager

from typing import Optional

from transformers.generation import *
from transformers.generation.utils import *

import torch
import transformer_engine as te
from transformer_engine.pytorch.attention import InferenceParams, RotaryPositionEmbedding
from transformer_engine.common.recipe import Format, DelayedScaling
from torch.cuda.amp import autocast

import transformers
from transformers.models.gemma.modeling_gemma import GemmaForCausalLM, GemmaConfig, GemmaModel

import torch.nn.functional as F


class TEGemmaDecoderLayer(te.pytorch.TransformerLayer):
    """
    Wrapper class over TE's `TransformerLayer`. This makes the wrapper very
    similar to HF's `GemmaDecoderLayer` and easier to replace it in the code.

    Args:
        config: GemmaConfig
        args: positional args (for compatibility with `GemmaDecoderLayer`)
        kwargs: keyword args (for compatibility with `GemmaDecoderLayer`)
    """
    def __init__(self, config : GemmaConfig, layer_idx : int, *args, **kwargs):
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
            layer_number=(layer_idx+1), # Layer numbers in TE starts from 1, not from 0 like in the HF.
            zero_centered_gamma=True
        )
        self.te_rope_emb = RotaryPositionEmbedding(256)(max_seq_len=config.max_position_embeddings).cuda()

    def forward(self, *args, **kwargs): # We need to pass positional encoding.
        # this args cannot be passed to TransformerLayer
        keys_to_remove = ["position_ids", "past_key_value", "output_attentions", "use_cache", "cache_position"]
        for key in keys_to_remove:
            kwargs.pop(key, None)
        # We need to return tuple to be compatible with HF.
        return (super().forward(*args, rotary_pos_emb=self.te_rope_emb, **kwargs),) 

class StaticGemmaModel(torch.nn.Module):
    """
        StaticGemma is based of HF GemmaModel class.
        It is adjusted to work properly with CUDA Graphs.
    """
    def __init__(
            self, 
            model : GemmaModel, 
            dtype : torch.dtype, 
            mask : torch.Tensor, 
            lm_head : torch.nn.Module, 
        ):
        super().__init__()
        self.model = model
        self.normalizer = torch.tensor(self.model.config.hidden_size**0.5, dtype=dtype)
        self.mask = mask
        self.lm_head = lm_head

    def set_inference_params(self, inference_params):
        self.inference_params = inference_params
    
    def forward(self, hidden_states : torch.Tensor, attention_mask : torch.Tensor = None):
        with torch.no_grad():
            hidden_states.data[:] = hidden_states.data[:] * self.normalizer # static operation - for CUDA graphs
            for decoder_layer in self.model.layers:
                hidden_states.data[:] = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    self_attn_mask_type=self.mask,
                    inference_params=self.inference_params
                )[0] # static copy - for CUDA graphs

        hidden_states.copy_(self.model.norm(hidden_states)) # static copy - for CUDA graphs
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        return logits


class GemmaGenerator(torch.nn.Module):
    """
        GemmaGenerator gets one layer of embeddins,
        makes forward pass and returns next tokens.
    """
    def __init__(self, model : GemmaModel, lm_head: torch.nn.Module, dtype : torch.dtype, qkv_format : str):
        super().__init__()
        self.model = model
        self.gemma_layers = StaticGemmaModel(model, dtype, 'padding', lm_head)
        self.qkv_format = qkv_format
    
    def set_inference_params(self, inference_params):
        self.inference_params = inference_params
        self.gemma_layers.set_inference_params(inference_params)

    def forward(self, hidden_states : torch.Tensor, mask : torch.Tensor = None):
        logits = self.gemma_layers(hidden_states, attention_mask=mask)

        assert logits.shape[0] == hidden_states.shape[0] # b
        assert logits.shape[1] == hidden_states.shape[1] # seq_len
        # logits.shape[2] = number of tokens
        logits = logits[:, -1, :]
        next_tokens = torch.argmax(logits, dim=1)

        hidden_states.copy_(self.model.embed_tokens(next_tokens).unsqueeze(1)) # static copy for CUDA graphs

        # self.inference_params contains for example kv_cache
        # This needs to be called before every pass, 
        # to update the information of sequence lengths.
        # Here we increase sequence offsets by one, 
        # because we generated one token for every sequence.
        self.inference_params.setup_before_new_input(next_tokens.unsqueeze(1))

        return next_tokens

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


class TEGemmaForCausalLM(GemmaForCausalLM):
    """
    Causal LM created with `GemmaModel`. The underlying `GemmaDecoderLayer`
    class is monkey-patched with `TEGemmaDecoderLayer` class before
    initializing the causal LM with `GemmaForCausalLM`.

    Args:
        config: GemmaConfig
    """

    def __init__(self, config: GemmaConfig):
        with replace_decoder(te_decoder_cls=TEGemmaDecoderLayer):
            super().__init__(config)
        self.to(torch.bfloat16).cuda()
        self.hidden_size = config.hidden_size
        self._model_generation_phase = GemmaGenerator(
            lm_head=self.lm_head,
            model=self.model, 
            dtype=torch.bfloat16,
            qkv_format=config.qkv_format
        )
        self._model_context_phase = StaticGemmaModel(self.model, torch.bfloat16, 'padding_causal', self.lm_head)

        if self.config.fp8:
            self.fp8_recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")

    
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
    
    def _next_64_multiply(self, x):
        return ((x + 63) // 64) * 64
    
    # This function is overriden in TeGEmmaForCausalLMCudaGraphs.
    def _create_hidden_states_buffer(self, input_ids : torch.Tensor):
        return torch.empty((input_ids.shape[0], input_ids.shape[1], self.hidden_size), device="cuda", dtype=torch.float32)

    # This function is overriden in TeGEmmaForCausalLMCudaGraphs.
    def _create_inference_params(self, max_batch_size : int, max_sequence_length : int):
        return InferenceParams(max_batch_size, max_sequence_length, qkv_format=self.config.qkv_format)

    # This function is overriden in TeGEmmaForCausalLMCudaGraphs.
    def _get_max_input_seq_len(self, input_ids):
        return input_ids.shape[1]
    
    # The buffer for generation is some part (beginning) of hidden states buffer.
    # This function returns pointer to it and also copies there data if provided.
    def _get_generation_buffer(self, hidden_states_buffer, data_to_copy=None):
        # hidden_states_buffer has shape [b, s, hd]
        # generation_buffer will have shape [b, 1, hd]
        # Notice that "generation_buffer = hidden_states_buffer[:, 0, :].unsqueeze(1)"
        # will return uncontiguous buffer, which we want to avoid.
        output = hidden_states_buffer.view(-1)[:hidden_states_buffer.shape[0] * hidden_states_buffer.shape[2]]
        if data_to_copy is not None:
            output.copy_(data_to_copy.reshape(-1))
        generation_buffer = output.view((hidden_states_buffer.shape[0], 1, hidden_states_buffer.shape[2]))
        return generation_buffer

    def _generate_context_phase(
            self,
            input_ids : torch.Tensor,
            inference_params : InferenceParams
    ):
        hidden_states = self._create_hidden_states_buffer(input_ids)
        hidden_states.data[:] = self.model.embed_tokens(input_ids)
        
        
        # We need to update offsets before every forward pass to make cache work properly.
        inference_params.setup_before_new_input(input_ids, pad_token_id=0, reset=True)
 
        hidden_states.data[:] = self.model.embed_tokens(input_ids)
        logits = self._model_context_phase(
            hidden_states, 
            attention_mask=((input_ids == 0) if self.config.qkv_format != "thd" else None)
        )

        # We choose logits coresponding with last token in each sequence,
        # which have various lengths - they are stored in (inference_params.incoming_seq_len - 1) Tensor
        # when qkv_format == "thd" and they are the last token in the sequence when qkv_format != "thd".
        if self.config.qkv_format == "thd":
            logits = logits[torch.arange(logits.size(0)), inference_params.input_sequence_lengths - 1, :]
        else:
            logits = logits[:, -1, :]
        next_tokens = torch.argmax(logits, dim=1)

        # self.hidden_states have shape [b, s, hd].
        # We return hidden state for the last token - output has shape [b, 1, hd]
        hidden_states = self._get_generation_buffer(hidden_states, self.model.embed_tokens(next_tokens))
        return hidden_states, next_tokens

    def _make_mask_one_token_longer(self, mask):
        return torch.cat(
            [mask, torch.zeros(mask.size(0), 1, 1, 1, dtype=torch.bool, device=mask.device)], 
            dim=-1
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pad_token_id: int = 0,
        max_new_tokens: int = 0,
        *args, **kwargs
    ): 
        self.eval()

        # We need both autocasts: FP8 for operations that can run in lower precision 
        # and BF16 for those that cannot.
        with autocast(dtype=torch.bfloat16, cache_enabled=False), \
             te.pytorch.fp8_autocast(
                 enabled=self.config.fp8, fp8_recipe=self.fp8_recipe if self.config.fp8 else None):
            
            batch_size, max_input_sequence_len = input_ids.shape[0], self._get_max_input_seq_len(input_ids)
            lengths = torch.sum(input_ids.ne(pad_token_id), dim=-1).squeeze() # [s]
            input_ids = F.pad(input_ids, (max_input_sequence_len - input_ids.shape[1], 0), "constant", 0)

            # InferenceParams is a cache, where keys and values of previous tokens are stored.
            # Moreover it stores length of both already generated and input sequences.
            inference_params = self._create_inference_params(
                max_batch_size=batch_size, 
                max_sequence_length=self._next_64_multiply(max_input_sequence_len + max_new_tokens)
            )

            self._model_context_phase.set_inference_params(inference_params)
            self._model_generation_phase.set_inference_params(inference_params)

            if self.config.qkv_format == "thd":
                # For thd layout padding is at the end, otherwise at the beginning.
                TEGemmaForCausalLM._padding_to_end(input_ids, lengths)

            hidden_states, next_tokens = self._generate_context_phase(
                input_ids,
                inference_params
            )

            # Generation phase.

            inference_params.setup_before_new_input(next_tokens.unsqueeze(1))
            
            output_tokens = [next_tokens]

            if self.config.qkv_format != "thd":
                mask = (input_ids == 0).unsqueeze(1).unsqueeze(1)

            for _ in range(max_new_tokens):
                if self.config.qkv_format != "thd":
                    # It will not work with cuda graphs, but it is not used for thd qkv_format.
                    mask = self._make_mask_one_token_longer(mask) 

                next_tokens = self._model_generation_phase(hidden_states, mask)
                # next_tokens is static output tensor, so we need to clone it - it gets changed every iteration.
                output_tokens.append(next_tokens.clone())

            result = torch.cat((input_ids, torch.stack(output_tokens).permute([1, 0])), dim=1)
            return result

class TEGemmaForCausalLMCudaGraphs(TEGemmaForCausalLM):
    """
        TEGemmaForCausalLMCudaGraphs is the version of the class TEGemmaForCausalLM using CUDA Graphs to speed it up.
        We need to make one trade-off. Namely, batch_size, max_seq_len and max_context_seq_len need to be static.
        It is necessary to run generation with the same value of these variables that we recorded graph on.
    """
    def __init__(self, config : GemmaConfig):
        super().__init__(config)
        assert config.qkv_format == "thd", "Generation with CUDA Graphs are implemented only for thd format."

        # Preparation of the static buffers.
        self.config = config 
        self.hidden_states_buffer = torch.empty(
            (config.cuda_graphs_static_batch_size, config.cuda_graphs_static_max_context_len, config.hidden_size)).cuda()
        # This is in fact part of the buffer for hidden_states.
        self.generation_buffer = self._get_generation_buffer(self.hidden_states_buffer) 
        self.inference_params = InferenceParams(
            max_batch_size=config.cuda_graphs_static_batch_size, 
            max_sequence_length=config.cuda_graphs_static_max_seq_len, 
            qkv_format="thd"
        )

        
        self._model_generation_phase.set_inference_params(self.inference_params)
        self._model_context_phase.set_inference_params(self.inference_params)
        
    def record(self):
        self.eval() # We want to record model in training=False, because it will be used in generation.

        # Here "the trick" happens. We override methods from TEGemmaForCausalLM
        # with their recorded version. After invocation of each of them,
        # captured graph will be replayed with minimal usage of CPU,
        # what will lead to huge speedup.
        input_shape = (self.config.cuda_graphs_static_batch_size, self.config.cuda_graphs_static_max_context_len)
        self.inference_params.setup_before_new_input(torch.randn(input_shape), reset=True)
        self._model_context_phase = self.record_graph(
            self._model_context_phase, self.hidden_states_buffer) # CUDA Graphs recording

        input_shape = torch.randn((self.config.cuda_graphs_static_batch_size, 1))
        self.inference_params.setup_before_new_input(input_shape, reset=True)        
        self._model_generation_phase = self.record_graph(
            self._model_generation_phase, self.generation_buffer) # CUDA Graphs recording

    """
        Functions _create_hidden_states_buffer and _create_inference_params from base class are overriden
        to make hidden_states and inference_params static 
        - not changing their position in memory between every invocation.
    """
    def _create_hidden_states_buffer(self, *args, **kwargs):
        return self.hidden_states_buffer

    def _create_inference_params(self, *args, **kwargs):
        return self.inference_params
    
    def _get_max_input_seq_len(self, _):
        return self.config.cuda_graphs_static_max_context_len

    @torch.no_grad()
    def record_graph(self, function, input_tensor):
        # function is invoked on argument (self.hidden_states,) and all kernels are recorded.
        # record_graph() returns captured function, which can be run later with minimal use of th CPU.
        fp8_format = Format.HYBRID
        fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=1024, amax_compute_algo="max")

        # We need both autocasts: FP8 for operations that can run in lower precision 
        # and BF16 for those that cannot.
        with autocast(dtype=torch.bfloat16, cache_enabled=False):
            graphed_function = te.pytorch.make_graphed_callables(
                function, 
                (input_tensor,), 
                fp8_enabled=self.config.fp8, 
                fp8_recipe=fp8_recipe, 
                allow_unused_input=True,
                num_warmup_iters=3
            )
        return graphed_function
