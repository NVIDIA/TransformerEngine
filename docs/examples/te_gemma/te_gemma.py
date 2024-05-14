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

import transformers
from transformers.models.gemma.modeling_gemma import GemmaForCausalLM, GemmaConfig, GemmaModel

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
        return super().forward(*args, rotary_pos_emb=self.te_rope_emb, **kwargs)


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
            inference_params : InferenceParams
        ):
        super().__init__()
        self.model = model
        self.normalizer = torch.tensor(self.model.config.hidden_size**0.5, dtype=dtype)
        self.mask = mask
        self.lm_head = lm_head
        self.inference_params = inference_params
    
    def forward(self, hidden_states : torch.Tensor):
        hidden_states.data[:] = hidden_states.data[:] * self.normalizer # static operation - for CUDA graphs
        for decoder_layer in self.model.layers:
            hidden_states.copy_(decoder_layer(
                hidden_states,
                attention_mask=None,
                self_attn_mask_type=self.mask,
                inference_params=self.inference_params
            )[0]) # static copy - for CUDA graphs

        hidden_states.copy_(self.model.norm(hidden_states)) # static copy - for CUDA graphs
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        return logits


class GemmaGenerator(torch.nn.Module):
    """
        GemmaGenerator gets one layer of embeddins,
        makes forward pass and returns next tokens.
    """
    def __init__(self, model : GemmaModel, lm_head: torch.nn.Module, inference_params : InferenceParams, dtype : torch.dtype):
        super().__init__()
        self.model = model
        self.gemma_layers = StaticGemmaModel(model, dtype, 'padding', lm_head, inference_params)
        self.inference_params = inference_params

    def forward(self, hidden_states : torch.Tensor):
        logits = self.gemma_layers(hidden_states)

        assert logits.shape[0] == hidden_states.shape[0] # b
        # logits.shape[1] = number of tokens
        assert logits.shape[2] == hidden_states.shape[2] # hidden_dim
        logits = logits[:, -1, :]
        next_tokens = torch.argmax(logits, dim=1)

        hidden_states.copy_(self.model.embed_tokens(next_tokens).unsqueeze(1))

        # self.inference_params contains for example kv_cache
        # This needs to be called before every pass, 
        # to update the information of sequence lengths.
        # Here we increase sequence offsets by one, 
        # because we generated one token for every sequence.
        self.inference_params.set_before_new_input(hidden_states, offsets_change="+1")

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
        self.hidden_states = None
    
    # This function is overriden in TeGEmmaForCausalLMCudaGraphs.
    @torch.no_grad()
    def _model_generation_phase(self, hidden_states : torch.Tensor, inference_params : InferenceParams=None):
        generator = GemmaGenerator(
            lm_head=self.lm_head,
            model=self.model, 
            inference_params=inference_params,
            dtype=hidden_states.dtype,
        )
        return generator(hidden_states,)

    # This function is overriden in TeGEmmaForCausalLMCudaGraphs.
    @torch.no_grad()
    def _model_context_phase(self, hidden_states : torch.Tensor, inference_params : InferenceParams=None):
        layers = StaticGemmaModel(self.model, torch.float32, 'padding_causal', self.lm_head, inference_params)
        return layers(hidden_states)

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
    
    # This function is overriden in TeGEmmaForCausalLMCudaGraphs.
    def _create_hidden_states_buffer(self, input_ids : torch.Tensor):
        return torch.empty_like(input_ids, device="cuda", dtype=torch.float32)

    # This function is overriden in TeGEmmaForCausalLMCudaGraphs.
    def _create_inference_params(self, max_batch_size : int, max_sequence_length : int):
        return InferenceParams(max_batch_size, max_sequence_length)

    def _generate_context_phase(
            self,
            input_ids : torch.Tensor,
            inference_params : InferenceParams
    ):
        hidden_states = self._create_hidden_states_buffer(input_ids)
        hidden_states.data[:] = self.model.embed_tokens(input_ids)

        logits = self._model_context_phase(self.hidden_states, inference_params)

        # We choose logits coresponding with last token in each sequence,
        # which have various lengths - they are stored in (inference_params.incoming_seq_len - 1) Tensor.
        logits = logits[torch.arange(logits.size(0)), inference_params.incoming_seq_len - 1, :]
        next_tokens = torch.argmax(logits, dim=1)

        # self.hidden_states have shape [b, s, hd].
        # We return hidden state for the last token - output has shape [b, 1, hd]
        self.hidden_states.data[:, 0, :] = self.model.embed_tokens(next_tokens)
        return self.hidden_states[:, 0, :].unsqueeze(1), [next_tokens]

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pad_token_id: int = 0,
        max_new_tokens: int = 0
    ): 
        batch_size, max_input_sequence_len = input_ids.shape
        lengths = torch.sum(input_ids.ne(pad_token_id), dim=-1).squeeze() # [s]

        # InferenceParams is a cache, where keys and values of previous tokens are stored.
        # Moreover it stores length of both already generated and input sequences.
        inference_params = self._create_inference_params(
            max_batch_size=batch_size, 
            max_sequence_length=max_input_sequence_len + max_new_tokens
        )

        # We need to update offsets before every forward pass to make cache work properly.
        inference_params.set_before_new_input(input_ids, padding_token=pad_token_id, offsets_change="all_zero")

        # Context phase
        TEGemmaForCausalLM._padding_to_end(input_ids, lengths)
        hidden_states, output_tokens = TEGemmaForCausalLM._generate_context_phase(
            self,
            input_ids,
            self.inference_params
        )

        # Generation phase.
        self.inference_params.set_before_new_input(hidden_states, offsets_change=None)
        for _ in range(max_new_tokens):
            next_tokens = self._model_generation_phase(hidden_states, self.inference_params)
            output_tokens.append(next_tokens.clone())

        result = torch.cat((input_ids, torch.stack(output_tokens).permute([1, 0])), dim=1)
        return result

class TEGemmaForCausalLMCudaGraphs(TEGemmaForCausalLM):
    """
        TEGemmaForCausalLMCudaGraphs is the version of the class TEGemmaForCausalLM using CUDA Graphs to speed it up.
        We need to make one trade-off. Namely, batch_size, max_seq_len and max_context_seq_len need to be static.
        It is necessary to run generation with the same value of these variables that we recorded graph on.
    """
    def __init__(self, config : GemmaConfig, batch_size : int, max_seq_len : int, max_context_seq_len : int):
        super.__init(config)

        # Preparation of the static buffers.
        self.batch_size = batch_size 
        self.max_seq_len = max_seq_len
        self.hidden_states_buffer = torch.empty((batch_size, max_context_seq_len, self.config.hidden_dim)).cuda()
        self.inference_params = InferenceParams(max_batch_size=batch_size, max_sequence_length=max_seq_len)
        
        # Here "the trick" happens. We override methods from TEGemmaForCausalLM
        # with their recorded version. After invocation of each of them,
        # captured graph will be replayed with minimal usage of CPU,
        # what will lead to huge speedup.
        self._model_generation_phase = self.record_graph(super()._model_generation_phase)
        self._model_context_phase = self.record_graph(super()._model_context_phase)

    """
        Functions _create_hidden_states_buffer and _create_inference_params from base class are overriden
        to make hidden_states and inference_params static 
        - not changing their position in memory between every invocation.
    """
    def _create_hidden_states_buffer(self, *args):
        return self.hidden_states_buffer

    def _create_inference_params(self, *args):
        return self.inference_params

    @torch.no_grad()
    def record_graph(self, function):
        # function is invoked on argument (self.hidden_states,) and all kernels are recorded.
        # record_graph() returns captured function, which can be run later with minimal use of th CPU.
        fp8_format = Format.HYBRID
        fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=32, amax_compute_algo="max")
        graphed_function = te.pytorch.make_graphed_callables(
            function, 
            (self.hidden_states,), 
            fp8_enabled=True, 
            fp8_recipe=fp8_recipe, 
            allow_unused_input=True,
            num_warmup_iters=3
        )
        return graphed_function
    
    @torch.no_grad()
    def generate(
            self,
            input_ids: Optional[torch.Tensor] = None,
            *args,
            **kwargs,
        ): 
        assert self.batch_size == input_ids.shape[0], \
            f"Input_ids shape {input_ids.shape} does not match batch_size={self.batch_size} of recorded graphs" 
        assert self.max_seq_len == input_ids.shape[1], \
            f"Input_ids shape {input_ids.shape} does not match max_seq_len={self.max_seq_len} of recorded graphs" 

        super().generate(input_ids, *args, **kwargs)