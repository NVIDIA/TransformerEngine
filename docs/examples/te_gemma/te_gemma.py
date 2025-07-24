# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from contextlib import contextmanager

from typing import Optional
from functools import partial
from collections import OrderedDict

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

    def __init__(self, config: GemmaConfig, layer_idx: int, *args, **kwargs):

        self.gemma_config = config

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
            kv_channels=self.gemma_config.head_dim,
            layer_number=(
                layer_idx + 1
            ),  # Layer numbers in TE starts from 1, not 0 like in the HF.
            zero_centered_gamma=True,
        )

    def forward(self, *args, **kwargs):  # We need to additionally pass positional encoding.

        # filter out HF specific args
        keys_to_remove = [
            "position_ids",
            "past_key_value",
            "output_attentions",
            "use_cache",
            "cache_position",
        ]
        for key in keys_to_remove:
            kwargs.pop(key, None)

        rope_emb = kwargs.pop("rope_emb", None)

        # Return tuple to be compatible with HF.
        return (super().forward(*args, rotary_pos_emb=rope_emb, **kwargs),)


class StaticGemmaModel(torch.nn.Module):
    """
    StaticGemma is based of HF GemmaModel class.
    It is adjusted to work properly with CUDA Graphs.
    """

    def __init__(
        self,
        model: GemmaModel,
        dtype: torch.dtype,
        lm_head: torch.nn.Module,
    ):
        super().__init__()
        self.model = model
        self.normalizer = torch.tensor(self.model.config.hidden_size**0.5, dtype=dtype)
        self.lm_head = lm_head

    def set_inference_params(self, inference_params):
        self.inference_params = inference_params

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        attn_mask_type: str = "arbitrary",
        rope_emb: torch.Tensor = None,
    ):
        with torch.no_grad():
            # static operation - for CUDA graphs
            hidden_states.data[:] = hidden_states.data[:] * self.normalizer

            for i, decoder_layer in enumerate(self.model.layers):
                hidden_states.data[:] = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    self_attn_mask_type=self.mask if attn_mask_type is None else attn_mask_type,
                    inference_params=self.inference_params,
                    rope_emb=rope_emb,
                )[
                    0
                ]  # static copy - for CUDA graphs

        hidden_states.copy_(self.model.norm(hidden_states))  # static copy - for CUDA graphs
        logits = self.lm_head(hidden_states)

        # @sudhakars: This is probably not needed, need to check.
        logits = logits.float()
        return logits


class GemmaGenerator(torch.nn.Module):
    """
    GemmaGenerator gets one layer of embeddins,
    makes forward pass and returns next tokens.
    """

    def __init__(
        self, model: GemmaModel, lm_head: torch.nn.Module, dtype: torch.dtype, qkv_format: str
    ):
        super().__init__()
        self.model = model
        self.gemma_layers = StaticGemmaModel(model, dtype, lm_head)
        self.qkv_format = qkv_format

    def set_inference_params(self, inference_params):
        self.inference_params = inference_params
        self.gemma_layers.set_inference_params(inference_params)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor = None,
        attn_mask_type: str = "arbitrary",
        rope_emb: torch.Tensor = None,
    ):
        logits = self.gemma_layers(
            hidden_states, attention_mask=mask, attn_mask_type=attn_mask_type, rope_emb=rope_emb
        )

        assert logits.shape[0] == hidden_states.shape[0]  # b
        assert logits.shape[1] == hidden_states.shape[1]  # seq_len

        # Fetch the logits for the last token
        logits = logits[:, -1, :]
        next_tokens = torch.argmax(logits, dim=1)

        # static copy for CUDA graphs
        hidden_states.copy_(self.model.embed_tokens(next_tokens).unsqueeze(1))

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

        dtype = torch.bfloat16
        with replace_decoder(te_decoder_cls=TEGemmaDecoderLayer):
            super().__init__(config)

        self.config = config
        self.to(dtype).cuda()
        self.hidden_size = config.hidden_size
        self._model_generation_phase = GemmaGenerator(
            lm_head=self.lm_head,
            model=self.model,
            dtype=dtype,
            qkv_format=config.qkv_format,
        )
        self._model_context_phase = StaticGemmaModel(self.model, dtype, self.lm_head)

        if self.config.fp8:
            self.fp8_recipe = DelayedScaling(
                fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max"
            )

        # Rotary position embedding remains the same for all the layers and so
        # created here. This makes it compatible with CUDA Graphs too.
        self.te_rope_emb = RotaryPositionEmbedding(self.config.head_dim)(
            max_seq_len=self.config.max_position_embeddings
        ).cuda()

    @staticmethod
    def _padding_to_end(inputs, lengths, max_seq_len=None):
        """
        Gets the tensor with sequence padded from the beginning and
        updates it inplace to be padded from its end.

        Parameters
        ----------
        inputs : Tensor, tensor with shape [b, s] containing token numbers.
                 It's padded from the beggining.
        lengths: Tensor, tensor with shape [s] with lengths of the sequences.

        """
        max_seq_len = torch.max(lengths) if max_seq_len is None else max_seq_len
        batch_size, max_seq_len = inputs.shape
        new_input_ids = inputs.clone()
        for i in range(batch_size):
            new_input_ids[i, : lengths[i]] = inputs[i, (max_seq_len - lengths[i]) : max_seq_len]
            new_input_ids[i, lengths[i] :] = inputs[i, 0 : (max_seq_len - lengths[i])]

        # Trim the inputs to no extra padding i.e. fix the max seq len to
        # the longest sequence in the batch
        actual_max_seq_len = max_seq_len
        inputs.data = new_input_ids[:, :actual_max_seq_len]

    def _next_64_multiply(self, x):
        return ((x + 63) // 64) * 64

    # This function is overriden in TeGEmmaForCausalLMCudaGraphs.
    def _create_hidden_states_buffer(self, input_ids: torch.Tensor):
        tensor = torch.empty(
            (input_ids.shape[0], input_ids.shape[1], self.hidden_size),
            device="cuda",
            dtype=torch.float32,
        )
        return tensor

    # This function is overriden in TeGEmmaForCausalLMCudaGraphs.
    def _create_inference_params(self, *args, **kwargs):
        infer_params = InferenceParams(*args, **kwargs)
        return infer_params

    # This function is overriden in TeGEmmaForCausalLMCudaGraphs.
    def _get_max_input_seq_len(self, input_ids):
        return (
            input_ids.shape[1]
            if not hasattr(self.config, "cuda_graphs_static_max_context_len")
            else self.config.cuda_graphs_static_max_context_len
        )

    # The buffer for generation is some part (beginning) of hidden states buffer.
    # This function returns pointer to it and also copies there data if provided.
    def _get_generation_buffer(self, hidden_states_buffer, data_to_copy=None):
        # hidden_states_buffer has shape [b, s, hd]
        # generation_buffer will have shape [b, 1, hd]
        # Notice that "generation_buffer = hidden_states_buffer[:, 0, :].unsqueeze(1)"
        # will return uncontiguous buffer, which we want to avoid.
        output = hidden_states_buffer.view(-1)[
            : hidden_states_buffer.shape[0] * hidden_states_buffer.shape[2]
        ]
        if data_to_copy is not None:
            output.copy_(data_to_copy.reshape(-1))
        generation_buffer = output.view(
            (hidden_states_buffer.shape[0], 1, hidden_states_buffer.shape[2])
        )
        return generation_buffer

    def _generate_context_phase(self, input_ids: torch.Tensor, inference_params: InferenceParams):
        hidden_states = self._create_hidden_states_buffer(input_ids)
        hidden_states.copy_(self.model.embed_tokens(input_ids))

        # Update offsets before every forward pass (including context/prefill phase) to make
        # cache work properly.
        lengths = input_ids.ne(0).sum(dim=1)
        inference_params.pre_step(OrderedDict(zip(list(range(len(lengths))), lengths.tolist())))

        logits = self._model_context_phase(
            hidden_states,
            attention_mask=((input_ids == 0) if self.config.qkv_format != "thd" else None),
            attn_mask_type="padding_causal" if self.config.qkv_format == "thd" else "arbitrary",
            rope_emb=self.te_rope_emb,
        )

        if self.config.qkv_format == "thd":
            logits = logits[torch.arange(logits.size(0)), lengths - 1, :]
        else:
            logits = logits[:, -1, :]

        next_tokens = torch.argmax(logits, dim=1)

        # self.hidden_states have shape [b, s, hd].
        # We return hidden state for the last token - output has shape [b, 1, hd]
        hidden_states = self._get_generation_buffer(
            hidden_states, self.model.embed_tokens(next_tokens)
        )
        return hidden_states, next_tokens

    def _make_mask_one_token_longer(self, mask):
        return torch.cat(
            [mask, torch.zeros(mask.size(0), 1, 1, 1, dtype=torch.bool, device=mask.device)], dim=-1
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pad_token_id: int = 0,
        max_new_tokens: int = 0,
        *args,
        **kwargs
    ):
        self.eval()

        # We need both autocasts: FP8 for operations that can run in lower precision
        # and BF16 for those that cannot.
        with autocast(dtype=torch.bfloat16, cache_enabled=False), te.pytorch.fp8_autocast(
            enabled=self.config.fp8, fp8_recipe=self.fp8_recipe if self.config.fp8 else None
        ):

            lengths = torch.sum(input_ids.ne(pad_token_id), dim=-1).squeeze()  # [s]

            batch_size, max_input_sequence_len = input_ids.shape[0], self._get_max_input_seq_len(
                input_ids
            )

            if self.config.qkv_format == "thd":
                # For thd layout padding is at the end, otherwise at the beginning.
                TEGemmaForCausalLM._padding_to_end(
                    input_ids,
                    lengths,
                    max_seq_len=(
                        self.config.cuda_graphs_static_max_context_len
                        if self.config.generation_cuda_graphs
                        else None
                    ),
                )

            # InferenceParams is a cache, where keys and values of previous tokens are stored.
            # Moreover it stores length of both already generated and input sequences.
            inference_params = self._create_inference_params(
                max_batch_size=batch_size,
                max_sequence_length=128,
                num_heads_kv=self.config.num_key_value_heads,
                # num_heads_q=self.config.num_attention_heads,
                head_dim_v=self.config.head_dim,
                head_dim_k=self.config.head_dim,
                dtype=torch.bfloat16,
                is_paged=self.config.is_paged,
                page_size=64,
                total_num_pages=64 * 128 // 64,  # 64 * 64 (max_sequence_length) / 64 (page_size)
            )

            self._model_context_phase.set_inference_params(inference_params)
            self._model_generation_phase.set_inference_params(inference_params)

            hidden_states, next_tokens = self._generate_context_phase(input_ids, inference_params)

            # Generation phase.
            if self.config.qkv_format == "thd":
                lengths_tensor = torch.ones((next_tokens.shape[0],), dtype=int)
                inference_params.pre_step(
                    OrderedDict(zip(list(range(len(lengths_tensor))), lengths_tensor.tolist()))
                )
            else:
                inference_params.setup_before_new_input(length=1)

            output_tokens = [next_tokens]

            mask = None
            if self.config.qkv_format != "thd":
                mask = (input_ids == 0).unsqueeze(1).unsqueeze(1)

            for _ in range(max_new_tokens):
                if self.config.qkv_format != "thd":
                    # It will not work with cuda graphs, but it is not used for thd qkv_format.
                    # Attention mask in bshd needs attn_mask increased by 1 to
                    # include the next token to be generated
                    mask = self._make_mask_one_token_longer(mask)

                next_tokens = self._model_generation_phase(
                    hidden_states,
                    mask=mask,
                    attn_mask_type="padding" if self.config.qkv_format == "thd" else "arbitrary",
                    rope_emb=self.te_rope_emb,
                )

                # self.inference_params contains for example kv_cache.
                # This needs to be called before every pass,
                # to update the information of sequence lengths.
                # Here we increase sequence offsets by one,
                # because we generated one token for every sequence.
                lengths_tensor = torch.ones((next_tokens.shape[0],), dtype=int)
                inference_params.pre_step(
                    OrderedDict(zip(list(range(len(lengths_tensor))), lengths_tensor.tolist()))
                )

                # next_tokens is static output tensor, so we need to clone it
                # - it gets changed every iteration.
                output_tokens.append(next_tokens.clone())

            result = torch.cat((input_ids, torch.stack(output_tokens).permute([1, 0])), dim=1)
            return result

    def forward(self, *args, **kwargs):
        self._model_context_phase.set_inference_params(None)
        hidden_states = self.model.embed_tokens(kwargs["input_ids"])
        logits = self._model_context_phase(
            hidden_states,
            attention_mask=(
                (kwargs["input_ids"] == 0) if self.config.qkv_format != "thd" else None
            ),
            attn_mask_type="padding_causal",
        )
        return logits


class TEGemmaForCausalLMCudaGraphs(TEGemmaForCausalLM):
    """
    TEGemmaForCausalLMCudaGraphs is the version of the class TEGemmaForCausalLM
    using CUDA Graphs to speed it up. We need to make one trade-off.
    Namely, batch_size, max_seq_len and max_context_seq_len need to be static.
    It is necessary to run generation with the same value of
    these variables that we recorded graph on.
    """

    def __init__(self, config: GemmaConfig):
        super().__init__(config)
        assert (
            config.qkv_format == "thd"
        ), "Generation with CUDA Graphs are implemented only for thd format."

        # Preparation of the static buffers.
        self.config = config
        self.hidden_states_buffer = torch.empty(
            (
                self.config.cuda_graphs_static_batch_size,
                self.config.cuda_graphs_static_max_context_len,
                self.config.hidden_size,
            )
        ).cuda()

        # This is in fact part of the buffer for hidden_states.
        self.generation_buffer = self._get_generation_buffer(self.hidden_states_buffer)
        self.inference_params = InferenceParams(
            max_batch_size=self.config.cuda_graphs_static_batch_size,
            max_sequence_length=self.config.cuda_graphs_static_max_seq_len,
            num_heads_kv=self.config.num_key_value_heads,
            head_dim_v=self.config.head_dim,
            head_dim_k=self.config.head_dim,
            dtype=torch.bfloat16,
            is_paged=self.config.is_paged,
            page_size=64,
            total_num_pages=(
                64 * self.config.cuda_graphs_static_max_seq_len // 64
            ),  # 64 * 64 (max_sequence_length) / 64 (page_size)
        )

        self._model_generation_phase.set_inference_params(self.inference_params)
        self._model_context_phase.set_inference_params(self.inference_params)

    def record(self):
        # We want to record model in training=False, because it will be used in generation.
        self.eval()

        # Here "the trick" happens. We override methods from TEGemmaForCausalLM
        # with their recorded version. After invocation of each of them,
        # captured graph will be replayed with minimal usage of CPU,
        # what will lead to huge speedup.
        input_shape = (
            self.config.cuda_graphs_static_batch_size,
            self.config.cuda_graphs_static_max_context_len,
        )

        # Forcing the inputs to be the same as lengths_tensor from TEGemmaForCausalLM
        lengths = torch.tensor(input_shape[0] * [input_shape[1]], device="cuda", dtype=torch.int32)
        # @sudhakars: Hardcoded value. Remove this.
        lengths.data[:] = torch.tensor([9] * self.config.cuda_graphs_static_batch_size)
        self.inference_params.pre_step(
            OrderedDict(zip(list(range(len(lengths))), lengths.tolist()))
        )

        self._model_context_phase = self.record_graph(
            self._model_context_phase,
            self.hidden_states_buffer,
            attn_mask_type="padding_causal",
            rope_emb=self.te_rope_emb,
        )  # CUDA Graphs recording

        # Setup the recording for generation phase.
        input_shape = (self.config.cuda_graphs_static_batch_size, 1)
        lengths = torch.tensor(input_shape[0] * [1], device="cuda", dtype=torch.int32)
        self.inference_params.pre_step(
            OrderedDict(zip(list(range(len(lengths))), lengths.tolist()))
        )

        self._model_generation_phase = self.record_graph(
            self._model_generation_phase,
            self.generation_buffer,
            attn_mask_type="padding",
            rope_emb=self.te_rope_emb,
        )  # CUDA Graphs recording

    """
        Functions _create_hidden_states_buffer and _create_inference_params
        from base class are overriden to make hidden_states and inference_params static
        - not changing their position in memory between every invocation.
    """

    def _create_hidden_states_buffer(self, *args, **kwargs):
        return self.hidden_states_buffer

    def _create_inference_params(self, *args, **kwargs):
        self.inference_params.reset()
        return self.inference_params

    def _get_max_input_seq_len(self, _):
        return self.config.cuda_graphs_static_max_context_len

    @torch.no_grad()
    def record_graph(self, function, input_tensor, **sample_kwargs):
        # function is invoked on argument (self.hidden_states,) and all kernels are recorded.
        # record_graph() returns captured function, which can be run later with lower of th CPU.
        fp8_format = Format.HYBRID
        fp8_recipe = DelayedScaling(
            fp8_format=fp8_format, amax_history_len=1024, amax_compute_algo="max"
        )

        # We need both autocasts: FP8 for operations that can run in lower precision
        # and BF16 for those that cannot.
        with autocast(dtype=torch.bfloat16, cache_enabled=False):
            graphed_function = te.pytorch.make_graphed_callables(
                function,
                (input_tensor,),
                fp8_enabled=self.config.fp8,
                fp8_recipe=fp8_recipe,
                allow_unused_input=True,
                num_warmup_iters=5,
                sample_kwargs=sample_kwargs,
            )
        return graphed_function
