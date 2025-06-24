# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaConfig, LlamaModel

import torch.nn.functional as F


def setup_cache_params_from_infer_params(inference_params, lengths_tensor, max_input_length):
    """
    Converts the `input_ids` to variables like `cu_seqlens_q/kv`, etc. which
    will be used later.

    (Currently a hack, this should be reformatted to a better method)
    """

    assert (
        lengths_tensor is not None and max_input_length is not None
    ), 'lengths_tensor and max_input_length should not be none for qkv_format = "thd"'
    torch.add(
        inference_params.cached_sequence_lengths,
        inference_params.input_sequence_lengths,
        out=inference_params.cached_sequence_lengths,
    )
    inference_params.input_sequence_lengths.copy_(lengths_tensor)
    inference_params.max_incoming_seq_len = max_input_length

    max_seqlen_q, max_seqlen_kv = (
        inference_params.max_incoming_seq_len,
        inference_params.max_sequence_length,
    )

    # # Allocation of buffers, it works correctly with CUDA Graphs.
    _allocator = StaticBufferAllocator()
    NR_BUFFERS = 4

    cu_seqlens_q, cu_seqlens_kv, cu_seqlens_q_padded, cu_seqlens_kv_padded = [
        _allocator(inference_params.max_batch_size + 1, dtype=torch.int32, device="cuda")
        for _ in range(NR_BUFFERS)
    ]

    torch.cumsum(inference_params.input_sequence_lengths, dim=0, out=cu_seqlens_q[1:])
    torch.cumsum(
        inference_params.cached_sequence_lengths + inference_params.input_sequence_lengths,
        dim=0,
        out=cu_seqlens_kv[1:],
    )
    # If layer has shape [b * s_layer, h, d]
    # offsets are of the form [k * s_layer * h * d for k = 0, ..., batch_size]
    cu_seqlens_q_padded.copy_(
        torch.arange(0, inference_params.max_batch_size + 1, device="cuda") * max_seqlen_q
    )
    cu_seqlens_kv_padded.copy_(
        torch.arange(0, inference_params.max_batch_size + 1, device="cuda") * max_seqlen_kv
    )

    # inference_params.step_dict = OrderedDict(zip(list(range(len(lengths_tensor))), lengths_tensor.tolist()))
    inference_params.pre_step(
        OrderedDict(zip(list(range(len(lengths_tensor))), lengths_tensor.tolist()))
    )

    # print(inference_params.step_dict)

    def get_cache_params_in_infer_params():
        return (
            max_seqlen_q,
            max_seqlen_kv,
            cu_seqlens_q,
            cu_seqlens_kv,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
        )

    # For the time being, create an ad-hoc field in `inference_params` to get the variables.
    # @sudhakars: to create a better way later.
    inference_params.get_cache_params_from_infer_params = get_cache_params_in_infer_params


# This class has been modified from
# https://github.com/huggingface/transformers/blob/98adf24883b007c2a7fb17bab1c01b1614673433/src/transformers/models/gemma/modeling_gemma.py
class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
        )
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = (
            device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
        return emb.unsqueeze(2)  # should return in [b, s, 1, d] format


class StaticBufferAllocator(torch.nn.Module):
    """
    This class is used when we use te.make_graphed_callable().
    CUDA Graphs require all tensors to be static. Neverthlessly,
    torch API make_graphed_callable() takes care of output of torch modules,
    and makes them static. Thus by wrapping allocation of memory into
    torch.nn.Module, we can greatly simplify our code.
    """

    # pylint: disable=no-self-use
    def forward(self, size, dtype, device):
        """
        Return buffer of given size, dtype and device.
        """
        return torch.zeros(size, dtype=dtype, device=device)


class TELlamaDecoderLayer(te.pytorch.TransformerLayer):
    """
    Wrapper class over TE's `TransformerLayer`. This makes the wrapper very
    similar to HF's `LlamaDecoderLayer` and easier to replace it in the code.

    Args:
        config: LlamaConfig
        args: positional args (for compatibility with `LlamaDecoderLayer`)
        kwargs: keyword args (for compatibility with `LlamaDecoderLayer`)
    """

    def __init__(self, config: LlamaConfig, layer_idx: int, *args, **kwargs):

        self.llama_config = config
        self.head_dim = self.llama_config.hidden_size // self.llama_config.num_attention_heads

        super().__init__(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            bias=False,  # LLaMA specific
            layernorm_epsilon=config.rms_norm_eps,
            hidden_dropout=0,
            attention_dropout=0,
            fuse_qkv_params=config.fuse_qkv_params,
            normalization="RMSNorm",
            activation="swiglu",  # LLaMA specific
            # attn_input_format=config.qkv_format,
            attn_input_format="bshd",
            num_gqa_groups=config.num_key_value_heads,
            kv_channels=self.head_dim,  # LLaMA specific
            layer_number=(
                layer_idx + 1
            ),  # Layer numbers in TE starts from 1, not 0 like in the HF.
            zero_centered_gamma=True,  # LLaMA specific
        )

    def alloc(self, size, dtype, device):
        """
        Allocated the buffer and works correctly with CUDA Graphs.
        """
        return self._allocator(size, dtype, device)

    def forward(self, *args, **kwargs):  # We need to additionally pass positional encoding.

        if "self_attn_mask_type" in kwargs:
            attn_mask_type = kwargs["self_attn_mask_type"]
        else:
            attn_mask_type = "whatever_default_is"

        if attn_mask_type == "arbitrary":
            # @sudhakars: following logic doesn't work for `thd`
            attn_mask = kwargs["attention_mask"]
            attention_mask_inv = ~attn_mask
            generation_case = torch.tensor(torch.tensor(attn_mask.shape).shape).item() > 2

            if generation_case:
                # @sudhakars: for some reason, `attention_mask` for generation is of the
                # form [b, 1, 1, s].
                attention_mask_inv = attention_mask_inv.squeeze(1).squeeze(1)
                assert torch.tensor(torch.tensor(attention_mask_inv.shape).shape).item() == 2

            # Create `position_ids` on the fly using `attention_mask` since HF
            # does the same in generation logic.
            position_ids = attention_mask_inv.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask_inv == 0, 1)

            if "position_ids" in kwargs and kwargs["position_ids"] is not None:
                assert torch.all(
                    torch.eq(position_ids, kwargs["position_ids"])
                ), "position ids don't match match exactly!"

            # convert [b, s] to [b, 1, s, s] since `arbitrary` is only set for
            # context phase and context phase gets [b, s] sized attn mask
            seq_len = (
                1
                if torch.tensor(torch.tensor(attn_mask.shape).shape).item() > 2
                else attention_mask_inv.shape[1]
            )
            arbitrary_attn_mask = torch.zeros(
                attention_mask_inv.shape[0], 1, seq_len, attention_mask_inv.shape[1]
            ).bool()
            for sample_idx in range(attn_mask.shape[0]):
                pad_len = attn_mask[sample_idx].sum().int().item()
                # set the columns to padded
                arbitrary_attn_mask[sample_idx, :, :, :pad_len] = True
                # set the rows to padded
                if not generation_case:
                    arbitrary_attn_mask[sample_idx, :, :pad_len, :] = True
                    arbitrary_attn_mask[sample_idx] = torch.tril(
                        arbitrary_attn_mask[sample_idx].logical_not()
                    ).logical_not()

            # Update the attention mask to arbitrary
            kwargs["attention_mask"] = arbitrary_attn_mask.cuda()

            # @sudhakars: `max_position_embeddings` is not even used inside GemmaRotaryEmbedding
            # @sudhakars: change the hardcoded `dim` to something like config.head_dim
            te_rope_emb = LlamaRotaryEmbedding(
                dim=self.head_dim, max_position_embeddings=self.llama_config.max_position_embeddings
            ).cuda()
            te_rope_emb = te_rope_emb(args[0], position_ids.cuda())
        else:
            # When the `attention_mask` is not `arbitrary`, then for the purpose
            # of this tutorial, we're using `padding_causal` (for context) and
            # `padding` (for generation)
            # @sudhakars: find a better way to provide the `tensor_format`
            te_rope_emb = RotaryPositionEmbedding(self.head_dim)(  # Use self.head_dim
                max_seq_len=self.llama_config.max_position_embeddings
            ).cuda()

        inference_params = kwargs["inference_params"]
        # @sudhakars: big assumption that the input is "sbhd"
        # batch_size = args[0].shape[0]
        if inference_params.qkv_format_legacy == "thd":
            (
                max_seqlen_q,
                max_seqlen_kv,
                cu_seqlens_q,
                cu_seqlens_kv,
                cu_seqlens_q_padded,
                cu_seqlens_kv_padded,
            ) = inference_params.get_cache_params_from_infer_params()

        # this args cannot be passed to TransformerLayer
        keys_to_remove = [
            "position_ids",
            "past_key_value",
            "output_attentions",
            "use_cache",
            "cache_position",
        ]
        for key in keys_to_remove:
            kwargs.pop(key, None)

        # import pdb; pdb.set_trace()
        # We need to return tuple to be compatible with HF.
        return (
            super().forward(
                *args,
                rotary_pos_emb=te_rope_emb,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                **kwargs
            ),
        )


class StaticLlamaModel(torch.nn.Module):
    """
    StaticLlama is based of HF LlamaModel class.
    It is adjusted to work properly with CUDA Graphs.
    """

    def __init__(
        self,
        model: LlamaModel,
        dtype: torch.dtype,
        mask: torch.Tensor,
        lm_head: torch.nn.Module,
    ):
        super().__init__()
        self.model = model
        self.llama_config = model.config  # Store LlamaConfig
        self.normalizer = torch.tensor(self.llama_config.hidden_size**0.5, dtype=dtype)
        self.mask = mask
        self.lm_head = lm_head

    def set_inference_params(self, inference_params):
        self.inference_params = inference_params

    # @sudhakars: is `arbitrary` fine being the default here?
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        attn_mask_type: str = "arbitrary",
    ):
        # import pdb; pdb.set_trace()
        if hidden_states.shape[1] > 1:
            torch.save(hidden_states, "input_ctxt.pth")

        with torch.no_grad():
            # static operation - for CUDA graphs
            hidden_states.data[:] = hidden_states.data[:] * self.normalizer

            for i, decoder_layer in enumerate(self.model.layers):
                hidden_states.data[:] = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    self_attn_mask_type=self.mask if attn_mask_type is None else attn_mask_type,
                    inference_params=self.inference_params,
                )[
                    0
                ]  # static copy - for CUDA graphs

        hidden_states.copy_(self.model.norm(hidden_states))  # static copy - for CUDA graphs
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        return logits


class LlamaGenerator(torch.nn.Module):
    """
    LlamaGenerator gets one layer of embeddins,
    makes forward pass and returns next tokens.
    """

    def __init__(
        self, model: LlamaModel, lm_head: torch.nn.Module, dtype: torch.dtype, qkv_format: str
    ):
        super().__init__()
        self.model = model
        self.llama_layers = StaticLlamaModel(model, dtype, "arbitrary", lm_head)
        self.qkv_format = qkv_format

    def set_inference_params(self, inference_params):
        self.inference_params = inference_params
        self.llama_layers.set_inference_params(inference_params)

    # @sudhakars: is `arbitrary` a good default value here?
    def forward(
        self, hidden_states: torch.Tensor, mask: torch.Tensor = None, mask_type: str = "arbitrary"
    ):
        logits = self.llama_layers(hidden_states, attention_mask=mask, attn_mask_type=mask_type)

        assert logits.shape[0] == hidden_states.shape[0]  # b
        assert logits.shape[1] == hidden_states.shape[1]  # seq_len
        # logits.shape[2] = number of tokens
        logits = logits[:, -1, :]
        next_tokens = torch.argmax(logits, dim=1)

        # static copy for CUDA graphs
        hidden_states.copy_(self.model.embed_tokens(next_tokens).unsqueeze(1))

        # self.inference_params contains for example kv_cache.
        # This needs to be called before every pass,
        # to update the information of sequence lengths.
        # Here we increase sequence offsets by one,
        # because we generated one token for every sequence.
        if self.qkv_format == "thd":
            # self.inference_params.setup_before_new_input(
            #     lengths_tensor=torch.ones((next_tokens.shape[0],), device="cuda"),
            #     max_input_length=1,
            # )
            setup_cache_params_from_infer_params(
                self.inference_params,
                lengths_tensor=torch.ones((next_tokens.shape[0],), dtype=int),
                max_input_length=1,
            )
        else:
            self.inference_params.setup_before_new_input(length=1)

        return next_tokens


class PartialForwardWrapper(torch.nn.Module):
    """
    This class wraps a `torch.nn.Module` while partially modifying its `forward`

    CUDAGraphs' `make_graphed_callables` method takes in a module but if only
    `functools.partial` is used to wrap the module, it changes the modules'
    type and that interferes with the `make_graphed_callables` intrinsics.
    """

    def __init__(self, module, **kwargs):
        super().__init__()
        self.module = module
        self.partial_forward = partial(self.module.forward, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.partial_forward(*args, **kwargs)

    # @sudhakars: should we use better abstraction?
    def set_inference_params(self, *args, **kwargs):
        return self.module.set_inference_params(*args, **kwargs)


@contextmanager
def replace_decoder(te_decoder_cls):
    """
    Replace `LlamaDecoderLayer` with custom `TELlamaDecoderLayer`.
    """
    original_llama_decoder_cls = transformers.models.llama.modeling_llama.LlamaDecoderLayer
    transformers.models.llama.modeling_llama.LlamaDecoderLayer = te_decoder_cls
    try:
        yield
    finally:
        transformers.models.llama.modeling_llama.LlamaDecoderLayer = original_llama_decoder_cls


class TELlamaForCausalLM(LlamaForCausalLM):
    """
    Causal LM created with `LlamaModel`. The underlying `LlamaDecoderLayer`
    class is monkey-patched with `TELlamaDecoderLayer` class before
    initializing the causal LM with `LlamaForCausalLM`.

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        with replace_decoder(te_decoder_cls=TELlamaDecoderLayer):
            super().__init__(config)
        self.config = config
        self.to(torch.bfloat16).cuda()
        self.hidden_size = config.hidden_size
        self._model_generation_phase = LlamaGenerator(
            lm_head=self.lm_head,
            model=self.model,
            dtype=torch.bfloat16,
            qkv_format=config.qkv_format,
        )
        self._model_context_phase = StaticLlamaModel(
            self.model, torch.bfloat16, "arbitrary", self.lm_head
        )

        if self.config.fp8:
            self.fp8_recipe = DelayedScaling(
                fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max"
            )

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
            new_input_ids[i, : lengths[i]] = inputs[i, (max_seq_len - lengths[i]) : max_seq_len]
            new_input_ids[i, lengths[i] :] = inputs[i, 0 : (max_seq_len - lengths[i])]

        # Disable the input preparation that involves extra padding
        # inputs.copy_(new_input_ids)

        # Trim the inputs to no extra padding i.e. fix the max seq len to
        # the longest sequence in the batch
        actual_max_seq_len = inputs.ne(0).sum(dim=1).max()
        inputs.data = new_input_ids[:, :actual_max_seq_len]

        # For Paged Attention, make the valid sequences, multiple of 64
        # inputs.data = new_input_ids[:, :4].repeat(1, 16)

    def _next_64_multiply(self, x):
        return ((x + 63) // 64) * 64

    # This function is overriden in TeGEmmaForCausalLMCudaGraphs.
    def _create_hidden_states_buffer(self, input_ids: torch.Tensor):
        return torch.empty(
            (input_ids.shape[0], input_ids.shape[1], self.hidden_size),
            device="cuda",
            dtype=torch.float32,
        )

    # This function is overriden in TeGEmmaForCausalLMCudaGraphs.
    def _create_inference_params(self, *args, **kwargs):
        infer_params = InferenceParams(*args, **kwargs)

        max_batch_size = kwargs["max_batch_size"]

        # Initialize some legacy params
        infer_params.cached_sequence_lengths = torch.zeros(
            (max_batch_size,), device="cuda", dtype=torch.int32
        )
        infer_params.input_sequence_lengths = torch.zeros(
            (max_batch_size,), device="cuda", dtype=torch.int32
        )

        return infer_params

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
        hidden_states.data[:] = self.model.embed_tokens(input_ids)

        # We need to update offsets before every forward pass to make cache work properly.
        lengths = input_ids.ne(0).sum(dim=1)
        # import pdb; pdb.set_trace()
        if self.config.qkv_format == "thd":
            # inference_params.setup_before_new_input(
            #     lengths_tensor=lengths, max_input_length=input_ids.shape[1]
            # )
            lengths = input_ids.ne(0).sum(dim=1)
            max_input_length = input_ids.shape[1]
            setup_cache_params_from_infer_params(inference_params, lengths, max_input_length)
        else:
            inference_params.setup_before_new_input(length=input_ids.shape[1])

        logits = self._model_context_phase(
            hidden_states,
            attention_mask=((input_ids == 0) if self.config.qkv_format != "thd" else None),
            attn_mask_type="padding_causal" if self.config.qkv_format == "thd" else "arbitrary",
        )

        # We choose logits coresponding with last token in each sequence,
        # which have various lengths - they are stored in (inference_params.incoming_seq_len - 1)
        # Tensor when qkv_format == "thd" and
        # they are the last token in the sequence when qkv_format != "thd".
        if self.config.qkv_format == "thd":
            logits = logits[
                torch.arange(logits.size(0)), inference_params.input_sequence_lengths - 1, :
            ]
        else:
            logits = logits[:, -1, :]
        torch.save(logits, "logits_ctxt.pth")
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
            # input_ids = F.pad(
            #                 input_ids, (max_input_sequence_len - input_ids.shape[1], 0), "constant", 0
            #             )

            if self.config.qkv_format == "thd":
                # For thd layout padding is at the end, otherwise at the beginning.
                TELlamaForCausalLM._padding_to_end(input_ids, lengths)

            batch_size, max_input_sequence_len = input_ids.shape[0], self._get_max_input_seq_len(
                input_ids
            )
            # import pdb; pdb.set_trace()

            # InferenceParams is a cache, where keys and values of previous tokens are stored.
            # Moreover it stores length of both already generated and input sequences.
            head_dim = self.config.hidden_size // self.config.num_attention_heads
            inference_params = self._create_inference_params(
                max_batch_size=batch_size,
                # num_layers=self.config.num_hidden_layers,
                max_sequence_length=self._next_64_multiply(max_input_sequence_len + max_new_tokens),
                num_heads_kv=self.config.num_key_value_heads,
                # num_heads_q=self.config.num_attention_heads,
                head_dim_v=head_dim,
                head_dim_k=head_dim,
                dtype=torch.bfloat16,
                is_paged=True,
                page_size=64,
                total_num_pages=64 * 3,  # 64 * 64 (max_sequence_length) / 64 (page_size)
                # is_cuda_graph=False
            )

            def init_cache_params_in_infer_params(inference_params):
                inference_params.cached_sequence_lengths = torch.zeros(
                    (batch_size,), device="cuda", dtype=torch.int32
                )
                inference_params.input_sequence_lengths = torch.zeros(
                    (batch_size,), device="cuda", dtype=torch.int32
                )

            init_cache_params_in_infer_params(inference_params)
            inference_params.qkv_format_legacy = self.config.qkv_format

            self._model_context_phase.set_inference_params(inference_params)
            self._model_generation_phase.set_inference_params(inference_params)

            hidden_states, next_tokens = self._generate_context_phase(input_ids, inference_params)

            # Generation phase.
            if self.config.qkv_format == "thd":
                # inference_params.setup_before_new_input(
                #     lengths_tensor=torch.ones((next_tokens.shape[0],), device="cuda"),
                #     max_input_length=1,
                # )
                setup_cache_params_from_infer_params(
                    inference_params,
                    lengths_tensor=torch.ones((next_tokens.shape[0],), dtype=int),
                    max_input_length=1,
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

                # setup_cache_params_from_infer_params(inference_params, input_ids)
                # @sudhakars: could create position_ids from mask here
                next_tokens = self._model_generation_phase(
                    hidden_states,
                    mask,
                    mask_type="padding" if self.config.qkv_format == "thd" else "arbitrary",
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
            attn_mask_type="arbitrary",
        )
        return logits


class TELlamaForCausalLMCudaGraphs(TELlamaForCausalLM):
    """
    TELlamaForCausalLMCudaGraphs is the version of the class TELlamaForCausalLM
    using CUDA Graphs to speed it up. We need to make one trade-off.
    Namely, batch_size, max_seq_len and max_context_seq_len need to be static.
    It is necessary to run generation with the same value of
    these variables that we recorded graph on.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        assert (
            config.qkv_format == "thd"
        ), "Generation with CUDA Graphs are implemented only for thd format."

        # Preparation of the static buffers.
        self.config = config
        self.hidden_states_buffer = torch.empty(
            (
                config.cuda_graphs_static_batch_size,
                config.cuda_graphs_static_max_context_len,
                config.hidden_size,
            )
        ).cuda()
        # This is in fact part of the buffer for hidden_states.
        self.generation_buffer = self._get_generation_buffer(self.hidden_states_buffer)
        self.inference_params = InferenceParams(
            max_batch_size=config.cuda_graphs_static_batch_size,
            max_sequence_length=config.cuda_graphs_static_max_seq_len,
            qkv_format="thd",
        )

        self._model_generation_phase.set_inference_params(self.inference_params)
        self._model_context_phase.set_inference_params(self.inference_params)

    def record(self):
        # We want to record model in training=False, because it will be used in generation.
        self.eval()

        # Here "the trick" happens. We override methods from TELlamaForCausalLM
        # with their recorded version. After invocation of each of them,
        # captured graph will be replayed with minimal usage of CPU,
        # what will lead to huge speedup.
        input_shape = (
            self.config.cuda_graphs_static_batch_size,
            self.config.cuda_graphs_static_max_context_len,
        )
        self.inference_params.reset()
        self.inference_params.setup_before_new_input(
            lengths_tensor=torch.tensor(input_shape[0] * [input_shape[1]], device="cuda"),
            max_input_length=input_shape[1],
        )
        self._model_context_phase = self.record_graph(
            PartialForwardWrapper(
                self._model_context_phase,
                attn_mask_type=(
                    "padding_causal" if self.inference_params.qkv_format == "thd" else "arbitrary"
                ),
            ),
            self.hidden_states_buffer,
        )  # CUDA Graphs recording

        input_shape = (self.config.cuda_graphs_static_batch_size, 1)
        self.inference_params.reset()
        self.inference_params.setup_before_new_input(
            lengths_tensor=torch.tensor(input_shape[0] * [input_shape[1]], device="cuda"),
            max_input_length=input_shape[1],
        )
        self._model_generation_phase = self.record_graph(
            PartialForwardWrapper(
                self._model_generation_phase,
                mask_type="padding" if self.inference_params.qkv_format == "thd" else "arbitrary",
            ),
            self.generation_buffer,
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
    def record_graph(self, function, input_tensor):
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
                num_warmup_iters=3,
            )
        return graphed_function
