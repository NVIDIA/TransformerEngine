# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from contextlib import contextmanager

from typing import Optional
from functools import partial
from collections import OrderedDict

import torch
from torch.amp import autocast

import transformer_engine as te
from transformer_engine.pytorch.attention import InferenceParams, RotaryPositionEmbedding
from transformer_engine.common.recipe import Format, DelayedScaling
from transformer_engine.pytorch.quantization import get_default_fp8_recipe
import transformers
from transformers.models.gemma.modeling_gemma import GemmaForCausalLM, GemmaConfig, GemmaModel

import torch.nn.functional as F

"""
Top level description of the classes used in the tutorial from this file.
----------------------------------------------------------------------

HuggingFace Gemma Model implementation hierarchy:
----------------------------------
GemmaDecoderLayer:
├── self_attn:
│   ├── norm: (nn.LayerNorm)
│   ├── qkv_proj: (nn.Linear)
│   ├── attention: (SDPA, FlashAttention, etc.)
│   └── o_proj: (nn.Linear)
├── ffn:
│   ├── norm: (nn.LayerNorm)
│   ├── gate_proj: (nn.Linear)
│   ├── up_proj: (nn.Linear)
│   └── down_proj: (nn.Linear)

GemmaModel:
├── embed_tokens         : Token embedding layer
├── layers               : GemmaDecoderLayer × N
├── norm                 : GemmaRMSNorm
└── rotary_emb           : GemmaRotaryEmbedding

GemmaForCausalLM:
├── model                : instance of GemmaModel
├── lm_head              : (nn.Linear) hidden states to vocabulary logits for generation
└── generate             : generate method (input prompt -> GemmaForCausalLM -> next tokens)

How `generate()` works in HF's GemmaForCausalLM:
    1. prefill (input prompt -> model -> lm_head -> logits -> next token)
    2. loop until max_new_tokens:
        - next token -> model -> lm_head -> logits -> next token
    3. return all tokens

NOTE: Notice how "prefill" and "loop until next tokens" are just part of the `generate()` method.
      This is a common pattern in HF models.


TransformerEngine's Gemma Model Hierarchy:
----------------------------------------
HF's `GemmaDecoderLayer` is monkey-patched with `TEGemmaDecoderLayer` before `GemmaForCausalLM` is initialized. This way,
while the model is downloaded from HuggingFace and most of the code runs from HF's `GemmaForCausalLM`, the underlying
blocks of "transformer layer" are actually from TransformerEngine.

TEGemmaDecoderLayer (inherits from te.TransformerLayer):
├── te.MultiHeadAttention:
│   ├── linear_qkv: (te.LayerNormLinear)
│   ├── attention: (te.DotProductAttention)
│   └── out_proj: (te.LayerNormLinear)
├── te.LayerNormMLP:
│   ├── fc1: (te.LayerNormLinear)
│   ├── fc2: (te.Linear)
│   └── activation: (te.GeGLU)

To be able to use `model.generate()`, an entry point is needed. `TEGemmaForCausalLM` is the entry point which
subclasses HF's `GemmaForCausalLM` and adds a few attributes and methods.

TEGemmaForCausalLM (inherits from HF's GemmaForCausalLM)
├─ model                    : inherited from HF's GemmaForCausalLM but with monkey-patched TEGemmaDecoderLayer × N
├─ lm_head                  : directly inherited from HF's GemmaForCausalLM
├─ te_rope_emb              : RotaryPositionEmbedding (reusing the same for all layers for CUDA graphs compatibility)
├─ hidden_states_buffer     : shape [b, max_ctx, h]                             (static)
├─ generation_buffer        : shape [b, 1, h] (view of `hidden_states_buffer`)  (static)
├─ inference_params         : TransformerEngine KV cache
├─ model_context_phase      : GemmaModelWrapper  → uses (model, lm_head, inference_params) for full-sequence prefill
├─ model_generation_phase   : GemmaGenerationWrapper → uses (model, lm_head, inference_params) for single-token decode
└─ generate                 : generate method (input prompt -> TEGemmaForCausalLM -> next tokens)

Notice how "prefill" and "loop until next tokens" are specialized to wrapper subroutines - "model_context_phase" and
"model_generation_phase" respectively which makes it easier to use CUDA Graphs. Just one more abstraction is needed:

TEGemmaForCausalLMCudaGraphs (inherits from TEGemmaForCausalLM)
├─ model                    : unchanged (HF's GemmaModel with monkey-patched TEGemmaDecoderLayer × N)
├─ lm_head                  : unchanged
├─ hidden_states_buffer     : unchanged
├─ generation_buffer        : unchanged
├─ inference_params         : unchanged
├─ record                   : utility function to record the graphed callable
├─ model_context_phase      : GraphedCallable(for Context/prefill) replaced by `record`
├─ model_generation_phase   : GraphedCallable(for Generation) replaced by `record`
└─ generate                 : unchanged

How `generate()` works in TEGemmaForCausalLM/TEGemmaForCausalLMCudaGraphs:
    1. model_context_phase (input prompt -> model -> lm_head -> logits -> next token)
    2. model_generation_phase:
        - loop until max_new_tokens:
            - next token -> model -> lm_head -> logits -> next token
    3. return all tokens

NOTE: In the tutorial, `record` is called when initializing the model.

Additional notes and clarifications
-----------------------------------
- Wrappers, not submodules:
  `model_context_phase` and `model_generation_phase` are convenience wrappers over the same
  `model` (GemmaModel) and `lm_head`. They own no parameters; they standardize buffer usage,
  masks (context uses "padding_causal", generation uses "padding"), rotary embeddings, and
  KV-cache (`InferenceParams`) flow for TE-optimized inference.

- Buffer relationship:
  `hidden_states_buffer` has shape [b, max_ctx, h]. `generation_buffer` is a contiguous view
  of size [b, 1, h] carved from its start to avoid non-contiguous indexing. Generation updates
  `generation_buffer` in-place with next-token embeddings.

- Padding policy:
  Inputs may arrive left-padded (HF-style). Before TE execution, padding is shifted to the end
  to match TE attention mask expectations and to keep shapes contiguous for capture/replay.

- CUDA Graphs specifics:
  `record()` captures two separate callables (context/prefill and generation) with fixed shapes and
  stable pointers, then replaces the wrappers with these GraphedCallables. Under graphs, the
  functional behavior is identical; only allocation/pointer churn and CPU overhead are removed.
"""


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


class GemmaModelWrapper(torch.nn.Module):
    """
    Encapsulates the HuggingFace GemmaModel class as a wrapper whose
    forward pass is compatible with CUDA Graphs.
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

        # This is not needed for generation but is needed for training
        # or finetuning.
        if self.training:
            logits = logits.float()

        return logits


class GemmaGenerationWrapper(torch.nn.Module):
    """
    Gets token embeddings for a batch of single tokens, runs forward pass, and
    returns the batch ofnext tokens. Also compatible with CUDA graphs. Not a
    subclass of `GemmaModel` since the model layers are simply reused here.
    """

    def __init__(
        self,
        model: GemmaModel,
        lm_head: torch.nn.Module,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.model = model
        self.gemma_layers = GemmaModelWrapper(model, dtype, lm_head)

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
    Monkey-patches `GemmaDecoderLayer` with the custom `TEGemmaDecoderLayer`
    class.
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
        config: Gemma model config that HF uses to initialize the model.
    """

    def __init__(self, config: GemmaConfig):

        dtype = torch.bfloat16
        with replace_decoder(te_decoder_cls=TEGemmaDecoderLayer):
            super().__init__(config)

        self.config = config
        self.to(dtype).cuda()
        self.hidden_size = config.hidden_size

        self._model_context_phase = GemmaModelWrapper(self.model, dtype, self.lm_head)

        self._model_generation_phase = GemmaGenerationWrapper(
            lm_head=self.lm_head,
            model=self.model,
            dtype=dtype,
        )

        if self.config.fp8:
            self.fp8_recipe = get_default_fp8_recipe()

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

    def _create_or_fetch_hidden_states_buffer(self, input_ids: torch.Tensor):
        """
        Returns a tensor of shape [b, s, hd] where `b` is the batch size,
        `s` is the sequence length, and `hd` is the hidden size.

        This function is overriden in TEGemmaForCausalLMCudaGraphs.
        """

        tensor = torch.empty(
            (input_ids.shape[0], input_ids.shape[1], self.hidden_size),
            device="cuda",
            dtype=torch.float32,
        )
        return tensor

    def _create_or_fetch_inference_params(self, *args, **kwargs):
        """
        Creates an InferenceParams object.

        This function is overriden in TEGemmaForCausalLMCudaGraphs.
        """

        infer_params = InferenceParams(*args, **kwargs)
        return infer_params

    def _get_generation_buffer(self, hidden_states_buffer, data_to_copy=None):
        """
        Returns a tensor of shape [b, 1, hd] where `b` is the batch size,
        `hd` is the hidden size.

        The buffer for generation is some part (beginning) of hidden states buffer.
        This function returns pointer to it and also copies there data if provided.
        """
        # hidden_states_buffer has shape [b, s, hd]
        # generation_buffer will have shape [b, 1, hd]
        # Notice that `hidden_states_buffer[:, 0, :].unsqueeze(1)` will return
        # uncontiguous buffer, which we want to avoid.
        output = hidden_states_buffer.view(-1)[
            : hidden_states_buffer.shape[0] * hidden_states_buffer.shape[2]
        ]
        if data_to_copy is not None:
            output.copy_(data_to_copy.reshape(-1))
        generation_buffer = output.view(
            (hidden_states_buffer.shape[0], 1, hidden_states_buffer.shape[2])
        )
        return generation_buffer

    def setup_and_run_context_phase(
        self, input_ids: torch.Tensor, inference_params: InferenceParams
    ):
        """
        Runs the context or prefill phase of the model.

        This function is overriden in TEGemmaForCausalLMCudaGraphs.
        """

        hidden_states = self._create_or_fetch_hidden_states_buffer(input_ids)
        hidden_states.copy_(self.model.embed_tokens(input_ids))

        # Update offsets before every forward pass (including context/prefill
        # phase) to make cache work properly.
        lengths = input_ids.ne(0).sum(dim=1)
        inference_params.pre_step(OrderedDict(zip(list(range(len(lengths))), lengths.tolist())))

        logits = self._model_context_phase(
            hidden_states,
            attention_mask=None,
            attn_mask_type="padding_causal",
            rope_emb=self.te_rope_emb,
        )

        logits = logits[torch.arange(logits.size(0)), lengths - 1, :]
        next_tokens = torch.argmax(logits, dim=1)

        # `self.hidden_states` has shape [b, s, hd].
        # Return hidden state for the last token - output has shape [b, 1, hd].
        hidden_states = self._get_generation_buffer(
            hidden_states, self.model.embed_tokens(next_tokens)
        )
        return hidden_states, next_tokens

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pad_token_id: int = 0,
        max_new_tokens: int = 0,
        *args,
        **kwargs,
    ):
        """
        Generates next tokens auto-regressively for a batch of input tokens.
        """
        self.eval()

        # Both autocasts are needed: FP8 for operations that can run in lower
        # precision and BF16 for those that cannot.
        with autocast("cuda", dtype=torch.bfloat16, cache_enabled=False), te.pytorch.autocast(
            enabled=self.config.fp8, recipe=self.fp8_recipe if self.config.fp8 else None
        ):
            lengths = torch.sum(input_ids.ne(pad_token_id), dim=-1).squeeze()
            # If padding is at the beginning, then shift it to the end
            TEGemmaForCausalLM._padding_to_end(
                input_ids,
                lengths,
                max_seq_len=(
                    self.config.cuda_graphs_static_max_context_len
                    if self.config.generation_cuda_graphs
                    else None
                ),
            )

            batch_size = input_ids.shape[0]
            # For benchmark generation run, this is being set explicitly.
            max_input_sequence_len = self.config.max_seq_length

            # InferenceParams is a cache, where keys and values of previous
            # tokens are stored. Moreover it stores the current running lengths
            # of the sequences in the current batch.
            # A helper function is used to create the inference params object
            # because this `generate` method is common for TEGemmaForCausalLM
            # and TEGemmaForCausalLMCudaGraphs. In case of CudaGraphs, this
            # function is overriden to simply return the inference params object
            # that is already created in TEGemmaForCausalLMCudaGraphs'
            # constructor.
            inference_params = self._create_or_fetch_inference_params(
                max_batch_size=batch_size,
                max_sequence_length=max_input_sequence_len,
                num_heads_kv=self.config.num_key_value_heads,
                head_dim_v=self.config.head_dim,
                head_dim_k=self.config.head_dim,
                dtype=torch.bfloat16,
                is_paged=self.config.is_paged,
                page_size=16,
                total_num_pages=batch_size * max_input_sequence_len // 16,
            )

            # Set the inference params for both the context/prefill phase and
            # generation phase objects.
            self._model_context_phase.set_inference_params(inference_params)
            self._model_generation_phase.set_inference_params(inference_params)

            # Context/prefill phase.
            hidden_states, next_tokens = self.setup_and_run_context_phase(
                input_ids, inference_params
            )

            # Generation phase.
            lengths_tensor = torch.ones((next_tokens.shape[0],), dtype=int)
            inference_params.pre_step(
                OrderedDict(zip(list(range(len(lengths_tensor))), lengths_tensor.tolist()))
            )
            output_tokens = [next_tokens]

            for _ in range(max_new_tokens):
                next_tokens = self._model_generation_phase(
                    hidden_states,
                    mask=None,
                    attn_mask_type="padding",
                    rope_emb=self.te_rope_emb,
                )

                # Increase sequence offsets by one because we generated one token
                # for every sequence.
                lengths_tensor = torch.ones((next_tokens.shape[0],), dtype=int)
                inference_params.pre_step(
                    OrderedDict(zip(list(range(len(lengths_tensor))), lengths_tensor.tolist()))
                )

                # `next_tokens` is a static output tensor, so we need to clone
                # it because it gets changed every iteration.
                output_tokens.append(next_tokens.clone())

            result = torch.cat((input_ids, torch.stack(output_tokens).permute([1, 0])), dim=1)
            return result

    def forward(self, *args, **kwargs):
        """
        Forward pass for the model. This is used in calibration step when
        forward pass is needed to generate FP8 calibration data.
        """

        self._model_context_phase.set_inference_params(None)
        hidden_states = self.model.embed_tokens(kwargs["input_ids"])
        logits = self._model_context_phase(
            hidden_states,
            attention_mask=(
                kwargs["input_ids"] == 0
            ),  # Hardcoded, this only applies to bshd/sbhd layouts.
            attn_mask_type="padding_causal",
        )
        return logits


class TEGemmaForCausalLMCudaGraphs(TEGemmaForCausalLM):
    """
    TEGemmaForCausalLMCudaGraphs is a wrapper over the class TEGemmaForCausalLM
    and uses CUDA Graphs to speed up the generation process. We need to make one
    trade-off - batch_size, max_seq_len and max_context_seq_len need to
    be static. It is necessary to run generation without changing the pointer
    to the variables that are recorded in the graph.
    """

    def __init__(self, config: GemmaConfig):
        super().__init__(config)

        self.config = config

        # Preparation of the static buffer to hold the hidden states that are
        # passed from one layer to the next.
        self.hidden_states_buffer = torch.empty(
            (
                self.config.cuda_graphs_static_batch_size,
                self.config.cuda_graphs_static_max_context_len,
                self.config.hidden_size,
            )
        ).cuda()

        # This is in fact part of the buffer for hidden_states. Refer to the
        # `_get_generation_buffer` function for more details.
        self.generation_buffer = self._get_generation_buffer(
            self.hidden_states_buffer,
        )

        # InferenceParams contains the keys and values cache. Refer to the
        # original call in TEGemmaForCausalLM's `generate` method for more
        # details.
        self.inference_params = InferenceParams(
            max_batch_size=self.config.cuda_graphs_static_batch_size,
            max_sequence_length=self.config.cuda_graphs_static_max_context_len,
            num_heads_kv=self.config.num_key_value_heads,
            head_dim_v=self.config.head_dim,
            head_dim_k=self.config.head_dim,
            dtype=torch.bfloat16,
            is_paged=self.config.is_paged,
            page_size=16,
            total_num_pages=self.config.cuda_graphs_static_batch_size
            * self.config.cuda_graphs_static_max_context_len
            // 16,
        )

        self._model_generation_phase.set_inference_params(self.inference_params)
        self._model_context_phase.set_inference_params(self.inference_params)

    def record(self):
        """
        Here "the trick" happens. `_model_context_phase` and
        `_model_generation_phase` from TEGemmaForCausalLM are replaced with
        their recorded version. Once the graphs are recorded, they can be
        replayed with minimal usage of CPU and that leads to speedup.
        """
        # Record the model with training=False, because it will be used in
        # generation.
        self.eval()

        # Setup the recording for context/prefill phase.
        input_shape = (
            self.config.cuda_graphs_static_batch_size,
            self.config.cuda_graphs_static_max_context_len,
        )

        # Hardcoded value for the context length.
        lengths = torch.tensor([9] * self.config.cuda_graphs_static_batch_size).to(
            device="cuda", dtype=torch.int32
        )
        self.inference_params.pre_step(
            OrderedDict(zip(list(range(len(lengths))), lengths.tolist()))
        )

        # Record the graph for context/prefill phase.
        self._model_context_phase = self.record_graph(
            self._model_context_phase,
            self.hidden_states_buffer,
            attn_mask_type="padding_causal",
            rope_emb=self.te_rope_emb,
        )

        # Setup the recording for generation phase.
        input_shape = (self.config.cuda_graphs_static_batch_size, 1)
        lengths = torch.tensor(input_shape[0] * [1], device="cuda", dtype=torch.int32)
        self.inference_params.pre_step(
            OrderedDict(zip(list(range(len(lengths))), lengths.tolist()))
        )

        # Record the graph for generation phase.
        self._model_generation_phase = self.record_graph(
            self._model_generation_phase,
            self.generation_buffer,
            attn_mask_type="padding",
            rope_emb=self.te_rope_emb,
        )

    def _create_or_fetch_hidden_states_buffer(self, *args, **kwargs):
        """
        Overriden to make `hidden_states` static i.e. not change its pointer
        in memory between every invocation.

        Returns the static buffer for `hidden states` which is already created
        in the constructor. This is the same buffer as used in the
        context/prefill phase.
        """
        return self.hidden_states_buffer

    def _create_or_fetch_inference_params(self, *args, **kwargs):
        """
        Overriden to make `inference_params` static i.e. not change its pointer
        in memory between every invocation.

        Returns the static buffer for `inference_params` which is already created
        in the constructor.
        """
        self.inference_params.reset()
        return self.inference_params

    @torch.no_grad()
    def record_graph(self, function, input_tensor, **sample_kwargs):
        """
        Records the graph for the given function. The function is invoked on
        argument (self.hidden_states,) and all kernels are recorded.
        It then returns the captured callable, which can be run later while
        minimizing CPU usage.
        """
        fp8_recipe = get_default_fp8_recipe()

        # We need both autocasts: FP8 for operations that can run in lower
        # precision and BF16 for those that cannot.
        with autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            graphed_function = te.pytorch.make_graphed_callables(
                function,
                (input_tensor,),
                enabled=self.config.fp8,
                recipe=fp8_recipe,
                allow_unused_input=True,
                num_warmup_iters=5,
                sample_kwargs=sample_kwargs,
            )
        return graphed_function
