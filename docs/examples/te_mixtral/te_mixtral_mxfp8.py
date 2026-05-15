# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""TE-native MXFP8 Mixtral model (tier 3).

MoE FFN is a TE ``Sequential`` of three fusible ops — ``GroupedLinear``
(gate_up), ``ScaledSwiGLU(glu_interleave_size=32)``, ``GroupedLinear``
(down) — that the OperationFuser collapses into the fused
``ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8`` and backward kernels under
MXFP8. HF gate (``w1``) and up (``w3``) weights are row-interleaved in
blocks of 32 to match the GLU interleaved layout that fused kernel reads.

The fused kernel is enabled by ``utils._enable_fused_mxfp8_grouped_mlp()``
(sets ``NVTE_CUTEDSL_FUSED_GROUPED_MLP=1`` and patches the SM-version /
cudnn-frontend signature checks). Requires
``nvidia-cudnn-frontend >= 1.23.0`` and SM>=10 (Blackwell B100/B200/B300+).
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from collections import OrderedDict
from contextlib import nullcontext
from typing import Any, ClassVar, ContextManager

import torch
import torch.distributed as dist
import torch.nn as nn

import transformer_engine.common.recipe as te_recipe
import transformer_engine.pytorch as te
from transformer_engine.pytorch.attention.inference import InferenceParams
from transformer_engine.pytorch.attention.rope import RotaryPositionEmbedding
from transformer_engine.pytorch.ops import (
    GroupedLinear as TEOpsGroupedLinear,
    ScaledSwiGLU,
    Sequential as TEOpsSequential,
)
from transformer_engine.pytorch.quantization import FP8GlobalStateManager
from transformer_engine.pytorch.router import fused_moe_aux_loss
from transformers import MixtralConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from te_moe_dispatch import AllToAllTokenDispatcher
from te_mixtral import (
    HFInferenceParams,
    _pad_input,
    _unpad_input,
)

logger = logging.getLogger(__name__)


# HF->TE checkpoint mapping is shared with the BF16 path in te_mixtral.py.
# ``GLU_INTERLEAVE_SIZE`` is the gate/up interleave block (32); the fused
# MXFP8 forward op only fires when ``ScaledSwiGLU`` is configured with it.
from hf_to_te_weights import (
    GLU_INTERLEAVE_SIZE,
    replace_params_mxfp8 as replace_params,
)


class NVMixtralMXFP8Config(MixtralConfig):
    """Improvement-9 config. Same surface as :class:`te_mixtral.NVMixtralConfig` but
    with the FFN mode locked to ``grouped_op`` + MXFP8."""

    attn_input_format: str = "thd"
    self_attn_mask_type: str = "padding_causal"
    expert_parallel_size: int = 1
    moe_aux_loss_coeff: float = 0.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.num_local_experts % self.expert_parallel_size != 0:
            raise ValueError(
                f"num_local_experts ({self.num_local_experts}) must be divisible by "
                f"expert_parallel_size ({self.expert_parallel_size})"
            )


class NVMixtralMXFP8PreTrainedModel(PreTrainedModel):
    """HF integration boilerplate for the tier-3 model."""

    config_class = NVMixtralMXFP8Config
    base_model_prefix = "model"
    _no_split_modules = ("NVMixtralMXFP8DecoderLayer",)
    _skip_keys_device_placement = ("past_key_values",)
    _do_not_quantize = ("lm_head", "model.layers.*.mlp.gate")

    def _init_weights(self, module):
        if module.__module__.startswith("transformer_engine.pytorch"):
            return
        super()._init_weights(module)

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        return {k: v for k, v in sd.items() if not k.endswith("_extra_state")}


class NVMixtralMXFP8SparseMoeBlock(nn.Module):
    """MoE block: router + EP dispatcher + fused MXFP8 grouped MLP."""

    def __init__(
        self,
        config: NVMixtralMXFP8Config,
        dispatcher: AllToAllTokenDispatcher | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.jitter_noise = config.router_jitter_noise

        self.ep_size = getattr(config, "expert_parallel_size", 1)
        self.num_local_experts = self.num_experts // self.ep_size
        self.moe_aux_loss_coeff = getattr(config, "moe_aux_loss_coeff", 0.0)
        self._aux_loss: torch.Tensor = torch.tensor(0.0)

        if self.intermediate_size % GLU_INTERLEAVE_SIZE != 0:
            raise ValueError(
                f"intermediate_size ({self.intermediate_size}) must be divisible by "
                f"GLU_INTERLEAVE_SIZE ({GLU_INTERLEAVE_SIZE})"
            )

        self.dispatcher = dispatcher or AllToAllTokenDispatcher(
            num_experts=self.num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=self.hidden_size,
            ep_size=self.ep_size,
        )

        device = "meta" if torch.get_default_device() == torch.device("meta") else "cuda"

        def _init_method(x: torch.Tensor) -> None:
            torch.nn.init.normal_(x, mean=0.0, std=config.initializer_range)

        with te.quantized_model_init(enabled=False):
            self.gate = te.Linear(
                self.hidden_size,
                self.num_experts,
                bias=False,
                device=device,
                params_dtype=config.dtype,
                init_method=_init_method,
            )

        self.experts_gate_up = TEOpsGroupedLinear(
            num_groups=self.num_local_experts,
            in_features=self.hidden_size,
            out_features=2 * self.intermediate_size,
            bias=False,
            dtype=config.dtype,
            device=device,
        )
        self.experts_swiglu = ScaledSwiGLU(glu_interleave_size=GLU_INTERLEAVE_SIZE)
        self.experts_down = TEOpsGroupedLinear(
            num_groups=self.num_local_experts,
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            bias=False,
            dtype=config.dtype,
            device=device,
        )
        # Wrap as TE Sequential to enable forward/backward op fusion
        # (ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8 / dswiglu).
        object.__setattr__(
            self,
            "_experts_ffn_op",
            TEOpsSequential(self.experts_gate_up, self.experts_swiglu, self.experts_down),
        )

    def set_ep_group(self, ep_group: dist.ProcessGroup, ep_mesh: Any) -> None:
        """Set the EP communication group on the dispatcher.

        Each EP rank owns its local slice of expert weights as ordinary
        Parameters (``weight0..weight{N-1}``) — no DTensor wrapping is
        needed because per-expert parameters are never replicated across the
        EP group.
        """
        del ep_mesh  # kept for API parity with te_mixtral.set_ep_groups
        self.dispatcher.set_ep_group(ep_group)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape

        if self.training and self.jitter_noise > 0:
            hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )

        if hidden_states.dim() == 3:
            hidden_states = hidden_states.reshape(-1, self.hidden_size)

        with te.autocast(enabled=False):
            router_logits = self.gate(hidden_states)  # [N, E]

        # Top-k routing weights, two algebraically equivalent forms.
        # Old::
        #
        #     probs   = softmax(logits)                  # (N, E)
        #     weights, idx = topk(probs, k)
        #     weights = weights / weights.sum(-1, keepdim=True)
        #
        # New (used here)::
        #
        #     topk_logits, idx = topk(logits, k)
        #     weights = softmax(topk_logits)             # softmax over (N, k)
        topk_logits, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = torch.nn.functional.softmax(topk_logits, dim=-1, dtype=torch.float32)

        # Bincount once, in the MoE block. ``AllToAllTokenDispatcher``
        # takes this as a required argument, so the dispatcher never
        # bincounts again.
        tokens_per_expert = torch.bincount(
            selected_experts.reshape(-1), minlength=self.num_experts
        ).to(torch.int32)

        if self.moe_aux_loss_coeff > 0:
            num_tokens = hidden_states.shape[0]
            softmax_probs = torch.nn.functional.softmax(
                router_logits, dim=-1, dtype=torch.float32
            )
            self._aux_loss = fused_moe_aux_loss(
                probs=softmax_probs,
                tokens_per_expert=tokens_per_expert,
                total_num_tokens=num_tokens,
                num_experts=self.num_experts,
                topk=self.top_k,
                coeff=self.moe_aux_loss_coeff,
            )
        else:
            self._aux_loss = torch.tensor(0.0, device=hidden_states.device)

        dispatch_out = self.dispatcher.dispatch(
            hidden_states,
            selected_experts,
            routing_weights,
            tokens_per_expert,
        )
        expert_input = dispatch_out.expert_input
        expert_probs = dispatch_out.expert_probs
        split_sizes = torch.tensor(
            dispatch_out.tokens_per_expert, dtype=torch.int32, device=expert_input.device
        )

        # Fused gate_up -> ScaledSwiGLU(probs) -> down.
        expert_output = self._experts_ffn_op(
            expert_input, split_sizes, expert_probs, split_sizes
        )

        output = self.dispatcher.combine(expert_output, dispatch_out.handle)
        return output.reshape(original_shape)


class NVMixtralMXFP8DecoderLayer(nn.Module):
    """Self-attention + tier-3 MoE block."""

    def __init__(
        self,
        config: NVMixtralMXFP8Config,
        layer_idx: int,
        dispatcher: AllToAllTokenDispatcher | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        device = "meta" if torch.get_default_device() == torch.device("meta") else "cuda"

        def _init_method(x: torch.Tensor) -> None:
            torch.nn.init.normal_(x, mean=0.0, std=config.initializer_range)

        self.self_attention = te.MultiheadAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_gqa_groups=config.num_key_value_heads,
            bias=False,
            layernorm_epsilon=config.rms_norm_eps,
            attention_dropout=0,
            fuse_qkv_params=True,
            qkv_weight_interleaved=True,
            normalization="RMSNorm",
            input_layernorm=True,
            qkv_format=config.attn_input_format,
            attn_mask_type=config.self_attn_mask_type,
            layer_number=layer_idx + 1,
            params_dtype=config.dtype,
            device=device,
            init_method=_init_method,
            output_layer_init_method=_init_method,
        )
        self.post_attention_layernorm = te.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
            device=device,
        )
        self.mlp = NVMixtralMXFP8SparseMoeBlock(config, dispatcher)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        rotary_pos_emb: torch.Tensor | None = None,
        inference_params: InferenceParams | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        attn_output = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            inference_params=inference_params,
            cu_seqlens_q=kwargs.get("cu_seqlens_q", None),
            cu_seqlens_kv=kwargs.get("cu_seqlens_kv", None),
            cu_seqlens_q_padded=kwargs.get("cu_seqlens_q_padded", None),
            cu_seqlens_kv_padded=kwargs.get("cu_seqlens_kv_padded", None),
            max_seqlen_q=kwargs.get("max_seqlen_q", None),
            max_seqlen_kv=kwargs.get("max_seqlen_kv", None),
            pad_between_seqs=kwargs.get("pad_between_seqs", None),
        )
        hidden_states = hidden_states + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class NVMixtralMXFP8Model(NVMixtralMXFP8PreTrainedModel):
    """Embedding + N decoder layers + RMSNorm. THD-packed under MXFP8."""

    def __init__(
        self,
        config: NVMixtralMXFP8Config,
        fp8_recipe: te_recipe.Recipe | None = None,
        dispatcher: AllToAllTokenDispatcher | None = None,
    ) -> None:
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self._fp8_recipe = fp8_recipe

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx, dtype=config.dtype
        )

        layers: list[NVMixtralMXFP8DecoderLayer] = [
            NVMixtralMXFP8DecoderLayer(config, i, dispatcher)
            for i in range(config.num_hidden_layers)
        ]
        self.layers = nn.ModuleList(layers)

        self.norm = te.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
            device="meta" if torch.get_default_device() == torch.device("meta") else "cuda",
        )

        self.rotary_emb = RotaryPositionEmbedding(config.hidden_size // config.num_attention_heads)
        self.rotary_emb.inv_freq = LlamaRotaryEmbedding(config=config).inv_freq

        self.gradient_checkpointing = False
        self.post_init()

    def set_ep_groups(self, ep_group: dist.ProcessGroup, ep_mesh: Any) -> None:
        for layer in self.layers:
            layer.mlp.set_ep_group(ep_group, ep_mesh)

    def _outer_autocast(self) -> ContextManager:
        if self._fp8_recipe is None:
            return nullcontext()
        return te.autocast(enabled=True, recipe=self._fp8_recipe)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: InferenceParams | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        del position_ids, use_cache  # not used in this minimal forward

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        has_thd_input = [
            x in kwargs for x in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k")
        ]
        should_pack_inputs = not any(has_thd_input) and self.config.attn_input_format == "thd"

        thd_remainder = 0
        thd_orig_tokens = 0
        indices = None
        batch_size = 0
        padded_seq_len = 0
        if should_pack_inputs:
            assert attention_mask is not None, "attention_mask required when packing BSHD."
            batch_size = hidden_states.size(0)
            padded_seq_len = hidden_states.size(1)
            hidden_states, indices, cu_seqlens, max_seqlen, _ = _unpad_input(
                hidden_states, attention_mask
            )

            # MXFP8 requires total tokens divisible by 32; pad the last seq.
            thd_orig_tokens = hidden_states.shape[0]
            thd_remainder = thd_orig_tokens % 32
            if thd_remainder != 0:
                thd_pad = 32 - thd_remainder
                hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, thd_pad))
                cu_seqlens = cu_seqlens.clone()
                cu_seqlens[-1] = cu_seqlens[-1] + thd_pad
                max_seqlen = max_seqlen + thd_pad

            kwargs["cu_seq_lens_q"] = kwargs["cu_seq_lens_k"] = cu_seqlens
            kwargs["max_length_q"] = kwargs["max_length_k"] = max_seqlen

        if (
            self.config.attn_input_format == "thd"
            and hidden_states.dim() == 3
            and hidden_states.size(0) == 1
        ):
            hidden_states = hidden_states.squeeze(0)

        if (
            self.config.attn_input_format == "bshd"
            and attention_mask is not None
            and attention_mask.dim() == 2
        ):
            attention_mask = ~attention_mask[:, None, None, :].bool()

        if isinstance(past_key_values, InferenceParams):
            lengths = (
                attention_mask.sum(dim=1).tolist()
                if attention_mask.shape == input_ids.shape
                else [1] * input_ids.shape[0]
            )
            past_key_values.pre_step(OrderedDict(zip(list(range(len(lengths))), lengths)))

        with torch.autocast(device_type="cuda", enabled=False):
            te_rope_emb = self.rotary_emb(max_seq_len=self.config.max_position_embeddings)

        with self._outer_autocast():
            for layer_idx, decoder_layer in enumerate(self.layers):
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=(
                        None if self.config.attn_input_format == "thd" else attention_mask
                    ),
                    rotary_pos_emb=te_rope_emb,
                    inference_params=past_key_values,
                    cu_seqlens_q=kwargs.get("cu_seq_lens_q", None),
                    cu_seqlens_kv=kwargs.get("cu_seq_lens_k", None),
                    cu_seqlens_q_padded=kwargs.get("cu_seq_lens_q_padded", None),
                    cu_seqlens_kv_padded=kwargs.get("cu_seq_lens_k_padded", None),
                    max_seqlen_q=kwargs.get("max_length_q", None),
                    max_seqlen_kv=kwargs.get("max_length_k", None),
                    pad_between_seqs=kwargs.get("pad_between_seqs", None),
                )

        hidden_states = self.norm(hidden_states)

        if should_pack_inputs:
            if thd_remainder != 0:
                hidden_states = hidden_states[:thd_orig_tokens]
            hidden_states = _pad_input(hidden_states, indices, batch_size, padded_seq_len)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
        )


class NVMixtralMXFP8ForCausalLM(NVMixtralMXFP8PreTrainedModel):
    """Causal LM wrapper with MXFP8 autocast."""

    _tied_weights_keys: ClassVar[list[str]] = []

    def __init__(
        self,
        config: NVMixtralMXFP8Config,
        fp8_recipe: te_recipe.Recipe | None = None,
        dispatcher: AllToAllTokenDispatcher | None = None,
    ) -> None:
        super().__init__(config)
        self.model = NVMixtralMXFP8Model(config, fp8_recipe=fp8_recipe, dispatcher=dispatcher)
        self.vocab_size = config.vocab_size
        with te.quantized_model_init(enabled=False):
            self.lm_head = te.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                params_dtype=config.dtype,
                device="meta" if torch.get_default_device() == torch.device("meta") else "cuda",
                init_method=lambda x: torch.nn.init.normal_(
                    x, mean=0.0, std=config.initializer_range
                ),
            )
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        shift_labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        with te.autocast(enabled=False):
            if hidden_states.ndim == 3:
                logits = self.lm_head(hidden_states[:, slice_indices, :])
            else:
                logits = self.lm_head(hidden_states[slice_indices, :])

        loss = None
        if labels is not None or shift_labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                shift_labels=shift_labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if self.config.moe_aux_loss_coeff > 0 and loss is not None:
            aux_loss = sum(layer.mlp._aux_loss for layer in self.model.layers)
            loss = loss + aux_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
