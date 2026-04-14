# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TransformerEngine-optimized Mixtral model with Mixture of Experts."""

import logging
import os
import re
import warnings
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, ClassVar, ContextManager, Protocol, Unpack

import torch
import torch.distributed as dist
import torch.nn as nn
import transformer_engine.common.recipe
import transformer_engine.pytorch
import transformers
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.device_mesh import DeviceMesh
from transformer_engine.pytorch.attention import InferenceParams
from transformer_engine.pytorch.attention.inference import PagedKVCacheManager
from transformer_engine.pytorch.attention.rope import RotaryPositionEmbedding
from transformers import MixtralConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.utils.generic import TransformersKwargs


logger = logging.getLogger(__name__)


AUTO_MAP = {
    "AutoConfig": "modeling_mixtral_te.NVMixtralConfig",
    "AutoModel": "modeling_mixtral_te.NVMixtralModel",
    "AutoModelForCausalLM": "modeling_mixtral_te.NVMixtralForCausalLM",
}


def _copy_param(target: torch.Tensor, source: torch.Tensor) -> None:
    """Copy a source tensor into a target tensor, preserving target dtype/device."""
    if isinstance(target, DTensor):
        target = target.to_local()
    target.copy_(source.to(device=target.device, dtype=target.dtype))


def _copy_qkv_proj_to_fused(
    fused_qkv: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_kind: str,
    config: MixtralConfig,
) -> None:
    """Copy one HF Q/K/V projection into the TE fused QKV layout."""
    if isinstance(fused_qkv, DTensor):
        fused_qkv = fused_qkv.to_local()

    head_num = config.num_attention_heads
    num_query_groups = config.num_key_value_heads
    heads_per_group = head_num // num_query_groups
    hidden_size = config.hidden_size
    head_size = hidden_size // head_num
    qkv_total_dim = head_num + 2 * num_query_groups

    fused_view = fused_qkv.view(qkv_total_dim, head_size, hidden_size)
    proj_weight = proj_weight.to(device=fused_view.device, dtype=fused_view.dtype)

    if proj_kind == "q":
        q_view = proj_weight.view(head_num, head_size, hidden_size)
        for i in range(num_query_groups):
            start = (heads_per_group + 2) * i
            end = start + heads_per_group
            fused_view[start:end].copy_(q_view[i * heads_per_group : (i + 1) * heads_per_group])
    elif proj_kind == "k":
        k_view = proj_weight.view(num_query_groups, head_size, hidden_size)
        for i in range(num_query_groups):
            fused_view[(heads_per_group + 2) * i + heads_per_group].copy_(k_view[i])
    elif proj_kind == "v":
        v_view = proj_weight.view(num_query_groups, head_size, hidden_size)
        for i in range(num_query_groups):
            fused_view[(heads_per_group + 2) * i + heads_per_group + 1].copy_(v_view[i])
    else:
        raise ValueError(f"Unsupported proj_kind: {proj_kind}")


def replace_params(hf_state_dict: dict, te_state_dict: dict, config: MixtralConfig):
    """Copy HF Mixtral weights into a TE Mixtral state dict.

    This is a Mixtral-specific, shard-friendly helper analogous to the simpler
    Llama example. It replaces the generic convert/state pipeline with direct
    key-based copies plus explicit packing for:

    - HF separate Q/K/V projections -> TE fused QKV weight
    - HF expert FFN weights -> TE stacked expert tensors

    The helper supports both packed HF MoE tensors (`mlp.experts.gate_up_proj`)
    and older per-expert tensors (`experts.{i}.w{1,2,3}.weight`).
    """
    all_layer_prefixes = set()
    for param_key in hf_state_dict.keys():
        m = re.match(r"model\.layers\.\d+\.", param_key)
        if m is not None:
            all_layer_prefixes.add(m.group())

    direct_top_level_mappings = {
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "model.norm.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
        "model.rotary_emb.inv_freq": "model.rotary_emb.inv_freq",
    }
    for hf_key, te_key in direct_top_level_mappings.items():
        if hf_key in hf_state_dict and te_key in te_state_dict:
            _copy_param(te_state_dict[te_key], hf_state_dict[hf_key])

    for layer_prefix in all_layer_prefixes:
        direct_layer_mappings = {
            layer_prefix + "input_layernorm.weight": layer_prefix + "self_attention.layernorm_qkv.layer_norm_weight",
            layer_prefix + "self_attn.o_proj.weight": layer_prefix + "self_attention.proj.weight",
            layer_prefix + "post_attention_layernorm.weight": layer_prefix + "post_attention_layernorm.weight",
        }
        for hf_key, te_key in direct_layer_mappings.items():
            if hf_key in hf_state_dict and te_key in te_state_dict:
                _copy_param(te_state_dict[te_key], hf_state_dict[hf_key])

        fused_qkv_key = layer_prefix + "self_attention.layernorm_qkv.weight"
        if fused_qkv_key in te_state_dict:
            qkv_sources = {
                "q": layer_prefix + "self_attn.q_proj.weight",
                "k": layer_prefix + "self_attn.k_proj.weight",
                "v": layer_prefix + "self_attn.v_proj.weight",
            }
            for proj_kind, hf_key in qkv_sources.items():
                if hf_key in hf_state_dict:
                    _copy_qkv_proj_to_fused(te_state_dict[fused_qkv_key], hf_state_dict[hf_key], proj_kind, config)

        gate_candidates = (
            layer_prefix + "mlp.gate.weight",
            layer_prefix + "block_sparse_moe.gate.weight",
        )
        te_gate_key = layer_prefix + "mlp.gate.weight"
        for hf_key in gate_candidates:
            if hf_key in hf_state_dict and te_gate_key in te_state_dict:
                _copy_param(te_state_dict[te_gate_key], hf_state_dict[hf_key])
                break

        packed_gate_up_candidates = (
            layer_prefix + "mlp.experts.gate_up_proj",
            layer_prefix + "block_sparse_moe.experts.gate_up_proj",
        )
        te_gate_up_key = layer_prefix + "mlp.experts_gate_up_weight"
        for hf_key in packed_gate_up_candidates:
            if hf_key in hf_state_dict and te_gate_up_key in te_state_dict:
                _copy_param(te_state_dict[te_gate_up_key], hf_state_dict[hf_key])
                break

        packed_down_candidates = (
            layer_prefix + "mlp.experts.down_proj",
            layer_prefix + "block_sparse_moe.experts.down_proj",
        )
        te_down_key = layer_prefix + "mlp.experts_down_weight"
        for hf_key in packed_down_candidates:
            if hf_key in hf_state_dict and te_down_key in te_state_dict:
                _copy_param(te_state_dict[te_down_key], hf_state_dict[hf_key])
                break

        # Older HF Mixtral checkpoints may store one tensor per expert.
        if te_gate_up_key in te_state_dict and te_down_key in te_state_dict:
            te_gate_up = te_state_dict[te_gate_up_key]
            te_down = te_state_dict[te_down_key]
            if isinstance(te_gate_up, DTensor):
                te_gate_up = te_gate_up.to_local()
            if isinstance(te_down, DTensor):
                te_down = te_down.to_local()

            num_local_experts = te_gate_up.shape[0]
            for expert_idx in range(num_local_experts):
                expert_prefixes = (
                    layer_prefix + f"mlp.experts.{expert_idx}.",
                    layer_prefix + f"block_sparse_moe.experts.{expert_idx}.",
                )
                for expert_prefix in expert_prefixes:
                    w1_key = expert_prefix + "w1.weight"
                    w3_key = expert_prefix + "w3.weight"
                    w2_key = expert_prefix + "w2.weight"

                    if w1_key in hf_state_dict:
                        te_gate_up[expert_idx, : config.intermediate_size].copy_(
                            hf_state_dict[w1_key].to(device=te_gate_up.device, dtype=te_gate_up.dtype)
                        )
                    if w3_key in hf_state_dict:
                        te_gate_up[expert_idx, config.intermediate_size :].copy_(
                            hf_state_dict[w3_key].to(device=te_gate_up.device, dtype=te_gate_up.dtype)
                        )
                    if w2_key in hf_state_dict:
                        te_down[expert_idx].copy_(
                            hf_state_dict[w2_key].to(device=te_down.device, dtype=te_down.dtype)
                        )

    return all_layer_prefixes


class NVMixtralConfig(MixtralConfig):
    """NVMixtral configuration."""

    # Attention input format:
    #   "bshd" = Batch, Sequence, Head, Dimension (standard padded format)
    #   "thd"  = Total tokens (packed/unpadded), Head, Dimension (sequence packing format)
    attn_input_format: str = "thd"
    self_attn_mask_type: str = "padding_causal"
    layer_precision: list[str | None] | None = None
    use_quantized_model_init: bool = False
    expert_parallel_size: int = 1
    moe_aux_loss_coeff: float = 0.0

    def __init__(self, **kwargs):
        """Initialize the NVMixtralConfig with additional TE-related config options."""
        super().__init__(**kwargs)

        if self.layer_precision is not None:
            if len(self.layer_precision) != self.num_hidden_layers:
                raise ValueError(f"layer_precision must be a list of length {self.num_hidden_layers}")
            for precision in self.layer_precision:
                if precision not in {"fp8", "fp4", None}:
                    raise ValueError(f'layer_precision element must be "fp8", "fp4", or None, got {precision!r}')

        if self.num_local_experts % self.expert_parallel_size != 0:
            raise ValueError(
                f"num_local_experts ({self.num_local_experts}) must be divisible by "
                f"expert_parallel_size ({self.expert_parallel_size})"
            )


@dataclass
class DispatchOutput:
    """Output of TokenDispatcher.dispatch().

    Attributes:
        expert_input: Tokens sorted by local expert, shape ``[total_recv_tokens, H]``.
        tokens_per_expert: Token count per local expert.
        handle: Opaque state needed by ``combine()`` to reverse the dispatch.
    """

    expert_input: torch.Tensor
    tokens_per_expert: list[int]
    handle: Any


class TokenDispatcher(Protocol):
    """Protocol for MoE token dispatch/combine strategies.

    Encapsulates the full dispatch cycle (permute -> communicate -> sort) and
    combine cycle (unsort -> communicate -> unpermute) so that the MoE block
    is agnostic to the communication backend (NCCL all-to-all, HybridEP, etc.).
    """

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> DispatchOutput:
        """Dispatch tokens to their assigned experts.

        Args:
            hidden_states: Flattened input tensor of shape ``[N, H]``.
            selected_experts: Expert assignments, shape ``[N, top_k]``, int.
            routing_weights: Normalized routing probabilities, shape ``[N, top_k]``, float32.

        Returns:
            DispatchOutput with expert-sorted tokens, per-expert counts, and an opaque handle.
        """
        ...

    def combine(
        self,
        expert_output: torch.Tensor,
        handle: Any,
    ) -> torch.Tensor:
        """Combine expert outputs back to the original token order.

        Args:
            expert_output: Expert output tensor of shape ``[total_recv_tokens, H]``.
            handle: Opaque state from ``dispatch()``.

        Returns:
            Combined output tensor of shape ``[N, H]`` with routing weights applied.
        """
        ...

    def set_ep_group(self, ep_group: dist.ProcessGroup) -> None:
        """Set the expert-parallel process group for communication."""
        ...


class NVMixtralPreTrainedModel(PreTrainedModel):
    """Base class for NVMixtral models."""

    config_class = NVMixtralConfig
    base_model_prefix = "model"
    _no_split_modules = ("NVMixtralDecoderLayer",)
    _skip_keys_device_placement = ("past_key_values",)
    _do_not_quantize = ("lm_head", "model.layers.*.mlp.gate")  # Flag for testing that these layers are not quantized.

    def init_empty_weights(self):
        """Handles moving the model from the meta device to the cuda device and initializing the weights."""
        for module in self.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        # After reset_parameters materializes GroupedLinear views on CUDA,
        # re-stack them into the authoritative stacked parameters.
        for module in self.modules():
            if isinstance(module, NVMixtralSparseMoeBlock):
                module._restack_from_views()

        self.model.embed_tokens.to_empty(device="cuda")
        self.model.embed_tokens.apply(self._init_weights)

        self.model.rotary_emb.inv_freq = LlamaRotaryEmbedding(config=self.model.config).inv_freq.to("cuda")

        self.tie_weights()

    def _init_weights(self, module):
        """Initialize module weights.

        We only use this method for standard pytorch modules, TE modules handle their own weight initialization through
        `init_method` parameters and the `reset_parameters` method.
        """
        if module.__module__.startswith("transformer_engine.pytorch"):
            return

        super()._init_weights(module)

    def state_dict(self, *args, **kwargs):
        """Override state_dict to filter out TransformerEngine's _extra_state keys."""
        state_dict = super().state_dict(*args, **kwargs)
        return {k: v for k, v in state_dict.items() if not k.endswith("_extra_state")}


class NVMixtralSparseMoeBlock(nn.Module):
    """Mixture of Experts block using TransformerEngine GroupedLinear."""

    def __init__(self, config: MixtralConfig, dispatcher: TokenDispatcher | None = None):
        """Initialize the sparse MoE block."""
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.jitter_noise = config.router_jitter_noise

        # Expert parallelism
        self.ep_size = getattr(config, "expert_parallel_size", 1)
        self.num_local_experts = self.num_experts // self.ep_size
        self.moe_aux_loss_coeff = getattr(config, "moe_aux_loss_coeff", 0.0)
        self._aux_loss: torch.Tensor = torch.tensor(0.0)
        self.initializer_range = config.initializer_range

        self.dispatcher: TokenDispatcher = dispatcher or AllToAllTokenDispatcher(
            self.num_experts,
            self.num_local_experts,
            self.hidden_size,
            self.ep_size,
        )

        device = "meta" if torch.get_default_device() == torch.device("meta") else "cuda"

        def _init_method(x):
            torch.nn.init.normal_(x, mean=0.0, std=config.initializer_range)

        # Router always outputs num_experts logits (replicated across EP ranks)
        with transformer_engine.pytorch.quantized_model_init(enabled=False):
            self.gate = transformer_engine.pytorch.Linear(
                self.hidden_size,
                self.num_experts,
                bias=False,
                device=device,
                params_dtype=config.dtype,
                init_method=_init_method,
            )

        # Expert FFNs — only num_local_experts per rank when EP > 1
        self.experts_gate_up = transformer_engine.pytorch.GroupedLinear(
            num_gemms=self.num_local_experts,
            in_features=self.hidden_size,
            out_features=2 * self.intermediate_size,
            bias=False,
            params_dtype=config.dtype,
            device=device,
            init_method=_init_method,
        )
        self.experts_down = transformer_engine.pytorch.GroupedLinear(
            num_gemms=self.num_local_experts,
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            bias=False,
            params_dtype=config.dtype,
            device=device,
            init_method=_init_method,
        )

        # Stack per-expert weights into single parameters (authoritative weight store).
        # GroupedLinear's _parameters dict is emptied; weight attributes are set as views
        # so that reset_parameters() / _get_weight_tensors() can still find them.
        self.experts_gate_up_weight = nn.Parameter(
            torch.stack(
                [self.experts_gate_up._parameters.pop(f"weight{i}").data for i in range(self.num_local_experts)]
            )
        )  # [num_local_experts, 2*intermediate_size, hidden_size]

        self.experts_down_weight = nn.Parameter(
            torch.stack([self.experts_down._parameters.pop(f"weight{i}").data for i in range(self.num_local_experts)])
        )  # [num_local_experts, hidden_size, intermediate_size]

        # Set views back on GroupedLinear so getattr(self, "weight{i}") still works
        # (needed by GroupedLinear.reset_parameters and _get_weight_tensors).
        self._sync_expert_views()

    def _restack_from_views(self) -> None:
        """Re-create stacked parameters on CUDA after meta init.

        Called by ``init_empty_weights()`` after ``reset_parameters()`` has been called
        on all TE modules. Since GroupedLinear has no registered parameters (we popped them),
        its ``reset_parameters()`` cannot move them from meta to CUDA. This method explicitly
        creates the stacked parameters on CUDA and reinitializes them.
        """
        device = torch.cuda.current_device()
        for attr_name in ("experts_gate_up_weight", "experts_down_weight"):
            old_param = getattr(self, attr_name)
            new_data = torch.empty_like(old_param, device=device)
            torch.nn.init.normal_(new_data, mean=0.0, std=self.initializer_range)
            setattr(self, attr_name, nn.Parameter(new_data))

        # Re-sync views to point to the new stacked parameter
        self._sync_expert_views()

    def _sync_expert_views(self) -> None:
        """Set GroupedLinear weight attributes as views of the stacked parameters.

        GroupedLinear internally uses ``getattr(self, f"weight{i}")`` in methods like
        ``reset_parameters()`` and ``_get_weight_tensors()``. After popping the original
        parameters, we set views of the stacked tensor so these methods keep working.
        Uses ``object.__setattr__`` to bypass ``nn.Module.__setattr__`` and avoid
        re-registering them as parameters.
        """
        gate_up_w = self.experts_gate_up_weight
        if isinstance(gate_up_w, DTensor):
            gate_up_w = gate_up_w.to_local()
        for i in range(self.num_local_experts):
            object.__setattr__(self.experts_gate_up, f"weight{i}", gate_up_w[i])

        down_w = self.experts_down_weight
        if isinstance(down_w, DTensor):
            down_w = down_w.to_local()
        for i in range(self.num_local_experts):
            object.__setattr__(self.experts_down, f"weight{i}", down_w[i])

    def set_ep_group(self, ep_group: dist.ProcessGroup, ep_mesh: DeviceMesh) -> None:
        """Set the expert-parallel process group and convert stacked weights to DTensors.

        Must be called before the first forward pass when ``ep_size > 1``.

        Args:
            ep_group: A ``torch.distributed.ProcessGroup`` whose world size equals ``self.ep_size``.
            ep_mesh: A 1-D ``DeviceMesh`` for expert parallelism. Used to wrap stacked weights
                as ``DTensor(Shard(0))`` so that DCP can save/load/reshard them automatically.
        """
        self.dispatcher.set_ep_group(ep_group)
        # Convert stacked parameters to DTensors with Shard(0) on the expert dimension.
        # Global shape is [num_experts, ...]; each rank stores [num_local_experts, ...].
        # Guard: only wrap plain tensors; skip if already DTensors (e.g. repeated calls).
        if not isinstance(self.experts_gate_up_weight.data, DTensor):
            self.experts_gate_up_weight = nn.Parameter(
                DTensor.from_local(self.experts_gate_up_weight.data, device_mesh=ep_mesh, placements=[Shard(0)])
            )
        if not isinstance(self.experts_down_weight.data, DTensor):
            self.experts_down_weight = nn.Parameter(
                DTensor.from_local(self.experts_down_weight.data, device_mesh=ep_mesh, placements=[Shard(0)])
            )

    def _expert_ffn(self, tokens: torch.Tensor, m_splits: list[int]) -> torch.Tensor:
        """Run the expert SwiGLU FFN (gate_up -> silu -> down).

        Args:
            tokens: Input tensor of shape [total_tokens, H], sorted by expert.
            m_splits: Number of tokens per local expert.

        Returns:
            Output tensor of shape [total_tokens, H].
        """
        gate_up_output = self.experts_gate_up(tokens, m_splits=m_splits)
        gate_output, up_output = gate_up_output.chunk(2, dim=-1)
        intermediate = torch.nn.functional.silu(gate_output) * up_output
        return self.experts_down(intermediate, m_splits=m_splits)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MoE block.

        Args:
            hidden_states: Input tensor of shape [B, S, H] (bshd) or [T, H] (thd).

        Returns:
            Output tensor of the same shape as the input.
        """
        original_shape = hidden_states.shape

        # Apply multiplicative jitter noise to hidden states during training to encourage load balancing
        if self.training and self.jitter_noise > 0:
            hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )

        # Flatten to [N, H] for routing
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.reshape(-1, self.hidden_size)

        # Router: compute expert assignments
        with transformer_engine.pytorch.autocast(enabled=False):
            # Keep the router logits in bf16 during FP8 training
            router_logits = self.gate(hidden_states)  # [N, num_experts]

        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)  # [N, top_k]
        # Normalize routing weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # Auxiliary load-balancing loss (switch transformer style)
        if self.moe_aux_loss_coeff > 0:
            num_tokens = hidden_states.shape[0]
            m_splits_tensor = torch.bincount(selected_experts.reshape(-1), minlength=self.num_experts).int()
            # f_i: fraction of tokens dispatched to each expert
            f = m_splits_tensor.float() / (num_tokens * self.top_k)
            # P_i: mean router probability per expert (over all tokens)
            router_probs = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float32)
            p = router_probs.mean(dim=0)
            self._aux_loss = self.moe_aux_loss_coeff * self.num_experts * (f * p).sum()
        else:
            self._aux_loss = torch.tensor(0.0, device=hidden_states.device)

        # Populate GroupedLinear weight attributes from stacked parameters.
        # For EP, the stacked parameter is a DTensor; .to_local() gives the local shard.
        self._sync_expert_views()

        dispatch_output = self.dispatcher.dispatch(hidden_states, selected_experts, routing_weights)
        expert_output = self._expert_ffn(dispatch_output.expert_input, dispatch_output.tokens_per_expert)
        output = self.dispatcher.combine(expert_output, dispatch_output.handle)

        return output.reshape(original_shape)


class NVMixtralDecoderLayer(nn.Module):
    """Mixtral decoder layer using TE attention and MoE MLP."""

    def __init__(self, config: MixtralConfig, layer_idx: int, dispatcher: TokenDispatcher | None = None):
        """Initialize the decoder layer."""
        super().__init__()
        self.hidden_size = config.hidden_size

        device = "meta" if torch.get_default_device() == torch.device("meta") else "cuda"

        def _init_method(x):
            torch.nn.init.normal_(x, mean=0.0, std=config.initializer_range)

        self.self_attention = transformer_engine.pytorch.MultiheadAttention(
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

        self.post_attention_layernorm = transformer_engine.pytorch.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
            device=device,
        )

        self.mlp = NVMixtralSparseMoeBlock(config, dispatcher)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        rotary_pos_emb: torch.Tensor | None = None,
        inference_params: InferenceParams | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for the decoder layer."""
        # Self attention with fused input layernorm
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

        # Residual connection
        hidden_states = hidden_states + attn_output

        # Post-attention layernorm + MoE MLP + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class NVMixtralModel(NVMixtralPreTrainedModel):
    """Mixtral model implemented in Transformer Engine."""

    def __init__(
        self,
        config: MixtralConfig,
        fp8_recipe: transformer_engine.common.recipe.Recipe | None = None,
        fp4_recipe: transformer_engine.common.recipe.Recipe | None = None,
        dispatcher: TokenDispatcher | None = None,
    ):
        """Initialize the NVMixtral model.

        Args:
            config: The configuration of the model.
            fp8_recipe: The FP8 recipe for the model.
            fp4_recipe: The FP4 recipe for the model.
            dispatcher: The token dispatcher for the model. If None, the default AllToAllTokenDispatcher will be used.
        """
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self._fp8_recipe: transformer_engine.common.recipe.Recipe | None = fp8_recipe
        self._fp4_recipe: transformer_engine.common.recipe.Recipe | None = fp4_recipe

        if fp8_recipe is not None and self.config.layer_precision is None:
            if fp4_recipe is not None:
                raise RuntimeError("Both FP8 and FP4 recipes provided, but no layer precision provided.")

            warnings.warn("No layer precision provided, using FP8 recipe for all layers.", UserWarning)
            self.config.layer_precision = ["fp8"] * self.config.num_hidden_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx, dtype=config.dtype)

        layers: list[NVMixtralDecoderLayer] = []
        for layer_idx in range(config.num_hidden_layers):
            with self.get_autocast_context(layer_idx, init=True):
                layers += [NVMixtralDecoderLayer(config, layer_idx, dispatcher)]

        self.layers = nn.ModuleList(layers)

        self.norm = transformer_engine.pytorch.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
            device="meta" if torch.get_default_device() == torch.device("meta") else "cuda",
        )

        self.rotary_emb = RotaryPositionEmbedding(config.hidden_size // config.num_attention_heads)
        self.rotary_emb.inv_freq = LlamaRotaryEmbedding(config=config).inv_freq

        self.gradient_checkpointing = False

        self.post_init()

    def set_ep_groups(self, ep_group: dist.ProcessGroup, ep_mesh: DeviceMesh) -> None:
        """Propagate an expert-parallel process group and mesh to every MoE block.

        Args:
            ep_group: The EP process group to set on each ``NVMixtralSparseMoeBlock``.
            ep_mesh: A 1-D ``DeviceMesh`` for expert parallelism.
        """
        for layer in self.layers:
            layer.mlp.set_ep_group(ep_group, ep_mesh)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: InferenceParams | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        """Forward pass for the NVMixtral model."""
        all_hidden_states = []
        output_hidden_states = kwargs.get("output_hidden_states", False)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # TE-specific input handling
        has_thd_input = [x in kwargs for x in ["cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"]]
        should_pack_inputs = not any(has_thd_input) and self.config.attn_input_format == "thd"

        if should_pack_inputs:
            assert attention_mask is not None, "Attention mask is required when packing BSHD inputs."
            batch_size = hidden_states.size(0)
            padded_seq_len = input_ids.size(1)
            hidden_states, indices, cu_seqlens, max_seqlen, _ = _unpad_input(hidden_states, attention_mask)
            kwargs["cu_seq_lens_q"] = kwargs["cu_seq_lens_k"] = cu_seqlens
            kwargs["max_length_q"] = kwargs["max_length_k"] = max_seqlen

        if self.config.attn_input_format == "thd" and hidden_states.dim() == 3 and hidden_states.size(0) == 1:
            hidden_states = hidden_states.squeeze(0)

        if self.config.attn_input_format == "bshd" and attention_mask is not None and attention_mask.dim() == 2:
            # Convert HF mask (1=attend, 0=pad) to TE boolean mask (True=masked, False=attend)
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

        with self.get_autocast_context(None, outer=True):
            for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
                if output_hidden_states:
                    all_hidden_states = (*all_hidden_states, hidden_states)

                with self.get_autocast_context(layer_idx):
                    hidden_states = decoder_layer(
                        hidden_states,
                        attention_mask=None if self.config.attn_input_format == "thd" else attention_mask,
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

        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        if should_pack_inputs:
            hidden_states = _pad_input(hidden_states, indices, batch_size, padded_seq_len)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states if output_hidden_states else None,
        )

    def get_autocast_context(
        self, layer_number: int | None, init: bool = False, outer: bool = False
    ) -> ContextManager:
        """Return the appropriate TE autocast context manager for a given layer.

        This function handles both the quantized_model_init during layer creation and the te.autocast() during layer
        forward pass.

        Args:
            layer_number: The 0-indexed layer number.
            init: Whether to return a `quantized_model_init` context for layer initialization.
            outer: Whether to return a global te.autocast() context to wrap the entire model stack.
        """
        if self.config.layer_precision is None:
            return nullcontext()

        if outer:
            if "fp8" not in self.config.layer_precision:
                return nullcontext()
            if self._fp8_recipe is None:
                warnings.warn("No FP8 recipe provided, using default recipe.", UserWarning)
            return transformer_engine.pytorch.autocast(enabled=True, recipe=self._fp8_recipe)

        precision = self.config.layer_precision[layer_number]
        recipe = {"fp8": self._fp8_recipe, "fp4": self._fp4_recipe}.get(precision)

        if init and self.config.use_quantized_model_init:
            if precision in ("fp8", "fp4"):
                return transformer_engine.pytorch.quantized_model_init(recipe=recipe)
            return nullcontext()

        if precision == "fp8":
            if recipe is None:
                warnings.warn("No FP8 recipe provided, using default recipe.", UserWarning)
            return transformer_engine.pytorch.autocast(enabled=True, recipe=recipe)
        if precision == "fp4":
            if recipe is None:
                raise RuntimeError("No FP4 recipe provided, but layer precision is set to FP4.")
            return transformer_engine.pytorch.autocast(enabled=True, recipe=recipe)
        return transformer_engine.pytorch.autocast(enabled=False)


class NVMixtralForCausalLM(NVMixtralPreTrainedModel, transformers.GenerationMixin):
    """Mixtral model with causal language head."""

    _tied_weights_keys: ClassVar[list[str]] = []

    def __init__(
        self,
        config,
        fp8_recipe: transformer_engine.common.recipe.Recipe | None = None,
        fp4_recipe: transformer_engine.common.recipe.Recipe | None = None,
        dispatcher: TokenDispatcher | None = None,
    ):
        """Initialize the NVMixtralForCausalLM model.

        Args:
            config: The configuration of the model.
            fp8_recipe: The FP8 recipe for the model.
            fp4_recipe: The FP4 recipe for the model.
            dispatcher: The token dispatcher for expert parallelism. If None, the default
                AllToAllTokenDispatcher will be used.
        """
        super().__init__(config)
        self.model = NVMixtralModel(config, fp8_recipe=fp8_recipe, fp4_recipe=fp4_recipe, dispatcher=dispatcher)
        self.vocab_size = config.vocab_size

        with transformer_engine.pytorch.quantized_model_init(enabled=False):
            self.lm_head = transformer_engine.pytorch.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                params_dtype=config.dtype,
                device="meta" if torch.get_default_device() == torch.device("meta") else "cuda",
                init_method=lambda x: torch.nn.init.normal_(x, mean=0.0, std=config.initializer_range),
            )

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor, ...], ...] | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        shift_labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        """Forward pass for the NVMixtralForCausalLM model."""
        outputs: BaseModelOutputWithPast = self.model(
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
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        with transformer_engine.pytorch.autocast(enabled=False):
            if hidden_states.ndim == 3:
                logits = self.lm_head(hidden_states[:, slice_indices, :])
            else:
                logits = self.lm_head(hidden_states[slice_indices, :])

        loss = None
        if labels is not None or shift_labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, shift_labels=shift_labels, vocab_size=self.config.vocab_size, **kwargs
            )

        # Collect auxiliary load-balancing loss from all MoE layers
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


def save_final_model_ep(
    model: NVMixtralForCausalLM,
    save_directory: str | os.PathLike,
    dist_config=None,
) -> None:
    """Gather all EP-sharded expert weights and save as safetensors.

    Uses ``get_model_state_dict(full_state_dict=True)`` to all-gather DTensors,
    matching the pattern from ``save_final_model_fsdp2`` in the llama3 checkpoint module.

    All ranks must call this function. Only rank 0 writes files.

    Args:
        model: The NVMixtral model (may have DTensor expert parameters).
        save_directory: Directory to save ``model.safetensors`` and config.
        dist_config: Optional distributed config with ``is_main_process()`` method.
            If ``None``, only rank 0 saves.
    """
    from safetensors.torch import save_file

    model_state_dict = get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )

    # Filter out TE _extra_state keys
    model_state_dict = {k: v for k, v in model_state_dict.items() if not k.endswith("_extra_state")}

    is_main = dist_config.is_main_process() if dist_config is not None else (dist.get_rank() == 0)
    if is_main:
        os.makedirs(save_directory, exist_ok=True)
        save_file(model_state_dict, os.path.join(save_directory, "model.safetensors"))
        model.config.save_pretrained(save_directory)
        logger.info(f"Saved final EP model to {save_directory}")


# Required for torch.compile'd functions below (_pad_input, _unpad_input, _build_expert_sort_indices)
# that use data-dependent scalar values (e.g., max_seqlen_in_batch.item()) or produce tensors
# whose shape depends on input data (e.g., repeat_interleave with tensor counts).
# These must be set at module level because torch.compile traces lazily on first call,
# so a scoped setting would not be active at trace time.
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True


@torch.compile
def _pad_input(hidden_states, indices, batch, seqlen):
    """Convert a THD tensor to a BSHD equivalent tensor.

    Adapted from huggingface/transformers/modeling_flash_attention_utils.py
    """
    dim = hidden_states.shape[1:]
    output = torch.zeros((batch * seqlen), *dim, device=hidden_states.device, dtype=hidden_states.dtype)
    output[indices] = hidden_states
    return output.view(batch, seqlen, *dim)


@torch.compile
def _unpad_input(hidden_states, attention_mask, unused_mask=None):
    """Convert a BSHD tensor to a THD equivalent tensor.

    Adapted from huggingface/transformers/modeling_flash_attention_utils.py
    """
    batch_size = hidden_states.size(0)
    seq_length = hidden_states.size(1)

    if attention_mask.shape[1] != seq_length:
        return (
            hidden_states.squeeze(1),
            torch.arange(batch_size, dtype=torch.int64, device=hidden_states.device),
            torch.arange(batch_size + 1, dtype=torch.int32, device=hidden_states.device),
            1,
            1,
        )

    all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

    return (
        hidden_states.reshape(-1, *hidden_states.shape[2:])[indices],
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        used_seqlens_in_batch,
    )


class HFInferenceParams(InferenceParams):
    """Extension of the InferenceParams class to support HF generate() and beam search."""

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the current cached sequence length.

        Required by HuggingFace transformers generate() to determine how many
        tokens have already been cached.
        """
        if not self.sequences:
            return 0
        return max(self.sequences.values())

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorder the cache based on the beam indices."""
        if isinstance(self.cache_manager, PagedKVCacheManager):
            raise NotImplementedError("Beam search is not supported for paged cache manager.")
        for layer_number, (key_cache, value_cache) in self.cache_manager.cache.items():
            updated_key_cache = key_cache.index_select(0, beam_idx)
            updated_value_cache = value_cache.index_select(0, beam_idx)
            self.cache_manager.cache[layer_number] = (updated_key_cache, updated_value_cache)


@torch.compile(fullgraph=True)
def _build_expert_sort_indices(recv_counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build sort and unsort index tensors for reordering received tokens by local expert.

    After all-to-all, tokens arrive grouped by source rank:
    ``[src0_exp0..src0_expL, src1_exp0..src1_expL, ...]``. ``GroupedLinear`` expects them
    grouped by expert: ``[all_exp0, all_exp1, ...]``.

    Uses only vectorized tensor operations (no ``.item()`` calls or Python-level loops)
    so that it is compatible with ``torch.compile(fullgraph=True)``.

    Args:
        recv_counts: Integer tensor of shape ``[ep_size, num_local_experts]`` giving the
            number of tokens received from each source rank for each local expert.

    Returns:
        A ``(sort_indices, unsort_indices)`` pair of 1-D ``int64`` tensors that can be
        used to reorder and restore the token dimension.
    """
    ep_size, num_local_experts = recv_counts.shape
    device = recv_counts.device
    num_blocks = ep_size * num_local_experts

    # Source-grouped (row-major) block offsets: [s0e0, s0e1, ..., s1e0, s1e1, ...]
    counts_src = recv_counts.reshape(-1).long()
    offsets_src = torch.zeros(num_blocks, dtype=torch.long, device=device)
    offsets_src[1:] = counts_src[:-1].cumsum(0)

    # Expert-grouped (column-major) block offsets: [e0s0, e0s1, ..., e1s0, e1s1, ...]
    counts_exp = recv_counts.t().contiguous().reshape(-1).long()
    offsets_exp = torch.zeros(num_blocks, dtype=torch.long, device=device)
    offsets_exp[1:] = counts_exp[:-1].cumsum(0)

    total = counts_src.sum()

    # Mapping from source block index (s * L + e) to expert block index (e * S + s)
    s_idx = torch.arange(ep_size, device=device).unsqueeze(1).expand(ep_size, num_local_experts)
    e_idx = torch.arange(num_local_experts, device=device).unsqueeze(0).expand(ep_size, num_local_experts)
    src_to_exp = (e_idx * ep_size + s_idx).reshape(-1)

    # Per-block positional shift from source layout to expert layout
    shifts = offsets_exp[src_to_exp] - offsets_src

    # Expand per-block shifts to per-token
    token_shifts = shifts.repeat_interleave(counts_src)

    # Map each source-grouped position to its expert-grouped destination
    src_positions = torch.arange(total, device=device)
    dst_positions = src_positions + token_shifts

    # sort_indices[exp_pos] = src_pos (gathers source tokens into expert order)
    sort_indices = torch.empty(total, dtype=torch.long, device=device)
    sort_indices[dst_positions] = src_positions

    # unsort_indices: inverse permutation (restores expert-ordered output to source order)
    unsort_indices = torch.empty_like(sort_indices)
    unsort_indices[sort_indices] = torch.arange(total, device=device)

    return sort_indices, unsort_indices


@dataclass
class _AllToAllHandle:
    """Opaque handle for AllToAllTokenDispatcher, storing state between dispatch and combine."""

    row_id_map: torch.Tensor
    routing_weights: torch.Tensor
    unsort_indices: torch.Tensor | None = None
    input_split_sizes: list[int] | None = None
    output_split_sizes: list[int] | None = None


class _DifferentiableAllToAll(torch.autograd.Function):
    """Differentiable wrapper around dist.all_to_all_single.

    The forward pass performs the standard all-to-all communication.
    The backward pass reverses the communication direction (swapping
    input/output split sizes) so that gradients flow correctly.
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        output_split_sizes: list[int],
        input_split_sizes: list[int],
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        """Perform all-to-all forward and save sizes for backward."""
        ctx.input_split_sizes = input_split_sizes
        ctx.output_split_sizes = output_split_sizes
        ctx.group = group
        output = torch.empty(
            sum(output_split_sizes),
            input.shape[1],
            device=input.device,
            dtype=input.dtype,
        )
        dist.all_to_all_single(output, input.contiguous(), output_split_sizes, input_split_sizes, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        """Reverse all-to-all: swap input and output split sizes."""
        grad_input = torch.empty(
            sum(ctx.input_split_sizes),
            grad_output.shape[1],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        dist.all_to_all_single(
            grad_input,
            grad_output.contiguous(),
            ctx.input_split_sizes,
            ctx.output_split_sizes,
            group=ctx.group,
        )
        return grad_input, None, None, None


class AllToAllTokenDispatcher:
    """TokenDispatcher using NCCL all-to-all for expert-parallel communication.

    Handles both EP=1 (no communication, just permute/unpermute) and EP>1
    (all-to-all token exchange between ranks) cases transparently.

    Args:
        num_experts: Total number of experts (global).
        num_local_experts: Number of experts on this rank.
        hidden_size: Hidden dimension size.
        ep_size: Expert parallel world size.
    """

    def __init__(self, num_experts: int, num_local_experts: int, hidden_size: int, ep_size: int):
        """Initialize the AllToAllTokenDispatcher."""
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.ep_size = ep_size
        self._ep_group: dist.ProcessGroup | None = None

    def set_ep_group(self, ep_group: dist.ProcessGroup) -> None:
        """Set the expert-parallel process group for all-to-all communication."""
        self._ep_group = ep_group

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> DispatchOutput:
        """Dispatch tokens to their assigned experts via permute and optional all-to-all.

        Args:
            hidden_states: Flattened input tensor of shape ``[N, H]``.
            selected_experts: Expert assignments, shape ``[N, top_k]``, int.
            routing_weights: Normalized routing probabilities, shape ``[N, top_k]``, float32.

        Returns:
            DispatchOutput with expert-sorted tokens, per-expert counts, and an opaque handle.
        """
        # Permute tokens by expert using TE moe_permute
        permuted_hidden, row_id_map = transformer_engine.pytorch.moe_permute(
            hidden_states, selected_experts.to(torch.int32), map_type="index"
        )

        # Compute m_splits: number of tokens per expert
        m_splits_tensor = torch.bincount(selected_experts.reshape(-1), minlength=self.num_experts).int()

        if self._ep_group is not None:
            ep_group = self._ep_group

            # Token counts per expert, reshaped to [ep_size, num_local_experts]
            send_counts = m_splits_tensor.reshape(self.ep_size, self.num_local_experts)

            # Exchange per-expert token counts between EP ranks
            recv_counts = torch.empty_like(send_counts)
            dist.all_to_all_single(recv_counts.flatten(), send_counts.flatten(), group=ep_group)

            # Derive split sizes for the token all-to-all
            input_split_sizes = send_counts.sum(dim=1).tolist()
            output_split_sizes = recv_counts.sum(dim=1).tolist()
            local_m_splits = recv_counts.sum(dim=0).int().tolist()

            # Dispatch tokens to expert-owning ranks (differentiable)
            recv_tokens = _DifferentiableAllToAll.apply(
                permuted_hidden, output_split_sizes, input_split_sizes, ep_group
            )

            # Sort received tokens by local expert index.
            # After all_to_all layout is [src0_exp0..src0_expL, src1_exp0..src1_expL, ...].
            # GroupedLinear needs [all_exp0, all_exp1, ...].
            sort_indices, unsort_indices = _build_expert_sort_indices(recv_counts)

            handle = _AllToAllHandle(
                row_id_map=row_id_map,
                routing_weights=routing_weights,
                unsort_indices=unsort_indices,
                input_split_sizes=input_split_sizes,
                output_split_sizes=output_split_sizes,
            )
            return DispatchOutput(
                expert_input=recv_tokens[sort_indices],
                tokens_per_expert=local_m_splits,
                handle=handle,
            )

        handle = _AllToAllHandle(row_id_map=row_id_map, routing_weights=routing_weights)
        return DispatchOutput(
            expert_input=permuted_hidden,
            tokens_per_expert=m_splits_tensor.tolist(),
            handle=handle,
        )

    def combine(self, expert_output: torch.Tensor, handle: _AllToAllHandle) -> torch.Tensor:
        """Combine expert outputs back to the original token order.

        Args:
            expert_output: Expert output tensor of shape ``[total_recv_tokens, H]``.
            handle: Handle from ``dispatch()`` containing state for the reverse operation.

        Returns:
            Combined output tensor of shape ``[N, H]`` with routing weights applied.
        """
        if self._ep_group is not None:
            assert handle.unsort_indices is not None
            # Unsort back to source-rank-grouped order and reverse all_to_all (differentiable)
            combined = _DifferentiableAllToAll.apply(
                expert_output[handle.unsort_indices],
                handle.input_split_sizes,
                handle.output_split_sizes,
                self._ep_group,
            )
        else:
            combined = expert_output

        # Unpermute and combine with routing weights (keep probs in float32 for numerical stability)
        return transformer_engine.pytorch.moe_unpermute(
            combined,
            handle.row_id_map,
            merging_probs=handle.routing_weights,
            map_type="index",
        )
