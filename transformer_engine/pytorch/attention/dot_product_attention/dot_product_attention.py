# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Attention."""
from contextlib import nullcontext
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings
import logging

import torch

import transformer_engine_torch as tex
from transformer_engine.pytorch.utils import get_cudnn_version
from transformer_engine.pytorch.fp8 import get_fp8_te_dtype
from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
from transformer_engine.pytorch.export import is_in_onnx_export_mode
from transformer_engine.pytorch.constants import (
    AttnMaskTypes,
    AttnTypes,
    dist_group_type,
)
from transformer_engine.pytorch.distributed import (
    get_distributed_world_size,
    checkpoint,
    set_all_rng_states,
    CudaRNGStatesTracker,
    graph_safe_rng_available,
)
from transformer_engine.pytorch.jit import no_torch_dynamo
from transformer_engine.pytorch.graph import is_graph_capturing
from transformer_engine.pytorch.attention.inference import InferenceParams

# Import attention utils
import transformer_engine.pytorch.attention.dot_product_attention.utils as dpa_utils
from transformer_engine.pytorch.attention.dot_product_attention.utils import (
    AttentionLogging as attn_log,
)

from transformer_engine.pytorch.attention.dot_product_attention.backends import (
    UnfusedDotProductAttention,
    FusedAttention,
    FlashAttention,
)


# Setup Attention Logging
attn_log.setup_logging()

# Global vars for available attention backends and ALiBi cache
_attention_backends = {
    "attention_params": None,
    "use_flash_attention": None,
    "flash_attention_backend": None,
    "use_fused_attention": None,
    "fused_attention_backend": None,
    "use_unfused_attention": None,
    "backend_selection_requires_update": False,
}

_alibi_cache = {
    "_num_heads": None,
    "_alibi_slopes": None,
    "_max_seqlen_q": None,
    "_max_seqlen_kv": None,
    "_bottom_right_alignment": True,
    "_alibi_bias": None,
    "_alibi_slopes_require_update": False,
    "_alibi_bias_require_update": False,
}

__all__ = ["DotProductAttention"]


class DotProductAttention(TransformerEngineBaseModule):
    """Allows the model to jointly attend to information from different
    representation subspaces as described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    .. note::

        Argument :attr:`attention_mask` in the `forward` call is only used when
        :attr:`attn_mask_type` includes '"padding"' or `"arbitrary"`.

    .. warning::

        FlashAttention uses a non-deterministic algorithm for optimal performance. To observe
        deterministic behavior at the cost of performance, use FlashAttention version >= `2.4.1`
        and set the environment variable :attr:`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`. In order
        to disable`flash-attn` entirely, set :attr:`NVTE_FLASH_ATTN=0`.

    .. note::

        Transformer Engine stores the FP8 metadata under a `._extra_state` key when checkpointing.
        As the FP8 attention support expands from one backend to multiple backends, the location
        of that key has also shifted (see `FP8 checkpoint compatibility <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/faq.html#fp8-checkpoint-compatibility>`_).


    Parameters
    ----------
    num_attention_heads : int
                         number of attention heads in the transformer layer.
    kv_channels : Union[int, Tuple[int, int]]
                the head size in key and value tensors. If the same, :attr:`kv_channels` can be
                an integer; if not, :attr:`kv_channels` should be a tuple of two integers.
    num_gqa_groups : Optional[int] = None
                    number of GQA groups in the transformer layer.
                    Grouped Query Attention is described in
                    `this paper <https://arxiv.org/pdf/2305.13245.pdf>`_.
                    This only affects the keys and values, not the queries.
                    GQA-1 is equivalent to Multi-Query Attention
                    (`MQA <https://arxiv.org/pdf/1911.02150.pdf>`_), while GQA-H
                    is equivalent to MHA, i.e. `num_gqa_groups = num_attention_heads`.
    attention_dropout: float, default = 0.0
                      dropout probability for the dropout op during multi-head attention.
    attn_mask_type: str, default = `causal`
                   type of attention mask passed into softmax operation, options are "`no_mask`",
                   "`padding`", "`causal`", "`padding,causal`", "`causal,padding`",
                   "`padding_causal`", "`causal_bottom_right`", "`padding_causal_bottom_right`", and
                   "`arbitrary`", where "`padding,causal`", "`causal,padding`" and "`padding_causal`"
                   are equivalent. This arg can be overridden by :attr:`attn_mask_type` in the
                   `forward` method. It is useful for cases involving compilation/tracing, e.g.
                   ONNX export, and the forward arg is useful for dynamically changing mask types,
                   e.g. a different mask for training and inference.
                   1. For "`no_mask`", no attention mask is applied.
                   2. For "`causal`", "`causal_bottom_right`", or the causal mask in
                   "`padding_causal`" and "`padding_causal_bottom_right`", Transformer Engine
                   calculates and applies an upper triangular mask to the softmax input.
                   No user input is needed. Causal masks without the "`bottom_right`" appendix align
                   the diagonal line to the top left corner of the softmax matrix. With
                   "`bottom_right`", the causal mask is aligned to the bottom right corner, which is
                   often used in inference/KV caching.
                   3. For "`padding`", or the padding mask in "`padding_causal`" and
                   "`padding_causal_bottom_right`", users need to provide the locations of padded
                   tokens, either via :attr:`cu_seqlens_q` and :attr:`cu_seqlens_kv` (both in shape
                   [batch_size + 1]), or via :attr:`attention_mask` (one tensor for self-attention
                   in shape [batch_size, 1, 1, max_seqlen_q], or two tensors in a tuple for
                   cross-attention in shapes [batch_size, 1, 1, max_seqlen_q] and
                   [batch_size, 1, 1, max_seqlen_kv]).
                   4. For "`arbitrary`", users need to provide a mask that is broadcastable to
                   the shape of softmax input [batch_size, num_heads, max_seqlen_q, max_seqlen_kv].
    window_size: Optional[Tuple[int, int]], default = `None`
                sliding window size for local attention, where query at position i attends to keys
                in [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q
                + window_size[1]] inclusive. Special cases (-1, -1) and (-1, 0) mean no sliding
                window and causal mask specifically. Both `causal` and `causal_bottom_right` masks
                map to `window_size = (-1, 0)` and Transformer Engine distinguishes them based on
                `attn_mask_type`. Similar to :attr:`attn_mask_type`, `window_size` can
                be overridden by :attr:`window_size` in `forward` as well.
    attention_type: str, default = `self`
                   type of attention, either "`self`" and "`cross`".
    layer_number: int, default = `None`
                 layer number of the current `DotProductAttention` when multiple such modules
                 are concatenated, for instance in consecutive transformer blocks.
    qkv_format: str, default = `sbhd`
               dimension format for `query_layer`, `key_layer` and `value_layer`,
               {`sbhd`, `bshd`, `thd`}. `s` stands for the sequence length, `b` batch size,
               `h` the number of heads, `d` head size, and `t` the total number of tokens
               in a batch, with `t = sum(s_i), for i = 0...b-1`. `sbhd` and `bshd` formats
               are used for when sequences in a batch are of equal length or padded to
               equal length, and the `thd` format is used for when sequences in a batch
               have different lengths. Please note that these formats do not reflect how
               tensors `query_layer`, `key_layer`, `value_layer` are laid out in memory.
               For that, please use `get_qkv_layout` to gain the layout information.
    softmax_scale: Optional[float], default = `None`
                softmax scale for the attention scores. If `None`, defaults to
                `1.0/math.sqrt(kv_channels if isinstance(kv_channels, int) else kv_channels[0])`.

    Parallelism parameters
    ----------------------
    sequence_parallel : bool, default = `False`
                       if set to `True`, uses sequence parallelism.
    tp_size : int, default = 1
             tensor parallel world size.
    tp_group : ProcessGroup, default = `None`
              tensor parallel process group.
    cp_group : Union[ProcessGroup, List[ProcessGroup]], default = `None`
              context parallel process group.
              ProcessGroup is for cp_comm_type of "p2p", "all_gather", and "a2a".
              List[ProcessGroup] is for cp_comm_type of "a2a+p2p", where cp_group[0]
              and cp_group[1] are for a2a and p2p communications respectively.
    cp_global_ranks : list of global rank IDs, default = `None`
                     global rank IDs of GPUs that are in cp_group.
    cp_stream : CUDA stream, default = `None`
               context parallelism splits flash attention into multiple steps for
               compute and communication overlapping. To address the wave quantization
               issue of each split step, we add an additional CUDA stream so that we
               can overlap two flash attention kernels.
    cp_comm_type : str, default = `p2p`
                  inter-gpu communication type for context parallelism.
                  Can be "p2p" or "all_gather" or "a2a" or "a2a+p2p".
                  "p2p": Exchange KV chunks with P2P communications in ring topology.
                         P2P is async and can be overlapped with attention compute.
                  "all_gather": All-gather to get full sequence of KV before attention.
                                The all-gather is not async, and cannot be overlapped.
                  "a2a": Like DeepSpeed Ulysses, scatter attention heads across the CP
                         group, and gather to get full sequence of QKV.
                  "a2a+p2p": hierarchical CP implementation. First applying a2a to QKV
                  across each CP sub-group (e.g., via NVLink), then exchanging KV with
                  p2p between sub-groups (e.g., via IBLink).
    """

    def __init__(
        self,
        num_attention_heads: int,
        kv_channels: Union[int, Tuple[int, int]],
        num_gqa_groups: Optional[int] = None,
        attention_dropout: float = 0.0,
        qkv_format: str = "sbhd",
        attn_mask_type: str = "causal",
        window_size: Optional[Tuple[int, int]] = None,
        sequence_parallel: bool = False,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        tp_group: Optional[dist_group_type] = None,
        layer_number: Optional[int] = None,
        attention_type: str = "self",
        cp_group: Optional[Union[dist_group_type, List[dist_group_type]]] = None,
        cp_global_ranks: List[int] = None,
        cp_stream: torch.cuda.Stream = None,
        cp_comm_type: str = "p2p",
        softmax_scale: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.logger = logging.getLogger("DotProductAttention")
        self.logger.setLevel(attn_log._log_level)
        if not self.logger.hasHandlers():
            self.logger.addHandler(attn_log._stream_handler)
        self.qkv_format = qkv_format
        attn_mask_type = attn_mask_type.replace(",", "_")
        if attn_mask_type == "causal_padding":
            attn_mask_type = "padding_causal"
        self.attn_mask_type = attn_mask_type
        self.window_size = dpa_utils.check_set_window_size(attn_mask_type, window_size)
        if tp_group is None:
            self.tp_size = tp_size
            if tp_size == 1:
                self.set_tensor_parallel_group(tp_group)
        else:
            self.tp_size = get_distributed_world_size(tp_group)
            self.set_tensor_parallel_group(tp_group)
        self.get_rng_state_tracker = get_rng_state_tracker
        self.num_attention_heads = num_attention_heads
        self.layer_number = 1 if layer_number is None else layer_number
        self.cp_group = cp_group
        self.cp_global_ranks = cp_global_ranks
        self.cp_stream = cp_stream
        self.cp_comm_type = cp_comm_type

        self.hidden_size_per_attention_head_k = (
            kv_channels if isinstance(kv_channels, int) else kv_channels[0]
        )
        self.hidden_size_per_attention_head_v = (
            kv_channels if isinstance(kv_channels, int) else kv_channels[1]
        )

        self.num_gqa_groups = num_attention_heads if num_gqa_groups is None else num_gqa_groups
        self.num_gqa_groups_per_partition = int(self.num_gqa_groups // self.tp_size)

        assert (
            num_attention_heads % self.num_gqa_groups == 0
        ), "The number of attention heads must be divisible by the number of GQA groups!"

        self.rng_states_tracker = None
        if sequence_parallel or get_rng_state_tracker is None:
            attention_dropout_ctx = nullcontext
        else:
            self.rng_states_tracker = get_rng_state_tracker()
            set_all_rng_states(self.rng_states_tracker.get_states())
            attention_dropout_ctx = self.rng_states_tracker.fork

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(
                kv_channels if isinstance(kv_channels, int) else kv_channels[0]
            )

        self.deterministic = (
            not bool(int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")))
            or torch.are_deterministic_algorithms_enabled()
        )
        # To use the workspace optimization path for determinism, please
        # set NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT=1 for cuDNN >=8.9.5 and <9.0.0,
        # and set NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 for cuDNN >=9.0.0.
        cudnn_version = get_cudnn_version()
        if (8, 9, 5) <= cudnn_version < (9, 0, 0):
            if self.deterministic:
                os.environ["NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT"] = "1"

            # CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT
            # - unset:       enables workspace optimization when required workspace is <= 256MB
            #                or when bias gradient needs to be computed
            # - n:           enables workspace optimization when required workspace is <= n bytes
            # - -1:          enables workspace optimization always
            # - 0:           disables workspace optimization always
            if "NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT" in os.environ:
                if os.environ["NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT"] == "0":
                    os.environ["CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT"] = "0"
                if os.environ["NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT"] == "1":
                    os.environ["CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT"] = "-1"

        assert attention_type in AttnTypes, f"attention_type {attention_type} not supported"

        self.attention_type = attention_type
        self.attention_dropout = attention_dropout

        attn_kwargs = {
            "attention_dropout": attention_dropout,
            "attention_dropout_ctx": attention_dropout_ctx,
        }

        self.flash_attention = FlashAttention(
            softmax_scale,
            attention_type=attention_type,
            layer_number=layer_number,
            deterministic=self.deterministic,
            **attn_kwargs,
        )

        # Instantiating three types since use of flash-attn and FusedAttention
        # might be ruled out due to forward inputs.
        self.fused_attention = FusedAttention(
            softmax_scale,
            attention_type=attention_type,
            layer_number=layer_number,
            deterministic=self.deterministic,
            **attn_kwargs,
        )

        self.unfused_attention = UnfusedDotProductAttention(
            softmax_scale,
            attention_type=attention_type,
            **attn_kwargs,
            layer_number=layer_number,
        )

        def remove_extra_states_check(self, incompatible_keys):  # pylint: disable=unused-argument
            """
            Temporarily remove core_attention._extra_state as a missing key
            when loading older Transformer Engine checkpoints. Will phase out
            this hook in Transformer Engine 2.0.
            """
            for key in incompatible_keys.missing_keys:
                if "core_attention._extra_state" in key:
                    incompatible_keys.missing_keys.remove(key)

        self.register_load_state_dict_post_hook(remove_extra_states_check)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """
        This function helps to load Transformer Engine 1.6 and 1.7 checkpoints, where FP8 attention
        metadata is stored under the `core_attention.fused_attention._extra_state` key and not the
        `core_attention._extra_state` key. Please see `FP8 checkpoint compatibility
        <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/faq.html#fp8-checkpoint-compatibility>`_ for more details.
        """
        fused_attn_key = False
        dot_product_attn_key = False
        for k in state_dict.keys():
            if "core_attention.fused_attention._extra_state" in k:
                fused_attn_key = True
            if "core_attention._extra_state" in k:
                dot_product_attn_key = True
        if fused_attn_key and not dot_product_attn_key:
            prefix = prefix + "fused_attention."
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def _checkpointed_attention_forward(
        self,
        attention_func: Callable,
        *forward_args: Tuple[torch.Tensor, ...],
        **forward_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """Forward method with activation checkpointing."""

        def custom_forward(*input_args, **input_kwargs):
            return attention_func(*input_args, **input_kwargs)

        hidden_states = checkpoint(
            custom_forward,
            distribute_saved_activations=False,
            get_rng_state_tracker=self.get_rng_state_tracker,
            tp_group=self.tp_group,
            *forward_args,
            **forward_kwargs,
        )

        return hidden_states

    def set_context_parallel_group(
        self,
        cp_group: Union[dist_group_type, List[dist_group_type], None],
        cp_global_ranks: List[int],
        cp_stream: torch.cuda.Stream,
        cp_comm_type: str = "p2p",
    ) -> None:
        """
        Set the context parallel attributes for the given
        module before executing the forward pass.

        Parameters
        ----------
        cp_group : Union[ProcessGroup, List[ProcessGroup]]
                  context parallel process group.
                  ProcessGroup is for cp_comm_type of "p2p", "all_gather", and "a2a".
                  List[ProcessGroup] is for cp_comm_type of "a2a+p2p", where cp_group[0]
                  and cp_group[1] are for a2a and p2p communications respectively.
        cp_global_ranks : List[int]
                         list of global ranks in the context group.
        cp_stream : torch.cuda.Stream
                   cuda stream for context parallel execution.
        cp_comm_type : str, default = `p2p`
                      inter-gpu communication type for context parallelism.
                      Can be "p2p" or "all_gather" or "a2a" or "a2a+p2p".
                      "p2p": Exchange KV chunks with P2P communications in ring topology.
                             P2P is async and can be overlapped with attention compute.
                      "all_gather": All-gather to get full sequence of KV before attention.
                                    The all-gather is not async, and cannot be overlapped.
                      "a2a": Like DeepSpeed Ulysses, scatter attention heads across the CP
                             group, and gather to get full sequence of QKV.
                      "a2a+p2p": hierarchical CP implementation. First applying a2a to QKV
                      across each CP sub-group (e.g., via NVLink), then exchanging KV with
                      p2p between sub-groups (e.g., via IBLink).
        """
        self.cp_group = cp_group
        self.cp_global_ranks = cp_global_ranks
        self.cp_stream = cp_stream
        self.cp_comm_type = cp_comm_type

    @no_torch_dynamo(recursive=False)
    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
        qkv_format: str = None,
        cu_seqlens_q: torch.Tensor = None,
        cu_seqlens_kv: torch.Tensor = None,
        cu_seqlens_q_padded: torch.Tensor = None,
        cu_seqlens_kv_padded: torch.Tensor = None,
        max_seqlen_q: int = None,
        max_seqlen_kv: int = None,
        attn_mask_type: Optional[str] = None,
        window_size: Optional[Tuple[int, int]] = None,
        checkpoint_core_attention: bool = False,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[torch.Tensor] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        fast_zero_fill: bool = True,
        inference_params: Optional[InferenceParams] = None,
        pad_between_seqs: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Dot Product Attention Layer.

        .. note::

            Argument :attr:`attention_mask` is only used when :attr:`attn_mask_type`
            includes '"padding"' or `"arbitrary"`.

        .. note::

            DotProductAttention supports three backends: 1) FlashAttention which calls
            HazyResearch/Dao-AILab's `flash-attn <https://arxiv.org/pdf/2305.13245.pdf>`_
            PyTorch API, 2) FusedAttention which has multiple fused attention implementations
            based on `cuDNN Graph API
            <https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#op-fusion>`_
            (see :attr:`FusedAttention` for more details on FusedAttention backends), and 3)
            UnfusedDotProductAttention which is the native PyTorch implementation
            with fused scaled masked softmax.

        .. note::

            Users can use environment variables :attr:`NVTE_FLASH_ATTN`, :attr:`NVTE_FUSED_ATTN`,
            and :attr:`NVTE_FUSED_ATTN_BACKEND` to control which DotProductAttention backend,
            and FusedAttention backend if applicable, to use. Transformer Engine prioritizes
            FlashAttention over FusedAttention and over UnfusedDotProductAttention.
            If FusedAttention is being used, users can also choose to switch to flash-attn's
            implementation for backward by setting :attr:`NVTE_FUSED_ATTN_USE_FAv2_BWD=1`
            (default: 0), because of the performance differences between various versions of
            flash-attn and FusedAttention. Further, :attr:`NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT`
            can be used to enable (:attr:`1`) or disable (:attr:`0`) the workspace related
            optimizations in FusedAttention. When unset, Transformer Engine determines the code path
            based on its internal logic. These optimizations trade memory for performance
            and should be used with care.

        .. note::
            .. _cu_seqlens note:

            When training data has variable sequence lengths, users have two options.

            1. Manipulate the data and pad all sequences to the same length. Use
               :attr:`qkv_format` = {"bshd", "sbhd"} and
               :attr:`attn_mask_type` = {"padding", "padding_causal", "padding_causal_bottom_right"}.
               Pass in :attr:`cu_seqlens_q` and :attr:`cu_seqlens_kv`, or :attr:`attention_mask`
               (which will be converted to :attr:`cu_seqlens_q` and :attr:`cu_seqlens_kv`), to provide
               the real sequence length information. For example, a batch of 3 sequences
               [a a a b b c c c c] can be padded to [a a a PAD b b PAD PAD c c c c], and the cumulative
               sequence length tensors would be
               :attr:`cu_seqlens_q` = :attr:`cu_seqlens_kv` = [0, 3, 5, 9] for self-attention.

            2. Do not perform padding on training data. Use :attr:`qkv_format` = "thd" and
               :attr:`attn_mask_type` = {"padding", "padding_causal", "padding_causal_bottom_right"}.
               Pass in :attr:`cu_seqlens_q` and :attr:`cu_seqlens_kv`, or :attr:`attention_mask`,
               as in option 1. For example, a batch of 3 sequences [a a a b b c c c c] can be processed
               without any padding, and the sequence length tensors would be
               :attr:`cu_seqlens_q` = :attr:`cu_seqlens_kv` = [0, 3, 5, 9] for self-attention.

               In certain use cases, a varying number of identifier tokens are inserted between
               sequences. These tokens do not participate in the attention calculation.
               :attr:`cu_seqlens_q_padded` and :attr:`cu_seqlens_kv_padded` must be specified
               in such cases to correctly identify the start and end of each sequence in a batch.
               For example, a batch of 3 sequences [a a a 1 b b 2 2 c c c c 3] would have
               :attr:`cu_seqlens_q` = :attr:`cu_seqlens_kv` = [0, 3, 5, 9], and
               :attr:`cu_seqlens_q_padded` = :attr:`cu_seqlens_kv_padded` = [0, 4, 8, 13]
               for self-attention.

        .. note::
            .. _max_seqlen note:

            When :attr:`qkv_format` = {"bshd", "sbhd"}, sequences are of equal length in a batch.
            :attr:`max_seqlen_q` and :attr:`max_seqlen_kv` should be the same as the "s" dimension of
            :attr:`query_layer` and :attr:`key_layer` tensors. When unset, Transformer Engine will
            infer them as such.

            When :attr:`qkv_format` = "thd", sequences have varying lengths. :attr:`max_seqlen_q` and
            :attr:`max_seqlen_kv` should be the maximum query and key/value sequence length in a batch.
            When unset, Transformer Engine deduces them from :attr:`cu_seqlens_q` and :attr:`cu_seqlens_kv`.
            This deduction costs a small kernel and some CPU-GPU synchronization, and to avoid this
            overhead, users are recommended to obtain the maximum sequence lengths from the data loaders
            and pass them in.

            - As the maximum sequence lengths, batch size, and number of tokens change from batch to batch,
              dynamic shapes need to be supported for tensor construction. FlashAttention and
              UnfusedDotProductAttention naturally do so, while FusedAttention requires parameters to be static
              to create graphs before performance heuristics analysis. To reduce the number of graphs created
              per run, Transformer Engine 1.13+ quantizes relevant parameters: for cuDNN < 9.6, {batch size,
              :attr:`max_seqlen_q`, :attr:`max_seqlen_kv`}, and for cuDNN >= 9.6, {"t" dimension of
              :attr:`query_layer`, "t" dimension of :attr:`key_layer`}.

        Parameters
        ----------
        query_layer : torch.Tensor
                     Query tensor.
        key_layer : torch.Tensor
                   Key tensor.
        value_layer : torch.Tensor
                     Value tensor.
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]],
             default = `None`. Boolean tensor(s) used to mask out attention softmax input.
             It should be `None` for causal masks and "`no_mask`". For padding masks, it should be
             a single tensor of [batch_size, 1, 1, seqlen_q] for self-attention, and a tuple of
             two tensors in shapes [batch_size, 1, 1, seqlen_q] and [batch_size, 1, 1, seqlen_kv]
             for cross-attention. For "`arbitrary`" mask, it should be in a shape broadcastable
             to [batch_size, num_heads, max_seqlen_q, max_seqlen_kv]. A `True` value means
             the corresponding position is masked out and a `False` means that position
             is allowed to participate in attention.
        qkv_format: str, default = `None`
                   If provided, overrides :attr:`qkv_format` from initialization.
        cu_seqlens_q: Optional[torch.Tensor], default = `None`
                   Cumulative sum of sequence lengths (without offset) in a batch for `query_layer`,
                   with shape [batch_size + 1] and dtype torch.int32.
                   See :ref:`note<cu_seqlens note>` for more details.
        cu_seqlens_kv: Optional[torch.Tensor], default = `None`
                   Cumulative sum of sequence lengths (without offset) in a batch for `key_layer`
                   and `value_layer`, with shape [batch_size + 1] and dtype torch.int32.
                   See :ref:`note<cu_seqlens note>` for more details.
        cu_seqlens_q_padded: Optional[torch.Tensor], default = `None`
                   Cumulative sum of sequence lengths (with offset) in a batch for
                   `query_layer`, with shape [batch_size + 1] and dtype torch.int32.
                   When there is no padding between sequences in a batch,
                   `cu_seqlens_q_padded = cu_seqlens_q`.
                   See :ref:`note<cu_seqlens note>` for more details.
        cu_seqlens_kv_padded: Optional[torch.Tensor], default = `None`
                   Cumulative sum of sequence lengths (with offset) in a batch for `key_layer`
                   and `value_layer`, with shape [batch_size + 1] and dtype torch.int32.
                   When there is no padding between sequences in a batch,
                   `cu_seqlens_kv_padded = cu_seqlens_kv`.
                   See :ref:`note<cu_seqlens note>` for more details.
        max_seqlen_q: Optional[int], default = `None`
                      Maximum sequence length in `query_layer`.
                      See :ref:`note<max_seqlen note>` for more details.
        max_seqlen_kv: Optional[int], default = `None`
                       Maximum sequence length in `key_layer` and `value_layer`.
                       See :ref:`note<max_seqlen note>` for more details.
        attn_mask_type: {'no_mask', 'padding', 'causal', 'padding,causal', 'causal,padding',
                       'padding_causal', 'causal_bottom_right', 'padding_causal_bottom_right',
                       'arbitrary'}, default = `None`. Type of attention mask passed into
                       softmax operation. 'padding,causal', 'causal,padding' and 'padding_causal'
                       are equivalent. By default, causal masks are aligned to the top left corner
                       of the softmax matrix. When "`bottom_right`" is specified in the mask type,
                       causal masks are aligned to the bottom right corner.
        window_size: Optional[Tuple[int, int]], default = `None`
                    Sliding window size for local attention.
        checkpoint_core_attention : bool, default = `False`
                                   If true, forward activations for attention are recomputed
                                   during the backward pass in order to save memory that would
                                   otherwise be occupied to store the forward activations until
                                   backprop.
        core_attention_bias_type: str, default = `no_bias`
                    Bias type, {`no_bias`, `pre_scale_bias`, `post_scale_bias`, `alibi`}
        core_attention_bias: Optional[torch.Tensor], default = `None`
                    Bias tensor for Q * K.T, shape [1, num_head, max_seqlen_q, max_seqlen_kv].
                    It should be 'None' for 'no_bias' and 'alibi' bias types.
        alibi_slopes: Optional[torch.Tensor], default = `None`
                     ALiBi slopes in FP32 and shape [nheads] or [batch_size, nheads].
                     It adds a bias of (-alibi_slope * (i + seqlen_k - seqlen_q - j))
                     to the attention score of query i and key j.
        fast_zero_fill: bool, default = `True`
                    Whether to use the fast path to set output tensors to 0 or not.
        inference_params: Optional[InferenceParams], default = `None`
            Optimizes execution performance during inference by caching Keys and Values of the
            current decoding iteration. These cached values are appended to the K and V values
            computed in previous iterations, eliminating the need to recalculate them for the
            entire sequence.
            Initialization of `inference_params` is required prior to use to ensure sufficient
            memory allocation.
            Adjustments of the sequence_len_offset should be done after a complete forward pass.
            If rotary positional embeddings (RoPE) are utilized, they must be prepared beforehand.
            Supports "sbhd" and "bshd" layouts, with the "sbhd" layout being more efficient.
        pad_between_seqs: Optional[bool], default = `None`
            If None, inferred from qkv_format, cu_seqlens and cu_seqlens_padded.
            If true, there are padding tokens between individual sequences in a packed batch.
        """

        with torch.cuda.device(query_layer.device), self.prepare_forward(
            query_layer,
            num_gemms=3,
            allow_non_contiguous=True,
        ) as query_layer:
            # checks for RNG
            if self.rng_states_tracker is not None and is_graph_capturing():
                assert isinstance(
                    self.rng_states_tracker, CudaRNGStatesTracker
                ), "Unsupported RNG states tracker."
                assert (
                    graph_safe_rng_available()
                ), "Upgrade PyTorch version to get RNG manipulation support for cuda graph capture."

            # checks for FP8
            if self.fp8:
                if self.fp8_meta["recipe"].fp8_mha:
                    if not self.fp8_meta["recipe"].fp8_dpa:
                        self.fp8_meta["recipe"].fp8_dpa = True
                        self.logger.warning(
                            """Forcing fp8_meta["recipe"].fp8_dpa=True due to """
                            """fp8_meta["recipe"].fp8_mha=True"""
                        )
            if self.fp8 and self.fp8_meta["recipe"].fp8_dpa:
                forward_dtype = get_fp8_te_dtype(self.fp8_meta["recipe"], fprop_tensor=True)
                backward_dtype = get_fp8_te_dtype(self.fp8_meta["recipe"], fprop_tensor=False)
                assert forward_dtype in [
                    tex.DType.kFloat8E4M3,
                    tex.DType.kFloat8E5M2,
                ] and backward_dtype in [
                    tex.DType.kFloat8E4M3,
                    tex.DType.kFloat8E5M2,
                ], """DotProductAttention only supports "E4M3" and "E5M2" FP8 data types."""

            # checks for q/k/v shapes
            assert (
                query_layer.is_cuda and key_layer.is_cuda and value_layer.is_cuda
            ), "DotProductAttention only supports CUDA tensors."
            assert (
                query_layer.dtype == key_layer.dtype and query_layer.dtype == value_layer.dtype
            ), "Queries, keys and values must have the same data type!"
            assert (
                key_layer.shape[:-1] == value_layer.shape[:-1]
            ), "Keys and values must have the same batch size, sequence length and number of heads!"
            num_attention_heads = query_layer.shape[-2]
            num_gqa_groups = key_layer.shape[-2]
            assert (
                query_layer.shape[-1] == key_layer.shape[-1]
            ), "Queries and keys must have the same head dimension!"
            head_dim_qk, head_dim_v = query_layer.shape[-1], value_layer.shape[-1]
            assert (
                head_dim_qk == self.hidden_size_per_attention_head_k
            ), f"Keys have head_dim = {head_dim_qk}, "
            "but expected head_dim = {self.hidden_size_per_attention_head_k}!"
            assert (
                head_dim_v == self.hidden_size_per_attention_head_v
            ), f"Values have head_dim = {head_dim_v}, "
            "but expected head_dim = {self.hidden_size_per_attention_head_v}!"
            assert num_gqa_groups == self.num_gqa_groups_per_partition, (
                "Keys and values must have num_gqa_group ="
                f" {self.num_gqa_groups_per_partition} heads! Found {num_gqa_groups}."
            )

            # checks for attention mask
            if attn_mask_type is None:
                attn_mask_type = self.attn_mask_type
            else:
                attn_mask_type = attn_mask_type.replace(",", "_")
                if attn_mask_type == "causal_padding":
                    attn_mask_type = "padding_causal"
            assert (
                attn_mask_type in AttnMaskTypes
            ), f"Attention mask type {attn_mask_type} is not supported!"

            # checks for sliding window
            if window_size is None:
                window_size = self.window_size
            window_size = dpa_utils.check_set_window_size(attn_mask_type, window_size)

            # checks for qkv_format
            if qkv_format is None:
                qkv_format = self.qkv_format
            assert qkv_format in [
                "sbhd",
                "bshd",
                "thd",
            ], "DotProductAttention only supports qkv_format = {'sbhd', 'bshd', 'thd'}!"
            batch_size = None
            if qkv_format in ["sbhd", "bshd"]:
                assert all(
                    len(x.shape) == 4 for x in (query_layer, key_layer, value_layer)
                ), f"Queries, keys and values must be 4D tensors when {qkv_format=}!"
                if qkv_format == "sbhd":
                    batch_size = query_layer.shape[1]
                    max_seqlen_q = query_layer.shape[0] if max_seqlen_q is None else max_seqlen_q
                    max_seqlen_kv = key_layer.shape[0] if max_seqlen_kv is None else max_seqlen_kv
                else:
                    batch_size = query_layer.shape[0]
                    max_seqlen_q = query_layer.shape[1] if max_seqlen_q is None else max_seqlen_q
                    max_seqlen_kv = key_layer.shape[1] if max_seqlen_kv is None else max_seqlen_kv
            if qkv_format == "thd":
                assert all(
                    len(x.shape) == 3 for x in (query_layer, key_layer, value_layer)
                ), "Queries, keys and values must be 3D tensors when qkv_format = thd!"
                assert (
                    "padding" in attn_mask_type
                ), "Attention mask type must be padding or padding_causal for qkv_format=thd!"
                assert (
                    cu_seqlens_q is not None and cu_seqlens_kv is not None
                ), "cu_seqlens_q and cu_seqlens_kv can not be None when qkv_format = thd!"
                assert (
                    cu_seqlens_q.shape == cu_seqlens_kv.shape
                    and len(cu_seqlens_q.shape) == 1
                    and len(cu_seqlens_kv.shape) == 1
                ), "cu_seqlens_q and cu_seqlens_q must both have shape [batch_size + 1]!"
                assert (
                    cu_seqlens_q.dtype == torch.int32 and cu_seqlens_kv.dtype == torch.int32
                ), "cu_seqlens_q and cu_seqlens_q must both be in dtype torch.int32!"
                batch_size = len(cu_seqlens_q) - 1
                if max_seqlen_q is None:
                    if cu_seqlens_q_padded is not None:
                        seqlens_q = cu_seqlens_q_padded[1:] - cu_seqlens_q_padded[:-1]
                    else:
                        seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
                    max_seqlen_q = int((seqlens_q.max().item() + 63) // 64 * 64)
                if max_seqlen_kv is None:
                    if cu_seqlens_kv_padded is not None:
                        seqlens_kv = cu_seqlens_kv_padded[1:] - cu_seqlens_kv_padded[:-1]
                    else:
                        seqlens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
                    max_seqlen_kv = int((seqlens_kv.max().item() + 63) // 64 * 64)

            # update KV cache and retrieve saved tokens from cache for inference
            if inference_params is not None:
                assert self.layer_number is not None, "Layer number must be set!"

                # convert top-left causal to bottom-right causal due to KV caching
                # users can still use the same attention mask for inference as for training
                assert "padding" in attn_mask_type, "KV caching requires padding mask!"
                if attn_mask_type == "padding_causal":
                    attn_mask_type = attn_mask_type + "_bottom_right"

                self.attention_type = "cross"
                self.flash_attention.attention_type = self.attention_type
                self.fused_attention.attention_type = self.attention_type
                self.unfused_attention.attention_type = self.attention_type

                query_layer, key_layer, value_layer = [
                    x.contiguous() if not x.is_contiguous() else x
                    for x in [query_layer, key_layer, value_layer]
                ]

                # get full K/V tensors from cache and adjust cu_seqlens, qkv_format based on the cache
                (
                    key_layer,
                    value_layer,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_kv,
                    qkv_format,
                ) = inference_params.step(
                    self.layer_number,
                    key_layer,
                    value_layer,
                    qkv_format,
                )
                cu_seqlens_q_padded = None
                cu_seqlens_kv_padded = None

            # get qkv's memory layout
            if all(isinstance(x, Float8Tensor) for x in [query_layer, key_layer, value_layer]):
                (
                    qkv_layout,
                    query_layer._data,
                    key_layer._data,
                    value_layer._data,
                    q_format,
                    kv_format,
                ) = dpa_utils.get_qkv_layout(
                    query_layer._data,
                    key_layer._data,
                    value_layer._data,
                    qkv_format=qkv_format,
                    inference_params=inference_params,
                )
            else:
                (
                    qkv_layout,
                    query_layer,
                    key_layer,
                    value_layer,
                    q_format,
                    kv_format,
                ) = dpa_utils.get_qkv_layout(
                    query_layer,
                    key_layer,
                    value_layer,
                    qkv_format=qkv_format,
                    inference_params=inference_params,
                )

            # adjust max_seqlen and cu_seqlens for CP
            cp_size = 1
            if isinstance(self.cp_group, dist_group_type):
                cp_size = get_distributed_world_size(self.cp_group)
            elif isinstance(self.cp_group, list):
                for group in self.cp_group:
                    cp_size *= get_distributed_world_size(group)
            context_parallel = cp_size > 1
            if q_format in ["sbhd", "bshd"]:
                max_seqlen_q *= cp_size
                if cu_seqlens_q is None:
                    if "padding" in attn_mask_type:
                        assert (
                            attention_mask is not None
                        ), "Please provide attention_mask for padding!"
                        if self.attention_type == "self":
                            cu_seqlens_q = dpa_utils.get_cu_seqlens(attention_mask)
                        else:
                            cu_seqlens_q = dpa_utils.get_cu_seqlens(attention_mask[0])
                    else:
                        cu_seqlens_q = dpa_utils.get_full_cu_seqlens(
                            batch_size,
                            max_seqlen_q,
                            query_layer.device,
                        )
            if kv_format in ["sbhd", "bshd"]:
                max_seqlen_kv *= cp_size
                if cu_seqlens_kv is None:
                    if "padding" in attn_mask_type:
                        assert (
                            attention_mask is not None
                        ), "Please provide attention_mask for padding!"
                        if self.attention_type == "self":
                            cu_seqlens_kv = dpa_utils.get_cu_seqlens(attention_mask)
                        else:
                            cu_seqlens_kv = dpa_utils.get_cu_seqlens(attention_mask[1])
                    else:
                        cu_seqlens_kv = dpa_utils.get_full_cu_seqlens(
                            batch_size,
                            max_seqlen_kv,
                            key_layer.device,
                        )

            # set ALiBi attributes
            global _alibi_cache
            if alibi_slopes is not None:
                assert (
                    core_attention_bias_type == "alibi"
                ), "core_attention_bias_type must be alibi in order to use alibi_slopes!"
                if self.layer_number == 1:
                    _alibi_cache["_alibi_slopes_require_update"] = True
                    _alibi_cache["_alibi_bias_require_update"] = True
            bottom_right_alignment = (attn_mask_type not in ["causal", "padding_causal"],)
            if core_attention_bias_type == "alibi":
                assert (
                    core_attention_bias is None
                ), "core_attention_bias must be None when core_attention_bias_type is alibi!"
                if (
                    _alibi_cache["_num_heads"] != query_layer.shape[-2]
                    or _alibi_cache["_max_seqlen_q"] != max_seqlen_q
                    or _alibi_cache["_max_seqlen_kv"] != max_seqlen_kv
                    or _alibi_cache["_bottom_right_alignment"] != bottom_right_alignment
                    or _alibi_cache["_alibi_slopes"] is None
                ):
                    _alibi_cache["_alibi_slopes_require_update"] = True
                    _alibi_cache["_alibi_bias_require_update"] = True

            # detect bias shape
            core_attention_bias_shape = None
            if core_attention_bias is not None:
                if (
                    core_attention_bias.shape[0] == batch_size
                    and core_attention_bias.shape[1] == query_layer.shape[-2]
                ):
                    core_attention_bias_shape = "bhss"
                elif (
                    core_attention_bias.shape[0] == 1
                    and core_attention_bias.shape[1] == query_layer.shape[-2]
                ):
                    core_attention_bias_shape = "1hss"
                elif (
                    core_attention_bias.shape[0] == batch_size and core_attention_bias.shape[1] == 1
                ):
                    core_attention_bias_shape = "b1ss"
                elif core_attention_bias.shape[0] == 1 and core_attention_bias.shape[1] == 1:
                    core_attention_bias_shape = "11ss"
                else:
                    assert (
                        False
                    ), "core_attention_bias must be in one of {bhss, 1hss, b1ss, 11ss} shapes"

            if pad_between_seqs is None:
                if qkv_format == "thd":
                    pad_between_seqs = (
                        cu_seqlens_q_padded is not None
                        and not torch.equal(cu_seqlens_q_padded[:-1], cu_seqlens_q[:-1])
                    ) or (
                        cu_seqlens_kv_padded is not None
                        and not torch.equal(cu_seqlens_kv_padded[:-1], cu_seqlens_kv[:-1])
                    )
                else:
                    pad_between_seqs = False

            # gather attention params for get_attention_backend
            attention_params = dpa_utils.AttentionParams(
                qkv_type=type(query_layer),
                qkv_dtype=query_layer.dtype,
                qkv_layout=qkv_layout,
                batch_size=batch_size,
                num_heads=num_attention_heads,
                num_gqa_groups=num_gqa_groups,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                head_dim_qk=head_dim_qk,
                head_dim_v=head_dim_v,
                attn_mask_type=attn_mask_type,
                window_size=window_size,
                alibi_slopes_shape=alibi_slopes.shape if alibi_slopes is not None else None,
                core_attention_bias_type=core_attention_bias_type,
                core_attention_bias_shape=core_attention_bias_shape,
                core_attention_bias_requires_grad=(
                    core_attention_bias.requires_grad if core_attention_bias is not None else False
                ),
                pad_between_seqs=pad_between_seqs,
                attention_dropout=self.attention_dropout,
                context_parallel=context_parallel,
                deterministic=self.deterministic,
                is_training=self.training,
                fp8=self.fp8,
                fp8_meta=self.fp8_meta,
                inference_params=inference_params,
            )
            global _attention_backends
            if is_in_onnx_export_mode():
                # We do not want to call get_attention_backend() in ONNX mode
                # and we want to avoid using any global variables like _attention_backends.
                use_flash_attention = False
                use_fused_attention = False
                use_unfused_attention = True
            else:
                if (
                    _attention_backends["attention_params"] is None
                    or attention_params != _attention_backends["attention_params"]
                ):
                    _attention_backends["attention_params"] = attention_params
                    _attention_backends["backend_selection_requires_update"] = True
                if _attention_backends["backend_selection_requires_update"]:
                    (
                        use_flash_attention,
                        flash_attention_backend,
                        use_fused_attention,
                        fused_attention_backend,
                        use_unfused_attention,
                        _,
                    ) = dpa_utils.get_attention_backend(attention_params)
                    # Set global _attention_backends var using return value
                    # from get_attention_backend()
                    _attention_backends["use_flash_attention"] = use_flash_attention
                    _attention_backends["flash_attention_backend"] = flash_attention_backend
                    _attention_backends["use_fused_attention"] = use_fused_attention
                    _attention_backends["fused_attention_backend"] = fused_attention_backend
                    _attention_backends["use_unfused_attention"] = use_unfused_attention
                    _attention_backends["backend_selection_requires_update"] = False
                    if use_flash_attention:
                        self.logger.info(
                            "Running with FlashAttention backend (version %s)",
                            flash_attention_backend,
                        )
                    elif use_fused_attention:
                        self.logger.info(
                            "Running with FusedAttention backend (sub-backend %s)",
                            int(fused_attention_backend),
                        )
                    elif use_unfused_attention:
                        self.logger.info("Running with UnfusedDotProductAttention backend")
                else:
                    use_flash_attention = _attention_backends["use_flash_attention"]
                    flash_attention_backend = _attention_backends["flash_attention_backend"]
                    use_fused_attention = _attention_backends["use_fused_attention"]
                    fused_attention_backend = _attention_backends["fused_attention_backend"]
                    use_unfused_attention = _attention_backends["use_unfused_attention"]

            # raise exception if no backend is available
            if sum([use_flash_attention, use_fused_attention, use_unfused_attention]) == 0:
                raise ValueError(
                    "No dot product attention backend is available for the provided inputs. Please"
                    " run with NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 to find out the reasons for"
                    " disabling all backends."
                )

            # run attention
            if use_flash_attention:
                if core_attention_bias_type == "alibi":
                    alibi_slopes, _ = dpa_utils.get_alibi(
                        _alibi_cache,
                        query_layer.shape[-2],
                        max_seqlen_q,
                        max_seqlen_kv,
                        alibi_slopes=alibi_slopes,
                    )
                return self.flash_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    attention_mask=attention_mask,
                    qkv_layout=qkv_layout,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_kv,
                    attn_mask_type=attn_mask_type,
                    window_size=window_size,
                    alibi_slopes=alibi_slopes,
                    cp_group=self.cp_group,
                    cp_global_ranks=self.cp_global_ranks,
                    cp_stream=self.cp_stream,
                    cp_comm_type=self.cp_comm_type,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                    fp8=self.fp8 and self.fp8_meta["recipe"].fp8_dpa,
                    fp8_meta=self.fp8_meta,
                    quantizers=self.quantizers,
                    inference_params=inference_params,
                    flash_attention_backend=flash_attention_backend,
                )

            if use_fused_attention:
                fu_core_attention_bias_type = core_attention_bias_type
                fu_core_attention_bias = core_attention_bias
                if core_attention_bias_type == "alibi" and (
                    alibi_slopes is not None or max_seqlen_q != max_seqlen_kv
                ):
                    fu_core_attention_bias_type = "post_scale_bias"
                    _, fu_core_attention_bias = dpa_utils.get_alibi(
                        _alibi_cache,
                        query_layer.shape[-2],
                        max_seqlen_q,
                        max_seqlen_kv,
                        alibi_slopes=alibi_slopes,
                        bias_dtype=query_layer.dtype,
                        bottom_right_alignment=attn_mask_type not in ["causal", "padding_causal"],
                    )
                # checkpoint_core_attention=False
                if checkpoint_core_attention:
                    return self._checkpointed_attention_forward(
                        self.fused_attention,
                        query_layer,
                        key_layer,
                        value_layer,
                        qkv_layout=qkv_layout,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_kv=cu_seqlens_kv,
                        cu_seqlens_q_padded=cu_seqlens_q_padded,
                        cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                        max_seqlen_q=max_seqlen_q,
                        max_seqlen_kv=max_seqlen_kv,
                        attn_mask_type=attn_mask_type,
                        attention_mask=attention_mask,
                        window_size=window_size,
                        fused_attention_backend=fused_attention_backend,
                        core_attention_bias_type=fu_core_attention_bias_type,
                        core_attention_bias=fu_core_attention_bias,
                        fast_zero_fill=fast_zero_fill,
                        cp_group=self.cp_group,
                        cp_global_ranks=self.cp_global_ranks,
                        cp_stream=self.cp_stream,
                        cp_comm_type=self.cp_comm_type,
                        fp8=self.fp8 and self.fp8_meta["recipe"].fp8_dpa,
                        fp8_meta=self.fp8_meta,
                        quantizers=self.quantizers,
                        pad_between_seqs=pad_between_seqs,
                        inference_params=inference_params,
                    )
                return self.fused_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    qkv_layout=qkv_layout,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_kv,
                    cu_seqlens_q_padded=cu_seqlens_q_padded,
                    cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                    attn_mask_type=attn_mask_type,
                    attention_mask=attention_mask,
                    window_size=window_size,
                    fused_attention_backend=fused_attention_backend,
                    core_attention_bias_type=fu_core_attention_bias_type,
                    core_attention_bias=fu_core_attention_bias,
                    fast_zero_fill=fast_zero_fill,
                    cp_group=self.cp_group,
                    cp_global_ranks=self.cp_global_ranks,
                    cp_stream=self.cp_stream,
                    cp_comm_type=self.cp_comm_type,
                    fp8=self.fp8 and self.fp8_meta["recipe"].fp8_dpa,
                    fp8_meta=self.fp8_meta,
                    quantizers=self.quantizers,
                    pad_between_seqs=pad_between_seqs,
                    inference_params=inference_params,
                )

            from transformer_engine.pytorch.cpu_offload import CPUOffloadEnabled

            if CPUOffloadEnabled:
                warnings.warn(
                    "Attention activation Offloading is only implemented"
                    "with Flash Attention and Fused Attention!"
                )

            if use_unfused_attention:
                if checkpoint_core_attention:
                    return self._checkpointed_attention_forward(
                        self.unfused_attention,
                        _alibi_cache,
                        query_layer,
                        key_layer,
                        value_layer,
                        qkv_layout=qkv_layout,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_kv=cu_seqlens_kv,
                        attn_mask_type=attn_mask_type,
                        attention_mask=attention_mask,
                        window_size=window_size,
                        core_attention_bias_type=core_attention_bias_type,
                        core_attention_bias=core_attention_bias,
                        alibi_slopes=alibi_slopes,
                        inference_params=inference_params,
                    )
                return self.unfused_attention(
                    _alibi_cache,
                    query_layer,
                    key_layer,
                    value_layer,
                    qkv_layout=qkv_layout,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_kv,
                    attn_mask_type=attn_mask_type,
                    attention_mask=attention_mask,
                    window_size=window_size,
                    core_attention_bias_type=core_attention_bias_type,
                    core_attention_bias=core_attention_bias,
                    alibi_slopes=alibi_slopes,
                    inference_params=inference_params,
                )
            return None
