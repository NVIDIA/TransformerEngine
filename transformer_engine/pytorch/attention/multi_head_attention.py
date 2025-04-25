# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Multi-head Attention."""
import collections
from typing import Callable, List, Optional, Tuple, Union
import torch

from transformer_engine.debug.pytorch.debug_state import TEDebugState
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
from transformer_engine.pytorch.module import LayerNormLinear, Linear
from transformer_engine.pytorch.utils import (
    SplitAlongDim,
    divide,
    get_default_init_method,
)
from transformer_engine.pytorch.constants import (
    AttnTypes,
    AttnBiasTypes,
    dist_group_type,
)
from transformer_engine.pytorch.distributed import (
    get_distributed_world_size,
    get_distributed_rank,
)

from transformer_engine.pytorch.attention.dot_product_attention import DotProductAttention
from transformer_engine.pytorch.attention.inference import InferenceParams
from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb
from transformer_engine.pytorch.tensor.quantized_tensor import QuantizedTensor


class MultiheadAttention(torch.nn.Module):
    r"""
    Multi-head Attention (MHA), including Query,
    Key, Value and Output projection.

    .. note::

        Argument :attr:`attention_mask` in the `forward` call is only used when
        :attr:`attn_mask_type` includes '"padding"' or `"arbitrary"`.

    Parameters
    ----------
    hidden_size : int
                 size of each input sample.
    num_attention_heads : int
                         number of attention heads in the transformer layer.
    kv_channels: int, default = `None`
                number of key-value channels. defaults to
                :attr:`hidden_size` / :attr:`num_attention_heads` if `None`.
    attention_dropout: float, default = 0.1
                      dropout probability for the dropout op during multi-head attention.
    layernorm_epsilon : float, default = 1e-5
                       a value added to the denominator of layer normalization
                       for numerical stability.
    init_method : Callable, default = `None`
                 used for initializing weights of QKV and FC1 weights in the following way:
                 `init_method(weight)`. When set to `None`, defaults to
                 `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    output_layer_init_method : Callable, default = `None`
                              used for initializing weights of PROJ and FC2 in the following way:
                              `output_layer_init_method(weight)`. When set to `None`, defaults to
                              `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    layer_number: int, default = `None`
                 layer number of the current `TransformerLayer` when multiple such modules are
                 concatenated to form a transformer block.
    attn_mask_type: {'no_mask', 'padding', 'causal', 'padding_causal', 'causal_bottom_right',
                   'padding_causal_bottom_right','arbitrary'},
                   default = `causal`
                   type of attention mask passed into softmax operation. Overridden by
                   :attr:`attn_mask_type` in the `forward` method. The forward
                   arg is useful for dynamically changing mask types, e.g. a different
                   mask for training and inference. The init arg is useful for cases
                   involving compilation/tracing, e.g. ONNX export.
    window_size: Optional[Tuple[int, int]], default = `None`
                sliding window size for local attention, where query at position i attends to keys
                in [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q
                + window_size[1]] inclusive. Special cases (-1, -1) and (-1, 0) mean no sliding
                window and causal mask specifically. Both `causal` and `causal_bottom_right` masks
                map to `window_size = (-1, 0)` and Transformer Engine distinguishes them based on
                `attn_mask_type`. Similar to :attr:`attn_mask_type`, `window_size` can
                be overridden by :attr:`window_size` in `forward` as well.
    num_gqa_groups : int, default = `None`
                         number of GQA groups in the transformer layer.
                         Grouped Query Attention is described in
                         `this paper <https://arxiv.org/pdf/2305.13245.pdf>`_.
                         This only affects the keys and values, not the querys.
                         GQA-1 is equivalent to Multi-Query Attention
                         (`MQA <https://arxiv.org/pdf/1911.02150.pdf>`_), while GQA-H
                         is equivalent to MHA, i.e. `num_gqa_groups = num_attention_heads`.
    return_layernorm_output : bool, default = `False`
                             if set to `True`, output of layernorm is returned from the forward
                             together with the output of the linear transformation.
                             Example use case: residual connection for transformer module is
                             taken post layernorm.
    input_layernorm: bool, default = `False`
                     if set to `True`, layer normalization to the input is applied.
    attention_type: { 'self', 'cross' }, default = 'self'
                   type of attention applied.
    zero_centered_gamma : bool, default = 'False'
                         if set to 'True', gamma parameter in LayerNorm is initialized to 0 and
                         the LayerNorm formula changes to

                         .. math::
                            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \varepsilon}} *
                            (1 + \gamma) + \beta
    normalization : { 'LayerNorm', 'RMSNorm' }, default = 'LayerNorm'
                   type of normalization applied.
    qkv_weight_interleaved : bool, default = `True`
                            if set to `False`, the QKV weight is interpreted as a concatenation of
                            query, key, and value weights along the `0th` dimension. The default
                            interpretation is that the individual `q`, `k`, and `v` weights for each
                            attention head are interleaved. This parameter is set to `False` when
                            using :attr:`fuse_qkv_params=False`.
    bias : bool, default = `True`
          if set to `False`, the transformer layer will not learn any additive biases.
    device : Union[torch.device, str], default = "cuda"
          The device on which the parameters of the model will be allocated. It is the user's
          responsibility to ensure all parameters are moved to the GPU before running the
          forward pass.
    qkv_format: str, default = `sbhd`
            dimension format for `query_layer`, `key_layer` and `value_layer`,
            {`sbhd`, `bshd`}. `s` stands for the sequence length, `b` batch size,
            `h` the number of heads and `d` head size. `sbhd` and `bshd` formats
            are used for when sequences in a batch are of equal length or padded to
            equal length. Please note that these formats do not reflect how
            tensors `query_layer`, `key_layer`, `value_layer` are laid out in memory.
            For that, please use `get_qkv_layout` to gain the layout information.
    name: str, default = `None`
        name of the module, currently used for debugging purposes.

    Parallelism parameters
    ----------------------
    set_parallel_mode : bool, default = `False`
                      if set to `True`, QKV and FC1 layers are used as Column Parallel
                      whereas PROJ and FC2 is used as Row Parallel as described
                      `here <https://arxiv.org/pdf/1909.08053.pdf>`_.
    sequence_parallel : bool, default = `False`
                       if set to `True`, uses sequence parallelism.
    tp_group : ProcessGroup, default = `None`
              tensor parallel process group.
    tp_size : int, default = 1
             used as TP (tensor parallel) world size when TP groups are not formed during
             initialization. In this case, users must call the
             `set_tensor_parallel_group(tp_group)` method on the initialized module before the
             forward pass to supply the tensor parallel group needed for tensor and sequence
             parallel collectives.

    Optimization parameters
    -----------------------
    fuse_wgrad_accumulation : bool, default = 'False'
                             if set to `True`, enables fusing of creation and accumulation of
                             the weight gradient. When enabled, it is assumed that the weights
                             have an additional `main_grad` attribute (used instead of the
                             regular `grad`) which is a pre-allocated buffer of the correct
                             size to accumulate gradients in.
    params_dtype : torch.dtype, default = `torch.get_default_dtype()`
                  it controls the type used to allocate the initial parameters. Useful when
                  the model is trained with lower precision and the original FP32 parameters
                  would not fit in GPU memory.
    return_bias : bool, default = `False`
                 when set to `True`, this module will not apply the additive bias itself, but
                 instead return the bias value during the forward pass together with the
                 output of the linear transformation :math:`y = xA^T`. This is useful when
                 the bias addition can be fused to subsequent operations.
    fuse_qkv_params: bool, default = 'False'
                    if set to `True`, `TransformerLayer` module exposes a single fused
                    parameter for query-key-value. This enables optimizations such as QKV
                    fusion without concatentations/splits and also enables the argument
                    `fuse_wgrad_accumulation`.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        kv_channels: Optional[int] = None,
        attention_dropout: float = 0.1,
        layernorm_epsilon: float = 1e-5,
        init_method: Optional[Callable] = None,
        output_layer_init_method: Optional[Callable] = None,
        layer_number: Optional[int] = None,
        attn_mask_type: str = "causal",
        window_size: Optional[Tuple[int, int]] = None,
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        num_gqa_groups: Optional[int] = None,
        fuse_wgrad_accumulation: bool = False,
        get_rng_state_tracker: Optional[Callable] = None,
        sequence_parallel: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        return_bias: bool = False,
        return_layernorm_output: bool = False,
        input_layernorm: bool = False,
        attention_type: str = "self",
        set_parallel_mode: bool = False,
        fuse_qkv_params: bool = False,
        zero_centered_gamma: bool = False,
        qkv_weight_interleaved: bool = True,
        ub_overlap_ag: bool = False,
        ub_overlap_rs: bool = False,
        ub_overlap_rs_dgrad: bool = False,
        ub_bulk_dgrad: bool = False,
        ub_bulk_wgrad: bool = False,
        bias: bool = True,
        normalization: str = "LayerNorm",
        device: Union[torch.device, str] = "cuda",
        qkv_format: str = "sbhd",
        name: str = None,
    ) -> None:
        super().__init__()

        self.qkv_format = qkv_format
        self.attn_mask_type = attn_mask_type
        self.window_size = window_size
        self.layer_number = 1 if layer_number is None else layer_number
        self.input_layernorm = input_layernorm
        self.attention_type = attention_type
        self.get_rng_state_tracker = get_rng_state_tracker
        self.tp_group = tp_group
        self.return_layernorm_output = return_layernorm_output
        self.params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.num_attention_heads = num_attention_heads
        self.return_bias = return_bias
        self.cp_size = 1
        self.cp_rank = 0

        kv_channels = kv_channels if kv_channels else (hidden_size // num_attention_heads)

        if init_method is None:
            init_method = get_default_init_method()
        if output_layer_init_method is None:
            output_layer_init_method = get_default_init_method()

        if not fuse_qkv_params:
            qkv_weight_interleaved = False
        self.qkv_weight_interleaved = qkv_weight_interleaved

        assert attention_type in AttnTypes, f"attention_type {attention_type} not supported"
        if layer_number is not None:
            assert layer_number > 0, "layer_number must be a positive integer"

        tp_size = tp_size if tp_group is None else get_distributed_world_size(tp_group)
        self.tp_size = tp_size
        self.sequence_parallel = (tp_size > 1) and sequence_parallel

        self.num_attention_heads_per_partition = divide(num_attention_heads, tp_size)
        self.num_gqa_groups = num_attention_heads if num_gqa_groups is None else num_gqa_groups
        assert (
            num_attention_heads % self.num_gqa_groups == 0
        ), "The number of attention heads must be divisible by the number of GQA groups!"
        assert (
            self.num_gqa_groups % tp_size == 0
        ), "The number of GQA groups must be divisible by tensor parallel size!"
        self.num_gqa_groups_per_partition = int(self.num_gqa_groups // tp_size)

        self.hidden_size_per_attention_head = kv_channels
        self.hidden_size_q = self.hidden_size_per_attention_head * num_attention_heads
        self.hidden_size_kv = self.hidden_size_per_attention_head * self.num_gqa_groups

        self.name = name

        common_gemm_kwargs = {
            "fuse_wgrad_accumulation": fuse_wgrad_accumulation,
            "tp_group": tp_group,
            "tp_size": tp_size,
            "get_rng_state_tracker": get_rng_state_tracker,
            "sequence_parallel": sequence_parallel,
            "params_dtype": self.params_dtype,
            "device": device,
        }

        qkv_parallel_mode = "column" if set_parallel_mode else None

        if self.attention_type == "self":
            parameters_split = None
            if not fuse_qkv_params:
                parameters_split = collections.OrderedDict(
                    [
                        ("query", self.hidden_size_q),
                        ("key", self.hidden_size_kv),
                        ("value", self.hidden_size_kv),
                    ]
                )
            if self.input_layernorm:
                self.layernorm_qkv = LayerNormLinear(
                    hidden_size,
                    self.hidden_size_q + 2 * self.hidden_size_kv,
                    eps=layernorm_epsilon,
                    init_method=init_method,
                    bias=bias,
                    return_bias=False,
                    parallel_mode=qkv_parallel_mode,
                    return_layernorm_output=return_layernorm_output,
                    parameters_split=parameters_split,
                    zero_centered_gamma=zero_centered_gamma,
                    ub_bulk_wgrad=ub_bulk_wgrad,
                    ub_bulk_dgrad=ub_bulk_dgrad,
                    ub_overlap_rs_dgrad=ub_overlap_rs_dgrad,
                    ub_overlap_ag=ub_overlap_ag,
                    normalization=normalization,
                    ub_name="qkv",
                    name=name + ".layernorm_linear_qkv" if name is not None else None,
                    **common_gemm_kwargs,
                )
            else:
                self.qkv = Linear(
                    hidden_size,
                    self.hidden_size_q + 2 * self.hidden_size_kv,
                    init_method=init_method,
                    bias=bias,
                    return_bias=False,
                    parallel_mode=qkv_parallel_mode,
                    parameters_split=parameters_split,
                    name=name + ".linear_qkv" if name is not None else None,
                    **common_gemm_kwargs,
                )
        elif self.attention_type == "cross":
            if self.input_layernorm:
                self.layernorm_query = LayerNormLinear(
                    hidden_size,
                    self.hidden_size_q,
                    eps=layernorm_epsilon,
                    init_method=init_method,
                    bias=bias,
                    return_bias=False,
                    parallel_mode=qkv_parallel_mode,
                    parameters_split=("query",) if not fuse_qkv_params else None,
                    return_layernorm_output=return_layernorm_output,
                    zero_centered_gamma=zero_centered_gamma,
                    ub_bulk_wgrad=ub_bulk_wgrad,
                    ub_bulk_dgrad=ub_bulk_dgrad,
                    ub_overlap_rs_dgrad=ub_overlap_rs_dgrad,
                    ub_overlap_ag=ub_overlap_ag,
                    normalization=normalization,
                    ub_name="qkv",
                    name=name + ".layernorm_linear_q" if name is not None else None,
                    **common_gemm_kwargs,
                )
            else:
                self.query_layer = Linear(
                    hidden_size,
                    self.hidden_size_q,
                    init_method=init_method,
                    bias=bias,
                    return_bias=False,
                    parallel_mode=qkv_parallel_mode,
                    **common_gemm_kwargs,
                )
            self.key_value = Linear(
                hidden_size,
                2 * self.hidden_size_kv,
                init_method=init_method,
                bias=bias,
                return_bias=False,
                parallel_mode=qkv_parallel_mode,
                parameters_split=("key", "value") if not fuse_qkv_params else None,
                name=name + ".linear_kv" if name is not None else None,
                **common_gemm_kwargs,
            )

        # Attention.
        self.core_attention = DotProductAttention(
            num_attention_heads,
            self.hidden_size_per_attention_head,
            num_gqa_groups=self.num_gqa_groups,
            attention_dropout=attention_dropout,
            qkv_format=self.qkv_format,
            tp_size=tp_size,
            get_rng_state_tracker=get_rng_state_tracker,
            sequence_parallel=sequence_parallel,
            tp_group=tp_group,
            layer_number=self.layer_number,
            attention_type=self.attention_type,
        )

        # Linear
        self.proj = Linear(
            self.hidden_size_q,
            hidden_size,
            init_method=output_layer_init_method,
            bias=bias,
            return_bias=return_bias,
            parallel_mode="row" if set_parallel_mode else None,
            ub_overlap_rs=ub_overlap_rs,
            ub_overlap_ag=ub_overlap_ag,
            ub_name="proj",
            name=name + ".proj" if name is not None else None,
            **common_gemm_kwargs,
        )

    def set_tensor_parallel_group(self, tp_group: Union[dist_group_type, None]) -> None:
        """
        Set the tensor parallel group for the given
        module before executing the forward pass.

        Parameters
        ----------
        tp_group : ProcessGroup, default = `None`
                  tensor parallel process group.
        """
        self.tp_group = tp_group

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
                      Can be "p2p" or "all_gather" or "a2a", "a2a+p2p".
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
        if isinstance(cp_group, dist_group_type):
            self.cp_size = get_distributed_world_size(cp_group)
            self.cp_rank = get_distributed_rank(cp_group)
        elif isinstance(cp_group, list):
            assert len(cp_group) == 2, "Current implementation only supports two-level CP groups!"
            assert (
                cp_comm_type == "a2a+p2p"
            ), "Only cp_comm_type of a2a+p2p requires hierarchical CP groups!"
            cp_size_a2a = get_distributed_world_size(cp_group[0])
            cp_rank_a2a = get_distributed_rank(cp_group[0])
            cp_size_p2p = get_distributed_world_size(cp_group[1])
            cp_rank_p2p = get_distributed_rank(cp_group[1])
            self.cp_size = cp_size_a2a * cp_size_p2p
            self.cp_rank = cp_size_a2a * cp_rank_p2p + cp_rank_a2a

        # Deep iterate but skip self to avoid infinite recursion.
        for index, child in enumerate(self.modules()):
            if index == 0:
                continue
            if hasattr(child, "set_context_parallel_group"):
                child.set_context_parallel_group(cp_group, cp_global_ranks, cp_stream, cp_comm_type)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        encoder_output: Optional[torch.Tensor] = None,
        attn_mask_type: Optional[str] = None,
        window_size: Optional[Tuple[int, int]] = None,
        is_first_microbatch: Optional[bool] = None,
        checkpoint_core_attention: bool = False,
        inference_params: Optional[InferenceParams] = None,
        rotary_pos_emb: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[torch.Tensor] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        fast_zero_fill: bool = True,
        pad_between_seqs: Optional[bool] = None,
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        """
        Forward propagation for MultiheadAttention layer.

        .. note::

            Argument :attr:`attention_mask` is only used when :attr:`attn_mask_type`
            includes `"padding"` or `"arbitrary"`.

        Parameters
        ----------
        hidden_states : torch.Tensor
             Input tensor.
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]],
             default = `None`. Boolean tensor(s) used to mask out attention softmax input.
             It should be `None` for causal masks and "`no_mask`". For padding masks, it should be
             a single tensor of [batch_size, 1, 1, seqlen_q] for self-attention, and a tuple of
             two tensors in shapes [batch_size, 1, 1, seqlen_q] and [batch_size, 1, 1, seqlen_kv]
             for cross-attention. For "`arbitrary`" mask, it should be in a shape broadcastable to
             [batch_size, num_heads, max_seqlen_q, max_seqlen_kv]. A `True` value means
             the corresponding position is masked out and a `False` means that position
             is allowed to participate in attention.
        attn_mask_type: {'no_mask', 'padding', 'causal', 'padding_causal', 'causal_bottom_right',
                       'padding_causal_bottom_right','arbitrary'},
                       default = `None`
                       type of attention mask passed into softmax operation. By default,
                       causal masks are aligned to the top left corner of the softmax matrix.
                       When "`bottom_right`" is specified in the mask type, causal masks are
                       aligned to the bottom right corner.
        window_size: Optional[Tuple[int, int]], default = `None`
                    sliding window size for local attention.
        encoder_output : Optional[torch.Tensor], default = `None`
             Output of the encoder block to be fed into the decoder block if using
             `layer_type="decoder"`.
        is_first_microbatch : {True, False, None}, default = None
                             During training using either gradient accumulation or
                             pipeline parallelism a minibatch of data is further split
                             into microbatches. Between the microbatches of the same minibatch
                             the model weights are not updated. Setting this parameter indicates
                             whether the current microbatch is the first in a minibatch or not.
                             When set, this parameter enables additional optimizations:

                             * during FP8 training, it allows caching of the FP8 versions of
                               the weights
                             * it also allows skipping gradient accumulation during the
                               first microbatch (since it is the first gradient being
                               produced)
        checkpoint_core_attention: bool, default = `False`
                                  If true, forward activations for core attention are recomputed
                                  during the backward pass in order to save memory that would
                                  otherwise be occupied to store the forward activations until
                                  backprop.
        rotary_pos_emb: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], default = `None`
                       Embeddings for query and key tensors for applying rotary position
                       embedding. By default no input embedding is applied.
        core_attention_bias_type: str, default = `no_bias`
                    Bias type, {`no_bias`, `pre_scale_bias`, 'post_scale_bias`, `alibi`}
        core_attention_bias: Optional[torch.Tensor], default = `None`
                    Bias tensor for Q * K.T, shape [1, num_head, max_seqlen_q, max_seqlen_kv].
                    It should be 'None' for 'no_bias' and 'alibi' bias types.
        alibi_slopes: Optional[torch.Tensor], default = `None`
                     ALiBi slopes in FP32 and shape [nheads] or [batch_size, nheads].
                     It adds a bias of (-alibi_slope * (i + seqlen_k - seqlen_q - j))
                     to the attention score of query i and key j.
        cu_seqlens_q: Optional[torch.Tensor], default = `None`
                   Cumulative sum of sequence lengths (without offset) in a batch for `query_layer`,
                   with shape [batch_size + 1] and dtype torch.int32.
        cu_seqlens_kv: Optional[torch.Tensor], default = `None`
                   Cumulative sum of sequence lengths (without offset) in a batch for `key_layer`
                   and `value_layer`, with shape [batch_size + 1] and dtype torch.int32.
        max_seqlen_q: Optional[int], default = `None`
                      Maximum sequence length in `query_layer`.
                      Calculated from `cu_seqlens_q` if not provided.
        max_seqlen_kv: Optional[int], default = `None`
                       Maximum sequence length in `key_layer` and `value_layer`.
                       Calculated from `cu_seqlens_kv` if not provided.
        fast_zero_fill: bool, default = `True`
                    Whether to set output tensors to 0 or not before use.
        pad_between_seqs: Optional[bool], default = `None`
            If None, inferred from qkv_format, cu_seqlens and cu_seqlens_padded.
            If true, there are padding tokens between individual sequences in a packed batch.
        """
        # hidden_states: [sq, b, h]

        if attn_mask_type is None:
            attn_mask_type = self.attn_mask_type
        if window_size is None:
            window_size = self.window_size

        if "padding" in attn_mask_type and attention_mask is not None:
            for mask in attention_mask:
                assert mask.dtype == torch.bool, "Attention mask must be in boolean type!"

        assert (
            core_attention_bias_type in AttnBiasTypes
        ), f"core_attention_bias_type {core_attention_bias_type} is not supported!"

        if TEDebugState.debug_enabled:
            TransformerEngineBaseModule._validate_name(self)

        # =================================================
        # Pre-allocate memory for key-value cache for inference
        # =================================================

        if (
            inference_params is not None
            and self.layer_number not in inference_params.cache_manager.cache
        ):
            inference_params.allocate_memory(self.layer_number)

        # ======================
        # Query, Key, and Value
        # ======================

        fp8_mha = (
            FP8GlobalStateManager.is_fp8_enabled()
            and FP8GlobalStateManager.get_fp8_recipe().fp8_mha
        )

        layernorm_output = None
        if self.attention_type == "self":
            # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn]
            if self.input_layernorm:
                layernorm_qkv_outputs = self.layernorm_qkv(
                    hidden_states,
                    is_first_microbatch=is_first_microbatch,
                    fp8_output=fp8_mha and rotary_pos_emb is None,
                )
                if self.return_layernorm_output:
                    mixed_x_layer, layernorm_output = layernorm_qkv_outputs
                else:
                    mixed_x_layer = layernorm_qkv_outputs
            else:
                mixed_x_layer = self.qkv(
                    hidden_states,
                    is_first_microbatch=is_first_microbatch,
                    fp8_output=fp8_mha and rotary_pos_emb is None,
                )

            num_queries_per_key_value = (
                self.num_attention_heads_per_partition // self.num_gqa_groups_per_partition
            )
            if self.qkv_weight_interleaved:
                # [sq, b, ng * (np/ng + 2) * hn] --> [sq, b, ng, (np/ng + 2), hn]
                new_tensor_shape = mixed_x_layer.size()[:-1] + (
                    self.num_gqa_groups_per_partition,
                    (num_queries_per_key_value + 2),
                    self.hidden_size_per_attention_head,
                )
                # split along second last dimension
                split_dim = -2
            else:
                # [sq, b, ng * (np/ng + 2) * hn] --> [sq, b, (np/ng + 2), ng, hn]
                new_tensor_shape = mixed_x_layer.size()[:-1] + (
                    (num_queries_per_key_value + 2),
                    self.num_gqa_groups_per_partition,
                    self.hidden_size_per_attention_head,
                )
                # split along third last dimension
                split_dim = -3

            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # qkv_weight_interleaved:
            #  [sq, b, ng, (np/ng + 2), hn]
            #  --> [sq, b, ng, np/ng, hn], [sq, b, ng, 1, hn], [sq, b, ng, 1, hn]
            # not qkv_weight_interleaved:
            #  [sq, b, (np/ng + 2), ng, hn]
            #  --> [sq, b, np/ng, np, hn], [sq, b, 1, ng, hn], [sq, b, 1, ng, hn]
            query_layer, key_layer, value_layer = SplitAlongDim.apply(
                mixed_x_layer, split_dim, (num_queries_per_key_value, 1, 1)
            )

            if self.qkv_format == "thd":
                query_layer, key_layer, value_layer = (
                    x.reshape(x.size(0), -1, self.hidden_size_per_attention_head)
                    for x in (query_layer, key_layer, value_layer)
                )
            else:
                # query: -> [sq, b, np, hn]
                # key, value: -> [sq, b, ng, hn]
                query_layer, key_layer, value_layer = (
                    x.reshape(x.size(0), x.size(1), -1, self.hidden_size_per_attention_head)
                    for x in (query_layer, key_layer, value_layer)
                )
        elif self.attention_type == "cross":
            # Attention heads [sk, b, h] --> [sk, b, (ng * 2 * hn)]
            mixed_kv_layer = self.key_value(
                encoder_output,
                is_first_microbatch=is_first_microbatch,
                fp8_output=fp8_mha and rotary_pos_emb is None,
            )

            if self.qkv_weight_interleaved:
                # [sq, b, (ng * 2 * hn)] --> [sq, b, ng, 2 * hn]
                new_tensor_shape = mixed_kv_layer.size()[:-1] + (
                    self.num_gqa_groups_per_partition,
                    2 * self.hidden_size_per_attention_head,
                )
                # split along last dimension
                split_dim = -1
            else:
                # [sq, b, (ng * 2 * hn)] --> [sq, b, 2 * ng, hn]
                new_tensor_shape = mixed_kv_layer.size()[:-1] + (
                    2 * self.num_gqa_groups_per_partition,
                    self.hidden_size_per_attention_head,
                )
                # split along second last dimension
                split_dim = -2

            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # mixed_kv_layer --> 2 [sk, b, ng, hn]
            key_layer, value_layer = SplitAlongDim.apply(
                mixed_kv_layer,
                split_dim,
                mixed_kv_layer.shape[split_dim] // 2,
            )
            key_layer, value_layer = (
                x.reshape(
                    x.size(0),
                    x.size(1),
                    -1,
                    self.hidden_size_per_attention_head,
                )
                for x in (key_layer, value_layer)
            )

            # Attention head [sq, b, h] --> [sq, b, hp]
            if self.input_layernorm:
                layernorm_query_outputs = self.layernorm_query(
                    hidden_states,
                    is_first_microbatch=is_first_microbatch,
                    fp8_output=fp8_mha and rotary_pos_emb is None,
                )
                if self.return_layernorm_output:
                    query_layer, layernorm_output = layernorm_query_outputs
                else:
                    query_layer = layernorm_query_outputs
            else:
                query_layer = self.query_layer(
                    hidden_states,
                    is_first_microbatch=is_first_microbatch,
                    fp8_output=fp8_mha and rotary_pos_emb is None,
                )

            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            query_layer = query_layer.view(*new_tensor_shape)

        # ======================================================
        # Apply relative positional encoding (rotary embedding)
        # ======================================================

        if rotary_pos_emb is not None:
            assert not isinstance(query_layer, Float8Tensor) and not isinstance(
                key_layer, Float8Tensor
            ), "RoPE is not supported for Float8Tensors!"
            # duplicate the pos_emb for self attention
            if not isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = (rotary_pos_emb,) * 2

            q_pos_emb, k_pos_emb = rotary_pos_emb

            # adjust key and value for inference
            if inference_params is not None:
                if self.qkv_format == "sbhd":
                    sequence_length = key_layer.size(0)
                elif self.qkv_format == "bshd":
                    sequence_length = key_layer.size(1)
                else:
                    raise ValueError(
                        f"qkv_format={self.qkv_format} not supported for KV caching and RoPE."
                    )

                sequence_start = inference_params.get_seqlens_pre_step()
                # sequence_start = inference_params.seqlens[0]
                sequence_end = sequence_start + sequence_length

                q_pos_emb = q_pos_emb[sequence_start:sequence_end, ...]
                k_pos_emb = k_pos_emb[sequence_start:sequence_end, ...]

            query_layer = apply_rotary_pos_emb(
                query_layer,
                q_pos_emb,
                self.qkv_format,
                fused=True,
                cu_seqlens=cu_seqlens_q,
                cp_size=self.cp_size,
                cp_rank=self.cp_rank,
            )
            key_layer = apply_rotary_pos_emb(
                key_layer,
                k_pos_emb,
                self.qkv_format,
                fused=True,
                cu_seqlens=cu_seqlens_kv,
                cp_size=self.cp_size,
                cp_rank=self.cp_rank,
            )

        # ===========================
        # Core attention computation
        # ===========================

        context_layer = self.core_attention(
            query_layer,
            key_layer,
            value_layer,
            qkv_format=self.qkv_format,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            attention_mask=attention_mask,
            attn_mask_type=attn_mask_type,
            window_size=window_size,
            checkpoint_core_attention=checkpoint_core_attention,
            core_attention_bias_type=core_attention_bias_type,
            core_attention_bias=core_attention_bias,
            alibi_slopes=alibi_slopes,
            fast_zero_fill=fast_zero_fill,
            inference_params=inference_params,
            pad_between_seqs=pad_between_seqs,
        )

        # ===================
        # Output. [sq, b, h]
        # ===================
        projection_output = self.proj(
            context_layer,
            is_first_microbatch=is_first_microbatch,
            fp8_grad=isinstance(context_layer, QuantizedTensor),
        )

        if self.return_bias:
            attention_output, attention_bias = projection_output
        else:
            attention_output, attention_bias = projection_output, None

        outputs = (attention_output,)
        if self.return_bias:
            outputs += (attention_bias,)
        if self.input_layernorm and self.return_layernorm_output:
            outputs += (layernorm_output,)
        return outputs if len(outputs) > 1 else outputs[0]
