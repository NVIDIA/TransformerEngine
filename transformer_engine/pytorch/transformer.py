# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Transformer."""
import os
import warnings
from contextlib import nullcontext
from typing import Any, Callable, Optional, Tuple, Union

import torch

import transformer_engine_extensions as tex
from transformer_engine.pytorch.module import LayerNormMLP, LayerNorm, RMSNorm
from transformer_engine.pytorch.attention import MultiHeadAttention
from transformer_engine.pytorch.jit import (
    set_jit_fusion_options,
    warmup_jit_bias_dropout_add_all_dtypes,
    get_bias_dropout_add,
    bias_dropout_add_fused_train,
    bias_dropout_add_fused_inference,
)
from transformer_engine.pytorch.utils import (
    cast_if_needed,
    get_default_init_method,
)
from transformer_engine.pytorch.constants import (
    AttnMaskTypes,
    LayerTypes,
    dist_group_type,
)
from transformer_engine.pytorch.distributed import get_distributed_world_size


warnings.filterwarnings("module", category=DeprecationWarning, module="transformer")


__all__ = ["TransformerLayer"]


class DropPath(torch.nn.Module):
    """Drop paths (Stochastic Depth) per sample
    (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """DropPath FWD"""

        if self.drop_prob == 0.0 or not self.training:
            return hidden_state
        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        shape = (hidden_state.shape[0],) + (1,) * (hidden_state.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=hidden_state.dtype, device=hidden_state.device
        )
        random_tensor.floor_()  # binarize
        output = hidden_state.div(keep_prob) * random_tensor
        return output


class TransformerLayer(torch.nn.Module):
    r"""
    TransformerLayer is made up of an attention block and a feedforward network (MLP).
    This standard layer is based on the paper "Attention Is All You Need".

    .. warning::

        Arguments :attr:`attention_softmax_in_fp32` and :attr:`apply_query_key_layer_scaling`
        are deprecated and will be fully removed in future releases.

    .. note::

        Argument :attr:`attention_mask` will be ignored in the `forward` call when
        :attr:`self_attn_mask_type` is set to `"causal"`.

    Parameters
    ----------
    hidden_size : int
                 size of each input sample.
    ffn_hidden_size : int
                     intermediate size to which input samples are projected.
    num_attention_heads : int
                         number of attention heads in the transformer layer.
    num_gqa_groups : int, default = `None`
                         number of GQA groups in the transformer layer.
                         Grouped Query Attention is described in
                         `this paper <https://arxiv.org/pdf/2305.13245.pdf>`_.
                         This only affects the keys and values, not the querys.
                         GQA-1 is equivalent to Multi-Query Attention
                         (`MQA <https://arxiv.org/pdf/1911.02150.pdf>`_), while GQA-H
                         is equivalent to MHA, i.e. `num_gqa_groups = num_attention_heads`.
    layernorm_epsilon : float, default = 1e-5
                       a value added to the denominator of layer normalization
                       for numerical stability.
    hidden_dropout: float, default = 0.1
                   dropout probability for the dropout op after FC2 layer.
    attention_dropout: float, default = 0.1
                      dropout probability for the dropout op during multi-head attention.
    init_method : Callable, default = `None`
                 used for initializing weights of QKV and FC1 weights in the following way:
                 `init_method(weight)`. When set to `None`, defaults to
                 `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    output_layer_init_method : Callable, default = `None`
                              used for initializing weights of PROJ and FC2 in the following way:
                              `output_layer_init_method(weight)`. When set to `None`, defaults to
                              `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    apply_residual_connection_post_layernorm : bool, default = `False`
                                              if set to `True`, residual connections are taken
                                              from the output of layer norm (default is taken
                                              from input of layer norm)
    layer_number: int, default = `None`
                 layer number of the current `TransformerLayer` when multiple such modules are
                 concatenated to form a transformer block.
    output_layernorm: bool, default = `False`
                     if set to `True`, layer normalization is applied on the output side,
                     after the final dropout-add. default behavior is to apply layer
                     normalization on the input side, before the QKV transformation.
    layer_type: {'encoder', 'decoder'}, default = `encoder`
               if set to `decoder`, an additional cross-attn block is added after self-attn.
               This can be used for structures like `T5` Transformer in conjunction with the
               `encoder` option.
    kv_channels: int, default = `None`
                number of key-value channels. defaults to
                :attr:`hidden_size` / :attr:`num_attention_heads` if `None`.
    self_attn_mask_type: {'causal', 'padding'}, default = `causal`
                        type of attention mask passed into softmax operation.
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
    activation : str, default = 'gelu'
          Type of activation used in MLP block.
          Options are: 'gelu', 'relu', 'reglu', 'geglu' and 'swiglu'.
    device : Union[torch.device, str], default = "cuda"
          The device on which the parameters of the model will allocated. It is the user's
          responsibility to ensure all parameters are moved to the GPU before running the
          forward pass.

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
    seq_length: int
               sequence length of input samples. Needed for JIT Warmup, a technique where jit
               fused functions are warmed up before training to ensure same kernels are used for
               forward propogation and activation recompute phase.
    micro_batch_size: int
                     batch size per training step. Needed for JIT Warmup, a technique where jit
                     fused functions are warmed up before training to ensure same kernels are
                     used for forward propogation and activation recompute phase.
    drop_path_rate: float, default = 0.0
                   when > 0.0, applies stochastic depth per sample in
                   the main path of the residual block.
    fuse_qkv_params: bool, default = 'False'
                    if set to `True`, `TransformerLayer` module exposes a single fused
                    parameter for query-key-value. This enables optimizations such as QKV
                    fusion without concatentations/splits and also enables the argument
                    `fuse_wgrad_accumulation`.
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_attention_heads: int,
        num_gqa_groups: Optional[int] = None,
        layernorm_epsilon: float = 1e-5,
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        init_method: Optional[Callable] = None,
        output_layer_init_method: Optional[Callable] = None,
        layer_number: Optional[int] = None,
        kv_channels: Optional[int] = None,
        self_attn_mask_type: str = "causal",
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        params_dtype: Optional[torch.dtype] = None,
        get_rng_state_tracker: Optional[Callable] = None,
        fuse_wgrad_accumulation: bool = False,
        apply_query_key_layer_scaling: bool = False, # pylint: disable=unused-argument
        attention_softmax_in_fp32: bool = True, # pylint: disable=unused-argument
        seq_length: Optional[int] = None,
        micro_batch_size: Optional[int] = None,
        sequence_parallel: bool = False,
        apply_residual_connection_post_layernorm: bool = False,
        output_layernorm: bool = False,
        layer_type: str = "encoder",
        drop_path_rate: float = 0.0,
        set_parallel_mode: bool = False,
        fuse_qkv_params: bool = False,
        zero_centered_gamma: bool = False,
        qkv_weight_interleaved: bool = True,
        ub_tp_comm_overlap: bool = False,
        bias: bool = True,
        activation: str = 'gelu',
        normalization: str = "LayerNorm",
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        super().__init__()

        warnings.warn(
            "Arguments `attention_softmax_in_fp32` and `apply_query_key_layer_scaling`"
            "are deprecated and will be fully removed in future releases.",
            category=DeprecationWarning,
        )

        if ub_tp_comm_overlap:
            assert (
                tex.userbuf_comm_available()
            ), "Userbuffer communication backend not available."

        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        ub_tp_comm_overlap = ub_tp_comm_overlap and bool(int(os.getenv("NVTE_UB_OVERLAP", "1")))
        ub_bulk_wgrad = ub_tp_comm_overlap and bool(int(os.getenv("NVTE_UB_BULK_WGRAD", "1")))
        ub_bulk_dgrad = ub_tp_comm_overlap and bool(int(os.getenv("NVTE_UB_BULK_DGRAD", "1")))
        ub_split_ag = ub_tp_comm_overlap and bool(int(os.getenv("NVTE_UB_SPLIT_AG", "1")))
        ub_split_rs = ub_tp_comm_overlap and bool(int(os.getenv("NVTE_UB_SPLIT_RS", "1")))
        bias_dropout_fusion = bool(int(os.getenv("NVTE_BIAS_DROPOUT_FUSION", "1")))
        self.layer_number = layer_number
        self.output_layernorm = output_layernorm
        self.layer_type = layer_type
        self.apply_residual_connection_post_layernorm = (
            apply_residual_connection_post_layernorm
        )
        self.self_attn_mask_type = self_attn_mask_type
        assert (
            self_attn_mask_type in AttnMaskTypes
        ), f"self_attn_mask_type {self_attn_mask_type} not supported"
        assert layer_type in LayerTypes, f"layer_type {layer_type} not supported"

        if not fuse_qkv_params:
            assert (
                not fuse_wgrad_accumulation
            ), "Gradient accumulation fusion requires single QKV parameter."

        if not fuse_qkv_params:
            qkv_weight_interleaved = False

        self.kv_channels = (
            kv_channels if kv_channels else (hidden_size // num_attention_heads)
        )

        if init_method is None:
            init_method = get_default_init_method()
        if output_layer_init_method is None:
            output_layer_init_method = get_default_init_method()

        self.tp_size = tp_size if tp_group is None else get_distributed_world_size(tp_group)
        self.sequence_parallel = (self.tp_size > 1) and sequence_parallel
        self.seq_length = seq_length

        self.get_rng_state_tracker = get_rng_state_tracker

        attention_args = (
            hidden_size,
            num_attention_heads,
            self.kv_channels,
            attention_dropout,
            layernorm_epsilon,
            init_method,
            output_layer_init_method,
        )
        common_attention_kwargs = {
            "layer_number": layer_number,
            "tp_group": tp_group,
            "tp_size": self.tp_size,
            "num_gqa_groups": num_gqa_groups,
            "fuse_wgrad_accumulation": fuse_wgrad_accumulation,
            "get_rng_state_tracker": get_rng_state_tracker,
            "sequence_parallel": self.sequence_parallel,
            "params_dtype": params_dtype,
            "return_layernorm_output": apply_residual_connection_post_layernorm,
            "set_parallel_mode": set_parallel_mode,
            "fuse_qkv_params": fuse_qkv_params,
            "zero_centered_gamma": zero_centered_gamma,
            "qkv_weight_interleaved" : qkv_weight_interleaved,
            "ub_bulk_wgrad" : ub_bulk_wgrad,
            "ub_bulk_dgrad" : ub_bulk_dgrad,
            "ub_split_ag" : ub_split_ag,
            "ub_split_rs" : ub_split_rs,
        }

        self.self_attention = MultiHeadAttention(
            *attention_args,
            **common_attention_kwargs,
            attn_mask_type=self_attn_mask_type,
            input_layernorm=not output_layernorm,
            attention_type="self",
            bias=bias,
            normalization=normalization,
            device=device,
        )

        if layer_type == "decoder":
            self.inter_attention = MultiHeadAttention(
                *attention_args,
                **common_attention_kwargs,
                attn_mask_type="padding",
                input_layernorm=True,
                attention_type="cross",
                bias=bias,
                normalization=normalization,
                device=device,
            )

        # LayerNorm -> activation(Linear + Bias) -> Linear
        # parallel_mode not supported for LayerNormMLP,
        # FC1 is CPL and FC2 is RPL
        # In the case of GLU activation, FC1 handles both
        # Linear layers before the activation
        self.layernorm_mlp = LayerNormMLP(
            hidden_size,
            ffn_hidden_size,
            eps=layernorm_epsilon,
            fuse_wgrad_accumulation=fuse_wgrad_accumulation,
            tp_group=tp_group,
            tp_size=self.tp_size,
            get_rng_state_tracker=get_rng_state_tracker,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            bias=bias,
            return_bias=True,
            sequence_parallel=self.sequence_parallel,
            params_dtype=params_dtype,
            return_layernorm_output=apply_residual_connection_post_layernorm,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            set_parallel_mode=set_parallel_mode,
            zero_centered_gamma=zero_centered_gamma,
            ub_bulk_wgrad=ub_bulk_wgrad,
            ub_bulk_dgrad=ub_bulk_dgrad,
            ub_split_rs=ub_split_rs,
            ub_split_ag=ub_split_ag,
            activation=activation,
            normalization=normalization,
            device=device,
        )

        self.hidden_dropout = hidden_dropout
        self.bias_dropout_fusion = bias_dropout_fusion
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else None

        # Set bias+dropout+add fusion grad_enable execution handler.
        TORCH_MAJOR = int(torch.__version__.split(".")[0])
        TORCH_MINOR = int(torch.__version__.split(".")[1])
        use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        self.bias_dropout_add_exec_handler = (
            nullcontext if use_nvfuser else torch.enable_grad
        )

        if self.bias_dropout_fusion:
            set_jit_fusion_options()
            if seq_length and micro_batch_size:
                if self.sequence_parallel:
                    seq_length = seq_length // self.tp_size
                warmup_jit_bias_dropout_add_all_dtypes(
                    hidden_size, seq_length, micro_batch_size
                )

        norm_module = {
                "LayerNorm": LayerNorm,
                "RMSNorm": RMSNorm,
        }
        if self.output_layernorm:
            self.layernorm = norm_module[normalization](
                hidden_size,
                eps=layernorm_epsilon,
                sequence_parallel=self.sequence_parallel,
                params_dtype=params_dtype,
                zero_centered_gamma=zero_centered_gamma,
                device=device,
            )

    def set_tensor_parallel_group(self, tp_group: Union[dist_group_type, None]) -> None:
        """Set TP group"""
        # Deep iterate but skip self to avoid infinite recursion.
        for index, child in enumerate(self.modules()):
            if index == 0:
                continue
            if hasattr(child, "set_tensor_parallel_group"):
                child.set_tensor_parallel_group(tp_group)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_output: Optional[torch.Tensor] = None,
        enc_dec_attn_mask: Optional[torch.Tensor] = None,
        is_first_microbatch: Optional[bool] = None,
        checkpoint_core_attention: bool = False,
        inference_params: Optional[Any] = None,
        rotary_pos_emb: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[torch.Tensor] = None,
        fast_zero_fill: bool = True,
    ) -> torch.Tensor:
        """
        Transformer Layer: attention block and a feedforward network (MLP)

        .. note::

            Argument :attr:`attention_mask` will be ignored when :attr:`self_attn_mask_type`
            is set to `"causal"`.

        Parameters
        ----------
        hidden_states : torch.Tensor
             Input tensor.
        attention_mask : Optional[torch.Tensor], default = `None`
             Boolean tensor used to mask out self-attention softmax input.
        encoder_output : Optional[torch.Tensor], default = `None`
             Output of the encoder block to be fed into the decoder block if using
             `layer_type="decoder"`.
        enc_dec_attn_mask : Optional[torch.Tensor], default = `None`
             Boolean tensor used to mask out inter-attention softmax input if using
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
                    Bias type, {`no_bias`, `pre_scale_bias`, 'post_scale_bias`}
        core_attention_bias: Optional[torch.Tensor], default = `None`
                    Bias tensor for Q * K.T
        fast_zero_fill: bool, default = `True`
                    Whether to set output tensors to 0 or not before use.
        """

        hidden_states = hidden_states.contiguous()

        if self.sequence_parallel and self.seq_length is not None:
            assert (
                hidden_states.shape[0] == self.seq_length // self.tp_size
            ), "Sequence dimension must be split across TP group when using sequence parallel."

        if self.self_attn_mask_type != "causal" and attention_mask is not None:
            assert (
                attention_mask.dtype == torch.bool
            ), "Attention mask must be a boolean tensor"

        # For AMP
        if torch.is_autocast_enabled():
            hidden_states = cast_if_needed(
                hidden_states, torch.get_autocast_gpu_dtype()
            )

        # Self attention.
        self_attention_outputs = self.self_attention(
            hidden_states,
            attention_mask,
            inference_params=inference_params,
            is_first_microbatch=is_first_microbatch,
            checkpoint_core_attention=checkpoint_core_attention,
            rotary_pos_emb=rotary_pos_emb,
            core_attention_bias_type=core_attention_bias_type,
            core_attention_bias=core_attention_bias,
            fast_zero_fill=fast_zero_fill,
        )

        if self.apply_residual_connection_post_layernorm and not self.output_layernorm:
            attention_output, attention_bias, residual = self_attention_outputs
        else:
            attention_output, attention_bias = self_attention_outputs
            residual = hidden_states

        # Set BDA func.
        if self.bias_dropout_fusion:
            if self.training:
                bias_dropout_add_func = bias_dropout_add_fused_train
            else:
                bias_dropout_add_func = bias_dropout_add_fused_inference
        else:
            bias_dropout_add_func = get_bias_dropout_add(self.training)

        # Bias dropoout add.
        if self.drop_path is None and attention_bias.numel() != 0:
            with self.bias_dropout_add_exec_handler():
                bda_output = bias_dropout_add_func(
                    attention_output, attention_bias, residual, self.hidden_dropout
                )
        else:
            if attention_bias.numel() != 0:
                attention_output = attention_output + attention_bias
            out = torch.nn.functional.dropout(
                attention_output,
                p=self.hidden_dropout,
                training=self.training,
            )
            if self.drop_path is not None:
                out = self.drop_path(out)
            bda_output = residual + out

        # Cross attention.
        if self.layer_type == "decoder":
            inter_attention_outputs = self.inter_attention(
                bda_output,
                enc_dec_attn_mask,
                encoder_output=encoder_output,
                is_first_microbatch=is_first_microbatch,
                checkpoint_core_attention=checkpoint_core_attention,
                core_attention_bias_type=core_attention_bias_type,
                core_attention_bias=core_attention_bias,
                fast_zero_fill=fast_zero_fill,
            )
            if self.apply_residual_connection_post_layernorm:
                attention_output, attention_bias, residual = inter_attention_outputs
            else:
                attention_output, attention_bias = inter_attention_outputs
                residual = bda_output

            if attention_bias.numel() != 0:
                with self.bias_dropout_add_exec_handler():
                    bda_output = bias_dropout_add_func(
                        attention_output, attention_bias, residual, self.hidden_dropout
                    )
            else:
                out = torch.nn.functional.dropout(
                    attention_output,
                    p=self.hidden_dropout,
                    training=self.training,
                )
                bda_output = residual + out
        # MLP.
        mlp_outputs = self.layernorm_mlp(
            bda_output, is_first_microbatch=is_first_microbatch
        )
        if self.apply_residual_connection_post_layernorm:
            mlp_output, mlp_bias, residual = mlp_outputs
        else:
            mlp_output, mlp_bias = mlp_outputs
            residual = bda_output

        # Bias dropoout add.
        if self.drop_path is None and mlp_bias.numel() != 0:
            with self.bias_dropout_add_exec_handler():
                output = bias_dropout_add_func(
                    mlp_output, mlp_bias, residual, self.hidden_dropout
                )
        else:
            if mlp_bias.numel() != 0:
                mlp_output = mlp_output + mlp_bias
            out = torch.nn.functional.dropout(
                mlp_output, p=self.hidden_dropout, training=self.training
            )
            if self.drop_path is not None:
                out = self.drop_path(out)
            output = residual + out

        # For BERT like architectures.
        if self.output_layernorm:
            output = self.layernorm(output)

        # output: [b, s, h]
        return output
