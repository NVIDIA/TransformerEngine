# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Wrapper module for Transformer related layers with FP8 support.
"""
import functools
import operator
from typing import Any, Callable, Iterable, List, Sequence, Tuple, Union

import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from jax import lax
from jax import nn as jax_nn
from jax import random as jax_random

from .dot import fp8_dot
from .fp8 import FP8GemmPackage, FP8Helper
from .layernorm import canonicalize_layernorm_type
from .layernorm import layernorm, layernorm_fp8_dot
from .mlp import fp8_ln_mlp, geglu
from .sharding import infer_sharding_type
from .softmax import is_softmax_kernel_available
from .sharding import MajorShardingType, ShardingType
from .softmax import softmax, SoftmaxType

PRNGKey = Any
Shape = Tuple[int, ...]
DType = jnp.dtype
Array = jnp.ndarray
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str], Tuple[lax.Precision,
                                                                       lax.Precision]]
Initializer = Callable[[PRNGKey, Shape, DType], Array]


def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int]:
    # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
    return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


def _canonicalize_tuple(x):
    if isinstance(x, Iterable):
        return tuple(x)
    return (x,)


def _create_layernorm_parameters(layernorm_type, shape, scale_init, scale_axes, bias_init,
                                 bias_axes, dtype):
    scale = nn_partitioning.param_with_axes('scale',
                                            scale_init,
                                            shape,
                                            jnp.float32,
                                            axes=scale_axes)
    scale = jnp.asarray(scale, dtype)

    layernorm_type = canonicalize_layernorm_type(layernorm_type)
    if layernorm_type == 'layernorm':
        bias = nn_partitioning.param_with_axes('ln_bias',
                                               bias_init,
                                               shape,
                                               jnp.float32,
                                               axes=bias_axes)
        bias = jnp.asarray(bias, dtype)
    else:
        assert layernorm_type == 'rmsnorm'
        bias = None

    return scale, bias


def _convert_to_activation_function(fn_or_string: Union[str, Callable]) -> Callable:
    """Convert a string to an activation function."""
    if fn_or_string == 'linear':
        return lambda x: x
    if isinstance(fn_or_string, str):
        return getattr(nn, fn_or_string)
    if callable(fn_or_string):
        return fn_or_string

    raise ValueError(f"don't know how to convert {fn_or_string} to an activation function")


def _combine_biases(*masks: List[Array]):
    """Combine attention biases."""
    masks = [m for m in masks if m is not None]
    if not masks:
        return None
    assert all(map(lambda x: x.ndim == masks[0].ndim,
                   masks)), (f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}')
    mask, *other_masks = masks
    for other_mask in other_masks:
        mask = mask + other_mask
    return mask


class Softmax(nn.Module):
    r"""
    Applies softmax over a mini-batch of inputs.
    The inputs's shape should be [batch, heads, q_seqlen, k_seqlen].

    Parameters
    ----------
    scale_factor : float, default = 1.0
        scale the inputs along the last dimension before running softmax.
    softmax_type : SoftmaxType, default = 'layernorm'
        indicate the type of softmax.

    Optimization parameters
    -----------------------
    sharding_type : ShardingType, default = ShardingType.SINGLE
        indicate the sharding pattern.
    """

    scale_factor: float = 1.0
    softmax_type: SoftmaxType = SoftmaxType.SCALED
    sharding_type: ShardingType = ShardingType.SINGLE

    @nn.compact
    def __call__(self, inputs: Array, mask: Array = None, bias: Array = None) -> jnp.ndarray:
        batch = inputs.shape[0]
        heads = inputs.shape[1]
        q_seqlen = inputs.shape[2]
        k_seqlen = inputs.shape[3]
        dtype = inputs.dtype
        logits = inputs

        if (self.softmax_type is not SoftmaxType.SCALED and is_softmax_kernel_available(
                self.softmax_type, batch, heads, q_seqlen, k_seqlen, inputs.dtype)):

            if bias is not None:
                logits = logits + bias.astype(dtype)

            mask_ = mask
            if self.softmax_type is not SoftmaxType.SCALED_MASKED:
                mask_ = None

            outputs = softmax(logits, mask_, self.scale_factor, self.softmax_type,
                              self.sharding_type)
        else:
            attention_bias = None
            if mask is not None:
                attention_bias = lax.select(mask > 0,
                                            jnp.full(mask.shape, -1e10).astype(dtype),
                                            jnp.full(mask.shape, 0.).astype(dtype))

            if bias is not None:
                attention_bias = _combine_biases(attention_bias, bias)

            if attention_bias is not None:
                logits = logits + attention_bias.astype(dtype)

            # For the case that self.softmax == SoftmaxType.SCALED_UPPER_TRIANG_MASKED
            # and kernel is unavailable, then try on pure scaled softmax custom calls.
            if is_softmax_kernel_available(SoftmaxType.SCALED, batch, heads, q_seqlen, k_seqlen,
                                           dtype):
                outputs = softmax(logits, None, self.scale_factor, SoftmaxType.SCALED,
                                  self.sharding_type)
            else:
                outputs = jax_nn.softmax(logits)

        return outputs


class LayerNorm(nn.Module):
    r"""
    Applies layer normalization over a mini-batch of inputs.
    There are two types of normalization supported by this module,
    regular and root mean square layer Normalization.

    The regular layer normalization is as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    size of each input sample.

    The root mean square layer normalization (RMSNorm) is as described in
    the paper `Root Mean Square Layer Normalization <https://arxiv.org/abs/1910.07467>`__

    .. math::
        y = \frac{x}{ \mathrm{RMS}[x] + \epsilon} * \gamma

    .. math::
        RMS = \sqrt{\mathrm{E}[x^2]}

    :math:`\gamma` is learnable affine transform parameters of
    size of each input sample.

    Parameters
    ----------
    epsilon : float, default = 1e-6
        a value added to the denominator of layer normalization for numerical stability.
    layernorm_type : {'layernorm', 'rmsnorm'}, default = 'layernorm'
        indicate the type of layer normalization.
    scale_init : Initializer, default = flax.linen.initializers.ones
        used for initializing scale factors :math:`\gamma`.
    scale_axes : Tuple[str, ...], default = ('embed', )
        the name of axes used to shard the scale factors :math:`\gamma` with a corresponding mesh.
    bias_init : Initializer, default = flax.linen.initializers.zeros
        used for initializing shift factors :math:`\beta`,
        only works when :attr:`layernorm_type='layernorm'`.
    bias_axes : Tuple[str, ...], default = ('embed', )
        The name of axes used to shard the shift factors :math:`\beta` with a corresponding mesh.
        only works when :attr:`layernorm_type='layernorm'`.

    Optimization parameters
    -----------------------
    dtype : jax.numpy.dtype, default  = jax.numpy.float32
        the data type used to allocate the initial parameters.
    transpose_batch_sequence : bool, default = True
        indicate whether the input tensors were switched axis of batch
        and sequence length dimension. if set to True, the input tensors
        should be in (seqlen, batch, hidden), otherwise (batch, seqlen, hidden).
    sharding_type : ShardingType, default = ShardingType.SINGLE
        indicate the sharding pattern.
    """
    epsilon: float = 1e-6
    layernorm_type: str = 'layernorm'
    scale_init: Initializer = nn.initializers.ones
    scale_axes: Tuple[str, ...] = ('embed',)
    bias_init: Initializer = nn.initializers.zeros
    bias_axes: Tuple[str, ...] = ('embed',)
    dtype: DType = jnp.float32
    transpose_batch_sequence: bool = True
    sharding_type: ShardingType = ShardingType.SINGLE

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies layer normalization to the input :attr:`inputs`.

        Parameters
        ----------
        inputs : jax.numpy.ndarray
            Input tensors.

        Returns
        -------
        outputs : jax.numpy.ndarray
            Output tensors.
        """

        features = x.shape[-1]
        scale, ln_bias = _create_layernorm_parameters(self.layernorm_type, (features,),
                                                      self.scale_init, self.scale_axes,
                                                      self.bias_init, self.bias_axes, self.dtype)

        return layernorm(x,
                         scale,
                         ln_bias,
                         self.layernorm_type,
                         sharding_type=self.sharding_type,
                         dp_dim_index=1 if self.transpose_batch_sequence else 0,
                         epsilon=self.epsilon)


class TransformerEngineBase(nn.Module):
    """
    Base class of transformer engine
    """

    @staticmethod
    def get_fp8_metas(num_of_gemm: int) -> List[jnp.ndarray]:
        """
        Get the FP8 metas
        """
        num_of_meta = num_of_gemm * FP8Helper.NUM_META_PER_GEMM
        axes = ('fp8_meta_axis', 'fp8_meta_history')

        fp8_max = nn_partitioning.variable_with_axes(FP8Helper.FP8_COLLECTION_NAME,
                                                     FP8Helper.FP8_MAX_NAME,
                                                     FP8Helper.generate_fp8_max_array,
                                                     num_of_meta,
                                                     axes=axes)
        fp8_metas_amax = nn_partitioning.variable_with_axes(
            FP8Helper.FP8_COLLECTION_NAME,
            FP8Helper.FP8_AMAX_NAME,
            jnp.zeros, (num_of_meta, FP8Helper.AMAX_HISTORY_LEN),
            jnp.float32,
            axes=axes)
        fp8_metas_scale = nn_partitioning.variable_with_axes(FP8Helper.FP8_COLLECTION_NAME,
                                                             FP8Helper.FP8_SCALE_NAME,
                                                             jnp.ones, (num_of_meta, 1),
                                                             jnp.float32,
                                                             axes=axes)
        fp8_metas_scale_inv = nn_partitioning.variable_with_axes(FP8Helper.FP8_COLLECTION_NAME,
                                                                 FP8Helper.FP8_SCALE_INV_NAME,
                                                                 jnp.ones, (num_of_meta, 1),
                                                                 jnp.float32,
                                                                 axes=axes)

        return fp8_max.value, fp8_metas_amax.value, fp8_metas_scale.value, fp8_metas_scale_inv.value

    @staticmethod
    def get_fp8_gemm_package(num_of_gemm: int, inputs: jnp.ndarray,
                             kernels: List[jnp.ndarray]) -> FP8GemmPackage:
        """
        Get the FP8 metas
        """
        assert num_of_gemm == len(kernels)
        fp8_max, fp8_metas_amax, fp8_metas_scale, fp8_metas_scale_inv = \
            TransformerEngineBase.get_fp8_metas(num_of_gemm)

        return FP8GemmPackage(num_of_gemm, inputs, kernels, fp8_max, fp8_metas_amax,
                              fp8_metas_scale, fp8_metas_scale_inv)


class DenseGeneral(TransformerEngineBase):
    """
    Applies a linear transformation to the incoming data :math:`y = xA^T + b`

    Parameters
    ----------
    features : Union[Iterable[int], int]
        the hidden size of each output sample.
    kernel_init : Initializer, default =
        flax.linen.initializers.variance_scaling(1.0, 'fan_in', 'truncated_normal')
        used for initializing weights.
    kernel_axes : Tuple[str, ...], default = ()
        the name of axes used to shard the weights with a corresponding mesh.
    use_bias: bool, default = False
        indicate whether to enable bias shifting.
        if set to False, the layer will not learn an additive bias.
    bias_init: Initializer, default = flax.linen.initializers.zeros
        used for initializing bias, only works when :attr:`use_bias=True`.
    bias_axes: Tuple[str, ...], default = ()
        the name of axes used to shard bias with a corresponding mesh,
        only works when :attr:`use_bias=True`.
    axis:  Union[Iterable[int], int], default = -1
        a integer of tuple with axes to apply the transformation on.

    Optimization parameters
    -----------------------
    dtype : jax.numpy.dtype, default  = jax.numpy.float32
        the data type used to allocate the initial parameters.
    transpose_batch_sequence : bool, default = True
        indicate whether the input tensors were switched axis of batch
        and sequence length dimension. if set to True, the input tensors
        should be in (seqlen, batch, hidden), otherwise (batch, seqlen, hidden).
    sharding_type : ShardingType, default = ShardingType.SINGLE
        indicate the sharding pattern.
    """

    features: Union[Iterable[int], int]
    kernel_init: Initializer = nn.initializers.variance_scaling(1.0, 'fan_in', 'truncated_normal')
    kernel_axes: Tuple[str, ...] = ()
    use_bias: bool = False
    bias_init: Initializer = nn.initializers.zeros
    bias_axes: Tuple[str, ...] = ()
    axis: Union[Iterable[int], int] = -1
    dtype: DType = jnp.float32
    transpose_batch_sequence: bool = True
    sharding_type: ShardingType = ShardingType.SINGLE

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """
        Apply the linear transformation to the input.

        Parameters
        ----------
        inputs : jax.numpy.ndarray
            Input tensors.

        Returns
        -------
        outputs : jax.numpy.ndarray
            Output tensors.
        """
        features = _canonicalize_tuple(self.features)
        axis = _canonicalize_tuple(self.axis)

        inputs = jnp.asarray(inputs, self.dtype)
        axis = _normalize_axes(axis, inputs.ndim)

        kernel_shape = tuple(inputs.shape[ax] for ax in axis) + features
        kernel_param_shape = (np.prod([inputs.shape[ax] for ax in axis]),) + features
        kernel = nn_partitioning.param_with_axes('kernel',
                                                 self.kernel_init,
                                                 kernel_param_shape,
                                                 jnp.float32,
                                                 axes=self.kernel_axes)

        kernel = jnp.reshape(kernel, kernel_shape)

        if self.use_bias:
            bias = nn_partitioning.param_with_axes('bias',
                                                   self.bias_init, (self.features,),
                                                   self.dtype,
                                                   axes=self.bias_axes)
        else:
            bias = None

        contract_ind = tuple(range(0, len(axis)))

        if FP8Helper.enable_fp8():
            fp8_gemm_package = \
                TransformerEngineBase.get_fp8_gemm_package(1, inputs, [kernel])
            y = fp8_dot(fp8_gemm_package,
                        FP8Helper.FWD_DTYPE,
                        FP8Helper.BWD_DTYPE, (axis, contract_ind),
                        sharding_type=self.sharding_type,
                        dp_dim_index=1 if self.transpose_batch_sequence else 0)
        else:
            kernel = jnp.asarray(kernel, self.dtype)
            y = lax.dot_general(inputs, kernel, ((axis, contract_ind), ((), ())))

        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y


class LayerNormDenseGeneral(TransformerEngineBase):
    r"""
    Applies layer normalization followed by linear transformation to the incoming data.

    Parameters
    ----------
    features : Union[Iterable[int], int]
        the hidden size of each output sample.
    enable_layernorm: bool, default = True
        indicate whether to enable layer normalization before linear transformation.
    layernorm_type : {'layernorm', 'rmsnorm'}, default = 'layernorm'
        indicate the type of layer normalization.
    epsilon : float, default = 1e-6
        a value added to the denominator of layer normalization for numerical stability.
    scale_init : Initializer, default = flax.linen.initializers.ones
        used for initializing scale factors :math:`\gamma`.
    scale_axes : Tuple[str, ...], default = ('embed', )
        the name of axes used to shard the scale factors :math:`\gamma` with a corresponding mesh,
        only works when :attr:`enable_layernorm=True`.
    ln_bias_init: Initializer, default = flax.linen.initializers.zeros
        used for initializing shift factors :math:`\beta`,
        only works when :attr:`enable_layernorm=True` and :attr:`layernorm_type='layernorm'`.
    ln_bias_axes: Tuple[str, ...], default = ('embed', )
        The name of axes used to shard the shift factors :math:`\beta` with a corresponding mesh.
        only works when :attr:`enable_layernorm=True` and :attr:`layernorm_type='layernorm'`.
    kernel_init : Initializer, default =
        flax.linen.initializers.variance_scaling(1.0, 'fan_in', 'truncated_normal')
        used for initializing weights.
    kernel_axes : Tuple[str, ...], default = ()
        the name of axes used to shard the weights with a corresponding mesh.
    use_bias: bool, default = False
        indicate whether to enable bias shifting.
        if set to False, the layer will not learn an additive bias.
    bias_init: Initializer, default = flax.linen.initializers.zeros
        used for initializing bias, only works when :attr:`use_bias=True`.
    bias_axes: Tuple[str, ...], default = ()
        the name of axes used to shard bias with a corresponding mesh,
        only works when :attr:`use_bias=True`.
    return_layernorm_output: bool, default = True
        indicate whether to return the output of layer normalization.
        If set False, return None as the second tensor in outputs.
    axis:  Union[Iterable[int], int], default = -1
        a integer of tuple with axes to apply the transformation on.

    Optimization parameters
    -----------------------
    dtype : jax.numpy.dtype, default  = jax.numpy.float32
        the data type used to allocate the initial parameters.
    transpose_batch_sequence : bool, default = True
        indicate whether the input tensors were switched axis of batch
        and sequence length dimension. if set to True, the input tensors
        should be in (seqlen, batch, hidden), otherwise (batch, seqlen, hidden).
    depth_scaling: float, default = None
        the factor to scale the output from `DenseGeneral`. It should be a float
        value or None. When None is set, then no scaling is applied.
    sharding_type : ShardingType, default = ShardingType.SINGLE
        indicate the sharding pattern.
    """

    features: Union[Iterable[int], int]
    enable_layernorm: bool = True
    layernorm_type: str = 'layernorm'
    epsilon: float = 1e-6
    scale_init: Initializer = nn.initializers.ones
    scale_axes: Tuple[str, ...] = ('embed',)
    ln_bias_init: Initializer = nn.initializers.zeros
    ln_bias_axes: Tuple[str, ...] = ('embed',)
    kernel_init: Initializer = nn.initializers.variance_scaling(1.0, 'fan_in', 'truncated_normal')
    kernel_axes: Tuple[str, ...] = ()
    use_bias: bool = False
    bias_init: Initializer = nn.initializers.zeros
    bias_axes: Tuple[str, ...] = ()
    return_layernorm_output: bool = True
    axis: Union[Iterable[int], int] = -1
    dtype: DType = jnp.float32
    transpose_batch_sequence: bool = True
    depth_scaling: float = None
    sharding_type: ShardingType = ShardingType.SINGLE

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """
        Apply layer normalization to the input followed by a linear transformation.

        Parameters
        ----------
        inputs: jax.numpy.ndarray
            Input tensor.

        Returns
        -------
        outputs : jax.numpy.ndarray
            Output tensors.
        ln_outputs: jax.numpy.ndarray
            The output tensors of layer normalization.
            If :attr:`return_layernorm_output=False`, then this woulb be None.
        """
        ln_output = None

        fuse_layernorm = FP8Helper.enable_fp8(
        ) and not self.return_layernorm_output and self.enable_layernorm

        if self.enable_layernorm:
            features = inputs.shape[-1]

            scale, ln_bias = _create_layernorm_parameters(self.layernorm_type, (features,),
                                                          self.scale_init, self.scale_axes,
                                                          self.ln_bias_init, self.ln_bias_axes,
                                                          self.dtype)

            if not fuse_layernorm:
                y = layernorm(inputs,
                              scale,
                              ln_bias,
                              layernorm_type=self.layernorm_type,
                              sharding_type=self.sharding_type,
                              dp_dim_index=1 if self.transpose_batch_sequence else 0,
                              epsilon=self.epsilon)
            else:
                assert not self.return_layernorm_output
                y = inputs
        else:
            y = inputs

        if self.return_layernorm_output:
            ln_output = y

        # DenseGeneral
        features = _canonicalize_tuple(self.features)
        axis = _canonicalize_tuple(self.axis)

        axis = _normalize_axes(axis, y.ndim)

        kernel_shape = tuple(y.shape[ax] for ax in axis) + features
        kernel_param_shape = (np.prod([inputs.shape[ax] for ax in axis]),) + features
        kernel = nn_partitioning.param_with_axes('kernel',
                                                 self.kernel_init,
                                                 kernel_param_shape,
                                                 jnp.float32,
                                                 axes=self.kernel_axes)

        kernel = jnp.reshape(kernel, kernel_shape)

        contract_ind = tuple(range(0, len(axis)))

        if FP8Helper.enable_fp8():
            fp8_gemm_package = \
                    TransformerEngineBase.get_fp8_gemm_package(1, y, [kernel])

            if not fuse_layernorm:
                z = fp8_dot(fp8_gemm_package,
                            FP8Helper.FWD_DTYPE,
                            FP8Helper.BWD_DTYPE, (axis, contract_ind),
                            sharding_type=self.sharding_type,
                            dp_dim_index=1 if self.transpose_batch_sequence else 0)
            else:
                z = layernorm_fp8_dot(fp8_gemm_package,
                                      scale,
                                      ln_bias,
                                      self.layernorm_type,
                                      FP8Helper.FWD_DTYPE,
                                      FP8Helper.BWD_DTYPE, (axis, contract_ind),
                                      sharding_type=self.sharding_type,
                                      dp_dim_index=1 if self.transpose_batch_sequence else 0,
                                      epsilon=self.epsilon)
        else:
            kernel = jnp.asarray(kernel, self.dtype)
            z = lax.dot_general(y, kernel, ((axis, contract_ind), ((), ())))

        bias = None
        if self.use_bias:
            bias = nn_partitioning.param_with_axes('bias',
                                                   self.bias_init, (self.features,),
                                                   self.dtype,
                                                   axes=self.bias_axes)

        if bias is not None:
            z += jnp.reshape(bias, (1,) * (z.ndim - 1) + (-1,))

        if self.depth_scaling is not None:
            z = z / self.depth_scaling

        return z, ln_output    # dense_output, layer_norm_output


class LayerNormMLP(TransformerEngineBase):
    r"""
    Applies layer normalization on the input followed by the MLP module,
    consisting of 2 successive linear transformations, separated by given activations.

    Parameters
    ----------
    intermediate_dim: int, default = 2048
        intermediate size to which input samples are projected.
    enable_layernorm: bool, default = True
        indicate whether to enable layer normalization before linear transformation.
    layernorm_type : {'layernorm', 'rmsnorm'}, default = 'layernorm'
        indicate the type of layer normalization.
    epsilon : float, default = 1e-6
        a value added to the denominator of layer normalization for numerical stability.
    scale_init : Initializer, default = flax.linen.initializers.ones
        used for initializing scale factors :math:`\gamma`.
    scale_axes : Tuple[str, ...], default = ('embed', )
        the name of axes used to shard the scale factors :math:`\gamma` with a corresponding mesh,
        only works when :attr:`enable_layernorm=True`.
    ln_bias_init: Initializer, default = flax.linen.initializers.zeros
        used for initializing shift factors :math:`\beta`,
        only works when :attr:`enable_layernorm=True` and :attr:`layernorm_type='layernorm'`.
    ln_bias_axes: Tuple[str, ...], default = ('embed', )
        The name of axes used to shard the shift factors :math:`\beta` with a corresponding mesh.
        only works when :attr:`enable_layernorm=True` and :attr:`layernorm_type='layernorm'`.
    kernel_init : Initializer, default =
        flax.linen.initializers.variance_scaling(1.0, 'fan_in', 'truncated_normal')
        used for initializing weight of both linear transformations.
    kernel_axes_1 : Tuple[str, ...], default = ('embed', 'act', 'mlp')
        the name of axes used to shard the weights with a corresponding mesh for
        the weight of the first linear transformations.
    kernel_axes_2 : Tuple[str, ...], default = ('mlp', 'embed')
        the name of axes used to shard the weights with a corresponding mesh for
        the weight of the second linear transformations.
    use_bias: bool, default = False
        indicate whether to enable bias shifting.
        if set to False, the layer will not learn an additive bias.
    bias_init: Initializer, default = flax.linen.initializers.zeros
        used for initializing bias, only works when :attr:`use_bias=True`.
    bias_axes_1: Tuple[str, ...], default = ('mlp',)
        the name of axes used to shard bias with a corresponding mesh  for
        the weight of the first linear transformations.
        only works when :attr:`use_bias=True`.
    bias_axes_2: Tuple[str, ...], default = ('embed',)
        the name of axes used to shard bias with a corresponding mesh  for
        the weight of the second linear transformations.
        only works when :attr:`use_bias=True`.
    return_layernorm_output: bool, default = True
        indicate whether to return the output of layer normalization.
        If set False, return None as the second tensor in outputs.
    activations: Sequence[Union[str, Callable]], default = ('relu',)
        the sequence of activation functions to apply after the first linear transformation.
        Each activation has its own transformation layer.
    intermediate_dropout_rate: float, default = 0.1
        dropout probability for the dropout op after the :attr:`activations`.
    axis:  Union[Iterable[int], int], default = -1
        a integer of tuple with axes to apply the transformation on.

    Optimization parameters
    -----------------------
    dtype : jax.numpy.dtype, default  = jax.numpy.float32
        the data type used to allocate the initial parameters.
    transpose_batch_sequence : bool, default = True
        indicate whether the input tensors were switched axis of batch
        and sequence length dimension. if set to True, the input tensors
        should be in (seqlen, batch, hidden), otherwise (batch, seqlen, hidden).
    major_sharding_type : MajorShardingType, default = MajorShardingType.SINGLE
        indicate the sharding pattern.
    """

    intermediate_dim: int = 2048
    enable_layernorm: bool = True
    layernorm_type: str = 'layernorm'
    epsilon: float = 1e-6
    scale_init: Initializer = nn.initializers.ones
    scale_axes: Tuple[str, ...] = ('embed',)
    ln_bias_init: Initializer = nn.initializers.zeros
    ln_bias_axes: Tuple[str, ...] = ('embed',)
    kernel_init: Initializer = nn.initializers.variance_scaling(1.0, 'fan_in', 'truncated_normal')
    kernel_axes_1: Tuple[str, ...] = ('embed', 'act', 'mlp')
    kernel_axes_2: Tuple[str, ...] = ('mlp', 'embed')
    use_bias: bool = False
    bias_init: Initializer = nn.initializers.zeros
    bias_axes_1: Tuple[str, ...] = ('mlp',)
    bias_axes_2: Tuple[str, ...] = ('embed',)
    return_layernorm_output: bool = True
    activations: Sequence[Union[str, Callable]] = ('relu',)
    intermediate_dropout_rate: float = 0.1
    axis: Union[Iterable[int], int] = -1
    dtype: DType = jnp.float32
    transpose_batch_sequence: bool = True
    major_sharding_type: MajorShardingType = MajorShardingType.SINGLE

    @nn.compact
    def __call__(self, inputs: Array, deterministic: bool = False) -> Array:
        """
        Apply layer normalization to the input followed by a feedforward network (MLP Block).

        Parameters
        ----------
        inputs: jax.numpy.ndarray
            Input tensor.
        deterministic: bool, default  = False
            Disable dropout ops if set to True.

        Returns
        -------
        outputs : jax.numpy.ndarray
            Output tensors.
        ln_outputs: jax.numpy.ndarray
            The output tensors of layer normalization.
            If :attr:`return_layernorm_output=False`, then this woulb be None.
        """
        ln_output = None

        fuse_layernorm = FP8Helper.enable_fp8(
        ) and not self.return_layernorm_output and self.enable_layernorm

        use_fused_ln_mlp = fuse_layernorm \
            and (not self.use_bias) and self.activations == ('gelu', 'linear') \
                and (self.intermediate_dropout_rate < 1e-3)

        first_sharding_type, second_sharding_type = infer_sharding_type(self.major_sharding_type)

        # LayerNorm
        if self.enable_layernorm:
            features = inputs.shape[-1]

            scale, ln_bias = _create_layernorm_parameters(self.layernorm_type, (features,),
                                                          self.scale_init, self.scale_axes,
                                                          self.ln_bias_init, self.ln_bias_axes,
                                                          self.dtype)

            if not fuse_layernorm:
                y = layernorm(inputs,
                              scale,
                              ln_bias,
                              layernorm_type=self.layernorm_type,
                              sharding_type=first_sharding_type,
                              dp_dim_index=1 if self.transpose_batch_sequence else 0,
                              epsilon=self.epsilon)
            else:
                assert not self.return_layernorm_output
                y = inputs
        else:
            y = inputs

        if self.return_layernorm_output:
            ln_output = y

        def kernel_1_init(key, num_kernels, stack_axis, *init_args):
            kernels = []
            for _ in range(num_kernels):
                key, init_key = jax_random.split(key)
                kernels.append(self.kernel_init(init_key, *init_args))
            return jnp.stack(kernels, axis=stack_axis, dtype=jnp.float32)

        num_of_gemm = 2
        if use_fused_ln_mlp:
            num_activations = len(self.activations)
            axis = _canonicalize_tuple(self.axis)
            axis = _normalize_axes(axis, inputs.ndim)

            intermediate_dim = _canonicalize_tuple((num_activations, self.intermediate_dim))
            kernel_1_shape = tuple(inputs.shape[ax] for ax in axis) + intermediate_dim
            kernel_1_each_shape = (np.prod([y.shape[ax] for ax in axis]), self.intermediate_dim)
            kernel_1 = nn_partitioning.param_with_axes('wi_kernel',
                                                       kernel_1_init,
                                                       num_activations,
                                                       -2,
                                                       kernel_1_each_shape,
                                                       jnp.float32,
                                                       axes=self.kernel_axes_1)
            kernel_1 = jnp.reshape(kernel_1, kernel_1_shape)
            hidden_size = inputs.shape[-1]
            hidden_size_tuple = _canonicalize_tuple(hidden_size)
            kernel_2_shape = (self.intermediate_dim,) + hidden_size_tuple
            kernel_2_param_shape = (self.intermediate_dim, np.prod(hidden_size_tuple))
            kernel_2 = nn_partitioning.param_with_axes('wo_kernel',
                                                       self.kernel_init,
                                                       kernel_2_param_shape,
                                                       jnp.float32,
                                                       axes=self.kernel_axes_2)
            kernel_2 = jnp.reshape(kernel_2, kernel_2_shape)
            contract_ind = tuple(range(0, len(axis)))

            fp8_gemm_package = \
                TransformerEngineBase.get_fp8_gemm_package(num_of_gemm, y, [kernel_1, kernel_2])
            out = fp8_ln_mlp(fp8_gemm_package,
                             scale,
                             ln_bias,
                             self.layernorm_type,
                             FP8Helper.FWD_DTYPE,
                             FP8Helper.BWD_DTYPE,
                             epsilon=self.epsilon,
                             contracting_dims=(axis, contract_ind),
                             major_sharding_type=self.major_sharding_type,
                             dp_dim_index=1 if self.transpose_batch_sequence else 0,
                             activations=self.activations)
        else:    # not use_fused_ln_mlp

            def fp8_meta_generator():
                fp8_max, fp8_metas_amax, fp8_metas_scale, fp8_metas_scale_inv = (None, None, None,
                                                                                 None)
                if FP8Helper.enable_fp8():
                    fp8_max, fp8_metas_amax, fp8_metas_scale, fp8_metas_scale_inv = \
                        TransformerEngineBase.get_fp8_metas(num_of_gemm)
                return fp8_max, fp8_metas_amax, fp8_metas_scale, fp8_metas_scale_inv

            fp8_max, fp8_metas_amax, fp8_metas_scale, fp8_metas_scale_inv = \
                fp8_meta_generator()

            # DenseGeneral 1
            activations = []
            num_activations = len(self.activations)
            axis = _canonicalize_tuple(self.axis)
            axis = _normalize_axes(axis, y.ndim)

            intermediate_dim = _canonicalize_tuple((num_activations, self.intermediate_dim))
            kernel_shape = tuple(y.shape[ax] for ax in axis) + intermediate_dim
            kernel_1_each_shape = (np.prod([y.shape[ax] for ax in axis]), self.intermediate_dim)
            kernel = nn_partitioning.param_with_axes('wi_kernel',
                                                     kernel_1_init,
                                                     num_activations,
                                                     -2,
                                                     kernel_1_each_shape,
                                                     jnp.float32,
                                                     axes=self.kernel_axes_1)
            kernel = jnp.reshape(kernel, kernel_shape)
            contract_ind = tuple(range(0, len(axis)))

            if FP8Helper.enable_fp8():
                fp8_gemm_package = FP8GemmPackage(
                    1, y, [kernel], fp8_max[:FP8Helper.NUM_META_PER_GEMM, :],
                    fp8_metas_amax[:FP8Helper.NUM_META_PER_GEMM, :],
                    fp8_metas_scale[:FP8Helper.NUM_META_PER_GEMM, :],
                    fp8_metas_scale_inv[:FP8Helper.NUM_META_PER_GEMM, :])

                if not fuse_layernorm:
                    x = fp8_dot(fp8_gemm_package,
                                FP8Helper.FWD_DTYPE,
                                FP8Helper.BWD_DTYPE, (axis, contract_ind),
                                sharding_type=first_sharding_type,
                                dp_dim_index=1 if self.transpose_batch_sequence else 0)
                else:
                    x = layernorm_fp8_dot(fp8_gemm_package,
                                          scale,
                                          ln_bias,
                                          self.layernorm_type,
                                          FP8Helper.FWD_DTYPE,
                                          FP8Helper.BWD_DTYPE, (axis, contract_ind),
                                          sharding_type=first_sharding_type,
                                          dp_dim_index=1 if self.transpose_batch_sequence else 0,
                                          epsilon=self.epsilon)
            else:    # not enable fp8
                kernel = jnp.asarray(kernel, self.dtype)
                x = lax.dot_general(y, kernel, ((axis, contract_ind), ((), ())))

            bias = None
            if self.use_bias:
                bias = nn_partitioning.param_with_axes('wi_bias',
                                                       self.bias_init, (self.intermediate_dim,),
                                                       self.dtype,
                                                       axes=self.bias_axes_1)
                x += jnp.reshape(bias, (1,) * (x.ndim - 1) + (-1,))

            if self.activations == ('gelu', 'linear'):
                z = geglu(x,
                          contracting_dims=(-2, -1),
                          sharding_type=second_sharding_type,
                          dp_dim_index=1 if self.transpose_batch_sequence else 0)
            else:
                x = jnp.split(x, num_activations, axis=-2)
                for idx, act_fn in enumerate(self.activations):
                    x_i = _convert_to_activation_function(act_fn)(x[idx])
                    activations.append(x_i)
                z = functools.reduce(operator.mul, activations)
                z = jnp.reshape(z, (*z.shape[:-2], -1))

            z = nn.Dropout(rate=self.intermediate_dropout_rate, broadcast_dims=(-2,))(
                z, deterministic=deterministic)    # Broadcast along length.

            # DenseGeneral 2
            hidden_size = inputs.shape[-1]
            hidden_size_tuple = _canonicalize_tuple(hidden_size)
            axis = _canonicalize_tuple(self.axis)
            axis = _normalize_axes(axis, z.ndim)

            kernel_shape = tuple(z.shape[ax] for ax in axis) + hidden_size_tuple
            kernel_param_shape = (np.prod([z.shape[ax] for ax in axis]), np.prod(hidden_size_tuple))
            kernel = nn_partitioning.param_with_axes('wo_kernel',
                                                     self.kernel_init,
                                                     kernel_param_shape,
                                                     jnp.float32,
                                                     axes=self.kernel_axes_2)
            kernel = jnp.reshape(kernel, kernel_shape)

            contract_ind = tuple(range(0, len(axis)))

            if FP8Helper.enable_fp8():
                fp8_gemm_package = FP8GemmPackage(
                    1, z, [kernel], fp8_max[FP8Helper.NUM_META_PER_GEMM:, :],
                    fp8_metas_amax[FP8Helper.NUM_META_PER_GEMM:, :],
                    fp8_metas_scale[FP8Helper.NUM_META_PER_GEMM:, :],
                    fp8_metas_scale_inv[FP8Helper.NUM_META_PER_GEMM:, :])

                out = fp8_dot(fp8_gemm_package,
                              FP8Helper.FWD_DTYPE,
                              FP8Helper.BWD_DTYPE, (axis, contract_ind),
                              sharding_type=second_sharding_type,
                              dp_dim_index=1 if self.transpose_batch_sequence else 0)
            else:
                kernel = jnp.asarray(kernel, self.dtype)
                out = lax.dot_general(z, kernel, ((axis, contract_ind), ((), ())))

            bias = None
            if self.use_bias:
                bias = nn_partitioning.param_with_axes('wo_bias',
                                                       self.bias_init, (hidden_size,),
                                                       self.dtype,
                                                       axes=self.bias_axes_2)
                out += jnp.reshape(bias, (1,) * (out.ndim - 1) + (-1,))

        return out, ln_output    # Output, layner_norm_output
