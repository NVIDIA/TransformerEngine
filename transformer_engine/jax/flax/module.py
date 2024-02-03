# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Wrapper module for Transformer related layers with FP8 support.
"""
import functools
import operator
import warnings
from typing import Any, Callable, Iterable, List, Sequence, Tuple, Union

import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from jax import lax
from jax import nn as jax_nn
from jax import random as jax_random
from jax.ad_checkpoint import checkpoint_name

from ..dot import type_safe_dot_general
from ..fp8 import FP8Helper, FP8MetaPackage
from ..layernorm import canonicalize_layernorm_type
from ..layernorm import layernorm, layernorm_fp8_dot
from ..mlp import layernorm_geglu_fp8_mlp, geglu
from ..mlp import layernorm_gelu_fp8_mlp, gelu
from ..softmax import is_softmax_kernel_available
from ..softmax import softmax, SoftmaxType
from ..sharding import with_sharding_constraint_by_logical_axes

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


def _obtain_default_layernorm_scale_init_if_need(original_init, zero_centered_gamma):
    if original_init is None:
        if not zero_centered_gamma:
            return nn.initializers.ones
    return nn.initializers.zeros


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


class Softmax(nn.Module):    # pylint: disable=too-few-public-methods
    r"""
    Applies softmax over a mini-batch of inputs.
    The input's shape should be [batch, heads, q_seqlen, k_seqlen].

    .. code-block:: python
        shifted_input = input + bias
        masked_scaled = (1 - mask)*(shifted_input * scale_factor)
        softmax_mask = mask * -1e-10
        output = softmax(masked_scaled + softmax_mask)

    Parameters
    ----------
    scale_factor : float, default = 1.0
        Scalar for the input to softmax.
    softmax_type : SoftmaxType, default = SoftmaxType.SCALED
        Indicate the type of softmax.
    """

    scale_factor: float = 1.0
    softmax_type: SoftmaxType = SoftmaxType.SCALED

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

            outputs = softmax(logits, mask_, self.scale_factor, self.softmax_type)
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
                outputs = softmax(logits, None, self.scale_factor, SoftmaxType.SCALED)
            else:
                outputs = jax_nn.softmax(logits * self.scale_factor)

        return outputs


class LayerNorm(nn.Module):    # pylint: disable=too-few-public-methods
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
        A value added to the denominator of layer normalization for numerical stability.
    layernorm_type : {'layernorm', 'rmsnorm'}, default = 'layernorm'
        Indicate the type of layer normalization.
    zero_centered_gamma : bool, default = False
        If set to `True`, the LayerNorm formula changes to

        .. math::
            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} *
            (1 + \gamma) + \beta

        This parameter is only applicable for 'layernorm'.
        The default of `scale_init` will also be changed. See `scale_init`.
    scale_init : Initializer, default = None
        Used for initializing scale factors :math:`\gamma`.
        If `None` is provided, scale_init is set according to the value of zero_centered_gamma.
        If zero_centered_gamma is set to `True`, then scale_init is `flax.linen.initializers.zeros`.
        Otherwise, scale_init is `flax.linen.initializers.ones`.
        It should be a callable object with three arguments (jax.random.PRNGKey, shape, dtype).
    scale_axes : Tuple[str, ...], default = ('embed', )
        The name of axes used to shard the scale factors :math:`\gamma` with a corresponding mesh.
    bias_init : Initializer, default = flax.linen.initializers.zeros
        Used for initializing shift factors :math:`\beta`,
        only used when :attr:`layernorm_type='layernorm'`.
        It should be a callable object with three arguments (jax.random.PRNGKey, shape, dtype).
    bias_axes : Tuple[str, ...], default = ('embed', )
        The name of axes used to shard the shift factors :math:`\beta` with a corresponding mesh.
        only used when :attr:`layernorm_type='layernorm'`.

    Optimization parameters
    -----------------------
    dtype : jax.numpy.dtype, default  = jax.numpy.float32
        the data type used to allocate the initial parameters.
    transpose_batch_sequence : bool, default = False
        Indicate whether the input tensors were switched axis of batch
        and sequence length dimension. If set to True, the input tensors
        should be in (seqlen, batch, hidden), otherwise (batch, seqlen, hidden).
    """
    epsilon: float = 1e-6
    layernorm_type: str = 'layernorm'
    zero_centered_gamma: bool = False
    scale_init: Initializer = None
    scale_axes: Tuple[str, ...] = ('embed',)
    bias_init: Initializer = nn.initializers.zeros
    bias_axes: Tuple[str, ...] = ('embed',)
    dtype: DType = jnp.float32
    transpose_batch_sequence: bool = False
    sharding_type = None

    def __post_init__(self):
        self.scale_init = _obtain_default_layernorm_scale_init_if_need(
            self.scale_init, self.zero_centered_gamma)
        super().__post_init__()

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
        warnings.warn("sharding_type of LayerNorm would be removed in the near feature",
                      DeprecationWarning)

        features = x.shape[-1]
        scale, ln_bias = _create_layernorm_parameters(self.layernorm_type, (features,),
                                                      self.scale_init, self.scale_axes,
                                                      self.bias_init, self.bias_axes, self.dtype)
        return layernorm(x,
                         scale,
                         ln_bias,
                         layernorm_type=self.layernorm_type,
                         zero_centered_gamma=self.zero_centered_gamma,
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
    def get_fp8_meta_package(num_of_gemm: int) -> FP8MetaPackage:
        """
        Get the FP8 metas
        """
        fp8_max, fp8_metas_amax, fp8_metas_scale, fp8_metas_scale_inv = \
            TransformerEngineBase.get_fp8_metas(num_of_gemm)

        return FP8MetaPackage(num_of_gemm, fp8_max, fp8_metas_amax, fp8_metas_scale,
                              fp8_metas_scale_inv)


class DenseGeneral(TransformerEngineBase):
    """
    Applies a linear transformation to the incoming data :math:`y = xA^T + b`

    Parameters
    ----------
    features : Union[Iterable[int], int]
        The hidden size of each output sample.
    kernel_init : Initializer, default =
        flax.linen.initializers.variance_scaling(1.0, 'fan_in', 'truncated_normal')
        Used for initializing weights.
        It should be a callable object with three arguments (jax.random.PRNGKey, shape, dtype).
    kernel_axes : Tuple[str, ...], default = ()
        The name of axes used to shard the weights with a corresponding mesh.
    use_bias: bool, default = False
        Indicate whether to enable bias shifting.
        If set to False, the layer will not learn an additive bias.
    bias_init: Initializer, default = flax.linen.initializers.zeros
        Used for initializing bias, only used when :attr:`use_bias=True`.
        It should be a callable object with three arguments (jax.random.PRNGKey, shape, dtype).
    bias_axes: Tuple[str, ...], default = ()
        The name of axes used to shard bias with a corresponding mesh,
        only used when :attr:`use_bias=True`.
    axis:  Union[Iterable[int], int], default = -1
        An integer tuple with axes to apply the transformation on.

    Optimization parameters
    -----------------------
    dtype : jax.numpy.dtype, default  = jax.numpy.float32
        The data type used to allocate the initial parameters.
    transpose_batch_sequence : bool, default = True
        Indicate whether the input tensors were switched axis of batch
        and sequence length dimension. If set to True, the input tensors
        should be in (seqlen, batch, hidden), otherwise (batch, seqlen, hidden).
    """

    features: Union[Iterable[int], int]
    kernel_init: Initializer = None
    kernel_axes: Tuple[str, ...] = ()
    use_bias: bool = True
    bias_init: Initializer = nn.initializers.zeros
    bias_axes: Tuple[str, ...] = ()
    axis: Union[Iterable[int], int] = -1
    dtype: DType = jnp.float32
    transpose_batch_sequence: bool = False
    sharding_type = None

    def __post_init__(self):
        if self.kernel_init is None:
            self.kernel_init = nn.initializers.variance_scaling(1.0, 'fan_in', 'truncated_normal')
        super().__post_init__()

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
        warnings.warn("sharding_type of DenseGeneral would be removed in the near feature",
                      DeprecationWarning)

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
                                                   self.bias_init,
                                                   features,
                                                   jnp.float32,
                                                   axes=self.bias_axes)
            bias = bias.astype(self.dtype)
        else:
            bias = None

        contract_ind = tuple(range(0, len(axis)))
        fp8_gemm_pkg = None
        if FP8Helper.is_fp8_enabled():
            fp8_gemm_pkg = \
                    TransformerEngineBase.get_fp8_meta_package(1)

        y = type_safe_dot_general(inputs,
                                  kernel,
                                  fp8_meta_pkg=fp8_gemm_pkg,
                                  contracting_dims=(axis, contract_ind))

        if bias is not None:
            bias_shape = (1,) * (y.ndim - bias.ndim) + bias.shape
            y += jnp.reshape(bias, bias_shape)
        return y


class LayerNormDenseGeneral(TransformerEngineBase):
    r"""
    Applies layer normalization followed by linear transformation to the incoming data.

    Parameters
    ----------
    features : Union[Iterable[int], int]
        The hidden size of each output sample.
    enable_layernorm: bool, default = True
        Indicate whether to enable layer normalization before linear transformation.
    layernorm_type : {'layernorm', 'rmsnorm'}, default = 'layernorm'
        Indicate the type of layer normalization.
    epsilon : float, default = 1e-6
        A value added to the denominator of layer normalization for numerical stability.
    zero_centered_gamma : bool, default = False
        If set to `True`, the LayerNorm formula changes to

        .. math::
            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} *
            (1 + \gamma) + \beta

        This parameter is only applicable for 'layernorm'.
        The default of `scale_init` will also be changed. See `scale_init`
    scale_init : Initializer, default = None
        Used for initializing scale factors :math:`\gamma`.
        If `None` is provided, scale_init is set according to the value of zero_centered_gamma.
        If zero_centered_gamma is set to `True`, then scale_init is `flax.linen.initializers.zeros`.
        Otherwise, scale_init is `flax.linen.initializers.ones`.
        It should be a callable object with three arguments (jax.random.PRNGKey, shape, dtype).
    scale_axes : Tuple[str, ...], default = ('embed', )
        The name of axes used to shard the scale factors :math:`\gamma` with a corresponding mesh,
        only used when :attr:`enable_layernorm=True`.
    ln_bias_init: Initializer, default = flax.linen.initializers.zeros
        Used for initializing shift factors :math:`\beta`,
        only used when :attr:`enable_layernorm=True` and :attr:`layernorm_type='layernorm'`.
        It should be a callable object with three arguments (jax.random.PRNGKey, shape, dtype).
    ln_bias_axes: Tuple[str, ...], default = ('embed', )
        The name of axes used to shard the shift factors :math:`\beta` with a corresponding mesh.
        It is only used when :attr:`enable_layernorm=True` and :attr:`layernorm_type='layernorm'`.
    kernel_init : Initializer, default =
        flax.linen.initializers.variance_scaling(1.0, 'fan_in', 'truncated_normal')
        Used for initializing weights.
        It should be a callable object with three arguments (jax.random.PRNGKey, shape, dtype).
    kernel_axes : Tuple[str, ...], default = ()
        The name of axes used to shard the weights with a corresponding mesh.
    use_bias: bool, default = False
        Indicate whether to enable bias shifting.
        If set to False, the layer will not learn an additive bias.
    bias_init: Initializer, default = flax.linen.initializers.zeros
        Used for initializing bias, only used when :attr:`use_bias=True`.
        It should be a callable object with three arguments (jax.random.PRNGKey, shape, dtype).
    bias_axes: Tuple[str, ...], default = ()
        The name of axes used to shard bias with a corresponding mesh,
        only used when :attr:`use_bias=True`.
    return_layernorm_output: bool, default = True
        Indicate whether to return the output of layer normalization.
        If set False, return None as the second tensor in outputs.
    axis:  Union[Iterable[int], int], default = -1
        An integer tuple with axes to apply the transformation on.
    layernorm_input_axes: Tuple[str, ...], default = None
        Indicate the logical axes of sharding constraint to the input of layernorm, like
        (BATCH_AXES, SEQLEN_AXES, HIDDEN_AXES). Default is None, which means not to insert
        sharding constraint.
    dot_input_axes: Tuple[str, ...], default = None
        Indicate the logical axes of sharding constraint to the input of dot, like
        (BATCH_AXES, SEQLEN_AXES, HIDDEN_AXES). Default is None, which means not to insert
        sharding constraint.

    Optimization parameters
    -----------------------
    dtype : jax.numpy.dtype, default  = jax.numpy.float32
        The data type used to allocate the initial parameters.
    transpose_batch_sequence : bool, default = True
        Indicate whether the input tensors were switched axis of batch
        and sequence length dimension. If set to True, the input tensors
        should be in (seqlen, batch, hidden), otherwise (batch, seqlen, hidden).
    depth_scaling: float, default = None
        The factor to scale the output from `DenseGeneral`. It should be a float
        value or None. When None is set, then no scaling is applied.
    """

    features: Union[Iterable[int], int]
    enable_layernorm: bool = True
    layernorm_type: str = 'layernorm'
    epsilon: float = 1e-6
    zero_centered_gamma: bool = False
    scale_init: Initializer = None
    scale_axes: Tuple[str, ...] = ('embed',)
    ln_bias_init: Initializer = nn.initializers.zeros
    ln_bias_axes: Tuple[str, ...] = ('embed',)
    kernel_init: Initializer = None
    kernel_axes: Tuple[str, ...] = ()
    use_bias: bool = False
    bias_init: Initializer = nn.initializers.zeros
    bias_axes: Tuple[str, ...] = ()
    return_layernorm_output: bool = True
    axis: Union[Iterable[int], int] = -1
    dtype: DType = jnp.float32
    transpose_batch_sequence: bool = True
    layernorm_input_axes: Tuple[str, ...] = None
    dot_input_axes: Tuple[str, ...] = None
    depth_scaling: float = None
    sharding_type = None

    def __post_init__(self):
        if self.kernel_init is None:
            self.kernel_init = nn.initializers.variance_scaling(1.0, 'fan_in', 'truncated_normal')
        self.scale_init = _obtain_default_layernorm_scale_init_if_need(
            self.scale_init, self.zero_centered_gamma)
        super().__post_init__()

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
            If :attr:`return_layernorm_output=False`, then this would be None.
        """
        warnings.warn("sharding_type of LayerNormDenseGeneral would be removed in the near feature",
                      DeprecationWarning)

        ln_output = None

        fuse_layernorm = FP8Helper.is_fp8_enabled(
        ) and not self.return_layernorm_output and self.enable_layernorm

        if self.enable_layernorm:
            inputs = with_sharding_constraint_by_logical_axes(inputs, self.layernorm_input_axes)

            assert self.axis == -1    # Only support axis = =-1 at this moment
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
                              zero_centered_gamma=self.zero_centered_gamma,
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

        fp8_meta_package = None
        if FP8Helper.is_fp8_enabled():
            fp8_meta_package = \
                    TransformerEngineBase.get_fp8_meta_package(1)

        if fuse_layernorm:
            z = layernorm_fp8_dot(y,
                                  kernel,
                                  scale,
                                  ln_bias,
                                  fp8_meta_package,
                                  self.layernorm_type,
                                  zero_centered_gamma=self.zero_centered_gamma,
                                  epsilon=self.epsilon,
                                  layernorm_input_axes=self.layernorm_input_axes,
                                  dot_input_axes=self.dot_input_axes)
        else:
            y = with_sharding_constraint_by_logical_axes(y, self.dot_input_axes)
            z = type_safe_dot_general(y,
                                      kernel,
                                      fp8_meta_pkg=fp8_meta_package,
                                      contracting_dims=(axis, contract_ind))

        bias = None
        if self.use_bias:
            bias = nn_partitioning.param_with_axes('bias',
                                                   self.bias_init,
                                                   features,
                                                   jnp.float32,
                                                   axes=self.bias_axes)
            bias = bias.astype(self.dtype)

        if bias is not None:
            bias_shape = (1,) * (z.ndim - bias.ndim) + bias.shape
            z += jnp.reshape(bias, bias_shape)

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
        Intermediate size to which input samples are projected.
    enable_layernorm: bool, default = True
        Indicate whether to enable layer normalization before linear transformation.
    layernorm_type : {'layernorm', 'rmsnorm'}, default = 'layernorm'
        Indicate the type of layer normalization.
    epsilon : float, default = 1e-6
        A value added to the denominator of layer normalization for numerical stability.
    zero_centered_gamma : bool, default = False
        If set to `True`, the LayerNorm formula changes to

        .. math::
            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} *
            (1 + \gamma) + \beta

        This parameter is only applicable for 'layernorm'.
        The default of `scale_init` will also be changed. See `scale_init`.
    scale_init : Initializer, default = None
        Used for initializing scale factors :math:`\gamma`.
        If `None` is provided, scale_init is set according to the value of zero_centered_gamma.
        If zero_centered_gamma is set to `True`, then scale_init is `flax.linen.initializers.zeros`.
        Otherwise, scale_init is `flax.linen.initializers.ones`.
        It should be a callable object with three arguments (jax.random.PRNGKey, shape, dtype).
    scale_axes : Tuple[str, ...], default = ('embed', )
        The name of axes used to shard the scale factors :math:`\gamma` with a corresponding mesh,
        only used when :attr:`enable_layernorm=True`.
    ln_bias_init: Initializer, default = flax.linen.initializers.zeros
        Used for initializing shift factors :math:`\beta`,
        only used when :attr:`enable_layernorm=True` and :attr:`layernorm_type='layernorm'`.
        It should be a callable object with three arguments (jax.random.PRNGKey, shape, dtype).
    ln_bias_axes: Tuple[str, ...], default = ('embed', )
        The name of axes used to shard the shift factors :math:`\beta` with a corresponding mesh.
        Only used when :attr:`enable_layernorm=True` and :attr:`layernorm_type='layernorm'`.
    kernel_init : Initializer, default =
        flax.linen.initializers.variance_scaling(1.0, 'fan_in', 'truncated_normal')
        Used for initializing the weights of both linear transformations.
        It should be a callable object with three arguments (jax.random.PRNGKey, shape, dtype).
    kernel_axes_1 : Tuple[str, ...], default = ('embed', 'act', 'mlp')
        The name of axes used to shard the weights with a corresponding mesh for
        the weight of the first linear transformations.
    kernel_axes_2 : Tuple[str, ...], default = ('mlp', 'embed')
        The name of axes used to shard the weights with a corresponding mesh for
        the weight of the second linear transformations.
    use_bias: bool, default = False
        Indicate whether to enable bias shifting.
        If set to False, the layer will not learn an additive bias.
    bias_init: Initializer, default = flax.linen.initializers.zeros
        Used for initializing bias, only used when :attr:`use_bias=True`.
        It should be a callable object with three arguments (jax.random.PRNGKey, shape, dtype).
    bias_axes_1: Tuple[str, ...], default = ('mlp',)
        The name of axes used to shard bias with a corresponding mesh  for
        the weight of the first linear transformations.
        Only used when :attr:`use_bias=True`.
    bias_axes_2: Tuple[str, ...], default = ('embed',)
        The name of axes used to shard bias with a corresponding mesh  for
        the weight of the second linear transformations.
        Only used when :attr:`use_bias=True`.
    return_layernorm_output: bool, default = True
        Indicate whether to return the output of layer normalization.
        If set False, return None as the second tensor in outputs.
    activations: Sequence[Union[str, Callable]], default = ('relu',)
        The sequence of activation functions to apply after the first linear transformation.
        Each activation has its own transformation layer.
    intermediate_dropout_rng_name: str, default = 'dropout'
        The key in given RNGs via flax.linen.Module.apply that for generating Dropout masks.
    intermediate_dropout_rate: float, default = 0.1
        Dropout probability for the dropout op after the :attr:`activations`.
    intermediate_hidden_dropout_dims: Sequence[int], default = ()
        Dimensions that will share the same dropout mask for hidden
    axis:  Union[Iterable[int], int], default = -1
        An integer tuple with axes to apply the transformation on.
    layernorm_input_axes: Tuple[str, ...], default = None
        Indicate the logical axes of sharding constraint to the input of layernorm, like
        (BATCH_AXES, SEQLEN_AXES, HIDDEN_AXES). Default is None, which means not to insert
        sharding constraint.
    dot_1_input_axes: Tuple[str, ...], default = None
        Indicate the logical axes of sharding constraint to the input of 1st dot, like
        (BATCH_AXES, SEQLEN_AXES, HIDDEN_AXES). Default is None, which means not to insert
        sharding constraint.
    dot_2_input_axes: Tuple[str, ...], default = None
        Indicate the logical axes of sharding constraint to the input of 2nd dot, like
        (BATCH_AXES, SEQLEN_AXES, HIDDEN_AXES). Default is None, which means not to insert
        sharding constraint.

    Optimization parameters
    -----------------------
    dtype : jax.numpy.dtype, default  = jax.numpy.float32
        The data type used to allocate the initial parameters.
    transpose_batch_sequence : bool, default = True
        Indicate whether the input tensors were switched axis of batch
        and sequence length dimension. If set to True, the input tensors
        should be in (seqlen, batch, hidden), otherwise (batch, seqlen, hidden).
    """

    intermediate_dim: int = 2048
    enable_layernorm: bool = True
    layernorm_type: str = 'layernorm'
    epsilon: float = 1e-6
    zero_centered_gamma: bool = False
    scale_init: Initializer = None
    scale_axes: Tuple[str, ...] = ('embed',)
    ln_bias_init: Initializer = nn.initializers.zeros
    ln_bias_axes: Tuple[str, ...] = ('embed',)
    kernel_init: Initializer = None
    kernel_axes_1: Tuple[str, ...] = ('embed', 'act', 'mlp')
    kernel_axes_2: Tuple[str, ...] = ('mlp', 'embed')
    use_bias: bool = False
    bias_init: Initializer = nn.initializers.zeros
    bias_axes_1: Tuple[str, ...] = ('act', 'mlp')
    bias_axes_2: Tuple[str, ...] = ('embed',)
    return_layernorm_output: bool = True
    activations: Sequence[Union[str, Callable]] = ('relu',)
    intermediate_dropout_rng_name: str = 'dropout'
    intermediate_dropout_rate: float = 0.1
    intermediate_hidden_dropout_dims: Sequence[int] = ()
    axis: Union[Iterable[int], int] = -1
    dtype: DType = jnp.float32
    transpose_batch_sequence: bool = True
    layernorm_input_axes: Tuple[str, ...] = None
    dot_1_input_axes: Tuple[str, ...] = None
    dot_2_input_axes: Tuple[str, ...] = None
    major_sharding_type = None

    def __post_init__(self):
        if self.kernel_init is None:
            self.kernel_init = nn.initializers.variance_scaling(1.0, 'fan_in', 'truncated_normal')
        self.scale_init = _obtain_default_layernorm_scale_init_if_need(
            self.scale_init, self.zero_centered_gamma)
        super().__post_init__()

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
            If :attr:`return_layernorm_output=False`, then this would be None.
        """
        warnings.warn("major_sharding_type of LayerNormMLP would be removed in the near feature",
                      DeprecationWarning)

        ln_output = None

        fuse_layernorm = FP8Helper.is_fp8_enabled(
        ) and not self.return_layernorm_output and self.enable_layernorm

        def is_geglu(acts):
            geglu_act_pool = [('gelu', 'linear'), ('linear', 'gelu')]

            normalize_acts = []
            for act in acts:
                if not isinstance(act, str):
                    return False
                normalize_acts.append(act.lower())
            return tuple(normalize_acts) in geglu_act_pool

        def is_gelu(acts):
            geglu_act_pool = [('gelu',)]

            normalize_acts = []
            for act in acts:
                if not isinstance(act, str):
                    return False
                normalize_acts.append(act.lower())
            return tuple(normalize_acts) in geglu_act_pool

        use_fused_ln_geglu_mlp = fuse_layernorm \
            and (not self.use_bias) and is_geglu(self.activations) \
                and (self.intermediate_dropout_rate < 1e-3)

        use_fused_ln_gelu_mlp = fuse_layernorm \
            and self.use_bias and is_gelu(self.activations) \
                and (self.intermediate_dropout_rate < 1e-3)

        # LayerNorm
        if self.enable_layernorm:
            assert self.axis == -1    # Only support axis == -1 at this moment
            inputs = with_sharding_constraint_by_logical_axes(inputs, self.layernorm_input_axes)

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
                              zero_centered_gamma=self.zero_centered_gamma,
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
        fp8_meta_package = None
        if FP8Helper.is_fp8_enabled():
            fp8_meta_package = \
                    TransformerEngineBase.get_fp8_meta_package(num_of_gemm)

        num_activations = len(self.activations)
        axis = _canonicalize_tuple(self.axis)
        axis = _normalize_axes(axis, y.ndim)

        intermediate_dim = _canonicalize_tuple((num_activations, self.intermediate_dim))
        kernel_1_shape = tuple(y.shape[ax] for ax in axis) + intermediate_dim
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

        ffn1_ckpt_name = 'ffn1'
        ffn2_ckpt_name = 'ffn2'

        if use_fused_ln_geglu_mlp:
            assert self.axis == -1    # Only support axis = =-1 at this moment

            out = layernorm_geglu_fp8_mlp(y,
                                          scale,
                                          ln_bias, [kernel_1, kernel_2],
                                          fp8_meta_package,
                                          self.layernorm_type,
                                          zero_centered_gamma=self.zero_centered_gamma,
                                          epsilon=self.epsilon,
                                          layernorm_input_axes=self.layernorm_input_axes,
                                          dot_1_input_axes=self.dot_1_input_axes,
                                          dot_2_input_axes=self.dot_2_input_axes,
                                          ffn1_ckpt_name=ffn1_ckpt_name,
                                          ffn2_ckpt_name=ffn2_ckpt_name)
        elif use_fused_ln_gelu_mlp:
            assert self.axis == -1    # Only support axis = =-1 at this moment

            bias_1 = nn_partitioning.param_with_axes('wi_bias',
                                                     self.bias_init,
                                                     intermediate_dim,
                                                     jnp.float32,
                                                     axes=self.bias_axes_1)
            bias_1 = bias_1.astype(self.dtype)

            bias_2 = nn_partitioning.param_with_axes('wo_bias',
                                                     self.bias_init, (hidden_size,),
                                                     jnp.float32,
                                                     axes=self.bias_axes_2)
            bias_2 = bias_2.astype(self.dtype)

            out = layernorm_gelu_fp8_mlp(y,
                                         scale,
                                         ln_bias, [kernel_1, kernel_2], [bias_1, bias_2],
                                         fp8_meta_package,
                                         self.layernorm_type,
                                         zero_centered_gamma=self.zero_centered_gamma,
                                         epsilon=self.epsilon,
                                         layernorm_input_axes=self.layernorm_input_axes,
                                         dot_1_input_axes=self.dot_1_input_axes,
                                         dot_2_input_axes=self.dot_2_input_axes,
                                         ffn1_ckpt_name=ffn1_ckpt_name,
                                         ffn2_ckpt_name=ffn2_ckpt_name)
        else:    # not use_fused_ln_geglu_mlp

            # DenseGeneral 1
            gemm1_fp8_meta_package = None if fp8_meta_package is None \
                                     else fp8_meta_package.get_package_by_gemm_idx(0)
            if fuse_layernorm:
                x = layernorm_fp8_dot(y,
                                      kernel_1,
                                      scale,
                                      ln_bias,
                                      gemm1_fp8_meta_package,
                                      self.layernorm_type,
                                      zero_centered_gamma=self.zero_centered_gamma,
                                      epsilon=self.epsilon,
                                      layernorm_input_axes=self.layernorm_input_axes,
                                      dot_input_axes=self.dot_1_input_axes)
            else:
                y = with_sharding_constraint_by_logical_axes(y, self.dot_1_input_axes)
                x = type_safe_dot_general(y,
                                          kernel_1,
                                          fp8_meta_pkg=gemm1_fp8_meta_package,
                                          contracting_dims=(axis, contract_ind))

            bias = None
            if self.use_bias:
                bias = nn_partitioning.param_with_axes('wi_bias',
                                                       self.bias_init,
                                                       intermediate_dim,
                                                       jnp.float32,
                                                       axes=self.bias_axes_1)
                bias = bias.astype(self.dtype)
                bias_shape = (1,) * (x.ndim - bias.ndim) + bias.shape
                x += jnp.reshape(bias, bias_shape)

            x = checkpoint_name(x, ffn1_ckpt_name)

            activations = []
            if is_geglu(self.activations):
                z = geglu(x)
            elif is_gelu(self.activations):
                z = gelu(x)
                z = jnp.reshape(z, (*z.shape[:-2], -1))
            else:
                x = jnp.split(x, num_activations, axis=-2)
                for idx, act_fn in enumerate(self.activations):
                    x_i = _convert_to_activation_function(act_fn)(x[idx])
                    activations.append(x_i)
                z = functools.reduce(operator.mul, activations)
                z = jnp.reshape(z, (*z.shape[:-2], -1))

            z = nn.Dropout(rate=self.intermediate_dropout_rate,
                           broadcast_dims=self.intermediate_hidden_dropout_dims,
                           rng_collection=self.intermediate_dropout_rng_name)(
                               z, deterministic=deterministic)

            z = with_sharding_constraint_by_logical_axes(z, self.dot_2_input_axes)

            # DenseGeneral 2
            gemm2_fp8_meta_package = None if fp8_meta_package is None \
                                     else fp8_meta_package.get_package_by_gemm_idx(1)

            out = type_safe_dot_general(z,
                                        kernel_2,
                                        fp8_meta_pkg=gemm2_fp8_meta_package,
                                        contracting_dims=(axis, contract_ind))

            bias = None
            if self.use_bias:
                bias = nn_partitioning.param_with_axes('wo_bias',
                                                       self.bias_init, (hidden_size,),
                                                       jnp.float32,
                                                       axes=self.bias_axes_2)
                bias = bias.astype(self.dtype)
                out += jnp.reshape(bias, (1,) * (out.ndim - 1) + (-1,))

            out = checkpoint_name(out, ffn2_ckpt_name)

        return out, ln_output    # Output, layner_norm_output
