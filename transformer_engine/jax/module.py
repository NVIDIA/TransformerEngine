# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Wrapper module for Transformer related layers with FP8 support.
"""
import functools
import operator

from typing import Any, Callable, Tuple, Union, Iterable, Sequence
import numpy as np

from flax import linen as nn
from flax.linen import partitioning as nn_partitioning

import jax.numpy as jnp
from jax import lax

from .fp8 import FP8Helper
from .dot import fp8_dot
from .layernorm import layernorm, layernorm_fp8_dot
from .mlp import fp8_ln_mlp

PRNGKey = Any
Shape = Tuple[int, ...]
DType = jnp.dtype
Array = Any
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str],
                      Tuple[lax.Precision, lax.Precision]]
Initializer = Callable[[PRNGKey, Shape, DType], Array]


def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int]:
    # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
    return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


def _canonicalize_tuple(x):
    if isinstance(x, Iterable):
        return tuple(x)
    return (x, )


def _convert_to_activation_function(
        fn_or_string: Union[str, Callable]) -> Callable:
    """Convert a string to an activation function."""
    if fn_or_string == 'linear':
        return lambda x: x
    if isinstance(fn_or_string, str):
        return getattr(nn, fn_or_string)
    if callable(fn_or_string):
        return fn_or_string

    raise ValueError(
        f"don't know how to convert {fn_or_string} to an activation function")


class TransformerEngineBase(nn.Module):
    """
    Base class of transformer engine
    """

    @staticmethod
    def get_fp8_metas(num_of_gemm):
        """
        Get the FP8 metas
        """
        num_of_meta = num_of_gemm * FP8Helper.NUM_META_PER_GEMM
        axes = ('fp8_meta_axis', 'fp8_meta_history')

        fp8_metas_amax = nn_partitioning.variable_with_axes(
            FP8Helper.FP8_COLLECTION_NAME,
            FP8Helper.FP8_AMAX_NAME,
            jnp.zeros, (num_of_meta, FP8Helper.AMAX_HISTORY_SIZE),
            jnp.float32,
            axes=axes)
        fp8_metas_scale = nn_partitioning.variable_with_axes(
            FP8Helper.FP8_COLLECTION_NAME,
            FP8Helper.FP8_SCALE_NAME,
            jnp.ones, (num_of_meta, 1),
            jnp.float32,
            axes=axes)
        fp8_metas_scale_inv = nn_partitioning.variable_with_axes(
            FP8Helper.FP8_COLLECTION_NAME,
            FP8Helper.FP8_SCALE_INV_NAME,
            jnp.ones, (num_of_meta, 1),
            jnp.float32,
            axes=axes)
        fp8_max = nn_partitioning.variable_with_axes(
            FP8Helper.FP8_COLLECTION_NAME,
            FP8Helper.FP8_MAX_NAME,
            FP8Helper.generate_fp8_max_array,
            num_of_meta,
            axes=axes)

        return fp8_metas_amax, fp8_metas_scale, fp8_metas_scale_inv, fp8_max


class DenseGeneral(TransformerEngineBase):
    """
    A linear transformation to the incoming data :math:`y = xW^T + b`
    with flexible axes and FP8 support.

    Parameters
    ----------
    features : Union[Iterable[int], int]
        size of each output sample.
    kernel_init: Initializer =
        flax.linen.initializers.variance_scaling(1.0, 'fan_in', 'truncated_normal')
        used for initializing weights.
    kernel_axes: Tuple[str, ...] = ()
        The name of axes, used to sharding the kernel with a corresponding mesh.
    use_bias: bool = `False`
        Whether to enable bias shifting for QKVO projections, FC1 and FC2.
    bias_init: Initializer = flax.linen.initializers.zeros
        used for initializing bias, only works when `use_bias`=`True`.
    bias_axes: Tuple[str, ...] = ()
        The name of axes, used to sharding bias with a corresponding mesh,
        only works when `use_bias`=`True`.
    axis:  Union[Iterable[int], int] = -1
        tuple with axes to apply the transformation on.

    Optimization parameters
    -----------------------
    dtype: Any = jnp.float32
        controls the type used to allocate the initial parameters.
    transpose_batch_sequence: bool = `True`
        Whether to switch axis of batch and sequence length dimension.
        if set to `True`, then transpose inputs from (batch, seqlen, hidden)
        to (seqlen, batch, hidden)
    """
    features: Union[Iterable[int], int]
    kernel_init: Initializer = nn.initializers.variance_scaling(
        1.0, 'fan_in', 'truncated_normal')
    kernel_axes: Tuple[str, ...] = ()
    use_bias: bool = False
    bias_init: Initializer = nn.initializers.zeros
    bias_axes: Tuple[str, ...] = ()
    axis: Union[Iterable[int], int] = -1
    dtype: DType = jnp.float32
    transpose_batch_sequence: bool = True

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """
        Applies a linear transformation to the inputs along multiple dimensions.

        Parameters
        ----------
        inputs: Array
            Input tensor.

        Returns
        -------
        y: Array
            Output tensor from the linear transformation.
        """
        features = _canonicalize_tuple(self.features)
        axis = _canonicalize_tuple(self.axis)

        inputs = jnp.asarray(inputs, self.dtype)
        axis = _normalize_axes(axis, inputs.ndim)

        kernel_shape = tuple(inputs.shape[ax] for ax in axis) + features
        kernel_param_shape = (np.prod([inputs.shape[ax]
                                       for ax in axis]), np.prod(features))
        kernel = nn_partitioning.param_with_axes('kernel',
                                                 self.kernel_init,
                                                 kernel_param_shape,
                                                 jnp.float32,
                                                 axes=self.kernel_axes)

        kernel = jnp.reshape(kernel, kernel_shape)

        if self.use_bias:
            bias = nn_partitioning.param_with_axes('bias',
                                                   self.bias_init,
                                                   (self.features, ),
                                                   self.dtype,
                                                   axes=self.bias_axes)
        else:
            bias = None

        contract_ind = tuple(range(0, len(axis)))

        if FP8Helper.enable_fp8():
            fp8_metas_amax, fp8_metas_scale, fp8_metas_scale_inv, fp8_max = \
                TransformerEngineBase.get_fp8_metas(1)
            y = fp8_dot(
                inputs,
                kernel,
                fp8_max.value,
                fp8_metas_amax.value,
                fp8_metas_scale.value,
                fp8_metas_scale_inv.value,
                0,
                FP8Helper.FWD_CTYPE,
                FP8Helper.BWD_CTYPE, (axis, contract_ind),
                batch_dim_index=1 if self.transpose_batch_sequence else 0)
        else:
            kernel = jnp.asarray(kernel, self.dtype)
            y = lax.dot_general(inputs, kernel,
                                ((axis, contract_ind), ((), ())))

        if bias is not None:
            y += jnp.reshape(bias, (1, ) * (y.ndim - 1) + (-1, ))
        return y


class LayerNormDenseGeneral(TransformerEngineBase):
    """
    A layner norm + linear transformation with flexible axes and FP8 support.

    Parameters
    ----------
    features: Union[Iterable[int], int]
        size of each output sample.
    enable_layernorm: bool = `True`
        Whether to enable layer normalization before DenseGeneral.
    epsilon: float = `1e-6`
        a value added to the denominator of layer normalization
        for numerical stability, only works when `enable_layernorm`
        is True.
    scale_init: Initializer = flax.linen.initializers.ones
        used for initializing scale factors in the layer norm,
        only works when `enable_layernorm` is True.
    scale_axes: Tuple[str, ...] = ('embed', )
        The name of axes, used to sharding the scale factors with a corresponding mesh,
        only works when `enable_layernorm` is True.
    kernel_init: Initializer =
        flax.liene.initializers.variance_scaling(1.0, 'fan_in', 'truncated_normal')
        used for initializing weights.
    kernel_axes: Tuple[str, ...] = ()
        The name of axes, used to sharding the kernel with a corresponding mesh.
    use_bias: bool = `False`
        Whether to enable bias shifting for QKVO projections, FC1 and FC2.
    bias_init: Initializer = flax.liene.initializers.zeros
        used for initializing bias, only works when `use_bias`=`True`.
    bias_axes: Tuple[str, ...] = ()
        The name of axes, used to sharding bias with a corresponding mesh,
        only works when `use_bias`=`True`.
    return_layernorm_output: bool = `True`
        Whether to return the output of layer normalization. If set `False`,
        return None as the second tensor in outputs.
    axis: Union[Iterable[int], int] = -1
        tuple with axes to apply the transformation on.

    Optimization parameters
    -----------------------
    dtype: Any = jnp.float32
        controls the type used to allocate the initial parameters.
    transpose_batch_sequence: bool = `True`
        Whether to switch axis of batch and sequence length dimension.
        if set to `True`, then transpose inputs from (batch, seqlen, hidden)
        to (seqlen, batch, hidden)
    depth_scaling: float = None
        The factor to scale the output from DenseGeneral. It should be a float
        value or None. When None is set, then no scaling is applied.
    """
    features: Union[Iterable[int], int]
    enable_layernorm: bool = True
    epsilon: float = 1e-6
    scale_init: Initializer = nn.initializers.ones
    scale_axes: Tuple[str, ...] = ('embed', )
    kernel_init: Initializer = nn.initializers.variance_scaling(
        1.0, 'fan_in', 'truncated_normal')
    kernel_axes: Tuple[str, ...] = ()
    use_bias: bool = False
    bias_init: Initializer = nn.initializers.zeros
    bias_axes: Tuple[str, ...] = ()
    return_layernorm_output: bool = True
    axis: Union[Iterable[int], int] = -1
    dtype: Any = jnp.float32
    transpose_batch_sequence: bool = True
    depth_scaling: float = None

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """
        Applies layer normalization + linear transformation to the inputs
        along multiple dimensions.

        Parameters
        ----------
        inputs: Array
            Input tensor.

        Returns
        -------
        z: Array
            Output tensor from the LN + linear transformation.
        ln_output: Array
            The output tensor of layer normalization. If `return_layernorm_output`
             is `False, then this woulb be None.
        """
        ln_output = None

        fuse_layernorm = FP8Helper.enable_fp8(
        ) and not self.return_layernorm_output and self.enable_layernorm

        if self.enable_layernorm:
            features = inputs.shape[-1]
            scale = nn_partitioning.param_with_axes('scale',
                                                    self.scale_init,
                                                    (features, ),
                                                    jnp.float32,
                                                    axes=self.scale_axes)
            scale = jnp.asarray(scale, self.dtype)
            if not fuse_layernorm:
                y = layernorm(
                    inputs,
                    scale,
                    batch_dim_index=1 if self.transpose_batch_sequence else 0,
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
        kernel_param_shape = (np.prod([y.shape[ax]
                                       for ax in axis]), np.prod(features))
        kernel = nn_partitioning.param_with_axes('kernel',
                                                 self.kernel_init,
                                                 kernel_param_shape,
                                                 jnp.float32,
                                                 axes=self.kernel_axes)

        kernel = jnp.reshape(kernel, kernel_shape)

        contract_ind = tuple(range(0, len(axis)))

        if FP8Helper.enable_fp8():
            fp8_metas_amax, fp8_metas_scale, fp8_metas_scale_inv, fp8_max = \
                TransformerEngineBase.get_fp8_metas(1)
            if not fuse_layernorm:
                z = fp8_dot(
                    y,
                    kernel,
                    fp8_max.value,
                    fp8_metas_amax.value,
                    fp8_metas_scale.value,
                    fp8_metas_scale_inv.value,
                    0,
                    FP8Helper.FWD_CTYPE,
                    FP8Helper.BWD_CTYPE, (axis, contract_ind),
                    batch_dim_index=1 if self.transpose_batch_sequence else 0)
            else:
                z = layernorm_fp8_dot(
                    inputs,
                    kernel,
                    scale,
                    fp8_max.value,
                    fp8_metas_amax.value,
                    fp8_metas_scale.value,
                    fp8_metas_scale_inv.value,
                    0,
                    FP8Helper.FWD_CTYPE,
                    FP8Helper.BWD_CTYPE, (axis, contract_ind),
                    batch_dim_index=1 if self.transpose_batch_sequence else 0,
                    epsilon=self.epsilon)
        else:
            kernel = jnp.asarray(kernel, self.dtype)
            z = lax.dot_general(y, kernel, ((axis, contract_ind), ((), ())))

        bias = None
        if self.use_bias:
            bias = nn_partitioning.param_with_axes('bias',
                                                   self.bias_init,
                                                   (self.features, ),
                                                   self.dtype,
                                                   axes=self.bias_axes)

        if bias is not None:
            z += jnp.reshape(bias, (1, ) * (z.ndim - 1) + (-1, ))

        if self.depth_scaling is not None:
            z = z / self.depth_scaling

        return z, ln_output  # dense_output, layer_norm_output


class LayerNormMlpBlock(TransformerEngineBase):
    """
    A layner norm + MlpBlock transformation with flexible axes and FP8 support.

    Parameters
    ----------
    intermediate_dim: int = 2048
        intermediate size to which input samples are projected.
    enable_layernorm: bool = `True`
        Whether to enable layer normalization before DenseGeneral.
    epsilon: float = `1e-6`
        a value added to the denominator of layer normalization
        for numerical stability, only works when `enable_layernorm`
        is True.
    scale_init: Initializer = flax.linen.initializers.ones
        used for initializing scale factors in the layer norm,
        only works when `enable_layernorm` is True.
    scale_axes: Tuple[str, ...] = ('embed', )
        The name of axes, used to sharding the scale factors with a corresponding mesh,
        only works when `enable_layernorm` is True.
    kernel_init: Initializer =
        flax.linen.initializers.variance_scaling(1.0, 'fan_in', 'truncated_normal')
        used for initializing weights.
    kernel_axes_1: Tuple[str, ...] = ('embed', 'mlp')
    kernel_axes_2: Tuple[str, ...] = ('mlp', 'embed')
    use_bias: bool = `False`
        The name of axes, used to sharding the kernel with a corresponding mesh.
    bias_init: Initializer = flax.linen.initializers.zeros
        Whether to enable bias shifting for QKVO projections, FC1 and FC2.
    bias_axes: Tuple[str, ...] = ()
        used for initializing bias, only works when `use_bias`=`True`.
    return_layernorm_output: bool = True
        Whether to return the output of layer normalization. If set `False`,
        return None as the second tensor in outputs.
    activations: Sequence[Union[str, Callable]] = ('relu', )
        the sequence of activation functions to apply after FC1. Each activation has
        its own FC1 layer.
    intermediate_dropout_rate: float = 0.1
        dropout probability for the dropout op after FC1 layer.
    axis: Union[Iterable[int], int] = -1
        tuple with axes to apply the transformation on.

    Optimization parameters
    -----------------------
    dtype: Any = jnp.float32
        controls the type used to allocate the initial parameters.
    transpose_batch_sequence: bool = `True`
        Whether to switch axis of batch and sequence length dimension.
        if set to `True`, then transpose inputs from (batch, seqlen, hidden)
        to (seqlen, batch, hidden)
    """
    intermediate_dim: int = 2048
    enable_layernorm: bool = True
    epsilon: float = 1e-6
    scale_init: Initializer = nn.initializers.ones
    scale_axes: Tuple[str, ...] = ('embed', )
    kernel_init: Initializer = nn.initializers.variance_scaling(
        1.0, 'fan_in', 'truncated_normal')
    kernel_axes_1: Tuple[str, ...] = ('embed', 'mlp')
    kernel_axes_2: Tuple[str, ...] = ('mlp', 'embed')
    use_bias: bool = False
    bias_init: Initializer = nn.initializers.zeros
    bias_axes: Tuple[str, ...] = ()
    return_layernorm_output: bool = True
    activations: Sequence[Union[str, Callable]] = ('relu', )
    intermediate_dropout_rate: float = 0.1
    axis: Union[Iterable[int], int] = -1
    dtype: Any = jnp.float32
    transpose_batch_sequence: bool = True
    fuse_wi: bool = True

    @nn.compact
    def __call__(self, inputs: Array, deterministic: bool = False) -> Array:
        """
        Applies layer normalization + MlpBlock transformation to the inputs
        along multiple dimensions.

        Parameters
        ----------
        inputs: Array
            Input tensor.
        deterministic: bool = `False`
            Disables dropout layers if set to True.

        Returns
        -------
        out: Array
            Output tensor from the LN + MlpBlock transformation.
        ln_output: Array
            The output tensor of layer normalization. If `return_layernorm_output`
             is `False, then this woulb be None.
        """
        ln_output = None

        fuse_wi = self.fuse_wi or len(self.activations) == 1

        fuse_layernorm = FP8Helper.enable_fp8(
        ) and not self.return_layernorm_output and self.enable_layernorm \
            and fuse_wi

        use_fused_ln_mlp = fuse_layernorm \
            and (not self.use_bias) and self.activations == ('gelu', 'linear') \
                and (self.intermediate_dropout_rate < 1e-3)

        # LayerNorm
        if self.enable_layernorm:
            features = inputs.shape[-1]
            scale = nn_partitioning.param_with_axes('scale',
                                                    self.scale_init,
                                                    (features, ),
                                                    jnp.float32,
                                                    axes=self.scale_axes)
            scale = jnp.asarray(scale, self.dtype)
            if not fuse_layernorm:
                y = layernorm(inputs, scale, self.epsilon)
            else:
                assert not self.return_layernorm_output
                y = inputs
        else:
            y = inputs

        if self.return_layernorm_output:
            ln_output = y

        if FP8Helper.enable_fp8():
            if fuse_wi:
                num_of_gemm = 2
            else:
                num_of_gemm = len(
                    self.activations) + 1  # 1 for 2-nd DenseGeneral_2
            fp8_metas_amax, fp8_metas_scale, fp8_metas_scale_inv, fp8_max = \
                TransformerEngineBase.get_fp8_metas(num_of_gemm)

        if use_fused_ln_mlp:
            num_activations = len(self.activations)
            axis = _canonicalize_tuple(self.axis)
            axis = _normalize_axes(axis, inputs.ndim)

            intermediate_dim = _canonicalize_tuple(
                self.intermediate_dim * num_activations, )
            kernel_1_shape = tuple(inputs.shape[ax]
                                   for ax in axis) + intermediate_dim
            kernel_1_param_shape = (np.prod([y.shape[ax] for ax in axis]),
                                    np.prod(intermediate_dim))
            kernel_1 = nn_partitioning.param_with_axes('wi_kernel',
                                                       self.kernel_init,
                                                       kernel_1_param_shape,
                                                       jnp.float32,
                                                       axes=self.kernel_axes_1)
            kernel_1 = jnp.reshape(kernel_1, kernel_1_shape)
            hidden_size = inputs.shape[-1]
            hidden_size_tuple = _canonicalize_tuple(hidden_size)
            kernel_2_shape = (self.intermediate_dim, ) + hidden_size_tuple
            kernel_2_param_shape = (self.intermediate_dim,
                                    np.prod(hidden_size_tuple))
            kernel_2 = nn_partitioning.param_with_axes('wo_kernel',
                                                       self.kernel_init,
                                                       kernel_2_param_shape,
                                                       jnp.float32,
                                                       axes=self.kernel_axes_2)
            kernel_2 = jnp.reshape(kernel_2, kernel_2_shape)
            contract_ind = tuple(range(0, len(axis)))
            out = fp8_ln_mlp(
                inputs,
                scale,
                kernel_1,
                kernel_2,
                fp8_max.value,
                fp8_metas_amax.value,
                fp8_metas_scale.value,
                fp8_metas_scale_inv.value,
                0,
                FP8Helper.FWD_CTYPE,
                FP8Helper.BWD_CTYPE,
                epsilon=self.epsilon,
                contracting_dims=(axis, contract_ind),
                batch_dim_index=1 if self.transpose_batch_sequence else 0,
                activations=self.activations)
        else:
            # DenseGeneral 1
            activations = []
            if fuse_wi:
                dense_name = 'wi'
                num_activations = len(self.activations)
                axis = _canonicalize_tuple(self.axis)
                axis = _normalize_axes(axis, y.ndim)

                intermediate_dim = _canonicalize_tuple(
                    self.intermediate_dim * num_activations, )
                kernel_shape = tuple(y.shape[ax]
                                     for ax in axis) + intermediate_dim
                kernel_param_shape = (np.prod([y.shape[ax] for ax in axis]),
                                      np.prod(intermediate_dim))
                kernel = nn_partitioning.param_with_axes(
                    f'{dense_name}_kernel',
                    self.kernel_init,
                    kernel_param_shape,
                    jnp.float32,
                    axes=self.kernel_axes_1)
                kernel = jnp.reshape(kernel, kernel_shape)
                contract_ind = tuple(range(0, len(axis)))
                if FP8Helper.enable_fp8():
                    if not fuse_layernorm:
                        x = fp8_dot(
                            y,
                            kernel,
                            fp8_max.value[:FP8Helper.NUM_META_PER_GEMM, :],
                            fp8_metas_amax.value[:FP8Helper.
                                                 NUM_META_PER_GEMM, :],
                            fp8_metas_scale.value[:FP8Helper.
                                                  NUM_META_PER_GEMM, :],
                            fp8_metas_scale_inv.value[:FP8Helper.
                                                      NUM_META_PER_GEMM, :],
                            0,
                            FP8Helper.FWD_CTYPE,
                            FP8Helper.BWD_CTYPE, (axis, contract_ind),
                            batch_dim_index=1
                            if self.transpose_batch_sequence else 0)
                    else:
                        x = layernorm_fp8_dot(
                            inputs,
                            kernel,
                            scale,
                            fp8_max.value[:FP8Helper.NUM_META_PER_GEMM, :],
                            fp8_metas_amax.value[:FP8Helper.
                                                 NUM_META_PER_GEMM, :],
                            fp8_metas_scale.value[:FP8Helper.
                                                  NUM_META_PER_GEMM, :],
                            fp8_metas_scale_inv.value[:FP8Helper.
                                                      NUM_META_PER_GEMM, :],
                            0,
                            FP8Helper.FWD_CTYPE,
                            FP8Helper.BWD_CTYPE, (axis, contract_ind),
                            batch_dim_index=1
                            if self.transpose_batch_sequence else 0,
                            epsilon=self.epsilon)
                else:
                    kernel = jnp.asarray(kernel, self.dtype)
                    x = lax.dot_general(y, kernel,
                                        ((axis, contract_ind), ((), ())))
                x = jnp.split(x, num_activations, axis=-1)
                for idx, act_fn in enumerate(self.activations):
                    x_i = _convert_to_activation_function(act_fn)(x[idx])
                    activations.append(x_i)
            else:
                for idx, act_fn in enumerate(self.activations):
                    dense_name = 'wi' if len(
                        self.activations) == 1 else f'wi_{idx}'

                    axis = _canonicalize_tuple(self.axis)
                    axis = _normalize_axes(axis, y.ndim)

                    intermediate_dim = _canonicalize_tuple(
                        self.intermediate_dim, )
                    kernel_shape = tuple(y.shape[ax]
                                         for ax in axis) + intermediate_dim
                    kernel_param_shape = (np.prod([y.shape[ax]
                                                   for ax in axis]),
                                          np.prod(intermediate_dim))
                    kernel = nn_partitioning.param_with_axes(
                        f'{dense_name}_kernel',
                        self.kernel_init,
                        kernel_param_shape,
                        jnp.float32,
                        axes=self.kernel_axes_1)
                    kernel = jnp.reshape(kernel, kernel_shape)

                    bias = None
                    if self.use_bias:
                        bias = nn_partitioning.param_with_axes(
                            f'{dense_name}_bias',
                            self.bias_init, (self.intermediate_dim, ),
                            self.dtype,
                            axes=self.bias_axes)

                    contract_ind = tuple(range(0, len(axis)))

                    if FP8Helper.enable_fp8():
                        fp8_meta_start = idx * FP8Helper.NUM_META_PER_GEMM
                        fp8_meta_end = fp8_meta_start + FP8Helper.NUM_META_PER_GEMM
                        z = fp8_dot(
                            y,
                            kernel,
                            fp8_max.value[fp8_meta_start:fp8_meta_end, :],
                            fp8_metas_amax.value[
                                fp8_meta_start:fp8_meta_end, :],
                            fp8_metas_scale.value[
                                fp8_meta_start:fp8_meta_end, :],
                            fp8_metas_scale_inv.value[
                                fp8_meta_start:fp8_meta_end, :],
                            0,
                            FP8Helper.FWD_CTYPE,
                            FP8Helper.BWD_CTYPE, (axis, contract_ind),
                            batch_dim_index=1
                            if self.transpose_batch_sequence else 0)
                    else:
                        kernel = jnp.asarray(kernel, self.dtype)
                        z = lax.dot_general(y, kernel,
                                            ((axis, contract_ind), ((), ())))

                    if bias is not None:
                        z += jnp.reshape(bias, (1, ) * (z.ndim - 1) + (-1, ))

                    z = _convert_to_activation_function(act_fn)(z)
                    activations.append(z)

            z = functools.reduce(operator.mul, activations)

            z = nn.Dropout(
                rate=self.intermediate_dropout_rate, broadcast_dims=(-2, ))(
                    z, deterministic=deterministic)  # Broadcast along length.
            if self.transpose_batch_sequence:
                z = nn_partitioning.with_sharding_constraint(
                    z, ('length', 'batch', 'mlp'))
            else:
                z = nn_partitioning.with_sharding_constraint(
                    z, ('batch', 'length', 'mlp'))

            # DenseGeneral 2
            hidden_size = inputs.shape[-1]
            hidden_size_tuple = _canonicalize_tuple(hidden_size)
            axis = _canonicalize_tuple(self.axis)
            axis = _normalize_axes(axis, z.ndim)

            kernel_shape = tuple(z.shape[ax]
                                 for ax in axis) + hidden_size_tuple
            kernel_param_shape = (np.prod([z.shape[ax] for ax in axis]),
                                  np.prod(hidden_size_tuple))
            kernel = nn_partitioning.param_with_axes('wo_kernel',
                                                     self.kernel_init,
                                                     kernel_param_shape,
                                                     jnp.float32,
                                                     axes=self.kernel_axes_2)
            kernel = jnp.reshape(kernel, kernel_shape)

            bias = None
            if self.use_bias:
                bias = nn_partitioning.param_with_axes('wo_bias',
                                                       self.bias_init,
                                                       (hidden_size, ),
                                                       self.dtype,
                                                       axes=self.bias_axes)

            contract_ind = tuple(range(0, len(axis)))

            if FP8Helper.enable_fp8():
                out = fp8_dot(
                    z,
                    kernel,
                    fp8_max.value[-FP8Helper.NUM_META_PER_GEMM:, :],
                    fp8_metas_amax.value[-FP8Helper.NUM_META_PER_GEMM:, :],
                    fp8_metas_scale.value[-FP8Helper.NUM_META_PER_GEMM:, :],
                    fp8_metas_scale_inv.value[
                        -FP8Helper.NUM_META_PER_GEMM:, :],
                    0,
                    FP8Helper.FWD_CTYPE,
                    FP8Helper.BWD_CTYPE, (axis, contract_ind),
                    batch_dim_index=1 if self.transpose_batch_sequence else 0)
            else:
                kernel = jnp.asarray(kernel, self.dtype)
                out = lax.dot_general(z, kernel,
                                      ((axis, contract_ind), ((), ())))

            if bias is not None:
                out += jnp.reshape(bias, (1, ) * (out.ndim - 1) + (-1, ))

        return out, ln_output  # Output, layner_norm_output
