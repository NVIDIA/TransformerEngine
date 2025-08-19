# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Wrapper module for Transformer related layers with FP8 support.
"""
from functools import reduce
import operator
from typing import Any, Callable, Iterable, List, Sequence, Tuple, Union, NewType

import numpy as np
import jax.numpy as jnp
from flax import linen as nn
from jax import lax
from jax import random as jax_random
from jax.ad_checkpoint import checkpoint_name

from transformer_engine.common import recipe

from ..dense import dense

from ..layernorm import canonicalize_norm_type
from ..layernorm import layernorm
from ..layernorm_dense import layernorm_dense
from ..layernorm_mlp import layernorm_mlp
from ..activation import activation
from ..softmax import softmax, SoftmaxType
from ..sharding import with_sharding_constraint_by_logical_axes
from ..cpp_extensions import (
    is_softmax_kernel_available,
    jax_scaled_softmax,
    jax_scaled_masked_softmax,
    jax_scaled_upper_triang_masked_softmax,
)
from ..quantize import QuantizerFactory, QuantizeConfig, QuantizeMeta, QuantizeMetaSet, ScalingMode

PRNGKey = Any
Shape = Tuple[int, ...]
DType = NewType("DType", jnp.dtype)
Array = NewType("Array", jnp.ndarray)
PrecisionLike = Union[
    None, str, lax.Precision, Tuple[str, str], Tuple[lax.Precision, lax.Precision]
]
Initializer = Callable[[PRNGKey, Shape, DType], Array]


def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int]:
    # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
    return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


def _canonicalize_tuple(x):
    if isinstance(x, Iterable):
        return tuple(x)
    return (x,)


def _obtain_default_layernorm_scale_init_if_need(original_init, zero_centered_gamma):
    if original_init is not None:
        return original_init

    if not zero_centered_gamma:
        return nn.initializers.ones
    return nn.initializers.zeros


def _create_layernorm_parameters(
    module,
    norm_type,
    shape,
    scale_init,
    scale_axes,
    bias_init,
    bias_axes,
    input_dtype,
    dtype,
):
    scale = module.param(
        "scale",
        nn.with_logical_partitioning(scale_init, scale_axes),
        shape,
        dtype,
    ).astype(input_dtype)

    norm_type = canonicalize_norm_type(norm_type)
    if norm_type == "layernorm":
        bias = module.param(
            "ln_bias",
            nn.with_logical_partitioning(bias_init, bias_axes),
            shape,
            dtype,
        ).astype(input_dtype)
    else:
        assert norm_type == "rmsnorm"
        bias = None

    return scale, bias


def _convert_to_activation_function(fn_or_string: Union[str, Callable]) -> Callable:
    """Convert a string to an activation function."""
    if fn_or_string == "linear":
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
    assert all(
        map(lambda x: x.ndim == masks[0].ndim, masks)
    ), f"masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}"
    mask, *other_masks = masks
    for other_mask in other_masks:
        mask = mask + other_mask
    return mask


def _apply_low_rank_adaptation(x, axis, features, lora_a_kernel, lora_b_kernel, alpha):
    """Low Rank Adaptation Implementation"""

    assert len(axis) <= 5
    hidden_in_names = "ijklm"[: len(axis)]
    assert len(features) <= 5
    hidden_out_names = "nopqr"[: len(features)]
    rank_name = "s"

    assert lora_a_kernel.shape[-1] == lora_b_kernel.shape[-2]
    rank = lora_a_kernel.shape[-1]
    scaling = alpha / rank if alpha is not None else 1.0

    x_einsum_express = f"...{hidden_in_names}"
    lora_a_einsum_express = f"{hidden_in_names}{hidden_out_names[:-1]}{rank_name}"
    lora_b_einsum_express = f"{hidden_out_names[:-1]}{rank_name}{hidden_out_names[-1]}"
    output_einsum_express = f"...{hidden_out_names}"
    final_einsum_express = (
        f"{x_einsum_express},{lora_a_einsum_express},{lora_b_einsum_express}"
        f"->{output_einsum_express}"
    )

    output = jnp.einsum(final_einsum_express, x, lora_a_kernel, lora_b_kernel)
    output = output * scaling
    return output


class Softmax(nn.Module):  # pylint: disable=too-few-public-methods
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
        input_dtype = inputs.dtype
        logits = inputs

        # use primitives
        if is_softmax_kernel_available(
            self.softmax_type, batch, heads, q_seqlen, k_seqlen, input_dtype
        ):
            if bias is not None:
                logits = logits + bias.astype(input_dtype)

            mask_ = mask
            if self.softmax_type is not SoftmaxType.SCALED_MASKED:
                mask_ = None

            outputs = softmax(logits, mask_, self.scale_factor, self.softmax_type)
        # use default jax based implementation
        else:
            if bias is not None:
                logits = logits + bias.astype(input_dtype)

            if self.softmax_type is SoftmaxType.SCALED:
                outputs = jax_scaled_softmax(logits, self.scale_factor)
            elif self.softmax_type is SoftmaxType.SCALED_MASKED:
                outputs = jax_scaled_masked_softmax(logits, mask, self.scale_factor)
            elif self.softmax_type is SoftmaxType.SCALED_UPPER_TRIANG_MASKED:
                outputs = jax_scaled_upper_triang_masked_softmax(logits, self.scale_factor)
            else:
                raise ValueError(
                    f"Unsupported softmax type: {self.softmax_type}. softmax_type must be [SCALED,"
                    " SCALED_MASKED, SCALED_UPPER_TRIANG_MASKED]"
                )
        assert input_dtype == outputs.dtype
        return outputs


class LayerNorm(nn.Module):  # pylint: disable=too-few-public-methods
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
    dtype: jax.numpy.dtype, default  = jax.numpy.float32
        The data type used to allocate the initial parameters.
    """

    epsilon: float = 1e-6
    layernorm_type: str = "layernorm"
    zero_centered_gamma: bool = False
    scale_init: Initializer = None
    scale_axes: Tuple[str, ...] = ("embed",)
    bias_init: Initializer = nn.initializers.zeros
    bias_axes: Tuple[str, ...] = ("embed",)
    dtype: DType = jnp.float32

    def __post_init__(self):
        self.scale_init = _obtain_default_layernorm_scale_init_if_need(
            self.scale_init,
            self.zero_centered_gamma,
        )
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
        input_dtype = x.dtype

        features = x.shape[-1]
        scale, ln_bias = _create_layernorm_parameters(
            self,
            self.layernorm_type,
            (features,),
            self.scale_init,
            self.scale_axes,
            self.bias_init,
            self.bias_axes,
            input_dtype,
            self.dtype,
        )
        out = layernorm(
            x,
            scale,
            ln_bias,
            norm_type=self.layernorm_type,
            zero_centered_gamma=self.zero_centered_gamma,
            epsilon=self.epsilon,
        )
        assert out.dtype == input_dtype
        return out


class TransformerEngineBase(nn.Module):  # pylint: disable=too-few-public-methods
    """
    Base class of transformer engine
    """

    def generate_quantizer_set(
        self, postfix: str = "", variable_collection: str = None, fp8_recipe=None
    ):
        """
        Generate a set of FP8 meta for a GEMM.
        """

        def generate_quantize_meta(quantizer_name: str):
            collection_name = (
                variable_collection
                if variable_collection is not None
                else QuantizeConfig.COLLECTION_NAME
            )
            scale = self.variable(
                collection_name,
                f"{quantizer_name}{postfix}_scale",
                jnp.ones,
                (1,),
                jnp.float32,
            ).value
            amax_history = self.variable(
                collection_name,
                f"{quantizer_name}{postfix}_amax_history",
                jnp.zeros,
                (QuantizeConfig.AMAX_HISTORY_LEN,),
                jnp.float32,
            ).value
            return QuantizeMeta(scale=scale, amax_history=amax_history)

        if QuantizeConfig.SCALING_MODE == ScalingMode.DELAYED_TENSOR_SCALING or isinstance(
            fp8_recipe, recipe.DelayedScaling
        ):
            x_meta = generate_quantize_meta("x")
            kernel_meta = generate_quantize_meta("kernel")
            grad_meta = generate_quantize_meta("grad")
            quantize_meta_set = QuantizeMetaSet(x=x_meta, kernel=kernel_meta, grad=grad_meta)
            kwargs = {"quantize_meta_set": quantize_meta_set}
        else:
            kwargs = {}

        quantizer_set = QuantizerFactory.create_set(fp8_recipe=fp8_recipe, **kwargs)
        return quantizer_set


class DenseGeneral(TransformerEngineBase):
    r"""
    Applies a dense layer transformation to the incoming data :math:`y = xA^T + b`.

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
    enable_low_rank_adaptation: bool, default = False
        Indicate whether to enable low rank adaptation for each dense layer.
    low_rank_adaptation_dim: int, default = 32
        The dimension for low rank adaptation, only used when
        :attr:`enable_low_rank_adaptation=True`
    low_rank_adaptation_alpha: float, default = None
        The alpha for computing the scaling factor of LoRA output.
        :math:`\frac{alpha}{rank} * lora_output`. None means no scaling.
    axis:  Union[Iterable[int], int], default = -1
        An integer tuple with axes to apply the transformation on.
    input_axes: Tuple[str, ...], default = None
        Indicate the logical axes of sharding constraint to the input, like
        (BATCH_AXES, SEQLEN_AXES, HIDDEN_AXES). Default is None, which means not to insert
        sharding constraint.

    Optimization parameters
    -----------------------
    dtype: jax.numpy.dtype, default  = jax.numpy.float32
        The data type used to allocate the initial parameters.
    """

    features: Union[Iterable[int], int]
    kernel_init: Initializer = None
    kernel_axes: Tuple[str, ...] = ()
    use_bias: bool = True
    bias_init: Initializer = nn.initializers.zeros
    bias_axes: Tuple[str, ...] = ()
    enable_low_rank_adaptation: bool = False
    low_rank_adaptation_dim: int = 32
    low_rank_adaptation_alpha: float = None
    axis: Union[Iterable[int], int] = -1
    dtype: DType = jnp.float32
    input_axes: Tuple[str, ...] = ()

    def __post_init__(self):
        if self.kernel_init is None:
            self.kernel_init = nn.initializers.variance_scaling(
                1.0, "fan_in", "truncated_normal", dtype=self.dtype
            )
        super().__post_init__()

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """
        Apply the dense layer transformation to the input.

        Parameters
        ----------
        inputs : jax.numpy.ndarray
            Input tensors.

        Returns
        -------
        outputs : jax.numpy.ndarray
            Output tensors.
        """

        input_dtype = inputs.dtype
        features = _canonicalize_tuple(self.features)
        axis = _canonicalize_tuple(self.axis)

        axis = _normalize_axes(axis, inputs.ndim)

        kernel_shape = tuple(inputs.shape[ax] for ax in axis) + features

        if self.kernel_axes:
            assert len(kernel_shape) == len(self.kernel_axes), (
                "Expected len(kernel_shape) to match len(kernel_axes),"
                f"got kernel_shape {kernel_shape} and kernel_axes {self.kernel_axes}"
            )
        kernel = self.param(
            "kernel",
            nn.with_logical_partitioning(self.kernel_init, self.kernel_axes),
            kernel_shape,
            self.dtype,
        )

        if not QuantizeConfig.is_fp8_enabled():
            kernel = kernel.astype(input_dtype)

        if self.use_bias:
            bias = self.param(
                "bias",
                nn.with_logical_partitioning(self.bias_init, self.bias_axes),
                features,
                self.dtype,
            ).astype(input_dtype)
        else:
            bias = None

        quantizer_set = self.generate_quantizer_set()
        contract_ind = tuple(range(0, len(axis)))
        y = dense(
            inputs,
            kernel,
            contracting_dims=(axis, contract_ind),
            input_axes=self.input_axes,
            kernel_axes=self.kernel_axes,
            quantizer_set=quantizer_set,
        )

        if self.enable_low_rank_adaptation:
            lora_a_kernel_shape = (
                *kernel_shape[: len(axis)],
                *features[:-1],
                self.low_rank_adaptation_dim,
            )
            lora_a_kernel_axes = (None,) * len(lora_a_kernel_shape)
            lora_a_kernel = self.param(
                "lora_a_kernel",
                nn.with_logical_partitioning(self.kernel_init, lora_a_kernel_axes),
                lora_a_kernel_shape,
                self.dtype,
            ).astype(input_dtype)

            lora_b_kernel_shape = (*features[:-1], self.low_rank_adaptation_dim, features[-1])
            lora_b_kernel_axes = (None,) * len(lora_b_kernel_shape)
            lora_b_kernel = self.param(
                "lora_b_kernel",
                nn.with_logical_partitioning(nn.initializers.zeros, lora_b_kernel_axes),
                lora_b_kernel_shape,
                self.dtype,
            ).astype(input_dtype)

            y += _apply_low_rank_adaptation(
                inputs, axis, features, lora_a_kernel, lora_b_kernel, self.low_rank_adaptation_alpha
            )

        if bias is not None:
            bias_shape = (1,) * (y.ndim - bias.ndim) + bias.shape
            y += jnp.reshape(bias, bias_shape)

        assert y.dtype == input_dtype
        return y


class LayerNormDenseGeneral(TransformerEngineBase):
    r"""
    Applies layer normalization followed by dense layer transformation to the incoming data.

    Parameters
    ----------
    features : Union[Iterable[int], int]
        The hidden size of each output sample.
    enable_layernorm: bool, default = True
        Indicate whether to enable layer normalization before dense layer transformation.
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
    enable_low_rank_adaptation: bool, default = False
        Indicate whether to enable low rank adaptation for each dense layer.
    low_rank_adaptation_dim: int, default = 32
        The dimension for low rank adaptation, only used when
        :attr:`enable_low_rank_adaptation=True`
    low_rank_adaptation_alpha: float, default = None
        The alpha for computing the scaling factor of LoRA output.
        :math:`\frac{alpha}{rank} * lora_output`. None means no scaling.
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
    dtype: jax.numpy.dtype, default  = jax.numpy.float32
        The data type used to allocate the initial parameters.
    depth_scaling: float, default = None
        The factor to scale the output from `DenseGeneral`. It should be a float
        value or None. When None is set, then no scaling is applied.
    """

    features: Union[Iterable[int], int]
    enable_layernorm: bool = True
    layernorm_type: str = "layernorm"
    epsilon: float = 1e-6
    zero_centered_gamma: bool = False
    scale_init: Initializer = None
    scale_axes: Tuple[str, ...] = ("embed",)
    ln_bias_init: Initializer = nn.initializers.zeros
    ln_bias_axes: Tuple[str, ...] = ("embed",)
    kernel_init: Initializer = None
    kernel_axes: Tuple[str, ...] = ()
    use_bias: bool = False
    bias_init: Initializer = nn.initializers.zeros
    bias_axes: Tuple[str, ...] = ()
    return_layernorm_output: bool = True
    enable_low_rank_adaptation: bool = False
    low_rank_adaptation_dim: int = 32
    low_rank_adaptation_alpha: float = None
    axis: Union[Iterable[int], int] = -1
    dtype: DType = jnp.float32
    layernorm_input_axes: Tuple[str, ...] = None
    dot_input_axes: Tuple[str, ...] = None
    depth_scaling: float = None

    def __post_init__(self):
        if self.kernel_init is None:
            self.kernel_init = nn.initializers.variance_scaling(
                1.0,
                "fan_in",
                "truncated_normal",
                dtype=self.dtype,
            )
        self.scale_init = _obtain_default_layernorm_scale_init_if_need(
            self.scale_init,
            self.zero_centered_gamma,
        )
        self.quantizer_set = QuantizerFactory.create_set()
        super().__post_init__()

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """
        Apply layer normalization to the input followed by a dense layer transformation.

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
        assert self.axis == -1, "Only support axis = =-1 at this moment"

        input_dtype = inputs.dtype
        ln_output = None

        quantizer_set = self.generate_quantizer_set()

        fuse_layernorm = (
            QuantizeConfig.is_fp8_enabled()
            and not self.return_layernorm_output
            and self.enable_layernorm
        )

        if self.enable_layernorm:
            inputs = with_sharding_constraint_by_logical_axes(inputs, self.layernorm_input_axes)
            features = inputs.shape[-1]
            scale, ln_bias = _create_layernorm_parameters(
                self,
                self.layernorm_type,
                (features,),
                self.scale_init,
                self.scale_axes,
                self.ln_bias_init,
                self.ln_bias_axes,
                input_dtype,
                self.dtype,
            )

            if not fuse_layernorm:
                y = layernorm(
                    inputs,
                    scale,
                    ln_bias,
                    norm_type=self.layernorm_type,
                    zero_centered_gamma=self.zero_centered_gamma,
                    epsilon=self.epsilon,
                )
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

        kernel_shape = (np.prod([inputs.shape[ax] for ax in axis]),) + features
        kernel = self.param(
            "kernel",
            nn.with_logical_partitioning(self.kernel_init, self.kernel_axes),
            kernel_shape,
            self.dtype,
        )
        if not QuantizeConfig.is_fp8_enabled():
            kernel = kernel.astype(input_dtype)

        contract_ind = tuple(range(0, len(axis)))

        if fuse_layernorm:
            z = layernorm_dense(
                y,
                kernel,
                scale,
                ln_bias,
                norm_type=self.layernorm_type,
                zero_centered_gamma=self.zero_centered_gamma,
                epsilon=self.epsilon,
                layernorm_input_axes=self.layernorm_input_axes,
                dot_input_axes=self.dot_input_axes,
                kernel_axes=self.kernel_axes,
                quantizer_set=quantizer_set,
            )
        else:
            y = with_sharding_constraint_by_logical_axes(y, self.dot_input_axes)
            z = dense(
                y,
                kernel,
                contracting_dims=(axis, contract_ind),
                input_axes=self.dot_input_axes,
                kernel_axes=self.kernel_axes,
                quantizer_set=quantizer_set,
            )

        if self.enable_low_rank_adaptation:
            lora_a_kernel_shape = (
                *kernel_shape[: len(axis)],
                *features[:-1],
                self.low_rank_adaptation_dim,
            )
            lora_a_kernel_axes = (None,) * len(lora_a_kernel_shape)
            lora_a_kernel = self.param(
                "lora_a_kernel",
                nn.with_logical_partitioning(self.kernel_init, lora_a_kernel_axes),
                lora_a_kernel_shape,
                self.dtype,
            ).astype(input_dtype)

            lora_b_kernel_shape = (*features[:-1], self.low_rank_adaptation_dim, features[-1])
            lora_b_kernel_axes = (None,) * len(lora_b_kernel_shape)
            lora_b_kernel = self.param(
                "lora_b_kernel",
                nn.with_logical_partitioning(nn.initializers.zeros, lora_b_kernel_axes),
                lora_b_kernel_shape,
                self.dtype,
            ).astype(input_dtype)

            z += _apply_low_rank_adaptation(
                y, axis, features, lora_a_kernel, lora_b_kernel, self.low_rank_adaptation_alpha
            )

        bias = None
        if self.use_bias:
            bias = self.param(
                "bias",
                nn.with_logical_partitioning(self.bias_init, self.bias_axes),
                features,
                self.dtype,
            ).astype(input_dtype)

        if bias is not None:
            bias_shape = (1,) * (z.ndim - bias.ndim) + bias.shape
            z += jnp.reshape(bias, bias_shape)

        if self.depth_scaling is not None:
            z = z / self.depth_scaling

        assert z.dtype == input_dtype, f"output_dtype={z.dtype}, input_dtype={input_dtype}"
        # z = z.reshape(*inputs.shape[: self.axis], *features)
        return z, ln_output  # dense_output, layer_norm_output


class LayerNormMLP(TransformerEngineBase):
    r"""
    Applies layer normalization on the input followed by the MLP module,
    consisting of 2 successive dense layer transformations, separated by given activations.

    Parameters
    ----------
    intermediate_dim: int, default = 2048
        Intermediate size to which input samples are projected.
    enable_layernorm: bool, default = True
        Indicate whether to enable layer normalization before dense layer transformation.
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
        Used for initializing the weights of both dense layer transformations.
        It should be a callable object with three arguments (jax.random.PRNGKey, shape, dtype).
    kernel_axes_1 : Tuple[str, ...], default = ('embed', 'act', 'mlp')
        The name of axes used to shard the weights with a corresponding mesh for
        the weight of the first dense layer transformation.
    kernel_axes_2 : Tuple[str, ...], default = ('mlp', 'embed')
        The name of axes used to shard the weights with a corresponding mesh for
        the weight of the second dense layer transformation.
    use_bias: bool, default = False
        Indicate whether to enable bias shifting.
        If set to False, the layer will not learn an additive bias.
    bias_init: Initializer, default = flax.linen.initializers.zeros
        Used for initializing bias, only used when :attr:`use_bias=True`.
        It should be a callable object with three arguments (jax.random.PRNGKey, shape, dtype).
    bias_axes_1: Tuple[str, ...], default = ('mlp',)
        The name of axes used to shard bias with a corresponding mesh  for
        the weight of the first dense layer transformation.
        Only used when :attr:`use_bias=True`.
    bias_axes_2: Tuple[str, ...], default = ('embed',)
        The name of axes used to shard bias with a corresponding mesh  for
        the weight of the second dense layer transformation.
        Only used when :attr:`use_bias=True`.
    return_layernorm_output: bool, default = True
        Indicate whether to return the output of layer normalization.
        If set False, return None as the second tensor in outputs.
    activations: Sequence[Union[str, Callable]], default = ('relu',)
        The sequence of activation functions to apply after the first dense layer transformation.
        Each activation has its own transformation layer.
    intermediate_dropout_rng_name: str, default = 'dropout'
        The key in given RNGs via flax.linen.Module.apply that for generating Dropout masks.
    intermediate_dropout_rate: float, default = 0.1
        Dropout probability for the dropout op after the :attr:`activations`.
    intermediate_hidden_dropout_dims: Sequence[int], default = ()
        Dimensions that will share the same dropout mask for hidden
    enable_low_rank_adaptation: bool, default = False
        Indicate whether to enable low rank adaptation for each dense layer.
    low_rank_adaptation_dim: int, default = 32
        The dimension for low rank adaptation, only used when
        :attr:`enable_low_rank_adaptation=True`.
    low_rank_adaptation_alpha: float, default = None
        The alpha for computing the scaling factor of LoRA output.
        :math:`\frac{alpha}{rank} * lora_output`. None means no scaling.
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
    ffn1_ckpt_name: str = "ffn1"
        Checkpoint name for the output of the first fully-connected layer in the MLP block.
    ffn2_ckpt_name: str = "ffn2"
        Checkpoint name for the output of the second fully-connected layer in the MLP block.


    Optimization parameters
    -----------------------
    dtype: jax.numpy.dtype, default  = jax.numpy.float32
        The data type used to allocate the initial parameters.
    """

    intermediate_dim: int = 2048
    enable_layernorm: bool = True
    layernorm_type: str = "layernorm"
    epsilon: float = 1e-6
    zero_centered_gamma: bool = False
    scale_init: Initializer = None
    scale_axes: Tuple[str, ...] = ("embed",)
    ln_bias_init: Initializer = nn.initializers.zeros
    ln_bias_axes: Tuple[str, ...] = ("embed",)
    kernel_init: Initializer = None
    kernel_axes_1: Tuple[str, ...] = ("embed", "act", "mlp")
    kernel_axes_2: Tuple[str, ...] = ("mlp", "embed")
    use_bias: bool = False
    bias_init: Initializer = nn.initializers.zeros
    bias_axes_1: Tuple[str, ...] = ("act", "mlp")
    bias_axes_2: Tuple[str, ...] = ("embed",)
    return_layernorm_output: bool = True
    activations: Sequence[Union[str, Callable]] = ("relu",)
    intermediate_dropout_rng_name: str = "dropout"
    intermediate_dropout_rate: float = 0.1
    intermediate_hidden_dropout_dims: Sequence[int] = ()
    enable_low_rank_adaptation: bool = False
    low_rank_adaptation_dim: int = 32
    low_rank_adaptation_alpha: float = None
    axis: Union[Iterable[int], int] = -1
    dtype: DType = jnp.float32
    layernorm_input_axes: Tuple[str, ...] = None
    dot_1_input_axes: Tuple[str, ...] = None
    dot_2_input_axes: Tuple[str, ...] = None
    ffn1_ckpt_name: str = "ffn1"
    ffn2_ckpt_name: str = "ffn2"

    def __post_init__(self):
        if self.kernel_init is None:
            self.kernel_init = nn.initializers.variance_scaling(
                1.0, "fan_in", "truncated_normal", dtype=self.dtype
            )
        self.scale_init = _obtain_default_layernorm_scale_init_if_need(
            self.scale_init,
            self.zero_centered_gamma,
        )
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
        assert self.axis == -1, "Only support axis == -1 at this moment"

        ffn1_quantizer_set = self.generate_quantizer_set("_0")
        ffn2_quantizer_set = self.generate_quantizer_set("_1")

        input_dtype = inputs.dtype
        ln_output = None

        # TODO(Phuong): use fuse_layernorm for high-precision
        # when NoOpQuantizer and Tensor are implemented
        fuse_layernorm = (
            QuantizeConfig.is_fp8_enabled()
            and not self.return_layernorm_output
            and self.enable_layernorm
        )

        gated_act_pool = [
            ("gelu", "linear"),
            ("silu", "linear"),
            ("relu", "linear"),
            ("quick_gelu", "linear"),
            ("squared_relu", "linear"),
        ]
        act_pool = [("gelu",), ("silu",), ("relu",), ("quick_gelu",), ("squared_relu",)]
        normalized_acts = []
        for act in self.activations:
            if not isinstance(act, str):
                return False
            normalized_acts.append(act.lower())
        normalized_acts = tuple(
            reversed(normalized_acts) if normalized_acts[0] == "linear" else normalized_acts
        )

        is_act_implemented = normalized_acts in (gated_act_pool + act_pool)

        use_fused_layernorm_mlp = (
            fuse_layernorm and is_act_implemented and self.intermediate_dropout_rate < 1e-3
        )
        # LayerNorm
        if self.enable_layernorm:
            inputs = with_sharding_constraint_by_logical_axes(inputs, self.layernorm_input_axes)

            features = inputs.shape[-1]

            scale, ln_bias = _create_layernorm_parameters(
                self,
                self.layernorm_type,
                (features,),
                self.scale_init,
                self.scale_axes,
                self.ln_bias_init,
                self.ln_bias_axes,
                input_dtype,
                self.dtype,
            )

            if not fuse_layernorm:
                y = layernorm(
                    inputs,
                    scale,
                    ln_bias,
                    norm_type=self.layernorm_type,
                    zero_centered_gamma=self.zero_centered_gamma,
                    epsilon=self.epsilon,
                )
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
            return jnp.stack(kernels, axis=stack_axis, dtype=self.dtype)

        num_activations = len(normalized_acts)
        axis = _canonicalize_tuple(self.axis)
        axis = _normalize_axes(axis, y.ndim)
        kernel_1_each_shape = (np.prod([inputs.shape[ax] for ax in axis]), self.intermediate_dim)
        kernel_1 = self.param(
            "wi_kernel",
            nn.with_logical_partitioning(kernel_1_init, self.kernel_axes_1),
            num_activations,
            -2,
            kernel_1_each_shape,
            self.dtype,
        )

        if not QuantizeConfig.is_fp8_enabled():
            kernel_1 = kernel_1.astype(input_dtype)

        hidden_size = inputs.shape[-1]
        hidden_size_tuple = _canonicalize_tuple(hidden_size)
        kernel_2_shape = (self.intermediate_dim,) + hidden_size_tuple
        kernel_2 = self.param(
            "wo_kernel",
            nn.with_logical_partitioning(self.kernel_init, self.kernel_axes_2),
            kernel_2_shape,
            self.dtype,
        )
        if not QuantizeConfig.is_fp8_enabled():
            kernel_2 = kernel_2.astype(input_dtype)

        contract_ind = tuple(range(0, len(axis)))

        if self.use_bias:
            bias_1_shape = (num_activations, self.intermediate_dim)
            bias_1 = self.param(
                "wi_bias",
                nn.with_logical_partitioning(self.bias_init, self.bias_axes_1),
                bias_1_shape,
                self.dtype,
            ).astype(input_dtype)

            bias_2_shape = (hidden_size,)
            bias_2 = self.param(
                "wo_bias",
                nn.with_logical_partitioning(self.bias_init, self.bias_axes_2),
                bias_2_shape,
                self.dtype,
            ).astype(input_dtype)
        else:
            bias_1 = None
            bias_2 = None

        if use_fused_layernorm_mlp:
            out = layernorm_mlp(
                y,
                scale,
                ln_bias,
                [kernel_1, kernel_2],
                [bias_1, bias_2],
                self.layernorm_type,
                zero_centered_gamma=self.zero_centered_gamma,
                epsilon=self.epsilon,
                norm_input_axes=self.layernorm_input_axes,
                dot_1_input_axes=self.dot_1_input_axes,
                dot_2_input_axes=self.dot_2_input_axes,
                kernel_1_axes=self.kernel_axes_1,
                kernel_2_axes=self.kernel_axes_2,
                ffn1_ckpt_name=self.ffn1_ckpt_name,
                ffn2_ckpt_name=self.ffn2_ckpt_name,
                activation_type=normalized_acts,
                quantizer_sets=(ffn1_quantizer_set, ffn2_quantizer_set),
            )
            out = out.reshape(*inputs.shape[: self.axis], *hidden_size_tuple)

        else:  # not use_fused_ln_geglu_mlp
            # DenseGeneral 1
            if fuse_layernorm:
                x = layernorm_dense(
                    y,
                    kernel_1,
                    scale,
                    ln_bias,
                    norm_type=self.layernorm_type,
                    zero_centered_gamma=self.zero_centered_gamma,
                    epsilon=self.epsilon,
                    layernorm_input_axes=self.layernorm_input_axes,
                    dot_input_axes=self.dot_1_input_axes,
                    kernel_axes=self.kernel_axes_1,
                    quantizer_set=ffn1_quantizer_set,
                )
            else:
                y = with_sharding_constraint_by_logical_axes(y, self.dot_1_input_axes)
                x = dense(
                    y,
                    kernel_1,
                    contracting_dims=(axis, contract_ind),
                    input_axes=self.dot_1_input_axes,
                    kernel_axes=self.kernel_axes_1,
                    quantizer_set=ffn1_quantizer_set,
                )

            if self.enable_low_rank_adaptation:
                wi_lora_a_kernel_each_shape = (
                    kernel_1_each_shape[: len(axis)],
                    self.low_rank_adaptation_dim,
                )
                wi_lora_a_kernel_axes = (None,) * len(wi_lora_a_kernel_each_shape + 1)
                wi_lora_a_kernel = self.param(
                    "wi_lora_a_kernel",
                    nn.with_logical_partitioning(kernel_1_init, wi_lora_a_kernel_axes),
                    num_activations,
                    -2,
                    wi_lora_a_kernel_each_shape,
                    self.dtype,
                ).astype(input_dtype)

                wi_lora_b_kernel_shape = (
                    num_activations,
                    self.low_rank_adaptation_dim,
                    self.intermediate_dim,
                )
                wi_lora_b_kernel_axes = (None,) * len(wi_lora_b_kernel_shape)
                wi_lora_b_kernel = self.param(
                    "wi_lora_b_kernel",
                    nn.with_logical_partitioning(nn.initializers.zeros, wi_lora_b_kernel_axes),
                    wi_lora_b_kernel_shape,
                    self.dtype,
                ).astype(input_dtype)

                x += _apply_low_rank_adaptation(
                    y,
                    axis,
                    (num_activations, self.intermediate_dim),
                    wi_lora_a_kernel,
                    wi_lora_b_kernel,
                    self.low_rank_adaptation_alpha,
                )

            if self.use_bias:
                x += jnp.reshape(bias_1, bias_1_shape)

            x = checkpoint_name(x, self.ffn1_ckpt_name)
            if is_act_implemented:
                z = activation(x, normalized_acts)
            else:
                activations = []
                x = jnp.split(x, num_activations, axis=-2)
                for idx, act_fn in enumerate(normalized_acts):
                    x_i = _convert_to_activation_function(act_fn)(x[idx])
                    activations.append(x_i)
                z = reduce(operator.mul, activations)
                z = jnp.squeeze(z, axis=-2)
            z = z.astype(input_dtype)

            z = nn.Dropout(
                rate=self.intermediate_dropout_rate,
                broadcast_dims=self.intermediate_hidden_dropout_dims,
                rng_collection=self.intermediate_dropout_rng_name,
            )(z, deterministic=deterministic)

            z = with_sharding_constraint_by_logical_axes(z, self.dot_2_input_axes)
            z = z.astype(input_dtype)

            # DenseGeneral 2
            out = dense(
                z,
                kernel_2,
                contracting_dims=(axis, contract_ind),
                input_axes=self.dot_2_input_axes,
                kernel_axes=self.kernel_axes_2,
                quantizer_set=ffn2_quantizer_set,
            )

            if self.enable_low_rank_adaptation:
                wo_lora_a_kernel_shape = (self.intermediate_dim, self.low_rank_adaptation_dim)
                wo_lora_a_kernel_axes = (None,) * len(wo_lora_a_kernel_shape)
                wo_lora_a_kernel = self.param(
                    "wo_lora_a_kernel",
                    nn.with_logical_partitioning(self.kernel_init, wo_lora_a_kernel_axes),
                    wo_lora_a_kernel_shape,
                    self.dtype,
                ).astype(input_dtype)

                wo_lora_b_kernel_shape = (self.low_rank_adaptation_dim, hidden_size)
                wo_lora_b_kernel_axes = (None,) * len(wo_lora_b_kernel_shape)
                wo_lora_b_kernel = self.param(
                    "wo_lora_b_kernel",
                    nn.with_logical_partitioning(nn.initializers.zeros, wo_lora_b_kernel_axes),
                    wo_lora_b_kernel_shape,
                    self.dtype,
                ).astype(input_dtype)

                out += _apply_low_rank_adaptation(
                    z,
                    axis,
                    hidden_size_tuple,
                    wo_lora_a_kernel,
                    wo_lora_b_kernel,
                    self.low_rank_adaptation_alpha,
                )

            if self.use_bias:
                out += jnp.reshape(bias_2, (1,) * (out.ndim - 1) + (-1,))

            out = checkpoint_name(out, self.ffn2_ckpt_name)

        assert out.dtype == input_dtype
        return out, ln_output  # Output, layner_norm_output
