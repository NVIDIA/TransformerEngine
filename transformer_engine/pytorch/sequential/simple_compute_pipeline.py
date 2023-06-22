from typing import Any, Callable
from sympy import sequence
from torch import device, dtype, nn
import torch

from ..module import Linear, LayerNorm, LayerNormLinear, LayerNormMLP


def linear(linear: nn.Linear):
    in_features = linear.in_features
    out_features = linear.out_features
    bias = linear.get_parameter("bias") is not None  # type: ignore
    dtype = linear.weight.dtype

    if linear.weight.device != device("cuda", torch.cuda.current_device()):
        return [linear]

    return [Linear(in_features, out_features, bias=bias, params_dtype=dtype)]


def layerNorm(layerNorm: nn.LayerNorm):
    # TODO: technically, te.Linear doesn't support elementwise_affine
    # as it initializes its parameters differently
    # but it should converge at some point any way.
    # It doesn't support the no-parameters mode, though
    if len(layerNorm.normalized_shape) != 1 or not layerNorm.elementwise_affine:
        return [layerNorm]

    hidden_size = layerNorm.normalized_shape[0]
    eps = layerNorm.eps

    # TODO: check if this is correct behavior
    # it is supposed to only increase performance
    sequence_parallel = True

    params_dtype = layerNorm.weight.dtype

    # TODO: this is an approximation of the elementwise_affine mode
    zero_centered_gamma = True

    return [
        LayerNorm(
            hidden_size,
            eps,
            sequence_parallel,
            params_dtype,
            zero_centered_gamma,
        )
    ]


def layerNormLinear(layerNorm: LayerNorm, linear: Linear):
    in_features = layerNorm.weight.shape[0]
    out_features = linear.out_features
    eps = layerNorm.eps
    sequence_parallel = (
        linear.sequence_parallel and layerNorm.weight.sequence_parallel
    )  # type: ignore
    fuse_wgrad_accumulation = linear.fuse_wgrad_accumulation
    tp_group = linear.tp_group
    tp_size = linear.tp_size
    get_rng_state_tracker = (
        None  # get_rng_state_tracker cannot be recovered, assume it is None
    )
    init_method = None  # init_method cannot be recovered, assume it is None
    bias = linear.use_bias
    return_bias = linear.return_bias

    linear_dtype: dtype
    try:
        linear_dtype = linear.weight_tensor.dtype  # type: ignore
    except AttributeError:
        linear_dtype = layerNorm.weight.dtype
    linear_dtype = linear_dtype or layerNorm.weight.dtype
    layernorm_dtype = layerNorm.weight.dtype
    params_dtype = linear_dtype if linear_dtype == layernorm_dtype else None

    parallel_mode = linear.parallel_mode
    return_layernorm_output = False
    skip_weight_param_allocation = not hasattr(linear, "weight_tensor")
    parameters_split = linear.parameters_split
    zero_centered_gamma = layerNorm.zero_centered_gamma
    ub_bulk_wgrad = False
    ub_bulk_dgrad = False
    ub_split_ag = linear.ub_split_ag

    return [
        LayerNormLinear(
            in_features,
            out_features,
            eps,
            sequence_parallel,
            fuse_wgrad_accumulation,
            tp_group,
            tp_size,
            get_rng_state_tracker,
            init_method,
            bias,
            return_bias,
            params_dtype,
            parallel_mode,
            return_layernorm_output,
            skip_weight_param_allocation,
            parameters_split,
            zero_centered_gamma,
            ub_bulk_wgrad,
            ub_bulk_dgrad,
            ub_split_ag,
        )
    ]


def layerNormMLP(layerNormLinear: LayerNormLinear, act: str, linear: Linear):
    if (
        layerNormLinear.in_features != linear.out_features
        or layerNormLinear.tp_group != linear.tp_group
        or layerNormLinear.use_bias != linear.use_bias
    ):
        activation = nn.GELU if act == "gelu" else nn.ReLU
        return [layerNormLinear, activation, linear]

    hidden_size = layerNormLinear.in_features
    ffn_hidden_size = linear.in_features
    eps = layerNormLinear.eps
    sequence_parallel = layerNormLinear.sequence_parallel and linear.sequence_parallel
    return_bias = linear.return_bias
    get_rng_state_tracker = (
        None  # get_rng_state_tracker cannot be recovered, assume it is None
    )
    tp_group = linear.tp_group
    tp_size = linear.tp_size
    init_method = None  # init_method cannot be recovered, assume it is None
    bias = linear.use_bias
    output_layer_init_method = (
        None  # output_layer_init_method cannot be recovered, assume it is None
    )
    fuse_wgrad_accumulation = linear.fuse_wgrad_accumulation
    params_dtype = (
        linear.weight_tensor.dtype
        if linear.weight_tensor.dtype == layerNormLinear.weight.dtype
        else None
    )
    return_layernorm_output = False
    seq_length = None
    micro_batch_size = None
    set_parallel_mode = False
    zero_centered_gamma = layerNormLinear.zero_centered_gamma
    ub_bulk_wgrad = layerNormLinear.ub_bulk_wgrad
    ub_bulk_dgrad = layerNormLinear.ub_bulk_dgrad
    ub_split_rs = linear.ub_split_rs
    ub_split_ag = layerNormLinear.ub_split_ag and linear.ub_split_ag

    return [
        LayerNormMLP(
            hidden_size,
            ffn_hidden_size,
            eps,
            sequence_parallel,
            return_bias,
            get_rng_state_tracker,
            tp_group,
            tp_size,
            init_method,
            bias,
            output_layer_init_method,
            fuse_wgrad_accumulation,
            params_dtype,
            return_layernorm_output,
            seq_length,
            micro_batch_size,
            set_parallel_mode,
            zero_centered_gamma,
            ub_bulk_wgrad,
            ub_bulk_dgrad,
            ub_split_rs,
            ub_split_ag,
        )
    ]


def layerNormMLPGELU(layerNormLinear: LayerNormLinear, gelu: nn.GELU, linear: Linear):
    del gelu  # unused
    return layerNormMLP(layerNormLinear, "gelu", linear)


def layerNormMLPReLU(layerNormLinear: LayerNormLinear, relu: nn.ReLU, linear: Linear):
    del relu  # unused
    return layerNormMLP(layerNormLinear, "relu", linear)


PATTERNS = {
    (nn.Linear,): linear,
    (nn.LayerNorm,): layerNorm,
    (LayerNorm, Linear): layerNormLinear,
    (LayerNormLinear, nn.GELU, Linear): layerNormMLPGELU,
    (LayerNormLinear, nn.ReLU, Linear): layerNormMLPReLU,
}


class ComputePipeline:
    def __init__(self, *modules: nn.Module) -> None:
        module_list = list(modules)

        for pattern, replacer in PATTERNS.items():
            ComputePipeline._replace(module_list, pattern, replacer)

        self.module = nn.Sequential(*module_list)

    @staticmethod
    def _replace(list: list[nn.Module], pattern: tuple[type, ...], replacer: Callable):
        for startPos in range(len(list) - len(pattern) + 1):
            if all(
                isinstance(list[startPos + i], pattern[i]) for i in range(len(pattern))
            ):
                list[startPos : startPos + len(pattern)] = replacer(
                    *list[startPos : startPos + len(pattern)]
                )

    def __call__(self, x: Any) -> Any:
        return self.module(x)
