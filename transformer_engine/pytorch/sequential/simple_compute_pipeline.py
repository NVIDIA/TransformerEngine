from typing import Any, Callable
from torch import dtype, nn
import torch

from transformer_engine.jax import layernorm
from ..module import Linear, LayerNorm, LayerNormLinear, LayerNormMLP
from ..attention import DotProductAttention


def linear(linear: nn.Linear) -> nn.Linear | Linear:
    in_features = linear.in_features
    out_features = linear.out_features
    bias = linear.get_parameter("bias") is not None  # type: ignore
    dtype = linear.weight.dtype

    if linear.weight.device != torch.cuda.current_device():  # type: ignore
        return linear

    return Linear(in_features, out_features, bias=bias, params_dtype=dtype)


def layerNorm(layerNorm: nn.LayerNorm) -> nn.LayerNorm | LayerNorm:
    normalized_shape = layerNorm.normalized_shape
    eps = layerNorm.eps
    elementwise_affine = layerNorm.elementwise_affine
    dtype = layerNorm.weight.dtype

    if len(normalized_shape) != 1 or elementwise_affine:
        return layerNorm

    return LayerNorm(normalized_shape[0], eps, params_dtype=dtype)


def layerNormLinear(layerNorm: LayerNorm, linear: Linear) -> LayerNormLinear:
    in_features = layerNorm.weight.shape[0]
    out_features = linear.out_features
    eps = layerNorm.eps
    sequence_parallel = linear.sequence_parallel and layerNorm.weight.sequence_parallel  # type: ignore
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
    return_layernorm_output = True
    skip_weight_param_allocation = hasattr(linear, "weight_tensor")
    parameters_split = linear.parameters_split
    zero_centered_gamma = layerNorm.zero_centered_gamma
    ub_bulk_wgrad = True
    ub_bulk_dgrad = True
    ub_split_ag = linear.ub_split_ag

    return LayerNormLinear(
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


def layerNormMLP(
    layerNormLinear: LayerNormLinear, act: str, linear: Linear
) -> LayerNormMLP:



def layerNormMLPGELU(
    layerNormLinear: LayerNormLinear, gelu: nn.GELU, linear: Linear
) -> LayerNormMLP:
    del gelu  # unused
    return layerNormMLP(layerNormLinear, "gelu", linear)


def layerNormMLPReLU(
    layerNormLinear: LayerNormLinear, relu: nn.ReLU, linear: Linear
) -> LayerNormMLP:
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
        for startPos in range(len(list) - len(pattern)):
            if all(
                isinstance(list[startPos + i], pattern[i]) for i in range(len(pattern))
            ):
                list[startPos : startPos + len(pattern)] = [replacer(*pattern)]

    def __call__(self, x: Any) -> Any:
        return self.module(x)
