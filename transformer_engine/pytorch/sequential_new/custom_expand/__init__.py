from torch import nn
from ..custom_expand_for_sequential import CUSTOM_EXPAND_FOR_SEQUENTIAL
from ..custom_expand_for_sequential import expand
from ..compile_env import CompileEnv
from ..ops import DType
from .. import ops
from ...module import Linear, LayerNorm, LayerNormLinear, LayerNormMLP


def expand_linear(module: nn.Module, env: CompileEnv) -> list[ops.Op]:
    assert isinstance(module, nn.Linear) or isinstance(module, Linear)
    has_bias = (
        getattr(module, "bias") is not None
        if isinstance(module, nn.Linear)
        else module.use_bias
    )
    in_features = module.in_features
    out_features = module.out_features
    tensor_type = DType.FP8 if env.fp8 else DType.default

    if has_bias:
        return [
            ops.Gemm(tensor_type, tensor_type, in_features, out_features),
            ops.Add(tensor_type, DType.infer, out_features),
        ]
    else:
        return [ops.Gemm(tensor_type, DType.infer, in_features, out_features)]


def expand_layerNorm(module: nn.Module, env: CompileEnv) -> list[ops.Op]:
    assert isinstance(module, nn.LayerNorm) or isinstance(module, LayerNorm)

    hidden_size = (
        module.weight.shape[0]
        if isinstance(module, LayerNorm)
        else module.normalized_shape[0]
    )
    eps = module.eps
    zero_centered_gamma = (
        True if isinstance(module, nn.LayerNorm) else module.zero_centered_gamma
    )
    tensor_type = DType.FP8 if env.fp8 else DType.default

    return [
        ops.LayerNorm(tensor_type, DType.infer, hidden_size, eps, zero_centered_gamma)
    ]


def expand_layerNormLinear(module: nn.Module, env: CompileEnv) -> list[ops.Op]:
    assert isinstance(module, LayerNormLinear)

    has_bias = module.use_bias
    in_features = module.in_features
    out_features = module.out_features
    eps = module.eps
    zero_centered_gamma = module.zero_centered_gamma
    tensor_type = DType.FP8 if env.fp8 else DType.default

    if has_bias:
        return [
            ops.LayerNorm(
                tensor_type, tensor_type, in_features, eps, zero_centered_gamma
            ),
            ops.Gemm(tensor_type, tensor_type, in_features, out_features),
            ops.Add(tensor_type, DType.infer, out_features),
        ]
    else:
        return [
            ops.LayerNorm(
                tensor_type, tensor_type, in_features, eps, zero_centered_gamma
            ),
            ops.Gemm(tensor_type, DType.infer, in_features, out_features),
        ]


def expand_layerNormMLP(module: nn.Module, env: CompileEnv) -> list[ops.Op]:
    assert isinstance(module, LayerNormMLP)

    has_bias = module.use_bias
    in_features = module.layer_norm_weight.shape[0]
    ffn_size = module.size_per_partition * module.tp_size
    eps = module.eps
    zero_centered_gamma = module.zero_centered_gamma
    tensor_type = DType.FP8 if env.fp8 else DType.default

    if has_bias:
        return [
            ops.LayerNorm(
                tensor_type, tensor_type, in_features, eps, zero_centered_gamma
            ),
            ops.Gemm(tensor_type, tensor_type, in_features, ffn_size),
            ops.Add(tensor_type, tensor_type, ffn_size),
            ops.Gelu(),
            ops.Gemm(tensor_type, tensor_type, ffn_size, in_features),
            ops.Add(tensor_type, DType.infer, in_features),
        ]
    else:
        return [
            ops.LayerNorm(
                tensor_type, tensor_type, in_features, eps, zero_centered_gamma
            ),
            ops.Gemm(tensor_type, tensor_type, in_features, ffn_size),
            ops.Gelu(),
            ops.Gemm(tensor_type, tensor_type, ffn_size, in_features),
        ]


def expand_sequential(module: nn.Module, env: CompileEnv):
    assert isinstance(module, nn.Sequential)
    return [op for submodule in module for op in expand(submodule, env)]


CUSTOM_EXPAND_FOR_SEQUENTIAL[nn.Linear] = expand_linear
CUSTOM_EXPAND_FOR_SEQUENTIAL[Linear] = expand_linear
CUSTOM_EXPAND_FOR_SEQUENTIAL[nn.LayerNorm] = expand_layerNorm
CUSTOM_EXPAND_FOR_SEQUENTIAL[LayerNorm] = expand_layerNorm
CUSTOM_EXPAND_FOR_SEQUENTIAL[LayerNormLinear] = expand_layerNormLinear
CUSTOM_EXPAND_FOR_SEQUENTIAL[LayerNormMLP] = expand_layerNormMLP
CUSTOM_EXPAND_FOR_SEQUENTIAL[nn.GELU] = lambda m, e: [ops.Gelu()]
CUSTOM_EXPAND_FOR_SEQUENTIAL[nn.ReLU] = lambda m, e: [ops.Relu()]
