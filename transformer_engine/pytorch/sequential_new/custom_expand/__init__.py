from torch import nn
from ..custom_expand_for_sequential import CUSTOM_EXPAND_FOR_SEQUENTIAL
from ..expand_for_sequential import expand
from ..compile_env import CompileEnv
from ..ops import Op
from ...module import Linear, LayerNorm, LayerNormLinear, LayerNormMLP


def expand_linear(module: nn.Module, env: CompileEnv):
    assert isinstance(module, nn.Linear) or isinstance(module, Linear)
    has_bias = (
        getattr(module, "bias") is not None
        if isinstance(module, nn.Linear)
        else module.use_bias
    )
    if env.fp8:
        if has_bias:
            return [Op.GEMM_FP8, Op.ADD_FP8]
        else:
            return [Op.GEMM_FP8]
    else:
        if has_bias:
            return [Op.GEMM, Op.ADD]
        else:
            return [Op.GEMM]


def expand_layerNorm(module: nn.Module, env: CompileEnv):
    assert isinstance(module, nn.LayerNorm) or isinstance(module, LayerNorm)
    if env.fp8:
        return [Op.LAYER_NORM_FP8]
    else:
        return [Op.LAYER_NORM]


def expand_layerNormLinear(module: nn.Module, env: CompileEnv):
    assert isinstance(module, LayerNormLinear)
    has_bias = module.use_bias
    if env.fp8:
        if has_bias:
            return [Op.LAYER_NORM_FP8, Op.GEMM_FP8, Op.ADD_FP8]
        else:
            return [Op.LAYER_NORM_FP8, Op.GEMM_FP8]
    else:
        if has_bias:
            return [Op.LAYER_NORM, Op.GEMM, Op.ADD]
        else:
            return [Op.LAYER_NORM, Op.GEMM]


def expand_layerNormMLP(module: nn.Module, env: CompileEnv):
    assert isinstance(module, LayerNormMLP)
    has_bias = module.use_bias
    if env.fp8:
        if has_bias:
            return [
                Op.LAYER_NORM_FP8,
                Op.GEMM_FP8,
                Op.ADD_FP8,
                Op.GELU_FP8,
                Op.GEMM_FP8,
                Op.ADD_FP8,
            ]
        else:
            return [Op.LAYER_NORM_FP8, Op.GEMM_FP8, Op.GELU_FP8, Op.GEMM_FP8]
    else:
        if has_bias:
            return [Op.LAYER_NORM, Op.GEMM, Op.ADD, Op.GELU, Op.GEMM, Op.ADD]
        else:
            return [Op.LAYER_NORM, Op.GEMM, Op.GELU, Op.GEMM]


def expand_sequential(module: nn.Module, env: CompileEnv):
    assert isinstance(module, nn.Sequential)
    return [op for submodule in module for op in expand(submodule, env)]


CUSTOM_EXPAND_FOR_SEQUENTIAL[nn.Linear] = expand_linear
CUSTOM_EXPAND_FOR_SEQUENTIAL[Linear] = expand_linear
CUSTOM_EXPAND_FOR_SEQUENTIAL[nn.LayerNorm] = expand_layerNorm
CUSTOM_EXPAND_FOR_SEQUENTIAL[LayerNorm] = expand_layerNorm
CUSTOM_EXPAND_FOR_SEQUENTIAL[LayerNormLinear] = expand_layerNormLinear
CUSTOM_EXPAND_FOR_SEQUENTIAL[LayerNormMLP] = expand_layerNormMLP
