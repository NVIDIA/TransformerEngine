# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Praxis Modules
"""
from functools import partial
from typing import Callable, Iterable, Sequence, Tuple, Union

from praxis import pax_fiddle
from praxis.base_layer import init_var
from praxis.base_layer import BaseLayer, WeightInit, WeightHParams, WeightHParamsCollection
from praxis.layers import flax_adapter
from praxis.pytypes import JTensor

from ..fp8 import FP8Helper
from ..flax.module import DenseGeneral, LayerNormDenseGeneral
from ..flax.module import LayerNorm as flax_LayerNorm
from ..flax.module import LayerNormMLP as flax_LayerNormMLP
from ..flax.module import Softmax
from ..softmax import SoftmaxType


def _generate_ln_scale_init(scale_init):
    if scale_init is not None:
        return TransformerEngineBaseLayer.generate_params_init("scale", scale_init)
    return scale_init


class TransformerEngineBaseLayer(BaseLayer):
    """TransformerEngineBaseLayer"""

    logical_axes_rules: Tuple[Tuple, ...] = None

    @staticmethod
    def generate_params_init(name: str, initializer: WeightInit):
        """generate_params_init"""

        def kernel_init(key, shape, dtype):
            wp = WeightHParams(shape=shape, init=initializer, dtype=dtype)
            return init_var(wp, key, name)

        return kernel_init

    def create_layer(self, name, flax_module_cls):
        """create_layer"""

        fp8_collection_map = {
            FP8Helper.FP8_COLLECTION_NAME: [
                WeightHParamsCollection.SKIP_LP_REGULARIZATION,
                WeightHParamsCollection.OVERWRITE_WITH_GRADIENT,
                WeightHParamsCollection.DISALLOW_BFLOAT16_CONVERSION,
            ]
        }

        flax_module_p = pax_fiddle.Config(
            flax_adapter.FlaxModuleAdapter,
            module_factory_method=flax_module_cls,
            logical_axes_rules=self.logical_axes_rules,
            var_collection_map=fp8_collection_map,
            ici_mesh_shape=self.ici_mesh_shape,
            dcn_mesh_shape=self.dcn_mesh_shape,
            mesh_axis_names=self.mesh_axis_names,
        )

        self.create_child(name, flax_module_p.clone())


class LayerNorm(TransformerEngineBaseLayer):
    """LayerNorm"""

    epsilon: float = 1e-6
    layernorm_type: str = "layernorm"
    zero_centered_gamma: bool = False
    scale_init: WeightInit = None
    scale_axes: Tuple[str, ...] = ()
    bias_init: WeightInit = WeightInit.Constant(0.0)
    bias_axes: Tuple[str, ...] = ()
    transpose_batch_sequence: bool = False

    def setup(self) -> None:
        """setup"""
        super().setup()

        ln_cls = partial(
            flax_LayerNorm,
            epsilon=self.epsilon,
            layernorm_type=self.layernorm_type,
            zero_centered_gamma=self.zero_centered_gamma,
            scale_init=_generate_ln_scale_init(self.scale_init),
            scale_axes=self.scale_axes,
            bias_init=TransformerEngineBaseLayer.generate_params_init("ln_bias", self.bias_init),
            bias_axes=self.bias_axes,
            dtype=self.dtype,
            transpose_batch_sequence=self.transpose_batch_sequence,
        )

        self.create_layer("layer_norm", ln_cls)

    def __call__(self, x: JTensor) -> JTensor:
        """__call__"""
        return self.layer_norm(x)


class FusedSoftmax(TransformerEngineBaseLayer):
    """FusedSoftmax"""

    scale_factor: float = 1.0
    softmax_type: SoftmaxType = SoftmaxType.SCALED

    def setup(self) -> None:
        """setup"""
        super().setup()

        fused_softmax_cls = partial(
            Softmax, scale_factor=self.scale_factor, softmax_type=self.softmax_type
        )

        self.create_layer("fused_softmax", fused_softmax_cls)

    def __call__(self, x: JTensor, mask: JTensor = None, bias: JTensor = None) -> JTensor:
        """__call__"""
        return self.fused_softmax(x, mask, bias)


class Linear(TransformerEngineBaseLayer):
    """Linear"""

    out_features: int = 512
    kernel_axes: Tuple[str, ...] = ()
    use_bias: bool = True
    bias_init: WeightInit = WeightInit.Constant(0.0)
    bias_axes: Tuple[str, ...] = ()
    enable_low_rank_adaptation: bool = False
    low_rank_adaptation_dim: int = 32
    low_rank_adaptation_alpha: float = None
    axis: Union[Iterable[int], int] = -1
    transpose_batch_sequence: bool = False

    def setup(self) -> None:
        """setup"""
        super().setup()

        dense_general_cls = partial(
            DenseGeneral,
            features=self.out_features,
            kernel_init=TransformerEngineBaseLayer.generate_params_init("kernel", self.params_init),
            kernel_axes=self.kernel_axes,
            use_bias=self.use_bias,
            bias_init=TransformerEngineBaseLayer.generate_params_init("bias", self.bias_init),
            bias_axes=self.bias_axes,
            enable_low_rank_adaptation=self.enable_low_rank_adaptation,
            low_rank_adaptation_dim=self.low_rank_adaptation_dim,
            low_rank_adaptation_alpha=self.low_rank_adaptation_alpha,
            axis=self.axis,
            dtype=self.dtype,
            transpose_batch_sequence=self.transpose_batch_sequence,
        )

        self.create_layer("linear", dense_general_cls)

    def __call__(self, x: JTensor) -> JTensor:
        """__call__"""
        return self.linear(x)


class LayerNormLinear(TransformerEngineBaseLayer):
    """LayerNormLinear"""

    out_features: int = 512
    enable_layernorm: bool = True
    layernorm_type: str = "layernorm"
    epsilon: float = 1e-6
    zero_centered_gamma: bool = False
    scale_init: WeightInit = None
    scale_axes: Tuple[str, ...] = ()
    ln_bias_init: WeightInit = WeightInit.Constant(1.0)
    ln_bias_axes: Tuple[str, ...] = ()
    kernel_axes: Tuple[str, ...] = ()
    use_bias: bool = False
    bias_init: WeightInit = WeightInit.Constant(0.0)
    bias_axes: Tuple[str, ...] = ()
    enable_low_rank_adaptation: bool = False
    low_rank_adaptation_dim: int = 32
    low_rank_adaptation_alpha: float = None
    return_layernorm_output: bool = True
    axis: Union[Iterable[int], int] = -1
    transpose_batch_sequence: bool = False
    depth_scaling: float = None

    def setup(self) -> None:
        """setup"""
        super().setup()

        ln_dense_general_cls = partial(
            LayerNormDenseGeneral,
            features=self.out_features,
            enable_layernorm=self.enable_layernorm,
            layernorm_type=self.layernorm_type,
            epsilon=self.epsilon,
            zero_centered_gamma=self.zero_centered_gamma,
            scale_init=_generate_ln_scale_init(self.scale_init),
            scale_axes=self.scale_axes,
            ln_bias_init=TransformerEngineBaseLayer.generate_params_init(
                "ln_bias", self.ln_bias_init
            ),
            ln_bias_axes=self.ln_bias_axes,
            kernel_init=TransformerEngineBaseLayer.generate_params_init("kernel", self.params_init),
            kernel_axes=self.kernel_axes,
            use_bias=self.use_bias,
            bias_init=TransformerEngineBaseLayer.generate_params_init("bias", self.bias_init),
            bias_axes=self.bias_axes,
            enable_low_rank_adaptation=self.enable_low_rank_adaptation,
            low_rank_adaptation_dim=self.low_rank_adaptation_dim,
            low_rank_adaptation_alpha=self.low_rank_adaptation_alpha,
            return_layernorm_output=self.return_layernorm_output,
            axis=self.axis,
            dtype=self.dtype,
            transpose_batch_sequence=self.transpose_batch_sequence,
            depth_scaling=self.depth_scaling,
        )

        self.create_layer("ln_linear", ln_dense_general_cls)

    def __call__(self, x: JTensor) -> JTensor:
        """__call__"""
        return self.ln_linear(x)


class LayerNormMLP(TransformerEngineBaseLayer):
    """LayerNormMLP"""

    intermediate_dim: int = 2048
    enable_layernorm: bool = True
    layernorm_type: str = "layernorm"
    epsilon: float = 1e-6
    zero_centered_gamma: bool = False
    scale_init: WeightInit = None
    scale_axes: Tuple[str, ...] = ()
    ln_bias_init: WeightInit = WeightInit.Constant(1.0)
    ln_bias_axes: Tuple[str, ...] = ()
    kernel_axes_1: Tuple[str, ...] = ()
    kernel_axes_2: Tuple[str, ...] = ()
    use_bias: bool = False
    bias_init: WeightInit = WeightInit.Constant(0.0)
    bias_axes_1: Tuple[str, ...] = ()
    bias_axes_2: Tuple[str, ...] = ()
    enable_low_rank_adaptation: bool = False
    low_rank_adaptation_dim: int = 32
    low_rank_adaptation_alpha: float = None
    return_layernorm_output: bool = True
    activations: Sequence[Union[str, Callable]] = ("relu",)
    intermediate_dropout_rate: float = 0.1
    intermediate_hidden_dropout_dims: Sequence[int] = ()
    axis: Union[Iterable[int], int] = -1
    transpose_batch_sequence: bool = False

    def setup(self) -> None:
        """setup"""
        super().setup()

        ln_mlp_cls = partial(
            flax_LayerNormMLP,
            intermediate_dim=self.intermediate_dim,
            enable_layernorm=self.enable_layernorm,
            layernorm_type=self.layernorm_type,
            epsilon=self.epsilon,
            zero_centered_gamma=self.zero_centered_gamma,
            scale_init=_generate_ln_scale_init(self.scale_init),
            scale_axes=self.scale_axes,
            ln_bias_init=TransformerEngineBaseLayer.generate_params_init(
                "ln_bias", self.ln_bias_init
            ),
            ln_bias_axes=self.ln_bias_axes,
            kernel_init=TransformerEngineBaseLayer.generate_params_init("kernel", self.params_init),
            kernel_axes_1=self.kernel_axes_1,
            kernel_axes_2=self.kernel_axes_2,
            use_bias=self.use_bias,
            bias_init=TransformerEngineBaseLayer.generate_params_init("bias", self.bias_init),
            bias_axes_1=self.bias_axes_1,
            bias_axes_2=self.bias_axes_2,
            enable_low_rank_adaptation=self.enable_low_rank_adaptation,
            low_rank_adaptation_dim=self.low_rank_adaptation_dim,
            low_rank_adaptation_alpha=self.low_rank_adaptation_alpha,
            return_layernorm_output=self.return_layernorm_output,
            activations=self.activations,
            intermediate_dropout_rate=self.intermediate_dropout_rate,
            intermediate_hidden_dropout_dims=self.intermediate_hidden_dropout_dims,
            axis=self.axis,
            dtype=self.dtype,
            transpose_batch_sequence=self.transpose_batch_sequence,
        )

        self.create_layer("ln_mlp", ln_mlp_cls)

    def __call__(self, x: JTensor, deterministic: bool = False) -> JTensor:
        """__call__"""
        return self.ln_mlp(x, deterministic)
