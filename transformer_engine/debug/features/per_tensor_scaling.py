# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PerTensorScaling Feature support for nvidia-dlframework-inspect"""

from typing import Optional

import torch

import nvdlfw_inspect.api as debug_api
from nvdlfw_inspect.registry import Registry, api_method

import transformer_engine_torch as tex
from transformer_engine.common.recipe import Format
from transformer_engine.pytorch.fp8 import _default_sf_compute
from transformer_engine.pytorch.tensor import Quantizer
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer, Float8Tensor
from transformer_engine.debug.features.api import TEConfigAPIMapper


def per_tensor_cast(
    tensor: torch.Tensor, fp8_dtype: tex.DType, margin: int = 0, out: Float8Tensor = None
) -> Float8Tensor:
    """
    This function ocmputes the scaling factors based on the tensor amax and then casts it to the fp8
    """

    assert tensor.dtype in (
        torch.float,
        torch.float16,
        torch.bfloat16,
    ), "[NVTORCH INSPECT ERROR] Unsupported tensor type for per tensor scaling"
    assert tensor.is_cuda, "[NVTORCH INSPECT ERROR] Must be a GPU tensor."
    assert fp8_dtype in {
        tex.DType.kFloat8E4M3,
        tex.DType.kFloat8E5M2,
    }, "[NVTORCH INSPECT ERROR] Only 2 FP8 types: E4M3 and E5M2 are supported in TE."

    if fp8_dtype == tex.DType.kFloat8E4M3:
        fp8_max = Format.E4M3.value.max_fwd
    else:
        fp8_max = Format.E5M2.value.max_fwd
    amax = tensor.abs().max().float()
    one = torch.ones(1, device=tensor.device)

    scale = _default_sf_compute(amax, one, fp8_max, margin)

    quantizer = Float8Quantizer(scale, amax, fp8_dtype)

    if out is not None:
        quantizer.update_quantized(tensor, out)
        return None
    return quantizer(tensor)


@Registry.register_feature(namespace="transformer_engine")
class PerTensorScaling(TEConfigAPIMapper):
    """
    Per Tensor Current Scaling feature for FP8 tensor in Transformer engine.

    If this feature is enabled, delayed scaling is disabled for the specified FP8 Tensor and GEMM.

    Config:
    To enable the feature in yaml config:
    transformer_engine:
        per_tensor_scaling:
        enabled: True
        ...

    Config fields:
    This feature works at a tensor level, you can set the following properties for each tensor:
    - margin: int, default is 0, amax = amax * (2^margin)
    - tensors/tensors_struct: tensors list or tensors_struct - please look into the Transformer Engine Precision Debug Tools documentation for more information.

    Gemm and Tensor structure is described below:
    """

    def _get_margin_default(self):
        """Returns default value of the margin parameter of the quantization."""
        return 0

    @api_method
    def fp8_gemm(
        self, config, layer_name: str, gemm: str, iteration: int
    ):  # pylint: disable=unused-argument
        """API call responsible for selecting between high-precision and FP8 GEMM execution."""
        return False

    @api_method
    def modify_tensor_enabled(
        self, config, layer_name: str, tensor_name: str, gemm: str, iteration: int
    ):  # pylint: disable=unused-argument
        """API call used to determine whether to run process_tensor() in the forward."""
        return True

    @api_method
    def modify_tensor(
        self,
        config,
        layer_name: str,
        gemm: str,
        tensor_name: str,
        tensor: torch.Tensor,
        iteration: int,
        default_quantizer: Quantizer,
        out: Optional[Float8Tensor] = None,
        dtype: Optional[torch.dtype] = None,
    ):  # pylint: disable=unused-argument
        """API call used to process the tensor."""
        for key in config.keys():
            if key not in ["gemm", "tensor", "margin"]:
                raise ValueError(f'[NVTORCH INSPECT ERROR] Unexpected key in config: "{key}".')

        assert default_quantizer is not None, (
            f"[NVTORCH INSPECT ERROR] Feature={self.__class__.__name__}, API=process_tensor:"
            f" Provide FP8 dtype when using process_tensor for per_tensor_scaling. {layer_name}"
        )

        debug_api.log_message(
            f"Feature={self.__class__.__name__}, API=process_tensor: {gemm}, {tensor_name}",
            layer_name,
            extra_cachable_args=(gemm, tensor_name),
        )

        margin = config.get("margin", self._get_margin_default())
        fp8_tensor = per_tensor_cast(tensor, default_quantizer.dtype, margin=margin, out=out)
        return fp8_tensor
