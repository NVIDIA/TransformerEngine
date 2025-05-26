# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FakeQuant Feature support for nvidia-dlframework-inspect"""

from typing import Optional

import torch

import nvdlfw_inspect.api as debug_api
from nvdlfw_inspect.registry import Registry, api_method
from nvdlfw_inspect.utils import append_parent_docstring


import transformer_engine_torch as tex
from transformer_engine.debug.features.api import TEConfigAPIMapper
from transformer_engine.common.recipe import Format
from transformer_engine.pytorch.tensor import Quantizer
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.fp8 import _default_sf_compute


def fake_quantize(tensor: torch.Tensor, fp8_format: tex.DType, out=None):
    """Input tensor is quantized to fp8 and then dequantized."""

    assert tensor.dtype in (
        torch.float,
        torch.float16,
        torch.bfloat16,
    ), "[NVTORCH INSPECT ERROR] Unsupported tensor type."
    assert tensor.is_cuda, "[NVTORCH INSPECT ERROR] Must be a GPU tensor."
    assert fp8_format in {
        "FP8E4M3",
        "FP8E5M2",
        "MXFP8E4M3",
        "MXFP8E5M2",
    }, (
        "[NVTORCH INSPECT ERROR] Only 4 FP8 types: FP8E4M3, FP8E5M2, MXFP8E4M3, MXFP8E5M2 are"
        " supported in TE."
    )
    if fp8_format in ["FP8E4M3", "FP8E5M2"]:
        if fp8_format == "FP8E4M3":
            fp8_max = Format.E4M3.value.max_fwd
            fp8_dtype = tex.DType.kFloat8E4M3
        else:
            fp8_max = Format.E5M2.value.max_fwd
            fp8_dtype = tex.DType.kFloat8E5M2
        amax = tensor.abs().max().float()
        one = torch.ones(1, device=tensor.device)
        scale = _default_sf_compute(amax, one, fp8_max, 0)

        quantizer = Float8Quantizer(scale, amax, fp8_dtype)
    else:
        quantizer = MXFP8Quantizer(fp8_dtype=fp8_format)
    if out is not None:
        out.copy_(quantizer(tensor).dequantize())
        return None
    return quantizer(tensor).dequantize()


@Registry.register_feature(namespace="transformer_engine")
@append_parent_docstring(parent=TEConfigAPIMapper)
class FakeQuant(TEConfigAPIMapper):
    """

    Disables FP8 GEMM. Fake quantizes chosen tensors to FP8 - using per-tensor scaling factor, not delayed scaling - and runs high-precision GEMM.

    .. figure:: ./img/fake_quant.svg
        :align: center

        Fig 1: Comparison of FP8 FPROP GEMM with the same GEMM in BF16 with fake quantization of activation tensor. Green tensors have the same values, but different dtypes.



    Parameters
    ----------

    gemms/gemms_struct: List[str]
        list of gemms to fake quantize

            - fprop
            - dgrad
            - wgrad
    tensors/tensors_struct: List[str]
        list of tensors to fake quantize

            - activation
            - gradient
            - weight
            - output
            - wgrad
            - dgrad

    quant_format: str
        specifies the FP8 format to use:

            - FP8E5M2
            - FP8E4M3

    Example
    -------
    .. code-block:: yaml

        example_fake_quant_fp8:
            enabled: True
            layers:
                layer_types: [transformer_layer.layernorm_mlp.fc1]
            transformer_engine:
                FakeQuant:
                    enabled: True
                    quant_format: FP8E5M2
                    gemms_struct:
                    - gemm: fprop
                        tensors: [activation, weight]
                    - gemm: dgrad
                        tensors: [gradient]
    """

    def _supported_formats(self):
        """Returns formats that one can fake quantize tensor to."""
        return ["FP8E4M3", "FP8E5M2", "MXFP8E4M3", "MXFP8E5M2"]

    @api_method
    def fp8_gemm_enabled(
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
        out: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
    ):  # pylint: disable=unused-argument
        """API call used to process the tensor."""

        for key in config.keys():
            if key not in ["gemm", "tensor", "quant_format"]:
                raise ValueError(f'[NVTORCH INSPECT ERROR] Unexpected key in config: "{key}".')

        if "quant_format" not in config:
            raise ValueError(
                f"[NVTORCH INSPECT ERROR] Feature={self.__class__.__name__}, API=process_tensor:"
                f" quant_format missing for Tensor: {tensor_name} in the config yaml for"
                " FakeQuant feature which is a required field"
            )
        if config["quant_format"] not in self._supported_formats():
            raise ValueError(
                f"[NVTORCH INSPECT ERROR] Feature={self.__class__.__name__}, API=process_tensor:"
                f" quant_format: {config['quant_format']} for Tensor: {tensor_name} in the config"
                " yaml for FakeQuant feature is not supported"
            )
        debug_api.log_message(
            f"Feature={self.__class__.__name__}, API=process_tensor: {gemm}, {tensor_name}",
            layer_name,
            extra_cachable_args=(gemm, tensor_name),
        )

        quant_format = config["quant_format"]
        q_tensor = fake_quantize(tensor, quant_format, out=out)
        if dtype is not None:
            q_tensor = q_tensor.to(dtype)
        return q_tensor
