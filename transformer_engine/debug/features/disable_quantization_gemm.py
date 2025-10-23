# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DisableQuantizationGEMM Feature support for nvidia-dlframework-inspect"""

from nvdlfw_inspect.registry import Registry, api_method
from transformer_engine.debug.features.api import TEConfigAPIMapper


@Registry.register_feature(namespace="transformer_engine")
class DisableQuantizationGEMM(TEConfigAPIMapper):
    """
    Disables specific GEMM operations from using quantization, forcing high-precision execution.
    
    Works with any quantization format (FP8, NVFP4, etc.).

    Parameters
    ----------

    gemms: List[str]
        list of gemms to disable quantization for

            - fprop
            - dgrad
            - wgrad

    Example
    -------
    .. code-block:: yaml

        example_disable_quantization_gemm:
            enabled: True
            layers:
                layer_types: [fc1]
            transformer_engine:
                DisableQuantizationGEMM:
                    enabled: True
                    gemms: [dgrad, wgrad]
    """

    @api_method
    def fp8_gemm_enabled(
        self, config, layer_name: str, gemm: str, iteration: int
    ):  # pylint: disable=unused-argument
        """API call responsible for choice between high-precision and quantized GEMM execution.
        
        Note: Method name kept as 'fp8_gemm_enabled' for backward compatibility with the debug API,
        but it applies to all quantization formats (FP8, NVFP4, etc.).
        """

        for key in config:
            if key != "gemm":
                raise ValueError(f'[NVTORCH INSPECT ERROR] Unexpected key in config: "{key}".')

        # If this feature is invoked, then quantized GEMM is disabled (returns to high precision).
        # If not, then default behavior in TransformerEngineAPI
        # is that fp8_gemm() API call returns True.
        return False, iteration + 1

