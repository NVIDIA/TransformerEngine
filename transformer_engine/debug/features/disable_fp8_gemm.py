# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DisableFP8GEMM Feature support for nvidia-dlframework-inspect"""

from nvdlfw_inspect.registry import Registry, api_method
from transformer_engine.debug.features.api import TEConfigAPIMapper


@Registry.register_feature(namespace="transformer_engine")
class DisableFP8GEMM(TEConfigAPIMapper):
    """
    GEMM operations are executed in higher precision, even when FP8 autocast is enabled.

    Parameters
    ----------

    gemms: List[str]
        list of gemms to disable

            - fprop
            - dgrad
            - wgrad

    Example
    -------
    .. code-block:: yaml

        example_disable_fp8_gemm:
            enabled: True
            layers:
                layer_types: [fc1]
            transformer_engine:
                DisableFP8GEMM:
                    enabled: True
                    gemms: [dgrad, wgrad]
    """

    @api_method
    def fp8_gemm_enabled(
        self, config, layer_name: str, gemm: str, iteration: int
    ):  # pylint: disable=unused-argument
        """API call responsible for choice between high-precision and FP8 GEMM execution."""

        for key in config:
            if key != "gemm":
                raise ValueError(f'[NVTORCH INSPECT ERROR] Unexpected key in config: "{key}".')

        # If this feature is invoked, then FP8 GEMM is disabled.
        # If not, then default behaviour in TransformerEngineAPI
        # is that fp8_gemm() API call returns True.
        return False
