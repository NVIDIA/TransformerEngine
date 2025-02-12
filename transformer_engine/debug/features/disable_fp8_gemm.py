# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DisableFp8Gemm Feature support for nvidia-dlframework-inspect"""

from nvdlfw_inspect.registry import Registry, api_method
from transformer_engine.debug.features.api import TEConfigAPIMapper


@Registry.register_feature(namespace="transformer_engine")
class DisableFp8Gemm(TEConfigAPIMapper):
    """
    Feature to disable FP8 GEMM in Transformer Engine.

    Config:

    To enable the feature in yaml config:
    transformer_engine:
      disable_fp8_gemm:
        enabled: True
        gemms: gemms list - please look into the Transformer Engine Precision Debug Tools documentation for more information.
    """

    @api_method
    def fp8_gemm_enabled(
        self, config, layer_name: str, gemm: str, iteration: int
    ):  # pylint: disable=unused-argument
        """API call responsible for choice between high-precision and FP8 GEMM execution."""

        for key in config:
            if key != "gemm":
                raise ValueError(f'[NVTORCH INSPECT ERROR] Unexpected key in config: "{key}".')

        # If this feature is invoked, then fp8 gemm is disabled.
        # If not, then default behaviour in TransformerEngineAPI
        # is that fp8_gemm() API call returns True.
        return False
