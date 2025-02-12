# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DisableFp8Layer Feature support for nvidia-dlframework-inspect"""

import nvdlfw_inspect.api as debug_api
from nvdlfw_inspect.registry import Registry, api_method


@Registry.register_feature(namespace="transformer_engine")
class DisableFp8Layer:
    """
    Feature to disable FP8 for entire layer or set of layers in Transformer Engine.

    Config:

    To enable the feature in yaml config:
    transformer_engine:
      disable_fp8_layer:
        enabled: True

    """

    @api_method
    def fp8_gemm_enabled(
        self, config, layer_name: str, gemm: str, iteration: int
    ):  # pylint: disable=unused-argument
        """API call responsible for selecting between high-precision and FP8 GEMM execution."""
        for key in config:
            if key not in ["enabled", "gemm"]:
                raise ValueError(f'[NVTORCH INSPECT ERROR] Unexpected key in config: "{key}".')
        # If FP8 training, disable FP8 for the selected layers if this feature is enabled in config.
        debug_api.log_message("FP8 Disabled", layer_name)

        # If this feature is invoked, then fp8 gemm is disabled.
        # If not, then default behaviour in TransformerEngineAPI
        # is that fp8_gemm() API call returns True.
        return False

    def parse_config_and_api(self, config, **_kwargs):
        """Determines whether to run the API
        DisableFp8Layer is the only feature provided by the Transformer Engine
        which does not inherit from TEConfigAPIMapper.

        Explanation of the parse_config_and_api can be found in the nvidia-dlframework-inspect documentation.
        """
        return config["enabled"], None
