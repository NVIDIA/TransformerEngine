# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DisableFP8GEMM Feature support for nvidia-dlframework-inspect

DEPRECATED: This is a backward compatibility alias for DisableQuantizationGEMM.
New code should use DisableQuantizationGEMM instead, which works with all quantization formats.
"""

from nvdlfw_inspect.registry import Registry
from transformer_engine.debug.features.disable_quantization_gemm import DisableQuantizationGEMM


@Registry.register_feature(namespace="transformer_engine")
class DisableFP8GEMM(DisableQuantizationGEMM):
    """
    GEMM operations are executed in higher precision, even when FP8 autocast is enabled.

    .. deprecated::
        Use :class:`DisableQuantizationGEMM` instead. This class is maintained for
        backward compatibility only. DisableQuantizationGEMM works with all quantization
        formats (FP8, NVFP4, etc.), not just FP8.

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
                DisableFP8GEMM:  # Deprecated: use DisableQuantizationGEMM
                    enabled: True
                    gemms: [dgrad, wgrad]
    """
