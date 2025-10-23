# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DisableFP8Layer Feature support for nvidia-dlframework-inspect

DEPRECATED: This is a backward compatibility alias for DisableQuantizationLayer.
New code should use DisableQuantizationLayer instead, which works with all quantization formats.
"""

from nvdlfw_inspect.registry import Registry
from transformer_engine.debug.features.disable_quantization_layer import DisableQuantizationLayer


@Registry.register_feature(namespace="transformer_engine")
class DisableFP8Layer(DisableQuantizationLayer):
    """
    Disables all FP8 GEMMs in the layer.

    .. deprecated::
        Use :class:`DisableQuantizationLayer` instead. This class is maintained for
        backward compatibility only. DisableQuantizationLayer works with all quantization
        formats (FP8, NVFP4, etc.), not just FP8.

    Example
    -------
    .. code-block:: yaml

        example_disable_fp8_layer:
            enabled: True
            layers:
                layer_types: [fc1]
            transformer_engine:
                DisableFP8Layer:  # Deprecated: use DisableQuantizationLayer
                    enabled: True
    """

    pass  # Inherits all functionality from DisableQuantizationLayer
