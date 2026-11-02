# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Configuration for NVFP4 quantize-transpose."""


class NVFP4QuantizeConfig:
    """Instantiation parameters of the CuTE DSL kernel"""

    def __init__(
        self,
        use_stochastic_rounding: bool,
        use_fast_math: bool,
        row_scaled_nvfp4: bool,
        return_transpose: bool,
    ):
        self.USE_STOCHASTIC_ROUNDING = use_stochastic_rounding
        self.USE_FAST_MATH = use_fast_math
        if row_scaled_nvfp4 and return_transpose:
            raise ValueError("row-scaled NVFP4 quantization does not produce a transposed output")
        self.ROW_SCALED_NVFP4 = row_scaled_nvfp4
        self.RETURN_TRANSPOSE = return_transpose

    def __str__(self):
        return (
            f"NVFP4QuantizeConfig(use_stochastic_rounding={self.USE_STOCHASTIC_ROUNDING}, "
            f"use_fast_math={self.USE_FAST_MATH}, "
            f"row_scaled_nvfp4={self.ROW_SCALED_NVFP4}, "
            f"return_transpose={self.RETURN_TRANSPOSE})"
        )

    __repr__ = __str__
