# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch import NVFP4Quantizer

recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)

seed = 12345
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def unpack_fp4(x: torch.Tensor) -> torch.Tensor:
    repeated = x.repeat_interleave(2, dim=1)
    repeated[:, 0::2] &= 0x0F
    repeated[:, 1::2] >>= 4
    return repeated


_FP4_LUT = torch.tensor(
    [
        0.0,  # 0: 0000 - zero
        0.5,  # 1: 0001 - smallest positive normal
        1.0,  # 2: 0010
        1.5,  # 3: 0011
        2.0,  # 4: 0100
        3.0,  # 5: 0101
        4.0,  # 6: 0110
        6.0,  # 7: 0111 - largest positive normal
        -0.0,  # 8: 1000 - negative zero
        -0.5,  # 9: 1001 - smallest negative normal
        -1.0,  # 10: 1010
        -1.5,  # 11: 1011
        -2.0,  # 12: 1100
        -3.0,  # 13: 1101
        -4.0,  # 14: 1110
        -6.0,  # 15: 1111 - largest negative normal
    ],
    dtype=torch.float32,
)


def fp4_to_fp32(fp4: torch.Tensor) -> torch.Tensor:
    # Convert FP4 indices to their corresponding floating point values
    # Each index (0-15) represents a 4-bit FP4 value in E2M1 format
    # Values based on the FP4 E2M1 specification
    fp4_lut = _FP4_LUT.to(fp4.device)
    return fp4_lut[fp4.to(torch.long)]


def dequantize_fp4(qx: torch.Tensor, sx: torch.Tensor, amax: torch.Tensor) -> torch.Tensor:
    sf = sx.repeat_interleave(16, dim=1).view(torch.float8_e4m3fn).to(torch.float32)
    dqx = fp4_to_fp32(unpack_fp4(qx))
    sf = sf[: dqx.shape[0], : dqx.shape[1]]
    dequant = dqx * sf * (amax / (6.0 * 448))
    return dequant


def RHT(x: torch.Tensor) -> torch.Tensor:
    def get_wgrad_sign_vector() -> torch.Tensor:
        """Hard-coded signs for Hadamard transform"""
        return torch.tensor(
            [
                1.0,
                1.0,
                1.0,
                -1.0,
                1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                1.0,
                -1.0,
                1.0,
                -1.0,
                -1.0,
            ],
            dtype=torch.float32,
        )

    def _build_hadamard_matrix(
        size: int, device: torch.device, dtype: torch.dtype, with_random_sign_mask: bool = True
    ) -> torch.Tensor:
        """Construct a Hadamard matrix of given power-of-two size with entries +-1.

        Uses Sylvester construction to avoid SciPy dependency.
        """
        assert (size & (size - 1)) == 0, "Hadamard size must be a power of two"
        h = torch.ones((1, 1), device=device, dtype=torch.float32)
        while h.shape[0] < size:
            h = torch.cat(
                [
                    torch.cat([h, h], dim=1),
                    torch.cat([h, -h], dim=1),
                ],
                dim=0,
            )
        if with_random_sign_mask:
            sign_mat = get_wgrad_sign_vector().to(device) * torch.eye(
                size, device=device, dtype=torch.float32
            )
            h = sign_mat @ h
        return h.to(dtype)

    rht_dim = 16
    # Build H and scale
    H = _build_hadamard_matrix(rht_dim, x.device, x.dtype)
    scale = 1.0 / float(rht_dim) ** 0.5

    # Perform blockwise transform along the last dimension
    original_shape = x.shape
    x_mat = x.contiguous().view(-1, rht_dim)
    # Random sign matrix is identity in this reference (no sign flipping)
    transform = H * scale
    out = x_mat @ transform
    return out.view(original_shape)


def quantize_fp4(
    x: torch.Tensor, use_stochastic_rounding: bool, use_2D: bool, use_RHT: bool
) -> torch.Tensor:
    nvfp4_quantizer = NVFP4Quantizer(
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=use_RHT,
        with_post_rht_amax=True,
        stochastic_rounding=use_stochastic_rounding,
        with_2d_quantization=use_2D,
    )

    x_nvfp4_sut = nvfp4_quantizer(x)
    # Extract data from NVFP4Tensor
    assert x_nvfp4_sut._rowwise_data is not None
    qx: torch.Tensor = x_nvfp4_sut._rowwise_data.view(dtype=torch.uint8)
    assert x_nvfp4_sut._rowwise_scale_inv is not None
    sx: torch.Tensor = x_nvfp4_sut._rowwise_scale_inv
    assert x_nvfp4_sut._columnwise_data is not None
    qx_t: torch.Tensor = x_nvfp4_sut._columnwise_data.view(dtype=torch.uint8)
    assert x_nvfp4_sut._columnwise_scale_inv is not None
    sx_t: torch.Tensor = x_nvfp4_sut._columnwise_scale_inv

    return qx, sx, qx_t, sx_t


def check_quantization_nvfp4_versus_reference(
    x_dtype: torch.dtype, M: int, N: int, use_2D: bool, use_RHT: bool
) -> None:
    device = "cuda"
    torch.manual_seed(seed)
    n_iters = 50

    x = torch.randn((M, N), dtype=x_dtype, device=device) * 2 - 1
    y = x.t().contiguous()
    if use_RHT:
        y = RHT(y)
    amax = torch.max(torch.abs(x)).float()
    q_rn, s_rn, q_t_rn, s_t_rn = quantize_fp4(
        x, use_stochastic_rounding=False, use_2D=use_2D, use_RHT=use_RHT
    )
    dq_rn = dequantize_fp4(q_rn, s_rn, amax)
    dq_t_rn = dequantize_fp4(q_t_rn, s_t_rn, amax)
    error_rn = (dq_rn - x).float()
    me_rn = torch.sqrt((error_rn * error_rn).mean())
    error_t_rn = (dq_t_rn - y).float()
    me_t_rn = torch.sqrt((error_t_rn * error_t_rn).mean())
    sr_result = torch.zeros_like(x).float()
    sr_t_result = torch.zeros_like(x).float().t().contiguous()
    for i in range(n_iters):
        q_sr, s_sr, q_t_sr, s_t_sr = quantize_fp4(
            x, use_stochastic_rounding=True, use_2D=use_2D, use_RHT=use_RHT
        )

        dq_sr = dequantize_fp4(q_sr, s_sr, amax)
        dq_t_sr = dequantize_fp4(q_t_sr, s_t_sr, amax)

        sr_result += dq_sr.float()
        sr_t_result += dq_t_sr.float()

        # sr_result_tmp = sr_result / (i + 1)
        # error_sr = (sr_result_tmp - x).float()
        # me_sr = torch.sqrt((error_sr * error_sr).mean())
        # sr_t_result_tmp = sr_t_result / (i + 1)
        # error_t_sr = (sr_t_result_tmp - y).float()
        # me_t_sr = torch.sqrt((error_t_sr * error_t_sr).mean())
        # print(f"Iteration {i}: RMSE SR: {me_sr:.3e} | RMSE RN: {me_rn:.3e}")
        # print(f"Iteration {i}: RMSE SR_t: {me_t_sr:.3e} | RMSE RN_t: {me_t_rn:.3e}")

    # Get the mean result of the stochastic rounding
    # It should be more accurate than the RN result
    sr_result /= n_iters
    error_sr = (sr_result - x).float()
    me_sr = torch.sqrt((error_sr * error_sr).mean())
    sr_t_result /= n_iters
    error_t_sr = (sr_t_result - y).float()
    me_t_sr = torch.sqrt((error_t_sr * error_t_sr).mean())

    print(f"RMSE SR: {me_sr:.3e} | RMSE RN: {me_rn:.3e}")
    print(f"RMSE SR_t: {me_t_sr:.3e} | RMSE RN_t: {me_t_rn:.3e}")
    assert me_sr < me_rn, "Stochastic rounding failed - error larger than the round to nearest."
    assert me_t_sr < me_t_rn, "Stochastic rounding failed - error larger than the round to nearest."


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        (8192, 8192),
        (8192, 8256),  # to test the nonfused RHT path
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("use_2D", [False, True], ids=str)
@pytest.mark.parametrize("use_RHT", [False, True], ids=str)
def test_quantization_block_tiling_versus_reference(
    x_dtype: torch.dtype,
    use_2D: bool,
    use_RHT: bool,
    M: int,
    N: int,
) -> None:
    if x_dtype == torch.float32 and use_RHT:
        pytest.skip("RHT is only supported with bfloat16")
    check_quantization_nvfp4_versus_reference(
        x_dtype=x_dtype,
        use_2D=use_2D,
        use_RHT=use_RHT,
        M=M,
        N=N,
    )
