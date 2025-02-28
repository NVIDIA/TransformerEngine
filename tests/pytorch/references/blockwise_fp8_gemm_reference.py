from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def fused_fma_kernel(y_ptr, x_ptr, s_ptr, M, N, y_str0, y_str1, BLOCK: tl.constexpr = 128):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < M * N

    row = idx // N
    col = idx % N

    y_offset = row * y_str0 + col * y_str1
    x_offset = row * N + col
    s_offset = row * N + col

    y = tl.load(y_ptr + y_offset, mask=mask)
    x = tl.load(x_ptr + x_offset, mask=mask)
    s = tl.load(s_ptr + s_offset, mask=mask)

    tl.store(y_ptr + y_offset, tl.fma(x, s, y), mask=mask)


def fused_fma(y, x, s, BLOCK=128):
    """
    Fused multiply-add operation (y = y + x * s).

    PyTorch does not provide a direct FMA equivalent (torch.addcmul is not bitwise equivalent to this operation).
    This function also supports cases where 'y' is non-contiguous in memory.
    """

    assert (
        y.shape == x.shape == s.shape and y.dim() == 2
    ), "All tensors must be 2D with the same shape"
    assert x.is_contiguous() and s.is_contiguous(), "x and s must be contiguous"

    M, N = y.shape
    grid = ((M * N + BLOCK - 1) // BLOCK,)

    fused_fma_kernel[grid](y, x, s, M, N, *y.stride(), BLOCK)

    return y


class CuBLASRefBlockwiseGemm:
    """
    A cuBLAS compatible reference implementation of subchannel GEMM.
    """

    def qgemm(
        self,
        qx: torch.Tensor,
        qw: torch.Tensor,
        out_dtype: torch.dtype,
        demunged_sx: torch.Tensor,
        demunged_sw: torch.Tensor,
        quant_tile_shape_x: Tuple[int, int],
        quant_tile_shape_w: Tuple[int, int],
        bias: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        accumulate: bool = False,
        use_split_accumulator: bool = False,
    ) -> torch.Tensor:
        # demunge scale shapes for cuBLAS
        is_a_1d_scaled = quant_tile_shape_x[0] == 1
        is_b_1d_scaled = quant_tile_shape_w[0] == 1
        M, K = qx.shape
        N, K = qw.shape

        # mm_tile_shape = (tile_m, tile_n, tile_k)
        mm_tile_shape = (
            quant_tile_shape_x[0],
            quant_tile_shape_w[0],
            quant_tile_shape_w[1],
        )
        if bias is not None and bias.numel():
            # To match cuBLAS more closely when bias is applied,
            # the reference accumulates into float32, and cast to
            # bfloat16 is deferred until after the GEMM.
            out_dtype_for_ref = torch.float32
        else:
            out_dtype_for_ref = out_dtype
        y = self.qgemm_blockwise_2d(
            qx,
            qw,
            out_dtype_for_ref,
            demunged_sx,
            demunged_sw,
            mm_tile_shape,
            use_split_accumulator,
            is_a_1d_scaled,
            is_b_1d_scaled,
        )
        if bias is not None and bias.numel():
            y += bias
            y = y.to(dtype=out_dtype)
        # cublas accumulation first convert to output dtype, then accumulate.
        if accumulate:
            assert out is not None
            y = y + out
        else:
            assert out is None, "Output tensor should be None when accumulate is False."

        return y

    @classmethod
    def qgemm_blockwise_2d(
        cls,
        qx: torch.Tensor,
        qw: torch.Tensor,
        out_dtype: torch.dtype,
        sx: torch.Tensor,
        sw: torch.Tensor,
        mm_tile_shape: Tuple[int, int, int],
        use_split_accumulator: bool,
        is_a_1d_scaled: bool,
        is_b_1d_scaled: bool,
    ) -> torch.Tensor:
        """
        Difference between cuBLAS and CUTLASS GEMM implementations:
            - cuBLAS accumulation equation: use different equation for each scaling mode.
            - For accumulation C in epiloge, it first convert C to output dtype, then accumulate.
        """

        M, K = qx.shape
        N, K_w = qw.shape
        assert K == K_w, "K dimension mismatch between qx and qw"

        tile_len = 128
        # Calculate grid sizes without padding
        grid_m = (M + tile_len - 1) // tile_len
        grid_n = (N + tile_len - 1) // tile_len
        grid_k = (K + tile_len - 1) // tile_len

        block_m, block_n, block_k = mm_tile_shape
        scale_m_per_tile = tile_len // block_m
        scale_n_per_tile = tile_len // block_n
        assert block_k == tile_len, "block_k must be equal to tile_len"

        # Notes on making the reference implementation numerically equivalent to Cast Blockwise FP8 GEMM:
        # 1) When using split_accumulate in FP8 GEMM, every 4 QMMA partial accumulation results are accumulated into float32 registers.
        # 2) Partial accumulation results are accumulated using FMA (Fused Multiply-Add) instructions to apply scaling factors, as in: y += partial_y * scale
        y = torch.zeros(M, N, dtype=torch.float32, device=qx.device)

        # Validate shapes of sx and sw
        scale_m_per_tensor = (M + block_m - 1) // block_m
        scale_n_per_tensor = (N + block_n - 1) // block_n
        assert sx.shape == (
            scale_m_per_tensor,
            grid_k,
        ), f"sx shape mismatch: expected ({scale_m_per_tensor}, {grid_k}), got {sx.shape}"
        assert sw.shape == (
            scale_n_per_tensor,
            grid_k,
        ), f"sw shape mismatch: expected ({scale_n_per_tensor}, {grid_k}), got {sw.shape}"

        for i in range(grid_m):
            m_start = i * tile_len
            m_end = min(m_start + tile_len, M)
            m_size = m_end - m_start

            for j in range(grid_n):
                n_start = j * tile_len
                n_end = min(n_start + tile_len, N)
                n_size = n_end - n_start

                y_block = y[m_start:m_end, n_start:n_end]

                for k in range(grid_k):
                    k_start = k * tile_len
                    k_end = min(k_start + tile_len, K)
                    k_size = k_end - k_start

                    qx_block = (
                        qx[m_start:m_end, k_start:k_end].clone().contiguous()
                    )  # Shape: [m_size, k_size]
                    qw_block = (
                        qw[n_start:n_end, k_start:k_end].clone().contiguous()
                    )  # Shape: [n_size, k_size]

                    # Extract scaling factors for the current blocks
                    sx_block = sx[i * scale_m_per_tile : (i + 1) * scale_m_per_tile, k].unsqueeze(
                        -1
                    )
                    sw_block = sw[j * scale_n_per_tile : (j + 1) * scale_n_per_tile, k].unsqueeze(0)

                    # Perform qgemm with scaling factors fused in the GEMM
                    # Accumulate should be in float32 format, which aligns with the split_accumulate in FP8 GEMM
                    one = torch.tensor(1.0, dtype=torch.float32, device=qx.device)
                    y_partial = torch._scaled_mm(
                        qx_block,
                        qw_block.t(),
                        scale_a=one,
                        scale_b=one,
                        out_dtype=torch.float32,
                        use_fast_accum=not use_split_accumulator,
                    )

                    # Accumulate the partial result
                    if is_a_1d_scaled and is_b_1d_scaled:
                        # 1Dx1D
                        # CuBLAS accumulation equation: y += (y * scale_a) * scale_b
                        y_partial = y_partial * sx_block
                        # Fuse multiplication and addition to align with the split_accumulate in FP8 GEMM
                        # y_block.add_(y_partial, alpha=scale.item())
                        fused_fma(
                            y_block,
                            y_partial,
                            sw_block.expand_as(y_partial).contiguous(),
                        )
                    elif not is_a_1d_scaled and is_b_1d_scaled:
                        # 2Dx1D
                        # CuBLAS accumulation equation: y += (y * scale_b) * scale_a
                        y_partial = y_partial * sw_block
                        fused_fma(
                            y_block,
                            y_partial,
                            sx_block.expand_as(y_partial).contiguous(),
                        )
                    elif is_a_1d_scaled and not is_b_1d_scaled:
                        # 1Dx2D
                        # CuBLAS accumulation equation: y += (y * scale_a) * scale_b
                        y_partial = y_partial * sx_block
                        fused_fma(
                            y_block,
                            y_partial,
                            sw_block.expand_as(y_partial).contiguous(),
                        )
                    else:
                        scale = sx_block * sw_block
                        fused_fma(y_block, y_partial, scale.expand_as(y_partial).contiguous())

        y = y.to(out_dtype)
        return y
