/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common/cast/nvfp4/core_nvfp4.cuh"
#include "common/cast/nvfp4/quantize_nvfp4_1x64.cuh"
#include "common/common.h"
#include "common/util/ptx.cuh"
#include "common/utils.cuh"

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {
namespace {

#if FP4_TYPE_SUPPORTED

using core::compute_global_encode_scaling_factor_FP4;
using ptx::FPx2;
using quantization_SF::compute_decoding_scaling_factor;

// One CUDA block = one 64x64 input tile in (M, N) row-major space.
//
// Grid layout: (blockIdx.x, blockIdx.y) = (n_window, m_tile), so each CTA
// owns exactly one 1x64 K-window for the rowwise pass *and* one 1x64
// (transposed-K) M-window for the columnwise pass. With 64 threads per
// CTA, threadIdx.x doubles as "row index in tile" during the rowwise pass
// and "column index in tile" during the columnwise pass.
//
// Either pass can be skipped at runtime by passing nullptr for its three
// output buffers; the SMEM tile load is shared.
//
// SMEM is padded to ``[64][65]`` so the columnwise transpose access
// (``in_sm[e][tid]`` walking down a column) does not fall on the same
// 32-bank lane for every row.
template <typename IType>
__global__ void __launch_bounds__(64)
    nvfp4_1x64_fused_per_tile(const IType* __restrict__ in, const size_t rows, const size_t cols,
                              const int ld_row_elts,
                              // Rowwise outputs (all three are non-null together, or all null).
                              uint8_t* __restrict__ q_row, fp8e4m3* __restrict__ s_dec_row,
                              float* __restrict__ w_amax_row, const size_t s_dec_row_stride,
                              const size_t w_amax_row_stride,
                              // Columnwise (transposed) outputs (all three together, or all null).
                              uint8_t* __restrict__ q_col, fp8e4m3* __restrict__ s_dec_col,
                              float* __restrict__ w_amax_col, const size_t s_dec_col_stride,
                              const size_t w_amax_col_stride, const float* __restrict__ noop) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  const int tile_n = static_cast<int>(blockIdx.x);
  const int tile_m = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);  // 0..63

  const int row_base = tile_m * 64;
  const int col_base = tile_n * 64;
  if (row_base >= static_cast<int>(rows) || col_base >= static_cast<int>(cols)) {
    return;
  }

  const bool do_row = (q_row != nullptr);
  const bool do_col = (q_col != nullptr);

  // 64x64 fp32 staging buffer with +1 column padding to side-step bank
  // conflicts on the columnwise transpose access pattern.
  __shared__ float in_sm[64][65];

  // Cooperative load: thread tid loads its assigned row (64 elements).
  // Padding with zeros keeps the amax reductions correct on M-tail rows
  // (the dispatcher already guarantees full 64-aligned tiles, so this is
  // strictly defensive).
  {
    const int gr = row_base + tid;
    if (gr < static_cast<int>(rows)) {
#pragma unroll
      for (int e = 0; e < 64; e++) {
        const int gc = col_base + e;
        in_sm[tid][e] = static_cast<float>(in[static_cast<size_t>(gr) * ld_row_elts + gc]);
      }
    } else {
#pragma unroll
      for (int e = 0; e < 64; e++) {
        in_sm[tid][e] = 0.f;
      }
    }
  }
  __syncthreads();

  using IType2 = FPx2<IType>;

  // ============================ ROWWISE PASS ============================
  if (do_row && (row_base + tid) < static_cast<int>(rows)) {
    const int r = row_base + tid;

    float wmx = 0.f;
#pragma unroll
    for (int e = 0; e < 64; e++) {
      wmx = fmaxf(wmx, fabsf(in_sm[tid][e]));
    }
    const float S_enc = compute_global_encode_scaling_factor_FP4(fmaxf(wmx, 1e-12f));

    w_amax_row[static_cast<size_t>(r) * w_amax_row_stride + tile_n] = wmx;

    uint8_t* row_out = q_row + static_cast<size_t>(r) * (cols / 2);
#pragma unroll
    for (int b = 0; b < 4; b++) {
      float bmx = 0.f;
      float vals[16];
#pragma unroll
      for (int e = 0; e < 16; e++) {
        const float v = in_sm[tid][b * 16 + e];
        vals[e] = v;
        bmx = fmaxf(bmx, fabsf(v));
      }
      const fp8e4m3 s_dec = compute_decoding_scaling_factor(bmx, S_enc);
      const float s_dec_f = static_cast<float>(s_dec);
      // Match the reference's all-zero-block branch (see
      // ``NVFP4Quantizer1x64Ref``): when ``bmx == 0`` ``s_dec`` saturates
      // to 0 and a naive ``S_enc / 0`` would NaN through the cvt.
      const float block_scale = (s_dec_f == 0.f) ? 0.f : __fdiv_rn(S_enc, s_dec_f);

      const int sub_blk_global = (col_base + b * 16) / 16;
      s_dec_row[static_cast<size_t>(r) * s_dec_row_stride + sub_blk_global] = s_dec;

      const size_t byte_off = static_cast<size_t>(col_base + b * 16) / 2;
#pragma unroll
      for (int q = 0; q < 4; q++) {
        const int e0 = q * 4;
        IType2 in01{static_cast<IType>(vals[e0]), static_cast<IType>(vals[e0 + 1])};
        IType2 in23{static_cast<IType>(vals[e0 + 2]), static_cast<IType>(vals[e0 + 3])};
        fp4e2m1x4 qu{};
        ptx::mul_cvt_4x(qu, in01, in23, block_scale);
        *reinterpret_cast<fp4e2m1x4*>(row_out + byte_off + static_cast<size_t>(2 * q)) = qu;
      }
    }
  }

  // No barrier between rowwise and colwise passes: ``in_sm`` is read-only
  // after the initial load+sync, all writes go to disjoint global memory
  // regions, so the two passes are safely concurrent at warp granularity.

  // =========================== COLUMNWISE PASS ===========================
  if (do_col && (col_base + tid) < static_cast<int>(cols)) {
    const int c = col_base + tid;

    float wmx = 0.f;
#pragma unroll
    for (int e = 0; e < 64; e++) {
      wmx = fmaxf(wmx, fabsf(in_sm[e][tid]));
    }
    const float S_enc = compute_global_encode_scaling_factor_FP4(fmaxf(wmx, 1e-12f));

    w_amax_col[static_cast<size_t>(c) * w_amax_col_stride + tile_m] = wmx;

    // Transposed output: q_col is laid out as (N, M/2). Each "row" of the
    // transposed tensor corresponds to one column of the input, so byte
    // stride is rows/2 along original-M.
    uint8_t* col_out = q_col + static_cast<size_t>(c) * (rows / 2);
#pragma unroll
    for (int b = 0; b < 4; b++) {
      float bmx = 0.f;
      float vals[16];
#pragma unroll
      for (int e = 0; e < 16; e++) {
        const float v = in_sm[b * 16 + e][tid];
        vals[e] = v;
        bmx = fmaxf(bmx, fabsf(v));
      }
      const fp8e4m3 s_dec = compute_decoding_scaling_factor(bmx, S_enc);
      const float s_dec_f = static_cast<float>(s_dec);
      const float block_scale = (s_dec_f == 0.f) ? 0.f : __fdiv_rn(S_enc, s_dec_f);

      const int sub_blk_global = (row_base + b * 16) / 16;
      s_dec_col[static_cast<size_t>(c) * s_dec_col_stride + sub_blk_global] = s_dec;

      const size_t byte_off = static_cast<size_t>(row_base + b * 16) / 2;
#pragma unroll
      for (int q = 0; q < 4; q++) {
        const int e0 = q * 4;
        IType2 in01{static_cast<IType>(vals[e0]), static_cast<IType>(vals[e0 + 1])};
        IType2 in23{static_cast<IType>(vals[e0 + 2]), static_cast<IType>(vals[e0 + 3])};
        fp4e2m1x4 qu{};
        ptx::mul_cvt_4x(qu, in01, in23, block_scale);
        *reinterpret_cast<fp4e2m1x4*>(col_out + byte_off + static_cast<size_t>(2 * q)) = qu;
      }
    }
  }
#endif  // __CUDA_ARCH__ >= 1000
}

#endif  // FP4_TYPE_SUPPORTED

}  // namespace

void quantize_1x64_local_encode(const Tensor& input, const Tensor& noop, Tensor* output,
                                const QuantizationConfig& /* quant_config */, cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  CheckNoopTensor(noop, "cast_noop");
  NVTE_CHECK(input.has_data(), "NVFP4 1x64: input has no data.");
  NVTE_CHECK(output->has_data() || output->has_columnwise_data(),
             "NVFP4 1x64: at least one of rowwise/columnwise output must be allocated.");
  NVTE_CHECK(!output->with_gemm_swizzled_scales,
             "NVFP4 1x64: expects compact (non-gemm) scales on both directions.");

  const size_t rows = input.flat_first_dim();
  const size_t cols = input.flat_last_dim();
  if (rows == 0 || cols == 0) {
    return;
  }
  // 1x16 sub-block inside a 1x64 K-window: both dimensions must be multiples
  // of 64 to keep all four sub-blocks of every window in-bounds for both
  // directions. The PyTorch wrapper (NVFP4Quantizer::create_tensor) sizes the
  // ``amax`` slot accordingly, so this check also pins down the Python side.
  NVTE_CHECK(cols % 64 == 0, "NVFP4 1x64: K (cols) must be a multiple of 64, got: ", cols);
  NVTE_CHECK(rows % 64 == 0, "NVFP4 1x64: M (rows) must be a multiple of 64, got: ", rows);

  uint8_t* q_row = nullptr;
  fp8e4m3* s_dec_row = nullptr;
  float* w_amax_row = nullptr;
  size_t s_dec_row_stride = 0;
  size_t w_amax_row_stride = 0;
  if (output->has_data()) {
    NVTE_CHECK(output->scale_inv.dptr != nullptr,
               "NVFP4 1x64: rowwise scale_inv must be allocated.");
    NVTE_CHECK(output->amax.dptr != nullptr,
               "NVFP4 1x64: rowwise amax (per-window) buffer is required.");
    q_row = reinterpret_cast<uint8_t*>(output->data.dptr);
    s_dec_row = reinterpret_cast<fp8e4m3*>(output->scale_inv.dptr);
    w_amax_row = reinterpret_cast<float*>(output->amax.dptr);
    s_dec_row_stride = output->scale_inv.shape.size() > 1 ? output->scale_inv.shape[1] : 1;
    w_amax_row_stride = output->amax.shape.size() > 1 ? output->amax.shape[1] : 1;
    NVTE_CHECK(s_dec_row_stride == cols / 16,
               "NVFP4 1x64: rowwise scale_inv stride must equal cols/16, got ", s_dec_row_stride);
    NVTE_CHECK(w_amax_row_stride == cols / 64,
               "NVFP4 1x64: rowwise amax stride must equal cols/64, got ", w_amax_row_stride);
  }

  uint8_t* q_col = nullptr;
  fp8e4m3* s_dec_col = nullptr;
  float* w_amax_col = nullptr;
  size_t s_dec_col_stride = 0;
  size_t w_amax_col_stride = 0;
  if (output->has_columnwise_data()) {
    NVTE_CHECK(output->columnwise_scale_inv.dptr != nullptr,
               "NVFP4 1x64: columnwise scale_inv must be allocated.");
    NVTE_CHECK(output->columnwise_amax.dptr != nullptr,
               "NVFP4 1x64: columnwise amax (per-window) buffer is required.");
    q_col = reinterpret_cast<uint8_t*>(output->columnwise_data.dptr);
    s_dec_col = reinterpret_cast<fp8e4m3*>(output->columnwise_scale_inv.dptr);
    w_amax_col = reinterpret_cast<float*>(output->columnwise_amax.dptr);
    s_dec_col_stride =
        output->columnwise_scale_inv.shape.size() > 1 ? output->columnwise_scale_inv.shape[1] : 1;
    w_amax_col_stride =
        output->columnwise_amax.shape.size() > 1 ? output->columnwise_amax.shape[1] : 1;
    NVTE_CHECK(s_dec_col_stride == rows / 16,
               "NVFP4 1x64: columnwise scale_inv stride must equal rows/16, got ",
               s_dec_col_stride);
    NVTE_CHECK(w_amax_col_stride == rows / 64,
               "NVFP4 1x64: columnwise amax stride must equal rows/64, got ", w_amax_col_stride);
  }

  const size_t n_win = cols / 64;
  const size_t m_tiles = rows / 64;
  dim3 grid(static_cast<unsigned>(n_win), static_cast<unsigned>(m_tiles), 1);
  constexpr int kBlock = 64;

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(input.dtype(), IType, {
    const IType* in_t = reinterpret_cast<const IType*>(input.data.dptr);
    nvfp4_1x64_fused_per_tile<IType><<<grid, kBlock, 0, stream>>>(
        in_t, rows, cols, static_cast<int>(cols), q_row, s_dec_row, w_amax_row, s_dec_row_stride,
        w_amax_row_stride, q_col, s_dec_col, w_amax_col, s_dec_col_stride, w_amax_col_stride,
        reinterpret_cast<const float*>(noop.data.dptr));
    NVTE_CHECK_CUDA(cudaGetLastError());
  });

#else
  (void)input;
  (void)noop;
  (void)output;
  (void)stream;
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif
}

}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine
