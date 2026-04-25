/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common/common.h"
#include "common/cast/nvfp4/quantize_nvfp4_1x64_rowwise.cuh"
#include "common/cast/nvfp4/core_nvfp4.cuh"
#include "common/util/ptx.cuh"
#include "common/utils.cuh"

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {
namespace {

#if FP4_TYPE_SUPPORTED

using ptx::FPx2;
using quantization_SF::compute_decoding_scaling_factor;
using core::compute_global_encode_scaling_factor_FP4;

// One CUDA block = one 1x64 K-tile in (row, K-window) layout.
// Threads reduce |x| over the tile, then S_enc = TE global formula on tile amax;
// 1x16 blocks share that S_enc.
template <typename IType>
__global__ void __launch_bounds__(64) nvfp4_rowwise_1x64_per_tile(
    const IType* __restrict__ in, const size_t rows, const size_t cols, const int ld_row_elts,
    uint8_t* __restrict__ out_data,   // raw fp4 bytes (same layout as other NVFP4 rowwise)
    fp8e4m3* __restrict__ row_scales, const size_t scale_stride, float* __restrict__ amax_global,
    const float* __restrict__ noop) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  const int w = static_cast<int>(blockIdx.x);
  const int r = static_cast<int>(blockIdx.y);
  const int c0 = w * 64;
  if (r >= static_cast<int>(rows) || c0 >= static_cast<int>(cols)) {
    return;
  }

  const int win_len = min(64, static_cast<int>(cols) - c0);

  __shared__ float sm[64];
  for (int i = threadIdx.x; i < 64; i += blockDim.x) {
    if (i < win_len) {
      sm[i] = fabsf(static_cast<float>(in[static_cast<size_t>(r) * ld_row_elts + c0 + i]));
    } else {
      sm[i] = 0.f;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float wmx = 0.f;
    for (int i = 0; i < 64; i++) {
      wmx = fmaxf(wmx, sm[i]);
    }
    sm[0] = wmx;
  }
  __syncthreads();

  const float tile_amax = sm[0];
  const float S_enc_tile = compute_global_encode_scaling_factor_FP4(fmaxf(tile_amax, 1e-12f));

  if (amax_global != nullptr) {
    atomicMaxFloat(amax_global, tile_amax);
  }

  if (threadIdx.x < 4) {
    const int b = static_cast<int>(threadIdx.x);
    const int cs = c0 + b * 16;
    if (b * 16 < win_len && cs + 16 <= static_cast<int>(cols)) {
      float bmx = 0.f;
      float vals[16];
      for (int e = 0; e < 16; e++) {
        const float v = static_cast<float>(in[static_cast<size_t>(r) * ld_row_elts + cs + e]);
        vals[e] = v;
        bmx = fmaxf(bmx, fabsf(v));
      }
      const fp8e4m3 s_dec = compute_decoding_scaling_factor(bmx, S_enc_tile);
      const float block_scale = __fdiv_rn(S_enc_tile, static_cast<float>(s_dec));

      const int c16 = cs / 16;
      row_scales[static_cast<size_t>(r) * scale_stride + static_cast<size_t>(c16)] = s_dec;

      using IType2 = FPx2<IType>;
      const size_t row_bytes = static_cast<size_t>(cols) / 2;  // fp4 packed: cols/2 bytes per row
      uint8_t* row_out = out_data + static_cast<size_t>(r) * row_bytes;
      for (int q = 0; q < 4; q++) {
        const int e0 = q * 4;
        IType2 in01{static_cast<IType>(vals[e0]), static_cast<IType>(vals[e0 + 1])};
        IType2 in23{static_cast<IType>(vals[e0 + 2]), static_cast<IType>(vals[e0 + 3])};
        fp4e2m1x4 qu{};
        ptx::mul_cvt_4x(qu, in01, in23, block_scale);
        *reinterpret_cast<fp4e2m1x4*>(row_out + static_cast<size_t>(cs / 2) + static_cast<size_t>(2 * q)) =
            qu;
      }
    }
  }
#endif
}

#endif  // FP4_TYPE_SUPPORTED

}  // namespace

void quantize_rowwise_1x64_local_encode(const Tensor& input, const Tensor& noop, Tensor* output,
                                      const QuantizationConfig& /* quant_config */, cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  CheckNoopTensor(noop, "cast_noop");
  NVTE_CHECK(input.has_data(), "NVFP4 1x64: input has no data.");
  NVTE_CHECK(output->has_data(), "NVFP4 1x64: output has no data.");
  NVTE_CHECK(!output->has_columnwise_data(),
             "NVFP4 rowwise 1x64: columnwise (transpose) path is not supported (no RHT / no GEMM).");
  NVTE_CHECK(output->scale_inv.dptr != nullptr, "NVFP4 1x64: rowwise scale_inv must be allocated.");
  NVTE_CHECK(!output->with_gemm_swizzled_scales, "NVFP4 1x64: expect compact (non-gemm) scales.");
  NVTE_CHECK(output->amax.dptr != nullptr, "NVFP4 1x64: rowwise amax buffer is required.");

  const size_t rows = input.flat_first_dim();
  const size_t cols = input.flat_last_dim();
  if (rows == 0 || cols == 0) {
    return;
  }
  NVTE_CHECK(cols % 16 == 0, "NVFP4 1x64: K must be a multiple of 16 (1x16 block size), got: ",
             cols);
  const size_t n_win = (cols + 63) / 64;

  uint8_t* out_ptr = reinterpret_cast<uint8_t*>(output->data.dptr);
  fp8e4m3* scales = reinterpret_cast<fp8e4m3*>(output->scale_inv.dptr);
  const size_t s_stride = output->scale_inv.shape.size() > 1 ? output->scale_inv.shape[1] : 1;
  float* amax = reinterpret_cast<float*>(output->amax.dptr);
  NVTE_CHECK_CUDA(cudaMemsetAsync(amax, 0, sizeof(float), stream));

  dim3 grid(static_cast<unsigned>(n_win), static_cast<unsigned>(rows), 1);
  constexpr int kBlock = 64;

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input.dtype(), IType, {
        const IType* in_t = reinterpret_cast<const IType*>(input.data.dptr);
        nvfp4_rowwise_1x64_per_tile<IType><<<grid, kBlock, 0, stream>>>(
            in_t, rows, cols, static_cast<int>(cols), out_ptr, scales, s_stride, amax,
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
