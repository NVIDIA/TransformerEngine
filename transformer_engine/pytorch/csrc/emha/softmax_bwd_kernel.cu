/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "ln_utils.cuh"  // This header is expected to be included before at least one of the includes below.
#include "softmax.h"
#include "softmax_bwd_kernels.cuh"
#include "softmax_kernel_traits.cuh"

using namespace softmax;

template <typename input_t, typename output_t, typename compute_t,
          typename index_t, int HIDDEN_SIZE, int CTAS_PER_ROW, int WARPS_M,
          int WARPS_N, MaskMode MASK_MODE, int BYTES_PER_LDG>
void launch_(LaunchParams<BwdParams> &launch_params,  // NOLINT(*)
             const bool configure_params) {
  using Kernel_traits =
      Kernel_traits<input_t, output_t, compute_t, index_t, HIDDEN_SIZE,
                    CTAS_PER_ROW, WARPS_M, WARPS_N, MASK_MODE, BYTES_PER_LDG>;
  auto kernel = &softmax_bwd_kernel<Kernel_traits>;
  auto &params = launch_params.params;

  dim3 grid(params.sq / Kernel_traits::ROWS_PER_CTA, params.h, params.b);
  assert(grid.x * Kernel_traits::ROWS_PER_CTA == params.sq);

  if (configure_params) {
    launch_params.elts_per_thread =
        Kernel_traits::LDGS * Kernel_traits::NUM_ELTS;
    return;
  }

  if (Kernel_traits::SMEM_BYTES_BWD >= 48 * 1024) {
    CHECK_CUDA(cudaFuncSetAttribute(kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    Kernel_traits::SMEM_BYTES_BWD));
  }

  auto stream = launch_params.stream;

  kernel<<<grid, Kernel_traits::THREADS_PER_CTA, Kernel_traits::SMEM_BYTES_BWD,
           stream>>>(params);
}

// Create forward launch function and register. Macro signature:
//  HIDDEN_SIZE, ITYPE, OTYPE, CTYPE, WARPS_M, WARPS_N

REGISTER_SOFTMAX_BWD_LAUNCHER(2048, fp16, fp16, fp32, 1, 4);
REGISTER_SOFTMAX_BWD_LAUNCHER(2048, bf16, bf16, fp32, 1, 4);
