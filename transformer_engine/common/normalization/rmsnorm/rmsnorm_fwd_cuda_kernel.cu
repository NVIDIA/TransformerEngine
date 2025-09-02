/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../common.h"
#include "../kernel_traits.h"
#include "rmsnorm_fwd_kernels.cuh"

using namespace transformer_engine::normalization;

template <typename weight_t, typename input_t, typename output_t, typename compute_t,
          typename index_t, int HIDDEN_SIZE, int CTAS_PER_ROW, int WARPS_M, int WARPS_N,
          int BYTES_PER_LDG>
void launch_rmsnorm_fwd_tuned_(LaunchParams<ForwardKernelParams> &launch_params,
                               const bool configure_params) {  // NOLINT(*)
  using Kernel_traits = Kernel_traits<weight_t, input_t, output_t, compute_t, index_t, HIDDEN_SIZE,
                                      CTAS_PER_ROW, WARPS_M, WARPS_N, BYTES_PER_LDG>;
  auto kernel = &rmsnorm_fwd_tuned_kernel<Kernel_traits>;

  if (configure_params) {
    int ctas_per_sm;
    NVTE_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &ctas_per_sm, kernel, Kernel_traits::THREADS_PER_CTA, Kernel_traits::SMEM_BYTES_FWD));
    launch_params.params.ctas_per_row = CTAS_PER_ROW;
    launch_params.params.ctas_per_col =
        launch_params.multiprocessorCount * ctas_per_sm / launch_params.params.ctas_per_row;
    if (Kernel_traits::CTAS_PER_ROW > 1) {
      launch_params.barrier_bytes = 2 * launch_params.params.ctas_per_col * sizeof(index_t);
      launch_params.workspace_bytes = launch_params.params.ctas_per_col * Kernel_traits::WARPS_M *
                                      Kernel_traits::CTAS_PER_ROW *
                                      sizeof(typename Kernel_traits::Stats::stats_t) * 2;
    }
    return;
  }

  if (Kernel_traits::SMEM_BYTES_FWD >= 48 * 1024) {
    NVTE_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         Kernel_traits::SMEM_BYTES_FWD));
  }
  auto stream = launch_params.stream;
  auto ctas_per_col = launch_params.params.ctas_per_col;
  auto ctas_per_row = launch_params.params.ctas_per_row;

  if (ctas_per_row == 1) {
    kernel<<<ctas_per_col, Kernel_traits::THREADS_PER_CTA, Kernel_traits::SMEM_BYTES_FWD, stream>>>(
        launch_params.params);
    NVTE_CHECK_CUDA(cudaGetLastError());
  } else {
    dim3 grid(ctas_per_row * ctas_per_col);
    dim3 block(Kernel_traits::THREADS_PER_CTA);
    void *params_ = reinterpret_cast<void *>(&launch_params.params);
    NVTE_CHECK_CUDA(cudaLaunchCooperativeKernel(reinterpret_cast<void *>(kernel), grid, block,
                                                reinterpret_cast<void **>(&params_),
                                                Kernel_traits::SMEM_BYTES_FWD, stream));
  }
}

template <typename weight_t, typename input_t, typename output_t, typename compute_t,
          typename index_t, int HIDDEN_SIZE, int WARPS_M, int WARPS_N, int BYTES_PER_LDG>
void launch_rmsnorm_fwd_general_(LaunchParams<ForwardKernelParams> &launch_params,
                                 const bool configure_params) {  // NOLINT(*)
  using Kernel_traits = Kernel_traits<weight_t, input_t, output_t, compute_t, index_t, HIDDEN_SIZE,
                                      1, WARPS_M, WARPS_N, BYTES_PER_LDG>;
  auto kernel = &rmsnorm_fwd_general_kernel<Kernel_traits>;
  auto ceil_div = [](int x, int y) -> int { return (x + y - 1) / y; };

  // Configure kernel params
  const int rows = launch_params.params.rows;
  const int cols = launch_params.params.cols;
  int ctas_per_col = launch_params.params.ctas_per_col;
  int ctas_per_row = launch_params.params.ctas_per_row;
  if (configure_params) {
    int ctas_per_sm;
    NVTE_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &ctas_per_sm, kernel, Kernel_traits::THREADS_PER_CTA, 0));
    const int max_ctas = launch_params.multiprocessorCount * ctas_per_sm;
    ctas_per_row = ceil_div(cols, HIDDEN_SIZE);
    ctas_per_col = std::min(ceil_div(rows, WARPS_M), max_ctas / ctas_per_row);
    launch_params.params.ctas_per_row = ctas_per_row;
    launch_params.params.ctas_per_col = ctas_per_col;
    if (launch_params.params.ctas_per_row > 1) {
      launch_params.barrier_bytes = 2 * ctas_per_col * sizeof(index_t);
      launch_params.workspace_bytes =
          (ctas_per_col * WARPS_M * ctas_per_row * sizeof(compute_t) * 2);
    }
    return;
  }

  // Launch kernel
  auto stream = launch_params.stream;
  dim3 grid(ctas_per_row * ctas_per_col);
  dim3 block(Kernel_traits::THREADS_PER_CTA);
  if (ctas_per_row == 1) {
    kernel<<<grid, block, 0, stream>>>(launch_params.params);
    NVTE_CHECK_CUDA(cudaGetLastError());
  } else {
    void *params_ = reinterpret_cast<void *>(&launch_params.params);
    NVTE_CHECK_CUDA(cudaLaunchCooperativeKernel(reinterpret_cast<void *>(kernel), grid, block,
                                                reinterpret_cast<void **>(&params_), 0, stream));
  }
}

#define REGISTER_NORM_LAUNCHER(NORM_TYPE, NORM_STAGE, LAUNCH_TYPE, HIDDEN_SIZE, WTYPE, ITYPE,                   \
                               OTYPE, CTYPE, ...)                                                               \
  namespace {                                                                                                   \
  void                                                                                                          \
      norm_##NORM_TYPE##_##NORM_STAGE##_##LAUNCH_TYPE##_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##OTYPE##_##CTYPE(  \
          LaunchParams<NORM_STAGE##KernelParams> &launch_params, const bool configure_params) {                 \
    launch_rmsnorm_fwd_##LAUNCH_TYPE##_<WTYPE, ITYPE, OTYPE, CTYPE, uint32_t, HIDDEN_SIZE,                      \
                                        __VA_ARGS__>(launch_params, configure_params);                          \
  }                                                                                                             \
  REGISTER_NORM_BASE(                                                                                           \
      NORM_TYPE, NORM_STAGE, LAUNCH_TYPE, HIDDEN_SIZE, WTYPE, ITYPE, OTYPE, CTYPE,                              \
      norm_##NORM_TYPE##_##NORM_STAGE##_##LAUNCH_TYPE##_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##OTYPE##_##CTYPE); \
  }  // namespace

// Create rmsnorm tuned launch function and register. Macro signature:
//  HIDDEN_SIZE, WTYPE, ITYPE, OTYPE, CTYPE, CTAS_PER_ROW, WARPS_M, WARPS_N, BYTES_PER_LDG

REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 512, bf16, bf16, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 512, fp16, fp16, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 512, fp32, fp32, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 512, fp32, fp32, fp32, fp32, 1, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 512, fp16, fp16, fp16, fp32, 1, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 512, bf16, bf16, bf16, fp32, 1, 4, 1, 16);

REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 768, bf16, bf16, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 768, fp16, fp16, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 768, fp32, fp32, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 768, fp32, fp32, fp32, fp32, 1, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 768, fp16, fp16, fp16, fp32, 1, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 768, bf16, bf16, bf16, fp32, 1, 4, 1, 16);

REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 1024, bf16, bf16, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 1024, fp16, fp16, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 1024, fp32, fp32, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 1024, fp32, fp32, fp32, fp32, 1, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 1024, fp16, fp16, fp16, fp32, 1, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 1024, bf16, bf16, bf16, fp32, 1, 4, 1, 16);

REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 2048, bf16, bf16, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 2048, fp16, fp16, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 2048, fp32, fp32, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 2048, fp32, fp32, fp32, fp32, 1, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 2048, fp16, fp16, fp16, fp32, 1, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 2048, bf16, bf16, bf16, fp32, 1, 4, 1, 16);

REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 4096, bf16, bf16, fp8e4m3, fp32, 1, 1, 4, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 4096, fp16, fp16, fp8e4m3, fp32, 1, 1, 4, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 4096, fp32, fp32, fp8e4m3, fp32, 1, 1, 4, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 4096, fp32, fp32, fp32, fp32, 1, 1, 4, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 4096, fp16, fp16, fp16, fp32, 1, 1, 4, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 4096, bf16, bf16, bf16, fp32, 1, 1, 4, 16);

REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 8192, bf16, bf16, fp8e4m3, fp32, 1, 1, 4, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 8192, fp16, fp16, fp8e4m3, fp32, 1, 1, 4, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 8192, fp32, fp32, fp8e4m3, fp32, 1, 1, 4, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 8192, fp32, fp32, fp32, fp32, 1, 1, 4, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 8192, fp16, fp16, fp16, fp32, 1, 1, 4, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, tuned, 8192, bf16, bf16, bf16, fp32, 1, 1, 4, 16);

// Create rmsnorm general launch function and register. Macro signature:
//  HIDDEN_SIZE, WTYPE, ITYPE, OTYPE, CTYPE, WARPS_M, WARPS_N, BYTES_PER_LDG

REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 128, bf16, bf16, fp8e4m3, fp32, 4, 1, 8);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 512, bf16, bf16, fp8e4m3, fp32, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 1024, bf16, bf16, fp8e4m3, fp32, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 2048, bf16, bf16, fp8e4m3, fp32, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 8192, bf16, bf16, fp8e4m3, fp32, 1, 4, 16);

REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 128, fp16, fp16, fp8e4m3, fp32, 4, 1, 8);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 512, fp16, fp16, fp8e4m3, fp32, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 1024, fp16, fp16, fp8e4m3, fp32, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 2048, fp16, fp16, fp8e4m3, fp32, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 8192, fp16, fp16, fp8e4m3, fp32, 1, 4, 16);

REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 128, fp32, fp32, fp8e4m3, fp32, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 512, fp32, fp32, fp8e4m3, fp32, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 1024, fp32, fp32, fp8e4m3, fp32, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 2048, fp32, fp32, fp8e4m3, fp32, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 8192, fp32, fp32, fp8e4m3, fp32, 1, 4, 16);

REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 128, fp32, fp32, fp32, fp32, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 128, fp16, fp16, fp16, fp32, 4, 1, 8);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 128, fp32, fp32, fp16, fp32, 4, 1, 8);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 128, bf16, bf16, bf16, fp32, 4, 1, 8);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 128, fp32, fp32, bf16, fp32, 4, 1, 8);

REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 512, fp32, fp32, fp32, fp32, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 512, fp16, fp16, fp16, fp32, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 512, fp32, fp32, fp16, fp32, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 512, bf16, bf16, bf16, fp32, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 512, fp32, fp32, bf16, fp32, 4, 1, 16);

REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 1024, fp32, fp32, fp32, fp32, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 1024, fp16, fp16, fp16, fp32, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 1024, fp32, fp32, fp16, fp32, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 1024, bf16, bf16, bf16, fp32, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 1024, fp32, fp32, bf16, fp32, 4, 1, 16);

REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 2048, fp32, fp32, fp32, fp32, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 2048, fp16, fp16, fp16, fp32, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 2048, fp32, fp32, fp16, fp32, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 2048, bf16, bf16, bf16, fp32, 4, 1, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 2048, fp32, fp32, bf16, fp32, 4, 1, 16);

REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 8192, fp32, fp32, fp32, fp32, 1, 4, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 8192, fp16, fp16, fp16, fp32, 1, 4, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 8192, fp32, fp32, fp16, fp32, 1, 4, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 8192, bf16, bf16, bf16, fp32, 1, 4, 16);
REGISTER_NORM_LAUNCHER(RMSNorm, Forward, general, 8192, fp32, fp32, bf16, fp32, 1, 4, 16);
