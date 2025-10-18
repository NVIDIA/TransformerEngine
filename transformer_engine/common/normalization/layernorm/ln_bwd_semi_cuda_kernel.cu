/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../../common.h"
#include "../common.h"
#include "../kernel_traits.h"
#include "ln_bwd_kernels.cuh"

using namespace transformer_engine::normalization;

template <typename weight_t, typename input_t, typename output_t, typename compute_t,
          typename index_t, int HIDDEN_SIZE, int CTAS_PER_ROW, int WARPS_M, int WARPS_N,
          int BYTES_PER_LDG_MAIN, int BYTES_PER_LDG_FINAL>
void launch_ln_bwd_tuned_(LaunchParams<BackwardKernelParams> &launch_params,
                          const bool configure_params) {  // NOLINT(*)
  using Kernel_traits = Kernel_traits<weight_t, input_t, output_t, compute_t, index_t, HIDDEN_SIZE,
                                      CTAS_PER_ROW, WARPS_M, WARPS_N, BYTES_PER_LDG_MAIN>;
  auto kernel = &ln_bwd_tuned_kernel<Kernel_traits>;

  if (configure_params) {
    int ctas_per_sm;
    NVTE_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &ctas_per_sm, kernel, Kernel_traits::THREADS_PER_CTA, Kernel_traits::SMEM_BYTES));
    launch_params.params.ctas_per_row = CTAS_PER_ROW;
    launch_params.params.ctas_per_col =
        launch_params.multiprocessorCount * ctas_per_sm / launch_params.params.ctas_per_row;
    if (Kernel_traits::CTAS_PER_ROW > 1) {
      launch_params.barrier_bytes = 2 * launch_params.params.ctas_per_col * sizeof(index_t);
      launch_params.workspace_bytes = launch_params.params.ctas_per_col * Kernel_traits::WARPS_M *
                                      Kernel_traits::CTAS_PER_ROW *
                                      sizeof(typename Kernel_traits::reduce_t) * 2;
    }
    launch_params.dgamma_part_bytes =
        launch_params.params.ctas_per_col * launch_params.params.cols * sizeof(compute_t);
    return;
  }

  if (Kernel_traits::SMEM_BYTES >= 48 * 1024) {
    NVTE_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         Kernel_traits::SMEM_BYTES));
  }
  auto stream = launch_params.stream;
  auto ctas_per_col = launch_params.params.ctas_per_col;
  auto ctas_per_row = launch_params.params.ctas_per_row;

  if (ctas_per_row == 1) {
    kernel<<<ctas_per_col, Kernel_traits::THREADS_PER_CTA, Kernel_traits::SMEM_BYTES, stream>>>(
        launch_params.params);
    NVTE_CHECK_CUDA(cudaGetLastError());
  } else {
    dim3 grid(ctas_per_row * ctas_per_col);
    dim3 block(Kernel_traits::THREADS_PER_CTA);
    void *params_ = reinterpret_cast<void *>(&launch_params.params);
    NVTE_CHECK_CUDA(cudaLaunchCooperativeKernel(reinterpret_cast<void *>(kernel), grid, block,
                                                reinterpret_cast<void **>(&params_),
                                                Kernel_traits::SMEM_BYTES, stream));
  }

  using Kernel_traits_f =
      Kernel_traits_finalize<HIDDEN_SIZE, weight_t, input_t, output_t, compute_t, index_t,
                             32 * 32,  // THREADS_PER_CTA
                             BYTES_PER_LDG_FINAL>;

  auto kernel_f = &ln_bwd_finalize_tuned_kernel<Kernel_traits_f>;
  kernel_f<<<Kernel_traits_f::CTAS, Kernel_traits_f::THREADS_PER_CTA, 0, stream>>>(
      launch_params.params);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename weight_t, typename input_t, typename output_t, typename compute_t,
          typename index_t, int HIDDEN_SIZE, int WARPS_M, int WARPS_N, int BYTES_PER_LDG_MAIN,
          int BYTES_PER_LDG_FINAL>
void launch_ln_bwd_general_(LaunchParams<BackwardKernelParams> &launch_params,
                            const bool configure_params) {  // NOLINT(*)
  auto ceil_div = [](int x, int y) -> int { return (x + y - 1) / y; };

  // Instantiate kernel
  using Kernel_traits = Kernel_traits<weight_t, input_t, output_t, compute_t, index_t, HIDDEN_SIZE,
                                      1, WARPS_M, WARPS_N, BYTES_PER_LDG_MAIN>;
  auto kernel = &ln_bwd_general_kernel<Kernel_traits>;

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
          (ctas_per_col * WARPS_M * ctas_per_row * sizeof(typename Kernel_traits::reduce_t) * 2);
    }
    launch_params.dgamma_part_bytes = ctas_per_col * cols * sizeof(compute_t);
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

  // Launch finalization kernel
  constexpr uint32_t WARPS_M_FINAL = 4;
  constexpr uint32_t WARPS_N_FINAL = 1;
  constexpr uint32_t ELTS_N_PER_CTA_FINAL =
      (Kernel_traits::THREADS_PER_WARP * WARPS_N_FINAL * BYTES_PER_LDG_FINAL / sizeof(compute_t));
  auto kernel_final =
      &ln_bwd_finalize_general_kernel<weight_t, compute_t, WARPS_M_FINAL, WARPS_N_FINAL,
                                      BYTES_PER_LDG_FINAL, Kernel_traits::THREADS_PER_WARP>;
  dim3 block_final(Kernel_traits::THREADS_PER_WARP * WARPS_N_FINAL, WARPS_M_FINAL);
  dim3 grid_final(ceil_div(cols, ELTS_N_PER_CTA_FINAL), 1);
  kernel_final<<<grid_final, block_final, 0, stream>>>(launch_params.params);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

#define REGISTER_NORM_LAUNCHER(NORM_TYPE, NORM_STAGE, LAUNCH_TYPE, HIDDEN_SIZE, WTYPE, ITYPE,                   \
                               OTYPE, CTYPE, ...)                                                               \
  namespace {                                                                                                   \
  void                                                                                                          \
      norm_##NORM_TYPE##_##NORM_STAGE##_##LAUNCH_TYPE##_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##OTYPE##_##CTYPE(  \
          LaunchParams<NORM_STAGE##KernelParams> &launch_params, const bool configure_params) {                 \
    launch_ln_bwd_##LAUNCH_TYPE##_<WTYPE, ITYPE, OTYPE, CTYPE, uint32_t, HIDDEN_SIZE,                           \
                                   __VA_ARGS__>(launch_params, configure_params);                               \
  }                                                                                                             \
  REGISTER_NORM_BASE(                                                                                           \
      NORM_TYPE, NORM_STAGE, LAUNCH_TYPE, HIDDEN_SIZE, WTYPE, ITYPE, OTYPE, CTYPE,                              \
      norm_##NORM_TYPE##_##NORM_STAGE##_##LAUNCH_TYPE##_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##OTYPE##_##CTYPE); \
  }  //  namespace

// Create tuned launch function and register. Macro signature:
//  HIDDEN_SIZE, WTYPE, ITYPE, OTYPE, CTYPE, CTAS_PER_ROW, ...
//                             WARPS_M, WARPS_N, BYTES_PER_LDG, BYTES_PER_LDG_FINAL

REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 768, fp32, fp32, fp32, fp32, 1, 4, 1, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 768, fp16, fp16, fp16, fp32, 1, 4, 1, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 768, fp16, fp32, fp16, fp32, 1, 4, 1, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 768, bf16, bf16, bf16, fp32, 1, 4, 1, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 768, bf16, fp32, bf16, fp32, 1, 4, 1, 16, 4);

REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 1024, fp32, fp32, fp32, fp32, 1, 4, 1, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 1024, fp16, fp16, fp16, fp32, 1, 4, 1, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 1024, fp16, fp32, fp16, fp32, 1, 4, 1, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 1024, bf16, bf16, bf16, fp32, 1, 4, 1, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 1024, bf16, fp32, bf16, fp32, 1, 4, 1, 16, 4);

REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 1536, fp32, fp32, fp32, fp32, 1, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 1536, fp16, fp16, fp16, fp32, 1, 1, 4, 8, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 1536, fp16, fp32, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 1536, bf16, bf16, bf16, fp32, 1, 1, 4, 8, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 1536, bf16, fp32, bf16, fp32, 1, 1, 4, 16, 4);

REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 2048, fp32, fp32, fp32, fp32, 1, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 2048, fp16, fp16, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 2048, fp16, fp32, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 2048, bf16, bf16, bf16, fp32, 1, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 2048, bf16, fp32, bf16, fp32, 1, 1, 4, 16, 4);

REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 2304, fp32, fp32, fp32, fp32, 1, 1, 4, 8, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 2304, fp16, fp16, fp16, fp32, 1, 1, 4, 4, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 2304, fp16, fp32, fp16, fp32, 1, 1, 4, 8, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 2304, bf16, bf16, bf16, fp32, 1, 1, 4, 4, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 2304, bf16, fp32, bf16, fp32, 1, 1, 4, 8, 4);

REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 3072, fp32, fp32, fp32, fp32, 1, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 3072, fp16, fp16, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 3072, fp16, fp32, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 3072, bf16, bf16, bf16, fp32, 1, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 3072, bf16, fp32, bf16, fp32, 1, 1, 4, 16, 4);

REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 3840, fp32, fp32, fp32, fp32, 1, 1, 4, 8, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 3840, fp16, fp16, fp16, fp32, 1, 1, 4, 4, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 3840, fp16, fp32, fp16, fp32, 1, 1, 4, 8, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 3840, bf16, bf16, bf16, fp32, 1, 1, 4, 4, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 3840, bf16, fp32, bf16, fp32, 1, 1, 4, 8, 4);

REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 4096, fp32, fp32, fp32, fp32, 1, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 4096, fp16, fp16, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 4096, fp16, fp32, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 4096, bf16, bf16, bf16, fp32, 1, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 4096, bf16, fp32, bf16, fp32, 1, 1, 4, 16, 4);

REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 5120, fp32, fp32, fp32, fp32, 1, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 5120, fp16, fp16, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 5120, fp16, fp32, fp16, fp32, 1, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 5120, bf16, bf16, bf16, fp32, 1, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 5120, bf16, fp32, bf16, fp32, 1, 1, 4, 16, 4);

REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 6144, fp32, fp32, fp32, fp32, 1, 1, 8, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 6144, fp16, fp16, fp16, fp32, 1, 1, 8, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 6144, fp16, fp32, fp16, fp32, 1, 1, 8, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 6144, bf16, bf16, bf16, fp32, 1, 1, 8, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 6144, bf16, fp32, bf16, fp32, 1, 1, 8, 16, 4);

REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 8192, fp32, fp32, fp32, fp32, 2, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 8192, fp16, fp16, fp16, fp32, 2, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 8192, fp16, fp32, fp16, fp32, 2, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 8192, bf16, bf16, bf16, fp32, 2, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 8192, bf16, fp32, bf16, fp32, 2, 1, 4, 16, 4);

REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 10240, fp32, fp32, fp32, fp32, 2, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 10240, fp16, fp16, fp16, fp32, 2, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 10240, fp16, fp32, fp16, fp32, 2, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 10240, bf16, bf16, bf16, fp32, 2, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 10240, bf16, fp32, bf16, fp32, 2, 1, 4, 16, 4);

REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 12288, fp32, fp32, fp32, fp32, 4, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 12288, fp16, fp16, fp16, fp32, 4, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 12288, fp16, fp32, fp16, fp32, 4, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 12288, bf16, bf16, bf16, fp32, 4, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 12288, bf16, fp32, bf16, fp32, 4, 1, 4, 16, 4);

REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 12800, fp32, fp32, fp32, fp32, 5, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 12800, fp16, fp16, fp16, fp32, 5, 1, 4, 8, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 12800, fp16, fp32, fp16, fp32, 5, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 12800, bf16, bf16, bf16, fp32, 5, 1, 4, 8, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 12800, bf16, fp32, bf16, fp32, 5, 1, 4, 16, 4);

REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 15360, fp32, fp32, fp32, fp32, 4, 1, 4, 8, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 15360, fp16, fp16, fp16, fp32, 4, 1, 4, 4, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 15360, fp16, fp32, fp16, fp32, 4, 1, 4, 8, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 15360, bf16, bf16, bf16, fp32, 4, 1, 4, 4, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 15360, bf16, fp32, bf16, fp32, 4, 1, 4, 8, 4);

REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 16384, fp32, fp32, fp32, fp32, 4, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 16384, fp16, fp16, fp16, fp32, 4, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 16384, fp16, fp32, fp16, fp32, 4, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 16384, bf16, bf16, bf16, fp32, 4, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 16384, bf16, fp32, bf16, fp32, 4, 1, 4, 16, 4);

REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 18432, fp32, fp32, fp32, fp32, 4, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 18432, fp16, fp16, fp16, fp32, 4, 1, 4, 8, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 18432, fp16, fp32, fp16, fp32, 4, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 18432, bf16, bf16, bf16, fp32, 4, 1, 4, 8, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 18432, bf16, fp32, bf16, fp32, 4, 1, 4, 16, 4);

REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 20480, fp32, fp32, fp32, fp32, 4, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 20480, fp16, fp16, fp16, fp32, 4, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 20480, fp16, fp32, fp16, fp32, 4, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 20480, bf16, bf16, bf16, fp32, 4, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 20480, bf16, fp32, bf16, fp32, 4, 1, 4, 16, 4);

REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 24576, fp32, fp32, fp32, fp32, 4, 1, 8, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 24576, fp16, fp16, fp16, fp32, 4, 1, 8, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 24576, fp16, fp32, fp16, fp32, 4, 1, 8, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 24576, bf16, bf16, bf16, fp32, 4, 1, 8, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 24576, bf16, fp32, bf16, fp32, 4, 1, 8, 16, 4);

REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 25600, fp32, fp32, fp32, fp32, 5, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 25600, fp16, fp16, fp16, fp32, 5, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 25600, fp16, fp32, fp16, fp32, 5, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 25600, bf16, bf16, bf16, fp32, 5, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 25600, bf16, fp32, bf16, fp32, 5, 1, 4, 16, 4);

REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 30720, fp32, fp32, fp32, fp32, 4, 1, 8, 8, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 30720, fp16, fp16, fp16, fp32, 4, 1, 8, 4, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 30720, fp16, fp32, fp16, fp32, 4, 1, 8, 8, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 30720, bf16, bf16, bf16, fp32, 4, 1, 8, 4, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 30720, bf16, fp32, bf16, fp32, 4, 1, 8, 8, 4);

REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 32768, fp32, fp32, fp32, fp32, 4, 1, 8, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 32768, fp16, fp16, fp16, fp32, 4, 1, 8, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 32768, fp16, fp32, fp16, fp32, 4, 1, 8, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 32768, bf16, bf16, bf16, fp32, 4, 1, 8, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 32768, bf16, fp32, bf16, fp32, 4, 1, 8, 16, 4);

REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 40960, fp32, fp32, fp32, fp32, 4, 1, 8, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 40960, fp16, fp16, fp16, fp32, 4, 1, 8, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 40960, fp16, fp32, fp16, fp32, 4, 1, 8, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 40960, bf16, bf16, bf16, fp32, 4, 1, 8, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 40960, bf16, fp32, bf16, fp32, 4, 1, 8, 16, 4);

REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 49152, fp32, fp32, fp32, fp32, 8, 1, 8, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 49152, fp16, fp16, fp16, fp32, 8, 1, 8, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 49152, fp16, fp32, fp16, fp32, 8, 1, 8, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 49152, bf16, bf16, bf16, fp32, 8, 1, 8, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 49152, bf16, fp32, bf16, fp32, 8, 1, 8, 16, 4);

REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 65536, fp32, fp32, fp32, fp32, 8, 1, 8, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 65536, fp16, fp16, fp16, fp32, 8, 1, 8, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 65536, fp16, fp32, fp16, fp32, 8, 1, 8, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 65536, bf16, bf16, bf16, fp32, 8, 1, 8, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, tuned, 65536, bf16, fp32, bf16, fp32, 8, 1, 8, 16, 4);

// Create general launch function and register. Macro signature:
//  HIDDEN_SIZE, WTYPE, ITYPE, OTYPE, CTYPE, ...
//                             WARPS_M, WARPS_N, BYTES_PER_LDG, BYTES_PER_LDG_FINAL

REGISTER_NORM_LAUNCHER(LayerNorm, Backward, general, 128, fp32, fp32, fp32, fp32, 4, 1, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, general, 128, fp16, fp16, fp16, fp32, 4, 1, 8, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, general, 128, fp16, fp32, fp16, fp32, 4, 1, 8, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, general, 128, bf16, bf16, bf16, fp32, 4, 1, 8, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, general, 128, bf16, fp32, bf16, fp32, 4, 1, 8, 4);

REGISTER_NORM_LAUNCHER(LayerNorm, Backward, general, 512, fp32, fp32, fp32, fp32, 4, 1, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, general, 512, fp16, fp16, fp16, fp32, 4, 1, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, general, 512, fp16, fp32, fp16, fp32, 4, 1, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, general, 512, bf16, bf16, bf16, fp32, 4, 1, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, general, 512, bf16, fp32, bf16, fp32, 4, 1, 16, 4);

REGISTER_NORM_LAUNCHER(LayerNorm, Backward, general, 1024, fp32, fp32, fp32, fp32, 4, 1, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, general, 1024, fp16, fp16, fp16, fp32, 4, 1, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, general, 1024, fp16, fp32, fp16, fp32, 4, 1, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, general, 1024, bf16, bf16, bf16, fp32, 4, 1, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, general, 1024, bf16, fp32, bf16, fp32, 4, 1, 16, 4);

REGISTER_NORM_LAUNCHER(LayerNorm, Backward, general, 2048, fp32, fp32, fp32, fp32, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, general, 2048, fp16, fp16, fp16, fp32, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, general, 2048, fp16, fp32, fp16, fp32, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, general, 2048, bf16, bf16, bf16, fp32, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, general, 2048, bf16, fp32, bf16, fp32, 1, 4, 16, 4);

REGISTER_NORM_LAUNCHER(LayerNorm, Backward, general, 4096, fp32, fp32, fp32, fp32, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, general, 4096, fp16, fp16, fp16, fp32, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, general, 4096, fp16, fp32, fp16, fp32, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, general, 4096, bf16, bf16, bf16, fp32, 1, 4, 16, 4);
REGISTER_NORM_LAUNCHER(LayerNorm, Backward, general, 4096, bf16, fp32, bf16, fp32, 1, 4, 16, 4);
