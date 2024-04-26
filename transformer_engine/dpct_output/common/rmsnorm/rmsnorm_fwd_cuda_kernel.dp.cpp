/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <dpct/dnnl_utils.hpp>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "rmsnorm.h"
#include "rmsnorm_fwd_kernels.dp.hpp"
#include "rmsnorm_kernel_traits.h"
#include <cmath>

using namespace transformer_engine::rmsnorm;

template <typename weight_t, typename input_t, typename output_t,
          typename compute_t, typename index_t, int HIDDEN_SIZE,
          int CTAS_PER_ROW, int WARPS_M, int WARPS_N, int BYTES_PER_LDG>
void launch_tuned_(LaunchParams<FwdParams> &launch_params,
                   const bool configure_params) try { // NOLINT(*)
    using Kernel_traits = Kernel_traits<weight_t, input_t, output_t, compute_t, index_t,
                                        HIDDEN_SIZE, CTAS_PER_ROW, WARPS_M, WARPS_N, BYTES_PER_LDG>;
    auto kernel = &rmsnorm_fwd_tuned_kernel<Kernel_traits>;

    if (configure_params) {
        int ctas_per_sm;
        /*
        DPCT1007:227: Migration of cudaOccupancyMaxActiveBlocksPerMultiprocessor
        is not supported.
        */
        dpct::err0 status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &ctas_per_sm, kernel, Kernel_traits::THREADS_PER_CTA,
            Kernel_traits::SMEM_BYTES_FWD);
        launch_params.params.ctas_per_row = CTAS_PER_ROW;
        launch_params.params.ctas_per_col =
            launch_params.multiprocessorCount * ctas_per_sm / launch_params.params.ctas_per_row;
        launch_params.barrier_size = 0;
        launch_params.workspace_bytes = 0;
        if (Kernel_traits::CTAS_PER_ROW > 1) {
            launch_params.barrier_size = 2 * launch_params.params.ctas_per_col;
            launch_params.workspace_bytes = launch_params.params.ctas_per_col *
                                            Kernel_traits::WARPS_M * Kernel_traits::CTAS_PER_ROW *
                                            sizeof(typename Kernel_traits::Stats::stats_t) * 2;
        }
        return;
    }

    if (Kernel_traits::SMEM_BYTES_FWD >= 48 * 1024) {
        /*
        DPCT1009:226: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        /*
        DPCT1027:228: The call to cudaFuncSetAttribute was replaced with 0
        because SYCL currently does not support corresponding setting.
        */
        NVTE_CHECK_CUDA(0);
    }
    auto stream = launch_params.stream;
    auto ctas_per_col = launch_params.params.ctas_per_col;
    auto ctas_per_row = launch_params.params.ctas_per_row;

    if (ctas_per_row == 1) {
        /*
        DPCT1049:10: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, ctas_per_col) *
                    sycl::range<3>(1, 1, Kernel_traits::THREADS_PER_CTA),
                sycl::range<3>(1, 1, Kernel_traits::THREADS_PER_CTA)),
            [=](sycl::nd_item<3> item_ct1) {
                (launch_params.params);
            });
    } else {
        sycl::range<3> grid(1, 1, ctas_per_row * ctas_per_col);
        sycl::range<3> block(1, 1, Kernel_traits::THREADS_PER_CTA);
        void *params_ = reinterpret_cast<void *>(&launch_params.params);
        /*
        DPCT1007:11: Migration of cudaLaunchCooperativeKernel is not supported.
        */
        cudaLaunchCooperativeKernel((void *)kernel, grid, block,
                                    (void **)&params_, // NOLINT(*)
                                    Kernel_traits::SMEM_BYTES_FWD, stream);
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

template <typename weight_t, typename input_t, typename output_t, typename compute_t,
          typename index_t, int HIDDEN_SIZE, int WARPS_M, int WARPS_N, int BYTES_PER_LDG>
void launch_general_(LaunchParams<FwdParams> &launch_params, const bool configure_params) {  // NOLINT(*)
    using Kernel_traits = Kernel_traits<weight_t, input_t, output_t, compute_t, index_t,
                                        HIDDEN_SIZE, 1, WARPS_M, WARPS_N, BYTES_PER_LDG>;
    auto kernel = &rmsnorm_fwd_general_kernel<Kernel_traits>;
    auto ceil_div = [](int x, int y) -> int { return (x + y - 1) / y; };

    // Configure kernel params
    const int rows = launch_params.params.rows;
    const int cols = launch_params.params.cols;
    int ctas_per_col = launch_params.params.ctas_per_col;
    int ctas_per_row = launch_params.params.ctas_per_row;
    if (configure_params) {
        int ctas_per_sm;
        /*
        DPCT1007:229: Migration of cudaOccupancyMaxActiveBlocksPerMultiprocessor
        is not supported.
        */
        dpct::err0 status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &ctas_per_sm, kernel, Kernel_traits::THREADS_PER_CTA, 0);
        const int max_ctas = launch_params.multiprocessorCount * ctas_per_sm;
        ctas_per_row = ceil_div(cols, HIDDEN_SIZE);
        ctas_per_col = std::min(ceil_div(rows, WARPS_M), max_ctas / ctas_per_row);
        launch_params.params.ctas_per_row = ctas_per_row;
        launch_params.params.ctas_per_col = ctas_per_col;

        launch_params.barrier_size = 0;
        launch_params.workspace_bytes = 0;
        if (launch_params.params.ctas_per_row > 1) {
            launch_params.barrier_size = 2 * ctas_per_col;
            launch_params.workspace_bytes =
                (ctas_per_col * WARPS_M * ctas_per_row * sizeof(compute_t) * 2);
        }
        return;
    }

    // Launch kernel
    auto stream = launch_params.stream;
    sycl::range<3> grid(1, 1, ctas_per_row * ctas_per_col);
    sycl::range<3> block(1, 1, Kernel_traits::THREADS_PER_CTA);
    if (ctas_per_row == 1) {
        /*
        DPCT1049:12: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->parallel_for(sycl::nd_range<3>(grid * block, block),
                             [=](sycl::nd_item<3> item_ct1) {
                                 (launch_params.params);
                             });
    } else {
        void *params_ = reinterpret_cast<void *>(&launch_params.params);
        /*
        DPCT1007:13: Migration of cudaLaunchCooperativeKernel is not supported.
        */
        cudaLaunchCooperativeKernel(reinterpret_cast<void *>(kernel), grid,
                                    block, reinterpret_cast<void **>(&params_),
                                    0, stream);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#define REGISTER_FWD_TUNED_LAUNCHER(HIDDEN_SIZE, WTYPE, ITYPE, OTYPE, CTYPE,                       \
                              CTAS_PER_ROW, WARPS_M, WARPS_N, BYTES_PER_LDG)                       \
    void rmsnorm_fwd_tuned_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##OTYPE##_##CTYPE(                  \
            LaunchParams<FwdParams> &launch_params,                                                \
            const bool configure_params) {                                                         \
        launch_tuned_<WTYPE, ITYPE, OTYPE, CTYPE, uint32_t, HIDDEN_SIZE, CTAS_PER_ROW,             \
        WARPS_M, WARPS_N, BYTES_PER_LDG>(                                                          \
            launch_params, configure_params);                                                      \
    }                                                                                              \
    static FwdTunedRegistrar<WTYPE, ITYPE, OTYPE, CTYPE, HIDDEN_SIZE>                              \
           reg_tuned_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##OTYPE##_##CTYPE(                        \
        rmsnorm_fwd_tuned_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##OTYPE##_##CTYPE)

#define REGISTER_FWD_GENERAL_LAUNCHER(HIDDEN_SIZE, WTYPE, ITYPE, OTYPE, CTYPE,                     \
                              WARPS_M, WARPS_N, BYTES_PER_LDG)                                     \
    void rmsnorm_fwd_general_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##OTYPE##_##CTYPE(                \
            LaunchParams<FwdParams> &launch_params,                                                \
            const bool configure_params) {                                                         \
        launch_general_<WTYPE, ITYPE, OTYPE, CTYPE, uint32_t, HIDDEN_SIZE,                         \
        WARPS_M, WARPS_N, BYTES_PER_LDG>(                                                          \
            launch_params, configure_params);                                                      \
    }                                                                                              \
    static FwdGeneralRegistrar<WTYPE, ITYPE, OTYPE, CTYPE, HIDDEN_SIZE>                            \
           reg_general_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##OTYPE##_##CTYPE(                      \
        rmsnorm_fwd_general_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##OTYPE##_##CTYPE)

////////////////////////////////////////////////////////////////////////////////////////////////////

// Create rmsnorm tuned launch function and register. Macro signature:
//  HIDDEN_SIZE, WTYPE, ITYPE, OTYPE, CTYPE, CTAS_PER_ROW, WARPS_M, WARPS_N, BYTES_PER_LDG

REGISTER_FWD_TUNED_LAUNCHER(512, bf16, bf16, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(512, fp16, fp16, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(512, fp32, fp32, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(512, fp32, fp32, fp32, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(512, fp16, fp16, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(512, bf16, bf16, bf16, fp32, 1, 4, 1, 16);

REGISTER_FWD_TUNED_LAUNCHER(768, bf16, bf16, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(768, fp16, fp16, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(768, fp32, fp32, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(768, fp32, fp32, fp32, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(768, fp16, fp16, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(768, bf16, bf16, bf16, fp32, 1, 4, 1, 16);

REGISTER_FWD_TUNED_LAUNCHER(1024, bf16, bf16, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(1024, fp16, fp16, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(1024, fp32, fp32, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(1024, fp32, fp32, fp32, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(1024, fp16, fp16, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(1024, bf16, bf16, bf16, fp32, 1, 4, 1, 16);

REGISTER_FWD_TUNED_LAUNCHER(2048, bf16, bf16, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(2048, fp16, fp16, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(2048, fp32, fp32, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(2048, fp32, fp32, fp32, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(2048, fp16, fp16, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(2048, bf16, bf16, bf16, fp32, 1, 4, 1, 16);

REGISTER_FWD_TUNED_LAUNCHER(4096, bf16, bf16, fp8e4m3, fp32, 1, 1, 4, 16);
REGISTER_FWD_TUNED_LAUNCHER(4096, fp16, fp16, fp8e4m3, fp32, 1, 1, 4, 16);
REGISTER_FWD_TUNED_LAUNCHER(4096, fp32, fp32, fp8e4m3, fp32, 1, 1, 4, 16);
REGISTER_FWD_TUNED_LAUNCHER(4096, fp32, fp32, fp32, fp32, 1, 1, 4, 16);
REGISTER_FWD_TUNED_LAUNCHER(4096, fp16, fp16, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_TUNED_LAUNCHER(4096, bf16, bf16, bf16, fp32, 1, 1, 4, 16);

// Create rmsnorm general launch function and register. Macro signature:
//  HIDDEN_SIZE, WTYPE, ITYPE, OTYPE, CTYPE, WARPS_M, WARPS_N, BYTES_PER_LDG

REGISTER_FWD_GENERAL_LAUNCHER(128, bf16, bf16, fp8e4m3, fp32, 4, 1, 8);
REGISTER_FWD_GENERAL_LAUNCHER(512, bf16, bf16, fp8e4m3, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(1024, bf16, bf16, fp8e4m3, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(2048, bf16, bf16, fp8e4m3, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(8192, bf16, bf16, fp8e4m3, fp32, 1, 4, 16);

REGISTER_FWD_GENERAL_LAUNCHER(128, fp16, fp16, fp8e4m3, fp32, 4, 1, 8);
REGISTER_FWD_GENERAL_LAUNCHER(512, fp16, fp16, fp8e4m3, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(1024, fp16, fp16, fp8e4m3, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(2048, fp16, fp16, fp8e4m3, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(8192, fp16, fp16, fp8e4m3, fp32, 1, 4, 16);

REGISTER_FWD_GENERAL_LAUNCHER(128, fp32, fp32, fp8e4m3, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(512, fp32, fp32, fp8e4m3, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(1024, fp32, fp32, fp8e4m3, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(2048, fp32, fp32, fp8e4m3, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(8192, fp32, fp32, fp8e4m3, fp32, 1, 4, 16);

REGISTER_FWD_GENERAL_LAUNCHER(128, fp32, fp32, fp32, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(128, fp16, fp16, fp16, fp32, 4, 1, 8);
REGISTER_FWD_GENERAL_LAUNCHER(128, fp32, fp32, fp16, fp32, 4, 1, 8);
REGISTER_FWD_GENERAL_LAUNCHER(128, bf16, bf16, bf16, fp32, 4, 1, 8);
REGISTER_FWD_GENERAL_LAUNCHER(128, fp32, fp32, bf16, fp32, 4, 1, 8);

REGISTER_FWD_GENERAL_LAUNCHER(512, fp32, fp32, fp32, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(512, fp16, fp16, fp16, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(512, fp32, fp32, fp16, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(512, bf16, bf16, bf16, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(512, fp32, fp32, bf16, fp32, 4, 1, 16);

REGISTER_FWD_GENERAL_LAUNCHER(1024, fp32, fp32, fp32, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(1024, fp16, fp16, fp16, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(1024, fp32, fp32, fp16, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(1024, bf16, bf16, bf16, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(1024, fp32, fp32, bf16, fp32, 4, 1, 16);

REGISTER_FWD_GENERAL_LAUNCHER(2048, fp32, fp32, fp32, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(2048, fp16, fp16, fp16, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(2048, fp32, fp32, fp16, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(2048, bf16, bf16, bf16, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(2048, fp32, fp32, bf16, fp32, 4, 1, 16);

REGISTER_FWD_GENERAL_LAUNCHER(8192, fp32, fp32, fp32, fp32, 1, 4, 16);
REGISTER_FWD_GENERAL_LAUNCHER(8192, fp16, fp16, fp16, fp32, 1, 4, 16);
REGISTER_FWD_GENERAL_LAUNCHER(8192, fp32, fp32, fp16, fp32, 1, 4, 16);
REGISTER_FWD_GENERAL_LAUNCHER(8192, bf16, bf16, bf16, fp32, 1, 4, 16);
REGISTER_FWD_GENERAL_LAUNCHER(8192, fp32, fp32, bf16, fp32, 1, 4, 16);
