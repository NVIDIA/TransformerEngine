/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_RMSNORM_RMSNORM_KERNEL_TRAITS_H_
#define TRANSFORMER_ENGINE_COMMON_RMSNORM_RMSNORM_KERNEL_TRAITS_H_

#include "../common.h"
#include "../layer_norm/ln_kernel_traits.h"
#include "../utils.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace transformer_engine {
namespace rmsnorm {

template <
    uint32_t HIDDEN_SIZE_, typename weight_t_, typename input_t_, typename output_t_,
    typename compute_t_, typename index_t_, uint32_t THREADS_PER_CTA_, uint32_t BYTES_PER_LDG_,
    typename Base =
        layer_norm::Kernel_traits_finalize<HIDDEN_SIZE_, weight_t_, input_t_, output_t_, compute_t_,
                                           index_t_, THREADS_PER_CTA_, BYTES_PER_LDG_> >
struct Kernel_traits_finalize : public Base {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename weight_t_, typename input_t_, typename output_t_, typename compute_t_,
          typename index_t_, uint32_t HIDDEN_SIZE_, uint32_t CTAS_PER_ROW_, uint32_t WARPS_M_,
          uint32_t WARPS_N_, uint32_t BYTES_PER_LDG_ = 16,
          typename Base = layer_norm::Kernel_traits<weight_t_, input_t_, output_t_, compute_t_,
                                                    index_t_, HIDDEN_SIZE_, CTAS_PER_ROW_, WARPS_M_,
                                                    WARPS_N_, BYTES_PER_LDG_> >
struct Kernel_traits : public Base {};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace rmsnorm
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_RMSNORM_RMSNORM_KERNEL_TRAITS_H_
