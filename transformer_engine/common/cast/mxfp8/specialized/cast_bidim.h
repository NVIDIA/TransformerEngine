/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file cast_bidim.h
 *  \brief Entry point for the register-resident bidimensional MXFP8 cast kernel.
 */

#ifndef TRANSFORMER_ENGINE_MXFP8_SPECIALIZED_CAST_BIDIM_H_
#define TRANSFORMER_ENGINE_MXFP8_SPECIALIZED_CAST_BIDIM_H_

#include <cuda_runtime.h>

#include "../../../common.h"

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {
namespace quantize_kernel {
namespace specialized {

/*! \brief Cast BF16 to MXFP8 with both rowwise and colwise scales.
 *
 * Produces two quantizations of the same tensor: a rowwise one, where 32
 * consecutive elements of a row share a scale, and a colwise one, where 32
 * consecutive elements of a column share a scale.  Both outputs keep the
 * input's row-major [rows, cols] layout; the colwise result is not transposed.
 *
 * Unlike the TMA kernel in mxfp8/specialized this one is register-resident: a
 * CTA reads its 32-row tile once and drives both passes from registers.  See
 * cast_bidim.cu for the layout and tuning rationale.
 *
 * Requires SM 10.0+ (Blackwell), matching MXFP8 support in the rest of TE.
 *
 *  \tparam     OType                 FP8 output type.
 *  \param[in]  input                 BF16 input, [rows, cols], row-major.
 *  \param[out] output_rowwise        FP8E4M3 rowwise-scaled output, [rows, cols].
 *  \param[out] scales_rowwise        E8M0 rowwise scales, one per 32 columns.
 *  \param[out] output_colwise        FP8E4M3 colwise-scaled output, [rows, cols].
 *  \param[out] scales_colwise        E8M0 colwise scales, one per 32 rows.
 *  \param[in]  rows                  Row count; must be a multiple of 32.
 *  \param[in]  cols                  Column count; must be a multiple of 256.
 *  \param[in]  scale_stride_rowwise  Rowwise scale elements per row.
 *  \param[in]  scale_stride_colwise  Colwise scale elements per 32-row band.
 *  \param[in]  stream                CUDA stream.
 */
template <typename OType>
void launch_cast_bidim(const void *input, void *output_rowwise, void *scales_rowwise,
                       void *output_colwise, void *scales_colwise, int rows, int cols,
                       int scale_stride_rowwise, int scale_stride_colwise, cudaStream_t stream);

}  // namespace specialized
}  // namespace quantize_kernel
}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_MXFP8_SPECIALIZED_CAST_BIDIM_H_
