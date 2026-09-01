/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file cast_rowwise.h
 *  \brief Entry point for the register-resident rowwise MXFP8 cast kernel.
 */

#ifndef TRANSFORMER_ENGINE_MXFP8_SPECIALIZED_CAST_ROWWISE_H_
#define TRANSFORMER_ENGINE_MXFP8_SPECIALIZED_CAST_ROWWISE_H_

#include <cuda_runtime.h>

#include "../../../common.h"

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {
namespace quantize_kernel {
namespace specialized {

/*! \brief Cast BF16 to rowwise-scaled MXFP8.
 *
 * Every run of 32 consecutive elements in a row forms one MX block sharing a
 * single E8M0 scale.  This is the register-resident member of the specialized
 * cast-only family: it keeps its tile in registers rather than staging it, and
 * is 10-20% faster than quantize_mxfp8_kernel_cast_only on that path.  See
 * cast_rowwise.cu for the layout and tuning rationale.
 *
 * Instantiated for the MXFP8 output types the specialized dispatch can reach,
 * fp8e4m3 and fp8e5m2.  Requires SM 10.0+ (Blackwell), matching MXFP8 support
 * in the rest of TE.
 *
 *  \tparam     OType         FP8 output type.
 *  \param[in]  input         BF16 input, [rows, cols], row-major.
 *  \param[out] output        FP8 output, [rows, cols], row-major.
 *  \param[out] scales        E8M0 scales, one byte per MX block.
 *  \param[in]  rows          Number of rows.
 *  \param[in]  cols          Number of columns; must be a multiple of 32.
 *  \param[in]  scale_stride  Scale elements per row.  Equals cols/32 for a
 *                            packed scale array, or more when it is padded.
 *  \param[in]  stream        CUDA stream.
 */
template <typename OType>
void launch_cast_rowwise(const void *input, void *output, void *scales, int rows, int cols,
                         int scale_stride, cudaStream_t stream);

}  // namespace specialized
}  // namespace quantize_kernel
}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_MXFP8_SPECIALIZED_CAST_ROWWISE_H_
