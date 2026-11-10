/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_GEMM_CUTLASS_MXFP4_MXFP8_GEMM_H_
#define TRANSFORMER_ENGINE_COMMON_GEMM_CUTLASS_MXFP4_MXFP8_GEMM_H_

#include <cuda_runtime.h>

#include "../common.h"

namespace transformer_engine {

/*! \brief Whether a GEMM mixes an MXFP4 operand with an MXFP8 operand. */
bool is_mxfp4_mxfp8_gemm(const Tensor &A, const Tensor &B);

/*! \brief MXFP4 x MXFP8 GEMM with CUTLASS (cuBLAS does not support mixed block-scaled inputs).
 *
 *  Follows the column-major conventions of cublas_gemm. A must be the MXFP4 operand and B the
 *  MXFP8 operand. A may be transposed (row-wise data, TN) or not (column-wise data, NN); B must
 *  not be transposed. Both operands need GEMM-swizzled scales.
 */
void cutlass_mxfp4_mxfp8_gemm(const Tensor *inputA, const Tensor *inputB, Tensor *outputD,
                              const Tensor *inputBias, bool transa, bool transb, float alpha,
                              float beta, void *workspace, size_t workspace_size,
                              cudaStream_t stream);

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_GEMM_CUTLASS_MXFP4_MXFP8_GEMM_H_
