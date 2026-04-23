/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file grouped_fp8_current_scaling.h
 *  \brief Functions for grouped FP8 current scaling quantization.
 *
 * This header provides functions for efficiently quantizing multiple tensors
 * simultaneously using FP8 current scaling. This is particularly useful for
 * Mixture of Experts (MoE) models where each expert's activations need to be
 * quantized independently.
 *
 * Workflow for FP8 Current Scaling:
 * 1. Compute amax for all tensors (nvte_group_amax_graph_safe)
 * 2. Compute scales from amaxes (nvte_multi_tensor_compute_scale_and_scale_inv)
 * 3. Perform FP8 quantization with scales (functions in this file)
 *
 * The three steps cannot be fused because step 2 depends on step 1's output.
 * However, processing multiple tensors in parallel within each step provides
 * significant performance benefits:
 * - Fewer kernel launches (3 instead of 3*N)
 * - Lower CPU overhead
 * - CUDA Graph compatible
 * - Better GPU utilization
 */

#ifndef TRANSFORMER_ENGINE_GROUPED_FP8_CURRENT_SCALING_H_
#define TRANSFORMER_ENGINE_GROUPED_FP8_CURRENT_SCALING_H_

#include <cuda_runtime.h>

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Perform grouped FP8 quantization with pre-computed scales (rowwise layout).
 *
 * This function quantizes multiple tensors from high precision to FP8 using
 * pre-computed scaling factors. The input and output tensors are stored in
 * grouped tensor format with rowwise (non-transposed) layout.
 *
 * Requirements:
 * - Input: NVTEGroupedTensor with high-precision data (FP32/BF16/FP16)
 * - Output: NVTEGroupedTensor with:
 *     * Allocated FP8 data buffer
 *     * Pre-computed scale values (one per tensor)
 *     * Same number of tensors as input
 *
 * Algorithm:
 * For each tensor i:
 *   For each element j:
 *     output[i][j] = cast_to_fp8(input[i][j] * scale[i])
 *
 * Performance characteristics:
 * - Single kernel launch for all tensors
 * - Coalesced memory access
 * - Vectorized loads when aligned
 * - CUDA Graph compatible
 *
 * \param[in]     input   Input grouped tensor (high precision)
 * \param[in,out] output  Output grouped tensor (FP8, scales must be set)
 * \param[in]     stream  CUDA stream for asynchronous execution
 *
 * Example:
 * \code
 *   // Step 1: Compute amaxes
 *   nvte_group_amax_graph_safe(input_grouped, output_grouped, stream);
 *
 *   // Step 2: Compute scales from amaxes
 *   nvte_multi_tensor_compute_scale_and_scale_inv(
 *       amax_list, scale_list, scale_inv_list, ...);
 *
 *   // Step 3: Quantize with computed scales
 *   nvte_grouped_fp8_quantize_rowwise(input_grouped, output_grouped, stream);
 * \endcode
 */
void nvte_grouped_fp8_quantize_rowwise(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                                       cudaStream_t stream);

/*! \brief Perform grouped FP8 quantization with transpose (columnwise layout).
 *
 * This function quantizes and transposes multiple tensors simultaneously.
 * The output is in columnwise (transposed) format, suitable for certain
 * GEMM layouts (TN, NT).
 *
 * For each 2D tensor with shape [M, N]:
 * - Input: [M, N] rowwise layout
 * - Output: [N, M] columnwise layout (transposed)
 *
 * Requirements:
 * - All tensors must be 2D
 * - Input: NVTEGroupedTensor with rowwise data
 * - Output: NVTEGroupedTensor with columnwise_data buffer allocated
 *
 * Algorithm:
 * For each tensor i with shape [M, N]:
 *   For each position (m, n):
 *     output_transposed[i][n][m] = cast_to_fp8(input[i][m][n] * scale[i])
 *
 * This is equivalent to:
 *   quantize(input[i]) followed by transpose
 * But performs both operations in a single kernel pass.
 *
 * \param[in]     input   Input grouped tensor (high precision, rowwise)
 * \param[in,out] output  Output grouped tensor (FP8, columnwise/transposed)
 * \param[in]     stream  CUDA stream for asynchronous execution
 *
 * Example:
 * \code
 *   // After computing scales...
 *
 *   // Quantize with transpose
 *   nvte_grouped_fp8_quantize_columnwise(input_grouped, output_grouped, stream);
 *
 *   // Output is now in transposed format suitable for TN/NT GEMM
 * \endcode
 */
void nvte_grouped_fp8_quantize_columnwise(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                                          cudaStream_t stream);

/*! \brief Perform both rowwise and columnwise grouped FP8 quantization.
 *
 * This function quantizes multiple tensors and produces both rowwise and
 * columnwise outputs simultaneously. This is useful when you need both
 * layouts (e.g., for forward and backward passes).
 *
 * Requirements:
 * - Output must have both data and columnwise_data buffers allocated
 *
 * This is equivalent to calling:
 *   nvte_grouped_fp8_quantize_rowwise() followed by
 *   nvte_grouped_fp8_quantize_columnwise()
 * But may be optimized to share computation.
 *
 * \param[in]     input   Input grouped tensor (high precision)
 * \param[in,out] output  Output grouped tensor (FP8, both layouts)
 * \param[in]     stream  CUDA stream for asynchronous execution
 *
 * Example:
 * \code
 *   // Allocate output with both rowwise and columnwise buffers
 *   output_grouped = GroupedTensor::make_grouped_tensor(
 *       num_tensors, shapes, quantizers, device);
 *
 *   // After computing scales...
 *
 *   // Quantize to both layouts
 *   nvte_grouped_fp8_quantize_both(input_grouped, output_grouped, stream);
 * \endcode
 */
void nvte_grouped_fp8_quantize_both(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                                    cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"

namespace transformer_engine {

// C++ wrapper functions for convenience

/*! \brief C++ wrapper for grouped FP8 rowwise quantization.
 *
 * \param input Input grouped tensor
 * \param output Output grouped tensor
 * \param stream CUDA stream
 */
void launch_grouped_fp8_quantize_rowwise(const GroupedTensor& input, GroupedTensor& output,
                                         cudaStream_t stream);

/*! \brief C++ wrapper for grouped FP8 columnwise quantization.
 *
 * \param input Input grouped tensor
 * \param output Output grouped tensor
 * \param stream CUDA stream
 */
void launch_grouped_fp8_quantize_columnwise(const GroupedTensor& input, GroupedTensor& output,
                                            cudaStream_t stream);

}  // namespace transformer_engine

#endif  // __cplusplus

#endif  // TRANSFORMER_ENGINE_GROUPED_FP8_CURRENT_SCALING_H_
