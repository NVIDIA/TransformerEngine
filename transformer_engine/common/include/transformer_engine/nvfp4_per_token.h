/*************************************************************************
 * Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_NVFP4_PER_TOKEN_H_
#define TRANSFORMER_ENGINE_NVFP4_PER_TOKEN_H_

#include <cuda_runtime_api.h>

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Composite K1+K2: per-row + per-col amax (K1) then FP4 + 1x16
 *         e4m3 SF encode (K2), back-to-back on the same stream.
 *
 *  Production entry point for the per-token cast on bf16 + 128-aligned shapes.
 *
 *  \param[in] with_rht  non-zero -> apply 16-pt RHT on the col direction in
 *                       both K1 and K2. Rowwise stays raw; zero is byte-equal
 *                       to the pre-RHT path.
 *  \param[in] random_sign_mask_t  low 16 bits = sign-flip pattern shared by
 *                       K1 and K2. Ignored when with_rht == 0.
 *  \param[in] with_swizzle  non-zero -> K2 emits rowwise scale_inv directly
 *                       in the cuBLAS LT swizzled tile layout (rowwise only;
 *                       colwise stays compact M-major).
 *  \param[in] with_sr   non-zero -> the K2 FP4 cast uses stochastic rounding
 *                       (per-element Philox dither). K1 amax stays
 *                       deterministic. Zero is byte-equal to the RN path.
 *  \param[in] rng_state int64 device tensor of shape [2] = {seed, offset}
 *                       (host-unpacked Philox). May be NULL iff with_sr == 0.
 */
void nvte_nvfp4_per_token_quantize(const NVTETensor input, const NVTETensor noop, NVTETensor output,
                                   int with_rht, int random_sign_mask_t, int with_swizzle,
                                   int with_sr, const NVTETensor rng_state, cudaStream_t stream);

/*! \brief Kernel 1 in isolation: per-row + per-col amax via TMA + atomicMax.
 *         Pre-zeroes the amax buffers and merges per-CTA partials into
 *         ``output->amax`` (size [M]) / ``output->columnwise_amax``
 *         (size [K]). Does NOT touch FP4 data / scale_inv slots.
 *
 *  \param[in] with_rht  non-zero -> apply 16-pt RHT on the col direction
 *                       before columnwise_amax (rowwise stays raw); zero is
 *                       byte-equal to the pre-RHT K1.
 *  \param[in] random_sign_mask_t  low 16 bits = sign-flip pattern; ignored
 *                       when with_rht == 0. Type matches prod's
 *                       nvte_hadamard_transform_amax convention.
 */
void nvte_nvfp4_per_token_amax(const NVTETensor input, const NVTETensor noop, NVTETensor output,
                               int with_rht, int random_sign_mask_t, cudaStream_t stream);

/*! \brief Kernel 2 in isolation: FP4 + 1x16 e4m3 SF encode given a
 *         pre-filled ``output->amax`` / ``output->columnwise_amax``. Reads
 *         the outer amax buffer(s) and writes the FP4 data / scale_inv
 *         tensors only.
 *
 *  \param[in] with_rht  non-zero -> col-wise cast applies the same 16-pt RHT
 *                       that K1 amax must have used (caller's responsibility
 *                       to thread the same flag + mask through K1 and K2).
 *  \param[in] random_sign_mask_t  low 16 bits = sign-flip pattern; ignored
 *                       when with_rht == 0.
 *  \param[in] with_swizzle  non-zero -> write rowwise scale_inv directly in
 *                       the cuBLAS LT swizzled tile layout (rowwise only;
 *                       colwise stays compact M-major).
 *  \param[in] with_sr   non-zero -> stochastic-rounding FP4 cast (per-element
 *                       Philox dither). Zero is byte-equal to the RN path.
 *  \param[in] rng_state int64 device tensor of shape [2] = {seed, offset}.
 *                       May be NULL iff with_sr == 0.
 */
void nvte_nvfp4_per_token_encode(const NVTETensor input, const NVTETensor noop, NVTETensor output,
                                 int with_rht, int random_sign_mask_t, int with_swizzle,
                                 int with_sr, const NVTETensor rng_state, cudaStream_t stream);

/*! \brief Returns 1 iff the per-token kernels accept ``(M, K, dtype)``.
 *
 *  Currently returns 1 iff ``dtype`` is bf16 AND ``M % 128 == 0`` AND
 *  ``K % 128 == 0``. Cheap host-side query (no CUDA call).
 *
 *  \param[in] M                 first-dim (rows).
 *  \param[in] K                 last-dim (cols).
 *  \param[in] input_dtype_enum  NVTE_DType cast to int.
 */
int nvte_nvfp4_per_token_can_dispatch(size_t M, size_t K, int input_dtype_enum);

/*! \brief Apply per-row * per-col outer-scale to a (M, N) bf16 GEMM output.
 *
 *  Computes:
 *
 *      d[i, j] = d[i, j] * row_amax_a[i] * row_amax_b[j]
 */
void nvte_nvfp4_per_token_post_scale(NVTETensor d, const NVTETensor row_amax_a,
                                     const NVTETensor row_amax_b, cudaStream_t stream);

/* ============================================================================
 * Grouped (multi-tensor) per-token quantize.
 *
 *  \param[in]     input          (sum_M, K) bf16/fp32, row-major contiguous
 *  \param[in,out] outputs        array of `num_tensors` NVTETensors; on
 *                                return, amax/columnwise_amax slots are filled.
 *  \param[in]     split_sections array of `num_tensors` size_t values,
 *                                each a multiple of 64; sum must equal sum_M.
 *  \param[in]     num_tensors    <= 64
 *  \param[in]     rowwise        emit per-row amax in `outputs[i].amax`
 *  \param[in]     columnwise     emit per-col amax in `outputs[i].columnwise_amax`
 *  \param[in]     with_rht       non-zero -> 16-pt RHT on the col direction
 *                                (rowwise stays raw).
 *  \param[in]     random_sign_mask_t  low 16 bits = sign-flip pattern; must
 *                                match the value passed to the matching cast
 *                                if amax + cast are launched separately.
 *  \param[in]     stream         CUDA stream
 */
void nvte_group_nvfp4_per_token_amax(const NVTETensor input, NVTETensor* outputs,
                                     const size_t* split_sections, size_t num_tensors, bool rowwise,
                                     bool columnwise, int with_rht, int random_sign_mask_t,
                                     cudaStream_t stream);

/*! \brief Grouped per-token encode (FP4 + 1x16 e4m3 inner SF) using the
 *         row_amax / col_amax values already populated by
 *         `nvte_group_nvfp4_per_token_amax`.
 *
 *  \param[in]     input          same as `nvte_group_nvfp4_per_token_amax`
 *  \param[in,out] outputs        on entry: amax/columnwise_amax populated;
 *                                on return: data/scale_inv + columnwise_data/
 *                                columnwise_scale_inv populated.
 *  \param[in]     split_sections same as `nvte_group_nvfp4_per_token_amax`
 *  \param[in]     num_tensors    <= 64
 *  \param[in]     rowwise        emit per-row FP4 + inner SF
 *  \param[in]     columnwise     emit per-col FP4 + inner SF
 *  \param[in]     with_rht       must match the preceding amax call's
 *                                with_rht; applies the same 16-pt RHT on the
 *                                colwise cast.
 *  \param[in]     random_sign_mask_t  low 16 bits = sign-flip pattern; must
 *                                match K1.
 *  \param[in]     stream         CUDA stream
 */
void nvte_group_nvfp4_per_token_cast(const NVTETensor input, NVTETensor* outputs,
                                     const size_t* split_sections, size_t num_tensors, bool rowwise,
                                     bool columnwise, int with_rht, int random_sign_mask_t,
                                     cudaStream_t stream);

/*! \brief Composite K1+K2 grouped per-token quantize. Calls the amax + cast
 *         kernels on the same stream. This is the external API
 *         `tex.split_quantize(per_token=True)` should call.
 *
 *  \param[in]     input          (sum_M, K) bf16/fp32, row-major contiguous
 *  \param[in,out] outputs        on entry: amax / columnwise_amax / data /
 *                                scale_inv / columnwise_data /
 *                                columnwise_scale_inv slots allocated;
 *                                on return: all populated.
 *  \param[in]     split_sections array of `num_tensors` size_t values,
 *                                each a multiple of 64; sum must equal sum_M.
 *  \param[in]     num_tensors    <= 64
 *  \param[in]     rowwise        emit rowwise output
 *  \param[in]     columnwise     emit columnwise output
 *  \param[in]     with_rht       non-zero -> 16-pt RHT on the col direction
 *                                in BOTH K1 and K2; zero is byte-equal to the
 *                                pre-RHT path.
 *  \param[in]     random_sign_mask_t  low 16 bits = sign-flip pattern shared
 *                                between K1 and K2; ignored when with_rht==0.
 *  \param[in]     stream         CUDA stream
 */
void nvte_group_nvfp4_per_token_quantize(const NVTETensor input, NVTETensor* outputs,
                                         const size_t* split_sections, size_t num_tensors,
                                         bool rowwise, bool columnwise, int with_rht,
                                         int random_sign_mask_t, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif  // TRANSFORMER_ENGINE_NVFP4_PER_TOKEN_H_
