/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file newton_schulz.h
 *  \brief Functions for distributed Newton-Schulz inverse square root.
 *
 *  This API is a TE-native binding to the cuSolverMp library.
 *  It computes an iterative Newton-Schulz inverse square root
 *  approximation on a distributed matrix.
 */

#ifndef TRANSFORMER_ENGINE_COMMON_NEWTON_SCHULZ_H_
#define TRANSFORMER_ENGINE_COMMON_NEWTON_SCHULZ_H_

#include <nccl.h>
#include <stdint.h>

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

typedef struct NVTECusolverMpCtx NVTECusolverMpCtx;

/*! \brief Create a cuSolverMp context for Newton-Schulz operations.
 *
 *  Creates a dedicated CUDA stream internally (cuSolverMp requires a
 *  non-default stream).
 *
 *  \param[in]  comm    NCCL communicator.
 *  \param[in]  nranks  Number of ranks.
 *  \param[in]  rank    Local rank.
 */
NVTECusolverMpCtx* nvte_cusolvermp_ctx_create(ncclComm_t comm, int nranks, int rank);

/*! \brief Destroy a cuSolverMp context.
 *
 *  \param[in]  ctx  Context to destroy.
 */
void nvte_cusolvermp_ctx_destroy(NVTECusolverMpCtx* ctx);

/*! \brief Compute Newton-Schulz inverse square root in-place.
 *
 *  Performs iterative Newton-Schulz approximation of the inverse square root
 *  on a distributed matrix using cuSolverMp.
 *
 *  \param[in]     ctx              cuSolverMp context.
 *  \param[in]     m                Global number of rows.
 *  \param[in]     n                Global number of columns.
 *  \param[in,out] x                Local part of the matrix (modified in-place).
 *  \param[in]     num_iterations   Number of Newton-Schulz iterations.
 *  \param[in]     coefficients     Array of polynomial coefficients (length depends on polynomial
 *                                  degree used internally by cuSolverMp).
 *  \param[in]     num_coefficients Number of elements in the coefficients array.
 */
void nvte_newton_schulz(NVTECusolverMpCtx* ctx, int64_t m, int64_t n, NVTETensor x,
                        int64_t num_iterations, const float* coefficients,
                        int64_t num_coefficients);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_COMMON_NEWTON_SCHULZ_H_
