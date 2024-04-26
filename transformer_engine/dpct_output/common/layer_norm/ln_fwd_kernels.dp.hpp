/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_LAYER_NORM_LN_FWD_KERNELS_CUH_
#define TRANSFORMER_ENGINE_COMMON_LAYER_NORM_LN_FWD_KERNELS_CUH_

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cfloat>
#include <cstdio>
#include "ln.h"
#include "../utils.dp.hpp"

namespace transformer_engine {
namespace layer_norm {
using namespace transformer_engine;

template <typename Ktraits>
/*
DPCT1110:47: The total declared local variable size in device function
ln_fwd_tuned_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/

void ln_fwd_tuned_kernel(FwdParams params, const sycl::nd_item<3> &item_ct1,
                         uint8_t *dpct_local, float *staging) {
    enum { ROWS_PER_CTA = Ktraits::ROWS_PER_CTA };
    enum { WARPS_N = Ktraits::WARPS_N };
    enum { WARPS_M = Ktraits::WARPS_M };
    enum { THREADS_PER_ROW = Ktraits::THREADS_PER_ROW };
    enum { VEC_COLS_PER_LDG = Ktraits::VEC_COLS_PER_LDG };
    enum { BYTES_PER_ROW = Ktraits::BYTES_PER_ROW };
    enum { LDGS = Ktraits::LDGS };
    enum { NUM_ELTS = Ktraits::NUM_ELTS };
    enum { CTAS_PER_ROW = Ktraits::CTAS_PER_ROW };

    using output_t = typename Ktraits::output_t;
    using index_t = typename Ktraits::index_t;
    using compute_t = typename Ktraits::compute_t;
    using Ivec = typename Ktraits::Ivec;
    using Ovec = typename Ktraits::Ovec;
    using Wvec = typename Ktraits::Wvec;

    using Stats = typename Ktraits::Stats;
    using stats_t = typename Stats::stats_t;

    auto smem_ = (char *)dpct_local;

    const index_t tidx = item_ct1.get_local_id(2);
    const index_t bidn = item_ct1.get_group(2) % CTAS_PER_ROW;
    const index_t bidm = item_ct1.get_group(2) / CTAS_PER_ROW;
    const index_t lane = tidx % THREADS_PER_WARP;
    const index_t warp = tidx / THREADS_PER_WARP;
    const index_t warp_m = warp / WARPS_N;
    const index_t warp_n = warp % WARPS_N;

    const index_t r = bidm * ROWS_PER_CTA + warp_m;
    const index_t c = bidn * THREADS_PER_ROW + warp_n * THREADS_PER_WARP + lane;

    Stats stats(params, bidm, bidn, warp_m, warp_n, lane, smem_);

    compute_t *mu_ptr = static_cast<compute_t *>(params.mu);
    compute_t *rs_ptr = static_cast<compute_t *>(params.rs);

    Wvec gamma[LDGS];
    Wvec beta[LDGS];
    index_t idx = c;
    #pragma unroll
    for ( int it = 0; it < LDGS; ++it ) {
        gamma[it].load_from(params.gamma, idx);
        beta[it].load_from(params.beta, idx);
        idx += VEC_COLS_PER_LDG;
    }

    constexpr compute_t rn = 1.f / compute_t(Ktraits::COLS);

    compute_t scale = 1.f;
    if (params.fp8_out) {
        scale = *reinterpret_cast<compute_t*>(params.scale);
    }
    compute_t amax = 0;

    for ( int row = r; row < params.rows; row += params.ctas_per_col * ROWS_PER_CTA ) {
        Ivec x[LDGS];
        index_t idx = row * Ktraits::VEC_COLS + c;
        compute_t xf[LDGS * NUM_ELTS];
        #pragma unroll
        for ( int it = 0; it < LDGS; it++ ) {
            x[it].load_from(params.x, idx);
            #pragma unroll
            for ( int jt = 0; jt < NUM_ELTS; jt++ ) {
                compute_t x_ij = compute_t(x[it].data.elt[jt]);
                xf[it * NUM_ELTS + jt] =  x_ij;
            }
            idx += VEC_COLS_PER_LDG;
        }

        /*
        DPCT1084:48: The function call "Stats::compute" has multiple migration
        results in different template instantiations that could not be unified.
        You may need to adjust the code.
        */
        stats_t s = stats.compute(xf, rn, item_ct1);

        compute_t mu = layer_norm::Get<0>::of<stats_t, compute_t>(s);
        compute_t m2 = layer_norm::Get<1>::of<stats_t, compute_t>(s);

        if ( bidn == 0 && warp_n == 0 && lane == 0 ) {
            mu_ptr[row] = mu;
        }

        compute_t rs = sycl::rsqrt(rn * m2 + params.epsilon);

        if ( bidn == 0 && warp_n == 0 && lane == 0 ) {
            rs_ptr[row] = rs;
        }

        Ovec z[LDGS];
        idx = row * Ktraits::VEC_COLS + c;
        #pragma unroll
        for ( int it = 0; it < LDGS; it++ ) {
            #pragma unroll
            for ( int jt = 0; jt < NUM_ELTS; jt++ ) {
                compute_t y_ij = rs * (xf[it * NUM_ELTS + jt] - mu);
                compute_t g_ij = gamma[it].data.elt[jt];
                if (params.zero_centered_gamma) {
                  g_ij += 1;
                }
                compute_t b_ij = beta[it].data.elt[jt];
                compute_t temp_output = g_ij * y_ij + b_ij;

                if (params.fp8_out) {
                    __builtin_assume(amax >= 0);
                    amax = sycl::fmax(amax, sycl::fabs(temp_output));
                    temp_output = temp_output * scale;
                }

                z[it].data.elt[jt] = output_t(temp_output);
            }
            z[it].store_to(params.z, idx);
            idx += VEC_COLS_PER_LDG;
        }
    }
    if (params.fp8_out) {
        amax = reduce_max<WARPS_M * WARPS_N>(amax, warp, item_ct1, staging);
        if (item_ct1.get_local_id(2) == 0 && item_ct1.get_local_id(1) == 0) {
            static_assert(std::is_same<compute_t, float>::value);
            atomicMaxFloat(reinterpret_cast<compute_t*>(params.amax), amax);
        }
    }
}

template <typename Ktraits>
/*
DPCT1110:49: The total declared local variable size in device function
ln_fwd_general_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/

void ln_fwd_general_kernel(FwdParams params, const sycl::nd_item<3> &item_ct1,
                           char *smem_, float *staging) {
    enum { LDGS = Ktraits::LDGS };
    enum { NUM_ELTS = Ktraits::NUM_ELTS };
    enum { WARPS_M = Ktraits::WARPS_M };
    enum { WARPS_N = Ktraits::WARPS_N };

    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;
    using output_t = typename Ktraits::output_t;
    using index_t = typename Ktraits::index_t;
    using compute_t = typename Ktraits::compute_t;
    using Ivec = typename Ktraits::Ivec;
    using Ovec = typename Ktraits::Ovec;
    using Wvec = typename Ktraits::Wvec;
    using Cvec = typename Ktraits::Cvec;

    const index_t tidx = item_ct1.get_local_id(2);
    const index_t lane = tidx % THREADS_PER_WARP;
    const index_t warp = tidx / THREADS_PER_WARP;
    const index_t warp_m = warp / WARPS_N;
    const index_t warp_n = warp % WARPS_N;

    const index_t bdimm = WARPS_M;
    const index_t bdimn = WARPS_N * THREADS_PER_WARP;
    const index_t bidm = item_ct1.get_group(2) / params.ctas_per_row;
    const index_t bidn = item_ct1.get_group(2) % params.ctas_per_row;

    const index_t gdimm = bdimm * params.ctas_per_col;
    const index_t gdimn = bdimn * params.ctas_per_row;
    const index_t gidm = bidm * bdimm + warp_m;
    const index_t gidn = (bidn * THREADS_PER_WARP
                          + warp_n * params.ctas_per_row * THREADS_PER_WARP
                          + lane);  // Order threads by warp x cta x lane

    // Objects for stats reductions
    using Reducer = DynamicReducer<compute_t, WARPS_M, WARPS_N>;
    constexpr int SMEM_BYTES = Reducer::SMEM_BYTES > 0 ? Reducer::SMEM_BYTES : 1;

    Reducer reducer(params, bidm, bidn, warp_m, warp_n, lane, smem_);
    Sum<compute_t> sum;
    const compute_t rn = 1.f / static_cast<compute_t>(params.cols);

    // Load weights
    Cvec gamma[LDGS];
    Cvec beta[LDGS];
    #pragma unroll
    for ( int it = 0, col = gidn * NUM_ELTS;
          it < LDGS && col < params.cols;
          ++it, col += gdimn * NUM_ELTS ) {
        Wvec gamma_in, beta_in;
        gamma_in.load_from_elts(params.gamma, col, params.cols - col);
        beta_in.load_from_elts(params.beta, col, params.cols - col);
        gamma_in.to(gamma[it]);
        beta_in.to(beta[it]);
    }

    // fp8 factors
    compute_t scale;
    if ( params.fp8_out ) {
        scale = *reinterpret_cast<compute_t*>(params.scale);
    }
    compute_t amax = 0;

    for ( int cta_row = bidm * bdimm;
          cta_row < params.rows;
          cta_row += gdimm ) {
        const int row = cta_row + warp_m;

        // Load input
        Cvec x[LDGS];
        #pragma unroll
        for ( int it = 0, col = gidn * NUM_ELTS;
              it < LDGS && row < params.rows && col < params.cols;
              it++, col += gdimn * NUM_ELTS ) {
            Ivec x_in;
            x_in.load_from_elts(params.x,
                                row * params.cols + col,
                                params.cols - col);
            x_in.to(x[it]);
        }

        // Compute mean
        compute_t mu = 0.f;
        #pragma unroll
        for ( int it = 0, col = gidn * NUM_ELTS;
              it < LDGS && row < params.rows && col < params.cols;
              it++, col += gdimn * NUM_ELTS ) {
            #pragma unroll
            for ( int jt = 0; jt < NUM_ELTS; jt++ ) {
                mu += x[it].data.elt[jt];
            }
        }
        mu = reducer.allreduce(mu, sum, item_ct1) * rn;

        // Compute variance
        compute_t sqsigma = 0.f;
        #pragma unroll
        for ( int it = 0, col = gidn * NUM_ELTS;
              it < LDGS && row < params.rows && col < params.cols;
              it++, col += gdimn * NUM_ELTS ) {
            #pragma unroll
            for ( int jt = 0; jt < NUM_ELTS; jt++ ) {
                if ( col + jt < params.cols ) {
                    compute_t diff = x[it].data.elt[jt] - mu;
                    sqsigma += diff * diff;
                }
            }
        }
        sqsigma = reducer.allreduce(sqsigma, sum, item_ct1) * rn;
        compute_t rs = sycl::rsqrt(sqsigma + params.epsilon);

        // Write statistics
        if ( gidn == 0 && row < params.rows ) {
            compute_t *mu_ptr = static_cast<compute_t *>(params.mu);
            compute_t *rs_ptr = static_cast<compute_t *>(params.rs);
            mu_ptr[row] = mu;
            rs_ptr[row] = rs;
        }

        // Compute output
        #pragma unroll
        for ( int it = 0, col = gidn * NUM_ELTS;
              it < LDGS && row < params.rows && col < params.cols;
              it++, col += gdimn * NUM_ELTS ) {
            // Compute output values
            Cvec z;
            #pragma unroll
            for ( int jt = 0; jt < NUM_ELTS; jt++ ) {
                compute_t y_ij = rs * (x[it].data.elt[jt] - mu);
                compute_t g_ij = gamma[it].data.elt[jt];
                if (params.zero_centered_gamma) {
                  g_ij += 1;
                }
                compute_t b_ij = beta[it].data.elt[jt];
                z.data.elt[jt] = g_ij * y_ij + b_ij;
            }

            // Apply fp8 factors
            if ( params.fp8_out ) {
                #pragma unroll
                for ( int jt = 0; jt < NUM_ELTS; jt++ ) {
                    if ( col + jt < params.cols ) {
                        compute_t z_ij = z.data.elt[jt];
                        __builtin_assume(amax >= 0);
                        amax = sycl::fmax(amax, sycl::fabs(z_ij));
                        z.data.elt[jt] = z_ij * scale;
                    }
                }
            }

            // Store output
            Ovec z_out;
            z.to(z_out);
            z_out.store_to_elts(params.z,
                                row * params.cols + col,
                                params.cols - col);
        }
    }

    // Finalize fp8 factors
    if ( params.fp8_out ) {
        amax = reduce_max<WARPS_M * WARPS_N>(amax, warp, item_ct1, staging);
        if (item_ct1.get_local_id(2) == 0) {
            static_assert(std::is_same<compute_t, float>::value);
            atomicMaxFloat(reinterpret_cast<compute_t*>(params.amax), amax);
        }
    }
}

}  // namespace layer_norm
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_LAYER_NORM_LN_FWD_KERNELS_CUH_
