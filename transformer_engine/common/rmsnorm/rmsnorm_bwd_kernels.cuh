/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_RMSNORM_RMSNORM_BWD_KERNELS_CUH_
#define TRANSFORMER_ENGINE_COMMON_RMSNORM_RMSNORM_BWD_KERNELS_CUH_

#include "../utils.cuh"

namespace transformer_engine {
namespace rmsnorm {
using namespace transformer_engine;

template <typename Ktraits>
__global__ __launch_bounds__(Ktraits::THREADS_PER_CTA) void rmsnorm_bwd_tuned_kernel(
    BwdParams params) {
  enum { ROWS_PER_CTA = Ktraits::ROWS_PER_CTA };
  enum { WARPS_M = Ktraits::WARPS_M };
  enum { WARPS_N = Ktraits::WARPS_N };
  enum { THREADS_PER_ROW = Ktraits::THREADS_PER_ROW };
  enum { COLS = Ktraits::COLS };
  enum { BYTES_PER_ROW = Ktraits::BYTES_PER_ROW };
  enum { LDGS = Ktraits::LDGS };
  enum { NUM_ELTS = Ktraits::ELTS_PER_LDG };
  enum { THREADS_PER_WARP = Ktraits::THREADS_PER_WARP };
  enum { CTAS_PER_ROW = Ktraits::CTAS_PER_ROW };

  using compute_t = typename Ktraits::compute_t;
  using index_t = typename Ktraits::index_t;
  using Ivec = typename Ktraits::Ivec;
  using Ovec = typename Ktraits::Ovec;
  using Wvec = typename Ktraits::Wvec;
  using Cvec = typename Ktraits::Cvec;
  using Reducer = typename Ktraits::Reducer;
  using reduce_t = typename Reducer::Type;

  extern __shared__ char smem_[];

  const index_t tidx = threadIdx.x;
  const index_t bidn = blockIdx.x % CTAS_PER_ROW;
  const index_t bidm = blockIdx.x / CTAS_PER_ROW;
  const index_t lane = tidx % THREADS_PER_WARP;
  const index_t warp = tidx / THREADS_PER_WARP;
  const index_t warp_m = warp / Ktraits::WARPS_N;
  const index_t warp_n = warp % Ktraits::WARPS_N;
  const index_t tid_r = warp_n * THREADS_PER_WARP + lane;

  const index_t r = bidm * Ktraits::ROWS_PER_CTA + warp_m;
  const index_t c = bidn * THREADS_PER_ROW + warp_n * THREADS_PER_WARP + lane;

  static_assert(COLS == THREADS_PER_ROW * LDGS * NUM_ELTS * CTAS_PER_ROW);

  Cvec dzy_sum[LDGS];

  memset(dzy_sum, 0, sizeof(dzy_sum));

  compute_t *smem_wgrad = reinterpret_cast<compute_t *>(smem_);
  char *smem_dgrad = smem_ + Ktraits::SMEM_BYTES_WGRAD;

  Reducer reducer(params, bidm, bidn, warp_m, warp_n, lane, smem_dgrad);

  Sum<reduce_t> sum;

  constexpr float rn = 1.f / static_cast<float>(COLS);
  Wvec gamma[LDGS];
  index_t idx = c;
#pragma unroll
  for (int it = 0; it < LDGS; it++) {
    gamma[it].load_from(params.gamma, idx);
    idx += Ktraits::VEC_COLS_PER_LDG;
  }
// TODO if ROWS_PER_CTA does not divide rows, we might get divergence in the
// last blocks with syncthreads!
// grid stride over rows
#pragma unroll 1
  for (int row = r; row < params.rows; row += params.ctas_per_col * ROWS_PER_CTA) {
    const compute_t rs_r = static_cast<const compute_t *>(params.rs)[row];
    Ivec x[LDGS];
    Ovec dz[LDGS];
    index_t idx = row * Ktraits::VEC_COLS + c;
#pragma unroll
    for (int it = 0; it < LDGS; it++) {
      dz[it].load_from(params.dz, idx);
      x[it].load_from(params.x, idx);
      idx += Ktraits::VEC_COLS_PER_LDG;
    }

    compute_t dy[LDGS * NUM_ELTS];
    compute_t y[LDGS * NUM_ELTS];

    compute_t mdyy_local = 0.f;
#pragma unroll
    for (int it = 0; it < LDGS; it++) {
#pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) {
        compute_t x_tmp = x[it].data.elt[jt];
        compute_t y_tmp = rs_r * (x_tmp);
        const compute_t dy_tmp_shift = (params.zero_centered_gamma) ? 1.0f : 0.f;
        compute_t dy_tmp = compute_t(gamma[it].data.elt[jt]) + dy_tmp_shift;
        dy_tmp *= compute_t(dz[it].data.elt[jt]);
        compute_t dz_tmp = dz[it].data.elt[jt];

        mdyy_local += dy_tmp * y_tmp;

        dy[it * NUM_ELTS + jt] = dy_tmp;
        y[it * NUM_ELTS + jt] = y_tmp;

        dzy_sum[it].data.elt[jt] += dz_tmp * y_tmp;
      }
    }

    reduce_t result = reducer.allreduce({0, mdyy_local}, sum);
    mdyy_local = Get<1>::of<reduce_t, compute_t>(result) * rn;

    Ivec dx[LDGS];
    idx = row * Ktraits::VEC_COLS + c;
#pragma unroll
    for (int it = 0; it < LDGS; it++) {
#pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) {
        compute_t dy_tmp = dy[it * NUM_ELTS + jt];
        compute_t y_tmp = y[it * NUM_ELTS + jt];
        compute_t dx_tmp = rs_r * (dy_tmp - (mdyy_local * y_tmp));
        dx[it].data.elt[jt] = dx_tmp;
      }
      dx[it].store_to(params.dx, idx);
      idx += Ktraits::VEC_COLS_PER_LDG;
    }
  }  // end: grid stride loop

  if (WARPS_M == 1) {
    idx = r * Ktraits::VEC_COLS + c;
#pragma unroll
    for (int it = 0; it < LDGS; it++) {
      dzy_sum[it].store_to(params.dgamma_part, idx);
      idx += Ktraits::VEC_COLS_PER_LDG;
    }
  } else {
    static_assert(WARPS_M == 1 || Ktraits::CTAS_PER_ROW == 1,
                  "Multiple rows per CTA not supported for Multi-CTA.");
    // Finalize reduction of part dgamma and dbeta for this CTA
    // by reducing over the rows held across the WARPS_M warps

    // Assumption: blockSize divides hidden size.
    enum { NUM_RES = COLS / Ktraits::THREADS_PER_CTA };
    static_assert(NUM_RES * Ktraits::THREADS_PER_CTA == COLS, "");

    idx = warp_m * Ktraits::VEC_COLS + tid_r;
#pragma unroll
    for (int it = 0; it < LDGS; it++) {
      dzy_sum[it].store_to(smem_wgrad, idx);
      idx += THREADS_PER_ROW;
    }
    __syncthreads();
    compute_t cta_dzy_sum[NUM_RES];
    memset(cta_dzy_sum, 0, sizeof(compute_t) * NUM_RES);
    for (int it = 0; it < ROWS_PER_CTA; it++) {
      for (int jt = 0; jt < NUM_RES; jt++) {
        cta_dzy_sum[jt] += smem_wgrad[it * COLS + tidx + jt * Ktraits::THREADS_PER_CTA];
      }
    }

    compute_t *dgamma_part = static_cast<compute_t *>(params.dgamma_part) + bidm * COLS + tidx;
    for (int jt = 0; jt < NUM_RES; jt++) {
      *dgamma_part = cta_dzy_sum[jt];
      dgamma_part += Ktraits::THREADS_PER_CTA;
    }
  }
}

template <typename Kernel_traits>
__global__ __launch_bounds__(Kernel_traits::THREADS_PER_CTA) void rmsnorm_bwd_finalize_tuned_kernel(
    BwdParams params) {
  using compute_t = typename Kernel_traits::compute_t;
  using weight_t = typename Kernel_traits::weight_t;
  using index_t = typename Kernel_traits::index_t;
  using Reducer = typename Kernel_traits::Reducer;
  using reduce_t = typename Reducer::Type;

  Sum<reduce_t> sum;
  enum { NUM_ELT = Kernel_traits::ELTS_PER_LDG };
  enum { THREADS_PER_WARP = Kernel_traits::THREADS_PER_WARP };

  __shared__ char smem_[Kernel_traits::SMEM_BYTES_PER_CTA];

  constexpr uint32_t bidm = 0;

  const uint32_t bidn = blockIdx.x;
  const uint32_t tidx = threadIdx.x;
  const uint32_t warp = tidx / THREADS_PER_WARP;
  const uint32_t lane = tidx % THREADS_PER_WARP;

  Reducer reducer(params, bidm, bidn, 0, 0, lane, smem_);

  const uint32_t c = bidn * THREADS_PER_WARP + lane;
  const uint32_t c_out = bidn * THREADS_PER_WARP / 2 + lane;
  constexpr uint32_t COL_STRIDE = Kernel_traits::CTAS * THREADS_PER_WARP;
  for (uint32_t col = c, col_out = c_out; col < Kernel_traits::COLS;
       col += COL_STRIDE, col_out += COL_STRIDE / 2) {
    // Each thread sums over NUM_ELT columns.
    Vec<compute_t, NUM_ELT> dbeta_local, dgamma_local;
    memset(&dgamma_local, 0, sizeof(dgamma_local));
    for (uint32_t row = warp; row < params.ctas_per_col; row += Kernel_traits::ROWS_PER_CTA) {
      index_t idx = row * Kernel_traits::COLS + col;

      Vec<compute_t, NUM_ELT> dgamma_part;
      dgamma_part.load_from(params.dgamma_part, idx);
#pragma unroll
      for (int it = 0; it < NUM_ELT; it++) {
        dgamma_local.data.elt[it] += dgamma_part.data.elt[it];
      }
    }

    void *smem_gamma = smem_;

    const int write_row = warp;
    const int write_col = lane ^ write_row;
    const int write_idx = write_row * THREADS_PER_WARP + write_col;

    dgamma_local.store_to(smem_gamma, write_idx);

    __syncthreads();

    // It would be probably safe to reuse the first row of smem_gamma
    void *smem_gamma_out = &smem_[2 * Kernel_traits::SMEM_BYTES_TRANSPOSE];

    // More than one iter iff ROWS_PER_CTA < 32.
    for (int w = warp; w < THREADS_PER_WARP; w += Kernel_traits::ROWS_PER_CTA) {
      const int read_row = lane;
      const int read_col = w ^ read_row;
      const int read_idx = read_row * THREADS_PER_WARP + read_col;

      memset(&dgamma_local, 0, sizeof(dgamma_local));

      // Load gamma transposed
      if (read_row < Kernel_traits::ROWS_PER_CTA) {
        dgamma_local.load_from(smem_gamma, read_idx);
      }

// Call reducer on the loaded value(s) and convert.
#pragma unroll
      for (int it = 0; it < NUM_ELT; it++) {
        compute_t g_i = dgamma_local.data.elt[it];
        g_i = reducer.allreduce(g_i, sum);

        dgamma_local.data.elt[it] = g_i;
      }

      // Leader stores the result at the current column.
      if (lane == 0) {
        dgamma_local.store_to(smem_gamma_out, w);
      }
    }

    // All writes done.
    __syncthreads();

    // Pack and store: 2-wide stores with half the threads.
    if (warp == Kernel_traits::ROWS_PER_CTA - 1 && lane < THREADS_PER_WARP / 2) {
      using src_t = typename TypeToVec2<compute_t>::Type;
      using dst_t = typename TypeToVec2<weight_t>::Type;
      Vec<src_t, NUM_ELT> dgamma_vec2;
      Vec<dst_t, NUM_ELT> dgamma_out2;

      dgamma_vec2.load_from(smem_gamma_out, lane);
#pragma unroll
      for (int it = 0; it < NUM_ELT; it++) {
        dgamma_out2.data.elt[it] = Converter<src_t, dst_t>::convert(dgamma_vec2.data.elt[it]);
      }
      dgamma_out2.store_to(params.dgamma, col_out);
    }
  }
}

template <typename Ktraits>
__global__ __launch_bounds__(Ktraits::THREADS_PER_CTA) void rmsnorm_bwd_general_kernel(
    BwdParams params) {
  enum { LDGS = Ktraits::LDGS };
  enum { NUM_ELTS = Ktraits::ELTS_PER_LDG };
  enum { WARPS_M = Ktraits::WARPS_M };
  enum { WARPS_N = Ktraits::WARPS_N };

  using input_t = typename Ktraits::input_t;
  using weight_t = typename Ktraits::weight_t;
  using compute_t = typename Ktraits::compute_t;
  using output_t = typename Ktraits::output_t;
  using index_t = typename Ktraits::index_t;
  using Ivec = typename Ktraits::Ivec;
  using Ovec = typename Ktraits::Ovec;
  using Wvec = typename Ktraits::Wvec;
  using Cvec = typename Ktraits::Cvec;

  const index_t tidx = threadIdx.x;
  const index_t lane = tidx % THREADS_PER_WARP;
  const index_t warp = tidx / THREADS_PER_WARP;
  const index_t warp_m = warp / WARPS_N;
  const index_t warp_n = warp % WARPS_N;

  const index_t bdimm = WARPS_M;
  const index_t bdimn = WARPS_N * THREADS_PER_WARP;
  const index_t bidm = blockIdx.x / params.ctas_per_row;
  const index_t bidn = blockIdx.x % params.ctas_per_row;

  const index_t gdimm = bdimm * params.ctas_per_col;
  const index_t gdimn = bdimn * params.ctas_per_row;
  const index_t gidm = bidm * bdimm + warp_m;
  const index_t gidn = (bidn * THREADS_PER_WARP + warp_n * params.ctas_per_row * THREADS_PER_WARP +
                        lane);  // Order threads by warp x cta x lane

  // Objects for weight grads
  Cvec dzy_sum[LDGS];
  memset(dzy_sum, 0, sizeof(dzy_sum));

  // Objects for stats reductions
  using reduce_t = typename Ktraits::Reducer::Type;
  using Reducer = DynamicReducer<reduce_t, WARPS_M, WARPS_N>;
  constexpr int SMEM_BYTES = Reducer::SMEM_BYTES > 0 ? Reducer::SMEM_BYTES : 1;
  __shared__ char smem_[SMEM_BYTES];
  Reducer reducer(params, bidm, bidn, warp_m, warp_n, lane, smem_);
  Sum<reduce_t> sum;
  const compute_t rn = 1.f / static_cast<compute_t>(params.cols);

  // Load weights
  Cvec gamma[LDGS];
#pragma unroll
  for (int it = 0, col = gidn * NUM_ELTS; it < LDGS && col < params.cols;
       it++, col += gdimn * NUM_ELTS) {
    Wvec gamma_in;
    gamma_in.load_from_elts(params.gamma, col, params.cols - col);
    gamma_in.to(gamma[it]);
  }

  for (int cta_row = bidm * bdimm; cta_row < params.rows; cta_row += gdimm) {
    const int row = cta_row + warp_m;
    compute_t rs = 0.f;
    if (row < params.rows) {
      rs = static_cast<const compute_t *>(params.rs)[row];
    }

    Cvec dy[LDGS];
    Cvec y[LDGS];
    compute_t mdy = 0.f;
    compute_t mdyy = 0.f;

#pragma unroll
    for (int it = 0, col = gidn * NUM_ELTS; it < LDGS && row < params.rows && col < params.cols;
         it++, col += gdimn * NUM_ELTS) {
      Ivec x;
      Ovec dz;
      x.load_from_elts(params.x, row * params.cols + col, params.cols - col);
      dz.load_from_elts(params.dz, row * params.cols + col, params.cols - col);
#pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) {
        compute_t x_ij = x.data.elt[jt];
        compute_t y_ij = rs * (x_ij);
        const compute_t g_ij_shift = (params.zero_centered_gamma) ? 1.0f : 0.f;
        compute_t g_ij = gamma[it].data.elt[jt] + g_ij_shift;
        compute_t dz_ij = dz.data.elt[jt];
        compute_t dy_ij = g_ij * dz_ij;

        y[it].data.elt[jt] = y_ij;
        dy[it].data.elt[jt] = dy_ij;

        mdy += dy_ij;
        mdyy += dy_ij * y_ij;

        dzy_sum[it].data.elt[jt] += dz_ij * y_ij;
      }
    }

    // Reduce over row
    reduce_t result = reducer.allreduce({mdy, mdyy}, sum);
    mdy = Get<0>::of<reduce_t, compute_t>(result) * rn;
    mdyy = Get<1>::of<reduce_t, compute_t>(result) * rn;

// Compute dx
#pragma unroll
    for (int it = 0, col = gidn * NUM_ELTS; it < LDGS && row < params.rows && col < params.cols;
         it++, col += gdimn * NUM_ELTS) {
      Ivec dx;
#pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) {
        compute_t dy_ij = dy[it].data.elt[jt];
        compute_t y_ij = y[it].data.elt[jt];
        dx.data.elt[jt] = rs * (dy_ij - (mdyy * y_ij));
      }
      dx.store_to_elts(params.dx, row * params.cols + col, params.cols - col);
    }
  }

  if constexpr (WARPS_M == 1) {
// Write out local weight grad contributions
#pragma unroll
    for (int it = 0, col = gidn * NUM_ELTS; it < LDGS && col < params.cols;
         it++, col += gdimn * NUM_ELTS) {
      dzy_sum[it].store_to_elts(params.dgamma_part, bidm * params.cols + col, params.cols - col);
    }
  } else {
    // Reduce weight grad contributions within CTA before writing
    __shared__ Cvec vecs_shared[LDGS][WARPS_M][WARPS_N][THREADS_PER_WARP + 1];

    // Reduce dzy
    __syncthreads();
#pragma unroll
    for (int it = 0, col = gidn * NUM_ELTS; it < LDGS && col < params.cols;
         it++, col += gdimn * NUM_ELTS) {
      if (it != warp_m) {
        dzy_sum[it].store_to(&vecs_shared[it][warp_m][warp_n][lane]);
      }
    }
    __syncthreads();
#pragma unroll
    for (int it = warp_m, col = (gidn + it * gdimn) * NUM_ELTS; it < LDGS && col < params.cols;
         it += WARPS_M, col += WARPS_M * gdimn * NUM_ELTS) {
#pragma unroll
      for (int kt = 0; kt < WARPS_M; kt++) {
        if (kt != warp_m) {
#pragma unroll
          for (int jt = 0; jt < NUM_ELTS; jt++) {
            dzy_sum[it].data.elt[jt] += vecs_shared[it][kt][warp_n][lane].data.elt[jt];
          }
        }
      }
      dzy_sum[it].store_to_elts(params.dgamma_part, bidm * params.cols + col, params.cols - col);
    }
  }
}

template <typename weight_t, typename compute_t, uint32_t WARPS_M, uint32_t WARPS_N,
          uint32_t BYTES_PER_LDG, uint32_t THREADS_PER_WARP>
__global__ __launch_bounds__(
    WARPS_M *WARPS_N *THREADS_PER_WARP) void rmsnorm_bwd_finalize_general_kernel(BwdParams params) {
  enum { NUM_ELTS = BYTES_PER_LDG / sizeof(compute_t) };
  using Wvec = Vec<weight_t, NUM_ELTS>;
  using Cvec = Vec<compute_t, NUM_ELTS>;

  const int lane = threadIdx.x % THREADS_PER_WARP;
  const int warp_m = threadIdx.y;
  const int warp_n = threadIdx.x / THREADS_PER_WARP;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Load grad contributions and accumulate locally
  Cvec dgamma;
  dgamma.clear();
  for (int row = warp_m; row < params.ctas_per_col && col < params.cols; row += WARPS_M) {
    Cvec dgamma_part;
    dgamma_part.load_from_elts(params.dgamma_part, row * params.cols + col, params.cols - col);
#pragma unroll
    for (int jt = 0; jt < NUM_ELTS; jt++) {
      dgamma.data.elt[jt] += dgamma_part.data.elt[jt];
    }
  }

  // Reduce dgamma within CTA
  __shared__ Cvec vecs_shared[WARPS_M][WARPS_N][THREADS_PER_WARP + 1];
  dgamma.store_to(&vecs_shared[warp_m][warp_n][lane]);
#pragma unroll
  for (int nrows = WARPS_M / 2; nrows > 0; nrows /= 2) {
    __syncthreads();
    if (warp_m < nrows) {
#pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) {
        vecs_shared[warp_m][warp_n][lane].data.elt[jt] +=
            vecs_shared[warp_m + nrows][warp_n][lane].data.elt[jt];
      }
    }
  }
  if (warp_m == 0 && col < params.cols) {
    Wvec dgamma_out;
    vecs_shared[warp_m][warp_n][lane].to(dgamma_out);
    dgamma_out.store_to_elts(params.dgamma, col, params.cols - col);
  }
}

}  // namespace rmsnorm
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_RMSNORM_RMSNORM_BWD_KERNELS_CUH_
