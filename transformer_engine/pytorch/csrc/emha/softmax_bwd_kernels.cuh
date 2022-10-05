/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once

namespace softmax {

#define REGISTER_SOFTMAX_BWD_LAUNCHER(HIDDEN_SIZE, ITYPE, OTYPE, CTYPE,      \
                                      WARPS_M, WARPS_N)                      \
  void softmax_bwd_##HIDDEN_SIZE##_##ITYPE##_##OTYPE##_##CTYPE(              \
      LaunchParams<BwdParams> &launch_params, const bool configure_params) { \
    launch_<ITYPE, OTYPE, CTYPE, uint32_t, HIDDEN_SIZE, 1, WARPS_M, WARPS_N, \
            SELF, 16>(launch_params, configure_params);                      \
  }                                                                          \
  static BwdRegistrar<ITYPE, OTYPE, CTYPE, HIDDEN_SIZE>                      \
      reg_##HIDDEN_SIZE##_##ITYPE##_##OTYPE##_##CTYPE(                       \
          softmax_bwd_##HIDDEN_SIZE##_##ITYPE##_##OTYPE##_##CTYPE)

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Ktraits>
__global__ __launch_bounds__(Ktraits::THREADS_PER_CTA) void softmax_bwd_kernel(
    BwdParams params) {
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
  using Cvec = typename Ktraits::Cvec;

  using Reducer = typename Ktraits::Reducer;
  using reduce_t = typename Reducer::Type;

  extern __shared__ char smem_[];

  const index_t bidq = blockIdx.x;
  const index_t bidh = blockIdx.y;
  const index_t bidb = blockIdx.z;

  const index_t tidx = threadIdx.x;
  const index_t lane = tidx % THREADS_PER_WARP;
  const index_t warp = tidx / THREADS_PER_WARP;
  const index_t warp_m = warp / WARPS_N;
  const index_t warp_n = warp % WARPS_N;

  const index_t bidbh = (bidb * gridDim.y + bidh);
  const index_t row_offset_bh = bidbh * gridDim.x * ROWS_PER_CTA;
  const index_t row_offset_sq = bidq * ROWS_PER_CTA + warp_m;

  const index_t r = row_offset_bh + row_offset_sq;
  const index_t c = warp_n * THREADS_PER_WARP + lane;

  // TODO special ctors for warp and block reducers
  Reducer reducer(params, 0, 0, warp_m, warp_n, lane, smem_);

  Sum<reduce_t> sum;

  Ovec dz[LDGS];
  Ovec sd[LDGS];
  index_t idx = r * Ktraits::VEC_COLS + c;
  compute_t dzf[LDGS * NUM_ELTS];
  compute_t sdf[LDGS * NUM_ELTS];
#pragma unroll
  for (int it = 0; it < LDGS; it++) {
    dz[it].load_from(params.dz, idx);
    sd[it].load_from(params.smat_dmask, idx);
#pragma unroll
    for (int jt = 0; jt < NUM_ELTS; jt++) {
      compute_t dz_ij = compute_t(dz[it].data.elt[jt]);
      compute_t sd_ij = compute_t(sd[it].data.elt[jt]);
      dzf[it * NUM_ELTS + jt] = dz_ij;
      sdf[it * NUM_ELTS + jt] = sd_ij;
    }
    idx += VEC_COLS_PER_LDG;
  }

  compute_t Z_local(0.f);
  compute_t rp_keep = 1.f / params.p_keep;
#pragma unroll
  for (int it = 0; it < LDGS; it++) {
#pragma unroll
    for (int jt = 0; jt < NUM_ELTS; jt++) {
      compute_t &dz_ij = dzf[it * NUM_ELTS + jt];
      compute_t &sd_ij = sdf[it * NUM_ELTS + jt];
      static_assert(sizeof(compute_t) == 4);
      const bool drop = reinterpret_cast<const uint32_t &>(sd_ij) & 0x80000000;

      // sd <- s
      sd_ij = fabsf(sd_ij);
      // dz <- ds * s
      dz_ij = drop ? compute_t(0) : dz_ij * sd_ij;

      sd_ij *= params.p_keep;
      Z_local = sum(dz_ij, Z_local);
    }
  }
  compute_t Z_global = reducer.allreduce(Z_local, sum);

  Ivec dx[LDGS];
  idx = r * Ktraits::VEC_COLS + c;
#pragma unroll
  for (int it = 0; it < LDGS; it++) {
#pragma unroll
    for (int jt = 0; jt < NUM_ELTS; jt++) {
      compute_t &ds_ij = dzf[it * NUM_ELTS + jt];
      compute_t &s_ij = sdf[it * NUM_ELTS + jt];

      output_t dx_ij =
          output_t((ds_ij - Z_global * s_ij) * params.scale_pre_softmax);
      dx[it].data.elt[jt] = dx_ij;
    }
    dx[it].store_to(params.dx, idx);
    idx += VEC_COLS_PER_LDG;
  }
}

}  // namespace softmax
