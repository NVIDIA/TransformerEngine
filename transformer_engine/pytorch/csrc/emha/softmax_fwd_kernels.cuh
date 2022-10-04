#pragma once
/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "philox.h"
#include "softmax.h"

namespace softmax {

#define REGISTER_SOFTMAX_FWD_LAUNCHER(HIDDEN_SIZE, ITYPE, OTYPE, CTYPE,       \
                                      WARPS_M, WARPS_N, MASK_TYPE)            \
  void softmax_fwd_##HIDDEN_SIZE##_##ITYPE##_##OTYPE##_##CTYPE##_##MASK_TYPE( \
      LaunchParams<FwdParams> &launch_params, const bool configure_params) {  \
    launch_<ITYPE, OTYPE, CTYPE, uint32_t, HIDDEN_SIZE, 1, WARPS_M, WARPS_N,  \
            MASK_TYPE, 16>(launch_params, configure_params);                  \
  }                                                                           \
  static FwdRegistrar<ITYPE, OTYPE, CTYPE, HIDDEN_SIZE, MASK_TYPE>            \
      reg_##HIDDEN_SIZE##_##ITYPE##_##OTYPE##_##CTYPE##_##MASK_TYPE(          \
          softmax_fwd_##HIDDEN_SIZE##_##ITYPE##_##OTYPE##_##CTYPE##_##MASK_TYPE)

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct Max {
  inline __device__ Max() {}
  inline __device__ T operator()(const T &a, const T &b) {
    return a > b ? a : b;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int NUM_ELTS>
struct Philox_helper {};

template <>
struct Philox_helper<4> {
  static __device__ inline void draw(Philox &ph, bool (&keep)[4],  // NOLINT(*)
                                     const float p_keep) {
    float4 r0 = uniform4(ph());
    keep[0 + 0] = r0.x <= p_keep;
    keep[0 + 1] = r0.y <= p_keep;
    keep[0 + 2] = r0.z <= p_keep;
    keep[0 + 3] = r0.w <= p_keep;
  }
};

template <>
struct Philox_helper<8> {
  static __device__ inline void draw(Philox &ph, bool (&keep)[8],  // NOLINT(*)
                                     const float p_keep) {
    float4 r0 = uniform4(ph());
    float4 r1 = uniform4(ph());
    keep[0 + 0] = r0.x <= p_keep;
    keep[0 + 1] = r0.y <= p_keep;
    keep[0 + 2] = r0.z <= p_keep;
    keep[0 + 3] = r0.w <= p_keep;

    keep[4 + 0] = r0.x <= p_keep;
    keep[4 + 1] = r0.y <= p_keep;
    keep[4 + 2] = r0.z <= p_keep;
    keep[4 + 3] = r0.w <= p_keep;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Ktraits>
__global__ __launch_bounds__(Ktraits::THREADS_PER_CTA) void softmax_fwd_kernel(
    FwdParams params) {
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

  typename Ktraits::Mask mask(params, bidb);

  Sum<reduce_t> sum;
  Max<reduce_t> max;

  auto seeds = at::cuda::philox::unpack(params.philox_args);

  const index_t tidx_global = (bidbh * gridDim.x + bidq) * blockDim.x + tidx;
  Philox ph(std::get<0>(seeds), tidx_global, std::get<1>(seeds));

  Ivec x[LDGS];
  index_t idx = r * Ktraits::VEC_COLS + c;
  compute_t xf[LDGS * NUM_ELTS];

  constexpr compute_t neg_inf = -std::numeric_limits<compute_t>::infinity();

  // Thread offset in the key sequence.
  index_t offset_sk = c * NUM_ELTS;
  compute_t local_max(neg_inf);
#pragma unroll
  for (int it = 0; it < LDGS; it++) {
    // Global load.
    x[it].load_from(params.x, idx);
#pragma unroll
    for (int jt = 0; jt < NUM_ELTS; jt++) {
      // Convert input to compute type.
      xf[it * NUM_ELTS + jt] = compute_t(x[it].data.elt[jt]);
      // Reference to the current element.
      compute_t &x_ij = xf[it * NUM_ELTS + jt];
      // Check if position is masked.
      bool is_masked = mask.is_masked(row_offset_sq, offset_sk + jt);
      // Apply masking or update x.
      x_ij = is_masked ? neg_inf : x_ij * params.scale_pre_softmax;
      // Compute thread-local max.
      local_max = max(x_ij, local_max);
    }
    idx += VEC_COLS_PER_LDG;
    // Advance position in key sequence.
    offset_sk += VEC_COLS_PER_LDG * NUM_ELTS;
  }

  // First reduction for max.
  compute_t global_max = reducer.allreduce(local_max, max);

  compute_t Z_local(0.f);
#pragma unroll
  for (int it = 0; it < LDGS; it++) {
#pragma unroll
    for (int jt = 0; jt < NUM_ELTS; jt++) {
      // Reference to the current element.
      compute_t &x_ij = xf[it * NUM_ELTS + jt];
      // Update x.
      x_ij = expf(x_ij - global_max);
      // Compute thread-local sum.
      Z_local = sum(x_ij, Z_local);
    }
  }
  // Second reduction for sum.
  compute_t Z_global = reducer.allreduce(Z_local, sum);

  // Normalizer and dropout scaling.
  // Defer multiplying this with the reciprocal of `params.p_keep` to better
  // precision. ref:
  // https://gitlab-master.nvidia.com/mkozuki/transformerengine/-/merge_requests/5
  compute_t Zinv = 1.f / Z_global;

  Ovec z[LDGS];
  idx = r * Ktraits::VEC_COLS + c;
  output_t rp_keep{1.0 / params.p_keep};
#pragma unroll
  for (int it = 0; it < LDGS; it++) {
    // Draw dropout flags.
    bool keep[NUM_ELTS];
    Philox_helper<NUM_ELTS>::draw(ph, keep, params.p_keep);
#pragma unroll
    for (int jt = 0; jt < NUM_ELTS; jt++) {
      // Apply dropout and softmax scaling.
      compute_t x_ij = Zinv * xf[it * NUM_ELTS + jt];
      // Encode dropout information.
      if (!keep[jt]) x_ij = -x_ij;
      // note: It seems the problem comes from not scaling with the output type
      // instead of in FP32 in line 173 (compliment: previously Zinv was defined
      // as 1.0 / (params.p_keep * rp_keep). This problem goes away when rp_keep
      // = 1.0.  This small difference seems to accumulate, since the GEMM
      // reduction dimension is 2048. Cast to output type.
      output_t y_ij = output_t(x_ij) * rp_keep;
      z[it].data.elt[jt] = y_ij;
    }
    // Global store.
    z[it].store_to(params.z, idx);
    idx += VEC_COLS_PER_LDG;
  }
}

}  // namespace softmax
