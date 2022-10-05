/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once


namespace softmax {

template <MaskMode MASK_MODE>
struct Mask {};

template <>
struct Mask<SELF> {
  template <typename Params>
  __device__ inline Mask(Params &params, const int bidb) {  // NOLINT(*)
    seqlen_ = params.cu_seqlens[bidb + 1] - params.cu_seqlens[bidb];
  }

  __device__ inline bool is_masked(const int idx_sq, const int idx_sk) {
    return idx_sq >= seqlen_ || idx_sk >= seqlen_;
  }

  int seqlen_;
};

template <>
struct Mask<CAUSAL> {
  template <typename Params>
  __device__ inline Mask(Params &params, const int bidb) {  // NOLINT(*)
    seqlen_ = params.cu_seqlens[bidb + 1] - params.cu_seqlens[bidb];
  }

  __device__ inline bool is_masked(const int idx_sq, const int idx_sk) {
    return idx_sq < idx_sk;
  }

  int seqlen_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <uint32_t HIDDEN_SIZE_, typename input_t_, typename output_t_,
          typename compute_t_, typename index_t_, uint32_t THREADS_PER_CTA_>
struct Kernel_traits_base {
  using input_t = input_t_;
  using output_t = output_t_;
  using compute_t = compute_t_;
  using index_t = index_t_;

  enum { HIDDEN_SIZE = HIDDEN_SIZE_ };
  enum { THREADS_PER_CTA = THREADS_PER_CTA_ };
  enum { THREADS_PER_WARP = 32 };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename input_t_, typename output_t_, typename compute_t_,
          typename index_t_, uint32_t HIDDEN_SIZE_, uint32_t CTAS_PER_ROW_,
          uint32_t WARPS_M_, uint32_t WARPS_N_, MaskMode MASK_MODE,
          uint32_t BYTES_PER_LDG_ = 16,
          typename Base = Kernel_traits_base<
              HIDDEN_SIZE_, input_t_, output_t_, compute_t_, index_t_,
              WARPS_M_ * WARPS_N_ * THREADS_PER_WARP> >
struct Kernel_traits : public Base {
  using input_t = typename Base::input_t;
  using compute_t = typename Base::compute_t;
  using output_t = typename Base::output_t;
  using index_t = typename Base::index_t;

  using Mask = Mask<MASK_MODE>;

  enum { CTAS_PER_ROW = CTAS_PER_ROW_ };
  enum { WARPS_M = WARPS_M_ };
  enum { WARPS_N = WARPS_N_ };
  enum { COLS = HIDDEN_SIZE_ };
  enum { HIDDEN_SIZE = HIDDEN_SIZE_ };
  enum { BYTES_PER_LDG = BYTES_PER_LDG_ };
  enum { NUM_ELTS = BYTES_PER_LDG / sizeof(input_t) };

  enum { THREADS_PER_ROW = WARPS_N * THREADS_PER_WARP };
  enum { THREADS_PER_CTA = WARPS_M * THREADS_PER_ROW };
  enum { ROWS_PER_CTA = WARPS_M };

  enum { BYTES_PER_ROW = COLS * sizeof(input_t) };
  enum { BYTES_PER_ROW_PER_CTA = THREADS_PER_ROW * BYTES_PER_LDG };
  // Multi-row per CTA not supported for multi-CTA => no smem for WGRAD needed
  static_assert(WARPS_M == 1 || CTAS_PER_ROW == 1);

  using reduce_t = compute_t;
  using Reducer = layer_norm::Reducer<reduce_t, CTAS_PER_ROW, WARPS_M, WARPS_N>;

  using Ivec = layer_norm::Vec<input_t, NUM_ELTS>;
  using Ovec = layer_norm::Vec<output_t, NUM_ELTS>;
  using Cvec = layer_norm::Vec<compute_t, NUM_ELTS>;
  enum { ELTS_PER_LDG = BYTES_PER_LDG / sizeof(input_t) };

  // Assume that each thread can handle the same number of elements in the
  // output as in the input.
  static_assert(sizeof(input_t) >= sizeof(output_t));
  // The number of columns fetched per load from input: one per thread.
  enum { VEC_COLS_PER_LDG = CTAS_PER_ROW * THREADS_PER_ROW };
  // The total number of vectorized loads/stores per hidden vector.
  enum { VEC_COLS = COLS / ELTS_PER_LDG };
  // The number of loads per thread for the input.
  enum { LDGS = VEC_COLS / VEC_COLS_PER_LDG };
  static_assert(LDGS * VEC_COLS_PER_LDG == VEC_COLS);
  // static_assert(LDGS * BYTES_PER_ROW_PER_CTA * CTAS_PER_ROW == BYTES_PER_ROW,
  // "");

  enum { SMEM_BYTES_FWD = Reducer::SMEM_BYTES };
  enum { SMEM_BYTES_BWD = Reducer::SMEM_BYTES };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace softmax
