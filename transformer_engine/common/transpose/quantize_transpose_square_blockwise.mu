/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <musa.h>
#include <musaTypedefs.h>
#include <musa_bf16.h>
#include <musa_runtime.h>

#include <cfloat>
#include <type_traits>
// #include <cuda/barrier>

#include "common/common.h"
#include "common/recipe/recipe_common.cuh"
#include "common/util/cuda_runtime.h"
#include "common/util/ptx.cuh"
#include "common/util/mtfp8_utils.muh"
#include "common/utils.cuh"

namespace transformer_engine {
namespace {

constexpr size_t BLOCK_TILE_DIM = 128;

using mtfp8::VlenTrait;
using mtfp8::warp_bits;
using mtfp8::warp_mask;
using mtfp8::warp_size;
using mtfp8::max_threads_per_block;
using mtfp8::max_warps_per_block;
using mtfp8::global_amax_min;
using mtfp8::ceil_div;
using mtfp8::is_power_of_2;
using mtfp8::io_bytes;


template <
    typename Param,
    float (*OP)(float, const Param &),
    typename IType,
    typename OType,
    typename CType,
    size_t VLEN,
    size_t NCol>
__global__ void fp8_nn_blockwise_n_to_1_kernel(
    const IType* inp,
    const CType* noop,
    OType* out,
    CType* sinv,
    size_t M,
    size_t N,
    size_t block_m,
    size_t block_n,
    size_t rounds,
    size_t n_warps,
    Param param) {
  if (noop != nullptr && noop[0] == 1.0f) return;
  using IVecT = Vec<IType, VLEN>;
  using CVecT = Vec<CType, VLEN>;
  using OVecT = Vec<OType, VLEN>;

  using Trait_VLEN = VlenTrait<VLEN>;
  using Trait_COL = VlenTrait<NCol>;

  extern __shared__ __align__(alignof(size_t)) char temp[];

  const size_t tid = threadIdx.y * blockDim.x + threadIdx.x;
  const size_t warp_id = tid >> warp_bits;
  const size_t lane_id = tid & warp_mask;

  auto* temp_base = reinterpret_cast<CVecT*>(temp);
  auto* offset_base = reinterpret_cast<size_t*>(temp + block_m * block_n * sizeof(CType)) + tid * rounds;

  const size_t stride = blockDim.x * blockDim.y;

  const size_t base_m = blockIdx.y * block_m;
  const size_t base_n = blockIdx.x * block_n;

  size_t idx = tid;
  size_t act_round = 0;
  IVecT vec_in;

  CType amax = 0;
  __shared__ CType staging[max_warps_per_block];

  for (; act_round < rounds; ++act_round) {
    size_t global_m = base_m;
    if constexpr (Trait_COL::is_power_of_2) {
      global_m += (idx >> Trait_COL::bits);
    } else {
      global_m += (idx / blockDim.x);
    }

    if (global_m < M) {
      size_t global_n = base_n;
      if constexpr (Trait_COL::is_power_of_2) {
        global_n += ((idx & Trait_COL::mask) << Trait_VLEN::bits);
      } else {
        global_n += ((idx % blockDim.x) << Trait_VLEN::bits);
      }

      size_t offset = global_m * N + global_n;
      *(offset_base + act_round) = offset;

      vec_in.load_from(inp + offset, 0);
      CVecT& temp = *(temp_base + idx);
#pragma unroll
      for (size_t j = 0; j < VLEN; ++j) {
        temp.data.elt[j] = (CType)(OP((float)vec_in.data.elt[j], param));
        amax = fmaxf(fabsf(temp.data.elt[j]), amax);
        amax = fmaxf(amax, global_amax_min);
      }
      idx += stride;
    } else {
      break;
    }
  }

  amax = warp_reduce_max<warp_size>(amax);
  if (lane_id == 0) {
    staging[warp_id] = amax;
  }
  __syncthreads_lm();

  amax = 0;
  if (warp_id == 0) {
    amax = tid < n_warps ? staging[tid] : 0;
    amax = warp_reduce_max<warp_size>(amax);
  }
  if (tid == 0) {
    staging[0] = amax;                                              // block_amax
    staging[1] = (CType)(Quantized_Limits<OType>::max_norm) / amax; // block_scale
  }
  __syncthreads_lm();
  amax = staging[1];

  OVecT vec_out;
  idx = tid;
  for (size_t i = 0; i < act_round; ++i) {
    CVecT& temp = *(temp_base + idx);
#pragma unroll
    for (size_t j = 0; j < VLEN; ++j) {
      vec_out.data.elt[j] = (OType)(temp.data.elt[j] * amax);
    }
    vec_out.store_to(out + *(offset_base + i), 0);
    idx += stride;
  }

  if (tid == 0) {
    const size_t sinv_offset = blockIdx.y * gridDim.x + blockIdx.x;
    *(sinv + sinv_offset) = staging[0] * (CType)(Quantized_Limits<OType>::max_norm_rcp);
  }
}

template <
    typename Param,
    float (*OP)(float, const Param &),
    typename IType,
    typename OType,
    typename CType,
    size_t VLEN>
__global__ void fp8_nn_blockwise_n_to_1_kernel_no_align(
    const IType* inp,
    const CType* noop,
    OType* out,
    CType* sinv,
    size_t M,
    size_t N,
    size_t block_m,
    size_t block_n,
    // dense
    int dense_m,
    int dense_n,
    // sparse
    int row_n,
    int col_n,
    Param param) {
  using IVecT = Vec<IType, VLEN>;
  using CVecT = Vec<CType, VLEN>;
  using OVecT = Vec<OType, VLEN>;

  if (noop != nullptr && noop[0] == 1.0f) return;

  const size_t tid = threadIdx.y * blockDim.x + threadIdx.x;
  const size_t warp_id = threadIdx.y;
  const size_t lane_id = threadIdx.x;

  const size_t base_m = blockIdx.y * block_m + warp_id;
  size_t base_n = blockIdx.x * block_n;

  extern __shared__ __align__(alignof(CType)) CType shm[];
  CType amax = 0;
  __shared__ CType staging[max_warps_per_block];

  const size_t data_strd = blockDim.y * N;

  size_t data_base = base_m * N;
  const IType* iptr = inp + data_base;

  size_t c_base;
  size_t c_strd;

  const bool is_dense = (blockIdx.x < dense_n && blockIdx.y < dense_m);
  if (is_dense) {
    base_n += lane_id * VLEN;
    c_base = tid;
    c_strd = blockDim.x * blockDim.y;

    CVecT* c_vec = reinterpret_cast<CVecT*>(shm) + c_base;
    iptr += base_n;

    IVecT vec_in;
    for (int i = 0; i < row_n; ++i, iptr+=data_strd, c_vec+=c_strd) {
      vec_in.load_from(iptr, 0);
      CVecT& c_tmp = *c_vec;
#pragma unroll
      for (size_t j = 0; j < VLEN; ++j) {
        c_tmp.data.elt[j] = (CType)(OP((float)vec_in.data.elt[j], param));
        amax = fmaxf(fabsf(c_tmp.data.elt[j]), amax);
      }
    }
  } else {
    base_n += lane_id;
    c_base = warp_id * block_n + lane_id;
    c_strd = blockDim.y * block_n;

    CType* cptr = shm + c_base;

    size_t id_m = base_m;
    for (int i = 0; (i < row_n) && (id_m < M); ++i, id_m+=blockDim.y, iptr+=data_strd, cptr+=c_strd) {
      size_t id_n = base_n;
      size_t ff_n = 0;
      for (int j = 0; (j < col_n) && (id_n < N); ++j, id_n+=blockDim.x, ff_n+=blockDim.x) {
        CType val = (CType)(OP((float)(*(iptr + id_n)), param));
        *(cptr + ff_n) = val;
        amax = fmaxf(fabsf(val), amax);
      }
    }
  }
  amax = fmaxf(amax, global_amax_min);

  amax = warp_reduce_max<warp_size>(amax);
  if (lane_id == 0) {
    staging[warp_id] = amax;
  }
  __syncthreads_lm();

  amax = 0;
  if (warp_id == 0) {
    amax = tid < blockDim.y ? staging[tid] : 0;
    amax = warp_reduce_max<warp_size>(amax);
  }
  if (tid == 0) {
    staging[0] = amax;                                              // block_amax
    staging[1] = (CType)(Quantized_Limits<OType>::max_norm) / amax; // block_scale
  }
  __syncthreads_lm();
  amax = staging[1];

  OType* optr = out + data_base;
  if (is_dense) {
    CVecT* c_vec = reinterpret_cast<CVecT*>(shm) + c_base;
    optr += base_n;

    OVecT vec_out;
    for (int i = 0; i < row_n; ++i, optr+=data_strd, c_vec+=c_strd) {
      CVecT& c_tmp = *c_vec;
#pragma unroll
      for (size_t j = 0; j < VLEN; ++j) {
        vec_out.data.elt[j] = (OType)(c_tmp.data.elt[j] * amax);
      }
      vec_out.store_to(optr, 0);
    }
  } else {
    CType* cptr = shm + c_base;

    size_t id_m = base_m;
    for (int i = 0; (i < row_n) && (id_m < M); ++i, id_m+=blockDim.y, optr+=data_strd, cptr+=c_strd) {
      size_t id_n = base_n;
      size_t ff_n = 0;
      for (int j = 0; (j < col_n) && (id_n < N); ++j, id_n+=blockDim.x, ff_n+=blockDim.x) {
        CType& val = *(cptr + ff_n);
        val *= amax;
        *(optr + id_n) = (OType)(val);
      }
    }
  }

  if (tid == 0) {
    const size_t sinv_offset = blockIdx.y * gridDim.x + blockIdx.x;
    *(sinv + sinv_offset) = staging[0] * (CType)(Quantized_Limits<OType>::max_norm_rcp);
  }
}

struct EmptyParam {};

__device__ inline float identity_op(float value, const EmptyParam&) { return value; }

template <
    typename IType,
    typename OType,
    typename CType>
inline void fp8_blockwise_cast_fused_transpose(
    const IType* inp,
    const CType* noop,
    OType* out,
    CType* sinv,
    size_t M,
    size_t N,
    size_t block_m,
    size_t block_n,
    musaStream_t stream) {
  NVTE_CHECK(block_m == block_n);

  const int block_x = (int)ceil_div(N, block_n);
  const int block_y = (int)ceil_div(M, block_m);
  dim3 blocks(block_x, block_y);

  if (N % block_n != 0) {
    NVTE_CHECK(is_power_of_2(block_n));

    const size_t dense_m = M / block_m;
    const size_t dense_n = N / block_n;
    const size_t full_vlen = block_n / warp_size;

    const auto col_n = ceil_div(block_n, warp_size);

    const auto thread_x = warp_size;
    const auto thread_y = std::min(block_m, max_warps_per_block);
    const auto row_n = ceil_div(block_m, thread_y);

    NVTE_CHECK(is_power_of_2(col_n));
    dim3 threads((int)(thread_x), (int)(thread_y));

    const size_t shm_trans = block_m * block_n * sizeof(CType);

    if (full_vlen == 4) {
      fp8_nn_blockwise_n_to_1_kernel_no_align<EmptyParam, identity_op, IType, OType, CType, 4>
          <<<blocks, threads, shm_trans, stream>>>(
              inp, noop, out, sinv, M, N, block_m, block_n, (int)dense_m, (int)dense_n,
              (int)row_n, (int)col_n, {});
    } else if (full_vlen == 2) {
      fp8_nn_blockwise_n_to_1_kernel_no_align<EmptyParam, identity_op, IType, OType, CType, 2>
          <<<blocks, threads, shm_trans, stream>>>(
              inp, noop, out, sinv, M, N, block_m, block_n, (int)dense_m, (int)dense_n,
              (int)row_n, (int)col_n, {});
    } else {
      NVTE_CHECK(block_n >= 32);
      fp8_nn_blockwise_n_to_1_kernel_no_align<EmptyParam, identity_op, IType, OType, CType, 1>
          <<<blocks, threads, shm_trans, stream>>>(
              inp, noop, out, sinv, M, N, block_m, block_n, (int)dense_m, (int)dense_n,
              (int)row_n, (int)col_n, {});
    }

    NVTE_CHECK_CUDA(musaGetLastError());
    return;
  }

  constexpr size_t VLEN = io_bytes / sizeof(IType);
  const size_t thread_x = block_n / VLEN;

  const size_t thread_y = std::min(block_m, max_threads_per_block / thread_x);
  dim3 threads((int)(thread_x), (int)(thread_y));

  const size_t n_threads = thread_x * thread_y;
  NVTE_CHECK(n_threads % warp_size == 0);
  const size_t n_warps = n_threads / warp_size;

  const size_t rounds = ceil_div(block_m, thread_y);
  const size_t shm_trans = block_m * block_n * sizeof(CType);
  const size_t shm_offsets = n_threads * rounds * sizeof(size_t);
  const size_t shm_total = shm_trans + shm_offsets;

  if (thread_x == 32) {
    fp8_nn_blockwise_n_to_1_kernel<EmptyParam, identity_op, IType, OType, CType, VLEN, 32>
        <<<blocks, threads, shm_total, stream>>>(inp, noop, out, sinv, M, N, block_m, block_n,
                                                 rounds, n_warps, {});
  } else if (thread_x == 16) {
    fp8_nn_blockwise_n_to_1_kernel<EmptyParam, identity_op, IType, OType, CType, VLEN, 16>
        <<<blocks, threads, shm_total, stream>>>(inp, noop, out, sinv, M, N, block_m, block_n,
                                                 rounds, n_warps, {});
  } else {
    fp8_nn_blockwise_n_to_1_kernel<EmptyParam, identity_op, IType, OType, CType, VLEN, 0>
        <<<blocks, threads, shm_total, stream>>>(inp, noop, out, sinv, M, N, block_m, block_n,
                                                 rounds, n_warps, {});
  }

  NVTE_CHECK_CUDA(musaGetLastError());
}

}  // namespace
}  // namespace transformer_engine

namespace transformer_engine::detail {

void quantize_transpose_square_blockwise(const SimpleTensor& input, SimpleTensor& scale_inv,
                                         SimpleTensor& scale_inv_t, SimpleTensor& output,
                                         SimpleTensor& output_t, const float epsilon,
                                         const bool return_transpose, const bool pow_2_scale,
                                         const SimpleTensor& noop_tensor, cudaStream_t stream) {
  NVTE_API_CALL(quantize_transpose_square_blockwise);
  checkCuDriverContext(stream);

  NVTE_CHECK(input.shape == output.shape, "Input and output must have the same shape.");
  const size_t row_length = input.shape.size() > 0 ? input.shape.back() : 1;
  size_t num_rows = 1;
  for (size_t i = 0; (i < input.shape.size() - 1) && (input.shape.size() > 0); ++i) {
    num_rows *= input.shape.at(i);
  }

  NVTE_CHECK(scale_inv.shape.size() == 2, "scale_inv must have 2 dimensions.");

  const float* noop_ptr = reinterpret_cast<const float*>(noop_tensor.dptr);

  (void)return_transpose;
  (void)scale_inv_t;
  (void)output_t;

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.dtype, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output.dtype, OutputType,
          fp8_blockwise_cast_fused_transpose<InputType, OutputType, float>(
              reinterpret_cast<const InputType*>(input.dptr), noop_ptr,
              reinterpret_cast<OutputType*>(output.dptr), reinterpret_cast<float*>(scale_inv.dptr),
              num_rows, row_length, BLOCK_TILE_DIM, BLOCK_TILE_DIM, stream);
          NVTE_CHECK_CUDA(musaGetLastError());
      )
  )

  NVTE_CHECK_CUDA(musaGetLastError());
}

}  // namespace transformer_engine::detail
