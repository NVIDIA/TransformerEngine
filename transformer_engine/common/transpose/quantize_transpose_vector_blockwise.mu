/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <musa.h>
#include <musaTypedefs.h>
#include <musa_bf16.h>
#include <musa_runtime.h>

#include <algorithm>
#include <cfloat>
#include <utility>

#include "common/common.h"
#include "common/recipe/recipe_common.cuh"
#include "common/transpose/cast_transpose.h"
#include "common/util/cuda_runtime.h"
#include "common/util/mtfp8_utils.muh"
#include "common/utils.cuh"

namespace transformer_engine {
namespace {

using transformer_engine::detail::FP8BlockwiseColumnwiseOption;
using transformer_engine::detail::FP8BlockwiseRowwiseOption;
using transformer_engine::mtfp8::global_amax_min;
using transformer_engine::mtfp8::next_power_of_2;
using transformer_engine::mtfp8::warp_size;

using CType = float;
constexpr size_t kGroupSize = 128;
constexpr size_t warps_per_tile = 4;
constexpr size_t block_size = warp_size * warps_per_tile;

template <typename T, int WIDTH = 32>
__device__ __forceinline__ T warpReduceMax(T v, unsigned mask = 0xffffffffu) {
  // static_assert((WIDTH & (WIDTH - 1)) == 0, "WIDTH must be power of 2");
  // static_assert(WIDTH <= 32, "WIDTH must be <= warpSize");
  #pragma unroll
  for (int offset = WIDTH >> 1; offset > 0; offset >>= 1) {
    T other = __shfl_xor_sync(mask, v, offset, WIDTH);
    v = fmaxf(v, other);
  }
  return v;
}

constexpr int max(int a, int b) {
  return a > b ? a : b;
}


template <typename IType, typename OType, size_t N_ELEMENTS_PER_THREAD_X = 4 /* VLEN */,
          size_t N_ELEMENTS_PER_THREAD_Y = 4, size_t BLOCK_SIZE_X = 32, size_t BLOCK_SIZE_Y = 16,
          size_t GROUP_SIZE = 128>
__global__ void mtfp8_cast_transpose_general_kernel_column_aligned(
    MUtensorDescriptor in_desc, const IType* __restrict__ const inp,
    const CType* __restrict__ const noop,
    OType* __restrict__ const out_c,                 
    OType* __restrict__ const out_t,                 
    CType* __restrict__ const scale_inv,             
    CType* __restrict__ const columnwise_scale_inv,  
    size_t ncols, size_t nrows) {

  // if (noop != nullptr && noop[0] == 1.0f) return;

  using input_vec_t = Vec<IType, N_ELEMENTS_PER_THREAD_X>;
  using out_vec_t = Vec<OType, N_ELEMENTS_PER_THREAD_X>;
  using scale_vec_t = Vec<CType, N_ELEMENTS_PER_THREAD_X>;
  using in_vec2 = Vec<IType, 2>;
  using f32_vec2 = Vec<float, 2>;
  using f32_vec_t = Vec<float, N_ELEMENTS_PER_THREAD_X>;

  const uint32_t local_col_base_id = threadIdx.x * N_ELEMENTS_PER_THREAD_X;
  const uint32_t global_col_base_id = blockIdx.x * GROUP_SIZE + local_col_base_id;
  const uint32_t local_row_base_id = threadIdx.y * N_ELEMENTS_PER_THREAD_Y;
  const uint32_t global_row_base_id = blockIdx.y * GROUP_SIZE;

  const uint32_t rowwise_scale_inv_stride = ncols / GROUP_SIZE;

  const IType* inp_load_ptr = inp + global_row_base_id * ncols + global_col_base_id;
  OType* out_c_store_ptr = out_c + global_row_base_id * ncols + global_col_base_id;
  CType* rowwise_scale_inv_ptr =
      scale_inv + global_row_base_id * rowwise_scale_inv_stride + blockIdx.x;
  OType* out_t_store_ptr = out_t + global_row_base_id * ncols + global_col_base_id;
  CType* columnwise_scale_inv_ptr = columnwise_scale_inv + blockIdx.y * ncols + global_col_base_id;

  const uint32_t rows_in_this_block = min(GROUP_SIZE, nrows - global_row_base_id);
  constexpr int REPEAT_Y =
      DIVUP(GROUP_SIZE, BLOCK_SIZE_Y * N_ELEMENTS_PER_THREAD_Y);  // 128 / (16 * 2) = 4

  __shared__ __align__(128) IType shm[GROUP_SIZE][GROUP_SIZE];
  __shared__ IType shm_amax_columnwise[BLOCK_SIZE_Y][GROUP_SIZE + 2];
  __shared__ float shm_rcp_amax_col_final[GROUP_SIZE];

  int local_tidx = threadIdx.y * blockDim.x + threadIdx.x;
  int warp_id = local_tidx >> 5;
  int lane_id = local_tidx & 31;

  float amax_rowwise;
  float amax_columnwise[N_ELEMENTS_PER_THREAD_X] = {0.f};

  #pragma unroll
  for (int ii_x = 0; ii_x < N_ELEMENTS_PER_THREAD_X; ++ii_x) {
    shm_amax_columnwise[threadIdx.y][local_col_base_id + ii_x] = 0.0f;
    amax_columnwise[ii_x] = 0.0f;
  }
  __syncthreads_lm();

  input_vec_t tmp_load_reg;
  out_vec_t tmp_store_reg;
  scale_vec_t scale_store_reg;

#pragma unroll
  for (int loop_y_id = 0; loop_y_id < REPEAT_Y; loop_y_id++) {
    // TODO: try prefetch

    int group_inner_y_id = loop_y_id * BLOCK_SIZE_Y * N_ELEMENTS_PER_THREAD_Y + local_row_base_id;
// load input values into shared memory
#pragma unroll
    for (int ii_y = 0; ii_y < N_ELEMENTS_PER_THREAD_Y; ii_y++) {
      amax_rowwise = 0.f;
      int ld_st_offset = group_inner_y_id + ii_y;
      if (ld_st_offset >= rows_in_this_block) {
         break;
      }
      *reinterpret_cast<input_vec_t*>(shm[group_inner_y_id + ii_y] + local_col_base_id) =
          *reinterpret_cast<const input_vec_t*>(inp_load_ptr + ld_st_offset * ncols);
      tmp_load_reg.load_from(shm[group_inner_y_id + ii_y] + local_col_base_id, 0);

#pragma unroll
      for (int ii_x = 0; ii_x < N_ELEMENTS_PER_THREAD_X; ii_x++) {
        float abs_val = fabsf(tmp_load_reg.data.elt[ii_x]);
        float val_with_min = fmaxf(abs_val, global_amax_min);
        amax_rowwise = fmaxf(amax_rowwise, val_with_min);
        amax_columnwise[ii_x] = fmaxf(amax_columnwise[ii_x], val_with_min);
        shm_amax_columnwise[threadIdx.y][local_col_base_id + ii_x] = amax_columnwise[ii_x];
      }

      amax_rowwise = warpReduceMax<float, 16>(amax_rowwise) * (float)(Quantized_Limits<OType>::max_norm_rcp);
      const float rcp_amax_rowwise =
          1.0f / amax_rowwise;  //(amax_rowwise > 0.f) ? (1.0f / amax_rowwise) : 0.f;
                                //// write back to scale_inv and out_c [rowwise result]
#pragma unroll
      for (int ii_x = 0; ii_x < N_ELEMENTS_PER_THREAD_X; ii_x++) {
        tmp_store_reg.data.elt[ii_x] =
            (OType)(float(tmp_load_reg.data.elt[ii_x]) * rcp_amax_rowwise);
      }
      tmp_store_reg.store_to(out_c_store_ptr + ld_st_offset * ncols, 0);
      if (threadIdx.x == 0) {
        rowwise_scale_inv_ptr[ld_st_offset * rowwise_scale_inv_stride] = amax_rowwise;
      }
    }
  }

  // RUN COLUMNWISE
  __syncthreads_lm();

  in_vec2 tid_amax_vec2;
  in_vec2 amax_col_vec2;
  int base_col = warp_id << 3;
  int col_pair = lane_id >> 3;
  int row_base = lane_id & 7;
  int col_idx = base_col + col_pair * 2;
  amax_col_vec2.load_from(&(shm_amax_columnwise[row_base][col_idx]));
  tid_amax_vec2 = amax_col_vec2;
#pragma unroll
  for (int k = 1; k < 4; k++) {
    int row_idx = row_base + k * 8;
    amax_col_vec2.load_from(&(shm_amax_columnwise[row_idx][col_idx]));
    tid_amax_vec2.data.elt[0] = fmaxf(tid_amax_vec2.data.elt[0], amax_col_vec2.data.elt[0]);
    tid_amax_vec2.data.elt[1] = fmaxf(tid_amax_vec2.data.elt[1], amax_col_vec2.data.elt[1]);
  }

  in_vec2 reduce_amax_col = tid_amax_vec2;
  for (int offset = 4; offset >= 1; offset /= 2) {
    reduce_amax_col.data.elt[0] =
        fmaxf(reduce_amax_col.data.elt[0],
              __shfl_down_sync(0xffffffff, reduce_amax_col.data.elt[0], offset));
    reduce_amax_col.data.elt[1] =
        fmaxf(reduce_amax_col.data.elt[1],
              __shfl_down_sync(0xffffffff, reduce_amax_col.data.elt[1], offset));
  }

  if (row_base == 0) {
    reduce_amax_col.store_to(&(shm_amax_columnwise[0][col_idx]));
  }
  __syncthreads_lm();



  input_vec_t amax_columnwise_vec_reg;
  amax_columnwise_vec_reg.load_from(&(shm_amax_columnwise[0][local_col_base_id]));
  f32_vec_t rcp_amax_col_final_vec_reg;
  if (threadIdx.y == 0) {
#pragma unroll
    for (int ii = 0; ii < N_ELEMENTS_PER_THREAD_X; ++ii) {
      amax_columnwise[ii] = (float)(amax_columnwise_vec_reg.data.elt[ii]) *
                            (Quantized_Limits<OType>::max_norm_rcp);
      shm_rcp_amax_col_final[local_col_base_id + ii] = 1.0f / amax_columnwise[ii];
      columnwise_scale_inv_ptr[ii] = amax_columnwise[ii];
    }
  }
  __syncthreads_lm();
  // write back to columnwise_scale_inv and out_t
  scale_vec_t rcp_amax_columnwise;
  rcp_amax_columnwise.load_from(&(shm_rcp_amax_col_final[local_col_base_id]));

#pragma unroll
  for (int loop_y_id = 0; loop_y_id < REPEAT_Y; loop_y_id++) {
    int group_inner_y_id = loop_y_id * BLOCK_SIZE_Y * N_ELEMENTS_PER_THREAD_Y + local_row_base_id;
#pragma unroll
    for (int ii_y = 0; ii_y < N_ELEMENTS_PER_THREAD_Y; ii_y++) {
      int group_inner_y_offset = group_inner_y_id + ii_y;
      //   int store_offset =
    //       (global_row_base_id + group_inner_y_offset) < nrows ? group_inner_y_offset : 0;
      if (group_inner_y_offset < rows_in_this_block) {
         tmp_load_reg.load_from(&(shm[group_inner_y_offset][local_col_base_id]));
#pragma unroll
      for (int ii_x = 0; ii_x < N_ELEMENTS_PER_THREAD_X; ii_x++) {
        
        float value = (float)(tmp_load_reg.data.elt[ii_x]) * rcp_amax_columnwise.data.elt[ii_x];
        tmp_store_reg.data.elt[ii_x] = (OType)(value);
      }
      tmp_store_reg.store_to(out_t_store_ptr + group_inner_y_offset * ncols, 0);
      }
    }
  }  

}
template <typename IType, typename OType, size_t N_ELEMENTS_PER_THREAD_X = 8 /* VLEN */,
          size_t N_ELEMENTS_PER_THREAD_Y = 4, size_t BLOCK_SIZE_X = 16, size_t BLOCK_SIZE_Y = 32,
          size_t GROUP_SIZE = 128>
__global__ void mtfp8_cast_rowise_aligned(
    MUtensorDescriptor in_desc, const IType* __restrict__ const inp,
    const CType* __restrict__ const noop,
    OType* __restrict__ const out_c,   
    CType* __restrict__ const scale_inv,
    size_t ncols, size_t nrows) {
  // if (noop != nullptr && noop[0] == 1.0f) return;

  using input_vec_t = Vec<IType, N_ELEMENTS_PER_THREAD_X>;
  using out_vec_t   = Vec<OType, N_ELEMENTS_PER_THREAD_X>;

  const uint32_t local_col_base_id  = threadIdx.x * N_ELEMENTS_PER_THREAD_X;
  const uint32_t global_col_base_id = blockIdx.x * GROUP_SIZE + local_col_base_id;

  const uint32_t local_row_base_id  = threadIdx.y * N_ELEMENTS_PER_THREAD_Y;
  const uint32_t global_row_base_id = blockIdx.y * GROUP_SIZE;

  const uint32_t rowwise_scale_inv_stride = ncols / GROUP_SIZE;

  OType* out_c_store_ptr = out_c + global_row_base_id * ncols + global_col_base_id;
  CType* rowwise_scale_inv_ptr =
      scale_inv + global_row_base_id * rowwise_scale_inv_stride + blockIdx.x;

  const uint32_t rows_in_this_block = min((uint32_t)GROUP_SIZE, (uint32_t)(nrows - global_row_base_id));
  constexpr int REPEAT_Y =
      DIVUP(GROUP_SIZE, BLOCK_SIZE_Y * N_ELEMENTS_PER_THREAD_Y);

  // Input tile staged into shared memory via TME
  __shared__ __align__(128) IType shm[GROUP_SIZE][GROUP_SIZE];

  int local_tidx = threadIdx.y * blockDim.x + threadIdx.x;
  int warp_id    = local_tidx >> 5;

  __musa::async_barrier bar(1);
  if (local_tidx == 0) {
    bar.init_arrival(1);
  }
  __syncthreads();

  int trans_count = GROUP_SIZE * GROUP_SIZE * sizeof(IType);
  __musa::tme_block_dim_v2 ld_dim(GROUP_SIZE, GROUP_SIZE);
  __musa::tme_block_pos_v2 ld_pos(blockIdx.x * GROUP_SIZE, blockIdx.y * GROUP_SIZE);

  if (warp_id == 0) {
    __musa::memcpy_async(bar, shm, &in_desc, ld_dim.get_param(), ld_pos.get_param(),
                         trans_count, 0, 3, 1);
    bar.arrive();
  }
  bar.wait(0);

  input_vec_t tmp_load_reg;
  out_vec_t   tmp_store_reg;

#pragma unroll
  for (int loop_y_id = 0; loop_y_id < REPEAT_Y; loop_y_id++) {
    int group_inner_y_id =
        loop_y_id * BLOCK_SIZE_Y * N_ELEMENTS_PER_THREAD_Y + local_row_base_id;

#pragma unroll
    for (int ii_y = 0; ii_y < (int)N_ELEMENTS_PER_THREAD_Y; ii_y++) {
      float amax_rowwise = 0.f;

      int ld_st_offset = group_inner_y_id + ii_y;
      if (ld_st_offset >= (int)rows_in_this_block) {
        break;
      }

      tmp_load_reg.load_from(shm[group_inner_y_id + ii_y] + local_col_base_id, 0);

#pragma unroll
      for (int ii_x = 0; ii_x < (int)N_ELEMENTS_PER_THREAD_X; ii_x++) {
        float abs_val     = fabsf((float)tmp_load_reg.data.elt[ii_x]);
        float val_with_min = fmaxf(abs_val, global_amax_min);
        amax_rowwise = fmaxf(amax_rowwise, val_with_min);
      }

      amax_rowwise =
          warpReduceMax<float, 16>(amax_rowwise) *
          (float)(Quantized_Limits<OType>::max_norm_rcp);

      const float rcp_amax_rowwise = 1.0f / amax_rowwise;

#pragma unroll
      for (int ii_x = 0; ii_x < (int)N_ELEMENTS_PER_THREAD_X; ii_x++) {
        tmp_store_reg.data.elt[ii_x] =
            (OType)((float)tmp_load_reg.data.elt[ii_x] * rcp_amax_rowwise);
      }
      tmp_store_reg.store_to(out_c_store_ptr + ld_st_offset * ncols, 0);

      if (threadIdx.x == 0) {
        rowwise_scale_inv_ptr[ld_st_offset * rowwise_scale_inv_stride] = (CType)amax_rowwise;
      }
    }
  }
}


template <typename IType, typename OType, size_t N_ELEMENTS_PER_THREAD_X = 8 /* VLEN */,
          size_t N_ELEMENTS_PER_THREAD_Y = 4, size_t BLOCK_SIZE_X = 16, size_t BLOCK_SIZE_Y = 32,
          size_t GROUP_SIZE = 128>
__global__ void mtfp8_cast_columnwise_aligned(
    MUtensorDescriptor in_desc, const IType* __restrict__ const inp,
    const CType* __restrict__ const noop,
    OType* __restrict__ const out_t,  
    CType* __restrict__ const columnwise_scale_inv,
    size_t ncols, size_t nrows) {
  // if (noop != nullptr && noop[0] == 1.0f) return;

  using input_vec_t = Vec<IType, N_ELEMENTS_PER_THREAD_X>;
  using out_vec_t   = Vec<OType, N_ELEMENTS_PER_THREAD_X>;
  using scale_vec_t = Vec<CType, N_ELEMENTS_PER_THREAD_X>;
  using in_vec2     = Vec<IType, 2>;
  using f32_vec_t   = Vec<float, N_ELEMENTS_PER_THREAD_X>;

  const uint32_t local_col_base_id  = threadIdx.x * N_ELEMENTS_PER_THREAD_X;
  const uint32_t global_col_base_id = blockIdx.x * GROUP_SIZE + local_col_base_id;

  const uint32_t local_row_base_id  = threadIdx.y * N_ELEMENTS_PER_THREAD_Y;
  const uint32_t global_row_base_id = blockIdx.y * GROUP_SIZE;

  OType* out_t_store_ptr = out_t + global_row_base_id * ncols + global_col_base_id;
  CType* columnwise_scale_inv_ptr =
      columnwise_scale_inv + blockIdx.y * ncols + global_col_base_id;

  const uint32_t rows_in_this_block = min((uint32_t)GROUP_SIZE, (uint32_t)(nrows - global_row_base_id));
  constexpr int REPEAT_Y =
      DIVUP(GROUP_SIZE, BLOCK_SIZE_Y * N_ELEMENTS_PER_THREAD_Y);

  __shared__ __align__(128) IType shm[GROUP_SIZE][GROUP_SIZE];
  __shared__ IType  shm_amax_columnwise[BLOCK_SIZE_Y][GROUP_SIZE + 2];
  __shared__ float  shm_rcp_amax_col_final[GROUP_SIZE];

  int local_tidx = threadIdx.y * blockDim.x + threadIdx.x;
  int warp_id    = local_tidx >> 5;
  int lane_id    = local_tidx & 31;

  __musa::async_barrier bar(1);
  if (local_tidx == 0) {
    bar.init_arrival(1);
  }
  __syncthreads();

  int trans_count = GROUP_SIZE * GROUP_SIZE * sizeof(IType);
  __musa::tme_block_dim_v2 ld_dim(GROUP_SIZE, GROUP_SIZE);
  __musa::tme_block_pos_v2 ld_pos(blockIdx.x * GROUP_SIZE, blockIdx.y * GROUP_SIZE);

  if (warp_id == 0) {
    __musa::memcpy_async(bar, shm, &in_desc, ld_dim.get_param(), ld_pos.get_param(),
                         trans_count, 0, 3, 1);
    bar.arrive();
  }
  bar.wait(0);

  // Build per-(threadIdx.y, col) partial columnwise max
  float amax_columnwise[N_ELEMENTS_PER_THREAD_X] = {0.f};

#pragma unroll
  for (int ii_x = 0; ii_x < (int)N_ELEMENTS_PER_THREAD_X; ++ii_x) {
    shm_amax_columnwise[threadIdx.y][local_col_base_id + ii_x] = (IType)0;
    amax_columnwise[ii_x] = 0.0f;
  }
  __syncthreads_lm();

  input_vec_t tmp_load_reg;

#pragma unroll
  for (int loop_y_id = 0; loop_y_id < REPEAT_Y; loop_y_id++) {
    int group_inner_y_id =
        loop_y_id * BLOCK_SIZE_Y * N_ELEMENTS_PER_THREAD_Y + local_row_base_id;

#pragma unroll
    for (int ii_y = 0; ii_y < (int)N_ELEMENTS_PER_THREAD_Y; ii_y++) {
      int ld_st_offset = group_inner_y_id + ii_y;
      if (ld_st_offset >= (int)rows_in_this_block) {
        break;
      }

      tmp_load_reg.load_from(shm[group_inner_y_id + ii_y] + local_col_base_id, 0);

#pragma unroll
      for (int ii_x = 0; ii_x < (int)N_ELEMENTS_PER_THREAD_X; ii_x++) {
        float abs_val      = fabsf((float)tmp_load_reg.data.elt[ii_x]);
        float val_with_min = fmaxf(abs_val, global_amax_min);
        amax_columnwise[ii_x] = fmaxf(amax_columnwise[ii_x], val_with_min);
        shm_amax_columnwise[threadIdx.y][local_col_base_id + ii_x] = (IType)amax_columnwise[ii_x];
      }
    }
  }

  // Reduce shm_amax_columnwise over "rows" dimension (32) to get final per-column amax
  __syncthreads_lm();

  in_vec2 tid_amax_vec2;
  in_vec2 amax_col_vec2;
  int base_col = warp_id << 3;      // warp_id * 8
  int col_pair = lane_id >> 3;      // / 8
  int row_base = lane_id & 7;       // % 8
  int col_idx  = base_col + col_pair * 2;

  amax_col_vec2.load_from(&(shm_amax_columnwise[row_base][col_idx]));
  tid_amax_vec2 = amax_col_vec2;

#pragma unroll
  for (int k = 1; k < 4; k++) {
    int row_idx = row_base + k * 8;
    amax_col_vec2.load_from(&(shm_amax_columnwise[row_idx][col_idx]));
    tid_amax_vec2.data.elt[0] = fmaxf((float)tid_amax_vec2.data.elt[0], (float)amax_col_vec2.data.elt[0]);
    tid_amax_vec2.data.elt[1] = fmaxf((float)tid_amax_vec2.data.elt[1], (float)amax_col_vec2.data.elt[1]);
  }

  in_vec2 reduce_amax_col = tid_amax_vec2;
  for (int offset = 4; offset >= 1; offset >>= 1) {
    reduce_amax_col.data.elt[0] =
        fmaxf((float)reduce_amax_col.data.elt[0],
              __shfl_down_sync(0xffffffff, (float)reduce_amax_col.data.elt[0], offset));
    reduce_amax_col.data.elt[1] =
        fmaxf((float)reduce_amax_col.data.elt[1],
              __shfl_down_sync(0xffffffff, (float)reduce_amax_col.data.elt[1], offset));
  }

  if (row_base == 0) {
    reduce_amax_col.store_to(&(shm_amax_columnwise[0][col_idx]));
  }
  __syncthreads_lm();

  // Compute (amax * max_norm_rcp), store columnwise_scale_inv, and cache rcp for output cast
  input_vec_t amax_columnwise_vec_reg;
  amax_columnwise_vec_reg.load_from(&(shm_amax_columnwise[0][local_col_base_id]));

  if (threadIdx.y == 0) {
#pragma unroll
    for (int ii = 0; ii < (int)N_ELEMENTS_PER_THREAD_X; ++ii) {
      float amax =
          (float)(amax_columnwise_vec_reg.data.elt[ii]) *
          (float)(Quantized_Limits<OType>::max_norm_rcp);

      shm_rcp_amax_col_final[local_col_base_id + ii] = 1.0f / amax;
      columnwise_scale_inv_ptr[ii] = (CType)amax;
    }
  }
  __syncthreads_lm();

  // Cast to FP8 using columnwise scale, write out_t
  scale_vec_t rcp_amax_columnwise;
  rcp_amax_columnwise.load_from(&(shm_rcp_amax_col_final[local_col_base_id]));

  out_vec_t tmp_store_reg;

#pragma unroll
  for (int loop_y_id = 0; loop_y_id < REPEAT_Y; loop_y_id++) {
    int group_inner_y_id =
        loop_y_id * BLOCK_SIZE_Y * N_ELEMENTS_PER_THREAD_Y + local_row_base_id;

#pragma unroll
    for (int ii_y = 0; ii_y < (int)N_ELEMENTS_PER_THREAD_Y; ii_y++) {
      int group_inner_y_offset = group_inner_y_id + ii_y;
      if (group_inner_y_offset < (int)rows_in_this_block) {
        tmp_load_reg.load_from(&(shm[group_inner_y_offset][local_col_base_id]));

#pragma unroll
        for (int ii_x = 0; ii_x < (int)N_ELEMENTS_PER_THREAD_X; ii_x++) {
          float value =
              (float)(tmp_load_reg.data.elt[ii_x]) * (float)rcp_amax_columnwise.data.elt[ii_x];
          tmp_store_reg.data.elt[ii_x] = (OType)value;
        }
        tmp_store_reg.store_to(out_t_store_ptr + group_inner_y_offset * ncols, 0);
      }
    }
  }
}

template <
    typename IType,
    typename OType,
    size_t N_ELEMENTS_PER_THREAD_X = 4/* VLEN */,
    size_t N_ELEMENTS_PER_THREAD_Y = 4,
    size_t BLOCK_SIZE_X = 32,
    size_t BLOCK_SIZE_Y = 16,
    size_t GROUP_SIZE = 128
>
__global__ void  mtfp8_cast_transpose_general_kernel_column_unaligned(
    const IType *__restrict__ const inp,
    const CType *__restrict__ const noop,
    OType *__restrict__ const out_c,
    OType *__restrict__ const out_t,
    CType *__restrict__ const scale_inv,
    CType *__restrict__ const columnwise_scale_inv,
    size_t ncols,
    size_t nrows) {
    // rowwise_group_size and columnwise_group_size should be equal

    // if (noop != nullptr && noop[0] == 1.0f) return;

    using input_vec_t = Vec<IType, N_ELEMENTS_PER_THREAD_X>;
    using out_vec_t = Vec<OType, N_ELEMENTS_PER_THREAD_X>;
    using scale_vec_t = Vec<CType, N_ELEMENTS_PER_THREAD_X>;

    const uint32_t local_col_base_id = threadIdx.x * N_ELEMENTS_PER_THREAD_X;
    uint32_t global_col_base_id = blockIdx.x * GROUP_SIZE;
    global_col_base_id += ((global_col_base_id + local_col_base_id) < ncols ? local_col_base_id : 0);
    const uint32_t local_row_base_id = threadIdx.y * N_ELEMENTS_PER_THREAD_Y;
    const uint32_t global_row_base_id = blockIdx.y * GROUP_SIZE;

    const uint32_t rowwise_scale_inv_stride = (ncols + GROUP_SIZE - 1) / GROUP_SIZE;

    const IType* inp_load_ptr = inp + global_row_base_id * ncols + global_col_base_id;
    OType* out_c_store_ptr = out_c + global_row_base_id * ncols + global_col_base_id;
    CType* rowwise_scale_inv_ptr = scale_inv + global_row_base_id * rowwise_scale_inv_stride + blockIdx.x;
    OType* out_t_store_ptr = out_t + global_row_base_id * ncols + global_col_base_id;
    CType* columnwise_scale_inv_ptr = columnwise_scale_inv + blockIdx.y * ncols + global_col_base_id;
    
    constexpr int REPEAT_Y = DIVUP(GROUP_SIZE, BLOCK_SIZE_Y * N_ELEMENTS_PER_THREAD_Y);
    constexpr int ELEMENTS_PER_BANK = 4 / sizeof(IType);  // dword of bank is 32 bits by default
    static_assert(ELEMENTS_PER_BANK != 0);
    constexpr int NDWORD = N_ELEMENTS_PER_THREAD_X / ELEMENTS_PER_BANK;
    static_assert(NDWORD != 0);

    // 0, 1, 2, ..., 31
    // 128 * 128 * 2 / 1024 + 128 * BLOCK_SIZE_Y * 2 / 1024
    // __shared__ IType shm[GROUP_SIZE][NDWORD][GROUP_SIZE / NDWORD];
    __shared__ IType shm[GROUP_SIZE][GROUP_SIZE];
    __shared__ IType shm_amax_columnwise[BLOCK_SIZE_Y][GROUP_SIZE + 2];

    float amax_rowwise;
    float amax_columnwise[N_ELEMENTS_PER_THREAD_X] = {0.f};

    input_vec_t tmp_load_reg;
    out_vec_t tmp_store_reg;
    scale_vec_t scale_store_reg;

    #pragma unroll
    for (int loop_y_id = 0; loop_y_id < REPEAT_Y; loop_y_id++) {
      // assume no multiple loads along X dimension

      // TODO: try prefetch

      int group_inner_y_id = loop_y_id * BLOCK_SIZE_Y * N_ELEMENTS_PER_THREAD_Y + local_row_base_id;
      // load input values into shared memory
      #pragma unroll
      for (int ii_y = 0; ii_y < N_ELEMENTS_PER_THREAD_Y; ii_y++) {
        amax_rowwise = 0.f;
        int ld_st_offset = global_row_base_id + group_inner_y_id + ii_y < nrows ?
                           group_inner_y_id + ii_y:
                           0;
        *reinterpret_cast<input_vec_t*>(shm[group_inner_y_id + ii_y] + local_col_base_id) = *reinterpret_cast<const input_vec_t*>(inp_load_ptr + ld_st_offset * ncols);
        tmp_load_reg.load_from(shm[group_inner_y_id + ii_y] + local_col_base_id, 0);
        
        #pragma unroll
        for (int ii_x = 0; ii_x < N_ELEMENTS_PER_THREAD_X; ii_x++) {
          amax_rowwise = fmaxf(fmaxf(amax_rowwise, fabsf(tmp_load_reg.data.elt[ii_x])), global_amax_min);
          amax_columnwise[ii_x] = fmaxf(fmaxf(amax_columnwise[ii_x], fabsf(tmp_load_reg.data.elt[ii_x])), global_amax_min);
        }

        amax_rowwise = warpReduceMax<float, 32>(amax_rowwise) * (float)(Quantized_Limits<OType>::max_norm_rcp);

        //// write back to scale_inv and out_c [rowwise result]
        for (int ii_x = 0; ii_x < N_ELEMENTS_PER_THREAD_X; ii_x++) {
            tmp_store_reg.data.elt[ii_x] = static_cast<OType>(float(tmp_load_reg.data.elt[ii_x]) / amax_rowwise);
        }
        tmp_store_reg.store_to(out_c_store_ptr + ld_st_offset * ncols, 0);
        if (threadIdx.x == 0) {
          rowwise_scale_inv_ptr[ld_st_offset * rowwise_scale_inv_stride] = amax_rowwise;
        }
      }

      for (int ii_x = 0; ii_x < N_ELEMENTS_PER_THREAD_X; ii_x++) {
        shm_amax_columnwise[threadIdx.y][local_col_base_id + ii_x] = amax_columnwise[ii_x];
      }
    }

    // RUN COLUMNWISE

    __syncthreads_lm();

    for (int i = threadIdx.y; i < GROUP_SIZE; i += blockDim.y) {
        IType amax = threadIdx.x < blockDim.y ? 
                     shm_amax_columnwise[threadIdx.x][i] :
                     (IType)0.f;
        amax = warpReduceMax<float, 32>((float)amax);
        if (threadIdx.x == 0) {
          shm_amax_columnwise[0][i] = amax;
        }
    }

    __syncthreads_lm();
    #pragma unroll
    for (int ii = 0; ii < N_ELEMENTS_PER_THREAD_X; ii++) {
      amax_columnwise[ii] = (float)shm_amax_columnwise[0][local_col_base_id + ii] * (float)(Quantized_Limits<OType>::max_norm_rcp);
    }

    // write back to columnwise_scale_inv and out_t
    for (int loop_y_id = 0; loop_y_id < REPEAT_Y; loop_y_id++) {
      int group_inner_y_id = loop_y_id * BLOCK_SIZE_Y * N_ELEMENTS_PER_THREAD_Y + local_row_base_id;
      for (int ii_y = 0; ii_y < N_ELEMENTS_PER_THREAD_Y; ii_y++) {
        int group_inner_y_offset = group_inner_y_id + ii_y;
        int store_offset = (global_row_base_id + group_inner_y_offset) < nrows ?
                            group_inner_y_offset :
                            0;
        
        for (int ii_x = 0; ii_x < N_ELEMENTS_PER_THREAD_X; ii_x++) {
          float value = (float)shm[group_inner_y_offset][local_col_base_id + ii_x] / amax_columnwise[ii_x];
          tmp_store_reg.data.elt[ii_x] = static_cast<OType>(value);
        }
        tmp_store_reg.store_to(out_t_store_ptr + store_offset * ncols, 0);
      }
    }
    if (threadIdx.y == 0) {
      #pragma unroll
      for (int i = 0; i < N_ELEMENTS_PER_THREAD_X; i++) {
        scale_store_reg.data.elt[i] = amax_columnwise[i];
      }
      scale_store_reg.store_to(columnwise_scale_inv_ptr, 0);
    }
}

}  // namespace
}  // namespace transformer_engine

namespace transformer_engine::detail {

void quantize_transpose_vector_blockwise(const SimpleTensor& input, SimpleTensor& scale_inv,
                                         SimpleTensor& scale_inv_t, SimpleTensor& output,
                                         SimpleTensor& output_t, const float epsilon,
                                         FP8BlockwiseRowwiseOption rowwise_option,
                                         FP8BlockwiseColumnwiseOption columnwise_option,
                                         const bool pow2_scale, const SimpleTensor& noop_tensor,
                                         cudaStream_t stream) {
  NVTE_API_CALL(quantize_transpose_vector_blockwise);
  checkCuDriverContext(stream);

  const size_t row_length = input.shape.size() > 0 ? input.shape.at(input.shape.size() - 1) : 1u;
  size_t num_elements = row_length;
  size_t num_rows = 1;
  for (size_t i = 0; (i < input.shape.size() - 1) && (input.shape.size() > 0); ++i) {
    num_rows *= input.shape.at(i);
    num_elements *= input.shape.at(i);
  }

  if (num_elements == 0) {
    return;
  }

  if (rowwise_option != FP8BlockwiseRowwiseOption::NONE
      && columnwise_option == FP8BlockwiseColumnwiseOption::NONE) {

    const size_t group_size = 128;

    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
        input.dtype, InputType,
        TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
            output.dtype, OutputType,

            constexpr int GROUP_SIZE = 128;  // TODO: extend other group_size
            NVTE_CHECK(group_size == GROUP_SIZE);
            constexpr int BLOCK_SIZE_Y = 32;
            constexpr int BLOCK_SIZE_X = 16;
            constexpr int N_ELEMENTS_PER_THREAD_X = std::min(GROUP_SIZE / BLOCK_SIZE_X, 8);
            constexpr int N_ELEMENTS_PER_THREAD_Y = 4;

            dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
            dim3 grid(DIVUP(row_length, group_size), DIVUP(num_rows, group_size));
            MUtensorDescriptor intensorDesc;
            MUtensorDescriptorDataType tensorDataType = MU_TENSOR_DESCRIPTOR_DATA_TYPE_BFLOAT16;
            uint32_t tensorRank = 2; const uint64_t globalDim[2] = {row_length, num_rows};
            const uint64_t globalStrides[1] = {row_length * sizeof(InputType)};
            MUtensorDescriptorInterleave interleave = MU_TENSOR_DESCRIPTOR_INTERLEAVE_NONE;
            uint64_t oobConstantFill = 0; NVTE_CHECK_MU(muTensorDescriptorEncode(
                &intensorDesc, tensorDataType, tensorRank, (void*)(input.dptr), globalDim,
                globalStrides, interleave, oobConstantFill));

            mtfp8_cast_rowise_aligned<
                InputType, OutputType, N_ELEMENTS_PER_THREAD_X, N_ELEMENTS_PER_THREAD_Y, BLOCK_SIZE_X,
                BLOCK_SIZE_Y, GROUP_SIZE><<<grid, block, 0, stream>>>(
                intensorDesc, reinterpret_cast<const InputType*>(input.dptr),
                reinterpret_cast<const CType*>(noop_tensor.dptr),
                reinterpret_cast<OutputType*>(output.dptr),
                reinterpret_cast<CType*>(scale_inv.dptr),
                row_length, num_rows);
        );
    );
  } else if (rowwise_option == FP8BlockwiseRowwiseOption::NONE
      && columnwise_option != FP8BlockwiseColumnwiseOption::NONE) {
    const size_t group_size = 128;

    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
        input.dtype, InputType,
        TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
            output_t.dtype, OutputType,

            constexpr int GROUP_SIZE = 128;  // TODO: extend other group_size
            NVTE_CHECK(group_size == GROUP_SIZE);
            constexpr int BLOCK_SIZE_Y = 32;
            constexpr int BLOCK_SIZE_X = 16;
            constexpr int N_ELEMENTS_PER_THREAD_X = std::min(GROUP_SIZE / BLOCK_SIZE_X, 8);
            constexpr int N_ELEMENTS_PER_THREAD_Y = 4;

            dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
            dim3 grid(DIVUP(row_length, group_size), DIVUP(num_rows, group_size));
            MUtensorDescriptor intensorDesc;
            MUtensorDescriptorDataType tensorDataType = MU_TENSOR_DESCRIPTOR_DATA_TYPE_BFLOAT16;
            uint32_t tensorRank = 2; const uint64_t globalDim[2] = {row_length, num_rows};
            const uint64_t globalStrides[1] = {row_length * sizeof(InputType)};
            MUtensorDescriptorInterleave interleave = MU_TENSOR_DESCRIPTOR_INTERLEAVE_NONE;
            uint64_t oobConstantFill = 0; NVTE_CHECK_MU(muTensorDescriptorEncode(
                &intensorDesc, tensorDataType, tensorRank, (void*)(input.dptr), globalDim,
                globalStrides, interleave, oobConstantFill));

            mtfp8_cast_columnwise_aligned<
                InputType, OutputType, N_ELEMENTS_PER_THREAD_X, N_ELEMENTS_PER_THREAD_Y, BLOCK_SIZE_X,
                BLOCK_SIZE_Y, GROUP_SIZE><<<grid, block, 0, stream>>>(
                intensorDesc, reinterpret_cast<const InputType*>(input.dptr),
                reinterpret_cast<const CType*>(noop_tensor.dptr),
                reinterpret_cast<OutputType*>(output_t.dptr),
                reinterpret_cast<CType*>(scale_inv_t.dptr), row_length, num_rows);
        );
    );

  } else if (rowwise_option != FP8BlockwiseRowwiseOption::NONE
      && columnwise_option != FP8BlockwiseColumnwiseOption::NONE) {

    const auto rowwise_sinv_n = scale_inv.shape[1];

    const size_t group_size = next_power_of_2(row_length / rowwise_sinv_n);

    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
        input.dtype, InputType,
        TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
            output.dtype, OutputType,

            constexpr int GROUP_SIZE = 128;  // TODO: extend other group_size
            NVTE_CHECK(group_size == GROUP_SIZE);

          if ((row_length % GROUP_SIZE) != 0) {
            constexpr int BLOCK_SIZE_Y = 16;
            constexpr int BLOCK_SIZE_X = 32;
            constexpr int N_ELEMENTS_PER_THREAD_X = std::min(GROUP_SIZE / BLOCK_SIZE_X, 8);
            constexpr int N_ELEMENTS_PER_THREAD_Y = 1;

            dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
            dim3 grid(DIVUP(row_length, group_size), DIVUP(num_rows, group_size));
            mtfp8_cast_transpose_general_kernel_column_unaligned<InputType, OutputType, N_ELEMENTS_PER_THREAD_X, N_ELEMENTS_PER_THREAD_Y, BLOCK_SIZE_X, BLOCK_SIZE_Y, GROUP_SIZE>
              <<<grid, block, 0, stream>>>(
                  reinterpret_cast<const InputType*>(input.dptr),
                  reinterpret_cast<const CType*>(noop_tensor.dptr),
                  reinterpret_cast<OutputType*>(output.dptr),
                  reinterpret_cast<OutputType*>(output_t.dptr),
                  reinterpret_cast<CType*>(scale_inv.dptr),
                  reinterpret_cast<CType*>(scale_inv_t.dptr),
                  row_length,
                  num_rows);
          } else {
            constexpr int BLOCK_SIZE_Y = 32;
            constexpr int BLOCK_SIZE_X = 16;
            constexpr int N_ELEMENTS_PER_THREAD_X = std::min(GROUP_SIZE / BLOCK_SIZE_X, 8);
            constexpr int N_ELEMENTS_PER_THREAD_Y = 4;

            dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
            dim3 grid(DIVUP(row_length, group_size), DIVUP(num_rows, group_size));
            MUtensorDescriptor intensorDesc;
            MUtensorDescriptorDataType tensorDataType = MU_TENSOR_DESCRIPTOR_DATA_TYPE_BFLOAT16;
            uint32_t tensorRank = 2; const uint64_t globalDim[2] = {row_length, num_rows};
            const uint64_t globalStrides[1] = {row_length * sizeof(InputType)};
            MUtensorDescriptorInterleave interleave = MU_TENSOR_DESCRIPTOR_INTERLEAVE_NONE;
            uint64_t oobConstantFill = 0; NVTE_CHECK_MU(muTensorDescriptorEncode(
                &intensorDesc, tensorDataType, tensorRank, (void*)(input.dptr), globalDim,
                globalStrides, interleave, oobConstantFill));

            mtfp8_cast_transpose_general_kernel_column_aligned<
                InputType, OutputType, N_ELEMENTS_PER_THREAD_X, N_ELEMENTS_PER_THREAD_Y, BLOCK_SIZE_X,
                BLOCK_SIZE_Y, GROUP_SIZE><<<grid, block, 0, stream>>>(
                intensorDesc, reinterpret_cast<const InputType*>(input.dptr),
                reinterpret_cast<const CType*>(noop_tensor.dptr),
                reinterpret_cast<OutputType*>(output.dptr),
                reinterpret_cast<OutputType*>(output_t.dptr),
                reinterpret_cast<CType*>(scale_inv.dptr),
                reinterpret_cast<CType*>(scale_inv_t.dptr), row_length, num_rows);
          }
        );
    );

  } else {
    NVTE_ERROR("Operation is not implemented.");
  }

  NVTE_CHECK_CUDA(musaGetLastError());
}

}  // namespace transformer_engine::detail
