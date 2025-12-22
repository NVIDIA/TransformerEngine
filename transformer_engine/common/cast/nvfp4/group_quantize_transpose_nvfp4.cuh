/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_transpose_nvfp4.cuh
 *  \brief CUDA kernels to cast to NVFP4 and transpose.
 */

#ifndef TRANSFORMER_ENGINE_GROUP_QUANTIZE_TRANSPOSE_NVFP4_CUH_
#define TRANSFORMER_ENGINE_GROUP_QUANTIZE_TRANSPOSE_NVFP4_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../util/math.h"
#include "../../util/ptx.cuh"
#include "../../utils.cuh"
#include "core_nvfp4.cuh"

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {

namespace group_quantize_transpose_kernel {

using namespace quantization_and_transposition_SF;
using namespace core;
using namespace ptx;

#if FP4_TYPE_SUPPORTED

constexpr int kMaxTensorsPerKernel = 64;  // Args must be <4 KB, expand 64 if needed
struct MultiAmaxCastTransposeFusionArgs {
  // Amax buffer for rowwise scaling
  void *rowwise_amax_list[kMaxTensorsPerKernel];
  // Rowwise scale pointers with 128x4 padding included for rowwise scaling
  void *output_rowwise_scale_inv_list[kMaxTensorsPerKernel];
  // (Unused for rowwise only scaling) Amax buffer for colwise scaling
  void *colwise_amax_list[kMaxTensorsPerKernel];
  // (Unused for rowwise only scaling) output data pointers for fp4 transposed output
  void *output_colwise_data_list[kMaxTensorsPerKernel];
  // (Unused for rowwise only scaling) output scale inverse pointers for each tensor
  void *output_colwise_scale_inv_list[kMaxTensorsPerKernel];
  // (Unused for rowwise only scaling) output scale stride for colwise scaling
  int output_colwise_scale_stride[kMaxTensorsPerKernel];
  // Prefix sum (with leading zero) of split_sections of each tensor of input
  int split_sections_range[kMaxTensorsPerKernel + 1];
  // Number of tensors (splits) being processed by kernel
  int num_tensors;
};

__device__ __forceinline__ int GetTensorId(MultiAmaxCastTransposeFusionArgs *kernel_args_ptr,
                                           int offset) {
  // check the kernel args and get the corresponding id
  int tensor_id = 0;
  while (kernel_args_ptr->split_sections_range[tensor_id + 1] <= offset) {
    ++tensor_id;
  }
  return tensor_id;
}

// Helper to get tensor id at offset, and also whether [offset_start, offset_end) crosses a split boundary.
__device__ __forceinline__ int GetTensorIdAndBoundary(
    MultiAmaxCastTransposeFusionArgs *kernel_args_ptr, int offset_start, int offset_end,
    bool *cross_boundary) {
  int tensor_id_start = 0;
  while (kernel_args_ptr->split_sections_range[tensor_id_start + 1] <= offset_start) {
    ++tensor_id_start;
  }
  int tensor_id_end = tensor_id_start;
  if (offset_end != offset_start) {
    if (kernel_args_ptr->split_sections_range[tensor_id_start + 1] < offset_end) {
      tensor_id_end = tensor_id_start + 1;
    }
  }
  if (cross_boundary) {
    *cross_boundary = (tensor_id_start != tensor_id_end);
  }
  return tensor_id_start;
}

__device__ __forceinline__ void UpdateEncodeDecodeScaleFP32(float *amax_ptr, float *s_enc_ptr,
                                                            float *s_dec_ptr) {
  float s_env_value =
      (amax_ptr == nullptr) ? 1.0f : compute_global_encode_scaling_factor_FP4(*amax_ptr);
  float s_dec_value = 1.0 / s_env_value;
  *s_enc_ptr = s_env_value;
  *s_dec_ptr = s_dec_value;
  return;
}

constexpr size_t SCALE_DIM = 16;  // NVFP4 block (x16 elts)

constexpr size_t CHUNK_DIM_Y = 128;
constexpr size_t CHUNK_DIM_X = 128;
constexpr size_t THREADS_NUM = 128;

constexpr size_t SCALES_PER_CHUNK_Y = CHUNK_DIM_Y / SCALE_DIM;
constexpr size_t SCALES_PER_CHUNK_X = CHUNK_DIM_X / SCALE_DIM;

constexpr size_t SCALES_PER_THREAD = 2 * (CHUNK_DIM_Y * CHUNK_DIM_X) / SCALE_DIM / THREADS_NUM;

// Each call generates 4x uint32_t random numbers
constexpr size_t RNG_GENS_PER_THREAD = SCALES_PER_THREAD / 4;

constexpr size_t TILE_DIM_Y = 32;
constexpr size_t TILE_DIM_X = 128;

// SHould this be SCALE_DIM or BLOCK_DIM? Both are 16, should work for both 1D and 2D
constexpr size_t SCALES_PER_TILE_Y = TILE_DIM_Y / SCALE_DIM;
constexpr size_t SCALES_PER_TILE_X = TILE_DIM_X / SCALE_DIM;  // 128 / 16 =  8

constexpr size_t TILES_Y = CHUNK_DIM_Y / TILE_DIM_Y;
constexpr size_t TILES_X = CHUNK_DIM_X / TILE_DIM_X;
constexpr size_t STAGES = TILES_Y * TILES_X;

constexpr size_t BUFFS_NUM = 2;
constexpr size_t BUFF_DIM_Y = TILE_DIM_Y;
constexpr size_t BUFF_DIM_X = TILE_DIM_X;
constexpr size_t BUFF_SIZE = BUFF_DIM_Y * BUFF_DIM_X;
constexpr size_t BUFF_SIZE_TOTAL = BUFF_SIZE * BUFFS_NUM;

// Input buffer (BF16)
constexpr size_t BUFF_IN_DIM_Y = BUFF_DIM_Y;
constexpr size_t BUFF_IN_DIM_X = BUFF_DIM_X;
constexpr size_t BUFF_IN_SIZE = BUFF_IN_DIM_Y * BUFF_IN_DIM_X;

// Output buffer (NVFP4)
constexpr size_t BUFF_OUT_DIM_Y = BUFF_DIM_Y;
constexpr size_t BUFF_OUT_DIM_X = (BUFF_DIM_X * 4) / 8;
constexpr size_t BUFF_OUT_SIZE = BUFF_OUT_DIM_Y * BUFF_OUT_DIM_X;

// Output transpose buffer (NVFP4)
constexpr size_t BUFF_OUT_T_DIM_Y = BUFF_DIM_X;
constexpr size_t BUFF_OUT_T_DIM_X = (BUFF_DIM_Y * 4) / 8;
constexpr size_t BUFF_OUT_T_SIZE = BUFF_OUT_T_DIM_Y * BUFF_OUT_T_DIM_X;

// Manual swizzling parameters to reduce SHMEM bank conflicts
constexpr size_t PACK_SIZE = 8;
constexpr size_t WAVES = SCALE_DIM / PACK_SIZE;

constexpr size_t SCALING_FACTORS_PER_TILE_X = TILE_DIM_X / SCALE_DIM;
constexpr size_t THREADS_X_ROWWISE = SCALING_FACTORS_PER_TILE_X;       // 128 / 16 = 8
constexpr size_t THREADS_Y_ROWWISE = THREADS_NUM / THREADS_X_ROWWISE;  // 128 / 8 = 16

constexpr size_t ITERATIONS_NORMAL = BUFF_DIM_Y / THREADS_Y_ROWWISE;  // 32/ 16 = 2
constexpr size_t ITERATIONS_TRANSPOSE = BUFF_IN_DIM_Y / SCALE_DIM;
constexpr size_t BUFF_OUT_IT_OFFSET = BUFF_OUT_T_DIM_X / ITERATIONS_TRANSPOSE;

static_assert(BUFF_DIM_Y >= SCALE_DIM &&
              "Number of buffer rows must be greater or equal to the size of the columwise "
              "scaling block\0");
static_assert(CHUNK_DIM_Y >= BUFF_DIM_Y);
static_assert(BUFF_DIM_Y >= THREADS_Y_ROWWISE &&
              "Number of buffer rows must be greater or equal to the number of rowwise "
              "processing threads in Y dimension\0");

// Number of 4-bit elements that span 32 banks (4-byte each) of shared memory
constexpr size_t TOTAL_BANKS_WIDTH = (32 * 4 * 8) / 4;  // 256

// Number of threads (rowwise scaling) that span 32 banks (4-byte banks) of shared memory
constexpr size_t THREADS_PER_BANK = TOTAL_BANKS_WIDTH / SCALE_DIM;  // 8 = 128 / 16

template <bool COMPUTE_ACTIVATIONS, typename ParamOP, float (*OP)(float, const ParamOP &),
          typename IType, bool USE_STOCHASTIC_ROUNDING, bool RETURN_TRANSPOSE>
__global__ void __launch_bounds__(THREADS_NUM)
    group_quantize_transpose_nvfp4_kernel(const __grid_constant__ CUtensorMap tensor_map_input,
                                          const __grid_constant__ CUtensorMap tensor_map_output,
                                          nvfp4_scale_t *const scales_ptr, const float *noop,
                                          const size_t rows, const size_t cols,
                                          const size_t scale_stride, const size_t *rng_state,
                                          MultiAmaxCastTransposeFusionArgs kernel_args) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  constexpr bool NO_ACTIVATIONS_NOT_FP32_INPUT =
      (!COMPUTE_ACTIVATIONS) && (!std::is_same_v<IType, float>);

  using IType2 = typename ptx::FPx2<IType>;

  if constexpr (!COMPUTE_ACTIVATIONS) {
    if (noop != nullptr && noop[0] == 1.0f) {
      return;
    }
  }

  const size_t rng_sequence =
      threadIdx.x + blockIdx.x * THREADS_NUM + blockIdx.y * gridDim.x * THREADS_NUM;
  const size_t rng_seed = rng_state != nullptr ? rng_state[0] : 0;
  const size_t rng_offset = rng_state != nullptr ? rng_state[1] : 0;
  transformer_engine::curanddx::detail::philox4x32_native_state<10> rng;
  rng.init(rng_seed, rng_sequence, rng_offset);
  uint4 random_uint4 = USE_STOCHASTIC_ROUNDING ? rng.generate4() : uint4{0, 0, 0, 0};
  // Index of the random number. It increments each time when used and resets to 0 if reaches 4x
  int rnd_idx = 0;

  constexpr bool IS_CACHED_ACT_OP = COMPUTE_ACTIVATIONS;

  const size_t block_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const size_t block_offset_X = blockIdx.x * CHUNK_DIM_X;

  // TODO(zhongbo): add back when transpose is supported
  // const size_t block_offset_Y_t = blockIdx.x * CHUNK_DIM_X;
  // const size_t block_offset_X_t = blockIdx.y * CHUNK_DIM_Y;

  const size_t chunk_rows = rows - block_offset_Y;

  const size_t scales_block_offset_Y_rowwise = blockIdx.y * CHUNK_DIM_Y;
  const size_t scales_block_offset_X_rowwise = blockIdx.x * SCALES_PER_CHUNK_X;
  // TODO(zhongbo): add back when transpose is supported
  // const size_t scales_block_offset_Y_t = blockIdx.x * CHUNK_DIM_X;
  // const size_t scales_block_offset_X_t = blockIdx.y * SCALES_PER_CHUNK_Y;

  const size_t tid_Y_rowwise = threadIdx.x / THREADS_X_ROWWISE;
  const size_t tid_X_rowwise = threadIdx.x % THREADS_X_ROWWISE;
  const size_t tid_X_colwise = threadIdx.x;
  const size_t tid_Y_t = tid_X_colwise;
  // const size_t tid_X_t = 0;

  const size_t thread_offset_Y_rowwise = tid_Y_rowwise;
  const size_t thread_offset_X_rowwise = tid_X_rowwise * SCALE_DIM;
  const size_t thread_offset_X_colwise = tid_X_colwise;

  const size_t row_base_rowwise = block_offset_Y + thread_offset_Y_rowwise;
  const size_t row_base_colwise = block_offset_Y;
  const size_t col_base_colwise = block_offset_X + thread_offset_X_colwise;

  const bool col_out_of_bounds_colwise = (col_base_colwise >= cols);

  const size_t scales_offset_Y_rowwise = scales_block_offset_Y_rowwise + tid_Y_rowwise;
  const size_t scales_offset_X_rowwise = scales_block_offset_X_rowwise + tid_X_rowwise;
  // TODO(zhongbo): add back when transpose is supported
  // const size_t scales_offset_Y_t = scales_block_offset_Y_t + tid_Y_t;
  // const size_t scales_offset_X_t = scales_block_offset_X_t;

  const size_t SFs_per_row = cols / SCALE_DIM;

  const bool rowwise_scale_is_within_bounds_X = scales_offset_X_rowwise < SFs_per_row;

  // TODO(zhongbo): add back when transpose is supported
  // const bool colwise_scale_is_within_bounds_Y = scales_offset_Y_t < cols;

  // Helps resolving bank conflicts in shmem
  const int thread_lane = threadIdx.x % THREADS_PER_WARP;
  const int bank_group = thread_lane / THREADS_PER_BANK;

  constexpr size_t buff_elems = BUFF_DIM_Y * BUFF_IN_DIM_X;
  constexpr size_t buff_elems_total = BUFFS_NUM * buff_elems;

  constexpr size_t buff_size_aligned_in =
      DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(IType), TMA_SHMEM_ALIGNMENT);
  constexpr size_t buff_size_aligned_out =
      DIVUP_TO_MULTIPLE((buff_elems_total * 4) / 8, TMA_SHMEM_ALIGNMENT);

  constexpr size_t in_mem = buff_size_aligned_in;

  constexpr size_t out_mem_rowwise_data = buff_size_aligned_out;
  constexpr size_t out_mem_colwise_data = buff_size_aligned_out;
  constexpr size_t out_mem_rowwise_scales = 0;

  extern __shared__ char dynamic_shmem[];
  uintptr_t base_shmem_ptr = reinterpret_cast<uintptr_t>(dynamic_shmem);
  // Manually align dynamic SHMEM per TMA requirements using padding
  // __align__(128) Does not guarantee the pointer to be aligned!
  uintptr_t dshmem = (base_shmem_ptr + TMA_SHMEM_ALIGNMENT - 1) &
                     ~(static_cast<uintptr_t>(TMA_SHMEM_ALIGNMENT - 1));

  // The destination shared memory buffer of a bulk tensor operation should be 16-byte aligned
  IType *in_sh = reinterpret_cast<IType *>(dshmem);
  fp4e2m1x2 *out_data_sh = reinterpret_cast<fp4e2m1x2 *>(dshmem + in_mem);
  fp4e2m1x2 *out_t_data_sh = reinterpret_cast<fp4e2m1x2 *>(dshmem + in_mem + out_mem_rowwise_data);

  nvfp4_scale_t *out_rowwise_scales_sh = reinterpret_cast<nvfp4_scale_t *>(
      dshmem + in_mem + out_mem_rowwise_data + out_mem_colwise_data);
  nvfp4_scale_t *out_colwise_scales_sh = reinterpret_cast<nvfp4_scale_t *>(
      dshmem + in_mem + out_mem_rowwise_data + out_mem_colwise_data + out_mem_rowwise_scales);
  IType *cached_act_sh = in_sh;  // in_sh is used as a cache buffer

  constexpr size_t shmem_buff_size = buff_size_aligned_in / BUFFS_NUM;

  const bool is_master_thread = (threadIdx.x == 0);

  // TODO (zhongbo): finish this
  float *amax_rowwise_ptr = nullptr;
  float *amax_colwise_ptr = nullptr;
  nvfp4_scale_t *split_rowwise_scale_ptr = nullptr;

  // suppose the amax is fixed for the current 128x128 tile (need 128 padding)
  bool need_update_tensor_id = true;
  int tensor_id = GetTensorIdAndBoundary(&kernel_args, block_offset_Y, block_offset_Y + CHUNK_DIM_Y,
                                         &need_update_tensor_id);
  size_t split_start = kernel_args.split_sections_range[tensor_id];
  size_t split_end = kernel_args.split_sections_range[tensor_id + 1];
  amax_rowwise_ptr = reinterpret_cast<float *>(kernel_args.rowwise_amax_list[tensor_id]);
  split_rowwise_scale_ptr =
      reinterpret_cast<nvfp4_scale_t *>(kernel_args.output_rowwise_scale_inv_list[tensor_id]);

  float S_enc_rowwise = 1.0f;
  float S_dec_rowwise = 1.0f;
  UpdateEncodeDecodeScaleFP32(amax_rowwise_ptr, &S_enc_rowwise, &S_dec_rowwise);

  // TODO (zhongbo): colwise scaling disabled for now because of transpose
  float S_enc_colwise = 1.0f;
  float S_dec_colwise = 1.0f;
  if (amax_colwise_ptr != nullptr) {
    UpdateEncodeDecodeScaleFP32(amax_colwise_ptr, &S_enc_colwise, &S_dec_colwise);
  } else {
    S_enc_colwise = S_enc_rowwise;
    S_dec_colwise = S_dec_rowwise;
  }

  float thread_amax = 0.0f;

// Initialize shared memory barrier with the number of threads participating in the barrier.
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[STAGES];

  initialize_barriers<STAGES, THREADS_NUM>(mbar, is_master_thread);

  copy_2d_to_shared(&in_sh[0], &tensor_map_input, block_offset_X, block_offset_Y, shmem_buff_size,
                    &mbar[0], is_master_thread);

#pragma unroll
  for (size_t stage = 0; stage < STAGES; ++stage) {
    const size_t buff = stage % BUFFS_NUM;
    const size_t next_stage = stage + 1;
    const size_t stage_offset_Y = stage * BUFF_DIM_Y;

    const size_t buff_offset_in = buff * BUFF_IN_SIZE;
    const size_t buff_offset_out = buff * BUFF_OUT_SIZE;
    const size_t buff_offset_out_t = buff * BUFF_OUT_T_SIZE;

    // for stages from 1 to STAGES - 1, we need to update the tensor id
    // skip updating tensor id if it's the last CTA, and some stages will be out of bounds
    if (need_update_tensor_id && stage > 0 && (block_offset_Y + stage_offset_Y < rows)) {
      int new_tensor_id = GetTensorId(&kernel_args, block_offset_Y + stage_offset_Y);
      if (new_tensor_id != tensor_id) {
        tensor_id = new_tensor_id;
        split_start = kernel_args.split_sections_range[tensor_id];
        split_end = kernel_args.split_sections_range[tensor_id + 1];
        amax_rowwise_ptr = reinterpret_cast<float *>(kernel_args.rowwise_amax_list[tensor_id]);
        UpdateEncodeDecodeScaleFP32(amax_rowwise_ptr, &S_enc_rowwise, &S_dec_rowwise);
        split_rowwise_scale_ptr =
            reinterpret_cast<nvfp4_scale_t *>(kernel_args.output_rowwise_scale_inv_list[tensor_id]);
        // TODO (zhongbo): colwise scaling disabled for now because of transpose
        // Skip fetching colwise amax pointer and scaling factor updates
      }
    }

    if (next_stage < STAGES) {
      // Wait for TMA transfer to have finished reading shared memory.
      // I.e. the buffer is ready to be written to
      ptx::cp_async_bulk_wait_group_read<1>();

      const size_t next_buff = next_stage % BUFFS_NUM;
      const size_t next_stage_offset_Y = next_stage * BUFF_DIM_Y;
      const size_t global_offset_Y = block_offset_Y + next_stage_offset_Y;
      const size_t global_offset_X = block_offset_X;
      const size_t next_buff_offset = next_buff * BUFF_IN_SIZE;

      copy_2d_to_shared(&in_sh[next_buff_offset], &tensor_map_input, global_offset_X,
                        global_offset_Y, shmem_buff_size, &mbar[next_stage], is_master_thread);
    }

    ptx::fence_proxy_async_shared_cta();

    // Wait for the data to have arrived
    ptx::mbarrier_wait_parity(&mbar[stage], 0);

    float block_amax = 0.0f;

    // COLWISE scaling
    if constexpr (RETURN_TRANSPOSE) {
#pragma unroll
      for (size_t it = 0; it < ITERATIONS_TRANSPOSE; ++it) {
        const size_t in_thread_offset_Y = 0 + it * SCALE_DIM;
        const size_t in_thread_offset_X = thread_offset_X_colwise;

        const size_t out_t_thread_offset_Y = thread_offset_X_colwise;
        const size_t out_t_thread_offset_X = 0 + it * BUFF_OUT_IT_OFFSET;

        const size_t shmem_offset_base_colwise_in =
            buff_offset_in + in_thread_offset_Y * BUFF_IN_DIM_X + in_thread_offset_X;
        const size_t shmem_offset_base_colwise_out_t =
            buff_offset_out_t + out_t_thread_offset_Y * BUFF_OUT_T_DIM_X + out_t_thread_offset_X;

        block_amax = 0.0f;
        float in_compute_colwise[SCALE_DIM];
        IType in_colwise_IType[SCALE_DIM];
        // 1. Read/Compute elements. Find NVFP4-block AMAX
        if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
          IType block_amax_f16 = static_cast<IType>(0.0f);
#pragma unroll
          for (int i = 0; i < SCALE_DIM; ++i) {
            const int shmem_offset_colwise = shmem_offset_base_colwise_in + i * BUFF_IN_DIM_X;
            in_colwise_IType[i] = in_sh[shmem_offset_colwise];
            block_amax_f16 = __hmax(block_amax_f16, __habs(in_colwise_IType[i]));
          }
          block_amax = static_cast<float>(block_amax_f16);
        } else {
#pragma unroll
          for (int i = 0; i < SCALE_DIM; ++i) {
            const int shmem_offset_colwise = shmem_offset_base_colwise_in + i * BUFF_IN_DIM_X;
            float elt = static_cast<float>(in_sh[shmem_offset_colwise]);
            if constexpr (COMPUTE_ACTIVATIONS) {
              elt = OP(elt, {});
            }
            // Numerical truncation: Downcast to IType (BF16/FP16), then upcast it back to FP32
            if constexpr (!std::is_same_v<IType, float>) {
              elt = static_cast<float>(static_cast<IType>(elt));
            }
            // Cache computed activations to avoid computing them again in the 2nd pass along another dimension
            if constexpr (IS_CACHED_ACT_OP) {
              cached_act_sh[shmem_offset_colwise] = static_cast<IType>(elt);
            }
            if constexpr (COMPUTE_ACTIVATIONS) {
              const bool row_out_of_bounds_colwise =
                  (row_base_colwise + stage_offset_Y + i >= rows);
              const bool out_of_bounds = (col_out_of_bounds_colwise || row_out_of_bounds_colwise);
              if (!out_of_bounds) {
                block_amax = fmaxf(block_amax, fabsf(elt));
              }
            } else {
              // If no activation, elt is 0 so we can safely do this
              block_amax = fmaxf(block_amax, fabsf(elt));
            }
            in_compute_colwise[i] = elt;
          }
        }
        // 2. Compute E4M3 scaling factor
        const nvfp4_scale_t S_dec_b_fp8 =
            compute_decoding_scaling_factor(block_amax, S_enc_colwise);

        // Store scaling factors through SHMEM
        const size_t scale_idx_sh =
            tid_Y_t * SCALES_PER_CHUNK_Y + stage * ITERATIONS_TRANSPOSE + it;
        out_colwise_scales_sh[scale_idx_sh] = S_dec_b_fp8;

        // Compute "correct" per-block encoding scaling factor
        constexpr float float_max = detail::TypeExtrema<float>::max;
        const float block_scale_inverse = fminf(
            1.0f / (static_cast<float>(S_dec_b_fp8) * S_dec_colwise), float_max);  // S_enc_b_fp8
        const float2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};

        // 3. Scale elements
        fp4e2m1x4 regs[SCALE_DIM / 4];

#pragma unroll
        for (int e = 0; e < SCALE_DIM / 4; ++e) {
          const uint32_t rbits = get_rbits(rng, random_uint4, rnd_idx);
          if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
            const uint64_t elts = *reinterpret_cast<uint64_t *>(&in_colwise_IType[4 * e]);
            regs[e] = ptx::mul_cvt_bf16_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
                elts, block_scale_inverse_2x, rbits);
          } else {
            const float2 in01 = *reinterpret_cast<float2 *>(&in_compute_colwise[4 * e]);
            const float2 in23 = *reinterpret_cast<float2 *>(&in_compute_colwise[4 * e + 2]);
            regs[e] = ptx::mul_cvt_fp32_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
                in01, in23, block_scale_inverse_2x, rbits);
          }
        }

        const int group = thread_lane / 16;
        uint32_t val[2];
        uint32_t *regs_4x = reinterpret_cast<uint32_t *>(regs);

        // Helps reducing bank conflicts
        switch (group) {
          case 0:
            val[0] = regs_4x[0];
            val[1] = regs_4x[1];
            break;
          case 1:
            val[0] = regs_4x[1];
            val[1] = regs_4x[0];

            break;
        }
        uint32_t *out_t_data_sh_as_uint32_t =
            reinterpret_cast<uint32_t *>(&out_t_data_sh[shmem_offset_base_colwise_out_t]);
        out_t_data_sh_as_uint32_t[group] = val[0];            // idx1 = (group + 0) % 2;
        out_t_data_sh_as_uint32_t[(group + 1) & 1] = val[1];  // idx2 = (group + 1) % 2;
      }
    }

    // ROWWISE scaling
    {
      const size_t stage_rowwise_scales_offset_Y = stage * BUFF_DIM_Y;
#pragma unroll
      for (size_t it = 0; it < ITERATIONS_NORMAL; ++it) {
        const size_t it_thread_offset_Y_rowwise = thread_offset_Y_rowwise + it * THREADS_Y_ROWWISE;

        const size_t shmem_offset_base_rowwise_in =
            buff_offset_in + it_thread_offset_Y_rowwise * BUFF_IN_DIM_X;
        const size_t shmem_offset_base_rowwise_out =
            buff_offset_out + it_thread_offset_Y_rowwise * BUFF_OUT_DIM_X;

        const size_t it_offset_Y = stage_offset_Y + it * THREADS_Y_ROWWISE;

        block_amax = 0.0f;
        float in_compute_rowwise[SCALE_DIM];
        Vec<IType, PACK_SIZE> in_cached[WAVES];

        // used as an IType container for BF16/FP16 --> NVFP4 CAST ONLY
        Vec<IType2, PACK_SIZE / 2> in_IType[WAVES];

        // 1. Read/Compute elements. Find NVFP4-block AMAX
        if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
          IType2 thread_amax_2x = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
#pragma unroll
          for (int w = 0; w < WAVES; ++w) {
            const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
            const size_t swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
            const size_t shmem_offset_rowwise = shmem_offset_base_rowwise_in + swizzled_thread_idx;
            // Load elements
            in_IType[w].load_from(&in_sh[shmem_offset_rowwise]);
#pragma unroll
            for (int e = 0; e < PACK_SIZE / 2; ++e) {
              ptx::abs_max_2x(thread_amax_2x, thread_amax_2x, in_IType[w].data.elt[e]);
            }
          }
          block_amax =
              static_cast<float>(__hmax(__habs(thread_amax_2x.x), __habs(thread_amax_2x.y)));
        } else if constexpr (IS_CACHED_ACT_OP) {
          // ensures that all writes to cache made in the section above are visible to all threads
          __syncthreads();
          IType2 thread_amax_2x = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
#pragma unroll
          for (int w = 0; w < WAVES; ++w) {
            const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
            const size_t swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
            const size_t shmem_offset_rowwise = shmem_offset_base_rowwise_in + swizzled_thread_idx;

            const bool row_out_of_bounds_rowwise = (row_base_rowwise + it_offset_Y >= rows);
            const bool swizzled_col_out_of_bounds = (block_offset_X + swizzled_thread_idx >= cols);
            const bool out_of_bounds = (row_out_of_bounds_rowwise || swizzled_col_out_of_bounds);

            // Load cached elements
            in_cached[w].load_from(&cached_act_sh[shmem_offset_rowwise]);
            // Since TMA requirement for the data alignment is 16B (i.e. cols % 8 == 0, in case of BF16 elements)
            // only single check (w.r.t. column direction) is sufficient to be sure the entire wave is inside the boundaries
            if (!out_of_bounds) {
              if constexpr (std::is_same_v<IType, float>) {
#pragma unroll
                for (int e = 0; e < PACK_SIZE; ++e) {
                  block_amax = fmaxf(block_amax, fabsf(in_cached[w].data.elt[e]));
                }
              } else {
#pragma unroll
                for (int e = 0; e < PACK_SIZE; e += 2) {
                  const IType2 in_cached_2x = {in_cached[w].data.elt[e],
                                               in_cached[w].data.elt[e + 1]};
                  ptx::abs_max_2x(thread_amax_2x, thread_amax_2x, in_cached_2x);
                }
              }
            }
          }
          if constexpr (!std::is_same_v<IType, float>) {
            block_amax =
                static_cast<float>(__hmax(__habs(thread_amax_2x.x), __habs(thread_amax_2x.y)));
          }
        } else {
#pragma unroll
          for (int w = 0; w < WAVES; ++w) {
            const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
            const size_t swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
            const size_t shmem_offset_rowwise = shmem_offset_base_rowwise_in + swizzled_thread_idx;

            Vec<IType, PACK_SIZE> in;
            Vec<IType, PACK_SIZE> act_in;

            in.load_from(&in_sh[shmem_offset_rowwise]);
#pragma unroll
            for (int e = 0; e < PACK_SIZE; ++e) {
              const size_t j = w * PACK_SIZE + e;
              // Compute element
              float elt = static_cast<float>(in.data.elt[e]);
              if constexpr (COMPUTE_ACTIVATIONS) {
                elt = OP(elt, {});
              }
              // Numerical truncation: Downcast to IType (BF16/FP16), then upcast it back to FP32
              if constexpr (!std::is_same_v<IType, float>) {
                elt = static_cast<float>(static_cast<IType>(elt));
              }
              if constexpr (COMPUTE_ACTIVATIONS) {
                const bool row_out_of_bounds_rowwise = (row_base_rowwise + it_offset_Y >= rows);
                const bool swizzled_col_out_of_bounds =
                    (block_offset_X + swizzled_thread_idx >= cols);
                const bool out_of_bounds =
                    (row_out_of_bounds_rowwise || swizzled_col_out_of_bounds);
                if (!out_of_bounds) {
                  block_amax = fmaxf(block_amax, fabsf(elt));
                }
              } else {
                // If no activation, elt is 0 so we can safely do this
                block_amax = fmaxf(block_amax, fabsf(elt));
              }
              in_compute_rowwise[j] = elt;
            }
          }
        }

        // 2. Compute E4M3 scaling factor
        const nvfp4_scale_t S_dec_b_fp8 =
            compute_decoding_scaling_factor(block_amax, S_enc_rowwise);

        // Check boundaries
        const size_t scales_offset_Y =
            scales_offset_Y_rowwise + stage * BUFF_DIM_Y + it * THREADS_Y_ROWWISE;
        const size_t scales_offset_X = scales_offset_X_rowwise;

        const bool rowwise_scale_is_within_bounds_Y =
            (stage_rowwise_scales_offset_Y + it * THREADS_Y_ROWWISE) < chunk_rows;

        // TODO(zhongbo): depending on input padding multiple (whether 128 or 64), use either scale_ptr or split_rowwise_scale_ptr
        // const size_t scale_idx_global = scales_offset_Y * scale_stride + scales_offset_X;
        // if (rowwise_scale_is_within_bounds_X && rowwise_scale_is_within_bounds_Y) {
        //   scales_ptr[scale_idx_global] = S_dec_b_fp8;
        // }

        // Map to local split coordinates
        const size_t split_rows = split_end - split_start;
        const size_t local_scale_row = scales_offset_Y - split_start;

        // Local bounds: 0 <= local_scale_row < split_rows
        const bool local_rowwise_scale_is_within_bounds_Y = local_scale_row < split_rows;

        // Index inside this split’s scale buffer
        const size_t scale_idx_local = local_scale_row * scale_stride + scales_offset_X;

        if (rowwise_scale_is_within_bounds_X && rowwise_scale_is_within_bounds_Y &&
            local_rowwise_scale_is_within_bounds_Y) {
          split_rowwise_scale_ptr[scale_idx_local] = S_dec_b_fp8;
        }

        // Compute "correct" per-block encoding scaling factor
        constexpr float float_max = detail::TypeExtrema<float>::max;
        const float block_scale_inverse = fminf(
            1.0f / (static_cast<float>(S_dec_b_fp8) * S_dec_rowwise), float_max);  // S_enc_b_fp8
        const float2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};

// 3. Scale elements
#pragma unroll
        for (int w = 0; w < WAVES; ++w) {
          Vec<fp4e2m1x4, PACK_SIZE / 4> out;
#pragma unroll
          for (int e = 0; e < PACK_SIZE / 4; ++e) {
            const uint32_t rbits = get_rbits(rng, random_uint4, rnd_idx);
            IType2 in01;
            IType2 in23;
            if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
              const uint64_t elts = *reinterpret_cast<uint64_t *>(&in_IType[w].data.elt[2 * e]);
              out.data.elt[e] = ptx::mul_cvt_bf16_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
                  elts, block_scale_inverse_2x, rbits);
            } else if constexpr (IS_CACHED_ACT_OP) {
              const uint64_t elts = *reinterpret_cast<uint64_t *>(&in_cached[w].data.elt[4 * e]);
              out.data.elt[e] = ptx::mul_cvt_bf16_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
                  elts, block_scale_inverse_2x, rbits);
            } else {
              const int j = w * PACK_SIZE + 4 * e;
              const float2 in01 = make_float2(in_compute_rowwise[j], in_compute_rowwise[j + 1]);
              const float2 in23 = make_float2(in_compute_rowwise[j + 2], in_compute_rowwise[j + 3]);
              out.data.elt[e] = ptx::mul_cvt_fp32_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
                  in01, in23, block_scale_inverse_2x, rbits);
            }
          }
          const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
          const size_t swizzled_idx = swizzled_group_idx + thread_offset_X_rowwise;
          const size_t shmem_offset_rowwise = shmem_offset_base_rowwise_out + swizzled_idx / 2;
          out.store_to(&out_data_sh[shmem_offset_rowwise]);
        }
      }
    }

    __builtin_assume(thread_amax >= 0);
    thread_amax = fmaxf(thread_amax, block_amax);

    // Wait for shared memory writes to be visible to TMA engine.
    ptx::fence_proxy_async_shared_cta();
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.

    // Initiate TMA transfer to copy shared memory to global memory
    if (is_master_thread) {
      const size_t global_offset_Y = block_offset_Y + stage_offset_Y;
      const size_t global_offset_X = block_offset_X;

      // TODO(zhongbo): add back when transpose is supported
      // const size_t global_offset_Y_t = block_offset_Y_t;
      // const size_t global_offset_X_t = block_offset_X_t + stage_offset_Y;

      ptx::cp_async_bulk_tensor_2d_shared_to_global(
          reinterpret_cast<const uint64_t *>(&tensor_map_output), global_offset_X, global_offset_Y,
          reinterpret_cast<uint64_t *>(&out_data_sh[buff_offset_out]));

      // TODO(zhongbo): add back when transpose is supported
      // if constexpr (RETURN_TRANSPOSE) {
      //   ptx::cp_async_bulk_tensor_2d_shared_to_global(
      //       reinterpret_cast<const uint64_t *>(&tensor_map_output_t), global_offset_X_t,
      //       global_offset_Y_t, reinterpret_cast<uint64_t *>(&out_t_data_sh[buff_offset_out_t]));
      // }

      // Create a "bulk async-group" out of the previous bulk copy operation.
      ptx::cp_async_bulk_commit_group();
    }
  }  // end of stages

  // TODO(zhongbo): add back when transpose is supported
  // Vectorized store scaling factors through SHMEM
  // if (RETURN_TRANSPOSE && colwise_scale_is_within_bounds_Y) {
  //   using ScalesVec = Vec<nvfp4_scale_t, SCALES_PER_CHUNK_Y>;
  //   const size_t scale_idx_sh = tid_Y_t * SCALES_PER_CHUNK_Y;
  //   ScalesVec &scales_vec = *reinterpret_cast<ScalesVec *>(&out_colwise_scales_sh[scale_idx_sh]);
  //   const size_t scale_idx_global = scales_offset_Y_t * scale_stride_t + scales_offset_X_t;
  //   const size_t count =  // number of scales in Y dimension of this chunk
  //       (chunk_rows >= CHUNK_DIM_Y) ? SCALES_PER_CHUNK_Y : (chunk_rows / SCALE_DIM);
  //   nvfp4_scale_t *dst = &scales_t_ptr[scale_idx_global];
  //   constexpr size_t vec_bytes = SCALES_PER_CHUNK_Y * sizeof(nvfp4_scale_t);
  //   if (count == SCALES_PER_CHUNK_Y && (reinterpret_cast<uintptr_t>(dst) % vec_bytes == 0)) {
  //     // Fast path: vectorized store when destination is properly aligned
  //     scales_vec.store_to(dst);
  //   } else {
  //     // Safe path: element-wise store for tails or unaligned destinations
  //     scales_vec.store_to_elts(dst, 0, count);
  //   }
  // }

  destroy_barriers<STAGES>(mbar, is_master_thread);
#else
  NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

#endif  // FP4_TYPE_SUPPORTED
}  // namespace group_quantize_transpose_kernel

template <bool use_2d_quantization>
void group_quantize_transpose(const Tensor &input, const Tensor *noop,
                              std::vector<Tensor *> &output_list, const size_t *split_sections,
                              size_t num_tensors, const QuantizationConfig *quant_config,
                              cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  using namespace group_quantize_transpose_kernel;
  using namespace ptx;
  bool use_stochastic_rounding = quant_config ? quant_config->stochastic_rounding : false;

  NVTE_CHECK(num_tensors == output_list.size(),
             "Number of output tensors should match number of tensors.");
  NVTE_CHECK(num_tensors <= kMaxTensorsPerKernel,
             "Number of tensors should be less than or equal to ", kMaxTensorsPerKernel);

  Tensor *output = nullptr;
  // loop over the list to find the first non-empty tensor
  for (size_t i = 0; i < num_tensors; ++i) {
    if (output_list[i]->has_data()) {
      output = output_list[i];
      break;
    }
  }
  NVTE_CHECK(output != nullptr, "No output tensor found.");
  // also check that the output has not null data pointer
  NVTE_CHECK(output->data.dptr != nullptr, "Output data pointer is null.");

  // If transposed output is allocated, return the transposed data. Otherwise, it's not necesary to
  // return the transposed data.
  bool return_transpose = output->has_columnwise_data();
  // forbid return transpose for now because group quantize transpose is not supported yet
  NVTE_CHECK(!return_transpose, "Return transpose is not supported for group quantize transpose.");

  // output_List is contiguous in memory, so take the first tensor as the contiguous output
  auto output_contiguous = output->data;

  constexpr bool COMPUTE_ACTIVATIONS = false;
  using ParamOP = Empty;
  constexpr float (*OP)(float, const ParamOP &) = nullptr;

  checkCuDriverContext(stream);
  CheckNoopTensor(*noop, "cast_noop");
  CheckInputTensor(input, "input");

  NVTE_CHECK(input.has_data(), "Cannot quantize tensor without rowwise data.");

  const size_t rows = input.flat_first_dim();
  const size_t cols = input.flat_last_dim();

  NVTE_CHECK(rows % 32 == 0,
             "Number of tensor rows must be a multiple of 32");  // 16B alignment for TMA
  NVTE_CHECK(cols % 32 == 0,
             "Number of tensor cols must be a multiple of 32");  // 16B alignment for TMA

  // process the output list and produce the multi-tensor args for grouped kernel
  MultiAmaxCastTransposeFusionArgs kernel_args;
  kernel_args.num_tensors = 0;
  kernel_args.split_sections_range[0] = 0;
  for (size_t i = 0; i < num_tensors; ++i) {
    if (split_sections[i] == 0) {
      continue;
    }
    kernel_args.rowwise_amax_list[kernel_args.num_tensors] =
        reinterpret_cast<void *>(output_list[i]->amax.dptr);
    kernel_args.output_rowwise_scale_inv_list[kernel_args.num_tensors] =
        reinterpret_cast<void *>(output_list[i]->scale_inv.dptr);
    // kernel_args.split_sections[kernel_args.num_tensors] = split_sections[i];
    kernel_args.split_sections_range[kernel_args.num_tensors + 1] =
        kernel_args.split_sections_range[kernel_args.num_tensors] + split_sections[i];
    // check overflow
    NVTE_CHECK(kernel_args.split_sections_range[kernel_args.num_tensors + 1] >= 0,
               "split_sections_range overflow the int32_t");
    kernel_args.num_tensors++;
  }

  const size_t blocks_Y = DIVUP(rows, CHUNK_DIM_Y);
  const size_t blocks_X = DIVUP(cols, CHUNK_DIM_X);
  const dim3 grid(blocks_X, blocks_Y);
  const size_t block_size = THREADS_NUM;

  // Note (zhongbo): for group quantize of [x1, x2, ..., xn]
  // for the rowwise sclaing, scaling factor stride is shared between all tensors
  // for the colwise scaling, scaling factor stride is different for each tensor because of transpose
  // since transpose puts token dimension splits in the last dimension of the tensor
  const size_t scale_stride = output->scale_inv.shape[1];
  // const size_t scale_stride_transpose =
  //     return_transpose ? output->columnwise_scale_inv.shape[1] : 0;

  nvfp4_scale_t *const scales_ptr = reinterpret_cast<nvfp4_scale_t *>(output->scale_inv.dptr);

  const float *noop_ptr = reinterpret_cast<const float *>(noop->data.dptr);

  const NVTETensor rng_state_tensor = (quant_config != nullptr) ? quant_config->rng_state : nullptr;
  const size_t *rng_state = nullptr;
  if (rng_state_tensor != nullptr) {
    Tensor &rng_state_te_tensor = *convertNVTETensor(rng_state_tensor);
    NVTE_CHECK(rng_state_te_tensor.dtype() == DType::kInt64,
               "RNG state should contain 2 64-bit values.");
    NVTE_CHECK(rng_state_te_tensor.data.shape == std::vector<size_t>{2},
               "Shape of the RNG state should be [2], but got ", rng_state_te_tensor.data.shape);
    rng_state = reinterpret_cast<const size_t *>(rng_state_te_tensor.data.dptr);
  }

  using IType = bf16;

  alignas(64) CUtensorMap tensor_map_input{};
  alignas(64) CUtensorMap tensor_map_output{};
  // alignas(64) CUtensorMap tensor_map_output_transpose{};

  create_2D_tensor_map(tensor_map_input, input.data, rows, cols, BUFF_DIM_Y, BUFF_DIM_X, cols, 0,
                       sizeof(IType) * 8);

  create_2D_tensor_map(tensor_map_output, output_contiguous, rows, cols, BUFF_DIM_Y, BUFF_DIM_X,
                       cols, 0, 4);
  // if (return_transpose) {
  //   create_2D_tensor_map(tensor_map_output_transpose, output->columnwise_data, cols, rows,
  //                        BUFF_DIM_X, BUFF_DIM_Y, rows, 0, 4);
  // }
  constexpr size_t buff_elems = BUFF_DIM_Y * BUFF_DIM_X;
  constexpr size_t buff_elems_total = BUFFS_NUM * buff_elems;
  constexpr size_t buff_size_aligned_in =
      DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(IType), TMA_SHMEM_ALIGNMENT);
  constexpr size_t buff_size_aligned_out =
      DIVUP_TO_MULTIPLE((buff_elems_total * 4) / 8, TMA_SHMEM_ALIGNMENT);
  constexpr size_t buff_size_scales = (CHUNK_DIM_Y * CHUNK_DIM_X) / 16 * sizeof(nvfp4_scale_t);

  constexpr size_t in_mem = buff_size_aligned_in;

  constexpr size_t out_data_mem = buff_size_aligned_out;
  constexpr size_t out_data_transpose_mem = buff_size_aligned_out;
  constexpr size_t out_scales_transpose_mem = buff_size_scales;

  constexpr size_t out_mem = out_data_mem + out_data_transpose_mem;

  constexpr size_t dshmem_size = in_mem + out_mem + out_scales_transpose_mem + TMA_SHMEM_ALIGNMENT;

  TRANSFORMER_ENGINE_SWITCH_CONDITION(
      use_stochastic_rounding, USE_STOCHASTIC_ROUNDING,

      TRANSFORMER_ENGINE_SWITCH_CONDITION(return_transpose, RETURN_TRANSPOSE, {
        auto kernel =
            group_quantize_transpose_nvfp4_kernel<COMPUTE_ACTIVATIONS, ParamOP, OP, IType,
                                                  USE_STOCHASTIC_ROUNDING, RETURN_TRANSPOSE>;

        if constexpr (use_2d_quantization) {
          NVTE_ERROR("2D quantization is not supported for group quantize transpose.");
        }

        NVTE_CHECK_CUDA(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem_size));
        kernel<<<grid, block_size, dshmem_size, stream>>>(tensor_map_input, tensor_map_output,
                                                          scales_ptr, noop_ptr, rows, cols,
                                                          scale_stride, rng_state, kernel_args);
        NVTE_CHECK_CUDA(cudaGetLastError());
      }););
#else
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif  // FP4_TYPE_SUPPORTED
}

}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_QUANTIZE_TRANSPOSE_NVFP4_CUH_
