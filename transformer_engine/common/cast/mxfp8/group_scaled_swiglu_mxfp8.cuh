/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file group_scaled_swiglu_mxfp8.cuh
 *  \brief Grouped scaled SwiGLU fused with columnwise MXFP8 quantization.
 *
 *  MoE backward recompute of the FC2 input, without re-running the FC1 GEMM:
 *
 *      input  : FC1 output, grouped, logical shape [T, 2F] (last dim = [act|gate]).
 *      prob   : per-token router weight, [T], in the input dtype.
 *      output : columnwise-MXFP8 of  (silu(act) * gate) * prob, grouped [T, F].
 *
 *  "SwiGLU" is TE's gated convention (same as gated_mxfp8.cuh): the first half of
 *  the last dim is the activation input, the second half is the gate, i.e.
 *  swiglu(x) = silu(x[:, :F]) * x[:, F:]. "scaled" is the per-token prob factor,
 *  applied after the activation.
 *
 *  Instantiating with ParamOP = ClampedSwiGLUParam and OP = clamped_silu gives the
 *  clamped variant, with the semantics of gated_mxfp8.cuh's forward path.
 */

#ifndef TRANSFORMER_ENGINE_GROUP_SCALED_SWIGLU_MXFP8_CUH_
#define TRANSFORMER_ENGINE_GROUP_SCALED_SWIGLU_MXFP8_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../util/cuda_runtime.h"
#include "../../util/math.h"
#include "../../util/ptx.cuh"
#include "../../utils.cuh"
#include "../core/common.cuh"
#include "swizzle.cuh"

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {
namespace group_scaled_swiglu_kernel {

using namespace dispatch::common;

// Reuse the same tiling as group_quantize_mxfp8 so the scheduler/TMA math match.
struct TunableConfig {
  static constexpr uint CHUNK_DIM_Y = 128;
  static constexpr uint CHUNK_DIM_X = 128;
  static constexpr uint THREADS_PER_CHUNK = 128;
  static constexpr uint STATIC_PERSISTENT_BLOCKS_PER_SM = 24;
};

constexpr size_t SCALE_DIM_Y = 32;
constexpr size_t SCALE_DIM_X = 32;

constexpr uint PREFETCH_STAGES = 1;
constexpr uint BUFFS_NUM = PREFETCH_STAGES + 1;

// Holding both the act and the gate input slice already costs this kernel 1.7x the
// shared memory of the plain quantize kernel, which is what caps resident blocks per
// SM. Single-buffering the (1 byte per element) output slice buys one more block back
// at the cost of overlapping the TMA store with the next stage's compute.
constexpr uint OUT_BUFFS_NUM = 1;
static_assert(OUT_BUFFS_NUM >= 1 && OUT_BUFFS_NUM <= BUFFS_NUM);

constexpr uint CHUNK_DIM_Y = TunableConfig::CHUNK_DIM_Y;
constexpr uint CHUNK_DIM_X = TunableConfig::CHUNK_DIM_X;
constexpr uint THREADS_PER_CHUNK = TunableConfig::THREADS_PER_CHUNK;

constexpr size_t ELTS_PER_CHUNK = CHUNK_DIM_Y * CHUNK_DIM_X;

constexpr uint THREADS_X = CHUNK_DIM_X / SCALE_DIM_X;
constexpr uint THREADS_Y = THREADS_PER_CHUNK / THREADS_X;

constexpr uint BUFF_DIM_Y = THREADS_Y;
constexpr uint BUFF_DIM_X = CHUNK_DIM_X;
constexpr uint BUFF_DIM = BUFF_DIM_Y * BUFF_DIM_X;
static_assert(BUFF_DIM_Y == 32);

constexpr uint STAGES = CHUNK_DIM_Y / BUFF_DIM_Y;
static_assert(STAGES >= 1);
static_assert(CHUNK_DIM_Y % BUFF_DIM_Y == 0);
static_assert(CHUNK_DIM_Y % SCALE_DIM_Y == 0);
static_assert(CHUNK_DIM_X % SCALE_DIM_X == 0);

// silu(x) = x * sigmoid(x) = h * (1 + tanh(h)), h = x/2. Deliberately approximate: one
// MUFU per element, against two for the ex2 form and a full expf/division chain for the
// generic path. The error cannot reach an MXFP8 mantissa of at most 3 bits; what it can
// do is move an e8m0 block exponent, which the C++ gtest bounds.
__device__ __forceinline__ float silu_approx(const float x) {
  const float h = 0.5f * x;
  float tanh_h;
  asm("tanh.approx.f32 %0, %1;" : "=f"(tanh_h) : "f"(h));
  return fmaf(h, tanh_h, h);
}

// clamped_silu's activation half, x * sigmoid(alpha * x), is the same identity with the
// tanh argument scaled. Kept separate from silu_approx rather than passing alpha = 1.0f
// so the plain path cannot regress on whether nvcc folds the multiply.
__device__ __forceinline__ float clamped_silu_approx(const float x, const float alpha) {
  const float h = 0.5f * x;
  float tanh_ah;
  asm("tanh.approx.f32 %0, %1;" : "=f"(tanh_ah) : "f"(alpha * h));
  return fmaf(h, tanh_ah, h);
}

// Columnwise scaled SwiGLU + MXFP8 quantization of one 32-row buffer slice.
// Each thread owns one column j and reduces amax over the BUFF_DIM_Y rows, then
// writes the e8m0 block scale and the scaled FP8 column.
template <typename ParamOP, float (*OP)(float, const ParamOP &), typename IType, typename OType,
          bool WITH_GEMM_SWIZZLED_SCALES>
__device__ __forceinline__ void process_colwise_gated_stage(
    const size_t buff, const size_t out_buff, const int stage, const size_t tid_X_colwise,
    const size_t scales_offset_Y_colwise, const size_t scales_offset_X_colwise,
    const size_t scale_stride_colwise, const size_t tensor_base_for_scales, const size_t rows,
    const size_t cols, const ParamOP &p, const float *const sProb, IType *sInAct_ptr,
    IType *sInGate_ptr, OType *sOutColwise_ptr, e8m0_t *scales_colwise) {
  using IType3D = IType[BUFFS_NUM][BUFF_DIM_Y][BUFF_DIM_X];
  using OType3D = OType[OUT_BUFFS_NUM][BUFF_DIM_Y][BUFF_DIM_X];

  const auto &sInAct = *reinterpret_cast<const IType3D *>(sInAct_ptr);
  const auto &sInGate = *reinterpret_cast<const IType3D *>(sInGate_ptr);
  auto &sOutColwise = *reinterpret_cast<OType3D *>(sOutColwise_ptr);

  const size_t global_scales_offset_Y = scales_offset_Y_colwise + stage;
  const size_t global_scales_offset_X = scales_offset_X_colwise;
  const bool colwise_scale_is_within_bounds = global_scales_offset_X < cols;

  size_t scale_idx = 0;
  if constexpr (WITH_GEMM_SWIZZLED_SCALES) {
    // The FC2 wgrad GEMM consumes this operand transposed, so its scale matrix
    // is the [cols, rows/32] transpose of the compact one, tiled 128x4. Each
    // expert gets its own swizzled block, sized exactly like its compact block
    // because per-expert row counts are 128-aligned.
    const size_t tensor_base_row = tensor_base_for_scales / cols;
    const size_t tensor_scales_offset_Y_base = tensor_base_row / SCALE_DIM_Y;
    const size_t tensor_scales_base = tensor_base_row * scale_stride_colwise / SCALE_DIM_Y;
    const size_t local_scales_offset_Y = global_scales_offset_Y - tensor_scales_offset_Y_base;
    scale_idx = tensor_scales_base +
                swizzle::gemm_swizzled_scale_idx(
                    global_scales_offset_X, local_scales_offset_Y,
                    DIVUP(rows, static_cast<size_t>(scale_tensor_alignment_Y_rowwise)));
  } else {
    scale_idx = global_scales_offset_Y * scale_stride_colwise + global_scales_offset_X;
  }

  const size_t j = tid_X_colwise;

  float rInCompute[BUFF_DIM_Y];
  float thread_amax = 0.0f;
#pragma unroll
  for (int i = 0; i < BUFF_DIM_Y; ++i) {
    const float act_elt = static_cast<float>(sInAct[buff][i][j]);
    float gate_elt = static_cast<float>(sInGate[buff][i][j]);
    // Staged in shared memory for the whole chunk by the caller: every thread needs
    // all rows, so reading it from global here would issue one broadcast load per
    // row per warp on the critical path.
    const float prob = sProb[stage * BUFF_DIM_Y + i];

    // Gate clamped on both sides then offset, activation clamped from above only --
    // the asymmetry is gated_mxfp8.cuh's forward path.
    //
    // The OP comparison must nest inside the ParamOP branch: OP's type carries ParamOP,
    // so `OP == &silu<fp32, fp32>` compares unrelated function pointer types once
    // ParamOP is ClampedSwiGLUParam, and an if-constexpr condition must be well formed
    // even where its branch is discarded.
    float act_x;
    if constexpr (std::is_same_v<ParamOP, ClampedSwiGLUParam>) {
      gate_elt = fminf(fmaxf(-p.limit, gate_elt), p.limit) + p.glu_linear_offset;
      if constexpr (OP == &clamped_silu<fp32, fp32>) {
        act_x = clamped_silu_approx(fminf(act_elt, p.limit), p.alpha);
      } else {
        act_x = OP(act_elt, p);
      }
    } else {
      if constexpr (OP == &silu<fp32, fp32>) {
        act_x = silu_approx(act_elt);
      } else {
        act_x = OP(act_elt, p);
      }
    }

    float elt = act_x * gate_elt * prob;

    // Match round-trip precision of the plain quantize path (cast through IType).
    if constexpr (!std::is_same_v<IType, float>) {
      elt = static_cast<float>(static_cast<IType>(elt));
    }
    thread_amax = fmaxf(thread_amax, fabsf(elt));
    rInCompute[i] = elt;
  }

  const e8m0_t biased_exponent =
      ptx::float_to_e8m0(thread_amax * Quantized_Limits<OType>::max_norm_rcp);
  scales_colwise[scale_idx] =
      colwise_scale_is_within_bounds ? biased_exponent : static_cast<e8m0_t>(0);

  const float block_scale_inverse = ptx::exp2f_rcp<float>(biased_exponent);
#pragma unroll
  for (int i = 0; i < SCALE_DIM_Y; ++i) {
    sOutColwise[out_buff][i][j] = static_cast<OType>(rInCompute[i] * block_scale_inverse);
  }
}

template <typename ParamOP, float (*OP)(float, const ParamOP &), typename IType, typename OType,
          bool WITH_GEMM_SWIZZLED_SCALES, ShapeRepresentation SHAPE_REP>
__global__ void __launch_bounds__(THREADS_PER_CHUNK) group_scaled_swiglu_mxfp8_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input_act_static,
    const __grid_constant__ CUtensorMap tensor_map_input_gate_static,
    const __grid_constant__ CUtensorMap tensor_map_output_colwise_static, const size_t num_tensors,
    const size_t first_logical_dim, const size_t last_logical_dim,
    const int64_t *const __restrict__ offsets_ptr, const int64_t *const __restrict__ first_dims_ptr,
    const int64_t *const __restrict__ last_dims_ptr, const IType *const __restrict__ prob_ptr,
    e8m0_t *const __restrict__ scales_colwise_ptr, const float *__restrict__ noop,
    const size_t work_blocks_X, const size_t work_blocks_Y, const ParamOP p) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  constexpr ShapeRepresentation shape_rep = SHAPE_REP;
  constexpr bool is_single_tensor = (shape_rep == SAME_BOTH_DIMS || shape_rep == VARYING_FIRST_DIM);
  // The shape-rep switch instantiates this kernel for all four reps, but only the
  // single-tensor ones are ever dispatched. Compile the others as no-ops.
  if constexpr (!is_single_tensor) {
    return;
  } else {
    const bool leading_thread = (threadIdx.x == 0);

    const size_t tid_X_colwise = threadIdx.x;

    constexpr size_t buff_elems = BUFF_DIM_Y * BUFF_DIM_X;
    constexpr size_t buff_elems_total = BUFFS_NUM * buff_elems;
    constexpr size_t buff_size_aligned_in =
        DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr size_t out_buff_elems_total = OUT_BUFFS_NUM * buff_elems;
    constexpr size_t buff_size_aligned_out =
        DIVUP_TO_MULTIPLE(out_buff_elems_total * sizeof(OType), TMA_SHMEM_ALIGNMENT);
    constexpr size_t prob_buff_size =
        DIVUP_TO_MULTIPLE(CHUNK_DIM_Y * sizeof(float), TMA_SHMEM_ALIGNMENT);

    // shmem layout: [act input][gate input][colwise output][prob]
    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char *dshmem = align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType *sInAct_ptr = reinterpret_cast<IType *>(dshmem);
    IType *sInGate_ptr = reinterpret_cast<IType *>(dshmem + buff_size_aligned_in);
    OType *sOutColwise_ptr = reinterpret_cast<OType *>(dshmem + 2 * buff_size_aligned_in);
    float *sProb_ptr =
        reinterpret_cast<float *>(dshmem + 2 * buff_size_aligned_in + buff_size_aligned_out);

    // Per-buffer byte count transferred by TMA (act + gate) into one slice.
    constexpr size_t shmem_buff_size = buff_size_aligned_in / BUFFS_NUM;

    const size_t total_work_blocks = work_blocks_X * work_blocks_Y;
    const size_t launch_block_id = blockIdx.y * gridDim.x + blockIdx.x;

    int IN_buff_readable_parity[BUFFS_NUM] = {0};

    if (launch_block_id >= total_work_blocks) {
      return;
    }
    int32_t ctaid_X = static_cast<int32_t>(launch_block_id % work_blocks_X);
    int32_t ctaid_Y = static_cast<int32_t>(launch_block_id / work_blocks_X);
    size_t static_block_stride = gridDim.x * gridDim.y;
    size_t static_next_block_id = launch_block_id + static_block_stride;

    bool job_finished = false;

    __shared__ uint64_t IN_buff_readable_mbar[BUFFS_NUM];
    initialize_barriers<BUFFS_NUM, 1>(IN_buff_readable_mbar, leading_thread);

    while (!job_finished) {
      const JobDescriptor current_job = decode_job<SHAPE_REP, CHUNK_DIM_Y, CHUNK_DIM_X>(
          num_tensors, first_logical_dim, last_logical_dim, work_blocks_X, ctaid_X, ctaid_Y,
          offsets_ptr, first_dims_ptr, last_dims_ptr);
      const bool current_job_is_valid =
          is_job_valid<SHAPE_REP>(current_job, total_work_blocks, offsets_ptr);
      if (!current_job_is_valid) {
        break;
      }
      if (!job_has_work(current_job)) {
        advance_to_next_job(job_finished, ctaid_X, ctaid_Y, static_next_block_id,
                            static_block_stride, total_work_blocks, work_blocks_X);
        continue;
      }

      const size_t rows = current_job.rows;
      const size_t cols = current_job.cols;
      const BlockDescriptor current_block =
          decode_block<SHAPE_REP, CHUNK_DIM_Y, CHUNK_DIM_X>(current_job, offsets_ptr);

      const size_t scale_alignment_X_colwise =
          static_cast<size_t>(scale_tensor_alignment_X_colwise);
      const size_t scale_stride_colwise = DIVUP_TO_MULTIPLE(cols, scale_alignment_X_colwise);

      // Only the swizzled layout needs the per-expert base; offsets_ptr may be null
      // otherwise (SAME_BOTH_DIMS), so keep the read inside the constexpr branch.
      size_t tensor_base_for_scales = 0;
      if constexpr (WITH_GEMM_SWIZZLED_SCALES) {
        tensor_base_for_scales = (num_tensors > 1)
                                     ? static_cast<size_t>(offsets_ptr[current_job.tensor_id])
                                     : current_block.tensor_base;
      }

      const size_t block_id_Y = current_block.block_id_Y;
      const size_t block_id_X = current_block.block_id_X;
      const size_t block_offset_Y = current_block.block_offset_Y;
      const size_t block_offset_X = current_block.block_offset_X;

      const size_t scales_block_offset_Y_colwise = block_id_Y * CHUNK_DIM_Y / SCALE_DIM_Y;
      const size_t scales_block_offset_X_colwise = block_id_X * CHUNK_DIM_X;
      const size_t scales_offset_Y_colwise = scales_block_offset_Y_colwise;
      const size_t scales_offset_X_colwise = scales_block_offset_X_colwise + tid_X_colwise;

      // Stage this chunk's per-token prob once. is_job_valid guarantees every row of a
      // valid 128-aligned block is a real token of this expert, so the absolute token
      // index is always in [0, T). prob rides along in the input (model) dtype,
      // matching cuDNN fc1_prob_tensor.
      for (size_t row = threadIdx.x; row < CHUNK_DIM_Y; row += THREADS_PER_CHUNK) {
        sProb_ptr[row] = static_cast<float>(prob_ptr[block_offset_Y + row]);
      }

      __syncthreads();

      int buff_in = 0;

// Prime the pipeline with the first PREFETCH_STAGES slices (act + gate).
#pragma unroll
      for (int stage = 0; stage < PREFETCH_STAGES; ++stage) {
        const size_t buff = stage;
        const size_t stage_offset_Y = stage * BUFF_DIM_Y;
        const size_t global_offset_Y = block_offset_Y + stage_offset_Y;
        const size_t global_offset_X = block_offset_X;
        const size_t buff_offset = buff * BUFF_DIM;
        uint64_t *barrier = &IN_buff_readable_mbar[buff];
        if (leading_thread) {
          ptx::mbarrier_arrive_expect_tx(barrier, 2 * shmem_buff_size);
          ptx::cp_async_bulk_tensor_2d_global_to_shared(
              reinterpret_cast<uint64_t *>(&sInAct_ptr[buff_offset]),
              reinterpret_cast<const uint64_t *>(&tensor_map_input_act_static), global_offset_X,
              global_offset_Y, barrier);
          ptx::cp_async_bulk_tensor_2d_global_to_shared(
              reinterpret_cast<uint64_t *>(&sInGate_ptr[buff_offset]),
              reinterpret_cast<const uint64_t *>(&tensor_map_input_gate_static), global_offset_X,
              global_offset_Y, barrier);
        }
      }

#pragma unroll
      for (int stage = 0; stage < STAGES; ++stage) {
        const size_t stage_offset_Y = stage * BUFF_DIM_Y;
        if (stage < STAGES - PREFETCH_STAGES) {
          const size_t next_prefetch_buff = (buff_in + PREFETCH_STAGES) % BUFFS_NUM;
          const size_t next_prefetch_stage = stage + PREFETCH_STAGES;
          const size_t next_prefetch_stage_offset_Y = next_prefetch_stage * BUFF_DIM_Y;
          const size_t global_offset_Y = block_offset_Y + next_prefetch_stage_offset_Y;
          const size_t global_offset_X = block_offset_X;
          const size_t next_prefetch_buff_offset = next_prefetch_buff * BUFF_DIM;
          uint64_t *barrier = &IN_buff_readable_mbar[next_prefetch_buff];
          if (leading_thread) {
            ptx::mbarrier_arrive_expect_tx(barrier, 2 * shmem_buff_size);
            ptx::cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t *>(&sInAct_ptr[next_prefetch_buff_offset]),
                reinterpret_cast<const uint64_t *>(&tensor_map_input_act_static), global_offset_X,
                global_offset_Y, barrier);
            ptx::cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t *>(&sInGate_ptr[next_prefetch_buff_offset]),
                reinterpret_cast<const uint64_t *>(&tensor_map_input_gate_static), global_offset_X,
                global_offset_Y, barrier);
          }
        }

        ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&IN_buff_readable_mbar[buff_in],
                                                         IN_buff_readable_parity[buff_in]);
        IN_buff_readable_parity[buff_in] ^= 1;
        // Wait until the store groups still holding an output slice have drained. Only
        // the leading thread commits those groups, so the wait is a no-op on the other
        // threads and the barrier is what stops them from overwriting a slice the TMA
        // unit has not finished reading.
        ptx::cp_async_bulk_wait_group_read<OUT_BUFFS_NUM - 1>();
        __syncthreads();

        const size_t buff = buff_in;
        const size_t out_buff = buff_in % OUT_BUFFS_NUM;
        process_colwise_gated_stage<ParamOP, OP, IType, OType, WITH_GEMM_SWIZZLED_SCALES>(
            buff, out_buff, stage, tid_X_colwise, scales_offset_Y_colwise, scales_offset_X_colwise,
            scale_stride_colwise, tensor_base_for_scales, rows, cols, p, sProb_ptr, sInAct_ptr,
            sInGate_ptr, sOutColwise_ptr, scales_colwise_ptr);

        ptx::fence_proxy_async_shared_cta();
        __syncthreads();

        const size_t global_offset_Y = block_offset_Y + stage_offset_Y;
        const size_t global_offset_X = block_offset_X;
        const size_t out_buff_offset = out_buff * BUFF_DIM;
        if (leading_thread) {
          ptx::cp_async_bulk_tensor_2d_shared_to_global(
              reinterpret_cast<const uint64_t *>(&tensor_map_output_colwise_static),
              global_offset_X, global_offset_Y,
              reinterpret_cast<uint64_t *>(&sOutColwise_ptr[out_buff_offset]));
          ptx::cp_async_bulk_commit_group();
        }

        buff_in = (buff_in + 1) % BUFFS_NUM;
      }

      advance_to_next_job(job_finished, ctaid_X, ctaid_Y, static_next_block_id, static_block_stride,
                          total_work_blocks, work_blocks_X);
    }

    destroy_barriers<BUFFS_NUM>(IN_buff_readable_mbar, leading_thread);
  }  // if constexpr (is_single_tensor)
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

}  // namespace group_scaled_swiglu_kernel

// Host launcher: grouped scaled SwiGLU -> columnwise MXFP8.
//   input  : GroupedTensor [T, 2F] ([act|gate]) in a floating input dtype.
//   prob   : Tensor [T] per-token weights, in the input (model) dtype.
//   output : GroupedTensor with columnwise_data / columnwise_scale_inv for [T, F].
//   p      : Empty for plain SwiGLU, ClampedSwiGLUParam for the clamped variant.
template <typename ParamOP, float (*OP)(float, const ParamOP &)>
void group_scaled_swiglu(const GroupedTensor *input, const Tensor *prob, const Tensor *noop,
                         GroupedTensor *output, const ParamOP &p,
                         const QuantizationConfig *quant_config, cudaStream_t stream) {
  using namespace group_scaled_swiglu_kernel;

  checkCuDriverContext(stream);
  CheckNoopTensor(*noop, "cast_noop");

  NVTE_CHECK(output->has_columnwise_data(),
             "group_scaled_swiglu requires columnwise output data to be allocated.");
  NVTE_CHECK(!output->has_data(),
             "group_scaled_swiglu produces a columnwise output only; "
             "rowwise is not implemented.");
  NVTE_CHECK(is_fp8_dtype(output->dtype()), "Output must have FP8 type.");
  NVTE_CHECK(input->num_tensors == output->num_tensors,
             "Number of input and output tensors must be same.");
  NVTE_CHECK(input->has_data(), "Cannot quantize tensor without rowwise data.");

  // Determine grouped shape representation from the output metadata.
  ShapeRepresentation shape_rep = ShapeRepresentation::SAME_BOTH_DIMS;
  if (output->all_same_shape()) {
    shape_rep = ShapeRepresentation::SAME_BOTH_DIMS;
  } else if (output->all_same_last_dim()) {
    shape_rep = ShapeRepresentation::VARYING_FIRST_DIM;
  } else {
    NVTE_CHECK(false,
               "group_scaled_swiglu requires all experts to share the same last dim F "
               "(grouped layout SAME_BOTH_DIMS or VARYING_FIRST_DIM).");
  }

  const bool with_gemm_swizzled_scales = output->with_gemm_swizzled_scales;

  // Output logical shape drives the schedule ([T, F]); input is [T, 2F].
  const size_t first_logical_dim = output->logical_shape.data[0];     // T
  const size_t out_last_logical_dim = output->logical_shape.data[1];  // F
  const size_t in_last_logical_dim = input->logical_shape.data[1];    // 2F

  NVTE_CHECK(in_last_logical_dim == 2 * out_last_logical_dim,
             "group_scaled_swiglu input last dim must be 2x the output last dim ([act|gate]).");
  NVTE_CHECK(input->logical_shape.data[0] == first_logical_dim,
             "group_scaled_swiglu input/output must share the token dimension T.");

  const size_t T = first_logical_dim;
  const size_t F = out_last_logical_dim;
  const size_t num_tensors = input->num_tensors;

  NVTE_CHECK(prob != nullptr && prob->data.dptr != nullptr, "prob tensor must be allocated.");
  // prob follows TE's cuDNN fc1_prob_tensor convention: model (input) dtype.
  NVTE_CHECK(prob->data.dtype == input->dtype(),
             "prob tensor must have the same dtype as the input (model dtype).");
  NVTE_CHECK(prob->data.numel() >= T, "prob tensor must have at least T elements.");

  // Single-tensor schedule: one virtual work grid over [T, F].
  const size_t work_blocks_Y = DIVUP(T, static_cast<size_t>(CHUNK_DIM_Y));
  const size_t work_blocks_X = DIVUP(F, static_cast<size_t>(CHUNK_DIM_X));

  NVTE_CHECK(T % 128 == 0, "group_scaled_swiglu requires T divisible by 128.");

  const size_t sm_num = static_cast<size_t>(transformer_engine::cuda::sm_count());
  const size_t static_grid_size = sm_num * TunableConfig::STATIC_PERSISTENT_BLOCKS_PER_SM;
  NVTE_CHECK(static_grid_size > 0, "Static persistent grid size must be greater than zero.");
  const dim3 grid(static_grid_size);
  const size_t block_size = THREADS_PER_CHUNK;

  const int64_t *const offsets_ptr = reinterpret_cast<const int64_t *>(output->tensor_offsets.dptr);
  const int64_t *const first_dims_ptr = reinterpret_cast<const int64_t *>(output->first_dims.dptr);
  const int64_t *const last_dims_ptr = reinterpret_cast<const int64_t *>(output->last_dims.dptr);

  if (with_gemm_swizzled_scales) {
    // The swizzled block is tiled 128x4 over the transposed [F, rows/32] scale
    // matrix, so a partial F tile would not map onto a whole number of tiles.
    NVTE_CHECK(F % 128 == 0,
               "group_scaled_swiglu with GEMM-swizzled scales requires the output "
               "last dim (F) to be divisible by 128, got ",
               F, ".");
    if (num_tensors > 1) {
      // Each expert owns a separate swizzled block whose extent depends on its
      // own token count, so per-expert first dims and offsets are mandatory.
      NVTE_CHECK(shape_rep == ShapeRepresentation::VARYING_FIRST_DIM,
                 "group_scaled_swiglu with GEMM-swizzled scales and multiple experts "
                 "requires per-expert first dims (pass first_dims / split_sections).");
      NVTE_CHECK(offsets_ptr != nullptr,
                 "group_scaled_swiglu with GEMM-swizzled scales requires tensor_offsets "
                 "to locate each expert's swizzled scale block.");
    }
  }

  const float *const noop_ptr = reinterpret_cast<const float *>(noop->data.dptr);
  e8m0_t *const scales_colwise_ptr = reinterpret_cast<e8m0_t *>(output->columnwise_scale_inv.dptr);
  NVTE_CHECK(scales_colwise_ptr != nullptr, "Columnwise scaling tensor must be allocated");

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input->dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,
          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              with_gemm_swizzled_scales, WITH_GEMM_SWIZZLED_SCALES,
              TRANSFORMER_ENGINE_GROUP_TENSOR_SHAPE_REPRESENTATION_SWITCH(
                  shape_rep, SHAPE_REP,
                  {
                    alignas(64) CUtensorMap tensor_map_input_act{};
                    alignas(64) CUtensorMap tensor_map_input_gate{};
                    alignas(64) CUtensorMap tensor_map_output_colwise{};

                    constexpr size_t input_type_bit_size = TypeInfo<IType>::size;
                    constexpr size_t output_type_bit_size = TypeInfo<OType>::size;

                    const IType *const prob_dptr = reinterpret_cast<const IType *>(prob->data.dptr);

                    // act half: [T, F] view of the [T, 2F] buffer, stride 2F, offset 0.
                    create_2D_tensor_map(tensor_map_input_act, input->data, T, F, BUFF_DIM_Y,
                                         BUFF_DIM_X, 2 * F, 0, input_type_bit_size);
                    // gate half: same view, offset F.
                    create_2D_tensor_map(tensor_map_input_gate, input->data, T, F, BUFF_DIM_Y,
                                         BUFF_DIM_X, 2 * F, F, input_type_bit_size);
                    // colwise output: [T, F] contiguous, stride F.
                    create_2D_tensor_map(tensor_map_output_colwise, output->columnwise_data, T, F,
                                         BUFF_DIM_Y, BUFF_DIM_X, F, 0, output_type_bit_size);

                    constexpr size_t buff_elems = BUFF_DIM_Y * BUFF_DIM_X;
                    constexpr size_t buff_elems_total = BUFFS_NUM * buff_elems;
                    constexpr size_t input_buff_size = (buff_elems_total * input_type_bit_size) / 8;
                    constexpr size_t out_buff_elems_total = OUT_BUFFS_NUM * buff_elems;
                    constexpr size_t output_buff_size =
                        (out_buff_elems_total * output_type_bit_size) / 8;
                    constexpr size_t buff_size_aligned_in =
                        DIVUP_TO_MULTIPLE(input_buff_size, TMA_SHMEM_ALIGNMENT);
                    constexpr size_t buff_size_aligned_out =
                        DIVUP_TO_MULTIPLE(output_buff_size, TMA_SHMEM_ALIGNMENT);

                    constexpr size_t prob_buff_size =
                        DIVUP_TO_MULTIPLE(CHUNK_DIM_Y * sizeof(float), TMA_SHMEM_ALIGNMENT);

                    // [act][gate][colwise out][prob]
                    const size_t dshmem_size = 2 * buff_size_aligned_in + buff_size_aligned_out +
                                               prob_buff_size + TMA_SHMEM_ALIGNMENT;

                    auto kernel =
                        group_scaled_swiglu_mxfp8_kernel<ParamOP, OP, IType, OType,
                                                         WITH_GEMM_SWIZZLED_SCALES, SHAPE_REP>;

                    NVTE_CHECK_CUDA(cudaFuncSetAttribute(
                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem_size));

                    kernel<<<grid, block_size, dshmem_size, stream>>>(
                        tensor_map_input_act, tensor_map_input_gate, tensor_map_output_colwise,
                        num_tensors, T, F, offsets_ptr, first_dims_ptr, last_dims_ptr, prob_dptr,
                        scales_colwise_ptr, noop_ptr, work_blocks_X, work_blocks_Y, p);

                    NVTE_CHECK_CUDA(cudaGetLastError());
                  });  // NOLINT(*)
          );           // NOLINT(*)
      );               // NOLINT(*)
  );                   // NOLINT(*)
}

}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine
#endif  // TRANSFORMER_ENGINE_GROUP_SCALED_SWIGLU_MXFP8_CUH_
