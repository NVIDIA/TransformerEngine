/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file gated_mxfp8_rowwise_swiglu.cuh
 *  \brief Optimized BWD/FWD Gated (SwiGLU) MXFP8 Rowwise kernel for BF16/FP16 inputs
 */

#ifndef TRANSFORMER_ENGINE_SPECIALIZED_GATED_MXFP8_ROWWISE_SWIGLU_CUH_
#define TRANSFORMER_ENGINE_SPECIALIZED_GATED_MXFP8_ROWWISE_SWIGLU_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include "../../../common.h"
#include "../../../util/math.h"
#include "../../../util/ptx.cuh"
#include "../../../utils.cuh"
#include "../../core/common.cuh"

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {
namespace gated_rowwise_kernel {

constexpr size_t SCALE_DIM = 32;

constexpr size_t CHUNK_DIM_X = 1024;
constexpr size_t SCALES_PER_CHUNK_X = CHUNK_DIM_X / SCALE_DIM;

constexpr size_t PREFETCH_STAGES = 1;
static_assert(PREFETCH_STAGES >= 1);

constexpr size_t BUFFS_NUM = PREFETCH_STAGES + 1;
constexpr size_t BUFF_DIM_X = 256;
constexpr size_t SCALES_PER_BUFF_X = BUFF_DIM_X / SCALE_DIM;

constexpr size_t ELEMS_PER_THREAD = 8;
constexpr size_t THREADS_NUM = 256;
constexpr size_t THREADS_X = BUFF_DIM_X / ELEMS_PER_THREAD;
constexpr size_t THREADS_Y = THREADS_NUM / THREADS_X;

constexpr size_t ITERS = 2;
constexpr size_t BUFF_DIM_Y = THREADS_Y * ITERS;

constexpr size_t CHUNK_DIM_Y = BUFF_DIM_Y;
constexpr size_t SCALES_PER_CHUNK_Y = CHUNK_DIM_Y;
constexpr size_t SCALES_PER_CHUNK = SCALES_PER_CHUNK_Y * SCALES_PER_CHUNK_X;

constexpr size_t BUFF_DIM = BUFF_DIM_Y * BUFF_DIM_X;

constexpr size_t STAGES = CHUNK_DIM_X / BUFF_DIM_X;
static_assert(STAGES >= 1);
static_assert(CHUNK_DIM_X % BUFF_DIM_X == 0);

constexpr size_t THREADS_PER_MX_BLOCK = SCALE_DIM / ELEMS_PER_THREAD;
constexpr size_t PACK_SIZE = 2;
static_assert(PACK_SIZE ==
              2);  // loads a pair of elements (4 Bytes of 2x BF16, i.e. 1x SMEM bank per load)

constexpr size_t WAVES = ELEMS_PER_THREAD / PACK_SIZE;  // number of pairs per thread
constexpr size_t GROUPS = WAVES;

template <typename IType2>
__device__ __forceinline__ float get_amax_of_pair(const IType2 xormax_pair) {
  return static_cast<float>(__hmax(__habs(xormax_pair.x), __habs(xormax_pair.y)));
}

template <typename IType2>
__device__ __forceinline__ void compute_bwd_gated_swiglu_tuned_ptx(
    IType2 &out_act, IType2 &out_gate, IType2 &out_xormax_act, IType2 &out_xormax_gate,
    const float2 in_act, const float2 in_gate, const float2 in_grad) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const IType2 in_xormax_act = out_xormax_act;
  const IType2 in_xormax_gate = out_xormax_gate;

  const float2 s{sigmoidf(in_act.x), sigmoidf(in_act.y)};

  if constexpr (std::is_same<IType2, typename ptx::bf16x2>::value) {
    asm volatile(
        "{\n\t"
        ".reg.b64 in_act, in_gate, in_grad, s; \n\t"
        "mov.b64 in_act, %6; \n\t"
        "mov.b64 in_gate, %7; \n\t"
        "mov.b64 in_grad, %8; \n\t"
        "mov.b64 s, %9; \n\t"

        ".reg.b64 act; \n\t"
        "mul.f32x2 act, in_act, s; \n\t"  // x * s
        ".reg.f32 one; \n\t"
        "mov.f32 one, 1.0; \n\t"
        ".reg.b64 ones; \n\t"
        "mov.b64 ones, {one, one}; \n\t"
        ".reg.b64 sub1s; \n\t"
        "sub.f32x2 sub1s, ones, s; \n\t"  // (1 - s)
        ".reg.b64 dact; \n\t"
        "fma.rn.f32x2 dact, act, sub1s, s; \n\t"  // act * (1 - s) + s
        ".reg.b64 out_act; \n\t"
        ".reg.b64 out_gate; \n\t"
        ".reg.b64 grad_gate; \n\t"
        "mul.f32x2 grad_gate, in_gate, in_grad; \n\t"
        "mul.f32x2 out_act, dact, grad_gate; \n\t"
        "mul.f32x2 out_gate, act, in_grad; \n\t"
        ".reg.f32 out_act1, out_act2; \n\t"
        ".reg.f32 out_gate1, out_gate2; \n\t"
        "mov.b64 {out_act1, out_act2}, out_act; \n\t"
        "mov.b64 {out_gate1, out_gate2}, out_gate; \n\t"
        ".reg.b32 cvt_out_act, cvt_out_gate; \n\t"
        "cvt.rn.satfinite.bf16x2.f32 cvt_out_act, out_act2, out_act1; \n\t"
        "cvt.rn.satfinite.bf16x2.f32 cvt_out_gate, out_gate2, out_gate1; \n\t"
        "mov.b32 %0, cvt_out_act; \n\t"
        "mov.b32 %1, cvt_out_gate; \n\t"
        "max.xorsign.abs.bf16x2 %2, %4, cvt_out_act; \n\t"
        "max.xorsign.abs.bf16x2 %3, %5, cvt_out_gate; \n\t"
        "}\n"
        : "=r"(reinterpret_cast<uint32_t &>(out_act)), "=r"(reinterpret_cast<uint32_t &>(out_gate)),
          "=r"(reinterpret_cast<uint32_t &>(out_xormax_act)),
          "=r"(reinterpret_cast<uint32_t &>(out_xormax_gate))
        : "r"(reinterpret_cast<const uint32_t &>(in_xormax_act)),
          "r"(reinterpret_cast<const uint32_t &>(in_xormax_gate)),
          "l"(reinterpret_cast<const uint64_t &>(in_act)),
          "l"(reinterpret_cast<const uint64_t &>(in_gate)),
          "l"(reinterpret_cast<const uint64_t &>(in_grad)),
          "l"(reinterpret_cast<const uint64_t &>(s)));
  } else if constexpr (std::is_same<IType2, typename ptx::fp16x2>::value) {
    asm volatile(
        "{\n\t"
        ".reg.b64 in_act, in_gate, in_grad, s; \n\t"
        "mov.b64 in_act, %6; \n\t"
        "mov.b64 in_gate, %7; \n\t"
        "mov.b64 in_grad, %8; \n\t"
        "mov.b64 s, %9; \n\t"

        ".reg.b64 act; \n\t"
        "mul.f32x2 act, in_act, s; \n\t"  // x * s
        ".reg.f32 one; \n\t"
        "mov.f32 one, 1.0; \n\t"
        ".reg.b64 ones; \n\t"
        "mov.b64 ones, {one, one}; \n\t"
        ".reg.b64 sub1s; \n\t"
        "sub.f32x2 sub1s, ones, s; \n\t"  // (1 - s)
        ".reg.b64 dact; \n\t"
        "fma.rn.f32x2 dact, act, sub1s, s; \n\t"  // act * (1 - s) + s
        ".reg.b64 out_act; \n\t"
        ".reg.b64 out_gate; \n\t"
        ".reg.b64 grad_gate; \n\t"
        "mul.f32x2 grad_gate, in_gate, in_grad; \n\t"
        "mul.f32x2 out_act, dact, grad_gate; \n\t"
        "mul.f32x2 out_gate, act, in_grad; \n\t"
        ".reg.f32 out_act1, out_act2; \n\t"
        ".reg.f32 out_gate1, out_gate2; \n\t"
        "mov.b64 {out_act1, out_act2}, out_act; \n\t"
        "mov.b64 {out_gate1, out_gate2}, out_gate; \n\t"
        ".reg.b32 cvt_out_act, cvt_out_gate; \n\t"
        "cvt.rn.satfinite.f16x2.f32 cvt_out_act, out_act2, out_act1; \n\t"
        "cvt.rn.satfinite.f16x2.f32 cvt_out_gate, out_gate2, out_gate1; \n\t"
        "mov.b32 %0, cvt_out_act; \n\t"
        "mov.b32 %1, cvt_out_gate; \n\t"
        "max.xorsign.abs.f16x2 %2, %4, cvt_out_act; \n\t"
        "max.xorsign.abs.f16x2 %3, %5, cvt_out_gate; \n\t"
        "}\n"
        : "=r"(reinterpret_cast<uint32_t &>(out_act)), "=r"(reinterpret_cast<uint32_t &>(out_gate)),
          "=r"(reinterpret_cast<uint32_t &>(out_xormax_act)),
          "=r"(reinterpret_cast<uint32_t &>(out_xormax_gate))
        : "r"(reinterpret_cast<const uint32_t &>(in_xormax_act)),
          "r"(reinterpret_cast<const uint32_t &>(in_xormax_gate)),
          "l"(reinterpret_cast<const uint64_t &>(in_act)),
          "l"(reinterpret_cast<const uint64_t &>(in_gate)),
          "l"(reinterpret_cast<const uint64_t &>(in_grad)),
          "l"(reinterpret_cast<const uint64_t &>(s)));
  }
#else
  NVTE_DEVICE_ERROR("compute_bwd_gated_swiglu is only supported on SM 10.0+.");
#endif
}

template <typename IType2>
__device__ __forceinline__ void compute_bwd_gated_swiglu(IType2 &out_act, IType2 &out_gate,
                                                         IType2 &out_xormax_act,
                                                         IType2 &out_xormax_gate,
                                                         const float2 in_act, const float2 in_gate,
                                                         const float2 in_grad) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const IType2 in_xormax_act = out_xormax_act;
  const IType2 in_xormax_gate = out_xormax_gate;

  const float2 s{sigmoidf(in_act.x), sigmoidf(in_act.y)};

  constexpr bool USE_TUNED_SWIGLU_PTX = true;
  if constexpr (USE_TUNED_SWIGLU_PTX) {
    compute_bwd_gated_swiglu_tuned_ptx<IType2>(out_act, out_gate, out_xormax_act, out_xormax_gate,
                                               in_act, in_gate, in_grad);
  } else {
    const float &x1 = in_act.x;
    const float &x2 = in_act.y;
    const float &s1 = s.x;
    const float &s2 = s.y;
    const float act1 = x1 * s1;
    const float act2 = x2 * s2;
    const float dact1 = x1 * s1 * (1 - s1) + s1;
    const float dact2 = x2 * s2 * (1 - s2) + s2;

    const float after_act_elt1 = dact1 * in_grad.x * in_gate.x;
    const float after_act_elt2 = dact2 * in_grad.y * in_gate.y;
    const float after_gate_elt1 = act1 * in_grad.x;
    const float after_gate_elt2 = act2 * in_grad.y;

    out_act = IType2{after_act_elt1, after_act_elt2};
    out_gate = IType2{after_gate_elt1, after_gate_elt2};

    ptx::abs_max_2x(out_xormax_act, in_xormax_act, out_act);
    ptx::abs_max_2x(out_xormax_gate, in_xormax_gate, out_gate);
  }
#else
  NVTE_DEVICE_ERROR("compute_bwd_gated_swiglu is only supported on SM 10.0+.");
#endif
}

template <typename IType2, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
__device__ __forceinline__ void compute_bwd_gated_activation(
    IType2 &out_act, IType2 &out_gate, IType2 &out_xormax_act, IType2 &out_xormax_gate,
    const float2 in_act, const float2 in_gate, const float2 in_grad, const ParamOP p) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  // SwiGLU activation
  constexpr bool IS_CLAMPED_SWIGLU = std::is_same<ParamOP, ClampedSwiGLUParam>::value;

  if constexpr (!IS_CLAMPED_SWIGLU) {
    if constexpr ((ActOP == &silu<fp32, fp32>) && (DActOP == &dsilu<fp32, fp32>)) {
      compute_bwd_gated_swiglu<IType2>(out_act, out_gate, out_xormax_act, out_xormax_gate, in_act,
                                       in_gate, in_grad);
      return;
    }
  }

  float after_act_elt[2];
  float after_gate_elt[2];
#pragma unroll
  for (int e = 0; e < 2; ++e) {
    const float act_elt = (e == 0) ? in_act.x : in_act.y;
    float gate_elt = (e == 0) ? in_gate.x : in_gate.y;
    const float grad_elt = (e == 0) ? in_grad.x : in_grad.y;

    bool dgate_elt = true;  // gating is ideally an identity function
    if constexpr (IS_CLAMPED_SWIGLU) {
      // In case of GPT OSS, clamp the activation and gate values
      dgate_elt = gate_elt <= p.limit && gate_elt >= -p.limit;  // Derivative of clamp
      gate_elt = min(max(-p.limit, gate_elt), p.limit) + 1.0f;
    }
    float act_x;
    float dact_x;
    if constexpr (IS_CLAMPED_SWIGLU) {
      const float x = min(act_elt, p.limit);
      const float s = sigmoidf(p.alpha * x);
      act_x = x * s;
      dact_x = act_elt <= p.limit ? s + s * (1 - s) * p.alpha * x : 0.0f;
    } else {
      const float x = act_elt;
      act_x = ActOP(x, p);
      dact_x = DActOP(x, p);
    }

    after_act_elt[e] = dact_x * grad_elt * gate_elt;
    after_gate_elt[e] = dgate_elt ? act_x * grad_elt : 0.0f;
  }
  out_act = IType2{after_act_elt[0], after_act_elt[1]};
  out_gate = IType2{after_gate_elt[0], after_gate_elt[1]};

  const IType2 in_xormax_act = out_xormax_act;
  const IType2 in_xormax_gate = out_xormax_gate;
  ptx::abs_max_2x(out_xormax_act, in_xormax_act, out_act);
  ptx::abs_max_2x(out_xormax_gate, in_xormax_gate, out_gate);

#else
  NVTE_DEVICE_ERROR("compute_bwd_gated_activation is only supported on SM 10.0+.");
#endif
}

template <typename IType2, typename ParamOP, float (*ActOP)(float, const ParamOP &)>
__device__ __forceinline__ void compute_fwd_gated_activation(IType2 &out_act,
                                                             IType2 &out_xormax_act,
                                                             const float2 in_act,
                                                             const float2 in_gate,
                                                             const ParamOP p) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  float after_act_elt[2];
#pragma unroll
  for (int e = 0; e < 2; ++e) {
    const float act_elt = (e == 0) ? in_act.x : in_act.y;
    float gate_elt = (e == 0) ? in_gate.x : in_gate.y;

    if constexpr (std::is_same<ParamOP, ClampedSwiGLUParam>::value) {
      // In case of GPT OSS, clamp the activation and gate values
      gate_elt = min(max(-p.limit, gate_elt), p.limit) + 1.0f;
    }
    after_act_elt[e] = ActOP(act_elt, p) * gate_elt;
  }
  out_act = IType2{after_act_elt[0], after_act_elt[1]};

  const IType2 in_xormax_act = out_xormax_act;
  ptx::abs_max_2x(out_xormax_act, in_xormax_act, out_act);
#else
  NVTE_DEVICE_ERROR("compute_fwd_gated_activation is only supported on SM 10.0+.");
#endif
}

template <bool IS_BWD, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &), typename IType, typename OType>
__global__ void __launch_bounds__(THREADS_NUM) quantize_gated_mxfp8_rowwise_kernel(
    const __grid_constant__ CUtensorMap tensor_map_grad,
    const __grid_constant__ CUtensorMap tensor_map_input_act,
    const __grid_constant__ CUtensorMap tensor_map_input_gate,
    const __grid_constant__ CUtensorMap tensor_map_output_act_rowwise,
    const __grid_constant__ CUtensorMap tensor_map_output_gate_rowwise,
    e8m0_t *const scales_rowwise, const size_t rows, const size_t cols,
    const size_t scale_stride_rowwise, const ParamOP p) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using IType2 = typename ptx::FPx2<IType>;
  using OType2 = typename ptx::FPx2<OType>;

  const size_t tid_Y = threadIdx.x / THREADS_X;
  const size_t tid_X = threadIdx.x % THREADS_X;

  const size_t thread_offset_Y = tid_Y;
  const size_t thread_offset_X = tid_X * ELEMS_PER_THREAD;

  const bool leading_thread = (threadIdx.x == 0);
  const bool SF_storing_thread = tid_X % THREADS_PER_MX_BLOCK == 0;

  const size_t gate_scale_idx_offset = (cols + SCALE_DIM - 1) / SCALE_DIM;

  // helps resolving bank conflicts in shmem
  const int thread_lane = threadIdx.x % THREADS_PER_WARP;
  const int bank_group = thread_lane / GROUPS;

  constexpr size_t SF_CHANNELS = IS_BWD ? 2 : 1;

  __shared__ e8m0_t __align__(4) scales_sh[SF_CHANNELS][CHUNK_DIM_Y][SCALES_PER_CHUNK_X];

  extern __shared__ unsigned char dynamic_shmem[];
  unsigned char *dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

  constexpr size_t buff_elems = BUFF_DIM_Y * BUFF_DIM_X;
  constexpr size_t buff_elems_total = BUFFS_NUM * buff_elems;
  constexpr size_t buff_size_aligned_in =
      DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(IType), TMA_SHMEM_ALIGNMENT);
  constexpr size_t buff_size_aligned_out =
      DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(OType), TMA_SHMEM_ALIGNMENT);

  const size_t grad_mem = (IS_BWD ? buff_size_aligned_in : 0);

  const size_t in_act_mem = buff_size_aligned_in;
  const size_t in_gate_mem = buff_size_aligned_in;
  const size_t in_mem = in_act_mem + in_gate_mem;

  const size_t out_act_mem = buff_size_aligned_out;
  const size_t out_gate_mem = (IS_BWD ? buff_size_aligned_out : 0);
  const size_t out_mem = out_act_mem + out_gate_mem;

  // The destination shared memory buffer of a bulk tensor operation should be 16-byte aligned
  IType *in_grad_sh_ptr = reinterpret_cast<IType *>(dshmem);
  IType *in_act_sh_ptr = reinterpret_cast<IType *>(dshmem + grad_mem);
  IType *in_gate_sh_ptr = reinterpret_cast<IType *>(dshmem + grad_mem + in_act_mem);

  OType *out_act_rowwise_sh_ptr = reinterpret_cast<OType *>(dshmem + grad_mem + in_mem);
  OType *out_gate_rowwise_sh_ptr =
      reinterpret_cast<OType *>(dshmem + grad_mem + in_mem + out_act_mem);

  using IType2x3D = IType2[BUFFS_NUM][BUFF_DIM_Y][BUFF_DIM_X / 2];
  using OType2x3D = OType2[BUFFS_NUM][BUFF_DIM_Y][BUFF_DIM_X / 2];

  auto &in_act = *reinterpret_cast<IType2x3D *>(in_act_sh_ptr);
  auto &in_gate = *reinterpret_cast<IType2x3D *>(in_gate_sh_ptr);
  auto &in_grad = *reinterpret_cast<IType2x3D *>(in_grad_sh_ptr);
  auto &out_act = *reinterpret_cast<OType2x3D *>(out_act_rowwise_sh_ptr);
  auto &out_gate = *reinterpret_cast<OType2x3D *>(out_gate_rowwise_sh_ptr);

  constexpr size_t shmem_buff_size = (IS_BWD ? 3 : 2) * (buff_size_aligned_in / BUFFS_NUM);

  __shared__ uint64_t workID_mbar;
  __shared__ __uint128_t workID_response;
  constexpr uint32_t workID_response_size = sizeof(workID_response);
  static_assert(workID_response_size == 16);

  __shared__ uint64_t IN_buff_readable_mbar[BUFFS_NUM];

  // Coordinates of the first chunk (CTA) to process
  int32_t ctaid_X = blockIdx.x;
  int32_t ctaid_Y = blockIdx.y;

  // Initialize shared memory barriers with the number of threads participating in them.
  if (leading_thread) {
#pragma unroll
    for (int buff = 0; buff < BUFFS_NUM; ++buff) {
      ptx::mbarrier_init(&IN_buff_readable_mbar[buff], 1);
    }
    ptx::mbarrier_init(&workID_mbar, 1);
    ptx::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  bool job_finished = false;
  int buff_in = 0;
  int buff_out = 0;

  int IN_buff_readable_parity[BUFFS_NUM] = {0, 0};
  int ctaid_parity = 0;

// Prefetch input data only when processing the first chunk,
// which enables the one-iteration overlap throughout the entire kernel life
#pragma unroll
  for (size_t stage = 0; stage < PREFETCH_STAGES; ++stage) {
    const size_t buff = stage;
    const size_t stage_offset_X = stage * BUFF_DIM_X;

    // Offsets change, because coordinates of the next "to-be-prefetched" CTA do also chage
    const size_t block_offset_Y = ctaid_Y * CHUNK_DIM_Y;
    const size_t block_offset_X = ctaid_X * CHUNK_DIM_X;

    const size_t global_offset_Y = block_offset_Y;
    const size_t global_offset_X = block_offset_X + stage_offset_X;

    uint64_t *barrier = &IN_buff_readable_mbar[buff];
    if (leading_thread) {
      uint64_t *dst_act = reinterpret_cast<uint64_t *>(&in_act[buff]);
      uint64_t *dst_gate = reinterpret_cast<uint64_t *>(&in_gate[buff]);

      const uint64_t *src_act = reinterpret_cast<const uint64_t *>(&tensor_map_input_act);
      const uint64_t *src_gate = reinterpret_cast<const uint64_t *>(&tensor_map_input_gate);

      // Arrive on the barrier and tell how many bytes are expected to come in.
      ptx::mbarrier_arrive_expect_tx(barrier, shmem_buff_size);

      // Initiate bulk tensor copy
      ptx::cp_async_bulk_tensor_2d_global_to_shared(dst_act, src_act, global_offset_X,
                                                    global_offset_Y, barrier);
      ptx::cp_async_bulk_tensor_2d_global_to_shared(dst_gate, src_gate, global_offset_X,
                                                    global_offset_Y, barrier);

      if constexpr (IS_BWD) {
        uint64_t *dst_grad = reinterpret_cast<uint64_t *>(&in_grad[buff]);
        const uint64_t *src_grad = reinterpret_cast<const uint64_t *>(&tensor_map_grad);
        ptx::cp_async_bulk_tensor_2d_global_to_shared(dst_grad, src_grad, global_offset_X,
                                                      global_offset_Y, barrier);
      }
    }
  }

  while (!job_finished) {
    const size_t block_offset_Y = ctaid_Y * CHUNK_DIM_Y;
    const size_t block_offset_X = ctaid_X * CHUNK_DIM_X;
    const size_t scales_block_offset_Y = ctaid_Y * CHUNK_DIM_Y;
    const size_t scales_block_offset_X = ctaid_X * SCALES_PER_CHUNK_X;

    const size_t row_base = block_offset_Y + thread_offset_Y;
    const size_t col_base = block_offset_X + thread_offset_X;

    if (leading_thread) {
      ptx::mbarrier_arrive_expect_tx_cta_relaxed_shared_cta(&workID_mbar, workID_response_size);
      ptx::clusterlaunchcontrol_try_cancel_async_shared_cta_mbarrier_complete_tx_bytes(
          &workID_mbar, &workID_response);
    }

#pragma unroll
    for (int stage = 0; stage < STAGES; ++stage) {
      const size_t stage_offset_X = stage * BUFF_DIM_X;

      if (stage == STAGES - PREFETCH_STAGES) {
        ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&workID_mbar, ctaid_parity);
        ptx::get_cancelled_cta_2D_id(&workID_response, ctaid_X, ctaid_Y);
        if (ctaid_X == -1 && ctaid_Y == -1) {
          job_finished = true;
        }
        ctaid_parity ^= 1;
      }

      // Prefetch next stage Input data
      if (!job_finished || (stage < STAGES - PREFETCH_STAGES)) {
        const size_t next_prefetch_buff = (buff_in + PREFETCH_STAGES) % BUFFS_NUM;
        const size_t next_prefetch_stage = (stage + PREFETCH_STAGES) % STAGES;
        const size_t next_prefetch_stage_offset_X = next_prefetch_stage * BUFF_DIM_X;

        // Offsets change, because coordinates of the next "to-be-prefetched" CTA do also chage
        const size_t block_offset_Y = ctaid_Y * CHUNK_DIM_Y;
        const size_t block_offset_X = ctaid_X * CHUNK_DIM_X;

        const size_t global_offset_Y = block_offset_Y;
        const size_t global_offset_X = block_offset_X + next_prefetch_stage_offset_X;

        uint64_t *barrier = &IN_buff_readable_mbar[next_prefetch_buff];
        if (leading_thread) {
          uint64_t *dst_act = reinterpret_cast<uint64_t *>(&in_act[next_prefetch_buff]);
          uint64_t *dst_gate = reinterpret_cast<uint64_t *>(&in_gate[next_prefetch_buff]);

          const uint64_t *src_act = reinterpret_cast<const uint64_t *>(&tensor_map_input_act);
          const uint64_t *src_gate = reinterpret_cast<const uint64_t *>(&tensor_map_input_gate);

          // Arrive on the barrier and tell how many bytes are expected to come in.
          ptx::mbarrier_arrive_expect_tx(barrier, shmem_buff_size);

          // Initiate bulk tensor copy
          ptx::cp_async_bulk_tensor_2d_global_to_shared(dst_act, src_act, global_offset_X,
                                                        global_offset_Y, barrier);
          ptx::cp_async_bulk_tensor_2d_global_to_shared(dst_gate, src_gate, global_offset_X,
                                                        global_offset_Y, barrier);

          if constexpr (IS_BWD) {
            uint64_t *dst_grad = reinterpret_cast<uint64_t *>(&in_grad[next_prefetch_buff]);
            const uint64_t *src_grad = reinterpret_cast<const uint64_t *>(&tensor_map_grad);
            ptx::cp_async_bulk_tensor_2d_global_to_shared(dst_grad, src_grad, global_offset_X,
                                                          global_offset_Y, barrier);
          }
        }
        ptx::fence_proxy_async_shared_cta();
      }

      // Wait for the data to have arrived
      ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&IN_buff_readable_mbar[buff_in],
                                                       IN_buff_readable_parity[buff_in]);
      IN_buff_readable_parity[buff_in] ^= 1;

      ptx::cp_async_bulk_wait_group_read<PREFETCH_STAGES>();

// Read data, compute activations, write to cache
#pragma unroll
      for (int it = 0; it < ITERS; ++it) {
        const size_t it_offset_Y = it * THREADS_Y;
        const size_t Y = tid_Y + it_offset_Y;

        IType2 ACT[WAVES];
        IType2 GATE[WAVES];
        IType2 xormax_act{0.0f, 0.0f};
        IType2 xormax_gate{0.0f, 0.0f};

#pragma unroll
        for (int w = 0; w < WAVES; ++w) {
          const size_t staggered_idx = (w + bank_group) % GROUPS;
          const size_t X = tid_X * WAVES + staggered_idx;

          const float2 act_elt = ptx::ld_shared_cvt_f32x2(&in_act[buff_in][Y][X]);
          const float2 gate_elt = ptx::ld_shared_cvt_f32x2(&in_gate[buff_in][Y][X]);

          if constexpr (IS_BWD) {
            const float2 grad_elt = ptx::ld_shared_cvt_f32x2(&in_grad[buff_in][Y][X]);
            compute_bwd_gated_activation<IType2, ParamOP, ActOP, DActOP>(
                ACT[w], GATE[w], xormax_act, xormax_gate, act_elt, gate_elt, grad_elt, p);
          } else {
            compute_fwd_gated_activation<IType2, ParamOP, ActOP>(ACT[w], xormax_act, act_elt,
                                                                 gate_elt, p);
          }
        }

// If channel==0: compute ACT
// if channel==1: compute GATE
#pragma unroll
        for (int channel = 0; channel < SF_CHANNELS; ++channel) {
          float amax = get_amax_of_pair((channel == 0) ? xormax_act : xormax_gate);
#pragma unroll
          for (int r = 1; r < SCALE_DIM / ELEMS_PER_THREAD; r *= 2) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, r));
          }

          const e8m0_t biased_exponent =
              ptx::float_to_e8m0(amax * Quantized_Limits<OType>::max_norm_rcp);
          const float block_scale_inverse = ptx::exp2f_rcp(biased_exponent);
          const ptx::floatx2 block_scale_inverse_2x = {block_scale_inverse, block_scale_inverse};

          if (SF_storing_thread) {
            const size_t scales_Y = Y;
            const size_t scales_X = stage * SCALES_PER_BUFF_X + tid_X / THREADS_PER_MX_BLOCK;
            scales_sh[channel][scales_Y][scales_X] = biased_exponent;
          }

#pragma unroll
          for (int w = 0; w < WAVES; ++w) {
            const size_t staggered_idx = (w + bank_group) % GROUPS;
            const size_t X = tid_X * WAVES + staggered_idx;
            if (channel == 0) {
              ptx::mul_cvt_2x(out_act[buff_out][Y][X], ACT[w], block_scale_inverse_2x);
            } else {
              ptx::mul_cvt_2x(out_gate[buff_out][Y][X], GATE[w], block_scale_inverse_2x);
            }
          }
        }
      }

      // Wait for shared memory writes to be visible to TMA engine.
      ptx::fence_proxy_async_shared_cta();
      __syncthreads();
      // After syncthreads, writes by all threads are visible to TMA engine.

      // Initiate TMA transfer to copy shared memory to global memory
      if (leading_thread) {
        const size_t global_offset_Y = block_offset_Y;
        const size_t global_offset_X = block_offset_X + stage_offset_X;

        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t *>(&tensor_map_output_act_rowwise), global_offset_X,
            global_offset_Y, reinterpret_cast<uint64_t *>(&out_act[buff_out]));
        if constexpr (IS_BWD) {
          ptx::cp_async_bulk_tensor_2d_shared_to_global(
              reinterpret_cast<const uint64_t *>(&tensor_map_output_gate_rowwise), global_offset_X,
              global_offset_Y, reinterpret_cast<uint64_t *>(&out_gate[buff_out]));
        }
        // Create a "bulk async-group" out of the previous bulk copy operation.
        ptx::cp_async_bulk_commit_group();
      }
      buff_in = (buff_in + 1) % BUFFS_NUM;
      buff_out = (buff_out + 1) % BUFFS_NUM;
    }

    // Store of SFs (S2G)
    // Fast vectorized store if SFs are 4-byte aligned. Each thread stores 4x SF (4-byte)
    const bool aligned_SFs = cols % (SCALE_DIM * 4) == 0;
    if (aligned_SFs) {
      constexpr size_t STORES_PER_CHUNK_X = SCALES_PER_CHUNK_X / 4;
      constexpr size_t THREADS_PER_CHANNEL = THREADS_NUM / SF_CHANNELS;
      constexpr size_t rows_per_iteration = THREADS_PER_CHANNEL / STORES_PER_CHUNK_X;
      constexpr size_t iters = DIVUP(CHUNK_DIM_Y, rows_per_iteration);

      const size_t channel = threadIdx.x / THREADS_PER_CHANNEL;
      const size_t tid_Y = (threadIdx.x % THREADS_PER_CHANNEL) / STORES_PER_CHUNK_X;
      const size_t tid_X = (threadIdx.x % THREADS_PER_CHANNEL) % STORES_PER_CHUNK_X;

      const size_t scale_idx_X = scales_block_offset_X + 4 * tid_X;
      const bool col_out_of_bounds = scale_idx_X >= cols / SCALE_DIM;

#pragma unroll
      for (int it = 0; it < iters; ++it) {
        const size_t row = tid_Y + it * rows_per_iteration;
        const size_t scale_idx_Y = scales_block_offset_Y + row;
        const size_t scale_idx =
            scale_idx_Y * scale_stride_rowwise + scale_idx_X + channel * gate_scale_idx_offset;
        const bool row_out_of_bounds = (scale_idx_Y >= rows) || (row >= CHUNK_DIM_Y);
        const bool out_of_bounds = row_out_of_bounds || col_out_of_bounds;
        if (!out_of_bounds) {
          uint32_t *scales_rowwise_4x = reinterpret_cast<uint32_t *>(&scales_rowwise[scale_idx]);
          const uint32_t SF_4x = *reinterpret_cast<uint32_t *>(&scales_sh[channel][row][4 * tid_X]);
          *scales_rowwise_4x = SF_4x;
        }
      }
    } else {
      // Slower scalar store of SFs
      constexpr size_t STORES_PER_CHUNK_X = SCALES_PER_CHUNK_X;
      constexpr size_t THREADS_PER_CHANNEL = THREADS_NUM / SF_CHANNELS;
      constexpr size_t rows_per_iteration = THREADS_PER_CHANNEL / STORES_PER_CHUNK_X;
      constexpr size_t iters = DIVUP(CHUNK_DIM_Y, rows_per_iteration);

      const size_t channel = threadIdx.x / THREADS_PER_CHANNEL;
      const size_t tid_Y = (threadIdx.x % THREADS_PER_CHANNEL) / STORES_PER_CHUNK_X;
      const size_t tid_X = (threadIdx.x % THREADS_PER_CHANNEL) % STORES_PER_CHUNK_X;

      const size_t scale_idx_X = scales_block_offset_X + tid_X;
      const bool col_out_of_bounds = scale_idx_X >= cols / SCALE_DIM;

#pragma unroll
      for (int it = 0; it < iters; ++it) {
        const size_t row = tid_Y + it * rows_per_iteration;
        const size_t scale_idx_Y = scales_block_offset_Y + row;
        const size_t scale_idx =
            scale_idx_Y * scale_stride_rowwise + scale_idx_X + channel * gate_scale_idx_offset;
        const bool row_out_of_bounds = (scale_idx_Y >= rows) || (row >= CHUNK_DIM_Y);
        const bool out_of_bounds = row_out_of_bounds || col_out_of_bounds;
        if (!out_of_bounds) {
          scales_rowwise[scale_idx] = scales_sh[channel][row][tid_X];
        }
      }
    }
    // Guarantees that all threads store scaling factors (S2G) before the data
    // are overwritten in the next iteration
    if (!job_finished) {
      __syncthreads();
    }
  }

  if (leading_thread) {
#pragma unroll
    for (int buff = 0; buff < BUFFS_NUM; ++buff) {
      ptx::mbarrier_invalid(&IN_buff_readable_mbar[buff]);
    }
    ptx::mbarrier_invalid(&workID_mbar);
  }
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}
}  // namespace gated_rowwise_kernel

template <bool IS_BWD, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
void quantize_gated_rowwise(const Tensor &grad, const Tensor &gated_input, Tensor *output,
                            ParamOP &p, cudaStream_t stream) {
  using namespace gated_rowwise_kernel;
  checkCuDriverContext(stream);

  NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated.");

  const size_t rows = gated_input.flat_first_dim();
  const size_t cols = gated_input.flat_last_dim() / 2;
  const size_t output_cols = (IS_BWD ? 2 : 1) * cols;

  const size_t blocks_Y = DIVUP(rows, CHUNK_DIM_Y);
  const size_t blocks_X = DIVUP(cols, CHUNK_DIM_X);

  const dim3 grid(blocks_X, blocks_Y);
  const dim3 block_size(THREADS_NUM);

  size_t scale_stride = output->scale_inv.shape[1];
  e8m0_t *const scales_ptr = reinterpret_cast<e8m0_t *>(output->scale_inv.dptr);

  TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(
      gated_input.dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,

          alignas(64) CUtensorMap tensor_map_grad{};
          alignas(64) CUtensorMap tensor_map_input_act{};
          alignas(64) CUtensorMap tensor_map_input_gate{};
          alignas(64) CUtensorMap tensor_map_output_act{};
          alignas(64) CUtensorMap tensor_map_output_gate{};

          constexpr size_t input_type_bit_size = TypeInfo<IType>::size;
          constexpr size_t output_type_bit_size = TypeInfo<OType>::size;

          if constexpr (IS_BWD) {
            create_2D_tensor_map(tensor_map_grad, grad.data, rows, cols, BUFF_DIM_Y, BUFF_DIM_X,
                                 cols, 0, input_type_bit_size);
          }

          const uint32_t tensor_stride_elems = output_cols;
          create_2D_tensor_map(tensor_map_input_act, gated_input.data, rows, cols, BUFF_DIM_Y,
                               BUFF_DIM_X, cols * 2, 0, input_type_bit_size);
          create_2D_tensor_map(tensor_map_input_gate, gated_input.data, rows, cols, BUFF_DIM_Y,
                               BUFF_DIM_X, cols * 2, cols, input_type_bit_size);

          create_2D_tensor_map(tensor_map_output_act, output->data, rows, cols, BUFF_DIM_Y,
                               BUFF_DIM_X, tensor_stride_elems, 0, output_type_bit_size);
          create_2D_tensor_map(tensor_map_output_gate, output->data, rows, cols, BUFF_DIM_Y,
                               BUFF_DIM_X, tensor_stride_elems, cols, output_type_bit_size);

          const size_t buff_elems_total = BUFFS_NUM * BUFF_DIM_Y * BUFF_DIM_X;
          const size_t input_buff_size = (buff_elems_total * input_type_bit_size) / 8;
          const size_t output_buff_size = (buff_elems_total * output_type_bit_size) / 8;
          const size_t buff_size_aligned_in =
              DIVUP_TO_MULTIPLE(input_buff_size, TMA_SHMEM_ALIGNMENT);
          const size_t buff_size_aligned_out =
              DIVUP_TO_MULTIPLE(output_buff_size, TMA_SHMEM_ALIGNMENT);

          const size_t grad_mem = (IS_BWD ? buff_size_aligned_in : 0);
          const size_t in_act_mem = buff_size_aligned_in;
          const size_t in_gate_mem = buff_size_aligned_in;
          const size_t in_mem = grad_mem + in_act_mem + in_gate_mem;

          const size_t out_act_mem = buff_size_aligned_out;
          const size_t out_gate_mem = (IS_BWD ? buff_size_aligned_out : 0);
          size_t out_mem = out_act_mem + out_gate_mem;

          const size_t shmem_size = in_mem + out_mem + TMA_SHMEM_ALIGNMENT;

          auto kernel =
              quantize_gated_mxfp8_rowwise_kernel<IS_BWD, ParamOP, ActOP, DActOP, IType, OType>;
          NVTE_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                               shmem_size));

          kernel<<<grid, block_size, shmem_size, stream>>>(
              tensor_map_grad, tensor_map_input_act, tensor_map_input_gate, tensor_map_output_act,
              tensor_map_output_gate, scales_ptr, rows, cols, scale_stride, p);

          NVTE_CHECK_CUDA(cudaGetLastError()););  // NOLINT(*)
  );                                              // NOLINT(*)
}

}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_SPECIALIZED_GATED_MXFP8_ROWWISE_SWIGLU_CUH_
