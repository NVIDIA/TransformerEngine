/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <transformer_engine/fused_rope.h>

#include "../common.h"
#include "../util/logging.h"

namespace transformer_engine {

template <typename scalar_t>
__global__ void fused_rope_forward_kernel(int sq, int b, int np, int hn,
                                          int hn2, const scalar_t *src,
                                          const scalar_t *cos,
                                          const scalar_t *sin, scalar_t *dst) {
  int sq_id = blockIdx.x, b_id = blockIdx.y;
  int offset_block = sq_id * b * np * hn + b_id * np * hn;
#pragma unroll
  for (int hn_id = threadIdx.x; hn_id < hn2; hn_id += blockDim.x) {
    scalar_t v_cos = cos[sq_id * hn2 + hn_id];
    scalar_t v_sin = sin[sq_id * hn2 + hn_id];
#pragma unroll
    for (int head_id = threadIdx.y; head_id < np; head_id += blockDim.y) {
      int offset_src_dst = offset_block + head_id * hn + hn_id;
      scalar_t v_src = src[offset_src_dst];
      scalar_t v_src_rotate = (hn_id + hn2 / 2 < hn2)
                                  ? -src[offset_src_dst + hn2 / 2]
                                  : src[offset_src_dst + hn2 / 2 - hn2];
      dst[offset_src_dst] = v_src * v_cos + v_src_rotate * v_sin;
    }
  }

  // copy the rest
  if (hn > hn2) {
#pragma unroll
    for (int head_id = threadIdx.y; head_id < np; head_id += blockDim.y) {
      int offset_head = offset_block + head_id * hn;
#pragma unroll
      for (int hn_id = hn2 + threadIdx.x; hn_id < hn; hn_id += blockDim.x) {
        int offset_src_dst = offset_head + hn_id;
        dst[offset_src_dst] = src[offset_src_dst];
      }
    }
  }
}

template <typename scalar_t>
__global__ void fused_rope_backward_kernel(int sq, int b, int np, int hn,
                                           int hn2, const scalar_t *src,
                                           const scalar_t *cos,
                                           const scalar_t *sin, scalar_t *dst) {
  int sq_id = blockIdx.x, b_id = blockIdx.y;
  int offset_block = sq_id * b * np * hn + b_id * np * hn;
#pragma unroll
  for (int hn_id = threadIdx.x; hn_id < hn2; hn_id += blockDim.x) {
    scalar_t v_cos = cos[sq_id * hn2 + hn_id];
    scalar_t v_sin = (hn_id + hn2 / 2 < hn2)
                         ? sin[sq_id * hn2 + hn_id + hn2 / 2]
                         : -sin[sq_id * hn2 + hn_id + hn2 / 2 - hn2];
#pragma unroll
    for (int head_id = threadIdx.y; head_id < np; head_id += blockDim.y) {
      int offset_src_dst = offset_block + head_id * hn + hn_id;
      scalar_t v_src = src[offset_src_dst];
      scalar_t v_src_rotate = (hn_id + hn2 / 2 < hn2)
                                  ? src[offset_src_dst + hn2 / 2]
                                  : src[offset_src_dst + hn2 / 2 - hn2];
      dst[offset_src_dst] = v_src * v_cos + v_src_rotate * v_sin;
    }
  }

  // handle the tail
  if (hn > hn2) {
#pragma unroll
    for (int head_id = threadIdx.y; head_id < np; head_id += blockDim.y) {
      int offset_head = offset_block + head_id * hn;
#pragma unroll
      for (int hn_id = hn2 + threadIdx.x; hn_id < hn; hn_id += blockDim.x) {
        dst[offset_head + hn_id] = 1.0;
      }
    }
  }
}

template <typename scalar_t>
void fused_rope_forward_launcher(int sq, int b, int np, int hn, int hn2,
                                 const scalar_t *input, const scalar_t *cos,
                                 const scalar_t *sin, scalar_t *output,
                                 cudaStream_t stream) {
  int warps_per_block = np < 16 ? 4 : 8;
  dim3 blocks(sq, b);
  dim3 threads(THREADS_PER_WARP, warps_per_block);

  fused_rope_forward_kernel<<<blocks, threads, 0, stream>>>(
      sq, b, np, hn, hn2, input, cos, sin, output);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename scalar_t>
void fused_rope_backward_launcher(int sq, int b, int np, int hn, int hn2,
                                  const scalar_t *incoming_grads,
                                  const scalar_t *cos, const scalar_t *sin,
                                  scalar_t *output_grads, cudaStream_t stream) {
  int warps_per_block = np < 16 ? 4 : 8;
  dim3 blocks(sq, b);
  dim3 threads(THREADS_PER_WARP, warps_per_block);

  fused_rope_backward_kernel<<<blocks, threads, 0, stream>>>(
      sq, b, np, hn, hn2, incoming_grads, cos, sin, output_grads);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void fused_rope_forward(const Tensor &input, const Tensor &cos,
                        const Tensor &sin, Tensor *output,
                        cudaStream_t stream) {
  const int sq = input.data.shape[0];
  const int b = input.data.shape[1];
  const int np = input.data.shape[2];
  const int hn = input.data.shape[3];
  const int hn2 = cos.data.shape[3];

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.data.dtype, scalar_t,
      fused_rope_forward_launcher(
          sq, b, np, hn, hn2,
          reinterpret_cast<const scalar_t *>(input.data.dptr),
          reinterpret_cast<const scalar_t *>(cos.data.dptr),
          reinterpret_cast<const scalar_t *>(sin.data.dptr),
          reinterpret_cast<scalar_t *>(output->data.dptr), stream););
}

void fused_rope_backward(const Tensor &incoming_grads, const Tensor &cos,
                         const Tensor &sin, Tensor *output_grads,
                         cudaStream_t stream) {
  const int sq = incoming_grads.data.shape[0];
  const int b = incoming_grads.data.shape[1];
  const int np = incoming_grads.data.shape[2];
  const int hn = incoming_grads.data.shape[3];
  const int hn2 = cos.data.shape[3];

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      incoming_grads.data.dtype, scalar_t,
      fused_rope_backward_launcher(
          sq, b, np, hn, hn2,
          reinterpret_cast<const scalar_t *>(incoming_grads.data.dptr),
          reinterpret_cast<const scalar_t *>(cos.data.dptr),
          reinterpret_cast<const scalar_t *>(sin.data.dptr),
          reinterpret_cast<scalar_t *>(output_grads->data.dptr), stream););
}

}  // end namespace transformer_engine

void nvte_fused_rope_forward(const NVTETensor input, const NVTETensor cos,
                             const NVTETensor sin, NVTETensor output,
                             cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_rope_forward);
  using namespace transformer_engine;
  fused_rope_forward(*reinterpret_cast<const Tensor *>(input),
                     *reinterpret_cast<const Tensor *>(cos),
                     *reinterpret_cast<const Tensor *>(sin),
                     reinterpret_cast<Tensor *>(output), stream);
}

void nvte_fused_rope_backward(const NVTETensor incoming_grads,
                              const NVTETensor cos, const NVTETensor sin,
                              NVTETensor output_grads, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_rope_backward);
  using namespace transformer_engine;
  fused_rope_backward(*reinterpret_cast<const Tensor *>(incoming_grads),
                      *reinterpret_cast<const Tensor *>(cos),
                      *reinterpret_cast<const Tensor *>(sin),
                      reinterpret_cast<Tensor *>(output_grads), stream);
}
