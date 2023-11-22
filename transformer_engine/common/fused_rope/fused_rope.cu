/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <transformer_engine/fused_rope.h>

#include "../common.h"
#include "../util/logging.h"
#include "../utils.cuh"

namespace transformer_engine {

template <typename scalar_t>
__global__ void fused_rope_forward_kernel(
    const scalar_t *src, const scalar_t *cos, const scalar_t *sin,
    scalar_t *dst, const int s, const int b, const int h, const int d,
    const int d2, const int stride_s, const int stride_b, const int stride_h,
    const int stride_d, const int o_stride_s, const int o_stride_b,
    const int o_stride_h, const int o_stride_d) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int offset_block = s_id * stride_s + b_id * stride_b;
  int offset_block_dst = s_id * o_stride_s + b_id * o_stride_b;
#pragma unroll
  for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
    scalar_t v_cos = cos[s_id * d2 + d_id];
    scalar_t v_sin = sin[s_id * d2 + d_id];
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      scalar_t v_src = src[offset_src];
      scalar_t v_src_rotate = (d_id + d2 / 2 < d2)
                                  ? -src[offset_src + (d2 / 2) * stride_d]
                                  : src[offset_src + (d2 / 2 - d2) * stride_d];
      dst[offset_dst] = v_src * v_cos + v_src_rotate * v_sin;
    }
  }

  // copy the rest
  if (d > d2) {
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_head = offset_block + h_id * stride_h;
      int offset_head_dst = offset_block_dst + h_id * o_stride_h;
#pragma unroll
      for (int d_id = d2 + threadIdx.x; d_id < d; d_id += blockDim.x) {
        dst[offset_head_dst + d_id * o_stride_d] =
            src[offset_head + d_id * stride_d];
      }
    }
  }
}

template <typename scalar_t>
__global__ void fused_rope_backward_kernel(
    const scalar_t *src, const scalar_t *cos, const scalar_t *sin,
    scalar_t *dst, const int s, const int b, const int h, const int d,
    const int d2, const int stride_s, const int stride_b, const int stride_h,
    const int stride_d, const int o_stride_s, const int o_stride_b,
    const int o_stride_h, const int o_stride_d) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int offset_block = s_id * stride_s + b_id * stride_b;
  int offset_block_dst = s_id * o_stride_s + b_id * o_stride_b;
#pragma unroll
  for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
    scalar_t v_cos = cos[s_id * d2 + d_id];
    scalar_t v_sin = (d_id + d2 / 2 < d2)
                         ? sin[s_id * d2 + d_id + d2 / 2]
                         : -sin[s_id * d2 + d_id + d2 / 2 - d2];
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      scalar_t v_src = src[offset_src];
      scalar_t v_src_rotate = (d_id + d2 / 2 < d2)
                                  ? src[offset_src + (d2 / 2) * stride_d]
                                  : src[offset_src + (d2 / 2 - d2) * stride_d];
      dst[offset_dst] = v_src * v_cos + v_src_rotate * v_sin;
    }
  }

  // handle the tail
  if (d > d2) {
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_head = offset_block + h_id * stride_h;
      int offset_head_dst = offset_block_dst + h_id * o_stride_h;
#pragma unroll
      for (int d_id = d2 + threadIdx.x; d_id < d; d_id += blockDim.x) {
        dst[offset_head_dst + d_id * o_stride_d] =
            src[offset_head + d_id * stride_d];
      }
    }
  }
}

template <typename scalar_t>
void fused_rope_forward_launcher(const scalar_t *input, const scalar_t *cos,
                                 const scalar_t *sin, scalar_t *output,
                                 const int s, const int b, const int h,
                                 const int d, const int d2, const int stride_s,
                                 const int stride_b, const int stride_h,
                                 const int stride_d, const int o_stride_s,
                                 const int o_stride_b, const int o_stride_h,
                                 const int o_stride_d, cudaStream_t stream) {
  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(s, b);
  dim3 threads(THREADS_PER_WARP, warps_per_block);

  fused_rope_forward_kernel<<<blocks, threads, 0, stream>>>(
      input, cos, sin, output, s, b, h, d, d2, stride_s, stride_b, stride_h,
      stride_d, o_stride_s, o_stride_b, o_stride_h, o_stride_d);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename scalar_t>
void fused_rope_backward_launcher(
    const scalar_t *incoming_grads, const scalar_t *cos, const scalar_t *sin,
    scalar_t *output_grads, const int s, const int b, const int h, const int d,
    const int d2, const int stride_s, const int stride_b, const int stride_h,
    const int stride_d, const int o_stride_s, const int o_stride_b,
    const int o_stride_h, const int o_stride_d, cudaStream_t stream) {
  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(s, b);
  dim3 threads(THREADS_PER_WARP, warps_per_block);

  fused_rope_backward_kernel<<<blocks, threads, 0, stream>>>(
      incoming_grads, cos, sin, output_grads, s, b, h, d, d2, stride_s,
      stride_b, stride_h, stride_d, o_stride_s, o_stride_b, o_stride_h,
      o_stride_d);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void fused_rope_forward(const Tensor &input, const Tensor &cos,
                        const Tensor &sin, Tensor *output, const int s,
                        const int b, const int h, const int d, const int d2,
                        const int stride_s, const int stride_b,
                        const int stride_h, const int stride_d,
                        const int o_stride_s, const int o_stride_b,
                        const int o_stride_h, const int o_stride_d,
                        cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.data.dtype, scalar_t,
      fused_rope_forward_launcher(
          reinterpret_cast<const scalar_t *>(input.data.dptr),
          reinterpret_cast<const scalar_t *>(cos.data.dptr),
          reinterpret_cast<const scalar_t *>(sin.data.dptr),
          reinterpret_cast<scalar_t *>(output->data.dptr), s, b, h, d, d2,
          stride_s, stride_b, stride_h, stride_d, o_stride_s, o_stride_b,
          o_stride_h, o_stride_d, stream););
}

void fused_rope_backward(const Tensor &incoming_grads, const Tensor &cos,
                         const Tensor &sin, Tensor *output_grads, const int s,
                         const int b, const int h, const int d, const int d2,
                         const int stride_s, const int stride_b,
                         const int stride_h, const int stride_d,
                         const int o_stride_s, const int o_stride_b,
                         const int o_stride_h, const int o_stride_d,
                         cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      incoming_grads.data.dtype, scalar_t,
      fused_rope_backward_launcher(
          reinterpret_cast<const scalar_t *>(incoming_grads.data.dptr),
          reinterpret_cast<const scalar_t *>(cos.data.dptr),
          reinterpret_cast<const scalar_t *>(sin.data.dptr),
          reinterpret_cast<scalar_t *>(output_grads->data.dptr), s, b, h, d, d2,
          stride_s, stride_b, stride_h, stride_d, o_stride_s, o_stride_b,
          o_stride_h, o_stride_d, stream););
}

}  // end namespace transformer_engine

void nvte_fused_rope_forward(const NVTETensor input, const NVTETensor cos,
                             const NVTETensor sin, NVTETensor output,
                             const int s, const int b, const int h, const int d,
                             const int d2, const int stride_s,
                             const int stride_b, const int stride_h,
                             const int stride_d, const int o_stride_s,
                             const int o_stride_b, const int o_stride_h,
                             const int o_stride_d, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_rope_forward);
  using namespace transformer_engine;
  fused_rope_forward(*reinterpret_cast<const Tensor *>(input),
                     *reinterpret_cast<const Tensor *>(cos),
                     *reinterpret_cast<const Tensor *>(sin),
                     reinterpret_cast<Tensor *>(output), s, b, h, d, d2,
                     stride_s, stride_b, stride_h, stride_d, o_stride_s,
                     o_stride_b, o_stride_h, o_stride_d, stream);
}

void nvte_fused_rope_backward(
    const NVTETensor incoming_grads, const NVTETensor cos, const NVTETensor sin,
    NVTETensor output_grads, const int s, const int b, const int h, const int d,
    const int d2, const int stride_s, const int stride_b, const int stride_h,
    const int stride_d, const int o_stride_s, const int o_stride_b,
    const int o_stride_h, const int o_stride_d, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_rope_backward);
  using namespace transformer_engine;
  fused_rope_backward(*reinterpret_cast<const Tensor *>(incoming_grads),
                      *reinterpret_cast<const Tensor *>(cos),
                      *reinterpret_cast<const Tensor *>(sin),
                      reinterpret_cast<Tensor *>(output_grads), s, b, h, d, d2,
                      stride_s, stride_b, stride_h, stride_d, o_stride_s,
                      o_stride_b, o_stride_h, o_stride_d, stream);
}
