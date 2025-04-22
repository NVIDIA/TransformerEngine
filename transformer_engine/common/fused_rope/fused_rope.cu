/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <assert.h>
#include <cuda_runtime.h>
#include <transformer_engine/fused_rope.h>

#include "../common.h"
#include "../util/logging.h"
#include "../utils.cuh"

namespace transformer_engine {

template <typename scalar_t>
__device__ void fused_rope_block_forward(const scalar_t *src, const float *freqs, scalar_t *dst,
                                         const bool interleaved, const int s_id,
                                         const int offset_block, const int offset_block_dst,
                                         const int h, const int d, const int d2, const int stride_h,
                                         const int stride_d, const int o_stride_h,
                                         const int o_stride_d) {
#pragma unroll
  for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
    float v_cos, v_sin;
    sincosf(freqs[s_id * d2 + d_id], &v_sin, &v_cos);
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      float v_src = src[offset_src];
      float v_src_rotate;
      if (!interleaved) {
        v_src_rotate = (d_id + d2 / 2 < d2)
                           ? -static_cast<float>(src[offset_src + (d2 / 2) * stride_d])
                           : static_cast<float>(src[offset_src + (d2 / 2 - d2) * stride_d]);
      } else {
        v_src_rotate = (d_id % 2 == 0)
                           // d_id + 1
                           ? -static_cast<float>(src[offset_src + stride_d])
                           // d_id - 1
                           : static_cast<float>(src[offset_src - stride_d]);
      }
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
        dst[offset_head_dst + d_id * o_stride_d] = src[offset_head + d_id * stride_d];
      }
    }
  }
}

template <typename scalar_t>
__device__ void fused_rope_block_backward(const scalar_t *src, const float *freqs, scalar_t *dst,
                                          const bool interleaved, const int s_id,
                                          const int offset_block, const int offset_block_dst,
                                          const int h, const int d, const int d2,
                                          const int stride_h, const int stride_d,
                                          const int o_stride_h, const int o_stride_d) {
#pragma unroll
  for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
    float v_cos = cosf(freqs[s_id * d2 + d_id]);
    float v_sin;
    if (!interleaved) {
      v_sin = (d_id + d2 / 2 < d2) ? sinf(freqs[s_id * d2 + d_id + d2 / 2])
                                   : -sinf(freqs[s_id * d2 + d_id + d2 / 2 - d2]);
    } else {
      v_sin =
          (d_id % 2 == 0) ? sinf(freqs[s_id * d2 + d_id + 1]) : -sinf(freqs[s_id * d2 + d_id - 1]);
    }
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      float v_src = src[offset_src];
      float v_src_rotate;
      if (!interleaved) {
        v_src_rotate = (d_id + d2 / 2 < d2)
                           ? static_cast<float>(src[offset_src + (d2 / 2) * stride_d])
                           : static_cast<float>(src[offset_src + (d2 / 2 - d2) * stride_d]);
      } else {
        v_src_rotate = (d_id % 2 == 0)
                           // d_id + 1
                           ? static_cast<float>(src[offset_src + stride_d])
                           // d_id - 1
                           : static_cast<float>(src[offset_src - stride_d]);
      }
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
        dst[offset_head_dst + d_id * o_stride_d] = src[offset_head + d_id * stride_d];
      }
    }
  }
}

template <typename scalar_t>
__global__ void fused_rope_forward_kernel(const scalar_t *src, const int *cu_seqlens,
                                          const float *freqs, const int *start_positions,
                                          scalar_t *dst, const bool interleaved, const int cp_size,
                                          const int cp_rank, const int s, const int h, const int d,
                                          const int d2, const int stride_s_or_t, const int stride_b,
                                          const int stride_h, const int stride_d,
                                          const int o_stride_s_or_t, const int o_stride_b,
                                          const int o_stride_h, const int o_stride_d) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int offset_block, offset_block_dst;
  int cur_seqlens;
  if (cu_seqlens != nullptr) {  // THD
    int start = cu_seqlens[b_id] / cp_size;
    int end = cu_seqlens[b_id + 1] / cp_size;
    int t_id = s_id + start;
    if (t_id >= end) return;
    offset_block = t_id * stride_s_or_t;
    offset_block_dst = t_id * o_stride_s_or_t;
    cur_seqlens = end - start;
  } else {  // SBHD/BSHD
    offset_block = s_id * stride_s_or_t + b_id * stride_b;
    offset_block_dst = s_id * o_stride_s_or_t + b_id * o_stride_b;
    cur_seqlens = s;
  }

  int s_id_for_freqs;
  if (cp_size > 1) {
    assert(cur_seqlens % 2 == 0);
    if (s_id < cur_seqlens / 2) {
      s_id_for_freqs = s_id + cp_rank * cur_seqlens / 2;
    } else {
      s_id_for_freqs =
          cur_seqlens * cp_size - (cp_rank + 1) * cur_seqlens / 2 + s_id - cur_seqlens / 2;
    }
  } else {
    int begin_offset = (start_positions == nullptr) ? 0 : start_positions[b_id];
    s_id_for_freqs = s_id + begin_offset;
  }

  fused_rope_block_forward(src, freqs, dst, interleaved, s_id_for_freqs, offset_block,
                           offset_block_dst, h, d, d2, stride_h, stride_d, o_stride_h, o_stride_d);
}

template <typename scalar_t>
__global__ void fused_rope_backward_kernel(
    const scalar_t *src, const int *cu_seqlens, const float *freqs, scalar_t *dst,
    const bool interleaved, const int cp_size, const int cp_rank, const int s, const int h,
    const int d, const int d2, const int stride_s_or_t, const int stride_b, const int stride_h,
    const int stride_d, const int o_stride_s_or_t, const int o_stride_b, const int o_stride_h,
    const int o_stride_d) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int offset_block, offset_block_dst;
  int cur_seqlens;
  if (cu_seqlens != nullptr) {  // THD
    int start = cu_seqlens[b_id] / cp_size;
    int end = cu_seqlens[b_id + 1] / cp_size;
    int t_id = s_id + start;
    if (t_id >= end) return;
    offset_block = t_id * stride_s_or_t;
    offset_block_dst = t_id * o_stride_s_or_t;
    cur_seqlens = end - start;
  } else {  // SBHD/BSHD
    offset_block = s_id * stride_s_or_t + b_id * stride_b;
    offset_block_dst = s_id * o_stride_s_or_t + b_id * o_stride_b;
    cur_seqlens = s;
  }

  int s_id_for_freqs;
  if (cp_size > 1) {
    assert(cur_seqlens % 2 == 0);
    if (s_id < cur_seqlens / 2) {
      s_id_for_freqs = s_id + cp_rank * cur_seqlens / 2;
    } else {
      s_id_for_freqs =
          cur_seqlens * cp_size - (cp_rank + 1) * cur_seqlens / 2 + s_id - cur_seqlens / 2;
    }
  } else {
    s_id_for_freqs = s_id;
  }

  fused_rope_block_backward(src, freqs, dst, interleaved, s_id_for_freqs, offset_block,
                            offset_block_dst, h, d, d2, stride_h, stride_d, o_stride_h, o_stride_d);
}

template <typename scalar_t>
void fused_rope_forward_launcher(const scalar_t *input, const int *cu_seqlens, const float *freqs,
                                 const int *start_positions, scalar_t *output,
                                 const NVTE_QKV_Format qkv_format, const bool interleaved,
                                 const int cp_size, const int cp_rank, const int s, const int b,
                                 const int h, const int d, const int d2, const int stride_s_or_t,
                                 const int stride_b, const int stride_h, const int stride_d,
                                 cudaStream_t stream) {
  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(s, b);
  dim3 threads(THREADS_PER_WARP, warps_per_block);
  int o_stride_s_or_t, o_stride_b;
  if (qkv_format == NVTE_QKV_Format::NVTE_THD) {
    NVTE_CHECK(cu_seqlens != nullptr, "cu_seqlens is required for THD format");
    o_stride_s_or_t = h * d;
    o_stride_b = 0;
  } else if (qkv_format == NVTE_QKV_Format::NVTE_SBHD) {
    o_stride_s_or_t = b * h * d;
    o_stride_b = h * d;
  } else {
    o_stride_s_or_t = h * d;
    o_stride_b = s * h * d;
  }
  const int o_stride_h = d;
  const int o_stride_d = 1;

  fused_rope_forward_kernel<<<blocks, threads, 0, stream>>>(
      input, cu_seqlens, freqs, start_positions, output, interleaved, cp_size, cp_rank, s, h, d, d2,
      stride_s_or_t, stride_b, stride_h, stride_d, o_stride_s_or_t, o_stride_b, o_stride_h,
      o_stride_d);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename scalar_t>
void fused_rope_backward_launcher(const scalar_t *output_grads, const int *cu_seqlens,
                                  const float *freqs, scalar_t *input_grads,
                                  const NVTE_QKV_Format qkv_format, const bool interleaved,
                                  const int cp_size, const int cp_rank, const int s, const int b,
                                  const int h, const int d, const int d2, const int stride_s_or_t,
                                  const int stride_b, const int stride_h, const int stride_d,
                                  cudaStream_t stream) {
  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(s, b);
  dim3 threads(THREADS_PER_WARP, warps_per_block);
  int o_stride_s_or_t, o_stride_b;
  if (qkv_format == NVTE_QKV_Format::NVTE_THD) {
    NVTE_CHECK(cu_seqlens != nullptr, "cu_seqlens is required for THD format");
    o_stride_s_or_t = h * d;
    o_stride_b = 0;
  } else if (qkv_format == NVTE_QKV_Format::NVTE_SBHD) {
    o_stride_s_or_t = b * h * d;
    o_stride_b = h * d;
  } else {
    o_stride_s_or_t = h * d;
    o_stride_b = s * h * d;
  }
  const int o_stride_h = d;
  const int o_stride_d = 1;

  fused_rope_backward_kernel<<<blocks, threads, 0, stream>>>(
      output_grads, cu_seqlens, freqs, input_grads, interleaved, cp_size, cp_rank, s, h, d, d2,
      stride_s_or_t, stride_b, stride_h, stride_d, o_stride_s_or_t, o_stride_b, o_stride_h,
      o_stride_d);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void fused_rope_forward(const Tensor &input, const Tensor &cu_seqlens, const Tensor &freqs,
                        const Tensor &start_positions, Tensor *output,
                        const NVTE_QKV_Format qkv_format, const bool interleaved, const int cp_size,
                        const int cp_rank, const int s, const int b, const int h, const int d,
                        const int d2, const int stride_s_or_t, const int stride_b,
                        const int stride_h, const int stride_d, cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.data.dtype, scalar_t,
      fused_rope_forward_launcher(reinterpret_cast<const scalar_t *>(input.data.dptr),
                                  reinterpret_cast<const int *>(cu_seqlens.data.dptr),
                                  reinterpret_cast<const float *>(freqs.data.dptr),
                                  reinterpret_cast<const int *>(start_positions.data.dptr),
                                  reinterpret_cast<scalar_t *>(output->data.dptr), qkv_format,
                                  interleaved, cp_size, cp_rank, s, b, h, d, d2, stride_s_or_t,
                                  stride_b, stride_h, stride_d, stream););
}

void fused_rope_backward(const Tensor &output_grads, const Tensor &cu_seqlens, const Tensor &freqs,
                         Tensor *input_grads, const NVTE_QKV_Format qkv_format,
                         const bool interleaved, const int cp_size, const int cp_rank, const int s,
                         const int b, const int h, const int d, const int d2,
                         const int stride_s_or_t, const int stride_b, const int stride_h,
                         const int stride_d, cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      output_grads.data.dtype, scalar_t,
      fused_rope_backward_launcher(reinterpret_cast<const scalar_t *>(output_grads.data.dptr),
                                   reinterpret_cast<const int *>(cu_seqlens.data.dptr),
                                   reinterpret_cast<const float *>(freqs.data.dptr),
                                   reinterpret_cast<scalar_t *>(input_grads->data.dptr), qkv_format,
                                   interleaved, cp_size, cp_rank, s, b, h, d, d2, stride_s_or_t,
                                   stride_b, stride_h, stride_d, stream););
}

}  // end namespace transformer_engine

void nvte_fused_rope_forward(const NVTETensor input, const NVTETensor cu_seqlens,
                             const NVTETensor freqs, const NVTETensor start_positions,
                             NVTETensor output, const NVTE_QKV_Format qkv_format,
                             const bool interleaved, const int cp_size, const int cp_rank,
                             const int s, const int b, const int h, const int d, const int d2,
                             const int stride_s_or_t, const int stride_b, const int stride_h,
                             const int stride_d, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_rope_forward);
  using namespace transformer_engine;
  fused_rope_forward(
      *reinterpret_cast<const Tensor *>(input), *reinterpret_cast<const Tensor *>(cu_seqlens),
      *reinterpret_cast<const Tensor *>(freqs), *reinterpret_cast<const Tensor *>(start_positions),
      reinterpret_cast<Tensor *>(output), qkv_format, interleaved, cp_size, cp_rank, s, b, h, d, d2,
      stride_s_or_t, stride_b, stride_h, stride_d, stream);
}

void nvte_fused_rope_backward(const NVTETensor output_grads, const NVTETensor cu_seqlens,
                              const NVTETensor freqs, NVTETensor input_grads,
                              const NVTE_QKV_Format qkv_format, const bool interleaved,
                              const int cp_size, const int cp_rank, const int s, const int b,
                              const int h, const int d, const int d2, const int stride_s_or_t,
                              const int stride_b, const int stride_h, const int stride_d,
                              cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_rope_backward);
  using namespace transformer_engine;
  fused_rope_backward(*reinterpret_cast<const Tensor *>(output_grads),
                      *reinterpret_cast<const Tensor *>(cu_seqlens),
                      *reinterpret_cast<const Tensor *>(freqs),
                      reinterpret_cast<Tensor *>(input_grads), qkv_format, interleaved, cp_size,
                      cp_rank, s, b, h, d, d2, stride_s_or_t, stride_b, stride_h, stride_d, stream);
}
