/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  extern __shared__ float shared_mem_cos_sin[];
  float *shared_mem_cos = shared_mem_cos_sin;
  float *shared_mem_sin = shared_mem_cos_sin + d2;
  int tid = threadIdx.x * blockDim.y + threadIdx.y;
  for (int i = tid; i < d2; i += blockDim.x * blockDim.y) {
    sincosf(freqs[s_id * d2 + i], &shared_mem_sin[i], &shared_mem_cos[i]);
  }
  __syncthreads();

#pragma unroll
  for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
#pragma unroll
    for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
      float v_cos = shared_mem_cos[d_id];
      float v_sin = shared_mem_sin[d_id];
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
    for (int d_id = d2 + threadIdx.x; d_id < d; d_id += blockDim.x) {
#pragma unroll
      for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
        int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
        int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
        dst[offset_dst] = src[offset_src];
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
  extern __shared__ float shared_mem_cos_sin[];
  float *shared_mem_cos = shared_mem_cos_sin;
  float *shared_mem_sin = shared_mem_cos_sin + d2;
  int tid = threadIdx.x * blockDim.y + threadIdx.y;
  for (int i = tid; i < d2; i += blockDim.x * blockDim.y) {
    sincosf(freqs[s_id * d2 + i], &shared_mem_sin[i], &shared_mem_cos[i]);
  }
  __syncthreads();

#pragma unroll
  for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
#pragma unroll
    for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
      int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      float v_src = src[offset_src];
      float v_cos = shared_mem_cos[d_id];
      float v_src_rotate, v_sin;
      if (!interleaved) {
        if (d_id + d2 / 2 < d2) {
          v_src_rotate = static_cast<float>(src[offset_src + (d2 / 2) * stride_d]);
          v_sin = shared_mem_sin[d_id + d2 / 2];
        } else {
          v_src_rotate = static_cast<float>(src[offset_src + (d2 / 2 - d2) * stride_d]);
          v_sin = -shared_mem_sin[d_id + d2 / 2 - d2];
        }
      } else {
        if (d_id % 2 == 0) {
          v_src_rotate = static_cast<float>(src[offset_src + stride_d]);
          v_sin = shared_mem_sin[d_id + 1];
        } else {
          v_src_rotate = static_cast<float>(src[offset_src - stride_d]);
          v_sin = -shared_mem_sin[d_id - 1];
        }
      }
      dst[offset_dst] = v_src * v_cos + v_src_rotate * v_sin;
    }
  }

  // copy the rest
  if (d > d2) {
#pragma unroll
    for (int d_id = d2 + threadIdx.x; d_id < d; d_id += blockDim.x) {
#pragma unroll
      for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
        int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
        int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
        dst[offset_dst] = src[offset_src];
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

  // Offset the RoPE embedding by start_positions if provided.
  int begin_offset = (start_positions == nullptr) ? 0 : start_positions[b_id];
  int s_id_for_freqs = s_id + begin_offset;

  // If CP_SIZE > 1, offset the RoPE embedding by cp_rank based on the dual-chunk order.
  if (cp_size > 1) {
    assert(cur_seqlens % 2 == 0);
    if (s_id < cur_seqlens / 2) {
      s_id_for_freqs += cp_rank * cur_seqlens / 2;
    } else {
      s_id_for_freqs += cur_seqlens * cp_size - (cp_rank + 1) * cur_seqlens / 2 - cur_seqlens / 2;
    }
  }

  fused_rope_block_forward(src, freqs, dst, interleaved, s_id_for_freqs, offset_block,
                           offset_block_dst, h, d, d2, stride_h, stride_d, o_stride_h, o_stride_d);
}

template <typename scalar_t>
__global__ void fused_rope_backward_kernel(
    const scalar_t *src, const int *cu_seqlens, const float *freqs, const int *start_positions,
    scalar_t *dst, const bool interleaved, const int cp_size, const int cp_rank, const int s,
    const int h, const int d, const int d2, const int stride_s_or_t, const int stride_b,
    const int stride_h, const int stride_d, const int o_stride_s_or_t, const int o_stride_b,
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

  // Offset the RoPE embedding by start_positions if provided.
  int begin_offset = (start_positions == nullptr) ? 0 : start_positions[b_id];
  int s_id_for_freqs = s_id + begin_offset;

  // If CP_SIZE > 1, offset the RoPE embedding by cp_rank based on the dual-chunk order.
  if (cp_size > 1) {
    assert(cur_seqlens % 2 == 0);
    if (s_id < cur_seqlens / 2) {
      s_id_for_freqs += cp_rank * cur_seqlens / 2;
    } else {
      s_id_for_freqs += cur_seqlens * cp_size - (cp_rank + 1) * cur_seqlens / 2 - cur_seqlens / 2;
    }
  }

  fused_rope_block_backward(src, freqs, dst, interleaved, s_id_for_freqs, offset_block,
                            offset_block_dst, h, d, d2, stride_h, stride_d, o_stride_h, o_stride_d);
}

template <typename scalar_t>
__device__ void fused_qkv_rope_block_forward(const scalar_t *src, const float *freqs, scalar_t *out,
                                             const bool interleaved, const int s_id,
                                             const int offset_block, const int offset_block_dst,
                                             const int h, const int d, const int d2,
                                             const int row_offset, const int in_row_length,
                                             const int out_row_length) {
  extern __shared__ float shared_mem_cos_sin_qk[];
  // Split the shared memory into cos and sin parts for q or k
  float *shared_mem_cos = nullptr;
  float *shared_mem_sin = nullptr;
  if (row_offset == 0) {  // q
    shared_mem_cos = shared_mem_cos_sin_qk;
    shared_mem_sin = shared_mem_cos_sin_qk + d2;
  } else {  // k
    shared_mem_cos = shared_mem_cos_sin_qk + 2 * d2;
    shared_mem_sin = shared_mem_cos_sin_qk + 3 * d2;
  }
  if (freqs != nullptr) {
    int tid = threadIdx.x * blockDim.y + threadIdx.y;
    for (int i = tid; i < d2; i += blockDim.x * blockDim.y) {
      sincosf(freqs[s_id * d2 + i], &shared_mem_sin[i], &shared_mem_cos[i]);
    }
  }
  __syncthreads();

#pragma unroll
  for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
#pragma unroll
    for (int i = 0; i < out_row_length; i += d) {
#pragma unroll
      for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
        int offset_src = offset_block + h_id * in_row_length + (row_offset + i) + d_id;
        int offset_dst = offset_block_dst + h_id * out_row_length + i + d_id;
        if (freqs != nullptr) {
          float v_cos, v_sin;
          v_cos = shared_mem_cos[d_id];
          v_sin = shared_mem_sin[d_id];
          float v_src = src[offset_src];
          float v_src_rotate;
          if (!interleaved) {
            v_src_rotate = (d_id + d2 / 2 < d2)
                               ? -static_cast<float>(src[offset_src + (d2 / 2)])
                               : static_cast<float>(src[offset_src + (d2 / 2 - d2)]);
          } else {
            v_src_rotate = (d_id % 2 == 0) ? -static_cast<float>(src[offset_src + 1])
                                           : static_cast<float>(src[offset_src - 1]);
          }
          out[offset_dst] = v_src * v_cos + v_src_rotate * v_sin;
        } else {
          out[offset_dst] = src[offset_src];
        }
      }
    }
  }
  // copy the rest
  if (d > d2) {
#pragma unroll
    for (int d_id = d2 + threadIdx.x; d_id < d; d_id += blockDim.x) {
#pragma unroll
      for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
#pragma unroll
        for (int i = 0; i < out_row_length; i += d) {
          int offset_src = offset_block + h_id * in_row_length + (row_offset + i) + d_id;
          int offset_dst = offset_block_dst + h_id * out_row_length + i + d_id;
          out[offset_dst] = src[offset_src];
        }
      }
    }
  }
}

template <typename scalar_t>
__device__ void fused_qkv_rope_block_backward(const scalar_t *grad_out, const float *freqs,
                                              scalar_t *out, const bool interleaved, const int s_id,
                                              const int offset_block, const int offset_block_dst,
                                              const int h, const int d, const int d2,
                                              const int row_offset, const int in_row_length,
                                              const int out_row_length) {
  extern __shared__ float shared_mem_cos_sin_qk[];
  float *shared_mem_cos = nullptr;
  float *shared_mem_sin = nullptr;
  // Split the shared memory into cos and sin parts for q or k
  if (row_offset == 0) {  // q
    shared_mem_cos = shared_mem_cos_sin_qk;
    shared_mem_sin = shared_mem_cos_sin_qk + d2;
  } else {  // k
    shared_mem_cos = shared_mem_cos_sin_qk + 2 * d2;
    shared_mem_sin = shared_mem_cos_sin_qk + 3 * d2;
  }
  if (freqs != nullptr) {
    int tid = threadIdx.x * blockDim.y + threadIdx.y;
    for (int i = tid; i < d2; i += blockDim.x * blockDim.y) {
      sincosf(freqs[s_id * d2 + i], &shared_mem_sin[i], &shared_mem_cos[i]);
    }
  }
  __syncthreads();
#pragma unroll
  for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
#pragma unroll
    for (int i = 0; i < out_row_length; i += d) {
#pragma unroll
      for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
        int offset_dst = offset_block + h_id * in_row_length + (row_offset + i) + d_id;
        int offset_src = offset_block_dst + h_id * out_row_length + i + d_id;

        float v_src = grad_out[offset_src];
        if (freqs != nullptr) {
          float v_cos, v_sin;
          v_cos = shared_mem_cos[d_id];
          float v_src_rotate;
          if (!interleaved) {
            if (d_id + d2 / 2 < d2) {
              v_src_rotate = static_cast<float>(grad_out[offset_src + (d2 / 2)]);
              v_sin = shared_mem_sin[d_id + d2 / 2];
            } else {
              v_src_rotate = static_cast<float>(grad_out[offset_src + (d2 / 2 - d2)]);
              v_sin = -shared_mem_sin[d_id + d2 / 2 - d2];
            }
          } else {
            if (d_id % 2 == 0) {
              v_src_rotate = static_cast<float>(grad_out[offset_src + 1]);
              v_sin = shared_mem_sin[d_id + 1];
            } else {
              v_src_rotate = static_cast<float>(grad_out[offset_src - 1]);
              v_sin = -shared_mem_sin[d_id - 1];
            }
          }
          out[offset_dst] = v_src * v_cos + v_src_rotate * v_sin;
        } else {
          out[offset_dst] = grad_out[offset_src];
        }
      }
    }
  }
  // copy the rest
  if (d > d2) {
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
#pragma unroll
      for (int i = 0; i < out_row_length; i += d) {
#pragma unroll
        for (int d_id = d2 + threadIdx.x; d_id < d; d_id += blockDim.x) {
          int offset_dst = offset_block + h_id * in_row_length + (row_offset + i) + d_id;
          int offset_src = offset_block_dst + h_id * out_row_length + i + d_id;
          out[offset_dst] = grad_out[offset_src];
        }
      }
    }
  }
}

template <typename scalar_t>
__global__ void fused_qkv_rope_forward_kernel(
    const scalar_t *qkv_input, const float *q_freqs, const float *k_freqs,
    const int *start_positions, scalar_t *q_out, scalar_t *k_out, scalar_t *v_out,
    const NVTE_QKV_Format qkv_format, const bool interleaved, const int cp_size, const int cp_rank,
    const int s, const int b, const int h, const int d, const int d2, const int q_split_arg,
    const int k_split_arg, const int v_split_arg) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int cur_seqlens = s;
  int total_d = q_split_arg + k_split_arg + v_split_arg;
  int offset_block, offset_block_dst_q, offset_block_dst_k, offset_block_dst_v;
  if (qkv_format == NVTE_QKV_Format::NVTE_SBHD) {
    offset_block = s_id * b * h * total_d + b_id * h * total_d;
    offset_block_dst_q = s_id * b * h * q_split_arg + b_id * h * q_split_arg;
    offset_block_dst_k = s_id * b * h * k_split_arg + b_id * h * k_split_arg;
    offset_block_dst_v = s_id * b * h * v_split_arg + b_id * h * v_split_arg;
  } else {
    offset_block = b_id * s * h * total_d + s_id * h * total_d;
    offset_block_dst_q = b_id * s * h * q_split_arg + s_id * h * q_split_arg;
    offset_block_dst_k = b_id * s * h * k_split_arg + s_id * h * k_split_arg;
    offset_block_dst_v = b_id * s * h * v_split_arg + s_id * h * v_split_arg;
  }

  int q_limit = q_split_arg;
  int k_limit = q_limit + k_split_arg;
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
  fused_qkv_rope_block_forward(qkv_input, q_freqs, q_out, interleaved, s_id_for_freqs, offset_block,
                               offset_block_dst_q, h, d, d2, 0, total_d, q_split_arg);
  fused_qkv_rope_block_forward(qkv_input, k_freqs, k_out, interleaved, s_id_for_freqs, offset_block,
                               offset_block_dst_k, h, d, d2, q_limit, total_d, k_split_arg);
  fused_qkv_rope_block_forward(qkv_input, nullptr, v_out, interleaved, s_id_for_freqs, offset_block,
                               offset_block_dst_v, h, d, d2, k_limit, total_d, v_split_arg);
}

template <typename scalar_t>
__global__ void fused_qkv_rope_backward_kernel(
    const scalar_t *grad_out_q, const scalar_t *grad_out_k, const scalar_t *grad_out_v,
    const float *q_freqs, const float *k_freqs, scalar_t *qkv_grad,
    const NVTE_QKV_Format qkv_format, const bool interleaved, const int cp_size, const int cp_rank,
    const int s, const int b, const int h, const int d, const int d2, const int q_split_arg,
    const int k_split_arg, const int v_split_arg) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int cur_seqlens = s;
  int offset_block, offset_block_dst_q, offset_block_dst_k, offset_block_dst_v;
  int total_d = q_split_arg + k_split_arg + v_split_arg;
  if (qkv_format == NVTE_QKV_Format::NVTE_SBHD) {
    offset_block = s_id * b * h * total_d + b_id * h * total_d;
    offset_block_dst_q = s_id * b * h * q_split_arg + b_id * h * q_split_arg;
    offset_block_dst_k = s_id * b * h * k_split_arg + b_id * h * k_split_arg;
    offset_block_dst_v = s_id * b * h * v_split_arg + b_id * h * v_split_arg;
  } else {
    offset_block = b_id * s * h * total_d + s_id * h * total_d;
    offset_block_dst_q = b_id * s * h * q_split_arg + s_id * h * q_split_arg;
    offset_block_dst_k = b_id * s * h * k_split_arg + s_id * h * k_split_arg;
    offset_block_dst_v = b_id * s * h * v_split_arg + s_id * h * v_split_arg;
  }
  int q_limit = q_split_arg;
  int k_limit = q_limit + k_split_arg;
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
  fused_qkv_rope_block_backward(grad_out_q, q_freqs, qkv_grad, interleaved, s_id_for_freqs,
                                offset_block, offset_block_dst_q, h, d, d2, 0, total_d,
                                q_split_arg);
  fused_qkv_rope_block_backward(grad_out_k, k_freqs, qkv_grad, interleaved, s_id_for_freqs,
                                offset_block, offset_block_dst_k, h, d, d2, q_limit, total_d,
                                k_split_arg);
  fused_qkv_rope_block_backward(grad_out_v, nullptr, qkv_grad, interleaved, s_id_for_freqs,
                                offset_block, offset_block_dst_v, h, d, d2, k_limit, total_d,
                                v_split_arg);
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
  const int shared_mem_size = 2 * d2 * sizeof(float);  // cos, sin
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

  fused_rope_forward_kernel<<<blocks, threads, shared_mem_size, stream>>>(
      input, cu_seqlens, freqs, start_positions, output, interleaved, cp_size, cp_rank, s, h, d, d2,
      stride_s_or_t, stride_b, stride_h, stride_d, o_stride_s_or_t, o_stride_b, o_stride_h,
      o_stride_d);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename scalar_t>
void fused_rope_backward_launcher(const scalar_t *output_grads, const int *cu_seqlens,
                                  const float *freqs, const int *start_positions,
                                  scalar_t *input_grads, const NVTE_QKV_Format qkv_format,
                                  const bool interleaved, const int cp_size, const int cp_rank,
                                  const int s, const int b, const int h, const int d, const int d2,
                                  const int stride_s_or_t, const int stride_b, const int stride_h,
                                  const int stride_d, cudaStream_t stream) {
  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(s, b);
  dim3 threads(THREADS_PER_WARP, warps_per_block);
  const int shared_mem_size = 2 * d2 * sizeof(float);  // cos, sin
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

  fused_rope_backward_kernel<<<blocks, threads, shared_mem_size, stream>>>(
      output_grads, cu_seqlens, freqs, start_positions, input_grads, interleaved, cp_size, cp_rank,
      s, h, d, d2, stride_s_or_t, stride_b, stride_h, stride_d, o_stride_s_or_t, o_stride_b,
      o_stride_h, o_stride_d);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename scalar_t>
void fused_qkv_rope_forward_launcher(const scalar_t *qkv_input, const float *q_freqs,
                                     const float *k_freqs, const int *start_positions,
                                     scalar_t *q_out, scalar_t *k_out, scalar_t *v_out,
                                     const NVTE_QKV_Format qkv_format, const bool interleaved,
                                     const int cp_size, const int cp_rank, const int s, const int b,
                                     const int h, const int d, const int d2,
                                     const int qkv_split_arg_list_0, const int qkv_split_arg_list_1,
                                     const int qkv_split_arg_list_2, cudaStream_t stream) {
  const int THREADS_PER_WARP = 32;
  int warps_per_block = (h <= 8) ? h : 8;
  dim3 blocks(s, b);
  dim3 threads(THREADS_PER_WARP, warps_per_block);
  const int shared_mem_size = 4 * d2 * sizeof(float);  // cos, sin * q ,k

  fused_qkv_rope_forward_kernel<<<blocks, threads, shared_mem_size, stream>>>(
      qkv_input, q_freqs, k_freqs, start_positions, q_out, k_out, v_out, qkv_format, interleaved,
      cp_size, cp_rank, s, b, h, d, d2, qkv_split_arg_list_0, qkv_split_arg_list_1,
      qkv_split_arg_list_2);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename scalar_t>
void fused_qkv_rope_backward_launcher(const scalar_t *q_grad_out, const scalar_t *k_grad_out,
                                      const scalar_t *v_grad_out, const float *q_freqs,
                                      const float *k_freqs, scalar_t *qkv_grad_input,
                                      const NVTE_QKV_Format qkv_format, const bool interleaved,
                                      const int cp_size, const int cp_rank, const int s,
                                      const int b, const int h, const int d, const int d2,
                                      const int qkv_split_arg_list_0,
                                      const int qkv_split_arg_list_1,
                                      const int qkv_split_arg_list_2, cudaStream_t stream) {
  const int THREADS_PER_WARP = 32;
  const int warps_per_block = (h <= 8) ? h : 8;
  dim3 blocks(s, b);
  dim3 threads(THREADS_PER_WARP, warps_per_block);
  const int shared_mem_size = 4 * d2 * sizeof(float);  // cos, sin * q ,k

  fused_qkv_rope_backward_kernel<<<blocks, threads, shared_mem_size, stream>>>(
      q_grad_out, k_grad_out, v_grad_out, q_freqs, k_freqs, qkv_grad_input, qkv_format, interleaved,
      cp_size, cp_rank, s, b, h, d, d2, qkv_split_arg_list_0, qkv_split_arg_list_1,
      qkv_split_arg_list_2);
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
                         const Tensor &start_positions, Tensor *input_grads,
                         const NVTE_QKV_Format qkv_format, const bool interleaved,
                         const int cp_size, const int cp_rank, const int s, const int b,
                         const int h, const int d, const int d2, const int stride_s_or_t,
                         const int stride_b, const int stride_h, const int stride_d,
                         cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      output_grads.data.dtype, scalar_t,
      fused_rope_backward_launcher(reinterpret_cast<const scalar_t *>(output_grads.data.dptr),
                                   reinterpret_cast<const int *>(cu_seqlens.data.dptr),
                                   reinterpret_cast<const float *>(freqs.data.dptr),
                                   reinterpret_cast<const int *>(start_positions.data.dptr),
                                   reinterpret_cast<scalar_t *>(input_grads->data.dptr), qkv_format,
                                   interleaved, cp_size, cp_rank, s, b, h, d, d2, stride_s_or_t,
                                   stride_b, stride_h, stride_d, stream););
}

void fused_qkv_rope_forward(const Tensor &qkv_input, const Tensor &q_freqs, const Tensor &k_freqs,
                            const Tensor &start_positions, Tensor *q_out, Tensor *k_out,
                            Tensor *v_out, const NVTE_QKV_Format qkv_format, const bool interleaved,
                            const int cp_size, const int cp_rank, const int s, const int b,
                            const int h, const int d, const int d2, const int qkv_split_arg_list_0,
                            const int qkv_split_arg_list_1, const int qkv_split_arg_list_2,
                            cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      qkv_input.data.dtype, scalar_t,
      fused_qkv_rope_forward_launcher(reinterpret_cast<const scalar_t *>(qkv_input.data.dptr),
                                      reinterpret_cast<const float *>(q_freqs.data.dptr),
                                      reinterpret_cast<const float *>(k_freqs.data.dptr),
                                      reinterpret_cast<const int *>(start_positions.data.dptr),
                                      reinterpret_cast<scalar_t *>(q_out->data.dptr),
                                      reinterpret_cast<scalar_t *>(k_out->data.dptr),
                                      reinterpret_cast<scalar_t *>(v_out->data.dptr), qkv_format,
                                      interleaved, cp_size, cp_rank, s, b, h, d, d2,
                                      qkv_split_arg_list_0, qkv_split_arg_list_1,
                                      qkv_split_arg_list_2, stream););
}

void fused_qkv_rope_backward(const Tensor &q_grad_out, const Tensor &k_grad_out,
                             const Tensor &v_grad_out, const Tensor &q_freqs, const Tensor &k_freqs,
                             Tensor *qkv_grad_input, const NVTE_QKV_Format qkv_format,
                             const bool interleaved, const int cp_size, const int cp_rank,
                             const int s, const int b, const int h, const int d, const int d2,
                             const int qkv_split_arg_list_0, const int qkv_split_arg_list_1,
                             const int qkv_split_arg_list_2, cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      q_grad_out.data.dtype, scalar_t,
      fused_qkv_rope_backward_launcher(reinterpret_cast<const scalar_t *>(q_grad_out.data.dptr),
                                       reinterpret_cast<const scalar_t *>(k_grad_out.data.dptr),
                                       reinterpret_cast<const scalar_t *>(v_grad_out.data.dptr),
                                       reinterpret_cast<const float *>(q_freqs.data.dptr),
                                       reinterpret_cast<const float *>(k_freqs.data.dptr),
                                       reinterpret_cast<scalar_t *>(qkv_grad_input->data.dptr),
                                       qkv_format, interleaved, cp_size, cp_rank, s, b, h, d, d2,
                                       qkv_split_arg_list_0, qkv_split_arg_list_1,
                                       qkv_split_arg_list_2, stream););
}
// ============================================================================
// MLA YARN RoPE kernels
// ============================================================================

__device__ int mla_get_thd_token_idx(const int *cu_seqlens, int pid_m, int seq_num, int cp_rank,
                                     int cp_size) {
  int token_idx = -1;
  int this_seq_len = 0;
  int last_cum = cu_seqlens[0] / cp_size;
  for (int seq_idx = 0; seq_idx < seq_num; seq_idx++) {
    int cur_cum = cu_seqlens[seq_idx + 1] / cp_size;
    if (token_idx == -1 && cur_cum > pid_m) {
      token_idx = pid_m - last_cum;
      this_seq_len = cur_cum - last_cum;
    }
    last_cum = cur_cum;
  }
  if (cp_size > 1) {
    if (token_idx < this_seq_len / 2) {
      token_idx = token_idx + cp_rank * this_seq_len / 2;
    } else {
      token_idx =
          (token_idx - this_seq_len / 2) + (2 * cp_size - cp_rank - 1) * this_seq_len / 2;
    }
  }
  return token_idx;
}

template <typename scalar_t>
__global__ void mla_yarn_rope_q_forward_kernel(const scalar_t *q_input, const float *cos_data,
                                               const float *sin_data, scalar_t *q_output,
                                               const int *cu_seqlens, const int qk_head_dim,
                                               const int emb_dim, const int h, const int d,
                                               const int s, const int b, const int cp_size,
                                               const int cp_rank) {
  int pid_m = blockIdx.x;
  const int half_emb = emb_dim / 2;
  const int stride_t = h * d;
  const int stride_h_val = d;

  int token_idx;
  if (cu_seqlens == nullptr) {
    int s_id = pid_m / b;
    token_idx = s_id;
    if (cp_size > 1) {
      if (s_id < s / 2) {
        token_idx = s_id + cp_rank * s / 2;
      } else {
        token_idx = s * cp_size - (cp_rank + 1) * s / 2 + s_id - s / 2;
      }
    }
  } else {
    token_idx = mla_get_thd_token_idx(cu_seqlens, pid_m, b, cp_rank, cp_size);
  }

  extern __shared__ float shared_mem_q_fwd[];
  float *sh_cos_l = shared_mem_q_fwd;
  float *sh_sin_l = shared_mem_q_fwd + half_emb;
  float *sh_cos_r = shared_mem_q_fwd + 2 * half_emb;
  float *sh_sin_r = shared_mem_q_fwd + 3 * half_emb;

  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int num_threads = blockDim.x * blockDim.y;
  for (int i = tid; i < half_emb; i += num_threads) {
    sh_cos_l[i] = cos_data[token_idx * emb_dim + i];
    sh_sin_l[i] = sin_data[token_idx * emb_dim + i];
    sh_cos_r[i] = cos_data[token_idx * emb_dim + half_emb + i];
    sh_sin_r[i] = sin_data[token_idx * emb_dim + half_emb + i];
  }
  __syncthreads();

  int base = pid_m * stride_t;

  for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
    int head_offset = base + h_id * stride_h_val;

    for (int i = threadIdx.x; i < qk_head_dim; i += blockDim.x) {
      q_output[head_offset + i] = q_input[head_offset + i];
    }

    int rope_in = head_offset + qk_head_dim;
    int rope_out = head_offset + qk_head_dim;
    for (int i = threadIdx.x; i < half_emb; i += blockDim.x) {
      float x1 = static_cast<float>(q_input[rope_in + i * 2]);
      float x2 = static_cast<float>(q_input[rope_in + i * 2 + 1]);

      q_output[rope_out + i] = static_cast<scalar_t>(x1 * sh_cos_l[i] - x2 * sh_sin_l[i]);
      q_output[rope_out + half_emb + i] =
          static_cast<scalar_t>(x2 * sh_cos_r[i] + x1 * sh_sin_r[i]);
    }
  }
}

template <typename scalar_t>
__global__ void mla_yarn_rope_q_backward_kernel(const scalar_t *grad_output,
                                                const float *cos_data, const float *sin_data,
                                                scalar_t *grad_input, const int *cu_seqlens,
                                                const int qk_head_dim, const int emb_dim,
                                                const int h, const int d, const int s, const int b,
                                                const int cp_size, const int cp_rank) {
  int pid_m = blockIdx.x;
  const int half_emb = emb_dim / 2;
  const int stride_t = h * d;
  const int stride_h_val = d;

  int token_idx;
  if (cu_seqlens == nullptr) {
    int s_id = pid_m / b;
    token_idx = s_id;
    if (cp_size > 1) {
      if (s_id < s / 2) {
        token_idx = s_id + cp_rank * s / 2;
      } else {
        token_idx = s * cp_size - (cp_rank + 1) * s / 2 + s_id - s / 2;
      }
    }
  } else {
    token_idx = mla_get_thd_token_idx(cu_seqlens, pid_m, b, cp_rank, cp_size);
  }

  extern __shared__ float shared_mem_q_bwd[];
  float *sh_cos_l = shared_mem_q_bwd;
  float *sh_sin_l = shared_mem_q_bwd + half_emb;
  float *sh_cos_r = shared_mem_q_bwd + 2 * half_emb;
  float *sh_sin_r = shared_mem_q_bwd + 3 * half_emb;

  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int num_threads = blockDim.x * blockDim.y;
  for (int i = tid; i < half_emb; i += num_threads) {
    sh_cos_l[i] = cos_data[token_idx * emb_dim + i];
    sh_sin_l[i] = sin_data[token_idx * emb_dim + i];
    sh_cos_r[i] = cos_data[token_idx * emb_dim + half_emb + i];
    sh_sin_r[i] = sin_data[token_idx * emb_dim + half_emb + i];
  }
  __syncthreads();

  int base = pid_m * stride_t;

  for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
    int head_offset = base + h_id * stride_h_val;

    for (int i = threadIdx.x; i < qk_head_dim; i += blockDim.x) {
      grad_input[head_offset + i] = grad_output[head_offset + i];
    }

    int rope_offset = head_offset + qk_head_dim;
    for (int i = threadIdx.x; i < half_emb; i += blockDim.x) {
      float gl = static_cast<float>(grad_output[rope_offset + i]);
      float gr = static_cast<float>(grad_output[rope_offset + half_emb + i]);

      grad_input[rope_offset + i * 2] = static_cast<scalar_t>(gl * sh_cos_l[i] + gr * sh_sin_r[i]);
      grad_input[rope_offset + i * 2 + 1] =
          static_cast<scalar_t>(-gl * sh_sin_l[i] + gr * sh_cos_r[i]);
    }
  }
}

template <typename scalar_t>
__global__ void mla_yarn_rope_kv_forward_kernel(
    const scalar_t *kv_input, const scalar_t *k_pos_emb, const float *cos_data,
    const float *sin_data, scalar_t *o_key, scalar_t *o_value, const int *cu_seqlens,
    const int emb_dim, const int k_dim, const int v_dim, const int h, const int s, const int b,
    const int cp_size, const int cp_rank) {
  int pid_m = blockIdx.x;
  const int half_emb = emb_dim / 2;
  const int kv_stride_t = h * (k_dim + v_dim);
  const int kv_stride_h = k_dim + v_dim;
  const int emb_stride_t = emb_dim;
  const int k_stride_t = h * (k_dim + emb_dim);
  const int k_stride_h = k_dim + emb_dim;
  const int v_stride_t = h * v_dim;
  const int v_stride_h = v_dim;

  int token_idx;
  if (cu_seqlens == nullptr) {
    int s_id = pid_m / b;
    token_idx = s_id;
    if (cp_size > 1) {
      if (s_id < s / 2) {
        token_idx = s_id + cp_rank * s / 2;
      } else {
        token_idx = s * cp_size - (cp_rank + 1) * s / 2 + s_id - s / 2;
      }
    }
  } else {
    token_idx = mla_get_thd_token_idx(cu_seqlens, pid_m, b, cp_rank, cp_size);
  }

  extern __shared__ float shared_mem_kv_fwd[];
  float *sh_cos_l = shared_mem_kv_fwd;
  float *sh_sin_l = shared_mem_kv_fwd + half_emb;
  float *sh_cos_r = shared_mem_kv_fwd + 2 * half_emb;
  float *sh_sin_r = shared_mem_kv_fwd + 3 * half_emb;
  float *sh_rot_left = shared_mem_kv_fwd + 4 * half_emb;
  float *sh_rot_right = shared_mem_kv_fwd + 5 * half_emb;

  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int num_threads = blockDim.x * blockDim.y;
  for (int i = tid; i < half_emb; i += num_threads) {
    sh_cos_l[i] = cos_data[token_idx * emb_dim + i];
    sh_sin_l[i] = sin_data[token_idx * emb_dim + i];
    sh_cos_r[i] = cos_data[token_idx * emb_dim + half_emb + i];
    sh_sin_r[i] = sin_data[token_idx * emb_dim + half_emb + i];
  }
  __syncthreads();

  for (int i = tid; i < half_emb; i += num_threads) {
    float x1 = static_cast<float>(k_pos_emb[pid_m * emb_stride_t + i * 2]);
    float x2 = static_cast<float>(k_pos_emb[pid_m * emb_stride_t + i * 2 + 1]);
    sh_rot_left[i] = x1 * sh_cos_l[i] - x2 * sh_sin_l[i];
    sh_rot_right[i] = x2 * sh_cos_r[i] + x1 * sh_sin_r[i];
  }
  __syncthreads();

  for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
    int kv_head = pid_m * kv_stride_t + h_id * kv_stride_h;
    int k_head = pid_m * k_stride_t + h_id * k_stride_h;
    int v_head = pid_m * v_stride_t + h_id * v_stride_h;

    for (int i = threadIdx.x; i < k_dim; i += blockDim.x) {
      o_key[k_head + i] = kv_input[kv_head + i];
    }

    for (int i = threadIdx.x; i < v_dim; i += blockDim.x) {
      o_value[v_head + i] = kv_input[kv_head + k_dim + i];
    }

    for (int i = threadIdx.x; i < half_emb; i += blockDim.x) {
      o_key[k_head + k_dim + i] = static_cast<scalar_t>(sh_rot_left[i]);
      o_key[k_head + k_dim + half_emb + i] = static_cast<scalar_t>(sh_rot_right[i]);
    }
  }
}

template <typename scalar_t>
__global__ void mla_yarn_rope_kv_backward_kernel(
    const scalar_t *dk, const scalar_t *dv, const float *cos_data, const float *sin_data,
    scalar_t *d_kv, scalar_t *d_emb, const int *cu_seqlens, const int emb_dim, const int k_dim,
    const int v_dim, const int h, const int s, const int b, const int cp_size, const int cp_rank) {
  int pid_m = blockIdx.x;
  const int half_emb = emb_dim / 2;
  const int dk_stride_t = h * (k_dim + emb_dim);
  const int dk_stride_h = k_dim + emb_dim;
  const int dv_stride_t = h * v_dim;
  const int dv_stride_h = v_dim;
  const int dkv_stride_t = h * (k_dim + v_dim);
  const int dkv_stride_h = k_dim + v_dim;

  int token_idx;
  if (cu_seqlens == nullptr) {
    int s_id = pid_m / b;
    token_idx = s_id;
    if (cp_size > 1) {
      if (s_id < s / 2) {
        token_idx = s_id + cp_rank * s / 2;
      } else {
        token_idx = s * cp_size - (cp_rank + 1) * s / 2 + s_id - s / 2;
      }
    }
  } else {
    token_idx = mla_get_thd_token_idx(cu_seqlens, pid_m, b, cp_rank, cp_size);
  }

  for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
    int dk_head = pid_m * dk_stride_t + h_id * dk_stride_h;
    int dv_head = pid_m * dv_stride_t + h_id * dv_stride_h;
    int dkv_head = pid_m * dkv_stride_t + h_id * dkv_stride_h;

    for (int i = threadIdx.x; i < k_dim; i += blockDim.x) {
      d_kv[dkv_head + i] = dk[dk_head + i];
    }
    for (int i = threadIdx.x; i < v_dim; i += blockDim.x) {
      d_kv[dkv_head + k_dim + i] = dv[dv_head + i];
    }
  }

  extern __shared__ float shared_mem_kv_bwd[];
  float *sh_cos_l = shared_mem_kv_bwd;
  float *sh_sin_l = shared_mem_kv_bwd + half_emb;
  float *sh_cos_r = shared_mem_kv_bwd + 2 * half_emb;
  float *sh_sin_r = shared_mem_kv_bwd + 3 * half_emb;

  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int num_threads = blockDim.x * blockDim.y;
  for (int i = tid; i < half_emb; i += num_threads) {
    sh_cos_l[i] = cos_data[token_idx * emb_dim + i];
    sh_sin_l[i] = sin_data[token_idx * emb_dim + i];
    sh_cos_r[i] = cos_data[token_idx * emb_dim + half_emb + i];
    sh_sin_r[i] = sin_data[token_idx * emb_dim + half_emb + i];
  }
  __syncthreads();

  for (int i = threadIdx.x; i < half_emb; i += blockDim.x) {
    if (threadIdx.y == 0) {
      float accum_l = 0.0f, accum_r = 0.0f;
      for (int h_id = 0; h_id < h; h_id++) {
        int dk_head = pid_m * dk_stride_t + h_id * dk_stride_h;
        accum_l += static_cast<float>(dk[dk_head + k_dim + i]);
        accum_r += static_cast<float>(dk[dk_head + k_dim + half_emb + i]);
      }
      float dx1 = accum_l * sh_cos_l[i] + accum_r * sh_sin_r[i];
      float dx2 = -accum_l * sh_sin_l[i] + accum_r * sh_cos_r[i];
      d_emb[pid_m * emb_dim + i * 2] = static_cast<scalar_t>(dx1);
      d_emb[pid_m * emb_dim + i * 2 + 1] = static_cast<scalar_t>(dx2);
    }
  }
}

template <typename scalar_t>
void mla_yarn_rope_q_forward_launcher(const scalar_t *q_input, const float *cos_data,
                                      const float *sin_data, scalar_t *q_output,
                                      const int *cu_seqlens, const int qk_head_dim,
                                      const int emb_dim, const int h, const int d,
                                      const int total_seqlen, const int s, const int b,
                                      const int cp_size, const int cp_rank, cudaStream_t stream) {
  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(total_seqlen);
  dim3 threads(THREADS_PER_WARP, warps_per_block);
  const int shared_mem_size = 4 * (emb_dim / 2) * sizeof(float);

  mla_yarn_rope_q_forward_kernel<<<blocks, threads, shared_mem_size, stream>>>(
      q_input, cos_data, sin_data, q_output, cu_seqlens, qk_head_dim, emb_dim, h, d, s, b,
      cp_size, cp_rank);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename scalar_t>
void mla_yarn_rope_q_backward_launcher(const scalar_t *grad_output, const float *cos_data,
                                       const float *sin_data, scalar_t *grad_input,
                                       const int *cu_seqlens, const int qk_head_dim,
                                       const int emb_dim, const int h, const int d,
                                       const int total_seqlen, const int s, const int b,
                                       const int cp_size, const int cp_rank, cudaStream_t stream) {
  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(total_seqlen);
  dim3 threads(THREADS_PER_WARP, warps_per_block);
  const int shared_mem_size = 4 * (emb_dim / 2) * sizeof(float);

  mla_yarn_rope_q_backward_kernel<<<blocks, threads, shared_mem_size, stream>>>(
      grad_output, cos_data, sin_data, grad_input, cu_seqlens, qk_head_dim, emb_dim, h, d, s, b,
      cp_size, cp_rank);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename scalar_t>
void mla_yarn_rope_kv_forward_launcher(const scalar_t *kv_input, const scalar_t *k_pos_emb,
                                       const float *cos_data, const float *sin_data,
                                       scalar_t *o_key, scalar_t *o_value, const int *cu_seqlens,
                                       const int emb_dim, const int k_dim, const int v_dim,
                                       const int h, const int total_seqlen, const int s,
                                       const int b, const int cp_size, const int cp_rank,
                                       cudaStream_t stream) {
  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(total_seqlen);
  dim3 threads(THREADS_PER_WARP, warps_per_block);
  const int shared_mem_size = 6 * (emb_dim / 2) * sizeof(float);

  mla_yarn_rope_kv_forward_kernel<<<blocks, threads, shared_mem_size, stream>>>(
      kv_input, k_pos_emb, cos_data, sin_data, o_key, o_value, cu_seqlens, emb_dim, k_dim, v_dim,
      h, s, b, cp_size, cp_rank);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename scalar_t>
void mla_yarn_rope_kv_backward_launcher(const scalar_t *dk, const scalar_t *dv,
                                        const float *cos_data, const float *sin_data,
                                        scalar_t *d_kv, scalar_t *d_emb, const int *cu_seqlens,
                                        const int emb_dim, const int k_dim, const int v_dim,
                                        const int h, const int total_seqlen, const int s,
                                        const int b, const int cp_size, const int cp_rank,
                                        cudaStream_t stream) {
  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(total_seqlen);
  dim3 threads(THREADS_PER_WARP, warps_per_block);
  const int shared_mem_size = 4 * (emb_dim / 2) * sizeof(float);

  mla_yarn_rope_kv_backward_kernel<<<blocks, threads, shared_mem_size, stream>>>(
      dk, dv, cos_data, sin_data, d_kv, d_emb, cu_seqlens, emb_dim, k_dim, v_dim, h, s, b,
      cp_size, cp_rank);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void fused_mla_rope_q_forward(const Tensor &q_input, const Tensor &cos, const Tensor &sin,
                              Tensor *q_output, const Tensor &cu_seqlens, const int qk_head_dim,
                              const int emb_dim, const int h, const int d, const int total_seqlen,
                              const int s, const int b, const int cp_size, const int cp_rank,
                              cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      q_input.data.dtype, scalar_t,
      mla_yarn_rope_q_forward_launcher(
          reinterpret_cast<const scalar_t *>(q_input.data.dptr),
          reinterpret_cast<const float *>(cos.data.dptr),
          reinterpret_cast<const float *>(sin.data.dptr),
          reinterpret_cast<scalar_t *>(q_output->data.dptr),
          reinterpret_cast<const int *>(cu_seqlens.data.dptr), qk_head_dim, emb_dim, h, d,
          total_seqlen, s, b, cp_size, cp_rank, stream););
}

void fused_mla_rope_q_backward(const Tensor &grad_output, const Tensor &cos, const Tensor &sin,
                               Tensor *grad_input, const Tensor &cu_seqlens,
                               const int qk_head_dim, const int emb_dim, const int h, const int d,
                               const int total_seqlen, const int s, const int b, const int cp_size,
                               const int cp_rank, cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      grad_output.data.dtype, scalar_t,
      mla_yarn_rope_q_backward_launcher(
          reinterpret_cast<const scalar_t *>(grad_output.data.dptr),
          reinterpret_cast<const float *>(cos.data.dptr),
          reinterpret_cast<const float *>(sin.data.dptr),
          reinterpret_cast<scalar_t *>(grad_input->data.dptr),
          reinterpret_cast<const int *>(cu_seqlens.data.dptr), qk_head_dim, emb_dim, h, d,
          total_seqlen, s, b, cp_size, cp_rank, stream););
}

void fused_mla_rope_kv_forward(const Tensor &kv_input, const Tensor &k_pos_emb, const Tensor &cos,
                               const Tensor &sin, Tensor *o_key, Tensor *o_value,
                               const Tensor &cu_seqlens, const int emb_dim, const int k_dim,
                               const int v_dim, const int h, const int total_seqlen, const int s,
                               const int b, const int cp_size, const int cp_rank,
                               cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      kv_input.data.dtype, scalar_t,
      mla_yarn_rope_kv_forward_launcher(
          reinterpret_cast<const scalar_t *>(kv_input.data.dptr),
          reinterpret_cast<const scalar_t *>(k_pos_emb.data.dptr),
          reinterpret_cast<const float *>(cos.data.dptr),
          reinterpret_cast<const float *>(sin.data.dptr),
          reinterpret_cast<scalar_t *>(o_key->data.dptr),
          reinterpret_cast<scalar_t *>(o_value->data.dptr),
          reinterpret_cast<const int *>(cu_seqlens.data.dptr), emb_dim, k_dim, v_dim, h,
          total_seqlen, s, b, cp_size, cp_rank, stream););
}

void fused_mla_rope_kv_backward(const Tensor &dk, const Tensor &dv, const Tensor &cos,
                                const Tensor &sin, Tensor *d_kv, Tensor *d_emb,
                                const Tensor &cu_seqlens, const int emb_dim, const int k_dim,
                                const int v_dim, const int h, const int total_seqlen, const int s,
                                const int b, const int cp_size, const int cp_rank,
                                cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      dk.data.dtype, scalar_t,
      mla_yarn_rope_kv_backward_launcher(
          reinterpret_cast<const scalar_t *>(dk.data.dptr),
          reinterpret_cast<const scalar_t *>(dv.data.dptr),
          reinterpret_cast<const float *>(cos.data.dptr),
          reinterpret_cast<const float *>(sin.data.dptr),
          reinterpret_cast<scalar_t *>(d_kv->data.dptr),
          reinterpret_cast<scalar_t *>(d_emb->data.dptr),
          reinterpret_cast<const int *>(cu_seqlens.data.dptr), emb_dim, k_dim, v_dim, h,
          total_seqlen, s, b, cp_size, cp_rank, stream););
}

}  // end namespace transformer_engine

void nvte_fused_mla_rope_q_forward(const NVTETensor q_input, const NVTETensor cos,
                                   const NVTETensor sin, NVTETensor q_output,
                                   const NVTETensor cu_seqlens, const int qk_head_dim,
                                   const int emb_dim, const int h, const int d,
                                   const int total_seqlen, const int s, const int b,
                                   const int cp_size, const int cp_rank, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_mla_rope_q_forward);
  using namespace transformer_engine;
  fused_mla_rope_q_forward(*convertNVTETensorCheck(q_input), *convertNVTETensorCheck(cos),
                           *convertNVTETensorCheck(sin), convertNVTETensorCheck(q_output),
                           *convertNVTETensorCheck(cu_seqlens), qk_head_dim, emb_dim, h, d,
                           total_seqlen, s, b, cp_size, cp_rank, stream);
}

void nvte_fused_mla_rope_q_backward(const NVTETensor grad_output, const NVTETensor cos,
                                    const NVTETensor sin, NVTETensor grad_input,
                                    const NVTETensor cu_seqlens, const int qk_head_dim,
                                    const int emb_dim, const int h, const int d,
                                    const int total_seqlen, const int s, const int b,
                                    const int cp_size, const int cp_rank, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_mla_rope_q_backward);
  using namespace transformer_engine;
  fused_mla_rope_q_backward(*convertNVTETensorCheck(grad_output), *convertNVTETensorCheck(cos),
                            *convertNVTETensorCheck(sin), convertNVTETensorCheck(grad_input),
                            *convertNVTETensorCheck(cu_seqlens), qk_head_dim, emb_dim, h, d,
                            total_seqlen, s, b, cp_size, cp_rank, stream);
}

void nvte_fused_mla_rope_kv_forward(const NVTETensor kv_input, const NVTETensor k_pos_emb,
                                    const NVTETensor cos, const NVTETensor sin, NVTETensor o_key,
                                    NVTETensor o_value, const NVTETensor cu_seqlens,
                                    const int emb_dim, const int k_dim, const int v_dim,
                                    const int h, const int total_seqlen, const int s, const int b,
                                    const int cp_size, const int cp_rank, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_mla_rope_kv_forward);
  using namespace transformer_engine;
  fused_mla_rope_kv_forward(
      *convertNVTETensorCheck(kv_input), *convertNVTETensorCheck(k_pos_emb),
      *convertNVTETensorCheck(cos), *convertNVTETensorCheck(sin), convertNVTETensorCheck(o_key),
      convertNVTETensorCheck(o_value), *convertNVTETensorCheck(cu_seqlens), emb_dim, k_dim, v_dim,
      h, total_seqlen, s, b, cp_size, cp_rank, stream);
}

void nvte_fused_mla_rope_kv_backward(const NVTETensor dk, const NVTETensor dv, const NVTETensor cos,
                                     const NVTETensor sin, NVTETensor d_kv, NVTETensor d_emb,
                                     const NVTETensor cu_seqlens, const int emb_dim,
                                     const int k_dim, const int v_dim, const int h,
                                     const int total_seqlen, const int s, const int b,
                                     const int cp_size, const int cp_rank, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_mla_rope_kv_backward);
  using namespace transformer_engine;
  fused_mla_rope_kv_backward(*convertNVTETensorCheck(dk), *convertNVTETensorCheck(dv),
                             *convertNVTETensorCheck(cos), *convertNVTETensorCheck(sin),
                             convertNVTETensorCheck(d_kv), convertNVTETensorCheck(d_emb),
                             *convertNVTETensorCheck(cu_seqlens), emb_dim, k_dim, v_dim, h,
                             total_seqlen, s, b, cp_size, cp_rank, stream);
}

void nvte_fused_rope_forward(const NVTETensor input, const NVTETensor cu_seqlens,
                             const NVTETensor freqs, const NVTETensor start_positions,
                             NVTETensor output, const NVTE_QKV_Format qkv_format,
                             const bool interleaved, const int cp_size, const int cp_rank,
                             const int s, const int b, const int h, const int d, const int d2,
                             const int stride_s_or_t, const int stride_b, const int stride_h,
                             const int stride_d, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_rope_forward);
  using namespace transformer_engine;
  fused_rope_forward(*convertNVTETensorCheck(input), *convertNVTETensorCheck(cu_seqlens),
                     *convertNVTETensorCheck(freqs), *convertNVTETensorCheck(start_positions),
                     convertNVTETensorCheck(output), qkv_format, interleaved, cp_size, cp_rank, s,
                     b, h, d, d2, stride_s_or_t, stride_b, stride_h, stride_d, stream);
}

void nvte_fused_rope_backward(const NVTETensor output_grads, const NVTETensor cu_seqlens,
                              const NVTETensor freqs, const NVTETensor start_positions,
                              NVTETensor input_grads, const NVTE_QKV_Format qkv_format,
                              const bool interleaved, const int cp_size, const int cp_rank,
                              const int s, const int b, const int h, const int d, const int d2,
                              const int stride_s_or_t, const int stride_b, const int stride_h,
                              const int stride_d, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_rope_backward);
  using namespace transformer_engine;
  fused_rope_backward(*convertNVTETensorCheck(output_grads), *convertNVTETensorCheck(cu_seqlens),
                      *convertNVTETensorCheck(freqs), *convertNVTETensorCheck(start_positions),
                      convertNVTETensorCheck(input_grads), qkv_format, interleaved, cp_size,
                      cp_rank, s, b, h, d, d2, stride_s_or_t, stride_b, stride_h, stride_d, stream);
}

void nvte_fused_qkv_rope_forward(const NVTETensor qkv_input, const NVTETensor q_freqs,
                                 const NVTETensor k_freqs, const NVTETensor start_positions,
                                 NVTETensor q_out, NVTETensor k_out, NVTETensor v_out,
                                 const NVTE_QKV_Format qkv_format, const bool interleaved,
                                 const int cp_size, const int cp_rank, const int s, const int b,
                                 const int h, const int d, const int d2,
                                 const int qkv_split_arg_list_0, const int qkv_split_arg_list_1,
                                 const int qkv_split_arg_list_2, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_qkv_rope_forward);
  using namespace transformer_engine;
  fused_qkv_rope_forward(*convertNVTETensorCheck(qkv_input), *convertNVTETensorCheck(q_freqs),
                         *convertNVTETensorCheck(k_freqs), *convertNVTETensorCheck(start_positions),
                         convertNVTETensorCheck(q_out), convertNVTETensorCheck(k_out),
                         convertNVTETensorCheck(v_out), qkv_format, interleaved, cp_size, cp_rank,
                         s, b, h, d, d2, qkv_split_arg_list_0, qkv_split_arg_list_1,
                         qkv_split_arg_list_2, stream);
}

void nvte_fused_qkv_rope_backward(const NVTETensor q_grad_out, const NVTETensor k_grad_out,
                                  const NVTETensor v_grad_out, const NVTETensor q_freqs,
                                  const NVTETensor k_freqs, NVTETensor qkv_grad_input,
                                  const NVTE_QKV_Format qkv_format, const bool interleaved,
                                  const int cp_size, const int cp_rank, const int s, const int b,
                                  const int h, const int d, const int d2,
                                  const int qkv_split_arg_list_0, const int qkv_split_arg_list_1,
                                  const int qkv_split_arg_list_2, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_qkv_rope_backward);
  using namespace transformer_engine;
  fused_qkv_rope_backward(*convertNVTETensorCheck(q_grad_out), *convertNVTETensorCheck(k_grad_out),
                          *convertNVTETensorCheck(v_grad_out), *convertNVTETensorCheck(q_freqs),
                          *convertNVTETensorCheck(k_freqs), convertNVTETensorCheck(qkv_grad_input),
                          qkv_format, interleaved, cp_size, cp_rank, s, b, h, d, d2,
                          qkv_split_arg_list_0, qkv_split_arg_list_1, qkv_split_arg_list_2, stream);
}
