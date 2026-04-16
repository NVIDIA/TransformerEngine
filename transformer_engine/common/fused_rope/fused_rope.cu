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
// ---------------------------------------------------------------------------
// MLA YARN RoPE
// ---------------------------------------------------------------------------

/*
 * MLA YARN RoPE for Q: in-place rotation on the *tail* emb_dim elements of each head.
 *
 * Layout:  Q[..., 0 : qk_head_dim] is untouched (compressed latent part).
 *          Q[..., qk_head_dim : qk_head_dim + emb_dim] gets YARN rotation.
 *
 * YARN rotation (interleaved-read, contiguous-write):
 *   x1 = input[even indices],  x2 = input[odd indices]        (interleaved read)
 *   cos_L, sin_L = cos/sin table first  half
 *   cos_R, sin_R = cos/sin table second half
 *   out[first  half] = x1 * cos_L - x2 * sin_L
 *   out[second half] = x2 * cos_R + x1 * sin_R
 *
 * COS / SIN are pre-computed and stored contiguously as [max_seq_len, emb_dim].
 */

template <typename scalar_t>
__device__ void mla_rope_q_block_forward(
    const scalar_t *src, const float *cos_table, const float *sin_table, scalar_t *dst,
    const int s_id_for_freqs, const int offset_block, const int offset_block_dst, const int h,
    const int qk_head_dim, const int emb_dim, const int stride_h, const int o_stride_h) {
  const int half_emb = emb_dim / 2;

  extern __shared__ float shared_mem[];
  float *sh_cos_L = shared_mem;
  float *sh_sin_L = shared_mem + half_emb;
  float *sh_cos_R = shared_mem + half_emb * 2;
  float *sh_sin_R = shared_mem + half_emb * 3;

  int tid = threadIdx.x * blockDim.y + threadIdx.y;
  int block_size = blockDim.x * blockDim.y;
  for (int i = tid; i < half_emb; i += block_size) {
    sh_cos_L[i] = cos_table[s_id_for_freqs * emb_dim + i];
    sh_sin_L[i] = sin_table[s_id_for_freqs * emb_dim + i];
    sh_cos_R[i] = cos_table[s_id_for_freqs * emb_dim + half_emb + i];
    sh_sin_R[i] = sin_table[s_id_for_freqs * emb_dim + half_emb + i];
  }
  __syncthreads();

#pragma unroll
  for (int d_id = threadIdx.x; d_id < qk_head_dim; d_id += blockDim.x) {
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int src_off = offset_block + h_id * stride_h + d_id;
      int dst_off = offset_block_dst + h_id * o_stride_h + d_id;
      dst[dst_off] = src[src_off];
    }
  }

#pragma unroll
  for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
#pragma unroll
    for (int i = threadIdx.x; i < half_emb; i += blockDim.x) {
      int even_off = offset_block + h_id * stride_h + qk_head_dim + 2 * i;
      int odd_off = offset_block + h_id * stride_h + qk_head_dim + 2 * i + 1;
      float x1 = static_cast<float>(src[even_off]);
      float x2 = static_cast<float>(src[odd_off]);

      float out_left = x1 * sh_cos_L[i] - x2 * sh_sin_L[i];
      float out_right = x2 * sh_cos_R[i] + x1 * sh_sin_R[i];

      int left_off = offset_block_dst + h_id * o_stride_h + qk_head_dim + i;
      int right_off = offset_block_dst + h_id * o_stride_h + qk_head_dim + half_emb + i;
      dst[left_off] = static_cast<scalar_t>(out_left);
      dst[right_off] = static_cast<scalar_t>(out_right);
    }
  }
}

/*
 * Backward of YARN rotation for Q.
 *
 * Given forward:
 *   L = x1 * cL - x2 * sL        R = x2 * cR + x1 * sR
 *
 * Backward (solve for dx1, dx2):
 *   dx1 = dL * cL + dR * sR
 *   dx2 = -dL * sL + dR * cR
 *
 * where dL / dR are the gradients stored contiguously, and
 * dx1 / dx2 are written back to interleaved positions.
 */
template <typename scalar_t>
__device__ void mla_rope_q_block_backward(
    const scalar_t *grad_out, const float *cos_table, const float *sin_table, scalar_t *grad_in,
    const int s_id_for_freqs, const int offset_block, const int offset_block_dst, const int h,
    const int qk_head_dim, const int emb_dim, const int stride_h, const int o_stride_h) {
  const int half_emb = emb_dim / 2;

  extern __shared__ float shared_mem[];
  float *sh_cos_L = shared_mem;
  float *sh_sin_L = shared_mem + half_emb;
  float *sh_cos_R = shared_mem + half_emb * 2;
  float *sh_sin_R = shared_mem + half_emb * 3;

  int tid = threadIdx.x * blockDim.y + threadIdx.y;
  int block_size = blockDim.x * blockDim.y;
  for (int i = tid; i < half_emb; i += block_size) {
    sh_cos_L[i] = cos_table[s_id_for_freqs * emb_dim + i];
    sh_sin_L[i] = sin_table[s_id_for_freqs * emb_dim + i];
    sh_cos_R[i] = cos_table[s_id_for_freqs * emb_dim + half_emb + i];
    sh_sin_R[i] = sin_table[s_id_for_freqs * emb_dim + half_emb + i];
  }
  __syncthreads();

#pragma unroll
  for (int d_id = threadIdx.x; d_id < qk_head_dim; d_id += blockDim.x) {
#pragma unroll
    for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
      int src_off = offset_block + h_id * stride_h + d_id;
      int dst_off = offset_block_dst + h_id * o_stride_h + d_id;
      grad_in[dst_off] = grad_out[src_off];
    }
  }

#pragma unroll
  for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
#pragma unroll
    for (int i = threadIdx.x; i < half_emb; i += blockDim.x) {
      int left_off = offset_block + h_id * stride_h + qk_head_dim + i;
      int right_off = offset_block + h_id * stride_h + qk_head_dim + half_emb + i;
      float dL = static_cast<float>(grad_out[left_off]);
      float dR = static_cast<float>(grad_out[right_off]);

      float dx1 = dL * sh_cos_L[i] + dR * sh_sin_R[i];
      float dx2 = -dL * sh_sin_L[i] + dR * sh_cos_R[i];

      int even_off = offset_block_dst + h_id * o_stride_h + qk_head_dim + 2 * i;
      int odd_off = offset_block_dst + h_id * o_stride_h + qk_head_dim + 2 * i + 1;
      grad_in[even_off] = static_cast<scalar_t>(dx1);
      grad_in[odd_off] = static_cast<scalar_t>(dx2);
    }
  }
}

// ---------------------------------------------------------------------------
// Global kernels for MLA Q
// ---------------------------------------------------------------------------

template <typename scalar_t>
__global__ void mla_rope_q_forward_kernel(
    const scalar_t *src, const int *cu_seqlens, const float *cos_table, const float *sin_table,
    scalar_t *dst, const int cp_size, const int cp_rank, const int s, const int h,
    const int qk_head_dim, const int emb_dim, const int stride_s_or_t, const int stride_b,
    const int stride_h, const int o_stride_s_or_t, const int o_stride_b,
    const int o_stride_h) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int offset_block, offset_block_dst;
  int cur_seqlens;
  if (cu_seqlens != nullptr) {
    int start = cu_seqlens[b_id] / cp_size;
    int end = cu_seqlens[b_id + 1] / cp_size;
    int t_id = s_id + start;
    if (t_id >= end) return;
    offset_block = t_id * stride_s_or_t;
    offset_block_dst = t_id * o_stride_s_or_t;
    cur_seqlens = end - start;
  } else {
    offset_block = s_id * stride_s_or_t + b_id * stride_b;
    offset_block_dst = s_id * o_stride_s_or_t + b_id * o_stride_b;
    cur_seqlens = s;
  }

  int s_id_for_freqs = s_id;
  if (cp_size > 1) {
    assert(cur_seqlens % 2 == 0);
    if (s_id < cur_seqlens / 2) {
      s_id_for_freqs += cp_rank * cur_seqlens / 2;
    } else {
      s_id_for_freqs += cur_seqlens * cp_size - (cp_rank + 1) * cur_seqlens / 2 - cur_seqlens / 2;
    }
  }

  mla_rope_q_block_forward(src, cos_table, sin_table, dst, s_id_for_freqs, offset_block,
                            offset_block_dst, h, qk_head_dim, emb_dim, stride_h, o_stride_h);
}

template <typename scalar_t>
__global__ void mla_rope_q_backward_kernel(
    const scalar_t *grad_out, const int *cu_seqlens, const float *cos_table,
    const float *sin_table, scalar_t *grad_in, const int cp_size, const int cp_rank, const int s,
    const int h, const int qk_head_dim, const int emb_dim, const int stride_s_or_t,
    const int stride_b, const int stride_h, const int o_stride_s_or_t,
    const int o_stride_b, const int o_stride_h) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int offset_block, offset_block_dst;
  int cur_seqlens;
  if (cu_seqlens != nullptr) {
    int start = cu_seqlens[b_id] / cp_size;
    int end = cu_seqlens[b_id + 1] / cp_size;
    int t_id = s_id + start;
    if (t_id >= end) return;
    offset_block = t_id * stride_s_or_t;
    offset_block_dst = t_id * o_stride_s_or_t;
    cur_seqlens = end - start;
  } else {
    offset_block = s_id * stride_s_or_t + b_id * stride_b;
    offset_block_dst = s_id * o_stride_s_or_t + b_id * o_stride_b;
    cur_seqlens = s;
  }

  int s_id_for_freqs = s_id;
  if (cp_size > 1) {
    assert(cur_seqlens % 2 == 0);
    if (s_id < cur_seqlens / 2) {
      s_id_for_freqs += cp_rank * cur_seqlens / 2;
    } else {
      s_id_for_freqs += cur_seqlens * cp_size - (cp_rank + 1) * cur_seqlens / 2 - cur_seqlens / 2;
    }
  }

  mla_rope_q_block_backward(grad_out, cos_table, sin_table, grad_in, s_id_for_freqs,
                             offset_block, offset_block_dst, h, qk_head_dim, emb_dim,
                             stride_h, o_stride_h);
}

// ---------------------------------------------------------------------------
// Global kernels for MLA KV
// ---------------------------------------------------------------------------

/*
 * MLA YARN RoPE for KV (forward).
 *
 * Inputs:
 *   KV       : [s, b, h, k_dim + v_dim]  or  [t, h, k_dim + v_dim]
 *   K_POS_EMB: [s, b, emb_dim]           or  [t, emb_dim]           (single-head)
 *   COS/SIN  : [max_seq_len, emb_dim]
 *
 * Outputs:
 *   O_KEY    : [s, b, h, k_dim + emb_dim]  or  [t, h, k_dim + emb_dim]
 *   O_VALUE  : [s, b, h, v_dim]            or  [t, h, v_dim]
 *
 * Operations per token:
 *   1) Copy k_dim from KV into O_KEY prefix (per-head).
 *   2) Copy v_dim from KV into O_VALUE (per-head).
 *   3) Apply YARN rotation to K_POS_EMB (single-head), broadcast result to all heads in O_KEY tail.
 */
template <typename scalar_t>
__global__ void mla_rope_kv_forward_kernel(
    const scalar_t *kv, const scalar_t *k_pos_emb, const float *cos_table,
    const float *sin_table, scalar_t *o_key, scalar_t *o_value, const int *cu_seqlens,
    const int cp_size, const int cp_rank, const int s, const int h, const int k_dim,
    const int v_dim, const int emb_dim, const int stride_kv_s, const int stride_kv_b,
    const int stride_kv_h, const int stride_emb_s, const int stride_emb_b,
    const int okey_stride_s, const int okey_stride_b, const int o_key_d,
    const int oval_stride_s, const int oval_stride_b, const int o_val_d) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int cur_seqlens;
  int kv_offset_base, emb_offset_base, okey_offset_base, oval_offset_base;

  if (cu_seqlens != nullptr) {
    int start = cu_seqlens[b_id] / cp_size;
    int end = cu_seqlens[b_id + 1] / cp_size;
    int t_id = s_id + start;
    if (t_id >= end) return;
    kv_offset_base = t_id * stride_kv_s;
    emb_offset_base = t_id * stride_emb_s;
    okey_offset_base = t_id * okey_stride_s;
    oval_offset_base = t_id * oval_stride_s;
    cur_seqlens = end - start;
  } else {
    kv_offset_base = s_id * stride_kv_s + b_id * stride_kv_b;
    emb_offset_base = s_id * stride_emb_s + b_id * stride_emb_b;
    okey_offset_base = s_id * okey_stride_s + b_id * okey_stride_b;
    oval_offset_base = s_id * oval_stride_s + b_id * oval_stride_b;
    cur_seqlens = s;
  }

  int s_id_for_freqs = s_id;
  if (cp_size > 1) {
    assert(cur_seqlens % 2 == 0);
    if (s_id < cur_seqlens / 2) {
      s_id_for_freqs += cp_rank * cur_seqlens / 2;
    } else {
      s_id_for_freqs += cur_seqlens * cp_size - (cp_rank + 1) * cur_seqlens / 2 - cur_seqlens / 2;
    }
  }

  const int half_emb = emb_dim / 2;

  extern __shared__ float shared_mem[];
  float *sh_cos_L = shared_mem;
  float *sh_sin_L = shared_mem + half_emb;
  float *sh_cos_R = shared_mem + half_emb * 2;
  float *sh_sin_R = shared_mem + half_emb * 3;

  int tid = threadIdx.x * blockDim.y + threadIdx.y;
  int block_size = blockDim.x * blockDim.y;
  for (int i = tid; i < half_emb; i += block_size) {
    sh_cos_L[i] = cos_table[s_id_for_freqs * emb_dim + i];
    sh_sin_L[i] = sin_table[s_id_for_freqs * emb_dim + i];
    sh_cos_R[i] = cos_table[s_id_for_freqs * emb_dim + half_emb + i];
    sh_sin_R[i] = sin_table[s_id_for_freqs * emb_dim + half_emb + i];
  }
  __syncthreads();

  float *sh_rope_left = shared_mem + half_emb * 4;
  float *sh_rope_right = shared_mem + half_emb * 5;

  for (int i = tid; i < half_emb; i += block_size) {
    float x1 = static_cast<float>(k_pos_emb[emb_offset_base + 2 * i]);
    float x2 = static_cast<float>(k_pos_emb[emb_offset_base + 2 * i + 1]);
    sh_rope_left[i] = x1 * sh_cos_L[i] - x2 * sh_sin_L[i];
    sh_rope_right[i] = x2 * sh_cos_R[i] + x1 * sh_sin_R[i];
  }
  __syncthreads();

#pragma unroll
  for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
#pragma unroll
    for (int d_id = threadIdx.x; d_id < k_dim; d_id += blockDim.x) {
      int kv_off = kv_offset_base + h_id * stride_kv_h + d_id;
      int ok_off = okey_offset_base + h_id * o_key_d + d_id;
      o_key[ok_off] = kv[kv_off];
    }

#pragma unroll
    for (int i = threadIdx.x; i < half_emb; i += blockDim.x) {
      int ok_left = okey_offset_base + h_id * o_key_d + k_dim + i;
      int ok_right = okey_offset_base + h_id * o_key_d + k_dim + half_emb + i;
      o_key[ok_left] = static_cast<scalar_t>(sh_rope_left[i]);
      o_key[ok_right] = static_cast<scalar_t>(sh_rope_right[i]);
    }

#pragma unroll
    for (int d_id = threadIdx.x; d_id < v_dim; d_id += blockDim.x) {
      int kv_off = kv_offset_base + h_id * stride_kv_h + k_dim + d_id;
      int ov_off = oval_offset_base + h_id * o_val_d + d_id;
      o_value[ov_off] = kv[kv_off];
    }
  }
}

/*
 * MLA YARN RoPE for KV (backward).
 */
template <typename scalar_t>
__global__ void mla_rope_kv_backward_kernel(
    const scalar_t *dk, const scalar_t *dv, const float *cos_table, const float *sin_table,
    scalar_t *dkv, scalar_t *d_emb, const int *cu_seqlens, const int cp_size, const int cp_rank,
    const int s, const int h, const int k_dim, const int v_dim, const int emb_dim,
    const int stride_dk_s, const int stride_dk_b, const int stride_dk_h, const int stride_dv_s,
    const int stride_dv_b, const int stride_dv_h,
    const int dkv_stride_s, const int dkv_stride_b, const int o_dkv_d,
    const int o_demb_stride_s, const int o_demb_stride_b) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int cur_seqlens;
  int dk_offset_base, dv_offset_base, dkv_offset_base, demb_offset_base;

  if (cu_seqlens != nullptr) {
    int start = cu_seqlens[b_id] / cp_size;
    int end = cu_seqlens[b_id + 1] / cp_size;
    int t_id = s_id + start;
    if (t_id >= end) return;
    dk_offset_base = t_id * stride_dk_s;
    dv_offset_base = t_id * stride_dv_s;
    dkv_offset_base = t_id * dkv_stride_s;
    demb_offset_base = t_id * o_demb_stride_s;
    cur_seqlens = end - start;
  } else {
    dk_offset_base = s_id * stride_dk_s + b_id * stride_dk_b;
    dv_offset_base = s_id * stride_dv_s + b_id * stride_dv_b;
    dkv_offset_base = s_id * dkv_stride_s + b_id * dkv_stride_b;
    demb_offset_base = s_id * o_demb_stride_s + b_id * o_demb_stride_b;
    cur_seqlens = s;
  }

  int s_id_for_freqs = s_id;
  if (cp_size > 1) {
    assert(cur_seqlens % 2 == 0);
    if (s_id < cur_seqlens / 2) {
      s_id_for_freqs += cp_rank * cur_seqlens / 2;
    } else {
      s_id_for_freqs += cur_seqlens * cp_size - (cp_rank + 1) * cur_seqlens / 2 - cur_seqlens / 2;
    }
  }

  const int half_emb = emb_dim / 2;

  extern __shared__ float shared_mem[];
  float *sh_cos_L = shared_mem;
  float *sh_sin_L = shared_mem + half_emb;
  float *sh_cos_R = shared_mem + half_emb * 2;
  float *sh_sin_R = shared_mem + half_emb * 3;
  float *sh_dL_accum = shared_mem + half_emb * 4;
  float *sh_dR_accum = shared_mem + half_emb * 5;

  int tid = threadIdx.x * blockDim.y + threadIdx.y;
  int block_size = blockDim.x * blockDim.y;
  for (int i = tid; i < half_emb; i += block_size) {
    sh_cos_L[i] = cos_table[s_id_for_freqs * emb_dim + i];
    sh_sin_L[i] = sin_table[s_id_for_freqs * emb_dim + i];
    sh_cos_R[i] = cos_table[s_id_for_freqs * emb_dim + half_emb + i];
    sh_sin_R[i] = sin_table[s_id_for_freqs * emb_dim + half_emb + i];
    sh_dL_accum[i] = 0.0f;
    sh_dR_accum[i] = 0.0f;
  }
  __syncthreads();

#pragma unroll
  for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
#pragma unroll
    for (int d_id = threadIdx.x; d_id < k_dim; d_id += blockDim.x) {
      int dk_off = dk_offset_base + h_id * stride_dk_h + d_id;
      int dkv_off = dkv_offset_base + h_id * o_dkv_d + d_id;
      dkv[dkv_off] = dk[dk_off];
    }

#pragma unroll
    for (int d_id = threadIdx.x; d_id < v_dim; d_id += blockDim.x) {
      int dv_off = dv_offset_base + h_id * stride_dv_h + d_id;
      int dkv_off = dkv_offset_base + h_id * o_dkv_d + k_dim + d_id;
      dkv[dkv_off] = dv[dv_off];
    }

#pragma unroll
    for (int i = threadIdx.x; i < half_emb; i += blockDim.x) {
      int left_off = dk_offset_base + h_id * stride_dk_h + k_dim + i;
      int right_off = dk_offset_base + h_id * stride_dk_h + k_dim + half_emb + i;
      atomicAdd(&sh_dL_accum[i], static_cast<float>(dk[left_off]));
      atomicAdd(&sh_dR_accum[i], static_cast<float>(dk[right_off]));
    }
  }
  __syncthreads();

  for (int i = tid; i < half_emb; i += block_size) {
    float dL = sh_dL_accum[i];
    float dR = sh_dR_accum[i];
    float dx1 = dL * sh_cos_L[i] + dR * sh_sin_R[i];
    float dx2 = -dL * sh_sin_L[i] + dR * sh_cos_R[i];
    d_emb[demb_offset_base + 2 * i] = static_cast<scalar_t>(dx1);
    d_emb[demb_offset_base + 2 * i + 1] = static_cast<scalar_t>(dx2);
  }
}

// ---------------------------------------------------------------------------
// MLA Launchers
// ---------------------------------------------------------------------------

template <typename scalar_t>
void mla_rope_q_forward_launcher(
    const scalar_t *input, const int *cu_seqlens, const float *cos_table, const float *sin_table,
    scalar_t *output, const NVTE_QKV_Format qkv_format, const int cp_size, const int cp_rank,
    const int s, const int b, const int h, const int d, const int qk_head_dim, const int emb_dim,
    const int stride_s_or_t, const int stride_b, const int stride_h,
    cudaStream_t stream) {
  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(s, b);
  dim3 threads(THREADS_PER_WARP, warps_per_block);
  const int shared_mem_size = 4 * (emb_dim / 2) * sizeof(float);

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

  mla_rope_q_forward_kernel<<<blocks, threads, shared_mem_size, stream>>>(
      input, cu_seqlens, cos_table, sin_table, output, cp_size, cp_rank, s, h, qk_head_dim,
      emb_dim, stride_s_or_t, stride_b, stride_h, o_stride_s_or_t, o_stride_b, o_stride_h);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename scalar_t>
void mla_rope_q_backward_launcher(
    const scalar_t *grad_out, const int *cu_seqlens, const float *cos_table,
    const float *sin_table, scalar_t *grad_in, const NVTE_QKV_Format qkv_format, const int cp_size,
    const int cp_rank, const int s, const int b, const int h, const int d, const int qk_head_dim,
    const int emb_dim, const int stride_s_or_t, const int stride_b, const int stride_h,
    cudaStream_t stream) {
  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(s, b);
  dim3 threads(THREADS_PER_WARP, warps_per_block);
  const int shared_mem_size = 4 * (emb_dim / 2) * sizeof(float);

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

  mla_rope_q_backward_kernel<<<blocks, threads, shared_mem_size, stream>>>(
      grad_out, cu_seqlens, cos_table, sin_table, grad_in, cp_size, cp_rank, s, h, qk_head_dim,
      emb_dim, stride_s_or_t, stride_b, stride_h, o_stride_s_or_t, o_stride_b, o_stride_h);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename scalar_t>
void mla_rope_kv_forward_launcher(
    const scalar_t *kv, const scalar_t *k_pos_emb, const float *cos_table,
    const float *sin_table, scalar_t *o_key, scalar_t *o_value, const int *cu_seqlens,
    const NVTE_QKV_Format qkv_format, const int cp_size, const int cp_rank, const int s,
    const int b, const int h, const int k_dim, const int v_dim, const int emb_dim,
    const int stride_kv_s, const int stride_kv_b, const int stride_kv_h, const int stride_emb_s,
    const int stride_emb_b, cudaStream_t stream) {
  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(s, b);
  dim3 threads(THREADS_PER_WARP, warps_per_block);
  const int o_key_d = k_dim + emb_dim;
  const int o_val_d = v_dim;
  const int shared_mem_size = 6 * (emb_dim / 2) * sizeof(float);

  int okey_stride_s, okey_stride_b, oval_stride_s, oval_stride_b;
  if (qkv_format == NVTE_QKV_Format::NVTE_THD) {
    okey_stride_s = h * o_key_d;  okey_stride_b = 0;
    oval_stride_s = h * o_val_d;  oval_stride_b = 0;
  } else if (qkv_format == NVTE_QKV_Format::NVTE_SBHD) {
    okey_stride_s = b * h * o_key_d;  okey_stride_b = h * o_key_d;
    oval_stride_s = b * h * o_val_d;  oval_stride_b = h * o_val_d;
  } else {
    okey_stride_s = h * o_key_d;       okey_stride_b = s * h * o_key_d;
    oval_stride_s = h * o_val_d;       oval_stride_b = s * h * o_val_d;
  }

  mla_rope_kv_forward_kernel<<<blocks, threads, shared_mem_size, stream>>>(
      kv, k_pos_emb, cos_table, sin_table, o_key, o_value, cu_seqlens, cp_size, cp_rank, s, h,
      k_dim, v_dim, emb_dim, stride_kv_s, stride_kv_b, stride_kv_h, stride_emb_s, stride_emb_b,
      okey_stride_s, okey_stride_b, o_key_d,
      oval_stride_s, oval_stride_b, o_val_d);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename scalar_t>
void mla_rope_kv_backward_launcher(
    const scalar_t *dk, const scalar_t *dv, const float *cos_table, const float *sin_table,
    scalar_t *dkv, scalar_t *d_emb, const int *cu_seqlens, const NVTE_QKV_Format qkv_format,
    const int cp_size, const int cp_rank, const int s, const int b, const int h, const int k_dim,
    const int v_dim, const int emb_dim, const int stride_dk_s, const int stride_dk_b,
    const int stride_dk_h, const int stride_dv_s, const int stride_dv_b, const int stride_dv_h,
    const int o_demb_stride_s, const int o_demb_stride_b, cudaStream_t stream) {
  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(s, b);
  dim3 threads(THREADS_PER_WARP, warps_per_block);
  const int o_dkv_d = k_dim + v_dim;
  const int shared_mem_size = 6 * (emb_dim / 2) * sizeof(float);

  int dkv_stride_s, dkv_stride_b;
  if (qkv_format == NVTE_QKV_Format::NVTE_THD) {
    dkv_stride_s = h * o_dkv_d;  dkv_stride_b = 0;
  } else if (qkv_format == NVTE_QKV_Format::NVTE_SBHD) {
    dkv_stride_s = b * h * o_dkv_d;  dkv_stride_b = h * o_dkv_d;
  } else {
    dkv_stride_s = h * o_dkv_d;       dkv_stride_b = s * h * o_dkv_d;
  }

  mla_rope_kv_backward_kernel<<<blocks, threads, shared_mem_size, stream>>>(
      dk, dv, cos_table, sin_table, dkv, d_emb, cu_seqlens, cp_size, cp_rank, s, h, k_dim,
      v_dim, emb_dim, stride_dk_s, stride_dk_b, stride_dk_h, stride_dv_s, stride_dv_b,
      stride_dv_h, dkv_stride_s, dkv_stride_b, o_dkv_d,
      o_demb_stride_s, o_demb_stride_b);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// MLA Internal C++ API (Tensor wrappers)
// ---------------------------------------------------------------------------

void mla_rope_q_forward(const Tensor &input, const Tensor &cu_seqlens,
                        const Tensor &cos_table, const Tensor &sin_table, Tensor *output,
                        const NVTE_QKV_Format qkv_format, const int cp_size,
                        const int cp_rank, const int s, const int b, const int h,
                        const int d, const int qk_head_dim, const int emb_dim,
                        const int stride_s_or_t, const int stride_b, const int stride_h,
                        cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.data.dtype, scalar_t,
      mla_rope_q_forward_launcher(
          reinterpret_cast<const scalar_t *>(input.data.dptr),
          reinterpret_cast<const int *>(cu_seqlens.data.dptr),
          reinterpret_cast<const float *>(cos_table.data.dptr),
          reinterpret_cast<const float *>(sin_table.data.dptr),
          reinterpret_cast<scalar_t *>(output->data.dptr), qkv_format, cp_size, cp_rank, s, b, h,
          d, qk_head_dim, emb_dim, stride_s_or_t, stride_b, stride_h, stream););
}

void mla_rope_q_backward(const Tensor &grad_out, const Tensor &cu_seqlens,
                          const Tensor &cos_table, const Tensor &sin_table,
                          Tensor *grad_in, const NVTE_QKV_Format qkv_format,
                          const int cp_size, const int cp_rank, const int s, const int b,
                          const int h, const int d, const int qk_head_dim, const int emb_dim,
                          const int stride_s_or_t, const int stride_b, const int stride_h,
                          cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      grad_out.data.dtype, scalar_t,
      mla_rope_q_backward_launcher(
          reinterpret_cast<const scalar_t *>(grad_out.data.dptr),
          reinterpret_cast<const int *>(cu_seqlens.data.dptr),
          reinterpret_cast<const float *>(cos_table.data.dptr),
          reinterpret_cast<const float *>(sin_table.data.dptr),
          reinterpret_cast<scalar_t *>(grad_in->data.dptr), qkv_format, cp_size, cp_rank, s, b, h,
          d, qk_head_dim, emb_dim, stride_s_or_t, stride_b, stride_h, stream););
}

void mla_rope_kv_forward(const Tensor &kv, const Tensor &k_pos_emb,
                          const Tensor &cos_table, const Tensor &sin_table, Tensor *o_key,
                          Tensor *o_value, const Tensor &cu_seqlens,
                          const NVTE_QKV_Format qkv_format, const int cp_size,
                          const int cp_rank, const int s, const int b, const int h,
                          const int k_dim, const int v_dim, const int emb_dim,
                          const int stride_kv_s, const int stride_kv_b,
                          const int stride_kv_h, const int stride_emb_s,
                          const int stride_emb_b, cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      kv.data.dtype, scalar_t,
      mla_rope_kv_forward_launcher(
          reinterpret_cast<const scalar_t *>(kv.data.dptr),
          reinterpret_cast<const scalar_t *>(k_pos_emb.data.dptr),
          reinterpret_cast<const float *>(cos_table.data.dptr),
          reinterpret_cast<const float *>(sin_table.data.dptr),
          reinterpret_cast<scalar_t *>(o_key->data.dptr),
          reinterpret_cast<scalar_t *>(o_value->data.dptr),
          reinterpret_cast<const int *>(cu_seqlens.data.dptr), qkv_format, cp_size, cp_rank, s, b,
          h, k_dim, v_dim, emb_dim, stride_kv_s, stride_kv_b, stride_kv_h, stride_emb_s,
          stride_emb_b, stream););
}

void mla_rope_kv_backward(const Tensor &dk, const Tensor &dv, const Tensor &cos_table,
                           const Tensor &sin_table, Tensor *dkv, Tensor *d_emb,
                           const Tensor &cu_seqlens, const NVTE_QKV_Format qkv_format,
                           const int cp_size, const int cp_rank, const int s, const int b,
                           const int h, const int k_dim, const int v_dim, const int emb_dim,
                           const int stride_dk_s, const int stride_dk_b,
                           const int stride_dk_h, const int stride_dv_s,
                           const int stride_dv_b, const int stride_dv_h,
                           const int o_demb_stride_s, const int o_demb_stride_b,
                           cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      dk.data.dtype, scalar_t,
      mla_rope_kv_backward_launcher(
          reinterpret_cast<const scalar_t *>(dk.data.dptr),
          reinterpret_cast<const scalar_t *>(dv.data.dptr),
          reinterpret_cast<const float *>(cos_table.data.dptr),
          reinterpret_cast<const float *>(sin_table.data.dptr),
          reinterpret_cast<scalar_t *>(dkv->data.dptr),
          reinterpret_cast<scalar_t *>(d_emb->data.dptr),
          reinterpret_cast<const int *>(cu_seqlens.data.dptr), qkv_format, cp_size, cp_rank, s, b,
          h, k_dim, v_dim, emb_dim, stride_dk_s, stride_dk_b, stride_dk_h, stride_dv_s,
          stride_dv_b, stride_dv_h, o_demb_stride_s, o_demb_stride_b, stream););
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

void nvte_mla_rope_q_forward(const NVTETensor input, const NVTETensor cu_seqlens,
                             const NVTETensor cos_table, const NVTETensor sin_table,
                             NVTETensor output, const NVTE_QKV_Format qkv_format,
                             const int cp_size, const int cp_rank, const int s, const int b,
                             const int h, const int d, const int qk_head_dim,
                             const int emb_dim, const int stride_s_or_t, const int stride_b,
                             const int stride_h, cudaStream_t stream) {
  NVTE_API_CALL(nvte_mla_rope_q_forward);
  using namespace transformer_engine;
  mla_rope_q_forward(*convertNVTETensorCheck(input), *convertNVTETensorCheck(cu_seqlens),
                     *convertNVTETensorCheck(cos_table), *convertNVTETensorCheck(sin_table),
                     convertNVTETensorCheck(output), qkv_format, cp_size, cp_rank, s, b, h,
                     d, qk_head_dim, emb_dim, stride_s_or_t, stride_b, stride_h, stream);
}

void nvte_mla_rope_q_backward(const NVTETensor grad_out, const NVTETensor cu_seqlens,
                              const NVTETensor cos_table, const NVTETensor sin_table,
                              NVTETensor grad_in, const NVTE_QKV_Format qkv_format,
                              const int cp_size, const int cp_rank, const int s, const int b,
                              const int h, const int d, const int qk_head_dim,
                              const int emb_dim, const int stride_s_or_t, const int stride_b,
                              const int stride_h, cudaStream_t stream) {
  NVTE_API_CALL(nvte_mla_rope_q_backward);
  using namespace transformer_engine;
  mla_rope_q_backward(*convertNVTETensorCheck(grad_out), *convertNVTETensorCheck(cu_seqlens),
                      *convertNVTETensorCheck(cos_table), *convertNVTETensorCheck(sin_table),
                      convertNVTETensorCheck(grad_in), qkv_format, cp_size, cp_rank, s, b, h,
                      d, qk_head_dim, emb_dim, stride_s_or_t, stride_b, stride_h, stream);
}

void nvte_mla_rope_kv_forward(
    const NVTETensor kv, const NVTETensor k_pos_emb, const NVTETensor cos_table,
    const NVTETensor sin_table, NVTETensor o_key, NVTETensor o_value, const NVTETensor cu_seqlens,
    const NVTE_QKV_Format qkv_format, const int cp_size, const int cp_rank, const int s,
    const int b, const int h, const int k_dim, const int v_dim, const int emb_dim,
    const int stride_kv_s, const int stride_kv_b, const int stride_kv_h, const int stride_emb_s,
    const int stride_emb_b, cudaStream_t stream) {
  NVTE_API_CALL(nvte_mla_rope_kv_forward);
  using namespace transformer_engine;
  mla_rope_kv_forward(
      *convertNVTETensorCheck(kv), *convertNVTETensorCheck(k_pos_emb),
      *convertNVTETensorCheck(cos_table), *convertNVTETensorCheck(sin_table),
      convertNVTETensorCheck(o_key), convertNVTETensorCheck(o_value),
      *convertNVTETensorCheck(cu_seqlens), qkv_format, cp_size, cp_rank, s, b, h, k_dim, v_dim,
      emb_dim, stride_kv_s, stride_kv_b, stride_kv_h, stride_emb_s, stride_emb_b, stream);
}

void nvte_mla_rope_kv_backward(
    const NVTETensor dk, const NVTETensor dv, const NVTETensor cos_table,
    const NVTETensor sin_table, NVTETensor dkv, NVTETensor d_emb, const NVTETensor cu_seqlens,
    const NVTE_QKV_Format qkv_format, const int cp_size, const int cp_rank, const int s,
    const int b, const int h, const int k_dim, const int v_dim, const int emb_dim,
    const int stride_dk_s, const int stride_dk_b, const int stride_dk_h, const int stride_dv_s,
    const int stride_dv_b, const int stride_dv_h, const int o_demb_stride_s,
    const int o_demb_stride_b, cudaStream_t stream) {
  NVTE_API_CALL(nvte_mla_rope_kv_backward);
  using namespace transformer_engine;
  mla_rope_kv_backward(
      *convertNVTETensorCheck(dk), *convertNVTETensorCheck(dv),
      *convertNVTETensorCheck(cos_table), *convertNVTETensorCheck(sin_table),
      convertNVTETensorCheck(dkv), convertNVTETensorCheck(d_emb),
      *convertNVTETensorCheck(cu_seqlens), qkv_format, cp_size, cp_rank, s, b, h, k_dim, v_dim,
      emb_dim, stride_dk_s, stride_dk_b, stride_dk_h, stride_dv_s, stride_dv_b, stride_dv_h,
      o_demb_stride_s, o_demb_stride_b, stream);
}
