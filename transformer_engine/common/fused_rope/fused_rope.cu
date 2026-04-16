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

// ============================================================
// MLA YARN RoPE – Q forward device function
//
// Input emb region layout:  interleaved pairs (x[2i], x[2i+1])
// Output emb region layout: non-interleaved   (first_half | second_half)
// cos/sin: [max_seq, emb_dim], split left ([:half]) / right ([half:])
// ============================================================
template <typename scalar_t>
__device__ void mla_rope_q_block_forward(
    const scalar_t *src, const float *cos, const float *sin, scalar_t *dst,
    const int s_id, const int offset_block, const int offset_block_dst,
    const int h, const int qk_head_dim, const int emb_dim,
    const int stride_h, const int o_stride_h) {
  const int half_dim = emb_dim / 2;
  extern __shared__ float smem[];
  float *scos = smem;
  float *ssin = smem + emb_dim;
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int nthreads = blockDim.x * blockDim.y;
  for (int i = tid; i < emb_dim; i += nthreads) {
    scos[i] = cos[s_id * emb_dim + i];
    ssin[i] = sin[s_id * emb_dim + i];
  }
  __syncthreads();
  for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y)
    for (int d = threadIdx.x; d < qk_head_dim; d += blockDim.x)
      dst[offset_block_dst + h_id * o_stride_h + d] = src[offset_block + h_id * stride_h + d];
  for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
    for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
      float x1 = static_cast<float>(src[offset_block + h_id * stride_h + qk_head_dim + 2 * i]);
      float x2 = static_cast<float>(src[offset_block + h_id * stride_h + qk_head_dim + 2 * i + 1]);
      dst[offset_block_dst + h_id * o_stride_h + qk_head_dim + i] =
          static_cast<scalar_t>(x1 * scos[i] - x2 * ssin[i]);
      dst[offset_block_dst + h_id * o_stride_h + qk_head_dim + half_dim + i] =
          static_cast<scalar_t>(x2 * scos[half_dim + i] + x1 * ssin[half_dim + i]);
    }
  }
}

// ============================================================
// MLA YARN RoPE – Q backward device function
//
// Inverts the forward rotation: non-interleaved grad → interleaved input grad.
// dx1[2i]   = dout_l[i]*cos_l[i]  + dout_r[i]*sin_r[i]
// dx2[2i+1] = -dout_l[i]*sin_l[i] + dout_r[i]*cos_r[i]
// ============================================================
template <typename scalar_t>
__device__ void mla_rope_q_block_backward(
    const scalar_t *grad, const float *cos, const float *sin, scalar_t *dst,
    const int s_id, const int offset_block, const int offset_block_dst,
    const int h, const int qk_head_dim, const int emb_dim,
    const int stride_h, const int o_stride_h) {
  const int half_dim = emb_dim / 2;
  extern __shared__ float smem[];
  float *scos = smem;
  float *ssin = smem + emb_dim;
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int nthreads = blockDim.x * blockDim.y;
  for (int i = tid; i < emb_dim; i += nthreads) {
    scos[i] = cos[s_id * emb_dim + i];
    ssin[i] = sin[s_id * emb_dim + i];
  }
  __syncthreads();
  for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y)
    for (int d = threadIdx.x; d < qk_head_dim; d += blockDim.x)
      dst[offset_block_dst + h_id * o_stride_h + d] = grad[offset_block + h_id * stride_h + d];
  for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
    for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
      float dl = static_cast<float>(grad[offset_block + h_id * stride_h + qk_head_dim + i]);
      float dr = static_cast<float>(grad[offset_block + h_id * stride_h + qk_head_dim + half_dim + i]);
      dst[offset_block_dst + h_id * o_stride_h + qk_head_dim + 2 * i] =
          static_cast<scalar_t>(dl * scos[i] + dr * ssin[half_dim + i]);
      dst[offset_block_dst + h_id * o_stride_h + qk_head_dim + 2 * i + 1] =
          static_cast<scalar_t>(-dl * ssin[i] + dr * scos[half_dim + i]);
    }
  }
}

// ============================================================
// MLA YARN RoPE – KV forward device function
//
// Splits packed kv → key (k_dim) + value (v_dim).
// Rotates per-token k_pos_emb (interleaved, h=1) and appends to key.
// Output key width = k_dim + emb_dim.
// ============================================================
template <typename scalar_t>
__device__ void mla_rope_kv_block_forward(
    const scalar_t *kv, const scalar_t *k_pos_emb,
    const float *cos, const float *sin,
    scalar_t *key_out, scalar_t *val_out,
    const int s_id, const int kv_offset, const int emb_offset,
    const int key_offset_dst, const int val_offset_dst,
    const int h, const int k_dim, const int v_dim, const int emb_dim,
    const int kv_stride_h, const int key_stride_h, const int val_stride_h) {
  const int half_dim = emb_dim / 2;
  extern __shared__ float smem[];
  float *scos = smem;
  float *ssin = smem + emb_dim;
  float *srot = smem + 2 * emb_dim;  // rotated k_pos_emb, broadcast to all heads
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int nthreads = blockDim.x * blockDim.y;
  for (int i = tid; i < emb_dim; i += nthreads) {
    scos[i] = cos[s_id * emb_dim + i];
    ssin[i] = sin[s_id * emb_dim + i];
  }
  __syncthreads();
  if (threadIdx.y == 0) {
    for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
      float x1 = static_cast<float>(k_pos_emb[emb_offset + 2 * i]);
      float x2 = static_cast<float>(k_pos_emb[emb_offset + 2 * i + 1]);
      srot[i] = x1 * scos[i] - x2 * ssin[i];
      srot[half_dim + i] = x2 * scos[half_dim + i] + x1 * ssin[half_dim + i];
    }
  }
  __syncthreads();
  for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
    for (int d = threadIdx.x; d < k_dim; d += blockDim.x)
      key_out[key_offset_dst + h_id * key_stride_h + d] = kv[kv_offset + h_id * kv_stride_h + d];
    for (int d = threadIdx.x; d < emb_dim; d += blockDim.x)
      key_out[key_offset_dst + h_id * key_stride_h + k_dim + d] =
          static_cast<scalar_t>(srot[d]);
    for (int d = threadIdx.x; d < v_dim; d += blockDim.x)
      val_out[val_offset_dst + h_id * val_stride_h + d] =
          kv[kv_offset + h_id * kv_stride_h + k_dim + d];
  }
}

// ============================================================
// MLA YARN RoPE – KV backward device function
//
// d_kv[:k_dim] = dk[:k_dim]; d_kv[k_dim:] = dv
// d_emb (interleaved) = sum over heads of inverse-rotated dk[k_dim:k_dim+emb_dim]
// ============================================================
template <typename scalar_t>
__device__ void mla_rope_kv_block_backward(
    const scalar_t *dk, const scalar_t *dv,
    const float *cos, const float *sin,
    scalar_t *d_kv_out, scalar_t *d_emb_out,
    const int s_id, const int dk_offset, const int dv_offset,
    const int d_kv_offset, const int d_emb_offset,
    const int h, const int k_dim, const int v_dim, const int emb_dim,
    const int dk_stride_h, const int dv_stride_h, const int d_kv_stride_h) {
  const int half_dim = emb_dim / 2;
  extern __shared__ float smem[];
  float *scos = smem;
  float *ssin = smem + emb_dim;
  float *sd_emb = smem + 2 * emb_dim;  // accumulator for d_emb (interleaved layout)
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int nthreads = blockDim.x * blockDim.y;
  for (int i = tid; i < emb_dim; i += nthreads) {
    scos[i] = cos[s_id * emb_dim + i];
    ssin[i] = sin[s_id * emb_dim + i];
    sd_emb[i] = 0.0f;
  }
  __syncthreads();
  for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
    for (int d = threadIdx.x; d < k_dim; d += blockDim.x)
      d_kv_out[d_kv_offset + h_id * d_kv_stride_h + d] =
          dk[dk_offset + h_id * dk_stride_h + d];
    for (int d = threadIdx.x; d < v_dim; d += blockDim.x)
      d_kv_out[d_kv_offset + h_id * d_kv_stride_h + k_dim + d] =
          dv[dv_offset + h_id * dv_stride_h + d];
    for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
      float dl = static_cast<float>(dk[dk_offset + h_id * dk_stride_h + k_dim + i]);
      float dr = static_cast<float>(dk[dk_offset + h_id * dk_stride_h + k_dim + half_dim + i]);
      atomicAdd(&sd_emb[2 * i],     dl * scos[i] + dr * ssin[half_dim + i]);
      atomicAdd(&sd_emb[2 * i + 1], -dl * ssin[i] + dr * scos[half_dim + i]);
    }
  }
  __syncthreads();
  for (int i = tid; i < emb_dim; i += nthreads)
    d_emb_out[d_emb_offset + i] = static_cast<scalar_t>(sd_emb[i]);
}

template <typename scalar_t>
__global__ void mla_rope_q_forward_kernel(
    const scalar_t *src, const int *cu_seqlens,
    const float *cos, const float *sin, scalar_t *dst,
    const int cp_size, const int cp_rank, const int s,
    const int h, const int qk_head_dim, const int emb_dim,
    const int stride_s_or_t, const int stride_b, const int stride_h,
    const int o_stride_s_or_t, const int o_stride_b, const int o_stride_h) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int offset_block, offset_block_dst, cur_seqlens;
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
  int s_id_for_cos = s_id;
  if (cp_size > 1) {
    assert(cur_seqlens % 2 == 0);
    if (s_id < cur_seqlens / 2)
      s_id_for_cos += cp_rank * cur_seqlens / 2;
    else
      s_id_for_cos += cur_seqlens * cp_size - (cp_rank + 1) * cur_seqlens / 2 - cur_seqlens / 2;
  }
  mla_rope_q_block_forward(src, cos, sin, dst, s_id_for_cos,
                            offset_block, offset_block_dst,
                            h, qk_head_dim, emb_dim, stride_h, o_stride_h);
}

template <typename scalar_t>
__global__ void mla_rope_q_backward_kernel(
    const scalar_t *grad, const int *cu_seqlens,
    const float *cos, const float *sin, scalar_t *dst,
    const int cp_size, const int cp_rank, const int s,
    const int h, const int qk_head_dim, const int emb_dim,
    const int stride_s_or_t, const int stride_b, const int stride_h,
    const int o_stride_s_or_t, const int o_stride_b, const int o_stride_h) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int offset_block, offset_block_dst, cur_seqlens;
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
  int s_id_for_cos = s_id;
  if (cp_size > 1) {
    assert(cur_seqlens % 2 == 0);
    if (s_id < cur_seqlens / 2)
      s_id_for_cos += cp_rank * cur_seqlens / 2;
    else
      s_id_for_cos += cur_seqlens * cp_size - (cp_rank + 1) * cur_seqlens / 2 - cur_seqlens / 2;
  }
  mla_rope_q_block_backward(grad, cos, sin, dst, s_id_for_cos,
                             offset_block, offset_block_dst,
                             h, qk_head_dim, emb_dim, stride_h, o_stride_h);
}

template <typename scalar_t>
__global__ void mla_rope_kv_forward_kernel(
    const scalar_t *kv, const scalar_t *k_pos_emb,
    const int *cu_seqlens,
    const float *cos, const float *sin,
    scalar_t *key_out, scalar_t *val_out,
    const int cp_size, const int cp_rank, const int s,
    const int h, const int k_dim, const int v_dim, const int emb_dim,
    const int kv_stride_s_or_t, const int kv_stride_b, const int kv_stride_h,
    const int emb_stride_s_or_t, const int emb_stride_b,
    const int key_stride_s_or_t, const int key_stride_b, const int key_stride_h,
    const int val_stride_s_or_t, const int val_stride_b, const int val_stride_h) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int kv_offset, emb_offset, key_offset_dst, val_offset_dst, cur_seqlens;
  if (cu_seqlens != nullptr) {
    int start = cu_seqlens[b_id] / cp_size;
    int end = cu_seqlens[b_id + 1] / cp_size;
    int t_id = s_id + start;
    if (t_id >= end) return;
    kv_offset = t_id * kv_stride_s_or_t;
    emb_offset = t_id * emb_stride_s_or_t;
    key_offset_dst = t_id * key_stride_s_or_t;
    val_offset_dst = t_id * val_stride_s_or_t;
    cur_seqlens = end - start;
  } else {
    kv_offset = s_id * kv_stride_s_or_t + b_id * kv_stride_b;
    emb_offset = s_id * emb_stride_s_or_t + b_id * emb_stride_b;
    key_offset_dst = s_id * key_stride_s_or_t + b_id * key_stride_b;
    val_offset_dst = s_id * val_stride_s_or_t + b_id * val_stride_b;
    cur_seqlens = s;
  }
  int s_id_for_cos = s_id;
  if (cp_size > 1) {
    assert(cur_seqlens % 2 == 0);
    if (s_id < cur_seqlens / 2)
      s_id_for_cos += cp_rank * cur_seqlens / 2;
    else
      s_id_for_cos += cur_seqlens * cp_size - (cp_rank + 1) * cur_seqlens / 2 - cur_seqlens / 2;
  }
  mla_rope_kv_block_forward(kv, k_pos_emb, cos, sin, key_out, val_out,
                             s_id_for_cos, kv_offset, emb_offset,
                             key_offset_dst, val_offset_dst,
                             h, k_dim, v_dim, emb_dim,
                             kv_stride_h, key_stride_h, val_stride_h);
}

template <typename scalar_t>
__global__ void mla_rope_kv_backward_kernel(
    const scalar_t *dk, const scalar_t *dv,
    const int *cu_seqlens,
    const float *cos, const float *sin,
    scalar_t *d_kv_out, scalar_t *d_emb_out,
    const int cp_size, const int cp_rank, const int s,
    const int h, const int k_dim, const int v_dim, const int emb_dim,
    const int dk_stride_s_or_t, const int dk_stride_b, const int dk_stride_h,
    const int dv_stride_s_or_t, const int dv_stride_b, const int dv_stride_h,
    const int d_kv_stride_s_or_t, const int d_kv_stride_b, const int d_kv_stride_h,
    const int d_emb_stride_s_or_t, const int d_emb_stride_b) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int dk_offset, dv_offset, d_kv_offset, d_emb_offset, cur_seqlens;
  if (cu_seqlens != nullptr) {
    int start = cu_seqlens[b_id] / cp_size;
    int end = cu_seqlens[b_id + 1] / cp_size;
    int t_id = s_id + start;
    if (t_id >= end) return;
    dk_offset = t_id * dk_stride_s_or_t;
    dv_offset = t_id * dv_stride_s_or_t;
    d_kv_offset = t_id * d_kv_stride_s_or_t;
    d_emb_offset = t_id * d_emb_stride_s_or_t;
    cur_seqlens = end - start;
  } else {
    dk_offset = s_id * dk_stride_s_or_t + b_id * dk_stride_b;
    dv_offset = s_id * dv_stride_s_or_t + b_id * dv_stride_b;
    d_kv_offset = s_id * d_kv_stride_s_or_t + b_id * d_kv_stride_b;
    d_emb_offset = s_id * d_emb_stride_s_or_t + b_id * d_emb_stride_b;
    cur_seqlens = s;
  }
  int s_id_for_cos = s_id;
  if (cp_size > 1) {
    assert(cur_seqlens % 2 == 0);
    if (s_id < cur_seqlens / 2)
      s_id_for_cos += cp_rank * cur_seqlens / 2;
    else
      s_id_for_cos += cur_seqlens * cp_size - (cp_rank + 1) * cur_seqlens / 2 - cur_seqlens / 2;
  }
  mla_rope_kv_block_backward(dk, dv, cos, sin, d_kv_out, d_emb_out,
                              s_id_for_cos, dk_offset, dv_offset,
                              d_kv_offset, d_emb_offset,
                              h, k_dim, v_dim, emb_dim,
                              dk_stride_h, dv_stride_h, d_kv_stride_h);
}

template <typename scalar_t>
void mla_rope_q_forward_launcher(
    const scalar_t *input, const int *cu_seqlens,
    const float *cos, const float *sin, scalar_t *output,
    const NVTE_QKV_Format qkv_format,
    const int cp_size, const int cp_rank,
    const int s, const int b, const int h,
    const int qk_head_dim, const int emb_dim,
    const int stride_s_or_t, const int stride_b, const int stride_h,
    cudaStream_t stream) {
  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(s, b);
  dim3 threads(THREADS_PER_WARP, warps_per_block);
  const int shared_mem_size = 2 * emb_dim * sizeof(float);
  const int total_d = qk_head_dim + emb_dim;
  int o_stride_s_or_t, o_stride_b;
  if (qkv_format == NVTE_QKV_Format::NVTE_THD) {
    o_stride_s_or_t = h * total_d;
    o_stride_b = 0;
  } else if (qkv_format == NVTE_QKV_Format::NVTE_SBHD) {
    o_stride_s_or_t = b * h * total_d;
    o_stride_b = h * total_d;
  } else {
    o_stride_s_or_t = h * total_d;
    o_stride_b = s * h * total_d;
  }
  const int o_stride_h = total_d;
  mla_rope_q_forward_kernel<<<blocks, threads, shared_mem_size, stream>>>(
      input, cu_seqlens, cos, sin, output,
      cp_size, cp_rank, s, h, qk_head_dim, emb_dim,
      stride_s_or_t, stride_b, stride_h, o_stride_s_or_t, o_stride_b, o_stride_h);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename scalar_t>
void mla_rope_q_backward_launcher(
    const scalar_t *grad, const int *cu_seqlens,
    const float *cos, const float *sin, scalar_t *dst,
    const NVTE_QKV_Format qkv_format,
    const int cp_size, const int cp_rank,
    const int s, const int b, const int h,
    const int qk_head_dim, const int emb_dim,
    const int stride_s_or_t, const int stride_b, const int stride_h,
    cudaStream_t stream) {
  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(s, b);
  dim3 threads(THREADS_PER_WARP, warps_per_block);
  const int shared_mem_size = 2 * emb_dim * sizeof(float);
  const int total_d = qk_head_dim + emb_dim;
  int o_stride_s_or_t, o_stride_b;
  if (qkv_format == NVTE_QKV_Format::NVTE_THD) {
    o_stride_s_or_t = h * total_d;
    o_stride_b = 0;
  } else if (qkv_format == NVTE_QKV_Format::NVTE_SBHD) {
    o_stride_s_or_t = b * h * total_d;
    o_stride_b = h * total_d;
  } else {
    o_stride_s_or_t = h * total_d;
    o_stride_b = s * h * total_d;
  }
  const int o_stride_h = total_d;
  mla_rope_q_backward_kernel<<<blocks, threads, shared_mem_size, stream>>>(
      grad, cu_seqlens, cos, sin, dst,
      cp_size, cp_rank, s, h, qk_head_dim, emb_dim,
      stride_s_or_t, stride_b, stride_h, o_stride_s_or_t, o_stride_b, o_stride_h);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename scalar_t>
void mla_rope_kv_forward_launcher(
    const scalar_t *kv, const scalar_t *k_pos_emb,
    const int *cu_seqlens,
    const float *cos, const float *sin,
    scalar_t *key_out, scalar_t *val_out,
    const NVTE_QKV_Format qkv_format,
    const int cp_size, const int cp_rank,
    const int s, const int b, const int h,
    const int k_dim, const int v_dim, const int emb_dim,
    const int kv_stride_s_or_t, const int kv_stride_b, const int kv_stride_h,
    const int emb_stride_s_or_t, const int emb_stride_b,
    cudaStream_t stream) {
  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(s, b);
  dim3 threads(THREADS_PER_WARP, warps_per_block);
  const int shared_mem_size = 3 * emb_dim * sizeof(float);  // cos, sin, srot
  const int key_d = k_dim + emb_dim;
  int key_stride_s_or_t, key_stride_b, val_stride_s_or_t, val_stride_b;
  if (qkv_format == NVTE_QKV_Format::NVTE_THD) {
    key_stride_s_or_t = h * key_d;
    key_stride_b = 0;
    val_stride_s_or_t = h * v_dim;
    val_stride_b = 0;
  } else if (qkv_format == NVTE_QKV_Format::NVTE_SBHD) {
    key_stride_s_or_t = b * h * key_d;
    key_stride_b = h * key_d;
    val_stride_s_or_t = b * h * v_dim;
    val_stride_b = h * v_dim;
  } else {
    key_stride_s_or_t = h * key_d;
    key_stride_b = s * h * key_d;
    val_stride_s_or_t = h * v_dim;
    val_stride_b = s * h * v_dim;
  }
  const int key_stride_h = key_d;
  const int val_stride_h = v_dim;
  mla_rope_kv_forward_kernel<<<blocks, threads, shared_mem_size, stream>>>(
      kv, k_pos_emb, cu_seqlens, cos, sin, key_out, val_out,
      cp_size, cp_rank, s, h, k_dim, v_dim, emb_dim,
      kv_stride_s_or_t, kv_stride_b, kv_stride_h,
      emb_stride_s_or_t, emb_stride_b,
      key_stride_s_or_t, key_stride_b, key_stride_h,
      val_stride_s_or_t, val_stride_b, val_stride_h);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename scalar_t>
void mla_rope_kv_backward_launcher(
    const scalar_t *dk, const scalar_t *dv,
    const int *cu_seqlens,
    const float *cos, const float *sin,
    scalar_t *d_kv_out, scalar_t *d_emb_out,
    const NVTE_QKV_Format qkv_format,
    const int cp_size, const int cp_rank,
    const int s, const int b, const int h,
    const int k_dim, const int v_dim, const int emb_dim,
    const int dk_stride_s_or_t, const int dk_stride_b, const int dk_stride_h,
    const int dv_stride_s_or_t, const int dv_stride_b, const int dv_stride_h,
    const int d_emb_stride_s_or_t, const int d_emb_stride_b,
    cudaStream_t stream) {
  int warps_per_block = h < 16 ? 4 : 8;
  dim3 blocks(s, b);
  dim3 threads(THREADS_PER_WARP, warps_per_block);
  const int shared_mem_size = 3 * emb_dim * sizeof(float);  // cos, sin, d_emb accum
  const int kv_d = k_dim + v_dim;
  int d_kv_stride_s_or_t, d_kv_stride_b;
  if (qkv_format == NVTE_QKV_Format::NVTE_THD) {
    d_kv_stride_s_or_t = h * kv_d;
    d_kv_stride_b = 0;
  } else if (qkv_format == NVTE_QKV_Format::NVTE_SBHD) {
    d_kv_stride_s_or_t = b * h * kv_d;
    d_kv_stride_b = h * kv_d;
  } else {
    d_kv_stride_s_or_t = h * kv_d;
    d_kv_stride_b = s * h * kv_d;
  }
  const int d_kv_stride_h = kv_d;
  mla_rope_kv_backward_kernel<<<blocks, threads, shared_mem_size, stream>>>(
      dk, dv, cu_seqlens, cos, sin, d_kv_out, d_emb_out,
      cp_size, cp_rank, s, h, k_dim, v_dim, emb_dim,
      dk_stride_s_or_t, dk_stride_b, dk_stride_h,
      dv_stride_s_or_t, dv_stride_b, dv_stride_h,
      d_kv_stride_s_or_t, d_kv_stride_b, d_kv_stride_h,
      d_emb_stride_s_or_t, d_emb_stride_b);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void mla_rope_q_forward(const Tensor &input, const Tensor &cu_seqlens,
                        const Tensor &cos, const Tensor &sin, Tensor *output,
                        const NVTE_QKV_Format qkv_format, const int cp_size, const int cp_rank,
                        const int s, const int b, const int h, const int qk_head_dim,
                        const int emb_dim, const int stride_s_or_t, const int stride_b,
                        const int stride_h, cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.data.dtype, scalar_t,
      mla_rope_q_forward_launcher(
          reinterpret_cast<const scalar_t *>(input.data.dptr),
          reinterpret_cast<const int *>(cu_seqlens.data.dptr),
          reinterpret_cast<const float *>(cos.data.dptr),
          reinterpret_cast<const float *>(sin.data.dptr),
          reinterpret_cast<scalar_t *>(output->data.dptr),
          qkv_format, cp_size, cp_rank, s, b, h, qk_head_dim, emb_dim,
          stride_s_or_t, stride_b, stride_h, stream););
}

void mla_rope_q_backward(const Tensor &grad, const Tensor &cu_seqlens,
                         const Tensor &cos, const Tensor &sin, Tensor *dst,
                         const NVTE_QKV_Format qkv_format, const int cp_size, const int cp_rank,
                         const int s, const int b, const int h, const int qk_head_dim,
                         const int emb_dim, const int stride_s_or_t, const int stride_b,
                         const int stride_h, cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      grad.data.dtype, scalar_t,
      mla_rope_q_backward_launcher(
          reinterpret_cast<const scalar_t *>(grad.data.dptr),
          reinterpret_cast<const int *>(cu_seqlens.data.dptr),
          reinterpret_cast<const float *>(cos.data.dptr),
          reinterpret_cast<const float *>(sin.data.dptr),
          reinterpret_cast<scalar_t *>(dst->data.dptr),
          qkv_format, cp_size, cp_rank, s, b, h, qk_head_dim, emb_dim,
          stride_s_or_t, stride_b, stride_h, stream););
}

void mla_rope_kv_forward(const Tensor &kv, const Tensor &k_pos_emb, const Tensor &cu_seqlens,
                         const Tensor &cos, const Tensor &sin,
                         Tensor *key_out, Tensor *val_out,
                         const NVTE_QKV_Format qkv_format, const int cp_size, const int cp_rank,
                         const int s, const int b, const int h,
                         const int k_dim, const int v_dim, const int emb_dim,
                         const int kv_stride_s_or_t, const int kv_stride_b, const int kv_stride_h,
                         const int emb_stride_s_or_t, const int emb_stride_b,
                         cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      kv.data.dtype, scalar_t,
      mla_rope_kv_forward_launcher(
          reinterpret_cast<const scalar_t *>(kv.data.dptr),
          reinterpret_cast<const scalar_t *>(k_pos_emb.data.dptr),
          reinterpret_cast<const int *>(cu_seqlens.data.dptr),
          reinterpret_cast<const float *>(cos.data.dptr),
          reinterpret_cast<const float *>(sin.data.dptr),
          reinterpret_cast<scalar_t *>(key_out->data.dptr),
          reinterpret_cast<scalar_t *>(val_out->data.dptr),
          qkv_format, cp_size, cp_rank, s, b, h, k_dim, v_dim, emb_dim,
          kv_stride_s_or_t, kv_stride_b, kv_stride_h,
          emb_stride_s_or_t, emb_stride_b, stream););
}

void mla_rope_kv_backward(const Tensor &dk, const Tensor &dv, const Tensor &cu_seqlens,
                          const Tensor &cos, const Tensor &sin,
                          Tensor *d_kv_out, Tensor *d_emb_out,
                          const NVTE_QKV_Format qkv_format, const int cp_size, const int cp_rank,
                          const int s, const int b, const int h,
                          const int k_dim, const int v_dim, const int emb_dim,
                          const int dk_stride_s_or_t, const int dk_stride_b, const int dk_stride_h,
                          const int dv_stride_s_or_t, const int dv_stride_b, const int dv_stride_h,
                          const int d_emb_stride_s_or_t, const int d_emb_stride_b,
                          cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      dk.data.dtype, scalar_t,
      mla_rope_kv_backward_launcher(
          reinterpret_cast<const scalar_t *>(dk.data.dptr),
          reinterpret_cast<const scalar_t *>(dv.data.dptr),
          reinterpret_cast<const int *>(cu_seqlens.data.dptr),
          reinterpret_cast<const float *>(cos.data.dptr),
          reinterpret_cast<const float *>(sin.data.dptr),
          reinterpret_cast<scalar_t *>(d_kv_out->data.dptr),
          reinterpret_cast<scalar_t *>(d_emb_out->data.dptr),
          qkv_format, cp_size, cp_rank, s, b, h, k_dim, v_dim, emb_dim,
          dk_stride_s_or_t, dk_stride_b, dk_stride_h,
          dv_stride_s_or_t, dv_stride_b, dv_stride_h,
          d_emb_stride_s_or_t, d_emb_stride_b, stream););
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
                              const NVTETensor cos, const NVTETensor sin, NVTETensor output,
                              const NVTE_QKV_Format qkv_format, const int cp_size, const int cp_rank,
                              const int s, const int b, const int h, const int qk_head_dim,
                              const int emb_dim, const int stride_s_or_t, const int stride_b,
                              const int stride_h, cudaStream_t stream) {
  NVTE_API_CALL(nvte_mla_rope_q_forward);
  using namespace transformer_engine;
  mla_rope_q_forward(*convertNVTETensorCheck(input), *convertNVTETensorCheck(cu_seqlens),
                     *convertNVTETensorCheck(cos), *convertNVTETensorCheck(sin),
                     convertNVTETensorCheck(output), qkv_format, cp_size, cp_rank,
                     s, b, h, qk_head_dim, emb_dim, stride_s_or_t, stride_b, stride_h, stream);
}

void nvte_mla_rope_q_backward(const NVTETensor grad, const NVTETensor cu_seqlens,
                               const NVTETensor cos, const NVTETensor sin, NVTETensor dst,
                               const NVTE_QKV_Format qkv_format, const int cp_size, const int cp_rank,
                               const int s, const int b, const int h, const int qk_head_dim,
                               const int emb_dim, const int stride_s_or_t, const int stride_b,
                               const int stride_h, cudaStream_t stream) {
  NVTE_API_CALL(nvte_mla_rope_q_backward);
  using namespace transformer_engine;
  mla_rope_q_backward(*convertNVTETensorCheck(grad), *convertNVTETensorCheck(cu_seqlens),
                      *convertNVTETensorCheck(cos), *convertNVTETensorCheck(sin),
                      convertNVTETensorCheck(dst), qkv_format, cp_size, cp_rank,
                      s, b, h, qk_head_dim, emb_dim, stride_s_or_t, stride_b, stride_h, stream);
}

void nvte_mla_rope_kv_forward(const NVTETensor kv, const NVTETensor k_pos_emb,
                               const NVTETensor cu_seqlens, const NVTETensor cos,
                               const NVTETensor sin, NVTETensor key_out, NVTETensor val_out,
                               const NVTE_QKV_Format qkv_format, const int cp_size, const int cp_rank,
                               const int s, const int b, const int h,
                               const int k_dim, const int v_dim, const int emb_dim,
                               const int kv_stride_s_or_t, const int kv_stride_b,
                               const int kv_stride_h,
                               const int emb_stride_s_or_t, const int emb_stride_b,
                               cudaStream_t stream) {
  NVTE_API_CALL(nvte_mla_rope_kv_forward);
  using namespace transformer_engine;
  mla_rope_kv_forward(*convertNVTETensorCheck(kv), *convertNVTETensorCheck(k_pos_emb),
                      *convertNVTETensorCheck(cu_seqlens), *convertNVTETensorCheck(cos),
                      *convertNVTETensorCheck(sin),
                      convertNVTETensorCheck(key_out), convertNVTETensorCheck(val_out),
                      qkv_format, cp_size, cp_rank, s, b, h, k_dim, v_dim, emb_dim,
                      kv_stride_s_or_t, kv_stride_b, kv_stride_h,
                      emb_stride_s_or_t, emb_stride_b, stream);
}

void nvte_mla_rope_kv_backward(const NVTETensor dk, const NVTETensor dv,
                                const NVTETensor cu_seqlens, const NVTETensor cos,
                                const NVTETensor sin, NVTETensor d_kv_out, NVTETensor d_emb_out,
                                const NVTE_QKV_Format qkv_format, const int cp_size,
                                const int cp_rank, const int s, const int b, const int h,
                                const int k_dim, const int v_dim, const int emb_dim,
                                const int dk_stride_s_or_t, const int dk_stride_b,
                                const int dk_stride_h, const int dv_stride_s_or_t,
                                const int dv_stride_b, const int dv_stride_h,
                                const int d_emb_stride_s_or_t, const int d_emb_stride_b,
                                cudaStream_t stream) {
  NVTE_API_CALL(nvte_mla_rope_kv_backward);
  using namespace transformer_engine;
  mla_rope_kv_backward(*convertNVTETensorCheck(dk), *convertNVTETensorCheck(dv),
                       *convertNVTETensorCheck(cu_seqlens), *convertNVTETensorCheck(cos),
                       *convertNVTETensorCheck(sin),
                       convertNVTETensorCheck(d_kv_out), convertNVTETensorCheck(d_emb_out),
                       qkv_format, cp_size, cp_rank, s, b, h, k_dim, v_dim, emb_dim,
                       dk_stride_s_or_t, dk_stride_b, dk_stride_h,
                       dv_stride_s_or_t, dv_stride_b, dv_stride_h,
                       d_emb_stride_s_or_t, d_emb_stride_b, stream);
}
