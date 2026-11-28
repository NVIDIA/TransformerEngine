/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*************************************************************************
 * Vectorized fused QKV RoPE kernels (forward and backward).
 *
 * Same math as transformer_engine/common/fused_rope/fused_rope.cu (fused_qkv_rope_*_kernel, non-interleaved):
 * per element, in FP32, out = x * cos + x_rot * sin with the same operands and the same expression, and
 * cos/sin from sincosf of the same FP32 angles. What changes is the memory access: each thread moves
 * 16 bytes (8 elements) per load/store, and a thread handles an element chunk together with its rotate
 * partner chunk (d2 / 2 away), so both halves of a head are read once. The scalar kernels issue 2-byte
 * accesses and re-read the partner element.
 *
 * Supported: 2-byte types (bf16, fp16), sbhd/bshd, interleaved == false, d2 % 16 == 0, d % 8 == 0,
 * d2 <= d, q/k/v splits that are multiples of d, 16-byte aligned pointers. can_use() checks this; the
 * caller falls back to the scalar kernels otherwise.
 ************************************************************************/
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace fused_qkv_rope_vec {

constexpr int kThreads = 256;
constexpr int kVec = 8;  // 2-byte elements per 16-byte access

inline bool aligned16(const void *p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }

inline bool can_use(int elem_size, bool interleaved, int h, int d, int d2, int q_split, int k_split,
                    int v_split) {
  if (elem_size != 2 || interleaved) return false;
  if (h <= 0 || d <= 0 || d2 <= 0 || d2 > d) return false;
  if (d2 % (2 * kVec) != 0 || d % kVec != 0) return false;
  if (q_split <= 0 || k_split <= 0 || v_split <= 0) return false;
  if (q_split % d != 0 || k_split % d != 0 || v_split % d != 0) return false;
  return true;
}

// Position used to index freqs; identical to the scalar kernels.
__device__ __forceinline__ int freqs_position(int s_id, int s, int cp_size, int cp_rank,
                                              const int *start_positions, int b_id) {
  if (cp_size > 1) {
    if (s_id < s / 2) return s_id + cp_rank * s / 2;
    return s * cp_size - (cp_rank + 1) * s / 2 + s_id - s / 2;
  }
  return s_id + ((start_positions == nullptr) ? 0 : start_positions[b_id]);
}

__device__ __forceinline__ void load_cos_sin(const float *freqs, int pos, int d2, float *cos_s,
                                             float *sin_s) {
  for (int i = threadIdx.x; i < d2; i += blockDim.x) {
    sincosf(freqs[pos * d2 + i], &sin_s[i], &cos_s[i]);
  }
}

template <typename T>
__device__ __forceinline__ void copy16(const T *src, T *dst) {
  *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(src);
}

// One block per (s, b) token. Tasks: q/k head chunks (a pair of 8-element chunks d2/2 apart, or one
// 8-element chunk of the non-rotated tail), then 8-element chunks of v.
template <typename T>
__global__ void __launch_bounds__(kThreads)
    fwd_kernel(const T *__restrict__ qkv, const float *__restrict__ q_freqs,
               const float *__restrict__ k_freqs, const int *__restrict__ start_positions,
               T *__restrict__ q_out, T *__restrict__ k_out, T *__restrict__ v_out, bool sbhd,
               int cp_size, int cp_rank, int s, int b, int h, int d, int d2, int q_split,
               int k_split, int v_split) {
  extern __shared__ float smem[];
  float *cos_q = smem, *sin_q = smem + d2, *cos_k = smem + 2 * d2, *sin_k = smem + 3 * d2;
  const int s_id = blockIdx.x, b_id = blockIdx.y;
  const int total_d = q_split + k_split + v_split;
  const int64_t tok = sbhd ? (int64_t)s_id * b + b_id : (int64_t)b_id * s + s_id;
  const T *src_tok = qkv + tok * h * total_d;
  T *q_tok = q_out + tok * h * q_split;
  T *k_tok = k_out + tok * h * k_split;
  T *v_tok = v_out + tok * h * v_split;

  const int pos = freqs_position(s_id, s, cp_size, cp_rank, start_positions, b_id);
  load_cos_sin(q_freqs, pos, d2, cos_q, sin_q);
  load_cos_sin(k_freqs, pos, d2, cos_k, sin_k);
  __syncthreads();

  const int half = d2 / 2, pairs = half / kVec, per_head = pairs + (d - d2) / kVec;
  const int qh = q_split / d, kh = k_split / d, vchunks = v_split / kVec;
  const int nq = h * qh * per_head, nk = h * kh * per_head, nv = h * vchunks;
  for (int t = threadIdx.x; t < nq + nk + nv; t += blockDim.x) {
    if (t < nq + nk) {
      const bool is_q = t < nq;
      int u = is_q ? t : t - nq;
      const int heads = is_q ? qh : kh;
      const int c = u % per_head;
      u /= per_head;
      const int head = u % heads, g = u / heads;
      const T *src = src_tok + g * total_d + (is_q ? 0 : q_split) + head * d;
      T *dst = is_q ? q_tok + g * q_split + head * d : k_tok + g * k_split + head * d;
      if (c < pairs) {
        const float *cs = is_q ? cos_q : cos_k;
        const float *sn = is_q ? sin_q : sin_k;
        const int lo = c * kVec, hi = lo + half;
        alignas(16) T xl[kVec], xh[kVec], ol[kVec], oh[kVec];
        *reinterpret_cast<uint4 *>(xl) = *reinterpret_cast<const uint4 *>(src + lo);
        *reinterpret_cast<uint4 *>(xh) = *reinterpret_cast<const uint4 *>(src + hi);
#pragma unroll
        for (int j = 0; j < kVec; ++j) {
          {  // first half: rotate partner is -x[d_id + d2/2]
            float v_cos = cs[lo + j], v_sin = sn[lo + j];
            float v_src = xl[j];
            float v_src_rotate = -static_cast<float>(xh[j]);
            ol[j] = v_src * v_cos + v_src_rotate * v_sin;
          }
          {  // second half: rotate partner is x[d_id - d2/2]
            float v_cos = cs[hi + j], v_sin = sn[hi + j];
            float v_src = xh[j];
            float v_src_rotate = static_cast<float>(xl[j]);
            oh[j] = v_src * v_cos + v_src_rotate * v_sin;
          }
        }
        *reinterpret_cast<uint4 *>(dst + lo) = *reinterpret_cast<const uint4 *>(ol);
        *reinterpret_cast<uint4 *>(dst + hi) = *reinterpret_cast<const uint4 *>(oh);
      } else {
        const int off = d2 + (c - pairs) * kVec;
        copy16(src + off, dst + off);
      }
    } else {
      const int u = t - nq - nk;
      const int c = u % vchunks, g = u / vchunks;
      copy16(src_tok + g * total_d + q_split + k_split + c * kVec, v_tok + g * v_split + c * kVec);
    }
  }
}

template <typename T>
__global__ void __launch_bounds__(kThreads)
    bwd_kernel(const T *__restrict__ gq, const T *__restrict__ gk, const T *__restrict__ gv,
               const float *__restrict__ q_freqs, const float *__restrict__ k_freqs,
               T *__restrict__ qkv_grad, bool sbhd, int cp_size, int cp_rank, int s, int b, int h,
               int d, int d2, int q_split, int k_split, int v_split) {
  extern __shared__ float smem[];
  float *cos_q = smem, *sin_q = smem + d2, *cos_k = smem + 2 * d2, *sin_k = smem + 3 * d2;
  const int s_id = blockIdx.x, b_id = blockIdx.y;
  const int total_d = q_split + k_split + v_split;
  const int64_t tok = sbhd ? (int64_t)s_id * b + b_id : (int64_t)b_id * s + s_id;
  T *dst_tok = qkv_grad + tok * h * total_d;
  const T *gq_tok = gq + tok * h * q_split;
  const T *gk_tok = gk + tok * h * k_split;
  const T *gv_tok = gv + tok * h * v_split;

  // The scalar backward kernel ignores start_positions.
  const int pos = freqs_position(s_id, s, cp_size, cp_rank, nullptr, b_id);
  load_cos_sin(q_freqs, pos, d2, cos_q, sin_q);
  load_cos_sin(k_freqs, pos, d2, cos_k, sin_k);
  __syncthreads();

  const int half = d2 / 2, pairs = half / kVec, per_head = pairs + (d - d2) / kVec;
  const int qh = q_split / d, kh = k_split / d, vchunks = v_split / kVec;
  const int nq = h * qh * per_head, nk = h * kh * per_head, nv = h * vchunks;
  for (int t = threadIdx.x; t < nq + nk + nv; t += blockDim.x) {
    if (t < nq + nk) {
      const bool is_q = t < nq;
      int u = is_q ? t : t - nq;
      const int heads = is_q ? qh : kh;
      const int c = u % per_head;
      u /= per_head;
      const int head = u % heads, g = u / heads;
      const T *src = is_q ? gq_tok + g * q_split + head * d : gk_tok + g * k_split + head * d;
      T *dst = dst_tok + g * total_d + (is_q ? 0 : q_split) + head * d;
      if (c < pairs) {
        const float *cs = is_q ? cos_q : cos_k;
        const float *sn = is_q ? sin_q : sin_k;
        const int lo = c * kVec, hi = lo + half;
        alignas(16) T gl[kVec], gh[kVec], ol[kVec], oh[kVec];
        *reinterpret_cast<uint4 *>(gl) = *reinterpret_cast<const uint4 *>(src + lo);
        *reinterpret_cast<uint4 *>(gh) = *reinterpret_cast<const uint4 *>(src + hi);
#pragma unroll
        for (int j = 0; j < kVec; ++j) {
          {  // first half: partner g[d_id + d2/2], sin[d_id + d2/2]
            float v_cos = cs[lo + j];
            float v_src = gl[j];
            float v_src_rotate = static_cast<float>(gh[j]);
            float v_sin = sn[hi + j];
            ol[j] = v_src * v_cos + v_src_rotate * v_sin;
          }
          {  // second half: partner g[d_id - d2/2], -sin[d_id - d2/2]
            float v_cos = cs[hi + j];
            float v_src = gh[j];
            float v_src_rotate = static_cast<float>(gl[j]);
            float v_sin = -sn[lo + j];
            oh[j] = v_src * v_cos + v_src_rotate * v_sin;
          }
        }
        *reinterpret_cast<uint4 *>(dst + lo) = *reinterpret_cast<const uint4 *>(ol);
        *reinterpret_cast<uint4 *>(dst + hi) = *reinterpret_cast<const uint4 *>(oh);
      } else {
        const int off = d2 + (c - pairs) * kVec;
        copy16(src + off, dst + off);
      }
    } else {
      const int u = t - nq - nk;
      const int c = u % vchunks, g = u / vchunks;
      copy16(gv_tok + g * v_split + c * kVec, dst_tok + g * total_d + q_split + k_split + c * kVec);
    }
  }
}

template <typename T>
cudaError_t launch_fwd(const T *qkv, const float *q_freqs, const float *k_freqs,
                       const int *start_positions, T *q_out, T *k_out, T *v_out, bool sbhd,
                       int cp_size, int cp_rank, int s, int b, int h, int d, int d2, int q_split,
                       int k_split, int v_split, cudaStream_t stream) {
  dim3 blocks(s, b);
  fwd_kernel<T><<<blocks, kThreads, 4 * d2 * sizeof(float), stream>>>(
      qkv, q_freqs, k_freqs, start_positions, q_out, k_out, v_out, sbhd, cp_size, cp_rank, s, b, h,
      d, d2, q_split, k_split, v_split);
  return cudaGetLastError();
}

template <typename T>
cudaError_t launch_bwd(const T *gq, const T *gk, const T *gv, const float *q_freqs,
                       const float *k_freqs, T *qkv_grad, bool sbhd, int cp_size, int cp_rank,
                       int s, int b, int h, int d, int d2, int q_split, int k_split, int v_split,
                       cudaStream_t stream) {
  dim3 blocks(s, b);
  bwd_kernel<T><<<blocks, kThreads, 4 * d2 * sizeof(float), stream>>>(
      gq, gk, gv, q_freqs, k_freqs, qkv_grad, sbhd, cp_size, cp_rank, s, b, h, d, d2, q_split,
      k_split, v_split);
  return cudaGetLastError();
}

}  // namespace fused_qkv_rope_vec
