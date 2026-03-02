/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <utility>
#include <assert.h>
#include <cuda_fp8.h>
#include <transformer_engine/multi_tensor.h>
#include <transformer_engine/transformer_engine.h>

#include "../common.h"
#include "../util/math.h"
#include "../utils.cuh"
#include "../util/ptx.cuh"
#include "multi_tensor_apply.cuh"

namespace transformer_engine {
namespace multi_tensor_adam {

#define BLOCK_SIZE 512
#define ILP 4
#define THREADS_PER_WARP 32

typedef enum {
  ADAM_MODE_0 = 0,  // L2 regularization mode
  ADAM_MODE_1 = 1   // Decoupled weight decay mode(AdamW)
} adamMode_t;

using MATH_T = float;
using fp8e4m3 = __nv_fp8_e4m3;
using fp8e5m2 = __nv_fp8_e5m2;
using e8m0_t = transformer_engine::e8m0_t;

template <typename T>
struct is_fp8 : std::false_type {};

template <>
struct is_fp8<fp8e4m3> : std::true_type {};

template <>
struct is_fp8<fp8e5m2> : std::true_type {};

template <bool is_fp8>
struct FP8Data {
  float scale;
  float *amax_ptr;
  float *scale_inv_ptr;
  float max;
  int warp_id;
};

template <>
struct FP8Data<false> {};

template <typename T>
__device__ __forceinline__ void adam_update(T &r_g, T &r_p, T &r_m, T &r_v, const float beta1,
                                            const float beta2, const float beta1_correction,
                                            const float beta2_correction, const float epsilon,
                                            const float lr, adamMode_t mode, const float decay) {
  if (mode == ADAM_MODE_0) {  // L2
    r_g = r_g + (decay * r_p);
    r_m = beta1 * r_m + (1 - beta1) * r_g;
    r_v = beta2 * r_v + (1 - beta2) * r_g * r_g;
    T next_m_unbiased = r_m / beta1_correction;
    T next_v_unbiased = r_v / beta2_correction;
    T denom = sqrtf(next_v_unbiased) + epsilon;
    T update = next_m_unbiased / denom;
    r_p = r_p - (lr * update);
  } else {  // weight decay
    r_m = beta1 * r_m + (1 - beta1) * r_g;
    r_v = beta2 * r_v + (1 - beta2) * r_g * r_g;
    T next_m_unbiased = r_m / beta1_correction;
    T next_v_unbiased = r_v / beta2_correction;
    T denom = sqrtf(next_v_unbiased) + epsilon;
    T update = (next_m_unbiased / denom) + (decay * r_p);
    r_p = r_p - (lr * update);
  }
}

template <typename PARAM_T, typename GRAD_T, typename FULL_T, typename MOMENT_T, typename index_t>
struct AdamFunctorMaster {
  static constexpr bool is_fp8_type = is_fp8<PARAM_T>::value;

  __device__ __forceinline__ void operator()(index_t chunk_size, volatile int *noop_gmem,
                                             TensorListMetadata<5, is_fp8_type> &tl,  // NOLINT(*)
                                             const float beta1, const float beta2,
                                             const float beta1_correction,
                                             const float beta2_correction, const float epsilon,
                                             const float lr, adamMode_t mode, const float decay) {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    FP8Data<is_fp8_type> fp8_data;

    index_t tensor_loc = tl.block_to_tensor[blockIdx.x];

    // potentially use to pass in list of scalar
    // int tensor_num = tl.start_tensor_this_launch + tensor_loc;

    index_t chunk_idx = tl.block_to_chunk[blockIdx.x];
    index_t n = tl.sizes[tensor_loc];

    GRAD_T *g = reinterpret_cast<GRAD_T *>(tl.addresses[0][tensor_loc]);
    g += chunk_idx * chunk_size;

    PARAM_T *p = reinterpret_cast<PARAM_T *>(tl.addresses[1][tensor_loc]);
    p += chunk_idx * chunk_size;

    MOMENT_T *m = reinterpret_cast<MOMENT_T *>(tl.addresses[2][tensor_loc]);
    m += chunk_idx * chunk_size;

    MOMENT_T *v = reinterpret_cast<MOMENT_T *>(tl.addresses[3][tensor_loc]);
    v += chunk_idx * chunk_size;

    FULL_T *p_master = reinterpret_cast<FULL_T *>(tl.addresses[4][tensor_loc]);
    p_master += chunk_idx * chunk_size;

    n -= chunk_idx * chunk_size;

    if constexpr (is_fp8_type) {
      float *scale_ptr = reinterpret_cast<float *>(tl.fp8_meta_addresses[0][tensor_loc]);
      fp8_data.scale = scale_ptr != nullptr ? *scale_ptr : 1;
      fp8_data.amax_ptr = reinterpret_cast<float *>(tl.fp8_meta_addresses[1][tensor_loc]);
      fp8_data.scale_inv_ptr = reinterpret_cast<float *>(tl.fp8_meta_addresses[2][tensor_loc]);
      fp8_data.warp_id = threadIdx.x / THREADS_PER_WARP;
      fp8_data.max = 0;
    }

    // see note in multi_tensor_scale_kernel.cu
    for (index_t i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * ILP) {
      MATH_T r_g[ILP];
      MATH_T r_p[ILP];
      MATH_T r_m[ILP];
      MATH_T r_v[ILP];
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        int i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size) {
          r_g[ii] = static_cast<MATH_T>(g[i]);
          r_p[ii] = static_cast<MATH_T>(p_master[i]);
          r_m[ii] = static_cast<MATH_T>(m[i]);
          r_v[ii] = static_cast<MATH_T>(v[i]);
        } else {
          r_g[ii] = MATH_T(0);
          r_p[ii] = MATH_T(0);
          r_m[ii] = MATH_T(0);
          r_v[ii] = MATH_T(0);
        }
      }
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        adam_update(r_g[ii], r_p[ii], r_m[ii], r_v[ii], beta1, beta2, beta1_correction,
                    beta2_correction, epsilon, lr, mode, decay);
      }

#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        int i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size) {
          p_master[i] = static_cast<FULL_T>(r_p[ii]);
          m[i] = static_cast<MOMENT_T>(r_m[ii]);
          v[i] = static_cast<MOMENT_T>(r_v[ii]);
          if constexpr (is_fp8_type) {
            __builtin_assume(fp8_data.max >= 0);
            fp8_data.max = fmaxf(fabsf(r_p[ii]), fp8_data.max);
            p[i] = static_cast<PARAM_T>(r_p[ii] * fp8_data.scale);
          } else {
            p[i] = static_cast<PARAM_T>(r_p[ii]);
          }
        }
      }
    }

    if constexpr (is_fp8_type) {
      fp8_data.max = transformer_engine::reduce_max<BLOCK_SIZE / THREADS_PER_WARP>(
          fp8_data.max, fp8_data.warp_id);
      if (threadIdx.x == 0) {
        if (fp8_data.amax_ptr != nullptr) {
          transformer_engine::atomicMaxFloat(fp8_data.amax_ptr, fp8_data.max);
        }
        if (fp8_data.scale_inv_ptr != nullptr) {
          *fp8_data.scale_inv_ptr = __frcp_rn(fp8_data.scale);
        }
      }
    }
  }
};

template <typename GRAD_T, typename FULL_T, typename MOMENT_T, typename index_t>
struct AdamFunctorMasterParamRemainder {
  __device__ __forceinline__ void operator()(index_t chunk_size, volatile int *noop_gmem,
                                             TensorListMetadata<5> &tl,  // NOLINT(*)
                                             const float beta1, const float beta2,
                                             const float beta1_correction,
                                             const float beta2_correction, const float epsilon,
                                             const float lr, adamMode_t mode, const float decay) {
    index_t tensor_loc = tl.block_to_tensor[blockIdx.x];

    index_t chunk_idx = tl.block_to_chunk[blockIdx.x];
    index_t n = tl.sizes[tensor_loc];

    GRAD_T *g = reinterpret_cast<GRAD_T *>(tl.addresses[0][tensor_loc]);
    g += chunk_idx * chunk_size;

    int16_t *p = reinterpret_cast<int16_t *>(tl.addresses[1][tensor_loc]);
    p += chunk_idx * chunk_size;

    MOMENT_T *m = reinterpret_cast<MOMENT_T *>(tl.addresses[2][tensor_loc]);
    m += chunk_idx * chunk_size;

    MOMENT_T *v = reinterpret_cast<MOMENT_T *>(tl.addresses[3][tensor_loc]);
    v += chunk_idx * chunk_size;

    int16_t *p_remainder = reinterpret_cast<int16_t *>(tl.addresses[4][tensor_loc]);
    p_remainder += chunk_idx * chunk_size;

    n -= chunk_idx * chunk_size;

    // see note in multi_tensor_scale_kernel.cu
    for (index_t i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * ILP) {
      union fp32_or_int162 {
        float fp32;
        int16_t int16[2];
      };
      fp32_or_int162 local_master_param[ILP];
      int16_t local_p[ILP];
      int16_t local_p_rem[ILP];
      MATH_T r_g[ILP];
      MATH_T r_m[ILP];
      MATH_T r_v[ILP];
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        int i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size) {
          r_g[ii] = static_cast<MATH_T>(g[i]);
          r_m[ii] = static_cast<MATH_T>(m[i]);
          r_v[ii] = static_cast<MATH_T>(v[i]);

          local_p[ii] = p[i];
          local_p_rem[ii] = p_remainder[i];
        } else {
          r_g[ii] = MATH_T(0);
          r_m[ii] = MATH_T(0);
          r_v[ii] = MATH_T(0);

          local_p[ii] = int16_t(0);
          local_p_rem[ii] = int16_t(0);
        }
      }
// Reconstruct FP32 params
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        if (local_p_rem[ii] < 0) local_p[ii]--;  // Undo rounding
        local_master_param[ii].int16[1] = local_p[ii];
        local_master_param[ii].int16[0] = local_p_rem[ii];
      }

      MATH_T *r_p = reinterpret_cast<MATH_T *>(local_master_param);

#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        if (mode == ADAM_MODE_0) {  // L2
          r_g[ii] = r_g[ii] + (decay * r_p[ii]);
          r_m[ii] = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
          r_v[ii] = beta2 * r_v[ii] + (1 - beta2) * r_g[ii] * r_g[ii];
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          MATH_T update = next_m_unbiased / denom;
          r_p[ii] = r_p[ii] - (lr * update);
        } else {  // weight decay
          r_m[ii] = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
          r_v[ii] = beta2 * r_v[ii] + (1 - beta2) * r_g[ii] * r_g[ii];
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          MATH_T update = (next_m_unbiased / denom) + (decay * r_p[ii]);
          r_p[ii] = r_p[ii] - (lr * update);
        }
      }

// Split into BF16 params (rounded-to-nearest) and remainders
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        local_p[ii] = local_master_param[ii].int16[1];
        local_p_rem[ii] = local_master_param[ii].int16[0];
        if (local_p_rem[ii] < 0) local_p[ii]++;  // Round up
      }

#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        int i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size) {
          p_remainder[i] = local_p_rem[ii];
          p[i] = local_p[ii];

          m[i] = static_cast<MOMENT_T>(r_m[ii]);
          v[i] = static_cast<MOMENT_T>(r_v[ii]);
        }
      }
    }
  }
};

template <typename PARAM_T, typename GRAD_T, typename FULL_T, typename MOMENT_T, typename index_t>
struct AdamFunctor {
  __device__ __forceinline__ void operator()(index_t chunk_size, volatile int *noop_gmem,
                                             TensorListMetadata<4> &tl,  // NOLINT(*)
                                             const float beta1, const float beta2,
                                             const float beta1_correction,
                                             const float beta2_correction, const float epsilon,
                                             const float lr, adamMode_t mode, const float decay) {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    index_t tensor_loc = tl.block_to_tensor[blockIdx.x];

    // potentially use to pass in list of scalar
    // int tensor_num = tl.start_tensor_this_launch + tensor_loc;

    index_t chunk_idx = tl.block_to_chunk[blockIdx.x];
    index_t n = tl.sizes[tensor_loc];

    GRAD_T *g = reinterpret_cast<GRAD_T *>(tl.addresses[0][tensor_loc]);
    g += chunk_idx * chunk_size;

    PARAM_T *p = reinterpret_cast<PARAM_T *>(tl.addresses[1][tensor_loc]);
    p += chunk_idx * chunk_size;

    MOMENT_T *m = reinterpret_cast<MOMENT_T *>(tl.addresses[2][tensor_loc]);
    m += chunk_idx * chunk_size;

    MOMENT_T *v = reinterpret_cast<MOMENT_T *>(tl.addresses[3][tensor_loc]);
    v += chunk_idx * chunk_size;

    n -= chunk_idx * chunk_size;

    // see note in multi_tensor_scale_kernel.cu
    for (index_t i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * ILP) {
      MATH_T r_g[ILP];
      MATH_T r_p[ILP];
      MATH_T r_m[ILP];
      MATH_T r_v[ILP];
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        int i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size) {
          r_g[ii] = static_cast<MATH_T>(g[i]);
          r_p[ii] = static_cast<MATH_T>(p[i]);
          r_m[ii] = static_cast<MATH_T>(m[i]);
          r_v[ii] = static_cast<MATH_T>(v[i]);
        } else {
          r_g[ii] = MATH_T(0);
          r_p[ii] = MATH_T(0);
          r_m[ii] = MATH_T(0);
          r_v[ii] = MATH_T(0);
        }
      }
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        if (mode == ADAM_MODE_0) {  // L2
          r_g[ii] = r_g[ii] + (decay * r_p[ii]);
          r_m[ii] = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
          r_v[ii] = beta2 * r_v[ii] + (1 - beta2) * r_g[ii] * r_g[ii];
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          MATH_T update = next_m_unbiased / denom;
          r_p[ii] = r_p[ii] - (lr * update);
        } else {  // weight decay
          r_m[ii] = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
          r_v[ii] = beta2 * r_v[ii] + (1 - beta2) * r_g[ii] * r_g[ii];
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          MATH_T update = (next_m_unbiased / denom) + (decay * r_p[ii]);
          r_p[ii] = r_p[ii] - (lr * update);
        }
      }
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        int i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size) {
          p[i] = static_cast<PARAM_T>(r_p[ii]);
          m[i] = static_cast<MOMENT_T>(r_m[ii]);
          v[i] = static_cast<MOMENT_T>(r_v[ii]);
        }
      }
    }
  }
};

template <typename T, typename FULL_T, typename MOMENT_T>
struct AdamCapturableFunctor {
  __device__ __forceinline__ void operator()(int chunk_size, volatile int *noop_gmem,
                                             TensorListMetadata<4> &tl,  // NOLINT(*)
                                             const float beta1, const float beta2, const int *step,
                                             const int bias_correction, const float epsilon,
                                             const float *lr, adamMode_t mode, const float decay,
                                             const float *inv_scale) {
    if (*noop_gmem == 1) return;

    float beta1_correction = 1.0f, beta2_correction = 1.0f;
    if (bias_correction == 1) {
      beta1_correction = 1 - pow(beta1, *step);
      beta2_correction = 1 - pow(beta2, *step);
    }

    int tensor_loc = tl.block_to_tensor[blockIdx.x];

    // potentially use to pass in list of scalar
    // int tensor_num = tl.start_tensor_this_launch + tensor_loc;

    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    T *g = reinterpret_cast<T *>(tl.addresses[0][tensor_loc]);
    g += chunk_idx * chunk_size;

    T *p = reinterpret_cast<T *>(tl.addresses[1][tensor_loc]);
    p += chunk_idx * chunk_size;

    MOMENT_T *m = reinterpret_cast<MOMENT_T *>(tl.addresses[2][tensor_loc]);
    m += chunk_idx * chunk_size;

    MOMENT_T *v = reinterpret_cast<MOMENT_T *>(tl.addresses[3][tensor_loc]);
    v += chunk_idx * chunk_size;

    n -= chunk_idx * chunk_size;

    // see note in multi_tensor_scale_kernel.cu
    for (int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * ILP) {
      MATH_T r_g[ILP];
      MATH_T r_p[ILP];
      MATH_T r_m[ILP];
      MATH_T r_v[ILP];
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        int i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size) {
          r_g[ii] = static_cast<MATH_T>(g[i]) * (*inv_scale);
          g[i] = static_cast<T>(r_g[ii]);
          r_p[ii] = static_cast<MATH_T>(p[i]);
          r_m[ii] = static_cast<MATH_T>(m[i]);
          r_v[ii] = static_cast<MATH_T>(v[i]);
        } else {
          r_g[ii] = MATH_T(0);
          r_p[ii] = MATH_T(0);
          r_m[ii] = MATH_T(0);
          r_v[ii] = MATH_T(0);
        }
      }
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        if (mode == ADAM_MODE_0) {  // L2
          r_g[ii] = r_g[ii] + (decay * r_p[ii]);
          r_m[ii] = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
          r_v[ii] = beta2 * r_v[ii] + (1 - beta2) * r_g[ii] * r_g[ii];
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          MATH_T update = next_m_unbiased / denom;
          r_p[ii] = r_p[ii] - (*lr * update);
        } else {  // weight decay
          r_m[ii] = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
          r_v[ii] = beta2 * r_v[ii] + (1 - beta2) * r_g[ii] * r_g[ii];
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          MATH_T update = (next_m_unbiased / denom) + (decay * r_p[ii]);
          r_p[ii] = r_p[ii] - (*lr * update);
        }
      }
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        int i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size) {
          p[i] = static_cast<T>(r_p[ii]);
          m[i] = static_cast<MOMENT_T>(r_m[ii]);
          v[i] = static_cast<MOMENT_T>(r_v[ii]);
        }
      }
    }
  }
};

template <typename T, typename FULL_T, typename MOMENT_T>
struct AdamCapturableMasterFunctor {
  __device__ __forceinline__ void operator()(int chunk_size, volatile int *noop_gmem,
                                             TensorListMetadata<5> &tl,  // NOLINT(*)
                                             const float beta1, const float beta2, const int *step,
                                             const int bias_correction, const float epsilon,
                                             const float *lr, adamMode_t mode, const float decay,
                                             const float *inv_scale) {
    if (*noop_gmem == 1) return;

    float beta1_correction = 1.0f, beta2_correction = 1.0f;
    if (bias_correction == 1) {
      beta1_correction = 1 - pow(beta1, *step);
      beta2_correction = 1 - pow(beta2, *step);
    }

    int tensor_loc = tl.block_to_tensor[blockIdx.x];

    // potentially use to pass in list of scalar
    // int tensor_num = tl.start_tensor_this_launch + tensor_loc;

    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    T *g = reinterpret_cast<T *>(tl.addresses[0][tensor_loc]);
    g += chunk_idx * chunk_size;

    T *p = reinterpret_cast<T *>(tl.addresses[1][tensor_loc]);
    p += chunk_idx * chunk_size;

    MOMENT_T *m = reinterpret_cast<MOMENT_T *>(tl.addresses[2][tensor_loc]);
    m += chunk_idx * chunk_size;

    MOMENT_T *v = reinterpret_cast<MOMENT_T *>(tl.addresses[3][tensor_loc]);
    v += chunk_idx * chunk_size;

    FULL_T *p_master = reinterpret_cast<FULL_T *>(tl.addresses[4][tensor_loc]);
    p_master += chunk_idx * chunk_size;

    n -= chunk_idx * chunk_size;

    // see note in multi_tensor_scale_kernel.cu
    for (int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * ILP) {
      MATH_T r_g[ILP];
      MATH_T r_p[ILP];
      MATH_T r_m[ILP];
      MATH_T r_v[ILP];
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        int i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size) {
          r_g[ii] = static_cast<MATH_T>(g[i]) * (*inv_scale);
          g[i] = static_cast<T>(r_g[ii]);
          r_p[ii] = static_cast<MATH_T>(p_master[i]);
          r_m[ii] = static_cast<MATH_T>(m[i]);
          r_v[ii] = static_cast<MATH_T>(v[i]);
        } else {
          r_g[ii] = MATH_T(0);
          r_p[ii] = MATH_T(0);
          r_m[ii] = MATH_T(0);
          r_v[ii] = MATH_T(0);
        }
      }
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        if (mode == ADAM_MODE_0) {  // L2
          r_g[ii] = r_g[ii] + (decay * r_p[ii]);
          r_m[ii] = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
          r_v[ii] = beta2 * r_v[ii] + (1 - beta2) * r_g[ii] * r_g[ii];
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          MATH_T update = next_m_unbiased / denom;
          r_p[ii] = r_p[ii] - (*lr * update);
        } else {  // weight decay
          r_m[ii] = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
          r_v[ii] = beta2 * r_v[ii] + (1 - beta2) * r_g[ii] * r_g[ii];
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          MATH_T update = (next_m_unbiased / denom) + (decay * r_p[ii]);
          r_p[ii] = r_p[ii] - (*lr * update);
        }
      }
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        int i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size) {
          p[i] = static_cast<T>(r_p[ii]);
          p_master[i] = static_cast<FULL_T>(r_p[ii]);
          m[i] = static_cast<MOMENT_T>(r_m[ii]);
          v[i] = static_cast<MOMENT_T>(r_v[ii]);
        }
      }
    }
  }
};

template <typename FP8_T>
__device__ __forceinline__ FP8_T cast_to_fp8(float x) {
  return static_cast<FP8_T>(x);
}

__device__ __forceinline__ float fp8_max_norm_rcp(uint8_t fp8_dtype) {
  if (fp8_dtype == static_cast<uint8_t>(transformer_engine::DType::kFloat8E4M3)) {
    return transformer_engine::Quantized_Limits<fp8e4m3>::max_norm_rcp;
  }
  return transformer_engine::Quantized_Limits<fp8e5m2>::max_norm_rcp;
}

template <typename PARAM_T, typename GRAD_T, typename MOMENT_T, typename index_t>
__global__ void adam_mxfp8_fused_kernel(
    int64_t chunk_size, volatile int *noop_gmem, MXFP8TensorListMetadata tl, float beta1,
    float beta2, float beta1_correction, float beta2_correction, float epsilon, float lr, int mode,
    float weight_decay) {
  // Stage 0: optional early-exit if a noop flag is set.
  if (noop_gmem != nullptr && *noop_gmem == 1) {
    return;
  }
  (void)chunk_size;

  // Stage 1: map this block to a specific tensor tile.
  const int block_idx = blockIdx.x;
  const int tensor_idx = tl.block_to_tensor[block_idx];
  const int tile_idx = tl.block_to_tile[block_idx];
  const int64_t rows_val = tl.rows[tensor_idx];
  const int64_t cols_val = tl.cols[tensor_idx];
  if (rows_val == 0 || cols_val == 0) {
    return;
  }

  const int64_t tiles_per_row = (cols_val + MXFP8_TILE - 1) / MXFP8_TILE;
  const int64_t tile_row = tile_idx / tiles_per_row;
  const int64_t tile_col = tile_idx % tiles_per_row;
  const int64_t row_base = tile_row * MXFP8_TILE;
  const int64_t col_base = tile_col * MXFP8_TILE;

  // Stage 2: load pointers for grads/params/moments and MXFP8 outputs/scales.
  GRAD_T *g = reinterpret_cast<GRAD_T *>(tl.addresses[0][tensor_idx]);
  PARAM_T *p = reinterpret_cast<PARAM_T *>(tl.addresses[1][tensor_idx]);
  MOMENT_T *m = reinterpret_cast<MOMENT_T *>(tl.addresses[2][tensor_idx]);
  MOMENT_T *v = reinterpret_cast<MOMENT_T *>(tl.addresses[3][tensor_idx]);

  auto *rowwise_data = reinterpret_cast<uint8_t *>(tl.addresses[4][tensor_idx]);
  auto *colwise_data = reinterpret_cast<uint8_t *>(tl.addresses[5][tensor_idx]);
  auto *rowwise_scale_inv = reinterpret_cast<uint8_t *>(tl.addresses[6][tensor_idx]);
  auto *colwise_scale_inv = reinterpret_cast<uint8_t *>(tl.addresses[7][tensor_idx]);

  const int64_t unpadded_scales_X_rowwise = (cols_val + MXFP8_TILE - 1) / MXFP8_TILE;
  constexpr int64_t kRowwiseScaleAlign = 4;
  const int64_t row_stride =
      DIVUP_TO_MULTIPLE(unpadded_scales_X_rowwise, kRowwiseScaleAlign);
  constexpr int64_t kColwiseScaleAlign = 128;
  const int64_t col_stride = DIVUP_TO_MULTIPLE(cols_val, kColwiseScaleAlign);
  const uint8_t dtype = tl.fp8_dtype[tensor_idx];
  const auto adam_mode = static_cast<transformer_engine::multi_tensor_adam::adamMode_t>(mode);

  // Stage 3: initialize shared amax accumulators per row/col within the tile.
  __shared__ float row_max_vals[MXFP8_TILE];
  __shared__ float col_max_vals[MXFP8_TILE];
  if (threadIdx.x < MXFP8_TILE) {
    row_max_vals[threadIdx.x] = 0.0f;
    col_max_vals[threadIdx.x] = 0.0f;
  }
  __syncthreads();

  for (int t = threadIdx.x; t < MXFP8_TILE_ELEMS; t += blockDim.x) {
    const int local_r = t / MXFP8_TILE;
    const int local_c = t % MXFP8_TILE;
    const int64_t r = row_base + local_r;
    const int64_t c = col_base + local_c;
    if (r >= rows_val || c >= cols_val) {
      continue;
    }
    const index_t idx = static_cast<index_t>(r * cols_val + c);

    float r_g = static_cast<float>(g[idx]);
    float r_p = static_cast<float>(p[idx]);
    float r_m = static_cast<float>(m[idx]);
    float r_v = static_cast<float>(v[idx]);

    // Stage 4: apply Adam update in FP32 and write back updated p/m/v.
    transformer_engine::multi_tensor_adam::adam_update(
        r_g, r_p, r_m, r_v, beta1, beta2, beta1_correction, beta2_correction, epsilon, lr,
        adam_mode, weight_decay);

    p[idx] = static_cast<PARAM_T>(r_p);
    m[idx] = static_cast<MOMENT_T>(r_m);
    v[idx] = static_cast<MOMENT_T>(r_v);

    // Stage 5: accumulate per-row/col absmax for MXFP8 scaling.
    const float abs_p = fabsf(r_p);
    transformer_engine::atomicMaxFloat(&row_max_vals[local_r], abs_p);
    transformer_engine::atomicMaxFloat(&col_max_vals[local_c], abs_p);
  }

  __syncthreads();

  // Stage 6: write rowwise/colwise scale-inverse exponents for the tile.
  const float max_norm_rcp = fp8_max_norm_rcp(dtype);

  for (int r = threadIdx.x; r < MXFP8_TILE; r += blockDim.x) {
    const int64_t row = row_base + r;
    if (row >= rows_val) {
      continue;
    }
    const float amax = row_max_vals[r];
    const ::transformer_engine::e8m0_t biased_exponent =
        transformer_engine::ptx::float_to_e8m0(amax * max_norm_rcp);
    const size_t scale_idx = static_cast<size_t>(row * row_stride + tile_col);
    rowwise_scale_inv[scale_idx] = reinterpret_cast<const uint8_t &>(biased_exponent);
  }

  for (int c = threadIdx.x; c < MXFP8_TILE; c += blockDim.x) {
    const int64_t col = col_base + c;
    if (col >= cols_val) {
      continue;
    }
    const float amax = col_max_vals[c];
    const ::transformer_engine::e8m0_t biased_exponent =
        transformer_engine::ptx::float_to_e8m0(amax * max_norm_rcp);
    const size_t scale_idx = static_cast<size_t>(tile_row * col_stride + col);
    colwise_scale_inv[scale_idx] = reinterpret_cast<const uint8_t &>(biased_exponent);
  }

  __syncthreads();

  // Stage 7: quantize updated params to MXFP8 using rowwise and colwise scales.
  for (int t = threadIdx.x; t < MXFP8_TILE_ELEMS; t += blockDim.x) {
    const int local_r = t / MXFP8_TILE;
    const int local_c = t % MXFP8_TILE;
    const int64_t r = row_base + local_r;
    const int64_t c = col_base + local_c;
    if (r >= rows_val || c >= cols_val) {
      continue;
    }
    const index_t idx = static_cast<index_t>(r * cols_val + c);
    const float r_p = static_cast<float>(p[idx]);

    const size_t row_scale_idx = static_cast<size_t>(r * row_stride + tile_col);
    const uint8_t row_raw = rowwise_scale_inv[row_scale_idx];
    const ::transformer_engine::e8m0_t row_biased =
        reinterpret_cast<const ::transformer_engine::e8m0_t &>(row_raw);
    const float row_scale_inv = transformer_engine::ptx::exp2f_rcp(row_biased);
    if (dtype == static_cast<uint8_t>(transformer_engine::DType::kFloat8E4M3)) {
      auto *out = reinterpret_cast<fp8e4m3 *>(rowwise_data);
      out[idx] = cast_to_fp8<fp8e4m3>(r_p * row_scale_inv);
    } else {
      auto *out = reinterpret_cast<fp8e5m2 *>(rowwise_data);
      out[idx] = cast_to_fp8<fp8e5m2>(r_p * row_scale_inv);
    }

    const size_t col_scale_idx = static_cast<size_t>(tile_row * col_stride + c);
    const uint8_t col_raw = colwise_scale_inv[col_scale_idx];
    const ::transformer_engine::e8m0_t col_biased =
        reinterpret_cast<const ::transformer_engine::e8m0_t &>(col_raw);
    const float col_scale_inv = transformer_engine::ptx::exp2f_rcp(col_biased);
    if (dtype == static_cast<uint8_t>(transformer_engine::DType::kFloat8E4M3)) {
      auto *out = reinterpret_cast<fp8e4m3 *>(colwise_data);
      out[idx] = cast_to_fp8<fp8e4m3>(r_p * col_scale_inv);
    } else {
      auto *out = reinterpret_cast<fp8e5m2 *>(colwise_data);
      out[idx] = cast_to_fp8<fp8e5m2>(r_p * col_scale_inv);
    }
  }
}

inline bool requires_64bit_indexing(const std::vector<std::vector<Tensor *>> &tensor_lists) {
  const size_t num_tensor_lists = tensor_lists.size();
  const size_t num_tensors_per_list = tensor_lists[0].size();
  for (size_t i = 0; i < num_tensor_lists; ++i) {
    for (size_t j = 0; j < num_tensors_per_list; ++j) {
      if (tensor_lists[i][j]->numel() >= INT_MAX) {
        return true;
      }
    }
  }
  return false;
}

void multi_tensor_adam_cuda(int chunk_size, Tensor noop_flag,
                            std::vector<std::vector<Tensor *>> tensor_lists, const float lr,
                            const float beta1, const float beta2, const float epsilon,
                            const int step, const int mode, const int bias_correction,
                            const float weight_decay, cudaStream_t stream) {
  // Handle bias correction mode
  float bias_correction1 = 1.0f, bias_correction2 = 1.0f;
  if (bias_correction == 1) {
    bias_correction1 = 1 - std::pow(beta1, step);
    bias_correction2 = 1 - std::pow(beta2, step);
  }

  // Check tensor list sizes
  // 4 tensor lists: g, p, m, v
  // 5 tensor lists: g, p, m, v, p_master
  const size_t num_tensor_lists = tensor_lists.size();
  NVTE_CHECK(num_tensor_lists == 4 || num_tensor_lists == 5,
             "Expected 4 or 5 tensor lists, but found ", num_tensor_lists);
  const size_t num_tensors_per_list = tensor_lists[0].size();
  for (size_t i = 1; i < num_tensor_lists; i++) {
    NVTE_CHECK(tensor_lists[i].size() == num_tensors_per_list, "Tensor list ", i,
               " has size=", tensor_lists[i].size(), ", but expected size=", num_tensors_per_list);
  }

  // Check tensor dtypes
  const auto g_in_type_te = tensor_lists[0][0]->dtype();
  const auto p_in_type_te = tensor_lists[1][0]->dtype();
  for (size_t j = 0; j < num_tensors_per_list; j++) {
    NVTE_CHECK(tensor_lists[0][j]->dtype() == g_in_type_te, "Grad tensor ", j,
               " has dtype=", to_string(tensor_lists[0][j]->dtype()),
               ", but expected dtype=", to_string(g_in_type_te));
    NVTE_CHECK(tensor_lists[1][j]->dtype() == p_in_type_te, "Param tensor ", j,
               " has dtype=", to_string(tensor_lists[1][j]->dtype()),
               ", but expected dtype=", to_string(p_in_type_te));
    {
      const bool m_is_fp32 = tensor_lists[2][j]->dtype() == DType::kFloat32;
      const bool m_is_bf16 = tensor_lists[2][j]->dtype() == DType::kBFloat16;
      const bool v_is_fp32 = tensor_lists[3][j]->dtype() == DType::kFloat32;
      const bool v_is_bf16 = tensor_lists[3][j]->dtype() == DType::kBFloat16;
      NVTE_CHECK((m_is_fp32 && v_is_fp32) || (m_is_bf16 && v_is_bf16),
                 "First and second moment tensors must both be Float32 or both be BFloat16, but "
                 "tensor ",
                 j, " has first moment dtype=", to_string(tensor_lists[2][j]->dtype()),
                 " and second moment dtype=", to_string(tensor_lists[3][j]->dtype()));
    }
    if (num_tensor_lists == 5) {
      NVTE_CHECK(tensor_lists[4][j]->dtype() == DType::kFloat32, "Master param tensor ", j,
                 " has dtype=", to_string(tensor_lists[4][j]->dtype()),
                 ", but expected dtype=", to_string(DType::kFloat32));
    }
  }

  const bool use_64bit_indexing = requires_64bit_indexing(tensor_lists);

  // Get moment dtype (m and v have the same dtype, already validated above)
  const auto moment_type_te = tensor_lists[2][0]->dtype();

  // Launch kernel
  if (use_64bit_indexing) {
    if (num_tensor_lists == 4) {
      // g, p, m, v
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
          p_in_type_te, p_in_type,
          TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
              g_in_type_te, g_in_type,
              TRANSFORMER_ENGINE_TYPE_SWITCH_FP32_BF16(
                  moment_type_te, moment_type,
                  multi_tensor_apply<4>(
                      (int64_t)BLOCK_SIZE, (int64_t)chunk_size, noop_flag, tensor_lists,
                      AdamFunctor<p_in_type, g_in_type, float, moment_type, int64_t>(), stream,
                      beta1, beta2, bias_correction1, bias_correction2, epsilon, lr,
                      (adamMode_t)mode, weight_decay);)));
    } else {
      // g, p, m, v, p_master
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
          p_in_type_te, p_in_type,
          TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
              g_in_type_te, g_in_type,
              TRANSFORMER_ENGINE_TYPE_SWITCH_FP32_BF16(
                  moment_type_te, moment_type,
                  multi_tensor_apply<5>(
                      (int64_t)BLOCK_SIZE, (int64_t)chunk_size, noop_flag, tensor_lists,
                      AdamFunctorMaster<p_in_type, g_in_type, float, moment_type, int64_t>(),
                      stream, beta1, beta2, bias_correction1, bias_correction2, epsilon, lr,
                      (adamMode_t)mode, weight_decay);)));
    }
  } else {
    if (num_tensor_lists == 4) {
      // g, p, m, v
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
          p_in_type_te, p_in_type,
          TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
              g_in_type_te, g_in_type,
              TRANSFORMER_ENGINE_TYPE_SWITCH_FP32_BF16(
                  moment_type_te, moment_type,
                  multi_tensor_apply<4>(
                      BLOCK_SIZE, chunk_size, noop_flag, tensor_lists,
                      AdamFunctor<p_in_type, g_in_type, float, moment_type, int32_t>(), stream,
                      beta1, beta2, bias_correction1, bias_correction2, epsilon, lr,
                      (adamMode_t)mode, weight_decay);)));
    } else {
      // g, p, m, v, p_master
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
          p_in_type_te, p_in_type,
          TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
              g_in_type_te, g_in_type,
              TRANSFORMER_ENGINE_TYPE_SWITCH_FP32_BF16(
                  moment_type_te, moment_type,
                  multi_tensor_apply<5>(
                      BLOCK_SIZE, chunk_size, noop_flag, tensor_lists,
                      AdamFunctorMaster<p_in_type, g_in_type, float, moment_type, int32_t>(),
                      stream, beta1, beta2, bias_correction1, bias_correction2, epsilon, lr,
                      (adamMode_t)mode, weight_decay);)));
    }
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void multi_tensor_adam_param_remainder_cuda(int chunk_size, Tensor noop_flag,
                                            std::vector<std::vector<Tensor *>> tensor_lists,
                                            const float lr, const float beta1, const float beta2,
                                            const float epsilon, const int step, const int mode,
                                            const int bias_correction, const float weight_decay,
                                            cudaStream_t stream) {
  // Handle bias correction mode
  float bias_correction1 = 1.0f, bias_correction2 = 1.0f;
  if (bias_correction == 1) {
    bias_correction1 = 1 - std::pow(beta1, step);
    bias_correction2 = 1 - std::pow(beta2, step);
  }

  // Check tensor list sizes
  // 5 tensor lists: g, p, m, v, p_remainder
  const size_t num_tensor_lists = tensor_lists.size();
  NVTE_CHECK(num_tensor_lists == 5, "Expected 5 tensor lists, but found ", num_tensor_lists);
  const size_t num_tensors_per_list = tensor_lists[0].size();
  for (size_t i = 1; i < num_tensor_lists; i++) {
    NVTE_CHECK(tensor_lists[i].size() == num_tensors_per_list, "Tensor list ", i,
               " has size=", tensor_lists[i].size(), ", but expected size=", num_tensors_per_list);
  }

  // Check tensor dtypes
  const auto g_in_type_te = tensor_lists[0][0]->dtype();
  for (size_t j = 0; j < num_tensors_per_list; j++) {
    NVTE_CHECK(tensor_lists[0][j]->dtype() == g_in_type_te, "Grad tensor ", j,
               " has dtype=", to_string(tensor_lists[0][j]->dtype()),
               ", but expected dtype=", to_string(g_in_type_te));
    NVTE_CHECK(tensor_lists[1][j]->dtype() == DType::kBFloat16, "Param tensor ", j,
               " has dtype=", to_string(tensor_lists[1][j]->dtype()),
               ", but expected dtype=", to_string(DType::kBFloat16));
    {
      const bool m_is_fp32 = tensor_lists[2][j]->dtype() == DType::kFloat32;
      const bool m_is_bf16 = tensor_lists[2][j]->dtype() == DType::kBFloat16;
      const bool v_is_fp32 = tensor_lists[3][j]->dtype() == DType::kFloat32;
      const bool v_is_bf16 = tensor_lists[3][j]->dtype() == DType::kBFloat16;
      NVTE_CHECK((m_is_fp32 && v_is_fp32) || (m_is_bf16 && v_is_bf16),
                 "First and second moment tensors must both be Float32 or both be BFloat16, but "
                 "tensor ",
                 j, " has first moment dtype=", to_string(tensor_lists[2][j]->dtype()),
                 " and second moment dtype=", to_string(tensor_lists[3][j]->dtype()));
    }
    NVTE_CHECK(tensor_lists[4][j]->dtype() == DType::kInt16, "Param remainder tensor ", j,
               " has dtype=", to_string(tensor_lists[4][j]->dtype()),
               ", but expected dtype=", to_string(DType::kInt16));
  }

  // Get moment dtype (m and v have the same dtype, already validated above)
  const auto moment_type_te = tensor_lists[2][0]->dtype();

  // Launch kernel
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      g_in_type_te, g_in_type,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP32_BF16(
          moment_type_te, moment_type,
          multi_tensor_apply<5>(
              (int64_t)BLOCK_SIZE, (int64_t)chunk_size, noop_flag, tensor_lists,
              AdamFunctorMasterParamRemainder<g_in_type, float, moment_type, int64_t>(), stream,
              beta1, beta2, bias_correction1, bias_correction2, epsilon, lr, (adamMode_t)mode,
              weight_decay);));
  NVTE_CHECK_CUDA(cudaGetLastError());
}

inline std::pair<float, float> compute_bias_correction(int bias_correction, float beta1,
                                                       float beta2, int step) {
  float bias_correction1 = 1.0f;
  float bias_correction2 = 1.0f;
  if (bias_correction == 1) {
    bias_correction1 = 1 - std::pow(beta1, step);
    bias_correction2 = 1 - std::pow(beta2, step);
  }
  return {bias_correction1, bias_correction2};
}

inline void check_tensor_list_sizes(const std::vector<std::vector<Tensor *>> &tensor_lists,
                                    size_t expected_lists) {
  const size_t num_tensor_lists = tensor_lists.size();
  NVTE_CHECK(num_tensor_lists == expected_lists, "Expected ", expected_lists,
             " tensor lists, but found ", num_tensor_lists);
  const size_t num_tensors_per_list = tensor_lists[0].size();
  for (size_t i = 1; i < num_tensor_lists; ++i) {
    NVTE_CHECK(tensor_lists[i].size() == num_tensors_per_list, "Tensor list ", i,
               " has size=", tensor_lists[i].size(), ", but expected size=", num_tensors_per_list);
  }
}


void multi_tensor_adam_fp8_cuda(int chunk_size, Tensor noop_flag,
                                std::vector<std::vector<Tensor *>> tensor_lists, const float lr,
                                const float beta1, const float beta2, const float epsilon,
                                const int step, const int mode, const int bias_correction,
                                const float weight_decay, const DType fp8_dtype,
                                cudaStream_t stream) {
  auto [bias_correction1, bias_correction2] =
      compute_bias_correction(bias_correction, beta1, beta2, step);
  check_tensor_list_sizes(tensor_lists, 8);
  const size_t num_tensor_lists = tensor_lists.size();
  const size_t num_tensors_per_list = tensor_lists[0].size();

  // Check tensor dtypes
  const auto g_in_type_te = tensor_lists[0][0]->dtype();
  for (size_t j = 0; j < num_tensors_per_list; j++) {
    NVTE_CHECK(tensor_lists[0][j]->dtype() == g_in_type_te, "Grad tensor ", j,
               " has dtype=", to_string(tensor_lists[0][j]->dtype()),
               ", but expected dtype=", to_string(g_in_type_te));
    NVTE_CHECK(
        tensor_lists[1][j]->dtype() == fp8_dtype || tensor_lists[1][j]->dtype() == DType::kByte,
        "Param tensor ", j, " has dtype=", to_string(tensor_lists[1][j]->dtype()),
        ", but expected dtype=", to_string(fp8_dtype));
    NVTE_CHECK(tensor_lists[2][j]->dtype() == DType::kFloat32, "First moment tensor ", j,
               " has dtype=", to_string(tensor_lists[2][j]->dtype()),
               ", but expected dtype=", to_string(DType::kFloat32));
    NVTE_CHECK(tensor_lists[3][j]->dtype() == DType::kFloat32, "Second moment tensor ", j,
               " has dtype=", to_string(tensor_lists[3][j]->dtype()),
               ", but expected dtype=", to_string(DType::kFloat32));
    NVTE_CHECK(tensor_lists[4][j]->dtype() == DType::kFloat32, "Master param tensor ", j,
               " has dtype=", to_string(tensor_lists[4][j]->dtype()),
               ", but expected dtype=", to_string(DType::kFloat32));
    NVTE_CHECK(tensor_lists[5][j]->dtype() == DType::kFloat32, "Scale tensor ", j,
               " has dtype=", to_string(tensor_lists[5][j]->dtype()),
               ", but expected dtype=", to_string(DType::kFloat32));
    NVTE_CHECK(tensor_lists[6][j]->dtype() == DType::kFloat32, "Absmax tensor ", j,
               " has dtype=", to_string(tensor_lists[6][j]->dtype()),
               ", but expected dtype=", to_string(DType::kFloat32));
    NVTE_CHECK(tensor_lists[7][j]->dtype() == DType::kFloat32, "Scale-inverse tensor ", j,
               " has dtype=", to_string(tensor_lists[7][j]->dtype()),
               ", but expected dtype=", to_string(DType::kFloat32));
  }

  const bool use_64bit_indexing = requires_64bit_indexing(tensor_lists);

  // Launch kernel
  if (use_64bit_indexing) {
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
        fp8_dtype, FP8_T,
        TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
            g_in_type_te, g_in_type,
            multi_tensor_apply<5, true>(
                (int64_t)BLOCK_SIZE, (int64_t)chunk_size, noop_flag, tensor_lists,
                AdamFunctorMaster<FP8_T, g_in_type, float, float, int64_t>(), stream, beta1, beta2,
                bias_correction1, bias_correction2, epsilon, lr, (adamMode_t)mode, weight_decay);));
  } else {
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
        fp8_dtype, FP8_T,
        TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
            g_in_type_te, g_in_type,
            multi_tensor_apply<5, true>(
                BLOCK_SIZE, chunk_size, noop_flag, tensor_lists,
                AdamFunctorMaster<FP8_T, g_in_type, float, float, int32_t>(), stream, beta1, beta2,
                bias_correction1, bias_correction2, epsilon, lr, (adamMode_t)mode, weight_decay);));
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void multi_tensor_adam_mxfp8_cuda(int chunk_size, Tensor noop_flag,
                                  std::vector<std::vector<Tensor *>> tensor_lists, const float lr,
                                  const float beta1, const float beta2, const float epsilon,
                                  const int step, const int mode, const int bias_correction,
                                  const float weight_decay, const DType fp8_dtype,
                                  cudaStream_t stream) {
  auto [bias_correction1, bias_correction2] =
      compute_bias_correction(bias_correction, beta1, beta2, step);
  check_tensor_list_sizes(tensor_lists, 8);
  const size_t num_tensor_lists = tensor_lists.size();
  const size_t num_tensors_per_list = tensor_lists[0].size();

  NVTE_CHECK(fp8_dtype == DType::kFloat8E4M3 || fp8_dtype == DType::kFloat8E5M2,
             "fp8_dtype must be E4M3 or E5M2 for MXFP8 fused Adam.");

  // Check tensor dtypes
  const auto g_in_type_te = tensor_lists[0][0]->dtype();
  const auto p_in_type_te = tensor_lists[1][0]->dtype();
  const auto moment_type_te = tensor_lists[2][0]->dtype();
  for (size_t j = 0; j < num_tensors_per_list; ++j) {
    NVTE_CHECK(tensor_lists[0][j]->dtype() == g_in_type_te, "Grad tensor ", j,
               " has dtype=", to_string(tensor_lists[0][j]->dtype()),
               ", but expected dtype=", to_string(g_in_type_te));
    NVTE_CHECK(tensor_lists[1][j]->dtype() == p_in_type_te, "Param tensor ", j,
               " has dtype=", to_string(tensor_lists[1][j]->dtype()),
               ", but expected dtype=", to_string(p_in_type_te));
    {
      const bool m_is_fp32 = tensor_lists[2][j]->dtype() == DType::kFloat32;
      const bool m_is_bf16 = tensor_lists[2][j]->dtype() == DType::kBFloat16;
      const bool v_is_fp32 = tensor_lists[3][j]->dtype() == DType::kFloat32;
      const bool v_is_bf16 = tensor_lists[3][j]->dtype() == DType::kBFloat16;
      NVTE_CHECK((m_is_fp32 && v_is_fp32) || (m_is_bf16 && v_is_bf16),
                 "First and second moment tensors must both be Float32 or both be BFloat16, but "
                 "tensor ",
                 j, " has first moment dtype=", to_string(tensor_lists[2][j]->dtype()),
                 " and second moment dtype=", to_string(tensor_lists[3][j]->dtype()));
    }
  }

  const bool use_64bit_indexing = requires_64bit_indexing(tensor_lists);

  if (use_64bit_indexing) {
    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
        p_in_type_te, p_in_type,
        TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
            g_in_type_te, g_in_type,
            TRANSFORMER_ENGINE_TYPE_SWITCH_FP32_BF16(
                moment_type_te, moment_type,
                multi_tensor_apply_mxfp8<
                    transformer_engine::multi_tensor_adam::adam_mxfp8_fused_kernel<
                        p_in_type, g_in_type, moment_type, int64_t>>(
                    chunk_size, noop_flag, tensor_lists, static_cast<uint8_t>(fp8_dtype), stream,
                    beta1, beta2, bias_correction1, bias_correction2, epsilon, lr, mode,
                    weight_decay);)));
  } else {
    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
        p_in_type_te, p_in_type,
        TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
            g_in_type_te, g_in_type,
            TRANSFORMER_ENGINE_TYPE_SWITCH_FP32_BF16(
                moment_type_te, moment_type,
                multi_tensor_apply_mxfp8<
                    transformer_engine::multi_tensor_adam::adam_mxfp8_fused_kernel<
                        p_in_type, g_in_type, moment_type, int32_t>>(
                    chunk_size, noop_flag, tensor_lists, static_cast<uint8_t>(fp8_dtype), stream,
                    beta1, beta2, bias_correction1, bias_correction2, epsilon, lr, mode,
                    weight_decay);)));
  }
}

void multi_tensor_adam_capturable_cuda(int chunk_size, Tensor noop_flag,
                                       std::vector<std::vector<Tensor *>> tensor_lists, Tensor lr,
                                       const float beta1, const float beta2, const float epsilon,
                                       Tensor step, const int mode, const int bias_correction,
                                       const float weight_decay, Tensor inv_scale,
                                       cudaStream_t stream) {
  // Check tensor list sizes
  // 4 tensor lists: g, p, m, v
  const size_t num_tensor_lists = tensor_lists.size();
  NVTE_CHECK(num_tensor_lists == 4, "Expected 4 tensor lists, but found ", num_tensor_lists);
  const size_t num_tensors_per_list = tensor_lists[0].size();
  for (size_t i = 1; i < num_tensor_lists; i++) {
    NVTE_CHECK(tensor_lists[i].size() == num_tensors_per_list, "Tensor list ", i,
               " has size=", tensor_lists[i].size(), ", but expected size=", num_tensors_per_list);
  }

  // Check tensor dtypes
  const auto g_in_type_te = tensor_lists[0][0]->dtype();
  for (size_t j = 0; j < num_tensors_per_list; j++) {
    NVTE_CHECK(tensor_lists[0][j]->dtype() == g_in_type_te, "Grad tensor ", j,
               " has dtype=", to_string(tensor_lists[0][j]->dtype()),
               ", but expected dtype=", to_string(g_in_type_te));
    NVTE_CHECK(tensor_lists[1][j]->dtype() == g_in_type_te, "Param tensor ", j,
               " has dtype=", to_string(tensor_lists[1][j]->dtype()),
               ", but expected dtype=", to_string(g_in_type_te));
    {
      const bool m_is_fp32 = tensor_lists[2][j]->dtype() == DType::kFloat32;
      const bool m_is_bf16 = tensor_lists[2][j]->dtype() == DType::kBFloat16;
      const bool v_is_fp32 = tensor_lists[3][j]->dtype() == DType::kFloat32;
      const bool v_is_bf16 = tensor_lists[3][j]->dtype() == DType::kBFloat16;
      NVTE_CHECK((m_is_fp32 && v_is_fp32) || (m_is_bf16 && v_is_bf16),
                 "First and second moment tensors must both be Float32 or both be BFloat16, but "
                 "tensor ",
                 j, " has first moment dtype=", to_string(tensor_lists[2][j]->dtype()),
                 " and second moment dtype=", to_string(tensor_lists[3][j]->dtype()));
    }
  }

  // Get moment dtype (m and v have the same dtype, already validated above)
  const auto moment_type_te = tensor_lists[2][0]->dtype();

  // Launch kernel
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      tensor_lists[0][0]->dtype(), dtype,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP32_BF16(
          moment_type_te, moment_type,
          multi_tensor_apply<4>(BLOCK_SIZE, chunk_size, noop_flag, tensor_lists,
                                AdamCapturableFunctor<dtype, float, moment_type>(), stream, beta1,
                                beta2, reinterpret_cast<int *>(step.data.dptr), bias_correction,
                                epsilon, reinterpret_cast<float *>(lr.data.dptr), (adamMode_t)mode,
                                weight_decay, reinterpret_cast<float *>(inv_scale.data.dptr));))

  NVTE_CHECK_CUDA(cudaGetLastError());
}

void multi_tensor_adam_capturable_master_cuda(int chunk_size, Tensor noop_flag,
                                              std::vector<std::vector<Tensor *>> tensor_lists,
                                              Tensor lr, const float beta1, const float beta2,
                                              const float epsilon, Tensor step, const int mode,
                                              const int bias_correction, const float weight_decay,
                                              Tensor inv_scale, cudaStream_t stream) {
  // Check tensor list sizes
  // 4 tensor lists: g, p, m, v, p_master
  const size_t num_tensor_lists = tensor_lists.size();
  NVTE_CHECK(num_tensor_lists == 5, "Expected 4 tensor lists, but found ", num_tensor_lists);
  const size_t num_tensors_per_list = tensor_lists[0].size();
  for (size_t i = 1; i < num_tensor_lists; i++) {
    NVTE_CHECK(tensor_lists[i].size() == num_tensors_per_list, "Tensor list ", i,
               " has size=", tensor_lists[i].size(), ", but expected size=", num_tensors_per_list);
  }

  // Check tensor dtypes
  const auto g_in_type_te = tensor_lists[0][0]->dtype();
  for (size_t j = 0; j < num_tensors_per_list; j++) {
    NVTE_CHECK(tensor_lists[0][j]->dtype() == g_in_type_te, "Grad tensor ", j,
               " has dtype=", to_string(tensor_lists[0][j]->dtype()),
               ", but expected dtype=", to_string(g_in_type_te));
    NVTE_CHECK(tensor_lists[1][j]->dtype() == g_in_type_te, "Param tensor ", j,
               " has dtype=", to_string(tensor_lists[1][j]->dtype()),
               ", but expected dtype=", to_string(g_in_type_te));
    {
      const bool m_is_fp32 = tensor_lists[2][j]->dtype() == DType::kFloat32;
      const bool m_is_bf16 = tensor_lists[2][j]->dtype() == DType::kBFloat16;
      const bool v_is_fp32 = tensor_lists[3][j]->dtype() == DType::kFloat32;
      const bool v_is_bf16 = tensor_lists[3][j]->dtype() == DType::kBFloat16;
      NVTE_CHECK((m_is_fp32 && v_is_fp32) || (m_is_bf16 && v_is_bf16),
                 "First and second moment tensors must both be Float32 or both be BFloat16, but "
                 "tensor ",
                 j, " has first moment dtype=", to_string(tensor_lists[2][j]->dtype()),
                 " and second moment dtype=", to_string(tensor_lists[3][j]->dtype()));
    }
    NVTE_CHECK(tensor_lists[4][j]->dtype() == DType::kFloat32, "Master param tensor ", j,
               " has dtype=", to_string(tensor_lists[4][j]->dtype()),
               ", but expected dtype=", to_string(DType::kFloat32));
  }

  // Get moment dtype (m and v have the same dtype, already validated above)
  const auto moment_type_te = tensor_lists[2][0]->dtype();

  // Launch kernel
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      tensor_lists[0][0]->dtype(), dtype,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP32_BF16(
          moment_type_te, moment_type,
          multi_tensor_apply<5>(BLOCK_SIZE, chunk_size, noop_flag, tensor_lists,
                                AdamCapturableMasterFunctor<dtype, float, moment_type>(), stream,
                                beta1, beta2, reinterpret_cast<int *>(step.data.dptr),
                                bias_correction, epsilon, reinterpret_cast<float *>(lr.data.dptr),
                                (adamMode_t)mode, weight_decay,
                                reinterpret_cast<float *>(inv_scale.data.dptr));))

  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace multi_tensor_adam
}  // namespace transformer_engine

void nvte_multi_tensor_adam_cuda(int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists,
                                 const size_t num_tensor_lists, const size_t num_tensors_per_list,
                                 const float lr, const float beta1, const float beta2,
                                 const float epsilon, const int step, const int mode,
                                 const int bias_correction, const float weight_decay,
                                 cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_adam_cuda);
  using namespace transformer_engine;

  multi_tensor_adam::multi_tensor_adam_cuda(
      chunk_size, *convertNVTETensorCheck(noop_flag),
      convert_tensor_array(tensor_lists, num_tensor_lists, num_tensors_per_list), lr, beta1, beta2,
      epsilon, step, mode, bias_correction, weight_decay, stream);
}

void nvte_multi_tensor_adam_param_remainder_cuda(
    int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists, const size_t num_tensor_lists,
    const size_t num_tensors_per_list, const float lr, const float beta1, const float beta2,
    const float epsilon, const int step, const int mode, const int bias_correction,
    const float weight_decay, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_adam_param_remainder_cuda);
  using namespace transformer_engine;

  multi_tensor_adam::multi_tensor_adam_param_remainder_cuda(
      chunk_size, *convertNVTETensorCheck(noop_flag),
      convert_tensor_array(tensor_lists, num_tensor_lists, num_tensors_per_list), lr, beta1, beta2,
      epsilon, step, mode, bias_correction, weight_decay, stream);
}

void nvte_multi_tensor_adam_fp8_cuda(int chunk_size, NVTETensor noop_flag,
                                     NVTETensor **tensor_lists, const size_t num_tensor_lists,
                                     const size_t num_tensors_per_list, const float lr,
                                     const float beta1, const float beta2, const float epsilon,
                                     const int step, const int mode, const int bias_correction,
                                     const float weight_decay, const NVTEDType fp8_dtype,
                                     cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_adam_fp8_cuda);
  using namespace transformer_engine;

  multi_tensor_adam::multi_tensor_adam_fp8_cuda(
      chunk_size, *convertNVTETensorCheck(noop_flag),
      convert_tensor_array(tensor_lists, num_tensor_lists, num_tensors_per_list), lr, beta1, beta2,
      epsilon, step, mode, bias_correction, weight_decay, static_cast<DType>(fp8_dtype), stream);
}

void nvte_multi_tensor_adam_mxfp8_cuda(
    int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists,
    const size_t num_tensor_lists, const size_t num_tensors_per_list, const NVTEDType fp8_dtype,
    const float lr, const float beta1, const float beta2, const float epsilon, const int step,
    const int mode, const int bias_correction, const float weight_decay, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_adam_mxfp8_cuda);
  using namespace transformer_engine;
  multi_tensor_adam::multi_tensor_adam_mxfp8_cuda(
      chunk_size, *convertNVTETensorCheck(noop_flag),
      convert_tensor_array(tensor_lists, num_tensor_lists, num_tensors_per_list), lr, beta1, beta2,
      epsilon, step, mode, bias_correction, weight_decay, static_cast<DType>(fp8_dtype), stream);
}

void nvte_multi_tensor_adam_capturable_cuda(
    int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists, const size_t num_tensor_lists,
    const size_t num_tensors_per_list, NVTETensor lr, const float beta1, const float beta2,
    const float epsilon, NVTETensor step, const int mode, const int bias_correction,
    const float weight_decay, NVTETensor inv_scale, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_adam_capturable_cuda);
  using namespace transformer_engine;

  multi_tensor_adam::multi_tensor_adam_capturable_cuda(
      chunk_size, *convertNVTETensorCheck(noop_flag),
      convert_tensor_array(tensor_lists, num_tensor_lists, num_tensors_per_list),
      *convertNVTETensorCheck(lr), beta1, beta2, epsilon, *convertNVTETensorCheck(step), mode,
      bias_correction, weight_decay, *convertNVTETensorCheck(inv_scale), stream);
}

void nvte_multi_tensor_adam_capturable_master_cuda(
    int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists, const size_t num_tensor_lists,
    const size_t num_tensors_per_list, NVTETensor lr, const float beta1, const float beta2,
    const float epsilon, NVTETensor step, const int mode, const int bias_correction,
    const float weight_decay, NVTETensor inv_scale, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_adam_capturable_master_cuda);
  using namespace transformer_engine;

  multi_tensor_adam::multi_tensor_adam_capturable_master_cuda(
      chunk_size, *convertNVTETensorCheck(noop_flag),
      convert_tensor_array(tensor_lists, num_tensor_lists, num_tensors_per_list),
      *convertNVTETensorCheck(lr), beta1, beta2, epsilon, *convertNVTETensorCheck(step), mode,
      bias_correction, weight_decay, *convertNVTETensorCheck(inv_scale), stream);
}
