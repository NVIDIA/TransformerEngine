#ifndef FILTER_CUH_
#define FILTER_CUH_

#include "nv_util.h"
#include <cuda/std/limits>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace nv {
template <typename T> __forceinline__ __device__ T cas(T *f, T val);

template <> __forceinline__ __device__ float cas(float *f, float val) {
  float res = __int_as_float(
      atomicCAS(reinterpret_cast<int *>(f),
                __float_as_int(cuda::std::numeric_limits<float>::max()),
                __float_as_int(val)));
  return res;
}

template <> __forceinline__ __device__ int cas(int *f, int val) {
  int res = atomicCAS(reinterpret_cast<int *>(f),
                      cuda::std::numeric_limits<int>::max(), val);
  return res;
}

template <> __forceinline__ __device__ __half cas(__half *f, __half val) {
  __half res =
      atomicCAS(reinterpret_cast<unsigned short int *>(f),
                __half_as_ushort(cuda::std::numeric_limits<__half>::max()),
                __half_as_ushort(val));
  return res;
}

template <>
__forceinline__ __device__ __nv_bfloat16 cas(__nv_bfloat16 *f,
                                             __nv_bfloat16 val) {
  __nv_bfloat16 res = atomicCAS(
      reinterpret_cast<unsigned short int *>(f),
      __bfloat16_as_ushort(cuda::std::numeric_limits<__nv_bfloat16>::max()),
      __bfloat16_as_ushort(val));
  return res;
}
template <typename T> __forceinline__ __device__ bool is_max(T val) {
  bool res = (val == cuda::std::numeric_limits<T>::max());
  return res;
}

// empirically, if k < len / BLOCK_DIM, this is faster than using shared memory
template <typename T, typename idxT>
__global__ void
filter_for_non_equal_value(const T *in, idxT len, const T *kth_value,
                           idxT *counter, T *out, idxT *out_idx, bool greater) {
  const T pivot = *kth_value;

  for (idxT i = blockIdx.x * blockDim.x + threadIdx.x; i < len;
       i += blockDim.x * gridDim.x) {
    const T val = in[i];
    if ((greater && val > pivot) || (!greater && val < pivot)) {
      idxT old_pos = atomicAdd(counter, (idxT)1);
      out[old_pos] = val;
      if (out_idx) {
        out_idx[old_pos] = i;
      }
    }
  }
}

template <typename T, typename idxT>
__global__ void filter_for_equal_value(const T *in, idxT len, idxT k,
                                       const T *kth_value, idxT *counter,
                                       T *out, idxT *out_idx) {
  if (*counter >= k) {
    return;
  }

  const T pivot = *kth_value;
  for (idxT i = blockIdx.x * blockDim.x + threadIdx.x; i < len;
       i += blockDim.x * gridDim.x) {
    if (*counter >= k) {
      return;
    }
    const T val = in[i];
    if (val == pivot) {
      idxT old_pos = atomicAdd(counter, (idxT)1);
      if (old_pos < k) {
        out[old_pos] = val;
        if (out_idx) {
          out_idx[old_pos] = i;
        }
      } else {
        return;
      }
    }
  }
}

template <typename T, typename idxT>
__global__ void batch_filter_for_non_equal_value(const T *in, idxT len, idxT k,
                                                 const T *kth_value,
                                                 idxT *counter, T *out,
                                                 idxT *out_idx, bool greater) {
  int batch_id = blockIdx.x;

  const T pivot = *(kth_value + batch_id);
  counter = counter + batch_id;
  in = in + batch_id * len;
  out = out + batch_id * k;
  out_idx = out_idx + batch_id * k;

  for (idxT i = blockIdx.y * blockDim.x + threadIdx.x; i < len;
       i += blockDim.x * gridDim.y) {
    const T val = in[i];
    if ((greater && val > pivot) || (!greater && val < pivot)) {
      idxT old_pos = atomicAdd(counter, (idxT)1);
      if (old_pos < k) {
        out[old_pos] = val;
        out_idx[old_pos] = i;
      }
    }
  }
}

template <typename T, typename idxT>
__global__ void batch_filter_for_equal_value(const T *in, idxT len, idxT k,
                                             const T *kth_value, idxT *counter,
                                             T *out, idxT *out_idx) {
  int batch_id = blockIdx.x;

  const T pivot = *(kth_value + batch_id);
  counter = counter + batch_id;
  in = in + batch_id * len;
  out = out + batch_id * k;
  out_idx = out_idx + batch_id * k;

  if (*counter >= k) {
    return;
  }

  for (idxT i = blockIdx.y * blockDim.x + threadIdx.x; i < len;
       i += blockDim.x * gridDim.y) {
    if (*counter >= k) {
      return;
    }
    const T val = in[i];
    if (val == pivot) {
      idxT old_pos = atomicAdd(counter, (idxT)1);
      if (old_pos < k) {
        out[old_pos] = val;
        if (out_idx) {
          out_idx[old_pos] = i;
        }
      } else {
        return;
      }
    }
  }
}

template <typename T, typename idxT>
__global__ void batch_filter(const T *in, const idxT len, const idxT k,
                             const T *kth_value, idxT *counter,
                             idxT *reverse_counter, T *out, idxT *out_idx,
                             bool greater) {
  int batch_id = blockIdx.y;

  const T pivot = *(kth_value + batch_id);
  counter = counter + batch_id;
  reverse_counter = reverse_counter + batch_id;
  in = in + batch_id * len;
  out = out + batch_id * k;
  out_idx = out_idx + batch_id * k;

  idxT old_pos;
  T val;
  bool if_set_equal = true;
  T val_old;
  for (idxT i = blockIdx.x * blockDim.x + threadIdx.x; i < len;
       i += blockDim.x * gridDim.x) {
    val = in[i];
    if ((greater && val > pivot) || (!greater && val < pivot)) {
      old_pos = atomicAdd(counter, (idxT)1);
      if (old_pos < k) {
        out[old_pos] = val;
        out_idx[old_pos] = i;
      }
    } else if (val == pivot && if_set_equal) {
      old_pos = atomicAdd(reverse_counter, (idxT)-1);
      if (old_pos >= 0) {
        // maybe we can set some flag if the cas failed, which means we don't
        // need to anything when val==pivot in the future
        val_old = cas(&out[old_pos], val);
        if (!is_max(val_old)) {
          if_set_equal = false;
        }
        // atomicCAS(reinterpret_cast<int
        // *>(&out[old_pos]),__float_as_int(FLT_MAX), __float_as_int(val));
        atomicCAS(reinterpret_cast<int *>(&out_idx[old_pos]), -1, int(i));
      }
    } // end else if
  }   // end for
} // end func

// both 'kth_value' and 'counter' point to scalar not array
// these scalars must reside in global memory
// and '*counter' must have value 0
template <typename T, typename idxT>
void filter(const T *in, idxT len, idxT k, const T *kth_value, idxT *counter,
            T *out, idxT *out_idx, bool greater, cudaStream_t stream) {
  const int BLOCK_DIM = 256;
  const int ITEM_PER_THREAD = 32;
  int nblock = ((len - 1) / BLOCK_DIM + 1) / ITEM_PER_THREAD;
  if (nblock == 0) {
    nblock = 1;
  }

  filter_for_non_equal_value<<<nblock, BLOCK_DIM, 0, stream>>>(
      in, len, kth_value, counter, out, out_idx, greater);

  filter_for_equal_value<<<nblock, BLOCK_DIM, 0, stream>>>(
      in, len, k, kth_value, counter, out, out_idx);
}

template <typename T, typename idxT>
void filter(const T *in, int batch_size, idxT len, idxT k, const T *kth_value,
            idxT *counter, T *out, idxT *out_idx, bool greater,
            cudaStream_t stream) {
  const int BLOCK_DIM = 256;
  const int ITEM_PER_THREAD = 32;
  int nblock = ((len - 1) / BLOCK_DIM + 1) / ITEM_PER_THREAD;
  if (nblock == 0) {
    nblock = 1;
  }

  dim3 blocks(batch_size, nblock);
  batch_filter_for_non_equal_value<<<blocks, BLOCK_DIM, 0, stream>>>(
      in, len, k, kth_value, counter, out, out_idx, greater);

  batch_filter_for_equal_value<<<blocks, BLOCK_DIM, 0, stream>>>(
      in, len, k, kth_value, counter, out, out_idx);
}

template <typename T, typename idxT>
void filter(const T *in, int batch_size, idxT len, idxT k, const T *kth_value,
            idxT *counter, idxT *reverse_counter, T *out, idxT *out_idx,
            bool greater, cudaStream_t stream) {
  const int BLOCK_DIM = 256;
  const int ITEM_PER_THREAD = 32;
  int nblock = ((len - 1) / BLOCK_DIM + 1) / ITEM_PER_THREAD;
  if (nblock == 0) {
    nblock = 1;
  }

  dim3 blocks(nblock, batch_size);
  /*batch_filter_for_non_equal_value<<<blocks, BLOCK_DIM, 0, stream>>>(
          in, len,k, kth_value, counter, out, out_idx, greater);

  batch_filter_for_equal_value<<<blocks, BLOCK_DIM, 0, stream>>>(
          in, len, k, kth_value, counter, out, out_idx);
   */
  batch_filter<<<blocks, BLOCK_DIM, 0, stream>>>(
      in, len, k, kth_value, counter, reverse_counter, out, out_idx, greater);

  TOPK_CUDA_CHECK(cudaGetLastError());
}

} // namespace nv
#endif
