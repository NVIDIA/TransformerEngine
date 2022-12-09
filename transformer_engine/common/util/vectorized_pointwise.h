/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_VECTORIZED_POINTWISE_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_VECTORIZED_POINTWISE_H_

#include <type_traits>
#include "../common.h"
#include "../utils.cuh"

namespace transformer_engine {

/* \brief Helper class that enables storing multiple values of type DType
          as 1 value of type LType.
*/
template <typename DType, int n>
class VectorizedStorage {
 public:
  using LType = typename transformer_engine::BytesToType<sizeof(DType) * n>::Type;
  constexpr static int nvec = n;
  union vectorized_storage {
    LType aligned;
    DType separate[nvec];  // NOLINT(*)

    inline __device__ vectorized_storage() {}
    inline __device__ ~vectorized_storage() {}
  } scratch_;

  inline __device__ VectorizedStorage() {}
  inline __device__ VectorizedStorage(const VectorizedStorage<DType, n> &y2) {
    scratch_.aligned = y2.scratch_.aligned;
  }
  inline __device__ VectorizedStorage(const LType &y2) { scratch_.aligned = y2; }
  inline __device__ VectorizedStorage<DType, n> &operator+=(
      const VectorizedStorage<DType, n> &rhs) {
#pragma unroll
    for (int i = 0; i < nvec; ++i) {
      scratch_.separate[i] = add_elem(scratch_.separate[i], rhs.scratch_.separate[i]);
    }
    return *this;
  }
  inline __device__ ~VectorizedStorage() {}
};

// Returns const LType is DType is const
template <typename DType, typename LType>
struct select_const {
  using type = LType;
};

template <typename DType, typename LType>
struct select_const<const DType, LType> {
  using type = const LType;
};

/* \brief Helper class that enables accessing multiple values of type DType
          as 1 value of type LType. Additional aligned template argument
          allows performance optimizations if the pointer and the size of
          the allocation is aligned to sizeof(LType) / sizeof(DType) elements.
*/
template <typename DType, int nvec, bool aligned = false>
class VectorizedAccessor {
 public:
  using StorageType = VectorizedStorage<typename std::remove_const<DType>::type, nvec>;
  using LType = typename select_const<DType, typename StorageType::LType>::type;
  StorageType storage_;

  LType *aligned_ptr_;
  DType *unaligned_ptr_;
  int alignment_;
  size_t n_elems_;

  inline __device__ VectorizedAccessor(DType *const ptr, const size_t size) {
    unaligned_ptr_ = ptr;
    if (aligned) {
      alignment_ = 0;
      aligned_ptr_ = reinterpret_cast<LType *>(ptr);
      n_elems_ = (size + nvec - 1) / nvec;
    } else {
      size_t ptr_as_number = reinterpret_cast<size_t>(ptr);
      alignment_ = (ptr_as_number % sizeof(LType)) / sizeof(DType);
      aligned_ptr_ = reinterpret_cast<LType *>(ptr - alignment_);
      n_elems_ = (size + alignment_ + nvec - 1) / nvec;
    }
  }

  /* \brief Alignment of the input pointer in elements. */
  inline __device__ int alignment() const { return alignment_; }

  /* \brief Access to separate elements. */
  inline __device__ DType *separate() { return storage_.scratch_.separate; }

  /* \brief Number of aligned elements that span the entire input tensor. */
  inline __device__ size_t num_aligned_elements() const { return n_elems_; }

  /* \brief Load values from the input.
     \param id Aligned index of the element.
     \param N size of the tensor.
  */
  inline __device__ void load(const size_t id, const size_t N) {
    if (aligned) {
      storage_.scratch_.aligned = aligned_ptr_[id];
    } else {
      if (id > 0 && id < n_elems_ - 1) {
        storage_.scratch_.aligned = aligned_ptr_[id];
      } else {
#pragma unroll
        for (int j = 0; j < nvec; ++j) {
          DType *ptr = reinterpret_cast<DType *>(&(aligned_ptr_[id])) + j;
          if (reinterpret_cast<size_t>(ptr) >= reinterpret_cast<size_t>(unaligned_ptr_) &&
              reinterpret_cast<size_t>(ptr) < reinterpret_cast<size_t>(unaligned_ptr_ + N)) {
            storage_.scratch_.separate[j] = *ptr;
          } else {
            storage_.scratch_.separate[j] = DType();
          }
        }
      }
    }
  }
};

/* \brief Class used for vectorized read-only access. */
template <typename DType, int nvec, bool aligned = false>
class VectorizedLoader : public VectorizedAccessor<const DType, nvec, aligned> {
 public:
  inline __device__ VectorizedLoader(const DType *ptr, const size_t N)
      : VectorizedAccessor<const DType, nvec, aligned>(ptr, N) {}
};

/* \brief Class used for vectorized writable access. */
template <typename DType, int nvec, bool aligned = false>
class VectorizedStorer : public VectorizedAccessor<DType, nvec, aligned> {
 public:
  inline __device__ VectorizedStorer(DType *ptr, const size_t N)
      : VectorizedAccessor<DType, nvec, aligned>(ptr, N) {}

  /* \brief Store values to the output.
     \param id Aligned index of the element.
     \param N size of the tensor.
  */
  inline __device__ void store(const size_t id, const size_t N) {
    if (aligned) {
      this->aligned_ptr_[id] = this->storage_.scratch_.aligned;
    } else {
      if (id > 0 && id < this->n_elems_ - 1) {
        this->aligned_ptr_[id] = this->storage_.scratch_.aligned;
      } else {
#pragma unroll
        for (int j = 0; j < nvec; ++j) {
          DType *ptr = reinterpret_cast<DType *>(&(this->aligned_ptr_[id])) + j;
          if (reinterpret_cast<size_t>(ptr) >= reinterpret_cast<size_t>(this->unaligned_ptr_) &&
              reinterpret_cast<size_t>(ptr) < reinterpret_cast<size_t>(this->unaligned_ptr_ + N)) {
            *ptr = this->storage_.scratch_.separate[j];
          }
        }
      }
    }
  }
};

constexpr int unary_kernel_threads = 512;

template <int nvec, bool aligned, typename ComputeType, typename Param,
          ComputeType (*OP)(ComputeType, const Param &), typename InputType, typename OutputType>
__launch_bounds__(unary_kernel_threads) __global__
    void unary_kernel(const InputType *input, OutputType *output, const ComputeType *scale,
                      ComputeType *scale_inv, ComputeType *amax, Param p, const size_t N,
                      const size_t num_aligned_elements) {
  VectorizedLoader<InputType, nvec, aligned> loader(input, N);
  VectorizedStorer<OutputType, nvec, aligned> storer(output, N);
  ComputeType max = 0;
  ComputeType s = 0;
  if constexpr (is_fp8<OutputType>::value) {
    if (scale != nullptr) s = *scale;
    if (blockIdx.x == 0 && threadIdx.x == 0 && scale_inv != nullptr) {
      reciprocal<ComputeType>(scale_inv, s);
    }
  }
  const int warp_id = threadIdx.x / THREADS_PER_WARP;

  const size_t M = num_aligned_elements;

  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < M; tid += gridDim.x * blockDim.x) {
    loader.load(tid, N);
#pragma unroll
    for (int i = 0; i < nvec; ++i) {
      const ComputeType val = static_cast<ComputeType>(loader.separate()[i]);
      ComputeType temp = OP(val, p);
      if constexpr (is_fp8<OutputType>::value) {
        __builtin_assume(max >= 0);
        max = fmaxf(fabsf(temp), max);

        temp = temp * s;
      }

      storer.separate()[i] = static_cast<OutputType>(temp);
    }
    storer.store(tid, N);
  }
  if constexpr (is_fp8<OutputType>::value) {
    /* warp tile amax reduce*/
    max = reduce_max<unary_kernel_threads / THREADS_PER_WARP>(max, warp_id);

    if (threadIdx.x == 0 && amax != nullptr) {
      static_assert(std::is_same<ComputeType, float>::value);
      atomicMaxFloat(amax, max);
    }
  }
}

namespace {

inline size_t get_num_aligned_elements(const void *ptr, const size_t lead_dim, const int nvec,
                                       const int size) {
  size_t ptr_as_number = reinterpret_cast<size_t>(ptr);
  int alignment = (ptr_as_number % (nvec * size)) / size;
  return DIVUP(lead_dim + alignment, static_cast<size_t>(nvec));
}

enum class Alignment {
  SAME_ALIGNED,    // All tensors aligned
  SAME_UNALIGNED,  // All tensors have the same misalignment
  DIFFERENT        // Tensors have different alignment
};

inline int CalcAlignment(const void *ptr, const int size) {
  size_t ptr_as_number = reinterpret_cast<size_t>(ptr);
  return ptr_as_number % size;
}

/* \brief Check alignment of the inputs and outputs when using vectorized accesses.
   \param lead_dim Leading dimension of the tensors.
   \param other_dim The size of the other dimensions of the tensors.
   \param nvec Length of the vector.
   \param inputs Inputs to the operator.
   \param outputs Outputs of the operator.
*/
template <typename InputType, typename OutputType>
Alignment CheckAlignment(const size_t lead_dim, const int nvec, const InputType *input,
                         const OutputType *output) {
  int align = -1;

  if (input != nullptr) {
    int new_align = CalcAlignment(input, sizeof(InputType) * nvec);
    if (align == -1) {
      align = new_align;
    } else {
      if (align != new_align) {
        return Alignment::DIFFERENT;
      }
    }
  }

  if (output != nullptr) {
    int new_align = CalcAlignment(output, sizeof(OutputType) * nvec);
    if (align == -1) {
      align = new_align;
    } else {
      if (align != new_align) {
        return Alignment::DIFFERENT;
      }
    }
  }

  if ((align == 0) && (lead_dim % nvec == 0)) {
    return Alignment::SAME_ALIGNED;
  } else {
    return Alignment::SAME_UNALIGNED;
  }
}

}  // namespace

template <int nvec, typename Param, fp32 (*OP)(fp32, const Param &), typename InputType,
          typename OutputType>
void VectorizedUnaryKernelLauncher(const InputType *input, OutputType *output, const fp32 *scale,
                                   fp32 *scale_inv, fp32 *amax, const size_t N, const Param params,
                                   cudaStream_t stream) {
  if (N != 0) {
    auto align = CheckAlignment(N, nvec, input, output);

    size_t num_aligned_elements = get_num_aligned_elements(input, N, nvec, sizeof(InputType));
    constexpr size_t threads = unary_kernel_threads;
    size_t num_blocks = DIVUP(num_aligned_elements, threads);
    constexpr size_t max_blocks = 65535;
    num_blocks = std::min(num_blocks, max_blocks);

    switch (align) {
      case Alignment::SAME_ALIGNED:
        unary_kernel<nvec, true, fp32, Param, OP><<<num_blocks, threads, 0, stream>>>(
            input, output, scale, scale_inv, amax, params, N, num_aligned_elements);
        break;
      case Alignment::SAME_UNALIGNED:
        unary_kernel<nvec, false, fp32, Param, OP><<<num_blocks, threads, 0, stream>>>(
            input, output, scale, scale_inv, amax, params, N, num_aligned_elements);
        break;
      case Alignment::DIFFERENT: {
        // If the pointers are aligned differently we cannot vectorize
        unary_kernel<1, true, fp32, Param, OP><<<num_blocks, threads, 0, stream>>>(
            input, output, scale, scale_inv, amax, params, N, N);
        break;
      }
    }
  }
}

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_VECTORIZED_POINTWISE_H_
