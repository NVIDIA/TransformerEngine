/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
                      ComputeType *amax, Param p, const size_t N,
                      const size_t num_aligned_elements) {
  VectorizedLoader<InputType, nvec, aligned> loader(input, N);
  VectorizedStorer<OutputType, nvec, aligned> storer(output, N);
  ComputeType max = 0;
  ComputeType s = 0;
  if constexpr (is_fp8<OutputType>::value) {
    if (scale != nullptr) s = *scale;
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

template <int nvec, bool aligned, typename ComputeType, typename Param,
          ComputeType (*OP)(ComputeType, const Param &), typename InputType, typename InputTypeGrad,
          typename OutputType>
__launch_bounds__(unary_kernel_threads) __global__
    void unary_grad_kernel(const InputTypeGrad *grad, const InputType *input, OutputType *output,
                           const ComputeType *scale, ComputeType *amax, Param p, const size_t N,
                           const size_t num_aligned_elements) {
  VectorizedLoader<InputType, nvec, aligned> loader(input, N);
  VectorizedLoader<InputTypeGrad, nvec, aligned> grad_loader(grad, N);
  VectorizedStorer<OutputType, nvec, aligned> storer(output, N);
  ComputeType max = 0;
  ComputeType s = 0;
  if constexpr (is_fp8<OutputType>::value) {
    if (scale != nullptr) s = *scale;
  }
  const int warp_id = threadIdx.x / THREADS_PER_WARP;

  const size_t M = num_aligned_elements;

  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < M; tid += gridDim.x * blockDim.x) {
    loader.load(tid, N);
    grad_loader.load(tid, N);
#pragma unroll
    for (int i = 0; i < nvec; ++i) {
      const ComputeType val = static_cast<ComputeType>(loader.separate()[i]);
      const ComputeType g = static_cast<ComputeType>(grad_loader.separate()[i]);
      ComputeType temp = OP(val, p) * g;
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
   \param ptrs Inputs and Outputs to the operator.
*/
template <typename... T>
Alignment CheckAlignment(const size_t lead_dim, const int nvec, const T... ptrs) {
  std::vector<int> alignments;
  alignments.reserve(sizeof...(T));

  // calculate the alignments of all ptrs and store them into alignments
  (..., alignments.push_back(CalcAlignment(ptrs, sizeof(*ptrs) * nvec)));

  bool all_same = std::all_of(alignments.cbegin(), alignments.cend(),
                              [alignments](int val) { return val == alignments.front(); });
  if (!all_same) {
    return Alignment::DIFFERENT;
  }

  if (alignments.front() == 0 && lead_dim % nvec == 0) {
    // all alignment are 0
    return Alignment::SAME_ALIGNED;
  } else {
    return Alignment::SAME_UNALIGNED;
  }
}

}  // namespace

template <int nvec, typename Param, fp32 (*OP)(const fp32, const Param &), typename InputType,
          typename OutputType>
void VectorizedUnaryKernelLauncher(const InputType *input, OutputType *output, const fp32 *scale,
                                   fp32 *amax, const size_t N, const Param params,
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
            input, output, scale, amax, params, N, num_aligned_elements);
        break;
      case Alignment::SAME_UNALIGNED:
        unary_kernel<nvec, false, fp32, Param, OP><<<num_blocks, threads, 0, stream>>>(
            input, output, scale, amax, params, N, num_aligned_elements);
        break;
      case Alignment::DIFFERENT: {
        // If the pointers are aligned differently we cannot vectorize
        unary_kernel<1, true, fp32, Param, OP>
            <<<num_blocks, threads, 0, stream>>>(input, output, scale, amax, params, N, N);
        break;
      }
    }
  }
}

template <int nvec, typename Param, fp32 (*OP)(fp32, const Param &), typename InputType,
          typename InputTypeGrad, typename OutputType>
void VectorizedUnaryGradKernelLauncher(const InputTypeGrad *grad, const InputType *input,
                                       OutputType *output, const fp32 *scale, fp32 *amax,
                                       const size_t N, const Param params, cudaStream_t stream) {
  if (N != 0) {
    auto align = CheckAlignment(N, nvec, input, grad, output);

    size_t num_aligned_elements = get_num_aligned_elements(input, N, nvec, sizeof(InputType));
    constexpr size_t threads = unary_kernel_threads;
    size_t num_blocks = DIVUP(num_aligned_elements, threads);
    constexpr size_t max_blocks = 65535;
    num_blocks = std::min(num_blocks, max_blocks);

    switch (align) {
      case Alignment::SAME_ALIGNED:
        unary_grad_kernel<nvec, true, fp32, Param, OP><<<num_blocks, threads, 0, stream>>>(
            grad, input, output, scale, amax, params, N, num_aligned_elements);
        break;
      case Alignment::SAME_UNALIGNED:
        unary_grad_kernel<nvec, false, fp32, Param, OP><<<num_blocks, threads, 0, stream>>>(
            grad, input, output, scale, amax, params, N, num_aligned_elements);
        break;
      case Alignment::DIFFERENT: {
        // If the pointers are aligned differently we cannot vectorize
        unary_grad_kernel<1, true, fp32, Param, OP>
            <<<num_blocks, threads, 0, stream>>>(grad, input, output, scale, amax, params, N, N);
        break;
      }
    }
  }
}

template <int nvec, bool aligned, typename ComputeType, typename Param,
          ComputeType (*Activation)(const ComputeType, const Param &), typename InputType,
          typename OutputType>
__launch_bounds__(unary_kernel_threads) __global__
    void gated_act_kernel(const InputType *input, OutputType *output, const ComputeType *scale,
                          ComputeType *amax, const size_t m, const size_t n, const Param p,
                          const size_t num_aligned_elements) {
  const size_t M = num_aligned_elements * m;
  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < M; tid += gridDim.x * blockDim.x) {
    const size_t id_x = tid % num_aligned_elements;
    const size_t id_y = tid / num_aligned_elements;
    VectorizedLoader<InputType, nvec, aligned> loader0(input + id_y * n * 2, n);
    VectorizedLoader<InputType, nvec, aligned> loader1(input + id_y * n * 2 + n, n);
    VectorizedStorer<OutputType, nvec, aligned> storer(output + id_y * n, n);
    ComputeType max = 0;
    ComputeType s = 0;
    if constexpr (is_fp8<OutputType>::value) {
      if (scale != nullptr) s = *scale;
    }
    const int warp_id = threadIdx.x / THREADS_PER_WARP;

    loader0.load(id_x, n);
    loader1.load(id_x, n);
#pragma unroll
    for (int i = 0; i < nvec; ++i) {
      const ComputeType val = static_cast<ComputeType>(loader0.separate()[i]);
      const ComputeType val2 = static_cast<ComputeType>(loader1.separate()[i]);
      ComputeType temp = static_cast<ComputeType>(Activation(val, p) * val2);
      if constexpr (is_fp8<OutputType>::value) {
        __builtin_assume(max >= 0);
        max = fmaxf(fabsf(temp), max);
        temp = temp * s;
      }
      storer.separate()[i] = static_cast<OutputType>(static_cast<ComputeType>(temp));
    }
    storer.store(id_x, n);

    if constexpr (is_fp8<OutputType>::value) {
      /* warp tile amax reduce*/
      max = reduce_max<unary_kernel_threads / THREADS_PER_WARP>(max, warp_id);

      if (threadIdx.x == 0 && amax != nullptr) {
        static_assert(std::is_same<ComputeType, float>::value);
        atomicMaxFloat(amax, max);
      }
    }
  }
}

template <int nvec, typename ComputeType, typename Param,
          ComputeType (*Activation)(const ComputeType, const Param &), typename InputType,
          typename OutputType>
void GatedActivationKernelLauncher(const InputType *input, OutputType *output, const fp32 *scale,
                                   fp32 *amax, const size_t m, const size_t n, const Param &p,
                                   cudaStream_t stream) {
  if (m != 0 && n != 0) {
    size_t num_aligned_elements = get_num_aligned_elements(input, n, nvec, sizeof(InputType));
    constexpr size_t threads = unary_kernel_threads;
    size_t num_blocks = DIVUP(num_aligned_elements * m, threads);
    constexpr size_t max_blocks = 65535;
    num_blocks = std::min(num_blocks, max_blocks);

    switch (auto align = CheckAlignment(n, nvec, input, input + n, output)) {
      case Alignment::SAME_ALIGNED:
        gated_act_kernel<nvec, true, ComputeType, Param, Activation>
            <<<num_blocks, threads, 0, stream>>>(input, output, scale, amax, m, n, p,
                                                 num_aligned_elements);
        break;
      case Alignment::SAME_UNALIGNED:
        gated_act_kernel<nvec, false, ComputeType, Param, Activation>
            <<<num_blocks, threads, 0, stream>>>(input, output, scale, amax, m, n, p,
                                                 num_aligned_elements);
        break;
      case Alignment::DIFFERENT: {
        // If the pointers are aligned differently we cannot vectorize
        gated_act_kernel<1, true, ComputeType, Param, Activation>
            <<<num_blocks, threads, 0, stream>>>(input, output, scale, amax, m, n, p, n);
        break;
      }
    }
  }
}

template <int nvec, bool aligned, typename ComputeType, typename Param,
          ComputeType (*Activation)(const ComputeType, const Param &),
          ComputeType (*Dactivation)(const ComputeType, const Param &), typename InputType,
          typename OutputType>
__launch_bounds__(unary_kernel_threads) __global__
    void dgated_act_kernel(const InputType *grad, const InputType *input, OutputType *output,
                           const size_t m, const size_t n, const Param p,
                           const size_t num_aligned_elements) {
  const size_t M = num_aligned_elements * m;
  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < M; tid += gridDim.x * blockDim.x) {
    const size_t id_x = tid % num_aligned_elements;
    const size_t id_y = tid / num_aligned_elements;
    VectorizedLoader<InputType, nvec, aligned> grad_loader(grad + id_y * n, n);
    VectorizedLoader<InputType, nvec, aligned> input_loader0(input + id_y * n * 2, n);
    VectorizedLoader<InputType, nvec, aligned> input_loader1(input + id_y * n * 2 + n, n);
    VectorizedStorer<OutputType, nvec, aligned> storer0(output + id_y * n * 2, n);
    VectorizedStorer<OutputType, nvec, aligned> storer1(output + id_y * n * 2 + n, n);

    grad_loader.load(id_x, n);
    input_loader0.load(id_x, n);
    input_loader1.load(id_x, n);

#pragma unroll
    for (int i = 0; i < nvec; ++i) {
      const ComputeType grad_val = static_cast<ComputeType>(grad_loader.separate()[i]);
      const ComputeType gelu_in = static_cast<ComputeType>(input_loader0.separate()[i]);
      const ComputeType gate_in = static_cast<ComputeType>(input_loader1.separate()[i]);

      ComputeType after_dgelu = Dactivation(gelu_in, p) * grad_val * gate_in;
      ComputeType after_dgate = grad_val * Activation(gelu_in, p);

      storer0.separate()[i] = static_cast<OutputType>(after_dgelu);
      storer1.separate()[i] = static_cast<OutputType>(after_dgate);
    }
    storer0.store(id_x, n);
    storer1.store(id_x, n);
  }
}

template <int nvec, typename ComputeType, typename Param,
          ComputeType (*Activation)(const ComputeType, const Param &),
          ComputeType (*Dactivation)(const ComputeType, const Param &), typename InputType,
          typename OutputType>
void DGatedActivationKernelLauncher(const InputType *grad, const InputType *input,
                                    OutputType *output, const size_t m, const size_t n,
                                    const Param &p, cudaStream_t stream) {
  if (m != 0 && n != 0) {
    size_t num_aligned_elements = get_num_aligned_elements(grad, n, nvec, sizeof(InputType));
    constexpr size_t threads = unary_kernel_threads;
    size_t num_blocks = DIVUP(num_aligned_elements * m, threads);
    constexpr size_t max_blocks = 65535;
    num_blocks = std::min(num_blocks, max_blocks);

    switch (auto align = CheckAlignment(n, nvec, input, input + n, output, output + n)) {
      case Alignment::SAME_ALIGNED:
        dgated_act_kernel<nvec, true, ComputeType, Param, Activation, Dactivation>
            <<<num_blocks, threads, 0, stream>>>(grad, input, output, m, n, p,
                                                 num_aligned_elements);
        break;
      case Alignment::SAME_UNALIGNED:
        dgated_act_kernel<nvec, false, ComputeType, Param, Activation, Dactivation>
            <<<num_blocks, threads, 0, stream>>>(grad, input, output, m, n, p,
                                                 num_aligned_elements);
        break;
      case Alignment::DIFFERENT: {
        // If the pointers are aligned differently we cannot vectorize
        dgated_act_kernel<1, true, ComputeType, Param, Activation, Dactivation>
            <<<num_blocks, threads, 0, stream>>>(grad, input, output, m, n, p, n);
        break;
      }
    }
  }
}

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_VECTORIZED_POINTWISE_H_
