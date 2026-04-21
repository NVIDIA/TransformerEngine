/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_bf16.h>

#include <cub/device/device_radix_sort.cuh>
#include <cub/util_allocator.cuh>
#include <vector>
namespace cg = cooperative_groups;

// Workspace pointer-alignment helpers.
inline size_t calc_aligned_size(const std::vector<size_t> &sizes) {
  const size_t ALIGN_BYTES = 256;
  const size_t ALIGN_MASK = ~(ALIGN_BYTES - 1);
  size_t total = 0;
  for (auto sz : sizes) total += (sz + ALIGN_BYTES - 1) & ALIGN_MASK;
  return total + ALIGN_BYTES - 1;
}
inline std::vector<void *> calc_aligned_pointers(const void *p, const std::vector<size_t> &sizes) {
  const size_t ALIGN_BYTES = 256;
  const size_t ALIGN_MASK = ~(ALIGN_BYTES - 1);
  char *ptr =
      reinterpret_cast<char *>((reinterpret_cast<size_t>(p) + ALIGN_BYTES - 1) & ALIGN_MASK);
  std::vector<void *> ptrs;
  ptrs.reserve(sizes.size());
  for (auto sz : sizes) {
    ptrs.push_back(ptr);
    ptr += (sz + ALIGN_BYTES - 1) & ALIGN_MASK;
  }
  return ptrs;
}

namespace nv {

constexpr int VECTORIZED_READ_SIZE = 16;
constexpr int WARP_SIZE = 32;
constexpr int WARP_BITS = 5;
constexpr unsigned FULL_WARP_MASK = 0xffffffff;

namespace topk {
using WideT = float4;

#ifdef __CUDA_ARCH__
using ::atomicAdd;
inline __device__ size_t atomicAdd(size_t *address, size_t value) {
  static_assert(sizeof(size_t) == sizeof(uint64_t));
  return atomicAdd(reinterpret_cast<uint64_t *>(address), static_cast<uint64_t>(value));
}
#endif

template <int BitsPerPass>
__host__ __device__ constexpr int calc_num_buckets() {
  return 1 << BitsPerPass;
}

/**
 * @brief Provide a ceiling division operation ie. ceil(a / b)
 * @tparam IntType supposed to be only integers for now!
 */
template <typename IntType>
constexpr __host__ __device__ IntType ceildiv(IntType a, IntType b) {
  return (a + b - 1) / b;
}

/**
 * @brief Provide an alignment function ie. ceil(a / b) * b
 * @tparam IntType supposed to be only integers for now!
 */
template <typename IntType>
constexpr __host__ __device__ IntType alignTo(IntType a, IntType b) {
  return ceildiv(a, b) * b;
}

template <typename T, int BitsPerPass>
__host__ __device__ constexpr int calc_num_passes() {
  return ceildiv<int>(sizeof(T) * 8, BitsPerPass);
}

__host__ __device__ int round(int num, int round_value) {
  return ((num - 1) / round_value + 1) * round_value;
}

/**
 * Bit 0 is the least significant (rightmost);
 * this implementation processes input from the most to the least significant
 * bit. This way, we can skip some passes in the end at the cost of having an
 * unsorted output.
 *
 * NB: Use pass=-1 for calc_mask().
 */
template <typename T, int BitsPerPass>
__device__ constexpr int calc_start_bit(int pass) {
  int start_bit = static_cast<int>(sizeof(T) * 8) - (pass + 1) * BitsPerPass;
  if (start_bit < 0) {
    start_bit = 0;
  }
  return start_bit;
}

template <typename T, int BitsPerPass>
__device__ constexpr unsigned calc_mask(int pass) {
  static_assert(BitsPerPass <= 31);
  int num_bits = calc_start_bit<T, BitsPerPass>(pass - 1) - calc_start_bit<T, BitsPerPass>(pass);
  return (1 << num_bits) - 1;
}

/**
 * Use CUB to twiddle bits - so that we can correctly compare bits of
 * floating-point values as well as of integers.
 */
template <typename T>
__device__ typename cub::Traits<T>::UnsignedBits twiddle_in(T key, bool select_min) {
  auto bits = reinterpret_cast<typename cub::Traits<T>::UnsignedBits &>(key);
  bits = cub::Traits<T>::TwiddleIn(bits);
  if (!select_min) {
    bits = ~bits;
  }
  return bits;
}

template <typename T>
__device__ T twiddle_out(typename cub::Traits<T>::UnsignedBits bits, bool select_min) {
  if (!select_min) {
    bits = ~bits;
  }
  bits = cub::Traits<T>::TwiddleOut(bits);
  return reinterpret_cast<T &>(bits);
}

template <typename T, int BitsPerPass>
__device__ int calc_bucket(T x, int start_bit, unsigned mask, bool select_min) {
  static_assert(BitsPerPass <= sizeof(int) * 8 - 1,
                "BitsPerPass is too large that the result type could not be int");
  return (twiddle_in(x, select_min) >> start_bit) & mask;
}

template <typename T, typename IdxT>
__host__ __device__ IdxT calc_buf_len(IdxT len) {
  // When writing is skipped, only read `in`(type T).
  // When writing is not skipped, read `in_buf`(T) and `in_idx_buf`(IdxT), and
  // write `out_buf`(T) and `out_idx_buf`(IdxT). The ratio between these cases
  // determines whether to skip writing and hence the buffer size. constexpr
  // float ratio = 2 + sizeof(IdxT) * 2.0 / sizeof(T);
  constexpr float ratio = 128;
  return len / ratio;
  // return len;
}

/**
 * Map a Func over the input data, using vectorized load instructions if
 * possible.
 *
 * NB: in future, we should move this to
 * cpp/include/raft/linalg/detail/unary_op.cuh, which currently does not support
 * the second lambda argument (index of an element)
 *
 * @tparam T element type
 * @tparam IdxT indexing type
 * @tparam Func void (T x, IdxT idx)
 *
 * @param thread_rank rank of the calling thread among all participating threads
 * @param num_threads number of the threads that participate in processing
 * @param in the input data
 * @param len the number of elements to read
 * @param f the lambda taking two arguments (T x, IdxT idx)
 */
template <typename T, typename idxT, typename Func>
__device__ void vectorized_process(size_t thread_rank, size_t num_threads, const T *in, idxT len,
                                   Func f) {
  if constexpr (sizeof(T) >= sizeof(WideT)) {
    for (idxT i = thread_rank; i < len; i += num_threads) {
      f(in[i], i);
    }
  } else {
    static_assert(sizeof(WideT) % sizeof(T) == 0);
    constexpr int items_per_scalar = sizeof(WideT) / sizeof(T);
    // TODO: it's UB
    union {
      WideT scalar;
      T array[items_per_scalar];  // NOLINT(runtime/arrays)
    } wide;

    int skip_cnt =
        (reinterpret_cast<size_t>(in) % sizeof(WideT))
            ? ((sizeof(WideT) - reinterpret_cast<size_t>(in) % sizeof(WideT)) / sizeof(T))
            : 0;
    if (skip_cnt > len) {
      skip_cnt = len;
    }
    const WideT *in_cast = reinterpret_cast<decltype(in_cast)>(in + skip_cnt);
    const idxT len_cast = (len - skip_cnt) / items_per_scalar;

    for (idxT i = thread_rank; i < len_cast; i += num_threads) {
      wide.scalar = in_cast[i];
      const idxT real_i = skip_cnt + i * items_per_scalar;
#pragma unroll
      for (int j = 0; j < items_per_scalar; ++j) {
        f(wide.array[j], real_i + j);
      }
    }

    static_assert(WARP_SIZE >= items_per_scalar);
    // and because items_per_scalar > skip_cnt, WARP_SIZE > skip_cnt
    // no need to use loop
    if (thread_rank < skip_cnt) {
      f(in[thread_rank], thread_rank);
    }
    // because len_cast = (len - skip_cnt) / items_per_scalar,
    // len_cast * items_per_scalar + items_per_scalar > len - skip_cnt;
    // and so
    // len - (skip_cnt + len_cast * items_per_scalar) < items_per_scalar <=
    // WARP_SIZE no need to use loop
    const idxT remain_i = skip_cnt + len_cast * items_per_scalar + thread_rank;
    if (remain_i < len) {
      f(in[remain_i], remain_i);
    }
  }
}

// sync_width should >= WARP_SIZE
template <typename T, typename idxT, typename Func>
__device__ void vectorized_process(const T *in, idxT len, Func f, int sync_width) {
  const idxT stride = blockDim.x * gridDim.x;
  const idxT tid = blockIdx.x * blockDim.x + threadIdx.x;
  if constexpr (sizeof(T) >= sizeof(WideT)) {
    for (idxT i = tid; i < len; i += stride) {
      f(in[i], i, true);
    }
  } else {
    static_assert(sizeof(WideT) % sizeof(T) == 0);
    constexpr int items_per_scalar = sizeof(WideT) / sizeof(T);
    union {
      WideT scalar;
      T array[items_per_scalar];  // NOLINT(runtime/arrays)
    } wide;

    int skip_cnt =
        (reinterpret_cast<size_t>(in) % sizeof(WideT))
            ? ((sizeof(WideT) - reinterpret_cast<size_t>(in) % sizeof(WideT)) / sizeof(T))
            : 0;
    if (skip_cnt > len) {
      skip_cnt = len;
    }
    const WideT *in_cast = reinterpret_cast<decltype(in_cast)>(in + skip_cnt);
    const idxT len_cast = (len - skip_cnt) / items_per_scalar;

    const idxT len_cast_for_sync = ((len_cast - 1) / sync_width + 1) * sync_width;
    for (idxT i = tid; i < len_cast_for_sync; i += stride) {
      bool valid = i < len_cast;
      if (valid) {
        wide.scalar = in_cast[i];
      }
      const idxT real_i = skip_cnt + i * items_per_scalar;
#pragma unroll
      for (int j = 0; j < items_per_scalar; ++j) {
        f(wide.array[j], real_i + j, valid);
      }
    }

    static_assert(WARP_SIZE >= items_per_scalar);
    // need at most one warp for skipped and remained elements,
    // and sync_width >= WARP_SIZE
    if (tid < sync_width) {
      bool valid = tid < skip_cnt;
      T value = valid ? in[tid] : T();
      f(value, tid, valid);

      const idxT remain_i = skip_cnt + len_cast * items_per_scalar + tid;
      valid = remain_i < len;
      value = valid ? in[remain_i] : T();
      f(value, remain_i, valid);
    }
  }
}

template <typename T, typename IdxT>
struct alignas(128) Counter {
  // We are processing the values in multiple passes, from most significant to
  // least significant. In each pass, we keep the length of input (`len`) and
  // the `k` of current pass, and update them at the end of the pass.
  IdxT k;
  IdxT len;

  //  `previous_len` is the length of input in previous pass. Note that
  //  `previous_len` rather than `len` is used for the filtering step because
  //  filtering is indeed for previous pass (see comments before
  //  `radix_kernel`).
  IdxT previous_len;

  // We determine the bits of the k_th value inside the mask processed by the
  // pass. The already known bits are stored in `kth_value_bits`. It's used to
  // discriminate a element is a result (written to `out`), a candidate for next
  // pass (written to `out_buf`), or not useful (discarded). The bits that are
  // not yet processed do not matter for this purpose.
  typename cub::Traits<T>::UnsignedBits kth_value_bits;

  // Record how many elements have passed filtering. It's used to determine the
  // position in the `out_buf` where an element should be written.
  alignas(128) IdxT filter_cnt;

  // For a row inside a batch, we may launch multiple thread blocks. This
  // counter is used to determine if the current block is the last running
  // block. If so, this block will execute scan() and choose_bucket().
  alignas(128) unsigned int finished_block_cnt;

  // Record how many elements have been written to the front of `out`. Elements
  // less (if select_min==true) than the k-th value are written from front to
  // back.
  alignas(128) IdxT out_cnt;

  // Record how many elements have been written to the back of `out`. Elements
  // equal to the k-th value are written from back to front. We need to keep
  // count of them separately because the number of elements that <= the k-th
  // value might exceed k.
  alignas(128) IdxT out_back_cnt;
};

/**
 * Fused filtering of the current pass and building histogram for the next pass
 * (see steps 4 & 1 in `radix_kernel` description).
 */
template <typename T, typename IdxT, int BitsPerPass, bool store_out>
__device__ void filter_and_histogram(const T *in_buf, const IdxT *in_idx_buf, T *out_buf,
                                     IdxT *out_idx_buf, T *out, IdxT *out_idx, IdxT previous_len,
                                     Counter<T, IdxT> *counter, IdxT *histogram, bool select_min,
                                     int pass, bool early_stop) {
  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
  __shared__ IdxT histogram_smem[num_buckets];
  for (IdxT i = threadIdx.x; i < num_buckets; i += blockDim.x) {
    histogram_smem[i] = 0;
  }
  __syncthreads();

  const int start_bit = calc_start_bit<T, BitsPerPass>(pass);
  const unsigned mask = calc_mask<T, BitsPerPass>(pass);

  if (pass == 0) {
    // Passed to vectorized_process, this function executes in all blocks in
    // parallel, i.e. the work is split along the input (both, in batches and
    // chunks of a single row). Later, the histograms are merged using
    // atomicAdd.
    auto f = [select_min, start_bit, mask](T value, IdxT) {
      int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
      atomicAdd(histogram_smem + bucket, static_cast<IdxT>(1));
    };
    vectorized_process(static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
                       static_cast<size_t>(blockDim.x) * gridDim.x, in_buf, previous_len, f);
  } else {
    IdxT *p_filter_cnt = &counter->filter_cnt;
    IdxT *p_out_cnt = &counter->out_cnt;
    const auto kth_value_bits = counter->kth_value_bits;
    const int previous_start_bit = calc_start_bit<T, BitsPerPass>(pass - 1);

    // See the remark above on the distributed execution of `f` using
    // vectorized_process.
    auto f = [in_idx_buf, out_buf, out_idx_buf, out, out_idx, select_min, start_bit, mask,
              previous_start_bit, kth_value_bits, p_filter_cnt, p_out_cnt,
              early_stop](T value, IdxT i) {
      const auto previous_bits = (twiddle_in(value, select_min) >> previous_start_bit)
                                 << previous_start_bit;
      if (previous_bits == kth_value_bits) {
        if (early_stop) {
          IdxT pos = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
          if constexpr (store_out) {
            out[pos] = value;
          }
          out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
        } else {
          if (out_buf) {
            IdxT pos = atomicAdd(p_filter_cnt, static_cast<IdxT>(1));
            out_buf[pos] = value;
            out_idx_buf[pos] = in_idx_buf ? in_idx_buf[i] : i;
          }

          int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
          atomicAdd(histogram_smem + bucket, static_cast<IdxT>(1));
        }
      }
      // the condition `(out_buf || early_stop)` is a little tricky:
      // If we skip writing to `out_buf` (when `out_buf` is nullptr), we should
      // skip writing to `out` too. So we won't write the same value to `out`
      // multiple times in different passes. And if we keep skipping the
      // writing, values will be written in `last_filter_kernel()` at last. But
      // when `early_stop` is true, we need to write to `out` since it's the
      // last chance.
      else if ((out_buf || early_stop) && previous_bits < kth_value_bits) {  // NOLINT
        IdxT pos = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
        if constexpr (store_out) {
          out[pos] = value;
        }
        out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
      }
    };
    vectorized_process(static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
                       static_cast<size_t>(blockDim.x) * gridDim.x, in_buf, previous_len, f);
  }
  if (early_stop) {
    return;
  }
  __syncthreads();

  // merge histograms produced by individual blocks
  for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
    if (histogram_smem[i] != 0) {
      atomicAdd(histogram + i, histogram_smem[i]);
    }
  }
}

/**
 * Replace histogram with its own prefix sum
 * (step 2 in `radix_kernel` description)
 */
template <typename IdxT, int BitsPerPass, int BlockSize>
__device__ void scan(volatile IdxT *histogram) {
  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
  if constexpr (num_buckets >= BlockSize) {
    static_assert(num_buckets % BlockSize == 0);
    constexpr int items_per_thread = num_buckets / BlockSize;
    typedef cub::BlockLoad<IdxT, BlockSize, items_per_thread, cub::BLOCK_LOAD_TRANSPOSE> BlockLoad;
    typedef cub::BlockStore<IdxT, BlockSize, items_per_thread, cub::BLOCK_STORE_TRANSPOSE>
        BlockStore;
    typedef cub::BlockScan<IdxT, BlockSize> BlockScan;

    __shared__ union {
      typename BlockLoad::TempStorage load;
      typename BlockScan::TempStorage scan;
      typename BlockStore::TempStorage store;
    } temp_storage;
    IdxT thread_data[items_per_thread];  // NOLINT(runtime/arrays)

    BlockLoad(temp_storage.load).Load(histogram, thread_data);
    __syncthreads();

    BlockScan(temp_storage.scan).InclusiveSum(thread_data, thread_data);
    __syncthreads();

    BlockStore(temp_storage.store).Store(histogram, thread_data);
  } else {
    typedef cub::BlockScan<IdxT, BlockSize> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    IdxT thread_data = 0;
    if (threadIdx.x < num_buckets) {
      thread_data = histogram[threadIdx.x];
    }

    BlockScan(temp_storage).InclusiveSum(thread_data, thread_data);
    __syncthreads();

    if (threadIdx.x < num_buckets) {
      histogram[threadIdx.x] = thread_data;
    }
  }
}

/**
 * Calculate in which bucket the k-th value will fall
 *  (steps 3 in `radix_kernel` description)
 */
template <typename T, typename IdxT, int BitsPerPass>
__device__ void choose_bucket(Counter<T, IdxT> *counter, const IdxT *histogram, const IdxT k,
                              const int pass) {
  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
  for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
    IdxT prev = (i == 0) ? 0 : histogram[i - 1];
    IdxT cur = histogram[i];

    // one and only one thread will satisfy this condition, so counter is
    // written by only one thread
    if (prev < k && cur >= k) {
      counter->k = k - prev;      // how many values still are there to find
      counter->len = cur - prev;  // number of values in next pass
      typename cub::Traits<T>::UnsignedBits bucket = i;
      int start_bit = calc_start_bit<T, BitsPerPass>(pass);
      counter->kth_value_bits |= bucket << start_bit;
    }
  }
}

template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
__device__ void scan_warp_version(cg::thread_block_tile<WARP_SIZE> const &warp,
                                  volatile IdxT *histogram, Counter<T, IdxT> *counter, const IdxT k,
                                  const int pass) {
  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();

  __shared__ IdxT warp_histogram[num_buckets >> WARP_BITS];
  for (int i = threadIdx.x; i < num_buckets; i += BlockSize) {
    IdxT data = histogram[i];
    IdxT warp_sum = cg::reduce(warp, data, cg::plus<IdxT>());

    if (i % WARP_SIZE == 0) {
      warp_histogram[i >> WARP_BITS] = warp_sum;
    }
  }
  __syncthreads();

  if (threadIdx.x < WARP_SIZE) {
    IdxT value = warp_histogram[threadIdx.x * 2] + warp_histogram[threadIdx.x * 2 + 1];
    IdxT prefix = value;
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
      IdxT n = __shfl_up_sync(FULL_WARP_MASK, prefix, offset, WARP_SIZE);
      if (threadIdx.x >= offset) prefix += n;
    }
    IdxT prefix_high = __shfl_sync(FULL_WARP_MASK, prefix, threadIdx.x - 1, WARP_SIZE);
    if (threadIdx.x == 0) prefix_high = 0;
    warp_histogram[threadIdx.x * 2] += prefix_high;
    warp_histogram[threadIdx.x * 2 + 1] = value + prefix_high;
    __syncwarp();

    // Find the target warp bucket
    IdxT target_warp = 2048;  // invalid value
    // bool is_one_in_warp=false;
    for (int i = threadIdx.x; i < 64 && target_warp == 2048; i += WARP_SIZE) {
      IdxT prev = (i == 0) ? 0 : warp_histogram[i - 1];
      IdxT cur = warp_histogram[i];
      bool is_selected = prev < k && cur >= k;
      unsigned mask = __ballot_sync(FULL_WARP_MASK, is_selected);
      if (__popc(mask) > 0) {
        // target_warp = __ffs(mask) -1 + (i/WARP_SIZE)*WARP_SIZE;
        target_warp = __ffs(mask) - 1 + ((i >> WARP_BITS) << WARP_BITS);
        // is_one_in_warp= (target_warp==0? warp_histogram[0]:
        // warp_histogram[target_warp]-warp_histogram[target_warp-1])==1?true:false;
      }
    }

    // Find the target bucket
    //  if(is_one_in_warp){
    //      bool is_one=histogram[target_warp*WARP_SIZE+threadIdx.x]==1?1:0;
    //      unsigned mask = __ballot_sync(FULL_WARP_MASK, is_one);
    //      IdxT target_bucket=__ffs(mask)-1+target_warp*WARP_SIZE;
    //      IdxT prev=target_warp==0? 0: warp_histogram[target_warp-1];
    //      IdxT cur=warp_histogram[target_warp];
    //      if(threadIdx.x==0) {
    //          counter->k = k - prev;      // how many values still are there
    //          to find counter->len = cur - prev;  // number of values in next
    //          pass typename cub::Traits<T>::UnsignedBits bucket =
    //          target_bucket; int start_bit = calc_start_bit<T,
    //          BitsPerPass>(pass); counter->kth_value_bits |= bucket <<
    //          start_bit;
    //      }
    //  }else{
    value = histogram[(target_warp << WARP_BITS) + threadIdx.x];
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
      IdxT n = __shfl_up_sync(FULL_WARP_MASK, value, offset, WARP_SIZE);
      if (threadIdx.x >= offset) value += n;
    }
    value += (target_warp == 0 ? 0 : warp_histogram[target_warp - 1]);

    for (int i = threadIdx.x; i < WARP_SIZE; i += WARP_SIZE) {
      IdxT prev = __shfl_up_sync(FULL_WARP_MASK, value, 1, WARP_SIZE);
      prev = (i == 0) ? (target_warp == 0 ? 0 : warp_histogram[target_warp - 1]) : prev;
      IdxT cur = value;
      if (prev < k && cur >= k) {
        counter->k = k - prev;      // how many values still are there to find
        counter->len = cur - prev;  // number of values in next pass
        typename cub::Traits<T>::UnsignedBits bucket = (target_warp << WARP_BITS) + i;
        int start_bit = calc_start_bit<T, BitsPerPass>(pass);
        counter->kth_value_bits |= bucket << start_bit;
      }
    }
    // }
  }
}
// For one-block version, last_filter() could be called when pass < num_passes
// - 1. So `pass` could not be constexpr
template <typename T, typename IdxT, int BitsPerPass, bool store_out>
__device__ void last_filter(const T *in_buf, const IdxT *in_idx_buf, T *out, IdxT *out_idx,
                            IdxT current_len, IdxT k, Counter<T, IdxT> *counter,
                            const bool select_min, const int pass) {
  const auto kth_value_bits = counter->kth_value_bits;
  const int start_bit = calc_start_bit<T, BitsPerPass>(pass);

  // changed in choose_bucket(); need to reload
  const IdxT needed_num_of_kth = counter->k;
  IdxT *p_out_cnt = &counter->out_cnt;
  IdxT *p_out_back_cnt = &counter->out_back_cnt;
  for (IdxT i = threadIdx.x; i < current_len; i += blockDim.x) {
    const T value = in_buf[i];
    const auto bits = (twiddle_in(value, select_min) >> start_bit) << start_bit;
    if (bits < kth_value_bits) {
      IdxT pos = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
      if constexpr (store_out) {
        out[pos] = value;
      }
      // For one-block version, `in_idx_buf` could be nullptr at pass 0.
      // For non one-block version, if writing has been skipped, `in_idx_buf`
      // could be nullptr if `in_buf` is `in`
      out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
    } else if (bits == kth_value_bits) {
      IdxT back_pos = atomicAdd(p_out_back_cnt, static_cast<IdxT>(1));
      if (back_pos < needed_num_of_kth) {
        IdxT pos = k - 1 - back_pos;
        if constexpr (store_out) {
          out[pos] = value;
        }
        out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
      }
    }
  }
}

template <typename T, typename IdxT, int BitsPerPass, bool store_out>
__global__ void last_filter_kernel(const T *in, const IdxT *in_idx, const T *in_buf,
                                   const IdxT *in_idx_buf, T *out, IdxT *out_idx, IdxT len, IdxT k,
                                   Counter<T, IdxT> *counters, const bool select_min) {
  const size_t batch_id = blockIdx.y;  // size_t to avoid multiplication overflow

  Counter<T, IdxT> *counter = counters + batch_id;
  IdxT previous_len = counter->previous_len;
  if (previous_len == 0) {
    return;
  }
  const IdxT buf_len = calc_buf_len<T>(len);
  if (previous_len > buf_len || in_buf == in) {
    in_buf = in + batch_id * len;
    in_idx_buf = in_idx ? (in_idx + batch_id * len) : nullptr;
    previous_len = len;
  } else {
    in_buf += batch_id * buf_len;
    in_idx_buf += batch_id * buf_len;
  }
  if constexpr (store_out) {
    out += batch_id * k;
  }
  out_idx += batch_id * k;

  constexpr int pass = calc_num_passes<T, BitsPerPass>() - 1;
  constexpr int start_bit = calc_start_bit<T, BitsPerPass>(pass);

  const auto kth_value_bits = counter->kth_value_bits;
  const IdxT needed_num_of_kth = counter->k;
  IdxT *p_out_cnt = &counter->out_cnt;
  IdxT *p_out_back_cnt = &counter->out_back_cnt;

  auto f = [k, select_min, kth_value_bits, needed_num_of_kth, p_out_cnt, p_out_back_cnt, in_idx_buf,
            out, out_idx](T value, IdxT i) {
    const auto bits = (twiddle_in(value, select_min) >> start_bit) << start_bit;
    if (bits < kth_value_bits) {
      IdxT pos = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
      if constexpr (store_out) {
        out[pos] = value;
      }
      out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
    } else if (bits == kth_value_bits) {
      IdxT back_pos = atomicAdd(p_out_back_cnt, static_cast<IdxT>(1));
      if (back_pos < needed_num_of_kth) {
        IdxT pos = k - 1 - back_pos;
        if constexpr (store_out) {
          out[pos] = value;
        }
        out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
      }
    }
  };

  vectorized_process(static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
                     static_cast<size_t>(blockDim.x) * gridDim.x, in_buf, previous_len, f);
}

/**
 *
 * It is expected to call this kernel multiple times (passes), in each pass we
 * process a radix, going from the most significant towards the least
 * significant bits (MSD).
 *
 * Conceptually, each pass consists of 4 steps:
 *
 * 1. Calculate histogram
 *      First, transform bits into a digit, the value of which is in the range
 *      [0, 2^{BITS_PER_PASS}-1]. Then count the frequency of each digit value
 * and the result is a histogram. That is, histogram[i] contains the count of
 * inputs having value i.
 *
 * 2. Scan the histogram
 *      Inclusive prefix sum is computed for the histogram. After this step,
 * histogram[i] contains the count of inputs having value <= i.
 *
 * 3. Find the bucket j of the histogram that the k-th value falls into
 *
 * 4. Filtering
 *      Input elements whose digit value <j are the top-k elements. We put them
 * into the result array out. The number of such elements is histogram[j-1].
 * Since the k-th value must be in the bucket j, we write all elements in bucket
 * j into a intermediate buffer out_buf. For the next pass, these elements are
 * used as input, and we would like to find the (k - histogram[j-1])-th value
 * among them. That is, the k in the next pass is set to (k - histogram[j-1]).
 *
 * In the implementation, the filtering step is delayed to the next pass so the
 * filtering and histogram computation are fused. In this way, inputs are read
 * once rather than twice.
 *
 * During the filtering step, we won't write candidates (elements in bucket j)
 * to `out_buf` if the number of candidates is larger than the length of
 * `out_buf` (this could happen when the leading bits of input values are almost
 * the same). And then in the next pass, inputs are read from `in` rather than
 * from `in_buf`. The benefit is that we can save the cost of writing candidates
 * and their indices.
 */
template <typename T, typename IdxT, int BitsPerPass, int BlockSize, bool fused_last_filter,
          bool store_out>
__device__ void radix_kernel_func(const T *in, const IdxT *in_idx, const T *in_buf,
                                  const IdxT *in_idx_buf, T *out_buf, IdxT *out_idx_buf, T *out,
                                  IdxT *out_idx, Counter<T, IdxT> *counter, IdxT *histogram,
                                  const IdxT len, const IdxT k, const bool select_min,
                                  const int pass) {
  if (len <= k) {
    if (pass == 0) {
      for (int index = threadIdx.x; index < len; index += BlockSize) {
        if constexpr (store_out) {
          out[index] = in[index];
        }
        out_idx[index] = in_idx ? in_idx[index] : index;
      }
      for (int index = threadIdx.x + len; index < k; index += BlockSize) {
        if constexpr (store_out) {
          out[index] = static_cast<T>(-1.0f);
        }
        out_idx[index] = -1;
      }
      return;
    } else {
      return;
    }
  }

  IdxT current_k;
  IdxT previous_len;
  IdxT current_len;
  if (pass == 0) {
    current_k = k;
    previous_len = len;
    // Need to do this so setting counter->previous_len for the next pass is
    // correct. This value is meaningless for pass 0, but it's fine because pass
    // 0 won't be the last pass in this implementation so pass 0 won't hit the
    // "if (pass == num_passes - 1)" branch. Maybe it's better to reload
    // counter->previous_len and use it rather than current_len in last_filter()
    current_len = len;
  } else {
    current_k = counter->k;
    current_len = counter->len;
    previous_len = counter->previous_len;
  }
  if (current_len == 0) {
    return;
  }

  // When k=len, early_stop will be true at pass 0. It means
  // filter_and_histogram() should handle correctly the case that pass=0 and
  // early_stop=true. However, this special case of k=len is handled in other
  // way in select_k() so such case is not possible here.
  const bool early_stop = (current_len == current_k);
  const IdxT buf_len = calc_buf_len<T>(len);
  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
  // "previous_len > buf_len" means previous pass skips writing buffer
  if (pass == 0 || pass == 1 || previous_len > buf_len) {
    in_buf = in;
    in_idx_buf = in_idx ? in_idx : nullptr;
    previous_len = len;
  }
  // "current_len > buf_len" means current pass will skip writing buffer
  if (pass == 0 || current_len > buf_len) {
    out_buf = nullptr;
    out_idx_buf = nullptr;
  }

  filter_and_histogram<T, IdxT, BitsPerPass, store_out>(in_buf, in_idx_buf, out_buf, out_idx_buf,
                                                        out, out_idx, previous_len, counter,
                                                        histogram, select_min, pass, early_stop);
  __threadfence();

  bool isLastBlock = false;
  if (threadIdx.x == 0) {
    unsigned int finished = atomicInc(&counter->finished_block_cnt, gridDim.x - 1);
    isLastBlock = (finished == (gridDim.x - 1));
  }

  if (__syncthreads_or(isLastBlock)) {
    if (early_stop) {
      if (threadIdx.x == 0) {
        // `last_filter_kernel()` requires setting previous_len
        counter->previous_len = 0;
        counter->len = 0;
      }
      return;
    }

    constexpr int num_passes = calc_num_passes<T, BitsPerPass>();

    scan<IdxT, BitsPerPass, BlockSize>(histogram);
    __syncthreads();
    choose_bucket<T, IdxT, BitsPerPass>(counter, histogram, current_k, pass);
    __syncthreads();

    // reset for next pass
    if (pass != num_passes - 1) {
      for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
        histogram[i] = 0;
      }
    }
    if (threadIdx.x == 0) {
      // `last_filter_kernel()` requires setting previous_len even in the last
      // pass
      counter->previous_len = current_len;
      // not necessary for the last pass, but put it here anyway
      counter->filter_cnt = 0;
    }

    if constexpr (fused_last_filter) {
      if (pass == num_passes - 1) {
        last_filter<T, IdxT, BitsPerPass, store_out>(
            out_buf ? out_buf : in_buf, out_idx_buf ? out_idx_buf : in_idx_buf, out, out_idx,
            out_buf ? current_len : len, k, counter, select_min, pass);
      }
    }
  }
}

template <typename T, typename IdxT, int BitsPerPass, int BlockSize, bool fused_last_filter,
          bool store_out>
__global__ void radix_kernel(const T *in, const IdxT *in_idx, const T *in_buf,
                             const IdxT *in_idx_buf, T *out_buf, IdxT *out_idx_buf, T *out,
                             IdxT *out_idx, Counter<T, IdxT> *counters, IdxT *histograms,
                             const IdxT len, const IdxT k, const bool select_min, const int pass,
                             IdxT *lengths) {
  const size_t batch_id = blockIdx.y;
  auto counter = counters + batch_id;
  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
  auto histogram = histograms + batch_id * num_buckets;

  in += batch_id * len;
  if (in_idx) {
    in_idx += batch_id * len;
  }
  if constexpr (store_out) {
    out += batch_id * k;
  }
  out_idx += batch_id * k;

  const IdxT buf_len = calc_buf_len<T>(len);
  in_buf += batch_id * buf_len;
  in_idx_buf += batch_id * buf_len;

  out_buf += batch_id * buf_len;
  out_idx_buf += batch_id * buf_len;

  IdxT actual_len = len;
  if (lengths != nullptr) {
    actual_len = lengths[batch_id];
  }
  radix_kernel_func<T, IdxT, BitsPerPass, BlockSize, fused_last_filter, store_out>(
      in, in_idx, in_buf, in_idx_buf, out_buf, out_idx_buf, out, out_idx, counter, histogram,
      actual_len, k, select_min, pass);
}

template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
unsigned calc_grid_dim(int batch_size, IdxT len, int sm_cnt) {
  static_assert(VECTORIZED_READ_SIZE / sizeof(T) >= 1);

  int active_blocks;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, radix_kernel<T, IdxT, BitsPerPass, BlockSize, false, false>, BlockSize, 0);
  active_blocks *= sm_cnt;

  IdxT best_num_blocks = 0;
  float best_tail_wave_penalty = 1.0f;
  const IdxT max_num_blocks = ceildiv<IdxT>(len, VECTORIZED_READ_SIZE / sizeof(T) * BlockSize);
  for (int num_waves = 1;; ++num_waves) {
    IdxT num_blocks = std::min(
        max_num_blocks, static_cast<IdxT>(std::max(num_waves * active_blocks / batch_size, 1)));
    IdxT items_per_thread = ceildiv<IdxT>(len, num_blocks * BlockSize);
    items_per_thread = alignTo<IdxT>(items_per_thread, VECTORIZED_READ_SIZE / sizeof(T));
    num_blocks = ceildiv<IdxT>(len, items_per_thread * BlockSize);
    float actual_num_waves = static_cast<float>(num_blocks) * batch_size / active_blocks;
    float tail_wave_penalty =
        (ceilf(actual_num_waves) - actual_num_waves) / ceilf(actual_num_waves);

    // 0.15 is determined experimentally. It also ensures breaking the loop
    // early, e.g. when num_waves > 7, tail_wave_penalty will always <0.15
    if (tail_wave_penalty < 0.15) {
      best_num_blocks = num_blocks;
      break;
    } else if (tail_wave_penalty < best_tail_wave_penalty) {
      best_num_blocks = num_blocks;
      best_tail_wave_penalty = tail_wave_penalty;
    }

    if (num_blocks == max_num_blocks) {
      break;
    }
  }
  return best_num_blocks;
}

template <typename T, typename IdxT>
__host__ __device__ void set_buf_pointers(const T *in, const IdxT *in_idx, T *buf1, IdxT *idx_buf1,
                                          T *buf2, IdxT *idx_buf2, int pass, const T *&in_buf,
                                          const IdxT *&in_idx_buf, T *&out_buf,
                                          IdxT *&out_idx_buf) {
  if (pass == 0) {
    in_buf = in;
    in_idx_buf = nullptr;
    out_buf = nullptr;
    out_idx_buf = nullptr;
  } else if (pass == 1) {
    in_buf = in;
    in_idx_buf = in_idx;
    out_buf = buf1;
    out_idx_buf = idx_buf1;
  } else if (pass % 2 == 0) {
    in_buf = buf1;
    in_idx_buf = idx_buf1;
    out_buf = buf2;
    out_idx_buf = idx_buf2;
  } else {
    in_buf = buf2;
    in_idx_buf = idx_buf2;
    out_buf = buf1;
    out_idx_buf = idx_buf1;
  }
}

// The following a few functions are for the one-block version, which uses
// single thread block for each row of a batch.
template <typename T, typename IdxT, int BitsPerPass, bool store_out, bool is_vectorized>
__device__ void filter_and_histogram_for_one_block(const T *in_buf, const IdxT *in_idx_buf,
                                                   T *out_buf, IdxT *out_idx_buf, T *out,
                                                   IdxT *out_idx, Counter<T, IdxT> *counter,
                                                   IdxT *histogram, bool select_min, int pass) {
  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
  for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
    histogram[i] = 0;
  }
  IdxT *p_filter_cnt = &counter->filter_cnt;
  if (threadIdx.x == 0) {
    *p_filter_cnt = 0;
  }
  __syncthreads();

  const int start_bit = calc_start_bit<T, BitsPerPass>(pass);
  const unsigned mask = calc_mask<T, BitsPerPass>(pass);
  const IdxT previous_len = counter->previous_len;

  if (pass == 0) {
    if constexpr (is_vectorized) {
      auto f = [histogram, select_min, start_bit, mask](T value, IdxT) {
        int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
        atomicAdd(histogram + bucket, static_cast<IdxT>(1));
      };
      vectorized_process(threadIdx.x, blockDim.x, in_buf, previous_len, f);
    } else {
      for (IdxT i = threadIdx.x; i < previous_len; i += blockDim.x) {
        const T value = in_buf[i];
        int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
        atomicAdd(histogram + bucket, static_cast<IdxT>(1));
      }
    }
  } else {
    // not use vectorized_process here because it increases #registers a lot
    IdxT *p_out_cnt = &counter->out_cnt;
    const auto kth_value_bits = counter->kth_value_bits;
    const int previous_start_bit = calc_start_bit<T, BitsPerPass>(pass - 1);

    for (IdxT i = threadIdx.x; i < previous_len; i += blockDim.x) {
      const T value = in_buf[i];
      const auto previous_bits = (twiddle_in(value, select_min) >> previous_start_bit)
                                 << previous_start_bit;
      if (previous_bits == kth_value_bits) {
#if CUDART_VERSION < 12000
        // Avoiding potential compiler bug in CUDA 11
        volatile
#endif
            IdxT pos = atomicAdd(p_filter_cnt, static_cast<IdxT>(1));
        out_buf[pos] = value;
        out_idx_buf[pos] = in_idx_buf ? in_idx_buf[i] : i;

        int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
        atomicAdd(histogram + bucket, static_cast<IdxT>(1));
      } else if (previous_bits < kth_value_bits) {
        IdxT pos = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
        if constexpr (store_out) {
          out[pos] = value;
        }
        out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
      }
    }
  }
}

template <typename T, typename IdxT, int BitsPerPass, int BlockSize, bool store_out,
          bool is_vectorized>
__device__ void radix_topk_one_block_func(const T *in, const IdxT *in_idx, const IdxT len,
                                          const IdxT k, T *out, IdxT *out_idx,
                                          const bool select_min, T *buf1, IdxT *idx_buf1, T *buf2,
                                          IdxT *idx_buf2) {
  if (len <= k) {
    for (int index = threadIdx.x; index < len; index += BlockSize) {
      if constexpr (store_out) {
        out[index] = in[index];
      }
      out_idx[index] = in_idx ? in_idx[index] : index;
    }
    for (int index = threadIdx.x + len; index < k; index += BlockSize) {
      if constexpr (store_out) {
        out[index] = static_cast<T>(-1.0f);
      }
      out_idx[index] = -1;
    }
    return;
  }

  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
  __shared__ Counter<T, IdxT> counter;
  __shared__ IdxT histogram[num_buckets];

  if (threadIdx.x == 0) {
    counter.k = k;
    counter.len = len;
    counter.previous_len = len;
    counter.kth_value_bits = 0;
    counter.out_cnt = 0;
    counter.out_back_cnt = 0;
  }
  __syncthreads();

  // const size_t batch_id = blockIdx.x; // size_t to avoid multiplication
  // overflow
  const T *in_buf = nullptr;
  const IdxT *in_idx_buf = nullptr;
  T *out_buf = nullptr;
  IdxT *out_idx_buf = nullptr;

  constexpr int num_passes = calc_num_passes<T, BitsPerPass>();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);
  for (int pass = 0; pass < num_passes; ++pass) {
    set_buf_pointers(in, in_idx, buf1, idx_buf1, buf2, idx_buf2, pass, in_buf, in_idx_buf, out_buf,
                     out_idx_buf);

    IdxT current_len = counter.len;
    IdxT current_k = counter.k;

    filter_and_histogram_for_one_block<T, IdxT, BitsPerPass, store_out, is_vectorized>(
        in_buf, in_idx_buf, out_buf, out_idx_buf, out, out_idx, &counter, histogram, select_min,
        pass);
    __syncthreads();

    scan<IdxT, BitsPerPass, BlockSize>(histogram);
    __syncthreads();

    choose_bucket<T, IdxT, BitsPerPass>(&counter, histogram, current_k, pass);
    // scan_warp_version<T, IdxT, BitsPerPass, BlockSize>(
    //     warp, histogram, &counter, current_k, pass);
    if (threadIdx.x == 0) {
      counter.previous_len = current_len;
    }
    __syncthreads();

    if (counter.len == counter.k || pass == num_passes - 1) {
      last_filter<T, IdxT, BitsPerPass, store_out>(pass == 0 ? in : out_buf,
                                                   pass == 0 ? in_idx : out_idx_buf, out, out_idx,
                                                   current_len, k, &counter, select_min, pass);
      break;
    }
  }  // end for pass
}  // end kernel

template <typename T, typename IdxT, int BitsPerPass, int BlockSize, bool store_out,
          bool is_vectorized>
__global__ void radix_topk_one_block_kernel(const T *in, const IdxT *in_idx, const IdxT len,
                                            const IdxT k, T *out, IdxT *out_idx,
                                            const bool select_min, T *buf1, IdxT *idx_buf1, T *buf2,
                                            IdxT *idx_buf2, IdxT *lengths) {
  const size_t batch_id = blockIdx.x;  // size_t to avoid multiplication overflow
  IdxT actual_len = len;
  if (lengths) {
    actual_len = lengths[batch_id];
  }

  in += batch_id * len;
  if (in_idx) {
    in_idx += batch_id * len;
  }

  out += batch_id * k;
  out_idx += batch_id * k;
  buf1 += batch_id * len;
  idx_buf1 += batch_id * len;
  buf2 += batch_id * len;
  idx_buf2 += batch_id * len;

  radix_topk_one_block_func<T, IdxT, BitsPerPass, BlockSize, store_out, is_vectorized>(
      in, in_idx, actual_len, k, out, out_idx, select_min, buf1, idx_buf1, buf2, idx_buf2);
}  // end kernel

}  // namespace topk

/***************Runtime API****************/

template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
void standalone_radix_topk_(void *buf, size_t &buf_size, const T *in, const IdxT *in_idx,
                            int batch_size, IdxT len, IdxT k, T *out, IdxT *out_idx,
                            bool select_min, bool fused_last_filter, unsigned grid_dim,
                            cudaStream_t stream, IdxT *lengths = nullptr) {
  static_assert(topk::calc_num_passes<T, BitsPerPass>() > 1);
  constexpr int num_buckets = topk::calc_num_buckets<BitsPerPass>();

  topk::Counter<T, IdxT> *counters = nullptr;
  IdxT *histograms = nullptr;
  T *buf1 = nullptr;
  IdxT *idx_buf1 = nullptr;
  T *buf2 = nullptr;
  IdxT *idx_buf2 = nullptr;
  {
    IdxT len_candidates = topk::calc_buf_len<T>(len);
    std::vector<size_t> sizes = {sizeof(*counters) * batch_size,
                                 sizeof(*histograms) * num_buckets * batch_size,
                                 sizeof(*buf1) * len_candidates * batch_size,
                                 sizeof(*idx_buf1) * len_candidates * batch_size,
                                 sizeof(*buf2) * len_candidates * batch_size,
                                 sizeof(*idx_buf2) * len_candidates * batch_size};
    size_t total_size = calc_aligned_size(sizes);
    if (!buf) {
      buf_size = total_size;
      return;
    }

    std::vector<void *> aligned_pointers = calc_aligned_pointers(buf, sizes);
    counters = static_cast<decltype(counters)>(aligned_pointers[0]);
    histograms = static_cast<decltype(histograms)>(aligned_pointers[1]);
    buf1 = static_cast<decltype(buf1)>(aligned_pointers[2]);
    idx_buf1 = static_cast<decltype(idx_buf1)>(aligned_pointers[3]);
    buf2 = static_cast<decltype(buf2)>(aligned_pointers[4]);
    idx_buf2 = static_cast<decltype(idx_buf2)>(aligned_pointers[5]);

    cudaMemsetAsync(
        buf, 0, static_cast<char *>(aligned_pointers[2]) - static_cast<char *>(aligned_pointers[0]),
        stream);
  }

  const T *in_buf = nullptr;
  const IdxT *in_idx_buf = nullptr;
  T *out_buf = nullptr;
  IdxT *out_idx_buf = nullptr;

  dim3 blocks(grid_dim, batch_size);

  constexpr int num_passes = topk::calc_num_passes<T, BitsPerPass>();

  auto kernel = topk::radix_kernel<T, IdxT, BitsPerPass, BlockSize, false, true>;

  for (int pass = 0; pass < num_passes; ++pass) {
    topk::set_buf_pointers(in, in_idx, buf1, idx_buf1, buf2, idx_buf2, pass, in_buf, in_idx_buf,
                           out_buf, out_idx_buf);

    if (fused_last_filter && pass == num_passes - 1 && out != nullptr) {
      kernel = topk::radix_kernel<T, IdxT, BitsPerPass, BlockSize, true, true>;
    } else if (fused_last_filter && pass == num_passes - 1 && out == nullptr) {
      kernel = topk::radix_kernel<T, IdxT, BitsPerPass, BlockSize, true, false>;
    } else if (out == nullptr) {
      kernel = topk::radix_kernel<T, IdxT, BitsPerPass, BlockSize, false, false>;
    }

    kernel<<<blocks, BlockSize, 0, stream>>>(in, in_idx, in_buf, in_idx_buf, out_buf, out_idx_buf,
                                             out, out_idx, counters, histograms, len, k, select_min,
                                             pass, lengths);
  }

  if (!fused_last_filter) {
    if (out != nullptr) {
      topk::last_filter_kernel<T, IdxT, BitsPerPass, true><<<blocks, BlockSize, 0, stream>>>(
          in, in_idx, out_buf, out_idx_buf, out, out_idx, len, k, counters, select_min);
    } else {
      topk::last_filter_kernel<T, IdxT, BitsPerPass, false><<<blocks, BlockSize, 0, stream>>>(
          in, in_idx, out_buf, out_idx_buf, out, out_idx, len, k, counters, select_min);
    }
  }
}

template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
void standalone_radix_topk_one_block_(void *buf, size_t &buf_size, const T *in, const IdxT *in_idx,
                                      int batch_size, IdxT len, IdxT k, T *out, IdxT *out_idx,
                                      bool select_min, cudaStream_t stream,
                                      IdxT *lengths = nullptr) {
  static_assert(topk::calc_num_passes<T, BitsPerPass>() > 1);

  T *buf1 = nullptr;
  IdxT *idx_buf1 = nullptr;
  T *buf2 = nullptr;
  IdxT *idx_buf2 = nullptr;
  {
    std::vector<size_t> sizes = {
        sizeof(*buf1) * len * batch_size, sizeof(*idx_buf1) * len * batch_size,
        sizeof(*buf2) * len * batch_size, sizeof(*idx_buf2) * len * batch_size};
    size_t total_size = calc_aligned_size(sizes);
    if (!buf) {
      buf_size = total_size;
      return;
    }

    std::vector<void *> aligned_pointers = calc_aligned_pointers(buf, sizes);
    buf1 = static_cast<decltype(buf1)>(aligned_pointers[0]);
    idx_buf1 = static_cast<decltype(idx_buf1)>(aligned_pointers[1]);
    buf2 = static_cast<decltype(buf2)>(aligned_pointers[2]);
    idx_buf2 = static_cast<decltype(idx_buf2)>(aligned_pointers[3]);
  }

  if (out != nullptr) {
    topk::radix_topk_one_block_kernel<T, IdxT, BitsPerPass, BlockSize, true, true>
        <<<batch_size, BlockSize, 0, stream>>>(in, in_idx, len, k, out, out_idx, select_min, buf1,
                                               idx_buf1, buf2, idx_buf2, lengths);
  } else {
    topk::radix_topk_one_block_kernel<T, IdxT, BitsPerPass, BlockSize, false, true>
        <<<batch_size, BlockSize, 0, stream>>>(in, in_idx, len, k, out, out_idx, select_min, buf1,
                                               idx_buf1, buf2, idx_buf2, lengths);
  }
}

template <typename T, typename idxT>
void standalone_topk(void *buf, size_t &buf_size, const T *in, int batch_size, idxT len, idxT k,
                     T *out, idxT *out_idx, bool greater, cudaStream_t stream = 0,
                     idxT *lengths = nullptr, bool is_prefill = false) {
  constexpr int items_per_thread = 32;
  constexpr int multi_block_dim = 256;
  constexpr int single_block_dim = 1024;
  constexpr bool fused_last_filter = false;
  if (len <= single_block_dim * items_per_thread || is_prefill) {
    standalone_radix_topk_one_block_<T, idxT, 11, single_block_dim>(
        buf, buf_size, in, static_cast<idxT *>(nullptr), batch_size, len, k, out, out_idx, !greater,
        stream, lengths);
  } else {
    // Cache sm_cnt per device to avoid repeated host-side queries.
    static int cached_dev = -1;
    static int cached_sm_cnt = -1;
    int sm_cnt;
    {
      int dev;
      NVTE_CHECK_CUDA(cudaGetDevice(&dev));
      if (dev != cached_dev) {
        NVTE_CHECK_CUDA(
            cudaDeviceGetAttribute(&cached_sm_cnt, cudaDevAttrMultiProcessorCount, dev));
        cached_dev = dev;
      }
      sm_cnt = cached_sm_cnt;
    }
    unsigned grid_dim = topk::calc_grid_dim<T, idxT, 11, multi_block_dim>(batch_size, len, sm_cnt);

    if (grid_dim == 1) {
      standalone_radix_topk_one_block_<T, idxT, 11, single_block_dim>(
          buf, buf_size, in, static_cast<idxT *>(nullptr), batch_size, len, k, out, out_idx,
          !greater, stream, lengths);
    } else {
      standalone_radix_topk_<T, idxT, 11, multi_block_dim>(
          buf, buf_size, in, static_cast<idxT *>(nullptr), batch_size, len, k, out, out_idx,
          !greater, fused_last_filter, grid_dim, stream, lengths);
    }
  }
}
}  // namespace nv
