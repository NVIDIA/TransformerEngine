#ifndef NV_UTIL_H_
#define NV_UTIL_H_
#include <cassert>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "cuda_runtime_api.h"

#define TOPK_CUDA_CHECK(val)                                                   \
  { nv::cuda_check_((val), __FILE__, __LINE__); }
#define TOPK_CUDA_CHECK_LAST_ERROR()                                           \
  { nv::cuda_check_last_error_(__FILE__, __LINE__); }

namespace nv {

constexpr unsigned FULL_WARP_MASK = 0xffffffff;
constexpr int WARP_SIZE = 32;

#ifdef __CUDA_ARCH__
using ::atomicAdd;
inline __device__ size_t atomicAdd(size_t *address, size_t value) {
  static_assert(sizeof(size_t) == sizeof(unsigned long long int));
  return atomicAdd((unsigned long long int *)address,
                   (unsigned long long int)value);
}
#endif

class CudaException : public std::runtime_error {
public:
  explicit CudaException(const std::string &what) : runtime_error(what) {}
};

inline void cuda_check_(cudaError_t val, const char *file, int line) {
  if (val != cudaSuccess) {
    throw CudaException(std::string(file) + ":" + std::to_string(line) +
                        ": CUDA error " + std::to_string(val) + ": " +
                        cudaGetErrorString(val));
  }
}

inline void cuda_check_last_error_(const char *file, int line) {
  cudaDeviceSynchronize();
  cudaError_t err = cudaPeekAtLastError();
  cuda_check_(err, file, line);
}

class CudaTimer {
public:
  CudaTimer() {
    TOPK_CUDA_CHECK(cudaEventCreate(&start_));
    TOPK_CUDA_CHECK(cudaEventCreate(&stop_));
    reset();
  }
  ~CudaTimer() {
    TOPK_CUDA_CHECK(cudaEventDestroy(start_));
    TOPK_CUDA_CHECK(cudaEventDestroy(stop_));
  }
  void reset() { TOPK_CUDA_CHECK(cudaEventRecord(start_)); }
  float elapsed_ms() {
    float elapsed;
    TOPK_CUDA_CHECK(cudaEventRecord(stop_));
    TOPK_CUDA_CHECK(cudaEventSynchronize(stop_));
    TOPK_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start_, stop_));
    return elapsed;
  }

private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
};

class Timer {
public:
  Timer() { reset(); }
  void reset() { start_time_ = std::chrono::steady_clock::now(); }
  float elapsed_ms() {
    auto end_time = std::chrono::steady_clock::now();
    auto dur =
        std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(
            end_time - start_time_);
    return dur.count();
  }

private:
  std::chrono::steady_clock::time_point start_time_;
};

inline size_t calc_aligned_size(const std::vector<size_t> &sizes) {
  const size_t ALIGN_BYTES = 256;
  const size_t ALIGN_MASK = ~(ALIGN_BYTES - 1);
  size_t total = 0;
  for (auto sz : sizes) {
    total += (sz + ALIGN_BYTES - 1) & ALIGN_MASK;
  }
  return total + ALIGN_BYTES - 1;
}

inline std::vector<void *>
calc_aligned_pointers(const void *p, const std::vector<size_t> &sizes) {
  const size_t ALIGN_BYTES = 256;
  const size_t ALIGN_MASK = ~(ALIGN_BYTES - 1);

  char *ptr = reinterpret_cast<char *>(
      (reinterpret_cast<size_t>(p) + ALIGN_BYTES - 1) & ALIGN_MASK);

  std::vector<void *> aligned_pointers;
  aligned_pointers.reserve(sizes.size());
  for (auto sz : sizes) {
    aligned_pointers.push_back(ptr);
    ptr += (sz + ALIGN_BYTES - 1) & ALIGN_MASK;
  }

  return aligned_pointers;
}

/*******************************************************/
/*                   Debug Function                    */
/*******************************************************/
template <typename T>
void printDevPtr(const T *d_cache, int len, char *name, bool print) {
  T *res = (T *)malloc(sizeof(T) * len);
  cudaMemcpy(res, d_cache, sizeof(T) * len, cudaMemcpyDeviceToHost);

  printf("%s ", name);
  int j = 0;
  int tmp = 0;
  for (int i = 0; i < len; i++) {
    tmp = tmp + res[i];
    if (print) {
      printf("%d(%e) ", i, (float)res[i]);
      if (j % 32 == 31) {
        printf("\n");
      }
    }
    j = j + 1;
  }
  printf("Get output number =%d sum=%d \n", j, tmp);
  free(res);
}

namespace math {

template <int size, typename T>
__host__ __device__ constexpr T round_up_to_multiple_of(T len) {
  if (len == 0) {
    return 0;
  }
  return ((len - 1) / size + 1) * size;
}

// from faiss/faiss/gpu/utils/StaticUtils.h
template <typename T> constexpr __host__ __device__ int log2_(T n, int p = 0) {
  return (n <= 1) ? p : log2_(n / 2, p + 1);
}

template <typename T> constexpr __host__ __device__ bool isPowerOf2(T v) {
  return (v && !(v & (v - 1)));
}

template <typename T> constexpr __host__ __device__ T nextHighestPowerOf2(T v) {
  // return (isPowerOf2(v) ? 2 * v : ((T)1 << (log2_(v) + 1)));
  return (isPowerOf2(v) ? v : ((T)1 << (log2_(v) + 1)));
}

} // namespace math

namespace numeric {

// a new type should specialize get_lower_bound() & get_upper_bound()
// rather than get_dummy()
template <typename T> constexpr T get_lower_bound() {
  static_assert(std::is_arithmetic<T>::value);
  if (std::numeric_limits<T>::has_infinity &&
      std::numeric_limits<T>::is_signed) {
    return -std::numeric_limits<T>::infinity();
  } else {
    return std::numeric_limits<T>::lowest();
  }
}

template <typename T> constexpr T get_upper_bound() {
  static_assert(std::is_arithmetic<T>::value);
  if (std::numeric_limits<T>::has_infinity) {
    return std::numeric_limits<T>::infinity();
  } else {
    return std::numeric_limits<T>::max();
  }
}

template <typename T> constexpr T get_dummy(bool greater) {
  // TODO: for unsigned and greater=true, dummy will be 0
  //       find better way to warn about this
  assert(!(std::is_unsigned<T>::value && greater));
  return greater ? get_lower_bound<T>() : get_upper_bound<T>();
}

template <bool greater, typename T>
__device__ bool is_better_than(T val, T baseline) {
  return (val > baseline && greater) || (val < baseline && !greater);
}

} // namespace numeric

} // namespace nv
#endif
