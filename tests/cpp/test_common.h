/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once

#include <array>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include <cudaTypedefs.h>
#define FP4_TYPE_SUPPORTED (CUDA_VERSION >= 12080)

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#if FP4_TYPE_SUPPORTED
#include <cuda_fp4.h>
#endif
#include <cuda_runtime_api.h>

#include <transformer_engine/transformer_engine.h>
#include "util/logging.h"

namespace test {
using namespace transformer_engine;

size_t typeToNumBits(DType type);
size_t product(const NVTEShape &shape);
size_t product(const std::vector<size_t> &shape);
size_t bytes(const NVTEShape& shape, const DType type);

template <size_t i>
struct BytesToType {};

template <>
struct BytesToType<1> {
  using Type = uint8_t;
};

template <>
struct BytesToType<2> {
  using Type = uint16_t;
};

template <>
struct BytesToType<4> {
  using Type = uint32_t;
};

template <>
struct BytesToType<8> {
  using Type = uint64_t;
};

using byte = uint8_t;
using int16 = int16_t;
using int32 = int32_t;
using int64 = int64_t;
using fp32 = float;
using fp16 = half;
using bf16 = nv_bfloat16;
using fp8e4m3 = __nv_fp8_e4m3;
using fp8e5m2 = __nv_fp8_e5m2;
using fp8e8m0 = uint8_t;
#if FP4_TYPE_SUPPORTED
using fp4e2m1 = __nv_fp4_e2m1;
using fp4e2m1x2 = __nv_fp4x2_e2m1;
using fp4e2m1x4 = __nv_fp4x4_e2m1;
#endif

template <typename T>
struct BitsNumber;

#if FP4_TYPE_SUPPORTED
template <>
struct BitsNumber<fp4e2m1> {
  static constexpr size_t num_bits = 4;
};
#endif

template <typename T>
struct BitsNumber {
  static constexpr size_t num_bits = 8 * sizeof(T);
};

template <typename T>
struct TypeInfo {
#if FP4_TYPE_SUPPORTED
    using types = std::tuple<byte, int16, int32, int64, fp32, fp16, bf16, fp8e4m3, fp8e5m2, fp8e8m0, fp4e2m1>;
#else
    using types = std::tuple<byte, int16, int32, int64, fp32, fp16, bf16, fp8e4m3, fp8e5m2, fp8e8m0>;
#endif

    template <typename U, DType current>
    struct Helper {
        constexpr static DType getType() {
            constexpr int i = static_cast<int>(current);
            if (std::is_same<U, typename std::tuple_element<i, types>::type>::value) {
                return current;
            } else {
                return Helper<U, static_cast<DType>(i + 1)>::getType();
            }
        }
    };

    template <typename U>
    struct Helper<U, DType::kNumTypes> {
        constexpr static DType getType() {
            return DType::kNumTypes;
        }
    };

    template <typename U>
    constexpr static DType getType() {
        return Helper<U, DType::kByte>::getType();
    }

    constexpr static DType dtype = getType<T>();
    constexpr static size_t size = BitsNumber<T>::num_bits;
};

// Deleter for CUDA buffer RAII class
struct CudaDeleter {
  void operator()(void* ptr) const { if (ptr != nullptr) cudaFree(ptr); }
};

// CUDA buffer RAII class
template <typename T = void>
using CudaPtr = std::unique_ptr<T, CudaDeleter>;

// Construct CUDA memory
template <typename T = void>
CudaPtr<T> cuda_alloc(size_t bytes) {
  void* ptr = nullptr;
  NVTE_CHECK_CUDA(cudaMalloc(&ptr, bytes));
  return CudaPtr<T>(static_cast<T*>(ptr));
}

/* Wrapper for Transformer Engine tensor
 *
 * Maintains matching GPU and CPU buffers.
 */
class Tensor {
 public:
  Tensor(const std::string& name,
         const NVTEShape &shape, const DType type,
         const bool rowwise = true,
         const bool columnwise = false,
         const NVTEScalingMode &mode = NVTE_DELAYED_TENSOR_SCALING);

  Tensor(const std::string& name,
         const std::vector<size_t> &shape,
         const DType type,
         const bool rowwise = true,
         const bool columnwise = false,
         const NVTEScalingMode &mode = NVTE_DELAYED_TENSOR_SCALING) :
    Tensor(name, nvte_make_shape(shape.data(), shape.size()), type, rowwise, columnwise, mode) {}

  Tensor() = default;

  Tensor& operator=(const Tensor &other) = delete;
  Tensor(const Tensor &other) = delete;

  Tensor(Tensor &&other) = default;
  Tensor& operator=(Tensor &&other) = default;

  ~Tensor() = default;

  NVTETensor data() const noexcept { return tensor_.data(); }

  NVTEShape rowwise_shape() const noexcept { return tensor_.get_rowwise_data().shape; }

  NVTEShape columnwise_shape() const noexcept { return tensor_.get_columnwise_data().shape; }

  NVTEShape rowwise_scale_inv_shape() const {
    NVTE_CHECK(rowwise_, "Tensor does not have rowwise data!");
    return tensor_.get_rowwise_scale_inv().shape;
  }

  NVTEShape columnwise_scale_inv_shape() const {
    NVTE_CHECK(columnwise_, "Tensor does not have columnwise data!");
    return tensor_.get_columnwise_scale_inv().shape;
  }

  NVTEScalingMode scaling_mode() const noexcept {
    return tensor_.scaling_mode();
  }

  DType dtype() const noexcept {
    return tensor_.dtype();
  }

  void *rowwise_dptr() const {
    NVTE_CHECK(rowwise_, "Tensor does not have rowwise data!");
    return tensor_.get_rowwise_data().data_ptr;
  }

  void *columnwise_dptr() const {
    NVTE_CHECK(columnwise_, "Tensor does not have columnwise data!");
    return tensor_.get_columnwise_data().data_ptr;
  }

  template <typename T>
  T *rowwise_cpu_dptr() {
    NVTE_CHECK(data_rowwise_, "Tensor does not have rowwise data!");
    NVTE_CHECK(TypeInfo<T>::dtype == data_rowwise_->dtype(), "Invalid type!");
    NVTE_CHECK(rowwise_, "Tensor does not have rowwise data!");
    return data_rowwise_->cpu_buffer<T>();
  }

  template <typename T>
  T *columnwise_cpu_dptr() {
    NVTE_CHECK(data_columnwise_, "Tensor does not have columnwise data!");
    NVTE_CHECK(TypeInfo<T>::dtype == data_columnwise_->dtype(), "Invalid type!");
    NVTE_CHECK(columnwise_, "Tensor does not have columnwise data!");
    return data_columnwise_->cpu_buffer<T>();
  }

  float amax() {
    NVTE_CHECK(amax_rowwise_);
    NVTE_CHECK(amax_rowwise_->size() == 1);
    NVTE_CHECK(amax_rowwise_->dtype() == DType::kFloat32);
    amax_rowwise_->to_cpu();
    return *amax_rowwise_->cpu_buffer<float>();
  }

  float amax_columnwise() {
    NVTE_CHECK(amax_columnwise_);
    NVTE_CHECK(amax_columnwise_->size() == 1);
    NVTE_CHECK(amax_columnwise_->dtype() == DType::kFloat32);
    amax_columnwise_->to_cpu();
    return *amax_columnwise_->cpu_buffer<float>();
  }

  float scale() {
    NVTE_CHECK(scale_);
    NVTE_CHECK(scale_->size() == 1);
    NVTE_CHECK(scale_->dtype() == DType::kFloat32);
    scale_->to_cpu();
    return *scale_->cpu_buffer<float>();
  }

  float rowwise_scale_inv(){
    NVTE_CHECK(scale_inv_rowwise_);
    NVTE_CHECK(scale_inv_rowwise_->size() == 1);
    NVTE_CHECK(scale_inv_rowwise_->dtype() == DType::kFloat32);
    scale_inv_rowwise_->to_cpu();
    return *scale_inv_rowwise_->cpu_buffer<float>();
  }

  template <typename T>
  T *rowwise_cpu_scale_inv_ptr(){
    NVTE_CHECK(scale_inv_rowwise_);
    scale_inv_rowwise_->to_cpu();
    return scale_inv_rowwise_->cpu_buffer<T>();
  }

  template <typename T>
  T *columnwise_cpu_scale_inv_ptr(){
    NVTE_CHECK(scale_inv_columnwise_);
    scale_inv_columnwise_->to_cpu();
    return scale_inv_columnwise_->cpu_buffer<T>();
  }

  template <typename T>
  T *cpu_rowwise_amax_ptr() {
    NVTE_CHECK(amax_rowwise_);
    amax_rowwise_->to_cpu();
    return amax_rowwise_->cpu_buffer<T>();
  }

  template <typename T>
  T *cpu_columnwise_amax_ptr() {
    NVTE_CHECK(amax_columnwise_);
    amax_columnwise_->to_cpu();
    return amax_columnwise_->cpu_buffer<T>();
  }

  size_t rowwise_amax_size() const noexcept {
    return amax_rowwise_ ? amax_rowwise_->size() : 0;
  }

  bool rowwise() const {
    return rowwise_;
  }

  bool columnwise() const {
    return columnwise_;
  }

  void set_tensor_amax_nullptr();

  void set_with_gemm_swizzled_scales(bool with_gemm_swizzled_scales);
  void set_row_scaled_nvfp4(bool row_scaled_nvfp4);

  void to_cpu();
  void from_cpu();

  void set_amax(float amax);
  void set_scale(float scale);
  void set_scale_inv(float scale_inv);
  void set_tensor_amax_columnwise(float amax);

  void fill_uniform_rowwise_scale_inv();
  void fill_uniform_columnwise_scale_inv();
  void fill_uniform_scale();

  std::mt19937& gen() { return gen_; }

 private:

  /* Manages matching GPU and CPU buffers. */
  class Buffer {
  public:

    Buffer(size_t size = 0, DType dtype = DType::kByte);
    ~Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&&) = default;
    Buffer& operator=(Buffer&&) = default;

    size_t size() const noexcept { return size_; }
    DType dtype() const noexcept { return dtype_; }

    // Void pointer accessors
    void *cpu_buffer() { return cpu_buffer_.get(); }
    const void *cpu_buffer() const { return cpu_buffer_.get(); }
    void *gpu_buffer() { return gpu_buffer_.get(); }
    const void *gpu_buffer() const { return gpu_buffer_.get(); }

    // Templated pointer accessors
    template <typename T>
    T *cpu_buffer() {
      return reinterpret_cast<T *>(cpu_buffer());
    }
    template <typename T>
    const T *cpu_buffer() const {
      return const_cast<Buffer *>(this)->cpu_buffer<T>();
    }
    template <typename T>
    T *gpu_buffer() {
      return reinterpret_cast<T *>(gpu_buffer());
    }
    template <typename T>
    const T *gpu_buffer() const {
      return const_cast<Buffer *>(this)->gpu_buffer<T>();
    }

    // Memory transfers between CPU and GPU
    void to_cpu();
    void from_cpu();

  private:
    std::unique_ptr<unsigned char[]> cpu_buffer_;
    CudaPtr<unsigned char[]> gpu_buffer_;
    size_t size_;
    DType dtype_;
    size_t bytes_;
  };

  // Transformer Engine tensor
  TensorWrapper tensor_;

  // Data buffers
  std::optional<Buffer> data_rowwise_;
  std::optional<Buffer> data_columnwise_;
  std::shared_ptr<Buffer> scale_inv_rowwise_;
  std::shared_ptr<Buffer> scale_inv_columnwise_;
  std::optional<Buffer> amax_rowwise_;
  std::optional<Buffer> amax_columnwise_;
  std::optional<Buffer> scale_;

  bool rowwise_;
  bool columnwise_;
  std::string name_;
  std::mt19937 gen_;
};

constexpr uint32_t FP32_EXPONENT_BIAS = 127;
constexpr uint32_t FP32_MANTISSA_BITS = 23;

// [128,4] rowwise and [4,128] colwise alignment requirement
constexpr size_t scale_tensor_alignment_Y_rowwise = 128;
constexpr size_t scale_tensor_alignment_X_rowwise = 4;
constexpr size_t scale_tensor_alignment_Y_colwise = 4;
constexpr size_t scale_tensor_alignment_X_colwise = 128;

inline size_t divide_round_up(const size_t N, const size_t M) {
    return ((N + M) - 1) / M;
}

inline size_t round_up_to_nearest_multiple(const size_t N, const size_t M) {
    return divide_round_up(N, M) * M;
}

template <typename T>
struct Numeric_Traits {
    static constexpr double minSubnorm = 1.0;
    static constexpr double maxSubnorm = 1.0;
    static constexpr double minNorm    = 1.0;
    static constexpr double maxNorm    = 1.0;
    static constexpr double artifInf   = 1.0;
    static constexpr int maxBiasedExponent = 1;
};

template <>
struct Numeric_Traits<fp8e4m3> {
    static constexpr double minSubnorm = 1.0   / static_cast<double>(1 << 9);   // std::pow(2.0, -9.0);
    static constexpr double maxSubnorm = 0.875 / static_cast<double>(1 << 6);   // std::pow(2.0, -6.0);
    static constexpr double minNorm    = 1.0   / static_cast<double>(1 << 6);   // std::pow(2.0, -6.0);
    static constexpr double maxNorm    = 448.0;
    static constexpr double artifInf   = 10.0 * maxNorm;                        // artificial Infinity
    static constexpr int maxBiasedExponentAsFP32 = 8 + FP32_EXPONENT_BIAS;
    static constexpr int maxUnbiasedExponentAsFP32 = 8;
    static constexpr int maxExpNorm    = 1 << maxUnbiasedExponentAsFP32;
};

template <>
struct Numeric_Traits<fp8e5m2> {
    static constexpr double minSubnorm = 1.0  / static_cast<double>(1 << 16);   // std::pow(2.0, -16.0);
    static constexpr double maxSubnorm = 0.75 / static_cast<double>(1 << 14);   // std::pow(2.0, -14.0);
    static constexpr double minNorm    = 1.0  / static_cast<double>(1 << 14);   // std::pow(2.0, -14.0);
    static constexpr double maxNorm    = 57344.0;
    static constexpr double artifInf   = 10.0 * maxNorm;                        // artificial Infinity
    static constexpr int maxBiasedExponentAsFP32 = 15 + FP32_EXPONENT_BIAS;
    static constexpr int maxUnbiasedExponentAsFP32 = 15;
    static constexpr int maxExpNorm    = 1 << maxUnbiasedExponentAsFP32;
};

template <>
struct Numeric_Traits<fp32> {
    static constexpr double minSubnorm = std::numeric_limits<fp32>::denorm_min();   // std::pow(2.0, -149.0);
    static constexpr double maxSubnorm = std::numeric_limits<fp32>::min()
                                         - std::numeric_limits<fp32>::denorm_min(); // minNormalized - minDenormalized
    static constexpr double minNorm    = std::numeric_limits<fp32>::min();          // std::pow(2.0, -126.0);
    static constexpr double maxNorm    = std::numeric_limits<fp32>::max();          // (1 - pow(2, -24)) * pow(2, 128)
    static constexpr double artifInf   = std::numeric_limits<fp32>::infinity();
    static constexpr int maxBiasedExponentAsFP32 = 255;
    static constexpr int maxUnbiasedExponentAsFP32 = 128;
};

template <typename T>
struct Quantized_Limits {
    static constexpr double ranges[]  = {
        0.0,
        Numeric_Traits<T>::minNorm,
        Numeric_Traits<T>::maxNorm,
        Numeric_Traits<T>::artifInf
    };
    static constexpr inline fp32 max() { return static_cast<fp32>(Numeric_Traits<T>::maxNorm); }
    static constexpr inline fp32 max_reciprocal() { return static_cast<fp32>(1.0 / max()); }
    static constexpr inline fp32 emax() { return static_cast<fp32>(Numeric_Traits<T>::maxExpNorm); }
    static constexpr inline fp32 emax_reciprocal() { return static_cast<fp32>(1.0 / emax()); }
    static constexpr inline int max_norm_biased_exponent() { return Numeric_Traits<T>::maxBiasedExponentAsFP32; }
    static constexpr inline int max_norm_unbiased_exponent() { return Numeric_Traits<T>::maxUnbiasedExponentAsFP32; }
};

// Input data filling cases
// Considering normal and subnormal magnitudes of E4M3 and E5M2 formats
// with nearest to even rounding per OFP8 specification
enum InputsFillCase {
    zero_to_minNorm             = 0,    // [0, min_normal)
    minNorm_to_maxNorm          = 1,    // [min_normal, max_normal)
    maxNorm_to_inf              = 2,    // [max_normal, inf)
    zeros                       = 3,    // {0}
    uniform                     = 4,    // std::uniform_real_distribution<> dis(-2.0, 1.0)
};

inline fp8e8m0 float_to_e8m0(float val) {
  // TODO: nan/inf needs to be set for any value
  // of nan/inf in input not just amax.
  if (std::isnan(val)) {
    return 0xFF;
  }
  if (std::isinf(val)) {
    return 0xFE;
  }
  if (val == 0.0f) {
    return 0x00;
  }
  uint32_t val_u32 = *reinterpret_cast<uint32_t*>(&val);
  fp8e8m0 exponent = (val_u32 >> FP32_MANTISSA_BITS);
  uint32_t mantissa = val_u32 & 0x7FFFFF;
  // Round up exponent and deal with satfinite.
  if ((mantissa > 0 && exponent != 0xFE) && !(exponent == 0 && mantissa <= 0x400000)) {
    ++exponent;
  }
  return exponent;
}

inline float exp2f_rcp(fp8e8m0 biased_exp) {
  int32_t int_val = 0;
  if (biased_exp == 255) {
    int_val = 0x7fffffff;
  } else if (biased_exp == 254) {
    int_val = 0x00400000;
  } else {
    int_val = (254 - biased_exp) << FP32_MANTISSA_BITS;   // 127 - (biased_exp - 127)
  }
  float fp32_val = *reinterpret_cast<float*>(&int_val);
  return fp32_val;
}

inline float identity(const float x) { return x; }
inline float gelu(const float x)     { return x * (0.5f + 0.5f * tanhf(x * (0.79788456f + 0.03567741f * x * x))); }
inline float dgelu(const float x) {
    const float tanh_out = tanhf(0.79788456f * x * (1 + 0.044715f * x * x));
    return 0.5f * x * ((1 - tanh_out * tanh_out) * (0.79788456f + 0.1070322243f * x * x))
           + 0.5f * (1 + tanh_out);
}
inline float sigmoid(const float x)  { return 1 / (1 + expf(-x)); }
inline float dsigmoid(const float x) { return sigmoid(x) * (1 - sigmoid(x)); }
inline float qgelu(const float x)    { return x * sigmoid(1.702f * x); }
inline float dqgelu(const float x)   { return 1.702f * x * dsigmoid(1.702f * x) + sigmoid(1.702f * x); }
inline float relu(const float x)     { return fmaxf(0, x); }
inline float drelu(const float x)    { return x > 0 ? 1 : 0; }
inline float silu(const float x)     { return x * sigmoid(x); }
inline float dsilu(const float x)    { return x * dsigmoid(x) + sigmoid(x); }
inline float srelu(const float x)    { return x > 0 ? x * x : 0; }
inline float dsrelu(const float x)   { return fmaxf(0, 2 * x); }

size_t first_dimension(const std::vector<size_t> &shape);
size_t last_dimension(const std::vector<size_t> &shape);

bool areShapesEqual(const NVTEShape &s1, const NVTEShape &s2);

void compareResults(const std::string &name, Tensor &test, const void *ref,
                    bool rowwise, double atol = 1e-5, double rtol = 1e-8, bool if_on_gpus = true,
                    const size_t tolerable_mismatches_limit = 0);
void compareResults(const std::string &name, const float test, const float ref,
                    double atol = 1e-5, double rtol = 1e-8);
void compareResults(const std::string &name, const uint8_t *test, const uint8_t *ref,
                    size_t N, float mismatch_rate_tol = 0.);
template <typename T>
void compare_scaling_factors(const std::string &name, const T *test, const T *ref,
                             const size_t row_blocks, const size_t col_blocks, const size_t stride,
                             size_t& mismatches_num,
                             const size_t scale_diff_abs_tolerance = 0,
                             const double abs_tolerable_mismatches_limit = 0,
                             const double rel_tolerable_mismatches_limit = 0);


std::array<size_t, 4> get_scale_tensor_dims(const size_t rows, const size_t cols,
                                            const size_t block_size_rows, const size_t block_size_cols);

std::pair<double, double> getTolerances(const DType type);

void fillUniform(Tensor *t);

template <typename InputEncoding>
void fillCase(Tensor *t, const InputsFillCase fill_case);

void setRandomScale(Tensor *t);
void setRandomScaleInv(Tensor *t);

constexpr int THREADS_PER_WARP = 32;

const std::string &typeName(DType type);
const std::string& caseName(InputsFillCase type);

extern std::vector<DType> all_fp_types;

bool isFp8Type(DType type);
bool isFp4Type(DType type);

int32_t getDeviceComputeCapability();
constexpr int32_t hopperComputeCapability = 90;
constexpr int32_t blackwellComputeCapability = 100;

// Custom deleter for RAII
struct GroupedTensorDeleter {
  void operator()(NVTEGroupedTensor h) const { if (h) nvte_destroy_grouped_tensor(h); }
};

// Grouped tensor RAII class
using GroupedTensorHandle = std::unique_ptr<std::remove_pointer_t<NVTEGroupedTensor>, GroupedTensorDeleter>;

// Helper owning GPU buffers that back NVTEGroupedTensor.
// NVTEGroupedTensor does not own memory; data/offsets/scales
// must be allocated and freed by the test.
struct GroupedBuffers {
  GroupedTensorHandle handle;
  CudaPtr<> data;
  CudaPtr<> scale_inv;
  CudaPtr<> columnwise_scale_inv;
  CudaPtr<int64_t> first_dims_dev;
  CudaPtr<int64_t> last_dims_dev;
  CudaPtr<int64_t> offsets_dev;
  CudaPtr<> columnwise_data;
  NVTEShape logical_shape{};
  std::vector<int64_t> offsets_host;
  std::vector<size_t> tensor_bytes;
  size_t num_tensors{0};
  size_t elem_size{0};
  DType dtype{DType::kFloat32};
  NVTEScalingMode scaling_mode{NVTE_DELAYED_TENSOR_SCALING};

  GroupedBuffers() = default;
  GroupedBuffers(const GroupedBuffers&) = delete;
  GroupedBuffers& operator=(const GroupedBuffers&) = delete;
  GroupedBuffers(GroupedBuffers&&) = default;
  GroupedBuffers& operator=(GroupedBuffers&&) = default;
  ~GroupedBuffers() = default;

  // Convenience accessors for raw pointers
  NVTEGroupedTensor get_handle() const { return handle.get(); }
  void* get_data() const { return data.get(); }
};

GroupedBuffers build_grouped_tensor(const std::vector<Tensor*>& tensors,
                                    const NVTEScalingMode scaling_mode);

}  // namespace test

#if FP4_TYPE_SUPPORTED
#define SWITCH_FP4_TYPE_HANDLE(type, ...) \
  case DType::kFloat4E2M1: {              \
    using type = fp4e2m1;                 \
    { __VA_ARGS__ }                       \
  } break;
#else
#define SWITCH_FP4_TYPE_HANDLE(type, ...) // do nothing
#endif

#define TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(dtype, type, ...) \
    switch (dtype) { \
        using namespace transformer_engine; \
        case DType::kByte: \
            { \
                using type = byte; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kInt32: \
            { \
                using type = int32; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kInt64: \
            { \
                using type = int64; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kFloat32: \
            { \
                using type = float; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kFloat16: \
            { \
                using type = fp16; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kBFloat16: \
            { \
                using type = bf16; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kFloat8E4M3: \
            { \
                using type = fp8e4m3; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kFloat8E5M2: \
            { \
                using type = fp8e5m2; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kFloat8E8M0: \
            { \
                using type = fp8e8m0; \
                {__VA_ARGS__} \
            } \
        break; \
        SWITCH_FP4_TYPE_HANDLE(type, __VA_ARGS__) \
        default: \
            printf("dtype: %d\n", static_cast<int>(dtype)); \
            NVTE_ERROR("Invalid type."); \
    }

#define TRANSFORMER_ENGINE_TYPE_SWITCH_FP8_ONLY(dtype, type, ...) \
    switch (dtype) { \
        using namespace transformer_engine; \
        case DType::kFloat8E4M3: \
            { \
                using type = fp8e4m3; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kFloat8E5M2: \
            { \
                using type = fp8e5m2; \
                {__VA_ARGS__} \
            } \
        break; \
        default: \
            NVTE_ERROR("Invalid type."); \
    }

#define TRANSFORMER_ENGINE_TYPE_SWITCH_FP4_ONLY(dtype, type, ...) \
    switch (dtype) { \
        using namespace transformer_engine; \
        SWITCH_FP4_HANDLE(type, __VA_ARGS__) \
        default: \
            NVTE_ERROR("Invalid type."); \
    }

#define TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(dtype, type, ...) \
    switch (dtype) { \
        using namespace transformer_engine; \
        case DType::kFloat32: \
            { \
                using type = float; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kFloat16: \
            { \
                using type = fp16; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kBFloat16: \
            { \
                using type = bf16; \
                {__VA_ARGS__} \
            } \
        break; \
        default: \
            NVTE_ERROR("Invalid type."); \
    }
