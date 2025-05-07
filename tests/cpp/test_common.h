/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once

#include <memory>
#include <vector>
#include <array>
#include <random>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include <transformer_engine/transformer_engine.h>
#include "util/logging.h"

namespace test {
using namespace transformer_engine;

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

template <typename T>
struct TypeInfo{
    using types = std::tuple<byte,
                             int16,
                             int32,
                             int64,
                             fp32,
                             fp16,
                             bf16,
                             fp8e4m3,
                             fp8e5m2,
                             fp8e8m0>;

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
    constexpr static size_t size = sizeof(T);
};

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

  Tensor() {}

  Tensor& operator=(const Tensor &other) = delete;
  Tensor(const Tensor &other) = delete;

  Tensor(Tensor &&other) = default;
  Tensor& operator=(Tensor &&other) = default;

  ~Tensor() {
    void *data_ptr = tensor_.dptr();
    void *scale_inv = tensor_.scale_inv();
    void *columnwise_data_ptr = tensor_.get_columnwise_data().data_ptr;
    void *columnwise_scale_inv = tensor_.get_columnwise_scale_inv().data_ptr;
    if (columnwise_data_ptr == data_ptr) {
      columnwise_data_ptr = nullptr;
    }
    if (columnwise_scale_inv == scale_inv) {
      columnwise_scale_inv = nullptr;
    }
    if (data_ptr != nullptr) {
      cudaFree(data_ptr);
    }
    if (scale_inv != nullptr) {
      cudaFree(scale_inv);
    }
    if (columnwise_data_ptr != nullptr) {
      cudaFree(columnwise_data_ptr);
    }
    if (columnwise_scale_inv != nullptr) {
      cudaFree(columnwise_scale_inv);
    }
  }

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
  T *rowwise_cpu_dptr() const {
    NVTE_CHECK(TypeInfo<T>::dtype == tensor_.dtype(), "Invalid type!");
    NVTE_CHECK(rowwise_, "Tensor does not have rowwise data!");
    return reinterpret_cast<T *>(cpu_data_rowwise_.get());
  }

  template <typename T>
  T *columnwise_cpu_dptr() const {
    NVTE_CHECK(TypeInfo<T>::dtype == tensor_.dtype(), "Invalid type!");
    NVTE_CHECK(columnwise_, "Tensor does not have columnwise data!");
    return reinterpret_cast<T *>(cpu_data_columnwise_.get());
  }

  float amax() const {
    if(amax_cpu_data_) {
      to_cpu();
      return *amax_cpu_data_;
    } else {
      return 0;
    }
  }

  float scale() const {
    if(scale_cpu_data_) {
      NVTE_CHECK(tensor_.scaling_mode() == NVTE_DELAYED_TENSOR_SCALING, "Invalid scaling_mode!");
      to_cpu();
      return *scale_cpu_data_;
    } else {
      return 1;
    }
  }

  template <typename T>
  T *rowwise_cpu_scale_inv_ptr(){
    if (tensor_.scaling_mode() == NVTE_DELAYED_TENSOR_SCALING){
      NVTE_CHECK(TypeInfo<T>::dtype == DType::kFloat32, "Invalid type!");
    } else if (tensor_.scaling_mode() == NVTE_BLOCK_SCALING_1D || tensor_.scaling_mode() == NVTE_BLOCK_SCALING_2D) {
      NVTE_CHECK(TypeInfo<T>::dtype == DType::kFloat32, "Invalid type!");
    } else {
      NVTE_CHECK(TypeInfo<T>::dtype == DType::kByte, "Invalid type!");
    }
    to_cpu();
    return reinterpret_cast<T*>(rowwise_scale_inv_cpu_data_.get());
  }

  template <typename T>
  T *columnwise_cpu_scale_inv_ptr(){
    if (tensor_.scaling_mode() == NVTE_DELAYED_TENSOR_SCALING){
      NVTE_CHECK(TypeInfo<T>::dtype == DType::kFloat32, "Invalid type!");
    } else if (tensor_.scaling_mode() == NVTE_BLOCK_SCALING_1D || tensor_.scaling_mode() == NVTE_BLOCK_SCALING_2D) {
      NVTE_CHECK(TypeInfo<T>::dtype == DType::kFloat32, "Invalid type!");
    } else {
      NVTE_CHECK(TypeInfo<T>::dtype == DType::kByte, "Invalid type!");
    }
    to_cpu();
    return reinterpret_cast<T*>(columnwise_scale_inv_cpu_data_.get());
  }

  float rowwise_scale_inv(){
    if(rowwise_scale_inv_cpu_data_) {
      float scale_inv = rowwise_cpu_scale_inv_ptr<float>()[0];
      return scale_inv;
    } else {
      return 1;
    }
  }

  bool rowwise() const {
    return rowwise_;
  }

  bool columnwise() const {
    return columnwise_;
  }

  void set_tensor_amax_nullptr(){
    tensor_.set_amax(nullptr, DType::kFloat32, tensor_.defaultShape);
  }

  void to_cpu() const;
  void from_cpu() const;
  void set_scale(float scale);
  void set_scale_inv(float scale_inv);
  void shareFP8Meta(const Tensor &other);

  std::mt19937& gen() { return gen_; }

 private:
  TensorWrapper tensor_;
  std::unique_ptr<unsigned char[]> cpu_data_rowwise_;
  std::unique_ptr<unsigned char[]> cpu_data_columnwise_;
  std::shared_ptr<float> amax_cpu_data_;
  std::shared_ptr<float> scale_cpu_data_;
  std::unique_ptr<unsigned char[]> rowwise_scale_inv_cpu_data_;
  std::unique_ptr<unsigned char[]> columnwise_scale_inv_cpu_data_;
  bool rowwise_;
  bool columnwise_;
  std::string name_;
  std::mt19937 gen_;
};

constexpr uint32_t FP32_EXPONENT_BIAS = 127;
constexpr uint32_t FP32_MANTISSA_BITS = 23;

// [128,4] rowwise and [4,128] colwise alignment requirement
constexpr size_t scale_tensor_alignment_X_rowwise = 4;
constexpr size_t scale_tensor_alignment_Y_rowwise = 128;
constexpr size_t scale_tensor_alignment_X_colwise = 128;
constexpr size_t scale_tensor_alignment_Y_colwise = 4;

inline size_t divide_round_up(const size_t N, const size_t M) {
    return (N - 1 + M) / M;
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
  return (biased_exp == 0) ? 1 : exp2f(FP32_EXPONENT_BIAS - static_cast<float>(biased_exp));
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

size_t typeToSize(DType type);
size_t product(const NVTEShape &shape);
size_t product(const std::vector<size_t> &shape);

size_t first_dimension(const std::vector<size_t> &shape);
size_t last_dimension(const std::vector<size_t> &shape);

bool areShapesEqual(const NVTEShape &s1, const NVTEShape &s2);

void compareResults(const std::string &name, const Tensor &test, const void *ref,
                    bool rowwise, double atol = 1e-5, double rtol = 1e-8, bool if_on_gpus = true);
void compareResults(const std::string &name, const float test, const float ref,
                    double atol = 1e-5, double rtol = 1e-8);
void compareResults(const std::string &name, const uint8_t *test, const uint8_t *ref,
                    size_t N, float mismatch_rate_tol = 0.);
void compare_e8m0_scaling_factors(const std::string &name, const uint8_t *test, const uint8_t *ref,
                                  const size_t row_blocks, const size_t col_blocks, const size_t stride);
void compare_e8m0_scaling_factors(const std::string &name, const uint8_t *test, const uint8_t *ref,
                                  const size_t N);

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

int32_t getDeviceComputeCapability();
constexpr int32_t hopperComputeCapability = 90;
constexpr int32_t blackwellComputeCapability = 100;

}  // namespace test

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
        default: \
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
