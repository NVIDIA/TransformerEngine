/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/


#include "test_common.h"

#include <algorithm>
#include <memory>
#include <random>
#include <cassert>
#include <cmath>

#include <gtest/gtest.h>
#include <omp.h>

#include <transformer_engine/transformer_engine.h>
#include "util/logging.h"

namespace test {

std::vector<DType> all_fp_types = {DType::kFloat32,
                                   DType::kFloat16,
                                   DType::kBFloat16,
                                   DType::kFloat8E5M2,
                                   DType::kFloat8E4M3};

bool areShapesEqual(const NVTEShape &s1, const NVTEShape &s2) {
  if (s1.ndim != s2.ndim) return false;

  for (size_t i = 0; i < s1.ndim; ++i) {
    if (s1.data[i] != s2.data[i]) return false;
  }

  return true;
}

size_t typeToSize(DType type) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(type, T,
  {
      return TypeInfo<T>::size;
  });
}

const std::string &typeName(DType type) {
  static const std::unordered_map<DType, std::string> name_map = {
    {DType::kByte, "byte"},
    {DType::kInt32, "int32"},
    {DType::kInt64, "int64"},
    {DType::kFloat32, "float32"},
    {DType::kFloat16, "float16"},
    {DType::kBFloat16, "bfloat16"},
    {DType::kFloat8E4M3, "float8e4m3"},
    {DType::kFloat8E5M2, "float8e5m2"}};
  return name_map.at(type);
}

const std::string& caseName(InputsFillCase type) {
  static const std::unordered_map<InputsFillCase, std::string> name_map = {
    {InputsFillCase::uniform, "uniform"},
    {InputsFillCase::zeros, "zeros"},
    {InputsFillCase::zero_to_minNorm, "zero_to_minNorm"},
    {InputsFillCase::minNorm_to_maxNorm, "minNorm_to_maxNorm"},
    {InputsFillCase::maxNorm_to_inf, "maxNorm_to_inf"}};
  return name_map.at(type);
}

size_t product(const NVTEShape &shape) {
    size_t ret = 1;
    for (size_t i = 0; i < shape.ndim; ++i) {
      ret *= shape.data[i];
    }
    return ret;
}
size_t product(const std::vector<size_t> shape) {
    size_t ret = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
      ret *= shape[i];
    }
    return ret;
}

size_t DIVUP(const size_t &x, const size_t &y){
  return (((x) + ((y)-1)) / (y));
}

inline bool is_tensor_scaling(const NVTEScalingMode &mode) { return (mode.x == -1) && (mode.y == -1); }

inline std::vector<size_t> get_scale_shape(const NVTEShape &shape, const NVTEScalingMode& scaling_mode){
    NVTE_CHECK(shape.ndim == 2,
               "Invalid shape of the tensor. Expected 2 dimensions for fine granularity scaling.");
  // Need (4, 128) alignment even for e8 scaling factor
  auto block_alignment = std::vector<size_t>{4ul, 128ul};
    auto alignment = block_alignment[scaling_mode.x < scaling_mode.y];
    auto scale_dim_0 = DIVUP(DIVUP(shape.data[0],
                                   static_cast<size_t>(scaling_mode.x)),
                             alignment) * alignment;
    alignment = block_alignment[scaling_mode.x > scaling_mode.y];
    auto scale_dim_1 = DIVUP(DIVUP(shape.data[1],
                                   static_cast<size_t>(scaling_mode.y)),
                             alignment) * alignment;
  return {scale_dim_0, scale_dim_1};
}

size_t get_scale_size(const NVTEShape &shape, const NVTEScalingMode & scaling_mode) {
  if (is_tensor_scaling(scaling_mode)) {
    return sizeof(float);
  } else {
    auto n_scales = product(get_scale_shape(shape, scaling_mode));
    return n_scales * sizeof(uint8_t);
  }
}

Tensor::Tensor(const NVTEShape &shape, const DType type, const NVTEScalingMode &scaling_mode, const bool _is_tensor_2x) : is_tensor_2x(_is_tensor_2x) {
  size_t total_size = product(shape) * typeToSize(type);
  void *dptr = nullptr;
  cpu_data_ = nullptr;
  amax_cpu_data_ = nullptr;
  scale_cpu_data_ = nullptr;
  scale_inv_cpu_data_ = nullptr;
  float *amax = nullptr, *scale = nullptr, *scale_inv = nullptr;
  void *columnwise_dptr = nullptr, *columnwise_scale_inv = nullptr;
  columnwise_cpu_data_ = nullptr; columnwise_scale_inv_cpu_data_ = nullptr;

  if (total_size != 0) {
    cudaMalloc((void**)&dptr, total_size);  // NOLINT(*)
    cudaMemset(dptr, 0, total_size);
    cpu_data_ = std::make_unique<unsigned char[]>(total_size);
    std::fill_n(cpu_data_.get(), total_size, 0);
  }

  if (isFp8Type(type)) {
    if (is_tensor_scaling(scaling_mode)) {
      cudaMalloc((void**)&amax, sizeof(float));  // NOLINT(*)
      cudaMemset(amax, 0, sizeof(float));
      cudaMalloc((void**)&scale, sizeof(float));  // NOLINT(*)
      cudaMemset(scale, 0, sizeof(float));
      amax_cpu_data_ = std::make_shared<float>(0);
      scale_cpu_data_ = std::make_shared<float>(0);
    }

    auto scale_size = get_scale_size(shape, scaling_mode);
    cudaMalloc((void**)&scale_inv, scale_size);  // NOLINT(*)
    cudaMemset(scale_inv, 0, scale_size);
    scale_inv_cpu_data_ = std::make_unique<unsigned char[]>(scale_size);
    std::fill_n(scale_inv_cpu_data_.get(), scale_size, 0);

    if (is_tensor_2x){
      cudaMalloc((void**)&columnwise_dptr, total_size);  // NOLINT(*)
      cudaMemset(columnwise_dptr, 0, total_size);
      columnwise_cpu_data_ = std::make_unique<unsigned char[]>(total_size);
      std::fill_n(columnwise_cpu_data_.get(), total_size, 0);

      auto columnwise_scale_size = get_scale_size(shape,
        NVTEScalingMode{scaling_mode.y, scaling_mode.x, scaling_mode.delayed_scaling});
      cudaMalloc((void**)&columnwise_scale_inv, columnwise_scale_size);  // NOLINT(*)
      cudaMemset(columnwise_scale_inv, 0, columnwise_scale_size);
      columnwise_scale_inv_cpu_data_ = std::make_unique<unsigned char[]>(columnwise_scale_size);
      std::fill_n(columnwise_scale_inv_cpu_data_.get(), columnwise_scale_size, 0);
    }
  }

  if (is_tensor_scaling(scaling_mode)) {
    tensor_ = TensorWrapper(dptr, shape, type, amax, scale, scale_inv);
  } else if (is_tensor_2x){
    auto scale_shape = get_scale_shape(shape, scaling_mode);
    auto columnwise_scale_shape = get_scale_shape(shape,
        NVTEScalingMode{scaling_mode.y, scaling_mode.x, scaling_mode.delayed_scaling});
    tensor_ = TensorWrapper(dptr, shape,
                            columnwise_dptr, shape,
                            type,
                            scale_inv, NVTEShape{scale_shape.data(), scale_shape.size()},
                            columnwise_scale_inv, NVTEShape{columnwise_scale_shape.data(), columnwise_scale_shape.size()},
                            DType::kByte,
                            scaling_mode);
  } else {
    auto scale_shape = get_scale_shape(shape, scaling_mode);
    tensor_ = TensorWrapper(dptr, shape, type, amax, scale, scale_inv,
                            NVTEShape{scale_shape.data(), scale_shape.size()},
                            scaling_mode);
  }
}

void Tensor::to_cpu() const {
  const size_t size = product(tensor_.shape()) * typeToSize(tensor_.dtype());
  cudaMemcpy(cpu_data_.get(), tensor_.dptr(), size, cudaMemcpyDeviceToHost);
  if (isFp8Type(dtype())) {
    if (is_tensor_scaling(tensor_.scaling_mode())) {
      cudaMemcpy(amax_cpu_data_.get(), tensor_.amax(), sizeof(float),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(scale_cpu_data_.get(), tensor_.scale(), sizeof(float),
                 cudaMemcpyDeviceToHost);
    }
    cudaMemcpy(scale_inv_cpu_data_.get(), tensor_.scale_inv(),
               product(tensor_.scale_inv_shape()) * typeToSize(tensor_.scale_inv_dtype()),
               cudaMemcpyDeviceToHost);
    if (is_tensor_2x){
      cudaMemcpy(columnwise_cpu_data_.get(), tensor_.columnwise_dptr(),
                 product(tensor_.columnwise_shape()) * typeToSize(tensor_.dtype()),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(columnwise_scale_inv_cpu_data_.get(), tensor_.columnwise_scale_inv(),
                 product(tensor_.columnwise_scale_inv_shape()) * typeToSize(tensor_.scale_inv_dtype()),
                 cudaMemcpyDeviceToHost);

    }
  }
}

void Tensor::from_cpu() const {
  const NVTEShape s = tensor_.shape();
  const size_t size = product(s) * typeToSize(tensor_.dtype());
  cudaMemcpy(tensor_.dptr(), cpu_data_.get(), size, cudaMemcpyHostToDevice);
  if (isFp8Type(dtype())) {
    if (is_tensor_scaling(tensor_.scaling_mode())) {
      cudaMemcpy(tensor_.amax(), amax_cpu_data_.get(), sizeof(float),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(tensor_.scale(), scale_cpu_data_.get(), sizeof(float),
                 cudaMemcpyHostToDevice);
    }
    cudaMemcpy(tensor_.scale_inv(), scale_inv_cpu_data_.get(),
               product(tensor_.scale_inv_shape()) * typeToSize(tensor_.scale_inv_dtype()),
               cudaMemcpyHostToDevice);
    if (is_tensor_2x){
      cudaMemcpy(tensor_.columnwise_dptr(), columnwise_cpu_data_.get(),
                 product(tensor_.columnwise_shape()) * typeToSize(tensor_.dtype()),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(tensor_.columnwise_scale_inv(), columnwise_scale_inv_cpu_data_.get(),
                 product(tensor_.columnwise_scale_inv_shape()) * typeToSize(tensor_.scale_inv_dtype()),
               cudaMemcpyHostToDevice);
    }
  }
}

void Tensor::set_scale(float scale) {
  if (isFp8Type(dtype())) {
    NVTE_CHECK(scale_cpu_data_);
  if (is_tensor_scaling(tensor_.scaling_mode())) {
      *scale_cpu_data_ = scale;
      from_cpu();
    }
  }
}

void Tensor::set_scale_inv() {
  if (isFp8Type(dtype())) {
    NVTE_CHECK(scale_inv_cpu_data_);
    auto num_scales = product(tensor_.scale_inv_shape());
    static std::mt19937 gen(12345);
    if (num_scales == 1){
      std::uniform_real_distribution<> dis(-2.0, 1.0);
      cpu_scale_inv_ptr<float>()[0] = dis(gen);
    } else{
      std::uniform_int_distribution<uint8_t> dis(0, 127);
      auto* scale_inv_ptr = cpu_scale_inv_ptr<uint8_t>();
      for (size_t i = 0; i < num_scales; i++){
        scale_inv_ptr[i] = dis(gen);
      }
      if (is_tensor_2x){
        NVTE_CHECK(columnwise_scale_inv_cpu_data_);
        auto* columnwise_scale_inv_ptr = columnwise_cpu_scale_inv_ptr<uint8_t>();
        auto columnwise_num_scales = product(tensor_.columnwise_scale_inv_shape());
        for (size_t i = 0; i < columnwise_num_scales; i++){
          columnwise_scale_inv_ptr[i] = dis(gen);
        }
      }
    }
    from_cpu();
  }
}

void Tensor::shareFP8Meta(const Tensor &other) {
  if(isFp8Type(dtype()) && isFp8Type(other.dtype())) {
    tensor_ = TensorWrapper(dptr(), shape(), dtype(),
                            other.tensor_.amax(),
                            other.tensor_.scale(),
                            other.tensor_.scale_inv(),
                            other.tensor_.scale_inv_shape(),
                            other.tensor_.scaling_mode());
    to_cpu();
  }
}

using std::to_string;

template <typename T>
std::string to_string(const std::vector<T> &v) {
  std::string s = "[";
  for (const auto x : v) {
    s += to_string(x) + ", ";
  }
  s.pop_back();
  s.pop_back();
  return s + "]";
}

std::vector<size_t> unravel(const size_t i, const NVTEShape &shape) {
  std::vector<size_t> ret;
  size_t current_i = i;
  for (size_t current = shape.ndim - 1;
       current > 0;
       --current) {
    ret.push_back(current_i % shape.data[current]);
    current_i /= shape.data[current];
  }
  ret.push_back(current_i);
  std::reverse(ret.begin(), ret.end());
  return ret;
}

void compareResults_sequential(const std::string &name, const Tensor &test, const void *ref,
                               double atol, double rtol, bool if_on_gpus) {
  if (if_on_gpus) test.to_cpu();
  const size_t N = product(test.shape());
  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(test.dtype(), T,
    const T *test_data = test.cpu_dptr<T>();
    const T *ref_data = reinterpret_cast<const T*>(ref);
    for (size_t i = 0; i < N; ++i) {
      double t = static_cast<double>(test_data[i]);
      double r = static_cast<double>(ref_data[i]);
      bool mismatch = fabs(t - r) > atol && (r == 0 || fabs((t - r) / r) > rtol);
      /* For Float32 the floating point comparison is enough to error out */
      bool assertion = mismatch && test.dtype() == DType::kFloat32;
      if (mismatch && !assertion) {
        /* Check if it is just a failure of round to nearest choosing different
           side of the real value */
        const double mean = (t + r) / 2;
        const double mean_p = mean >= 0 ? mean * (1 + 1e-6) : mean * (1 - 1e-6);
        const double mean_m = mean >= 0 ? mean * (1 - 1e-6) : mean * (1 + 1e-6);
        const double cast_mean_p = static_cast<double>(static_cast<T>(mean_p));
        const double cast_mean_m = static_cast<double>(static_cast<T>(mean_m));
        assertion = !(cast_mean_m == std::min(t,r) && cast_mean_p == std::max(t,r));
      }
      ASSERT_FALSE(assertion) << "Error in tensor " << name << std::endl
                              << "Mismatch at place " << to_string(unravel(i, test.shape()))
                              << " (" << std::to_string(i) << "): " << t << " vs " << r;
    }
  );
}

template <typename T>
static size_t getFirstMismatchIdx(const DType data_type, const T* test_data, const T* ref_data,
                                  const size_t N, const double atol, const double rtol) {
  int first_mismatch_idx = N;

  bool is_mismatch_found = false;
  #pragma omp parallel for schedule(static) firstprivate(is_mismatch_found) \
    reduction(min: first_mismatch_idx) proc_bind(spread)
  for (size_t i = 0; i < N; ++i) {
    if (is_mismatch_found) {    // early escape of the omp thread
      continue;
    }

    double t = static_cast<double>(test_data[i]);
    double r = static_cast<double>(ref_data[i]);

    bool mismatch = fabs(t - r) > atol && (r == 0 || fabs((t - r) / r) > rtol);
    /* For Float32 the floating point comparison is enough to error out */
    bool assertion = mismatch && (data_type == DType::kFloat32);
    if (mismatch && !assertion) {
      /* Check if it is just a failure of round to nearest choosing different
          side of the real value */
      const double mean = (t + r) / 2;
      const double mean_p = mean >= 0 ? mean * (1 + 1e-6) : mean * (1 - 1e-6);
      const double mean_m = mean >= 0 ? mean * (1 - 1e-6) : mean * (1 + 1e-6);
      const double cast_mean_p = static_cast<double>(static_cast<T>(mean_p));
      const double cast_mean_m = static_cast<double>(static_cast<T>(mean_m));
      assertion = !(cast_mean_m == std::min(t,r) && cast_mean_p == std::max(t,r));
    }
    if (assertion && i < first_mismatch_idx) {
      first_mismatch_idx = i;
      is_mismatch_found = true;
    }
  }
  return first_mismatch_idx;
}

void compareResults_parallel(const std::string &name, const Tensor &test, const void *ref,
                    double atol, double rtol, bool if_on_gpus) {
  if (if_on_gpus) test.to_cpu();
  const size_t N = product(test.shape());
  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(test.dtype(), T,
    const T *test_data = test.cpu_dptr<T>();
    const T *ref_data = reinterpret_cast<const T*>(ref);

    const size_t i = getFirstMismatchIdx<T>(test.dtype(), test_data, ref_data, N, atol, rtol);
    if (i != N) {
      const double t = static_cast<double>(test_data[i]);
      const double r = static_cast<double>(ref_data[i]);
      ASSERT_FALSE(true) << "Error in tensor " << name << std::endl
                         << "Mismatch at place " << to_string(unravel(i, test.shape()))
                         << " (" << std::to_string(i) << "): " << t << " vs " << r;
    }
  );
}

void compareResults(const std::string &name, const Tensor &test, const void *ref,
                    double atol, double rtol, bool if_on_gpus) {
  constexpr bool sequential = false;
  if constexpr (sequential) {
    compareResults_sequential(name, test, ref, atol, rtol, if_on_gpus);
  } else {
    compareResults_parallel(name, test, ref, atol, rtol, if_on_gpus);
  }
}

void compareResults(const std::string &name, const float test, const float ref,
                    double atol, double rtol) {
  double t = static_cast<double>(test);
  double r = static_cast<double>(ref);
  bool mismatch = fabs(t - r) > atol && (r == 0 || fabs((t - r) / r) > rtol);
  ASSERT_FALSE(mismatch) << "Error in " << name << std::endl
                         << "Mismatch: " << t << " vs " << r;

}


void compareResults(const std::string &name, const uint8_t *test, const uint8_t *ref,
                    size_t N, float mismatch_rate_tol) {
  size_t max_mismatches = std::ceil(N * mismatch_rate_tol);
  size_t n_mismatches = 0;
  std::vector<size_t> mismatch_indices;
  for (int i = 0; i < N; i++){
    bool mismatch = test[i] != ref[i];
    if (mismatch){
      n_mismatches++;
      mismatch_indices.push_back(i);
    }
    if (n_mismatches > max_mismatches){
      std::cout << "Error in " << name << std::endl;
      for (auto &index : mismatch_indices)
        std::cout << "Mismatch at (" << index << "):" << static_cast<int>(test[i]) << " vs "
        << static_cast<int>(ref[i]) << std::endl;
      GTEST_FAIL() << n_mismatches << " mismatche(s) which is more than mismatch tol.";
    }
  }
}

void compare_e8m0_scaling_factors(const std::string &name, const uint8_t *test, const uint8_t *ref,
                    size_t N) {
  for (int i = 0; i < N; i++){
    ASSERT_FALSE(test[i] != ref[i]) << "Error in " << name << std::endl
      << "Mismatch: " << static_cast<int>(test[i]) << " vs "
      << static_cast<int>(ref[i]) << " at index " << i;
  }
}

std::pair<double, double> getTolerances(const DType type) {
  switch(type) {
    case DType::kFloat32:
      return {1e-6, 5e-6};
    case DType::kFloat16:
      return {1e-5, 1e-3};
    case DType::kBFloat16:
      return {1e-5, 1e-2};
    case DType::kFloat8E4M3:
    case DType::kFloat8E5M2:
      return {1e-2, 1e-2};
    default:
      NVTE_CHECK("Invalid type!");
  }
  return {0, 0};
}

template <typename T>
void generate_data_uniformly(T* data, const size_t size) {
  const int seed = 12345;
  #pragma omp parallel proc_bind(spread)
  {
    std::mt19937 gen(seed);
    gen.discard(omp_get_thread_num() * 599);
    std::uniform_real_distribution<> dis(-2.0, 1.0);
    #pragma omp for schedule(static)
    for (size_t i = 0; i < size; ++i) {
      data[i] = static_cast<T>(dis(gen));
    }
  }
}

void fillUniform(Tensor *t) {
  const size_t size = product(t->shape());
  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(t->dtype(), T,
    {
      generate_data_uniformly(t->cpu_dptr<T>(), product(t->shape()));
      if (t->tensor_2x()) generate_data_uniformly(t->columnwise_cpu_dptr<T>(), product(t->columnwise_shape()));

    }
  );
  t->set_scale_inv();
  t->from_cpu();
}

template<typename InputEncoding, InputsFillCase Case>
void fillCase_special(Tensor *t) {
  const size_t size = product(t->shape());
  const size_t rows = t->shape().data[0];
  const size_t cols = t->shape().data[1];

  if constexpr (Case == InputsFillCase::zeros) {
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(t->dtype(), InputType, {
      InputType *data = t->cpu_dptr<InputType>();
      for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<InputType>(0);
      }
    });
  } else {
    double minAbs = -2.0;
    double maxAbs =  1.0;
    if constexpr (Case != InputsFillCase::uniform) {
      minAbs = Quantized_Limits<InputEncoding>::ranges[Case];
      maxAbs = Quantized_Limits<InputEncoding>::ranges[Case + 1];
    }
    static std::mt19937 gen(12345);
    std::uniform_real_distribution<> dis(minAbs, maxAbs);
    std::uniform_real_distribution<> dis_sign(-1.0, 1.0);
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(t->dtype(), InputType, {
      InputType *data = t->cpu_dptr<InputType>();
      for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
          const size_t idx = i * cols + j;
          const bool is_negative = (dis_sign(gen) < 0.0);
          double val = dis(gen);
          if (is_negative) {
            val = -val;
          }
          data[idx] = static_cast<InputType>(val);
        }
      }
    });
  }
  t->set_scale_inv();
  t->from_cpu();
}

template <typename InputEncoding>
void fillCase(Tensor *t, const InputsFillCase fill_case) {
  switch (fill_case) {
    case InputsFillCase::uniform:
        fillCase_special<InputEncoding, InputsFillCase::uniform>(t); break;
    case InputsFillCase::zeros:
        fillCase_special<InputEncoding, InputsFillCase::zeros>(t); break;
    case InputsFillCase::zero_to_minNorm:
        fillCase_special<InputEncoding, InputsFillCase::zero_to_minNorm>(t); break;
    case InputsFillCase::minNorm_to_maxNorm:
        fillCase_special<InputEncoding, InputsFillCase::minNorm_to_maxNorm>(t); break;
    case InputsFillCase::maxNorm_to_inf:
        fillCase_special<InputEncoding, InputsFillCase::maxNorm_to_inf>(t); break;
  }
}

template void fillCase<fp8e4m3>(Tensor *t, const InputsFillCase fill_case);
template void fillCase<fp8e5m2>(Tensor *t, const InputsFillCase fill_case);
template void fillCase<fp32>(Tensor *t, const InputsFillCase fill_case);

void setRandomScale(Tensor *t) {
  static std::mt19937 gen(12345);
  std::uniform_real_distribution<> dis(-2.0, 1.0);
  const float scale = dis(gen);
  t->set_scale(scale);
}

void setRandomScaleInv(Tensor *t) {
  t->set_scale_inv();
}

bool isFp8Type(DType type) {
    return type == DType::kFloat8E4M3 || type == DType::kFloat8E5M2;
}

int32_t getDeviceComputeCapability()
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    return 10 * deviceProp.major + deviceProp.minor;
}

}  // namespace test
