/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

inline bool is_tensor_scaling(const NVTEScalingMode &mode) { return (mode.x == -1) && (mode.y == -1); }

std::tuple<size_t, size_t> get_num_scales_and_scale_size(const NVTEShape &shape, const NVTEScalingMode & scaling_mode) {
  if (is_tensor_scaling(scaling_mode)) {
    return {1, sizeof(float)};
  } else {
    assert(shape.ndim == 2);
    auto n_scales = std::ceil(shape.data[0] / scaling_mode.x) * std::ceil(shape.data[1] / scaling_mode.y);
    return {n_scales, n_scales * sizeof(uint8_t)};
  }
}

Tensor::Tensor(const NVTEShape &shape, const DType type, const NVTEScalingMode &scaling_mode) {
    size_t s = typeToSize(type);
    size_t total_size = product(shape) * s;
    void *dptr = nullptr;
    cpu_data_ = nullptr;
    amax_cpu_data_ = nullptr;
    scale_cpu_data_ = nullptr;
    scale_inv_cpu_data_ = nullptr;
    float *amax = nullptr, *scale = nullptr, *scale_inv = nullptr;
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
    auto scale_size = std::get<1>(get_num_scales_and_scale_size(shape, scaling_mode));
    cudaMalloc((void**)&scale_inv, scale_size);  // NOLINT(*)
    cudaMemset(scale_inv, 0, scale_size);
    scale_inv_cpu_data_ = std::make_unique<unsigned char[]>(scale_size);
    std::fill_n(scale_inv_cpu_data_.get(), scale_size, 0);
  }
  if (is_tensor_scaling(scaling_mode)) {
    tensor_ = TensorWrapper(dptr, shape, type, amax, scale, scale_inv);
  } else {
    std::vector<size_t> scale_inv_shape = {
      (shape.data[0] + scaling_mode.x - 1) / scaling_mode.x,
      (shape.data[1] + scaling_mode.y - 1) / scaling_mode.y};
    tensor_ = TensorWrapper(dptr, shape, type, amax, scale, scale_inv,
                            NVTEShape{scale_inv_shape.data(), scale_inv_shape.size()},
                            scaling_mode);
  }
}

void Tensor::to_cpu() const {
  const NVTEShape s = tensor_.shape();
  const size_t size = product(s) * typeToSize(tensor_.dtype());
  cudaMemcpy(cpu_data_.get(), tensor_.dptr(), size, cudaMemcpyDeviceToHost);
  if (isFp8Type(dtype())) {
    if (is_tensor_scaling(tensor_.scaling_mode())) {
      cudaMemcpy(amax_cpu_data_.get(), tensor_.amax(), sizeof(float),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(scale_cpu_data_.get(), tensor_.scale(), sizeof(float),
                 cudaMemcpyDeviceToHost);
    }
    auto scale_size = std::get<1>(get_num_scales_and_scale_size(tensor_.shape(), tensor_.scaling_mode()));
    cudaMemcpy(scale_inv_cpu_data_.get(), tensor_.scale_inv(), scale_size,
               cudaMemcpyDeviceToHost);
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
    auto scale_size = std::get<1>(get_num_scales_and_scale_size(tensor_.shape(), tensor_.scaling_mode()));
    cudaMemcpy(tensor_.scale_inv(), scale_inv_cpu_data_.get(), scale_size,
               cudaMemcpyHostToDevice);
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

void Tensor::set_scale_inv(float scale_inv) {
  if (isFp8Type(dtype())) {
    NVTE_CHECK(scale_inv_cpu_data_);
    auto num_scales = std::get<0>(get_num_scales_and_scale_size(tensor_.shape(), tensor_.scaling_mode()));
    if (num_scales == 1){
      cpu_scale_inv_ptr<float>()[0] = scale_inv;
    } else{
      static std::mt19937 gen(12345);
      std::uniform_int_distribution<uint8_t> dis(0, 127);
      for (size_t i = 0; i < num_scales; i++){
        cpu_scale_inv_ptr<uint8_t>()[i] = dis(gen);
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

void compareResults(const std::string &name, const Tensor &test, const void *ref,
                    double atol, double rtol) {
  test.to_cpu();
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

void compareResults(const std::string &name, const float test, const float ref,
                    double atol, double rtol) {
  double t = static_cast<double>(test);
  double r = static_cast<double>(ref);
  bool mismatch = fabs(t - r) > atol && (r == 0 || fabs((t - r) / r) > rtol);
  ASSERT_FALSE(mismatch) << "Error in " << name << std::endl
                         << "Mismatch: " << t << " vs " << r;

}

void compareResults(const std::string &name, const uint8_t *test, const uint8_t *ref,
                    size_t N) {
  for (int i = 0; i < N; i++){
    ASSERT_EQ(int(test[i]), int(ref[i])) << "Error in " << name
      << ". Mismatch at index " << i << std::endl;
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

void fillUniform(Tensor *t) {
  const size_t size = product(t->shape());
  static std::mt19937 gen(12345);
  std::uniform_real_distribution<> dis(-2.0, 1.0);
  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(t->dtype(), T, {
      T *data = t->cpu_dptr<T>();
      for (size_t i = 0; i < size; ++i) {
          data[i] = T(dis(gen));
      }
  });
  t->set_scale_inv(dis(gen));
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
  t->set_scale_inv(1.0);
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
  static std::mt19937 gen(12345);
  std::uniform_real_distribution<> dis(-2.0, 1.0);
  const float scale_inv = dis(gen);
  t->set_scale_inv(scale_inv);
}

bool isFp8Type(DType type) {
    return type == DType::kFloat8E4M3 || type == DType::kFloat8E5M2;
}

}  // namespace test
