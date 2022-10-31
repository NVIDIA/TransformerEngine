/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/


#include "test_common.h"
#include "transformer_engine/logging.h"
#include "transformer_engine/transformer_engine.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <memory>
#include <random>

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
    {DType::kFloat32, "float32"},
    {DType::kFloat16, "float16"},
    {DType::kBFloat16, "bfloat16"},
    {DType::kFloat8E4M3, "float8e4m3"},
    {DType::kFloat8E5M2, "float8e5m2"}};
  return name_map.at(type);
}

size_t product(const NVTEShape &shape) {
    size_t ret = 1;
    for (size_t i = 0; i < shape.ndim; ++i) {
      ret *= shape.data[i];
    }
    return ret;
}

Tensor::Tensor(const NVTEShape &shape, const DType type) {
    size_t s = typeToSize(type);
    size_t total_size = product(shape) * s;
    void *dptr = nullptr;
    cpu_data_ = nullptr;
    if (total_size != 0) {
        cudaMalloc((void**)&dptr, total_size);  // NOLINT(*)
        cudaMemset(dptr, 0, total_size);
        cpu_data_ = std::make_unique<unsigned char[]>(total_size);
    }
    tensor_ = TensorWrapper(dptr, shape, type);
}

void Tensor::to_cpu() const {
  const NVTEShape s = tensor_.shape();
  const size_t size = product(s) * typeToSize(tensor_.dtype());
  cudaMemcpy(cpu_data_.get(), tensor_.dptr(), size, cudaMemcpyDeviceToHost);
}

void Tensor::from_cpu() const {
  const NVTEShape s = tensor_.shape();
  const size_t size = product(s) * typeToSize(tensor_.dtype());
  cudaMemcpy(tensor_.dptr(), cpu_data_.get(), size, cudaMemcpyHostToDevice);
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

void fillUniform(const Tensor &t) {
  const size_t size = product(t.shape());
  static std::mt19937 gen(12345);
  std::uniform_real_distribution<> dis(-2.0, 1.0);
  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(t.dtype(), T, {
      T *data = t.cpu_dptr<T>();
      for (size_t i = 0; i < size; ++i) {
          data[i] = T(dis(gen));
      }
  });
  t.from_cpu();
}

}  // namespace test
