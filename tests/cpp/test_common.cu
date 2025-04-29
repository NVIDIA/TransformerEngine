/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/


#include "test_common.h"

#include <algorithm>
#include <memory>
#include <random>
#include <iostream>
#include <cassert>
#include <cmath>
#include <string>

#include <gtest/gtest.h>
#include <omp.h>

#include <transformer_engine/transformer_engine.h>
#include "util/logging.h"

namespace test {

size_t create_seed_from_tensor_name(const std::string& tensor_name) {
  auto full_name = std::string(testing::UnitTest::GetInstance()->current_test_info()->name()) +
                   "/" + tensor_name;
  return std::hash<std::string>{}(full_name);
}

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
    {DType::kFloat8E5M2, "float8e5m2"},
    {DType::kFloat8E8M0, "float8e8m0"}};
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

size_t product(const NVTEShape &shape, size_t begin, size_t end) {
    size_t ret = 1;
    NVTE_CHECK(end <= shape.ndim);
    for (size_t i = begin; i < end; ++i) {
      ret *= shape.data[i];
    }
    return ret;
}

size_t product(const NVTEShape &shape) {
  return product(shape, 0, shape.ndim);
}

size_t product(const std::vector<size_t> shape, size_t begin, size_t end) {
    size_t ret = 1;
    NVTE_CHECK(end <= shape.size());
    for (size_t i = begin; i < end; ++i) {
      ret *= shape[i];
    }
    return ret;
}

size_t product(const std::vector<size_t>& shape) {
  return product(shape, 0, shape.size());
}

size_t DIVUP(const size_t &x, const size_t &y){
  return (((x) + ((y)-1)) / (y));
}

struct scale_inv_meta {
  std::vector<size_t> shape;
  DType type;
  size_t type_size;
};

NVTEShape convertShape(const std::vector<size_t>& s) {
  return nvte_make_shape(s.data(), s.size());
}

std::pair<scale_inv_meta, scale_inv_meta> get_scales(const NVTEShape& shape,
                                                     const NVTEScalingMode scaling_mode) {
  if (scaling_mode == NVTE_DELAYED_TENSOR_SCALING) {
    scale_inv_meta ret;
    ret.shape = {1};
    ret.type = DType::kFloat32;
    ret.type_size = sizeof(float);
    return {ret, ret};
  }
  if (scaling_mode == NVTE_MXFP8_1D_SCALING) {
    std::vector<size_t> shape_vec;
    for (size_t i = 0; i < shape.ndim; ++i) {
      shape_vec.push_back(shape.data[i]);
    }
    size_t first_dim = first_dimension(shape_vec);
    size_t last_dim = last_dimension(shape_vec);

    scale_inv_meta ret_rowwise, ret_colwise;

    auto block_alignment = std::vector<size_t>{128ul, 4ul};
    {
      auto alignment = block_alignment[0];
      auto scale_dim_0 = DIVUP(DIVUP(first_dim, static_cast<size_t>(1)), alignment) * alignment;
      alignment = block_alignment[1];
      auto scale_dim_1 = DIVUP(DIVUP(last_dim, static_cast<size_t>(32)), alignment) * alignment;
      ret_rowwise.shape = {scale_dim_0, scale_dim_1};
    }
    {
      auto alignment = block_alignment[1];
      auto scale_dim_0 = DIVUP(DIVUP(first_dim, static_cast<size_t>(32)), alignment) * alignment;
      alignment = block_alignment[0];
      auto scale_dim_1 = DIVUP(DIVUP(last_dim, static_cast<size_t>(1)), alignment) * alignment;
      ret_colwise.shape = {scale_dim_0, scale_dim_1};
    }
    ret_rowwise.type = DType::kFloat8E8M0;
    ret_colwise.type = DType::kFloat8E8M0;
    ret_rowwise.type_size = sizeof(uint8_t);
    ret_colwise.type_size = sizeof(uint8_t);

    return {ret_rowwise, ret_colwise};
  }
  if (scaling_mode == NVTE_BLOCK_SCALING_2D) {
    std::vector<size_t> shape_vec;
    for (size_t i = 0; i < shape.ndim; ++i) {
      shape_vec.push_back(shape.data[i]);
    }
    size_t first_dim = first_dimension(shape_vec);
    size_t last_dim = last_dimension(shape_vec);

    scale_inv_meta ret_rowwise, ret_colwise;

    {
      auto scale_dim_0 = DIVUP(first_dim, static_cast<size_t>(128));
      auto scale_dim_1 = DIVUP(DIVUP(last_dim, static_cast<size_t>(128)), 4) * 4;
      ret_rowwise.shape = {scale_dim_0, scale_dim_1};
    }
    {
      auto scale_dim_0 = DIVUP(last_dim, static_cast<size_t>(128));
      auto scale_dim_1 = DIVUP(DIVUP(first_dim, static_cast<size_t>(128)), 4) * 4;
      ret_colwise.shape = {scale_dim_0, scale_dim_1};
    }
    ret_rowwise.type = DType::kFloat32;
    ret_colwise.type = DType::kFloat32;
    ret_rowwise.type_size = sizeof(float);
    ret_colwise.type_size = sizeof(float);

    return {ret_rowwise, ret_colwise};
  }
  if (scaling_mode == NVTE_BLOCK_SCALING_1D) {
    std::vector<size_t> shape_vec;
    for (size_t i = 0; i < shape.ndim; ++i) {
      shape_vec.push_back(shape.data[i]);
    }
    size_t first_dim = first_dimension(shape_vec);
    size_t last_dim = last_dimension(shape_vec);
    scale_inv_meta ret_rowwise, ret_colwise;

    {
      auto scale_dim_0 = DIVUP(last_dim, static_cast<size_t>(128));
      auto scale_dim_1 = DIVUP(first_dim, 4) * 4;
      ret_rowwise.shape = {scale_dim_0, scale_dim_1};
    }
    {
      auto scale_dim_0 = DIVUP(first_dim, static_cast<size_t>(128));
      auto scale_dim_1 = DIVUP(last_dim, 4) * 4;
      ret_colwise.shape = {scale_dim_0, scale_dim_1};
    }
    ret_rowwise.type = DType::kFloat32;
    ret_colwise.type = DType::kFloat32;
    ret_rowwise.type_size = sizeof(float);
    ret_colwise.type_size = sizeof(float);
    return {ret_rowwise, ret_colwise};
  }

  NVTE_ERROR("Invalid scaling mode!");
}

Tensor::Tensor(const std::string& name,
               const NVTEShape &shape, const DType type,
               const bool rowwise, const bool columnwise,
               const NVTEScalingMode &scaling_mode) {
  name_ = name;
  const size_t seed = create_seed_from_tensor_name(name);
  gen_.seed(seed);
  rowwise_ = rowwise;
  columnwise_ = columnwise;
  size_t s = typeToSize(type);
  size_t total_size = product(shape) * s;
  void *dptr_rowwise = nullptr;
  void *dptr_columnwise = nullptr;
  cpu_data_rowwise_ = nullptr;
  cpu_data_columnwise_ = nullptr;
  amax_cpu_data_ = nullptr;
  scale_cpu_data_ = nullptr;
  rowwise_scale_inv_cpu_data_ = nullptr;
  columnwise_scale_inv_cpu_data_ = nullptr;
  float *amax = nullptr, *scale = nullptr;
  float *rowwise_scale_inv = nullptr, *columnwise_scale_inv = nullptr;
  if (columnwise) {
    NVTE_CHECK(shape.ndim >= 2);
  }
  std::vector<size_t> normalized_shape_v = {product(shape, 0, shape.ndim - 1),
                                            shape.data[shape.ndim - 1]};
  NVTEShape normalized_shape = convertShape(normalized_shape_v);
  NVTEShape columnwise_shape = {};

  std::vector<size_t> columnwise_shape_vec;
  if (scaling_mode == NVTE_DELAYED_TENSOR_SCALING || scaling_mode == NVTE_BLOCK_SCALING_1D || scaling_mode == NVTE_BLOCK_SCALING_2D) {
    // Transpose when tensor scaling
    columnwise_shape_vec.emplace_back(shape.data[shape.ndim - 1]);
    for (size_t i = 0; i < shape.ndim - 1; ++i) {
      columnwise_shape_vec.emplace_back(shape.data[i]);
    }
  } else {
    // Same shape for MX
    for (size_t i = 0; i < shape.ndim; ++i) {
      columnwise_shape_vec.emplace_back(shape.data[i]);
    }
  }

  if (columnwise) {
    columnwise_shape = nvte_make_shape(columnwise_shape_vec.data(), columnwise_shape_vec.size());
  }

  tensor_ = TensorWrapper(scaling_mode);

  if (total_size != 0) {
    if (rowwise) {
      cudaMalloc((void**)&dptr_rowwise, total_size);  // NOLINT(*)
      cudaMemset(dptr_rowwise, 0, total_size);
      cpu_data_rowwise_ = std::make_unique<unsigned char[]>(total_size);
      std::fill_n(cpu_data_rowwise_.get(), total_size, 0);
    }
    if (columnwise) {
      cudaMalloc((void**)&dptr_columnwise, total_size);  // NOLINT(*)
      cudaMemset(dptr_columnwise, 0, total_size);
      cpu_data_columnwise_ = std::make_unique<unsigned char[]>(total_size);
      std::fill_n(cpu_data_columnwise_.get(), total_size, 0);
    }
  }
  tensor_.set_rowwise_data(dptr_rowwise, type, shape);
  tensor_.set_columnwise_data(dptr_columnwise, type, columnwise_shape);

  if (isFp8Type(type)) {
    if (scaling_mode == NVTE_DELAYED_TENSOR_SCALING) {
      cudaMalloc((void**)&amax, sizeof(float));  // NOLINT(*)
      cudaMemset(amax, 0, sizeof(float));
      cudaMalloc((void**)&scale, sizeof(float));  // NOLINT(*)
      cudaMemset(scale, 0, sizeof(float));
      amax_cpu_data_ = std::make_shared<float>(0);
      scale_cpu_data_ = std::make_shared<float>(0);
      tensor_.set_amax(amax, DType::kFloat32, std::vector<size_t>{1});
      tensor_.set_scale(scale, DType::kFloat32, std::vector<size_t>{1});
      cudaMalloc((void**)&rowwise_scale_inv, sizeof(float));  // NOLINT(*)
      if (rowwise) {
        tensor_.set_rowwise_scale_inv(rowwise_scale_inv, DType::kFloat32,
                                      std::vector<size_t>{1});
        rowwise_scale_inv_cpu_data_ = std::make_unique<unsigned char[]>(sizeof(float));
        std::fill_n(rowwise_scale_inv_cpu_data_.get(), sizeof(float), 0);
      }
      if (columnwise) {
        tensor_.set_columnwise_scale_inv(rowwise_scale_inv, DType::kFloat32,
                                         std::vector<size_t>{1});
        columnwise_scale_inv_cpu_data_ = std::make_unique<unsigned char[]>(sizeof(float));
        std::fill_n(columnwise_scale_inv_cpu_data_.get(), sizeof(float), 0);
      }
    } else {
      auto [rowwise_scale_meta, colwise_scale_meta] =
          get_scales(normalized_shape, tensor_.scaling_mode());
      auto rowwise_scale_size = product(rowwise_scale_meta.shape) * rowwise_scale_meta.type_size;
      auto columnwise_scale_size = product(colwise_scale_meta.shape) * colwise_scale_meta.type_size;
      auto scale_shape = rowwise_scale_meta.shape;
      auto columnwise_scale_shape = colwise_scale_meta.shape;
      if (rowwise) {
        cudaMalloc((void **)&rowwise_scale_inv, rowwise_scale_size);  // NOLINT(*)
        cudaMemset(rowwise_scale_inv, 0, rowwise_scale_size);
        rowwise_scale_inv_cpu_data_ = std::make_unique<unsigned char[]>(rowwise_scale_size);
        std::fill_n(rowwise_scale_inv_cpu_data_.get(), rowwise_scale_size, 0);
        auto scale_dtype = rowwise_scale_meta.type;
        tensor_.set_rowwise_scale_inv(rowwise_scale_inv, scale_dtype, scale_shape);
      }
      if (columnwise) {
        cudaMalloc((void**)&columnwise_scale_inv, columnwise_scale_size);  // NOLINT(*)
        cudaMemset(columnwise_scale_inv, 0, columnwise_scale_size);
        columnwise_scale_inv_cpu_data_ = std::make_unique<unsigned char[]>(columnwise_scale_size);
        std::fill_n(columnwise_scale_inv_cpu_data_.get(), columnwise_scale_size, 0);
        auto scale_dtype = colwise_scale_meta.type;
        tensor_.set_columnwise_scale_inv(columnwise_scale_inv, scale_dtype, columnwise_scale_shape);
      }
    }
  }
}

void Tensor::to_cpu() const {
  const NVTEShape s = tensor_.shape();
  const size_t size = product(s) * typeToSize(tensor_.dtype());
  if (rowwise_) {
    cudaMemcpy(cpu_data_rowwise_.get(),
               tensor_.get_rowwise_data().data_ptr,
               size,
               cudaMemcpyDeviceToHost);
  }
  if (columnwise_) {
    cudaMemcpy(cpu_data_columnwise_.get(),
               tensor_.get_columnwise_data().data_ptr,
               size,
               cudaMemcpyDeviceToHost);
  }
  if (isFp8Type(dtype())) {
    if (tensor_.scaling_mode() == NVTE_DELAYED_TENSOR_SCALING) {
      if (tensor_.amax() != nullptr){
        cudaMemcpy(amax_cpu_data_.get(),
                  tensor_.amax(),
                  sizeof(float),
                  cudaMemcpyDeviceToHost);
      }
      cudaMemcpy(scale_cpu_data_.get(),
                 tensor_.scale(),
                 sizeof(float),
                 cudaMemcpyDeviceToHost);
    }
    auto [rowwise_scale_meta, colwise_scale_meta] =
        get_scales(s, tensor_.scaling_mode());
    if (rowwise_) {
      auto scale_size = product(rowwise_scale_meta.shape) * rowwise_scale_meta.type_size;
      cudaMemcpy(rowwise_scale_inv_cpu_data_.get(),
                 tensor_.get_rowwise_scale_inv().data_ptr,
                 scale_size,
                 cudaMemcpyDeviceToHost);
    }
    if (columnwise_) {
      auto scale_size = product(colwise_scale_meta.shape) * colwise_scale_meta.type_size;
      cudaMemcpy(columnwise_scale_inv_cpu_data_.get(),
                 tensor_.get_columnwise_scale_inv().data_ptr,
                 scale_size,
                 cudaMemcpyDeviceToHost);
    }
  }
}

void Tensor::from_cpu() const {
  const NVTEShape s = tensor_.shape();
  const size_t size = product(s) * typeToSize(tensor_.dtype());
  if (rowwise_) {
    cudaMemcpy(tensor_.get_rowwise_data().data_ptr,
               cpu_data_rowwise_.get(), size, cudaMemcpyHostToDevice);
  }
  if (columnwise_) {
    cudaMemcpy(tensor_.get_columnwise_data().data_ptr,
               cpu_data_columnwise_.get(), size, cudaMemcpyHostToDevice);
  }
  if (isFp8Type(dtype())) {
    if (tensor_.scaling_mode() == NVTE_DELAYED_TENSOR_SCALING) {
      if (tensor_.amax() != nullptr){
        cudaMemcpy(tensor_.amax(), amax_cpu_data_.get(), sizeof(float),
                  cudaMemcpyHostToDevice);
      }
      cudaMemcpy(tensor_.scale(), scale_cpu_data_.get(), sizeof(float),
                 cudaMemcpyHostToDevice);
    }
    auto [rowwise_scale_meta, colwise_scale_meta] =
        get_scales(s, tensor_.scaling_mode());
    if (rowwise_) {
      auto scale_size = product(rowwise_scale_meta.shape) * rowwise_scale_meta.type_size;
      cudaMemcpy(tensor_.get_rowwise_scale_inv().data_ptr,
                 rowwise_scale_inv_cpu_data_.get(), scale_size,
                 cudaMemcpyHostToDevice);
    }
    if (columnwise_) {
      auto scale_size = product(colwise_scale_meta.shape) * colwise_scale_meta.type_size;
      cudaMemcpy(tensor_.get_columnwise_scale_inv().data_ptr,
                 columnwise_scale_inv_cpu_data_.get(), scale_size,
                 cudaMemcpyHostToDevice);
    }
  }
}

void Tensor::set_scale(float scale) {
  if (isFp8Type(dtype())) {
    NVTE_CHECK(scale_cpu_data_);
    if (tensor_.scaling_mode() == NVTE_DELAYED_TENSOR_SCALING) {
      *scale_cpu_data_ = scale;
      from_cpu();
    }
  }
}

void Tensor::set_scale_inv(float scale_inv) {
  if (isFp8Type(dtype())) {
    if (rowwise_) {
      NVTE_CHECK(rowwise_scale_inv_cpu_data_);
    }
    if (columnwise_) {
      NVTE_CHECK(columnwise_scale_inv_cpu_data_);
    }

    auto [rowwise_scale_meta, colwise_scale_meta] =
        get_scales(tensor_.shape(), tensor_.scaling_mode());
    if (rowwise_) {
      auto num_scales = product(rowwise_scale_meta.shape);
      if (num_scales == 1) {
        rowwise_cpu_scale_inv_ptr<float>()[0] = scale_inv;
      } else {
        std::uniform_int_distribution<uint8_t> dis(0, 127);
        auto *scale_inv_ptr = rowwise_cpu_scale_inv_ptr<uint8_t>();
        for (size_t i = 0; i < num_scales; i++) {
          scale_inv_ptr[i] = dis(gen_);
        }
      }
    }
    if (columnwise_) {
      auto num_scales = product(colwise_scale_meta.shape);
      if (num_scales == 1) {
        columnwise_cpu_scale_inv_ptr<float>()[0] = scale_inv;
      } else {
        std::uniform_int_distribution<uint8_t> dis(0, 127);
        auto *scale_inv_ptr = columnwise_cpu_scale_inv_ptr<uint8_t>();
        for (size_t i = 0; i < num_scales; i++) {
          scale_inv_ptr[i] = dis(gen_);
        }
      }
    }
    from_cpu();
  }
}

void Tensor::shareFP8Meta(const Tensor &other) {
  if (isFp8Type(dtype()) && isFp8Type(other.dtype())) {
    auto new_tensor = TensorWrapper(other.tensor_.scaling_mode());
    auto my_rowwise_data = tensor_.get_rowwise_data();
    new_tensor.set_rowwise_data(my_rowwise_data.data_ptr, static_cast<DType>(my_rowwise_data.dtype),
                                my_rowwise_data.shape);
    auto my_columnwise_data = tensor_.get_columnwise_data();
    new_tensor.set_columnwise_data(my_columnwise_data.data_ptr,
                                   static_cast<DType>(my_columnwise_data.dtype),
                                   my_columnwise_data.shape);
    auto other_amax = other.tensor_.get_amax();
    new_tensor.set_amax(other_amax.data_ptr, static_cast<DType>(other_amax.dtype),
                        other_amax.shape);
    auto other_scale = other.tensor_.get_scale();
    new_tensor.set_scale(other_scale.data_ptr, static_cast<DType>(other_scale.dtype),
                         other_scale.shape);
    auto other_row_scale_inv = other.tensor_.get_rowwise_scale_inv();
    new_tensor.set_rowwise_scale_inv(other_row_scale_inv.data_ptr,
                                     static_cast<DType>(other_row_scale_inv.dtype),
                                     other_row_scale_inv.shape);
    auto other_col_scale_inv = other.tensor_.get_columnwise_scale_inv();
    new_tensor.set_columnwise_scale_inv(other_col_scale_inv.data_ptr,
                                        static_cast<DType>(other_col_scale_inv.dtype),
                                        other_col_scale_inv.shape);
    tensor_ = std::move(new_tensor);
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
  for (size_t current = shape.ndim - 1; current > 0; --current) {
    ret.push_back(current_i % shape.data[current]);
    current_i /= shape.data[current];
  }
  ret.push_back(current_i);
  std::reverse(ret.begin(), ret.end());
  return ret;
}

void compareResults_sequential(const std::string &name, const Tensor &test,
                               const void *ref, const bool rowwise,
                               double atol, double rtol, bool if_on_gpus) {
  if (if_on_gpus) test.to_cpu();
  const auto& shape = rowwise ? test.rowwise_shape() : test.columnwise_shape();
  const size_t N = product(shape);
  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(test.dtype(), T,
    const T *test_data = rowwise ? test.rowwise_cpu_dptr<T>() : test.columnwise_cpu_dptr<T>();
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
      std::string direction = rowwise ? "rowwise" : "columnwise";
      ASSERT_FALSE(assertion) << "Error in tensor " << name << " in "
                              << direction << " direction." << std::endl
                              << "Mismatch at place " << to_string(unravel(i, shape))
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
                             const bool rowwise, double atol, double rtol, bool if_on_gpus) {
  if (if_on_gpus) test.to_cpu();
  const auto& shape = rowwise ? test.rowwise_shape() : test.columnwise_shape();
  const size_t N = product(shape);
  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(test.dtype(), T,
    const T *test_data = rowwise ? test.rowwise_cpu_dptr<T>() : test.columnwise_cpu_dptr<T>();
    const T *ref_data = reinterpret_cast<const T*>(ref);

    const size_t i = getFirstMismatchIdx<T>(test.dtype(), test_data, ref_data, N, atol, rtol);
    if (i != N) {
      const double t = static_cast<double>(test_data[i]);
      const double r = static_cast<double>(ref_data[i]);
      std::string direction = rowwise ? "rowwise" : "columnwise";
      ASSERT_FALSE(true) << "Error in tensor " << name << " in "
                         << direction << " direction." << std::endl
                         << "Mismatch at place " << to_string(unravel(i, shape))
                         << " (" << std::to_string(i) << "): " << t << " vs " << r;
    }
  );
}

void compareResults(const std::string &name, const Tensor &test, const void *ref,
                    const bool rowwise, double atol, double rtol, bool if_on_gpus) {
  constexpr bool sequential = false;
  if constexpr (sequential) {
    compareResults_sequential(name, test, ref, rowwise, atol, rtol, if_on_gpus);
  } else {
    compareResults_parallel(name, test, ref, rowwise, atol, rtol, if_on_gpus);
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
                                  const size_t row_blocks, const size_t col_blocks, const size_t stride)
{
  for (int i = 0; i < row_blocks; ++i) {
    for (int j = 0; j < col_blocks; ++j) {
      const int idx = i * stride + j;
      ASSERT_FALSE(test[idx] != ref[idx]) << "Error in " << name << std::endl
        << "Mismatch: " << static_cast<int>(test[idx]) << " vs "
        << static_cast<int>(ref[idx]) << " at index " << idx;
    }
  }
}

void compare_e8m0_scaling_factors(const std::string &name, const uint8_t *test, const uint8_t *ref,
                                  const size_t N)
{
  for (int i = 0; i < N; i++) {
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
    case DType::kFloat8E8M0:
      return {1e-2, 1e-2};
    default:
      NVTE_CHECK("Invalid type!");
  }
  return {0, 0};
}

template <typename T>
void generate_data_uniformly(T* data, const size_t size, std::mt19937* gen) {
  #pragma omp parallel proc_bind(spread)
  {
    std::mt19937 gen_local = *gen;
    const int thread_ID = omp_get_thread_num();
    const int threads_num = omp_get_max_threads();
    const int chunk_size = (size + threads_num - 1) / threads_num;
    const int idx_min = chunk_size * thread_ID;
    const int idx_max = std::min(chunk_size * (thread_ID + 1), static_cast<int>(size));
    gen_local.discard(idx_min);
    std::uniform_real_distribution<> dis(-2.0, 1.0);

    for (int i = idx_min; i < idx_max; ++i) {
      data[i] = static_cast<T>(dis(gen_local));
    }
  }
  gen->discard(size);
}

void fillUniform(Tensor *t) {
  if (t->rowwise()) {
    const size_t size = product(t->rowwise_shape());
    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(t->dtype(), T,
      {
        T *data = t->rowwise_cpu_dptr<T>();
        generate_data_uniformly(data, size, &(t->gen()));
      }
    );
  } else {
    const size_t size = product(t->columnwise_shape());
    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(t->dtype(), T,
      {
        T *data = t->columnwise_cpu_dptr<T>();
        generate_data_uniformly(data, size, &(t->gen()));
      }
    );
  }
  std::uniform_real_distribution<> dis(-2.0, 1.0);
  t->set_scale_inv(dis(t->gen()));
  t->from_cpu();
}

template<typename InputEncoding, InputsFillCase Case>
void fillCase_special(Tensor *t) {
  const size_t size = product(t->rowwise_shape());

  if constexpr (Case == InputsFillCase::zeros) {
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(t->dtype(), InputType, {
      InputType *data = t->rowwise_cpu_dptr<InputType>();
      for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<InputType>(0);
      }
    });
  } else {
    double minAbs = -2.0;
    double maxAbs = 1.0;
    if constexpr (Case != InputsFillCase::uniform) {
      minAbs = Quantized_Limits<InputEncoding>::ranges[Case];
      maxAbs = Quantized_Limits<InputEncoding>::ranges[Case + 1];
    }
    std::uniform_real_distribution<> dis(minAbs, maxAbs);
    std::uniform_real_distribution<> dis_sign(-1.0, 1.0);
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(t->dtype(), InputType, {
      InputType *data = t->rowwise_cpu_dptr<InputType>();
      for (size_t idx = 0; idx < size; ++idx) {
        const bool is_negative = (dis_sign(t->gen()) < 0.0);
        double val = dis(t->gen());
        if (is_negative) {
          val = -val;
        }
        data[idx] = static_cast<InputType>(val);
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
  std::uniform_real_distribution<> dis(-2.0, 1.0);
  const float scale = dis(t->gen());
  t->set_scale(scale);
}

void setRandomScaleInv(Tensor *t) {
  std::uniform_real_distribution<> dis(-2.0, 1.0);
  const float scale_inv = dis(t->gen());
  t->set_scale_inv(scale_inv);
}

bool isFp8Type(DType type) {
  return type == DType::kFloat8E4M3 || type == DType::kFloat8E5M2 || type == DType::kFloat8E8M0;
}

int32_t getDeviceComputeCapability() {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  return 10 * deviceProp.major + deviceProp.minor;
}

size_t first_dimension(const std::vector<size_t> &shape) {
  if (shape.size() == 0) return 1;
  if (shape.size() == 1) return 1;
  return product(shape, 0, shape.size() - 1);
}

size_t last_dimension(const std::vector<size_t> &shape) {
  if (shape.size() == 0) return 1;
  return shape[shape.size() - 1];
}

std::array<size_t, 4> get_scale_tensor_dims(const size_t rows,
                                            const size_t cols,
                                            const size_t block_size_rows,
                                            const size_t block_size_cols) {
    const bool is_rowwise = (block_size_rows == 1) && (block_size_cols == 32);

    const size_t alignment_Y = is_rowwise
                               ? scale_tensor_alignment_Y_rowwise
                               : scale_tensor_alignment_Y_colwise;
    const size_t alignment_X = is_rowwise
                               ? scale_tensor_alignment_X_rowwise
                               : scale_tensor_alignment_X_colwise;

    const size_t unpadded_blocks_Y = divide_round_up(rows, block_size_rows);
    const size_t unpadded_blocks_X = divide_round_up(cols, block_size_cols);

    const size_t blocks_Y = round_up_to_nearest_multiple(unpadded_blocks_Y, alignment_Y);
    const size_t blocks_X = round_up_to_nearest_multiple(unpadded_blocks_X, alignment_X);
    return {unpadded_blocks_Y, unpadded_blocks_X, blocks_Y, blocks_X};
}

}  // namespace test
