/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common.h"

#include "c10/util/ArrayRef.h"
#include "pybind.h"
#include "transformer_engine/transformer_engine.h"
namespace transformer_engine::pytorch {

std::vector<size_t> getTensorShape(at::Tensor t) {
  std::vector<size_t> shape;
  for (auto s : t.sizes()) {
    shape.push_back(s);
  }
  return shape;
}

std::unique_ptr<Quantizer> convert_quantizer(py::handle quantizer) {
  init_extension();
  if (quantizer.is_none()) {
    return std::make_unique<NoneQuantizer>(quantizer);
  }
  for (auto [_check_type, check_quantizer_type, _create_tensor, create_quantizer] :
       detail::custom_types_converters) {
    if (check_quantizer_type(quantizer.ptr())) {
      return create_quantizer(quantizer);
    }
  }

  NVTE_ERROR("Unexpected type for quantizer");
}

transformer_engine::DType getTransformerEngineFP8Type(bool e4m3_if_hybrid,
                                                      const std::string& fp8_recipe) {
  // if e4m3 or hybrid + forward
  if ((fp8_recipe == "E4M3") || ((fp8_recipe == "HYBRID") && e4m3_if_hybrid)) {
    return transformer_engine::DType::kFloat8E4M3;
  }
  return transformer_engine::DType::kFloat8E5M2;
}

TensorWrapper makeTransformerEngineTensor(py::handle tensor, py::handle quantizer) {
  NVTE_CHECK(!tensor.is_none(), "Tensor is not allocated!");
  std::unique_ptr<Quantizer> my_quantizer = convert_quantizer(quantizer);
  for (auto [check_type, check_quantizer_type, create_tensor, _] :
       detail::custom_types_converters) {
    if (check_type(tensor.ptr())) {
      NVTE_CHECK(quantizer.is_none() || check_quantizer_type(quantizer.ptr()),
                 "Unexpected quantization params type.");
      auto x = create_tensor(tensor, my_quantizer.get());
      return x;
    }
  }

  // Regular pyTorch tensor
  at::Tensor torch_tensor = tensor.cast<at::Tensor>();

  // #TODO (pgadzinski) - needed in attention for non-contiguous tensors.
  //if (!torch_tensor.is_contiguous()) {
  //  torch_tensor = torch_tensor.contiguous();
  //}
  auto ret = TensorWrapper(my_quantizer->get_scaling_mode());
  ret.set_rowwise_data(torch_tensor.data_ptr(),
                       GetTransformerEngineDType(torch_tensor.scalar_type()),
                       getTensorShape(torch_tensor));
  my_quantizer->set_quantization_params(&ret);
  return ret;
}

transformer_engine::TensorWrapper makeTransformerEngineTensor(
    void* data_ptr, const NVTEShape& shape, const transformer_engine::DType type) {
  return transformer_engine::TensorWrapper(data_ptr, shape, type);
}

transformer_engine::TensorWrapper makeTransformerEngineTensor(
    void* data_ptr, const std::vector<size_t>& shape, const transformer_engine::DType type) {
  return transformer_engine::TensorWrapper(data_ptr, shape, type);
}

transformer_engine::TensorWrapper makeTransformerEngineTensor(at::Tensor tensor) {
  transformer_engine::DType dtype = GetTransformerEngineDType(tensor.scalar_type());
  std::vector<size_t> shape;
  for (auto s : tensor.sizes()) {
    shape.push_back(s);
  }
  return makeTransformerEngineTensor(tensor.data_ptr(), shape, dtype);
}

transformer_engine::TensorWrapper makeTransformerEngineTensor(
    void* data_ptr, const std::vector<size_t>& shape, const transformer_engine::DType type,
    void* amax_ptr, void* scale_ptr, void* scale_inv_ptr, std::vector<size_t> scale_inv_shape,
    NVTEScalingMode scaling_mode) {
  TensorWrapper ret(scaling_mode);
  ret.set_rowwise_data(data_ptr, type, shape);
  const std::vector<size_t> meta_shape{1};
  ret.set_amax(amax_ptr, DType::kFloat32, meta_shape);
  ret.set_scale(scale_ptr, DType::kFloat32, meta_shape);
  auto scale_inv_dtype =
      (scaling_mode == NVTE_MXFP8_1D_SCALING) ? DType::kFloat8E8M0 : DType::kFloat32;
  ret.set_rowwise_scale_inv(scale_inv_ptr, scale_inv_dtype, scale_inv_shape);
  return ret;
}

transformer_engine::TensorWrapper makeTransformerEngineTensor(
    void* data_ptr, void* columnwise_data_ptr, const std::vector<size_t>& shape,
    const std::vector<size_t>& columnwise_shape, const transformer_engine::DType type,
    void* amax_ptr, void* scale_ptr, void* scale_inv_ptr, void* columnwise_scale_inv_ptr,
    const std::vector<size_t>& scale_inv_shape,
    const std::vector<size_t>& columnwise_scale_inv_shape, NVTEScalingMode scaling_mode) {
  TensorWrapper ret(scaling_mode);
  ret.set_rowwise_data(data_ptr, type, shape);
  ret.set_columnwise_data(columnwise_data_ptr, type, columnwise_shape);
  const std::vector<size_t> meta_shape{1};
  ret.set_amax(amax_ptr, DType::kFloat32, meta_shape);
  ret.set_scale(scale_ptr, DType::kFloat32, meta_shape);
  auto scale_inv_dtype =
      (scaling_mode == NVTE_MXFP8_1D_SCALING) ? DType::kFloat8E8M0 : DType::kFloat32;
  ret.set_rowwise_scale_inv(scale_inv_ptr, scale_inv_dtype, scale_inv_shape);
  ret.set_columnwise_scale_inv(columnwise_scale_inv_ptr, scale_inv_dtype,
                               columnwise_scale_inv_shape);
  return ret;
}

transformer_engine::TensorWrapper makeTransformerEngineTensor(at::Tensor tensor, at::Tensor amax,
                                                              const at::Tensor scale,
                                                              at::Tensor scale_inv,
                                                              NVTEScalingMode scaling_mode) {
  transformer_engine::DType dtype = GetTransformerEngineDType(tensor.scalar_type());

  auto tensor_shape = getTensorShape(tensor);
  auto scale_inv_shape = getTensorShape(scale_inv);

  NVTE_CHECK(amax.scalar_type() == at::kFloat);
  NVTE_CHECK(scale.scalar_type() == at::kFloat);
  NVTE_CHECK(scale_inv.scalar_type() == at::kFloat);

  return makeTransformerEngineTensor(tensor.data_ptr(), tensor_shape, dtype, amax.data_ptr(),
                                     scale.data_ptr(), scale_inv.data_ptr(), scale_inv_shape,
                                     scaling_mode);
}

template <typename T>
T product(const std::vector<T>& shape) {
  T ret = 1;
  for (auto s : shape) {
    ret *= s;
  }
  return ret;
}

template size_t product<size_t>(const std::vector<size_t>& shape);
template int64_t product<int64_t>(const std::vector<int64_t>& shape);

size_t product(const NVTEShape& shape, size_t begin, size_t end) {
  NVTE_CHECK(begin <= end && end <= shape.ndim, "Attempted to access entries ", begin, " to ", end,
             " in a shape with ", shape.ndim, " entries");
  size_t ret = 1;
  for (size_t i = begin; i < end; ++i) {
    ret *= shape.data[i];
  }
  return ret;
}

std::vector<size_t> nvte_shape_to_vector(const NVTEShape& nvte_shape) {
  std::vector<size_t> shape;
  for (size_t i = 0; i < nvte_shape.ndim; i++) {
    shape.push_back(nvte_shape.data[i]);
  }
  return shape;
}

at::Tensor allocateSpace(const std::vector<size_t>& shape, const transformer_engine::DType type,
                         bool init_to_zeros) {
  std::vector<int64_t> shape_int64(shape.begin(), shape.end());
  c10::IntArrayRef ar_shape(shape_int64);
  if (init_to_zeros) {
    return at::zeros(ar_shape, at::CUDA(GetATenDType(type)));
  } else {
    return at::empty(ar_shape, at::CUDA(GetATenDType(type)));
  }
}

at::Tensor allocateSpace(const NVTEShape& shape, const transformer_engine::DType type,
                         bool init_to_zeros) {
  auto size = shape.ndim;
  if (size == 2 && init_to_zeros) {
    return at::zeros({static_cast<int64_t>(shape.data[0]), static_cast<int64_t>(shape.data[1])},
                     at::CUDA(GetATenDType(type)));
  } else if (size == 2) {
    return at::empty({static_cast<int64_t>(shape.data[0]), static_cast<int64_t>(shape.data[1])},
                     at::CUDA(GetATenDType(type)));
  } else if (size == 1 && init_to_zeros) {
    return at::zeros({static_cast<int64_t>(shape.data[0])}, at::CUDA(GetATenDType(type)));
  } else if (size == 1) {
    return at::empty({static_cast<int64_t>(shape.data[0])}, at::CUDA(GetATenDType(type)));
  }
  NVTE_CHECK(false, "Should never reach here! func: allocateSpace");
}

at::Tensor allocateTorchTensor(int M, int N, transformer_engine::DType dtype) {
  return at::empty({static_cast<int64_t>(M), static_cast<int64_t>(N)},
                   at::CUDA(GetATenDType(dtype)));
}

at::Tensor allocateTorchTensor(int M, transformer_engine::DType dtype) {
  return at::empty({static_cast<int64_t>(M)}, at::CUDA(GetATenDType(dtype)));
}

void* getDataPtr(at::Tensor tensor, int offset) {
  void* dptr = nullptr;
  if (tensor.numel() > 0) {
    dptr = tensor.data_ptr();
  }
  if (dptr != nullptr && offset != 0) {
    char* char_ptr = reinterpret_cast<char*>(dptr);
    char_ptr += offset * tensor.element_size();
    dptr = reinterpret_cast<void*>(char_ptr);
  }
  return dptr;
}

std::vector<size_t> convertShape(const NVTEShape& shape) {
  return std::vector<size_t>(shape.data, shape.data + shape.ndim);
}

int roundup(const int value, const int multiple) {
  assert(multiple > 0);
  return ((value + multiple - 1) / multiple) * multiple;
}

}  // namespace transformer_engine::pytorch
