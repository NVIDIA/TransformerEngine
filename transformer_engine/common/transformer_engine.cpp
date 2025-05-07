/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/transformer_engine.h>

#include <cstring>
#include <iostream>

#include "common.h"
#include "common/util/cuda_runtime.h"

namespace transformer_engine {

size_t typeToSize(const DType type) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(type, T,
                                     return TypeInfo<T>::size;);  // NOLINT(*)
}

bool is_fp8_dtype(const DType t) { return t == DType::kFloat8E4M3 || t == DType::kFloat8E5M2; }

std::string to_string(const DType type) {
  switch (type) {
    case DType::kByte:
      return "Byte";
    case DType::kBFloat16:
      return "BFloat16";
    case DType::kFloat16:
      return "Float16";
    case DType::kFloat32:
      return "Float32";
    case DType::kFloat8E4M3:
      return "Float8E4M3";
    case DType::kFloat8E5M2:
      return "Float8E5M2";
    case DType::kFloat8E8M0:
      return "Float8E8M0";
    case DType::kInt32:
      return "Int32";
    case DType::kInt64:
      return "Int64";
    default:
      return concat_strings("Invalid type ", static_cast<int>(type));
  }
}

std::string to_string(const NVTEScalingMode &mode) {
  switch (mode) {
    case NVTE_DELAYED_TENSOR_SCALING:
      return "NVTE_DELAYED_TENSOR_SCALING";
    case NVTE_MXFP8_1D_SCALING:
      return "NVTE_MXFP8_1D_SCALING";
    case NVTE_INVALID_SCALING:
      return "NVTE_INVALID_SCALING";
  }
  return "Invalid Scaling";
}

void CheckNoopTensor(const Tensor &t, const std::string &name) {
  if (t.data.dptr != nullptr) {
    NVTE_CHECK(t.numel() == 1, "Expected 1 element for ", name, " noop, but found ", t.numel(),
               ".");
    NVTE_CHECK(t.data.dtype == DType::kFloat32, "Found wrong dtype for ", name,
               " noop. Expected kFloat32.");
  }
}

void CheckScaleTensorShape(const Tensor &t, const std::string &name) {
  NVTE_CHECK(t.scaling_mode != NVTE_INVALID_SCALING, "Invalid scaling mode!");
  if (is_tensor_scaling(t.scaling_mode)) {
    // per-tensor scaling
    if (t.has_data()) {
      NVTE_CHECK(t.scale_inv.numel() == 1, "Tensor \"", name,
                 "\" has invalid scale_inv shape (expected (1), got ", t.scale_inv.shape, ")");
    }
    if (t.has_columnwise_data()) {
      NVTE_CHECK(t.columnwise_scale_inv.numel() == 1, "Tensor \"", name,
                 "\" has invalid columnwise_scale_inv shape (expected (1), got ",
                 t.columnwise_scale_inv.shape, ")");
    }
  } else {
    if (t.scaling_mode == NVTE_MXFP8_1D_SCALING) {
      // Need (4, 128) alignment even for e8 scaling factor
      auto block_alignment = std::vector<size_t>{128ul, 4ul};
      size_t expected_x, expected_y, alignment;

      if (t.has_data()) {
        alignment = block_alignment[0];
        expected_x =
            DIVUP(DIVUP(t.flat_first_dim(), static_cast<size_t>(1)), alignment) * alignment;
        alignment = block_alignment[1];
        expected_y =
            DIVUP(DIVUP(t.flat_last_dim(), static_cast<size_t>(32)), alignment) * alignment;
        const auto &expected = std::vector<size_t>{expected_x, expected_y};
        NVTE_CHECK(t.scale_inv.shape == expected, "Tensor \"", name,
                   "\" has invalid scale_inv shape (expected ", expected, ", got ",
                   t.scale_inv.shape, ")");
      }
      if (t.has_columnwise_data()) {
        alignment = block_alignment[1];
        expected_x =
            DIVUP(DIVUP(t.flat_first_dim(), static_cast<size_t>(32)), alignment) * alignment;
        alignment = block_alignment[0];
        expected_y = DIVUP(DIVUP(t.flat_last_dim(), static_cast<size_t>(1)), alignment) * alignment;
        const auto &expected = std::vector<size_t>{expected_x, expected_y};
        NVTE_CHECK(t.columnwise_scale_inv.shape == expected, "Tensor \"", name,
                   "\"  has invalid columnwise_scale_inv shape (expected ", expected, ", got ",
                   t.columnwise_scale_inv.shape, ")");
      }
    }
  }
}

void CheckInputTensor(const Tensor &t, const std::string &name) {
  const DType type = t.dtype();
  if (is_fp8_dtype(type)) {
    // FP8 input needs to have scale_inv
    if (t.has_data()) {
      NVTE_CHECK(t.scale_inv.dptr != nullptr, "FP8 scaling factor input ", name,
                 "_scale_inverse must be allocated");
      NVTE_CHECK(t.scale_inv.dtype == DType::kFloat32 || t.scale_inv.dtype == DType::kFloat8E8M0,
                 "FP8 scaling factor input ", name,
                 "_scale_inverse has invalid dtype "
                 "(expected Float32 or Byte, got ",
                 to_string(t.scale_inv.dtype), ")");
    }
    if (t.has_columnwise_data()) {
      NVTE_CHECK(t.columnwise_scale_inv.dptr != nullptr, "FP8 scaling factor input ", name,
                 "_columnwise_scale_inverse must be allocated");
      NVTE_CHECK(t.columnwise_scale_inv.dtype == DType::kFloat32 ||
                     t.columnwise_scale_inv.dtype == DType::kFloat8E8M0,
                 "FP8 scaling factor input ", name,
                 "_columnwise_scale_inverse has invalid dtype "
                 "(expected Float32 or Byte, got ",
                 to_string(t.columnwise_scale_inv.dtype), ")");
    }
  } else {
    NVTE_CHECK(t.scale.dptr == nullptr, "Scale is not supported for non-FP8 input ", name);
    NVTE_CHECK(t.amax.dptr == nullptr, "Amax is not supported for non-FP8 input ", name);
    NVTE_CHECK(t.scale_inv.dptr == nullptr, "Scale_inv is not supported for non-FP8 input ", name);
    NVTE_CHECK(t.columnwise_scale_inv.dptr == nullptr,
               "Scale_inv is not supported for non-FP8 input ", name);
  }
  NVTE_CHECK(t.has_data() || t.has_columnwise_data(), "Input ", name, " is not allocated!");

  CheckScaleTensorShape(t, name);
}

void CheckOutputTensor(const Tensor &t, const std::string &name, bool allow_empty) {
  const DType type = t.dtype();
  if (is_fp8_dtype(type)) {
    // FP8 output needs to have scale, scale_inv and (if delayed scaling) amax
    if (t.scaling_mode == NVTE_DELAYED_TENSOR_SCALING && t.amax.dptr != nullptr) {
      NVTE_CHECK(t.amax.dtype == DType::kFloat32, "Invalid amax dtype (expected ",
                 to_string(DType::kFloat32), ", got ", to_string(t.amax.dtype), ")");
      NVTE_CHECK(product(t.amax.shape) == 1, "Invalid shape of amax in output ", name,
                 " (expected 1 entry, got shape=", t.amax.shape, ")");
    }
    if (t.has_data()) {
      NVTE_CHECK(t.scale_inv.dptr != nullptr, "FP8 scaling factor output ", name,
                 "_scale_inverse must be allocated");
      NVTE_CHECK(t.scale_inv.dtype == DType::kFloat32 || t.scale_inv.dtype == DType::kFloat8E8M0,
                 "FP8 scaling factor output ", name,
                 "_scale_inverse has invalid dtype "
                 "(expected Float32 or Float8E8M0, got ",
                 to_string(t.scale_inv.dtype), ")");
    }
    if (t.has_columnwise_data()) {
      NVTE_CHECK(t.columnwise_scale_inv.dptr != nullptr, "FP8 scaling factor output ", name,
                 "_columnwise_scale_inverse must be allocated");
      NVTE_CHECK(t.columnwise_scale_inv.dtype == DType::kFloat32 ||
                     t.columnwise_scale_inv.dtype == DType::kFloat8E8M0,
                 "FP8 scaling factor output ", name,
                 "_columnwise_scale_inverse has invalid dtype "
                 "(expected Float32 or Float8E8M0, got ",
                 to_string(t.columnwise_scale_inv.dtype), ")");
    }
  } else {
    NVTE_CHECK(t.scale.dptr == nullptr, "Scale is not supported for non-FP8 output ", name);
    NVTE_CHECK(t.amax.dptr == nullptr, "Amax is not supported for non-FP8 output ", name);
    NVTE_CHECK(t.scale_inv.dptr == nullptr, "Scale_inv is not supported for non-FP8 output ", name);
    NVTE_CHECK(t.columnwise_scale_inv.dptr == nullptr,
               "Scale_inv is not supported for non-FP8 input ", name);
  }

  if (!allow_empty) {
    NVTE_CHECK(t.has_data() || t.has_columnwise_data(), "Output ", name, " is not allocated!");
  }

  CheckScaleTensorShape(t, name);
}

}  // namespace transformer_engine

NVTETensor nvte_create_tensor(NVTEScalingMode scaling_mode) {
  transformer_engine::Tensor *ret = new transformer_engine::Tensor;
  ret->scaling_mode = scaling_mode;
  return ret;
}

void nvte_destroy_tensor(NVTETensor tensor) {
  if (tensor == nullptr) return;
  auto *t = reinterpret_cast<transformer_engine::Tensor *>(tensor);
  delete t;
}

NVTEDType nvte_tensor_type(const NVTETensor tensor) {
  if (tensor == nullptr) return kNVTEFloat32;
  return static_cast<NVTEDType>(
      reinterpret_cast<const transformer_engine::Tensor *>(tensor)->dtype());
}

NVTEShape nvte_make_shape(const size_t *data, size_t ndim) {
  NVTEShape ret;
  if (ndim == 0) {
    ret.ndim = 0;
    return ret;
  }
  NVTE_CHECK(ndim <= sizeof(ret.data) / sizeof(ret.data[0]),
             "Too many dims for NVTEShape (requested: ", ndim,
             ", max: ", sizeof(ret.data) / sizeof(ret.data[0]), ")");
  std::copy(data, data + ndim, ret.data);
  ret.ndim = ndim;
  return ret;
}

NVTEShape nvte_tensor_shape(const NVTETensor tensor) {
  if (tensor == nullptr) {
    NVTE_ERROR("Invalid tensor");
  }

  // Determine tensor shape depending on tensor format
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  std::vector<size_t> shape = t.shape();

  return nvte_make_shape(shape.data(), shape.size());
}

NVTEShape nvte_tensor_columnwise_shape(const NVTETensor tensor) {
  if (tensor == nullptr) {
    NVTE_ERROR("Invalid tensor");
  }
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  return nvte_make_shape(t.columnwise_data.shape.data(), t.columnwise_data.shape.size());
}

size_t nvte_tensor_ndims(const NVTETensor tensor) { return nvte_tensor_shape(tensor).ndim; }

size_t nvte_tensor_size(const NVTETensor tensor, const size_t dim) {
  const auto &shape = nvte_tensor_shape(tensor);
  NVTE_CHECK(0 <= dim && dim < shape.ndim, "Attempted to access index ", dim,
             " in a shape array with ", shape.ndim, " entries");
  return shape.data[dim];
}

size_t nvte_tensor_numel(const NVTETensor tensor) {
  const auto &shape = nvte_tensor_shape(tensor);
  size_t numel = 1;
  for (size_t i = 0; i < shape.ndim; i++) {
    numel *= shape.data[i];
  }
  return numel;
}

size_t nvte_tensor_element_size(const NVTETensor tensor) {
  if (tensor == nullptr) return sizeof(float);
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  return transformer_engine::typeToSize(t.dtype());
}

void *nvte_tensor_data(const NVTETensor tensor) {
  if (tensor == nullptr) return nullptr;
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  return t.data.dptr;
}

void *nvte_tensor_columnwise_data(const NVTETensor tensor) {
  if (tensor == nullptr) return nullptr;
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  return t.columnwise_data.dptr;
}

float *nvte_tensor_amax(const NVTETensor tensor) {
  if (tensor == nullptr) return nullptr;
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  NVTE_CHECK(t.amax.dtype == transformer_engine::DType::kFloat32,
             "Tensor's amax must have Float32 type!");
  return reinterpret_cast<float *>(t.amax.dptr);
}

float *nvte_tensor_scale(const NVTETensor tensor) {
  if (tensor == nullptr) return nullptr;
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  NVTE_CHECK(t.scale.dtype == transformer_engine::DType::kFloat32,
             "Tensor's scale must have Float32 type!");
  return reinterpret_cast<float *>(t.scale.dptr);
}

float *nvte_tensor_scale_inv(const NVTETensor tensor) {
  if (tensor == nullptr) return nullptr;
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  return reinterpret_cast<float *>(t.scale_inv.dptr);
}

void *nvte_tensor_columnwise_scale_inv(const NVTETensor tensor) {
  if (tensor == nullptr) return nullptr;
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  return t.columnwise_scale_inv.dptr;
}

NVTEShape nvte_tensor_scale_inv_shape(const NVTETensor tensor) {
  if (tensor == nullptr) {
    return nvte_make_shape(nullptr, 0);
  }
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  return nvte_make_shape(t.scale_inv.shape.data(), t.scale_inv.shape.size());
}

void nvte_set_tensor_param(NVTETensor *tensor, NVTETensorParam param_name,
                           const NVTEBasicTensor *param) {
  NVTE_CHECK(tensor != nullptr, "Tensor pointer can't be NULL.");
  NVTE_CHECK(*tensor != nullptr, "Tensor is not allocated.");
  auto &t = *reinterpret_cast<transformer_engine::Tensor *>(*tensor);
  switch (param_name) {
    case kNVTERowwiseData:
      t.data = *param;
      break;
    case kNVTEColumnwiseData:
      t.columnwise_data = *param;
      break;
    case kNVTEScale:
      t.scale = *param;
      break;
    case kNVTEAmax:
      t.amax = *param;
      break;
    case kNVTERowwiseScaleInv:
      t.scale_inv = *param;
      break;
    case kNVTEColumnwiseScaleInv:
      t.columnwise_scale_inv = *param;
      break;
    default:
      NVTE_ERROR("Unknown tensor parameter!");
  }
}

NVTEBasicTensor nvte_get_tensor_param(const NVTETensor tensor, NVTETensorParam param_name) {
  if (tensor == nullptr) {
    return {nullptr, kNVTEFloat32, nvte_make_shape(nullptr, 0)};
  }
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  switch (param_name) {
    case kNVTERowwiseData:
      return t.data;
    case kNVTEColumnwiseData:
      return t.columnwise_data;
    case kNVTEScale:
      return t.scale;
    case kNVTEAmax:
      return t.amax;
    case kNVTERowwiseScaleInv:
      return t.scale_inv;
    case kNVTEColumnwiseScaleInv:
      return t.columnwise_scale_inv;
    default:
      NVTE_ERROR("Unknown tensor parameter!");
  }
}

NVTEScalingMode nvte_tensor_scaling_mode(const NVTETensor tensor) {
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  return t.scaling_mode;
}

void nvte_tensor_pack_create(NVTETensorPack *pack) {
  for (int i = 0; i < pack->MAX_SIZE; i++) {
    pack->tensors[i] = reinterpret_cast<NVTETensor>(new transformer_engine::Tensor);
  }
}

void nvte_tensor_pack_destroy(NVTETensorPack *pack) {
  for (int i = 0; i < pack->MAX_SIZE; i++) {
    auto *t = reinterpret_cast<transformer_engine::Tensor *>(pack->tensors[i]);
    delete t;
  }
}

void nvte_zero_tensor(const NVTETensor tensor, cudaStream_t stream) {
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  // Zero out tensor data if allocated
  if (t.data.dptr != nullptr) {
    size_t size_in_bytes = nvte_tensor_element_size(tensor) * nvte_tensor_numel(tensor);
    cudaMemsetAsync(t.data.dptr, 0, size_in_bytes, stream);
  }
  // Set amax to 0 if allocated
  if (t.amax.dptr != nullptr) {
    cudaMemsetAsync(t.amax.dptr, 0, sizeof(float), stream);
  }
}

NVTEQuantizationConfig nvte_create_quantization_config() {
  return new transformer_engine::QuantizationConfig;
}

void nvte_get_quantization_config_attribute(NVTEQuantizationConfig config,
                                            NVTEQuantizationConfigAttribute attr, void *buf,
                                            size_t size_in_bytes, size_t *size_written) {
  // Write attribute size
  NVTE_CHECK(attr < kNVTEQuantizationConfigNumAttributes,
             "Invalid NVTEQuantizationConfigAttribute (got ", static_cast<int>(attr), ")");
  NVTE_CHECK(size_written != nullptr, "Invalid size_written (got NULL)");
  const auto &attr_size = transformer_engine::QuantizationConfig::attr_sizes[attr];
  *size_written = attr_size;

  // Return immediately if buffer is not provided
  if (buf == nullptr) {
    return;
  }

  // Check buffer size
  NVTE_CHECK(size_in_bytes >= attr_size,
             "Buffer is too small for quantization config attribute "
             "(attribute ",
             static_cast<int>(attr), " needs ", attr_size, " bytes, but buffer has ", size_in_bytes,
             " bytes)");

  // Write to buffer
  NVTE_CHECK(config != nullptr, "Invalid NVTEQuantizationConfig (got NULL)");
  const auto &config_ = *reinterpret_cast<const transformer_engine::QuantizationConfig *>(config);
  switch (attr) {
    case kNVTEQuantizationConfigForcePow2Scales:
      std::memcpy(buf, &config_.force_pow_2_scales, attr_size);
      break;
    case kNVTEQuantizationConfigAmaxEpsilon:
      std::memcpy(buf, &config_.amax_epsilon, attr_size);
      break;
    case kNVTEQuantizationConfigNoopTensor:
      std::memcpy(buf, &config_.noop_tensor, attr_size);
      break;
    default:
      NVTE_ERROR("Unsupported NVTEQuantizationConfigAttribute (got ", static_cast<int>(attr), ")");
  }
}

void nvte_set_quantization_config_attribute(NVTEQuantizationConfig config,
                                            NVTEQuantizationConfigAttribute attr, const void *buf,
                                            size_t size_in_bytes) {
  // Check attribute and buffer
  NVTE_CHECK(attr < kNVTEQuantizationConfigNumAttributes,
             "Invalid NVTEQuantizationConfigAttribute (got ", static_cast<int>(attr), ")");
  const auto &attr_size = transformer_engine::QuantizationConfig::attr_sizes[attr];
  NVTE_CHECK(size_in_bytes >= attr_size,
             "Buffer is too small for quantization config attribute "
             "(attribute ",
             static_cast<int>(attr), " needs ", attr_size, " bytes, but buffer has ", size_in_bytes,
             " bytes)");
  NVTE_CHECK(buf != nullptr, "Invalid buffer (got NULL)");

  // Read from buffer
  NVTE_CHECK(config != nullptr, "Invalid NVTEQuantizationConfig (got NULL)");
  auto &config_ = *reinterpret_cast<transformer_engine::QuantizationConfig *>(config);
  switch (attr) {
    case kNVTEQuantizationConfigForcePow2Scales:
      std::memcpy(&config_.force_pow_2_scales, buf, attr_size);
      break;
    case kNVTEQuantizationConfigAmaxEpsilon:
      std::memcpy(&config_.amax_epsilon, buf, attr_size);
      break;
    case kNVTEQuantizationConfigNoopTensor:
      std::memcpy(&config_.noop_tensor, buf, attr_size);
      break;
    default:
      NVTE_ERROR("Unsupported NVTEQuantizationConfigAttribute (got ", static_cast<int>(attr), ")");
  }
}

void nvte_destroy_quantization_config(NVTEQuantizationConfig config) {
  if (config != nullptr) {
    delete reinterpret_cast<transformer_engine::QuantizationConfig *>(config);
  }
}

int nvte_is_non_tn_fp8_gemm_supported() {
  int deviceComputeCapability =
      transformer_engine::cuda::sm_arch(transformer_engine::cuda::current_device());

  // Note: this is temporary restriction and should be lifted in the future.
  // (remove the note once it's done.)
  return (deviceComputeCapability >= 100 && deviceComputeCapability < 120) ||
         deviceComputeCapability >= 130;
}
