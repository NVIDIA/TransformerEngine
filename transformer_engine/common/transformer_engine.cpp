/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/transformer_engine.h>

#include <atomic>
#include <climits>
#include <cstring>
#include <iostream>
#include <mutex>
#include <utility>

#include "common.h"
#include "common/util/cuda_runtime.h"
#include "common/util/logging.h"

namespace transformer_engine {

size_t typeToNumBits(const DType type) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(type, T,
                                     return TypeInfo<T>::size;);  // NOLINT(*)
}

size_t typeToSize(const DType type) {
  NVTE_CHECK(type != DType::kFloat4E2M1, "typeToSize() Does not support FP4 data type.");
  return typeToNumBits(type) / 8;
}

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
    case DType::kFloat4E2M1:
      return "Float4E2M1";
    case DType::kInt16:
      return "Int16";
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
    case NVTE_BLOCK_SCALING_1D:
      return "NVTE_BLOCK_SCALING_1D";
    case NVTE_BLOCK_SCALING_2D:
      return "NVTE_BLOCK_SCALING_2D";
    case NVTE_NVFP4_1D_SCALING:
      return "NVTE_NVFP4_1D_SCALING";
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
      const size_t block_size_rowwise = 32;
      const size_t block_size_colwise = 32;

      if (t.has_data()) {
        alignment = block_alignment[0];
        expected_x =
            DIVUP(DIVUP(t.flat_first_dim(), static_cast<size_t>(1)), alignment) * alignment;
        alignment = block_alignment[1];
        expected_y =
            DIVUP(DIVUP(t.flat_last_dim(), static_cast<size_t>(block_size_rowwise)), alignment) *
            alignment;

        const auto &expected = std::vector<size_t>{expected_x, expected_y};
        NVTE_CHECK(t.scale_inv.shape == expected, "Tensor \"", name,
                   "\" has invalid scale_inv shape (expected ", expected, ", got ",
                   t.scale_inv.shape, ")");
      }
      if (t.has_columnwise_data()) {
        alignment = block_alignment[1];
        expected_x =
            DIVUP(DIVUP(t.flat_first_dim(), static_cast<size_t>(block_size_colwise)), alignment) *
            alignment;
        alignment = block_alignment[0];
        expected_y = DIVUP(DIVUP(t.flat_last_dim(), static_cast<size_t>(1)), alignment) * alignment;

        const auto &expected = std::vector<size_t>{expected_x, expected_y};
        NVTE_CHECK(t.columnwise_scale_inv.shape == expected, "Tensor \"", name,
                   "\"  has invalid columnwise_scale_inv shape (expected ", expected, ", got ",
                   t.columnwise_scale_inv.shape, ")");
      }
    } else if (t.scaling_mode == NVTE_NVFP4_1D_SCALING) {
      if (t.has_data()) {
        const size_t expected_y = DIVUP_TO_MULTIPLE(t.flat_first_dim(), 128);
        const size_t expected_x = DIVUP_TO_MULTIPLE(DIVUP(t.flat_last_dim(), 16lu), 4);
        const auto &expected = std::vector<size_t>{expected_y, expected_x};
        NVTE_CHECK(t.scale_inv.shape == expected, "Tensor \"", name,
                   "\" has invalid scale_inv shape (expected ", expected, ", got ",
                   t.scale_inv.shape, ")");
      }
      if (t.has_columnwise_data()) {
        const size_t expected_y = DIVUP_TO_MULTIPLE(t.flat_last_dim(), 128);
        const size_t expected_x = DIVUP_TO_MULTIPLE(DIVUP(t.flat_first_dim(), 16lu), 4);
        const auto &expected = std::vector<size_t>{expected_y, expected_x};
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
  } else if (is_fp4_dtype(type)) {
    // TODO(ksivaman): Fix this to check for amaxes and other details.
    // For now only needed for swizzle.
    if (t.has_data()) {
      NVTE_CHECK(t.scale_inv.dptr != nullptr, "FP4 scaling factor input ", name,
                 "_scale_inverse must be allocated");
      NVTE_CHECK(t.scale_inv.dtype == DType::kFloat8E4M3, "FP4 scaling factor input ", name,
                 "_scale_inverse has invalid dtype "
                 "(expected DType::kFloat8E4M3, got ",
                 to_string(t.scale_inv.dtype), ")");
    }
    if (t.has_columnwise_data()) {
      NVTE_CHECK(t.columnwise_scale_inv.dptr != nullptr, "FP4 scaling factor input ", name,
                 "_columnwise_scale_inverse must be allocated");
      NVTE_CHECK(t.columnwise_scale_inv.dtype == DType::kFloat8E4M3, "FP8 scaling factor input ",
                 name,
                 "_columnwise_scale_inverse has invalid dtype "
                 "(expected DType::kFloat8E4M3, got ",
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
  } else if (is_fp4_dtype(type)) {
    // FP4 output needs to have the scale_inv
    if (t.has_data()) {
      NVTE_CHECK(t.scale_inv.dptr != nullptr, "FP4 scaling factor output ", name,
                 "_scale_inverse must be allocated");
      NVTE_CHECK(t.scale_inv.dtype == DType::kFloat8E4M3, "FP4 scaling factor output ", name,
                 "_scale_inverse has invalid dtype "
                 "(expected Float8E4M3, got ",
                 to_string(t.scale_inv.dtype), ")");
    }
    if (t.has_columnwise_data()) {
      NVTE_CHECK(t.columnwise_scale_inv.dptr != nullptr, "FP4 scaling factor output ", name,
                 "_columnwise_scale_inverse must be allocated");
      NVTE_CHECK(t.columnwise_scale_inv.dtype == DType::kFloat8E4M3, "FP4 scaling factor output ",
                 name,
                 "_columnwise_scale_inverse has invalid dtype "
                 "(expected Float8E4M3, got ",
                 to_string(t.columnwise_scale_inv.dtype), ")");
    }
  } else {
    NVTE_CHECK(t.scale.dptr == nullptr, "Scale is not supported for non-FP8 output ", name);
    // Unfused quant with level 2 nvfp4 scaling will produce high precision tensors with amax.
    // NVTE_CHECK(t.amax.dptr == nullptr, "Amax is not supported for non-FP8 output ", name);
    NVTE_CHECK(t.scale_inv.dptr == nullptr, "Scale_inv is not supported for non-FP8 output ", name);
    NVTE_CHECK(t.columnwise_scale_inv.dptr == nullptr,
               "Scale_inv is not supported for non-FP8 input ", name);
  }

  if (!allow_empty) {
    NVTE_CHECK(t.has_data() || t.has_columnwise_data(), "Output ", name, " is not allocated!");
  }

  CheckScaleTensorShape(t, name);
}

inline std::pair<size_t, size_t> get_block_size(NVTEScalingMode scaling_mode, bool is_colwise) {
  size_t block_x = 1, block_y = 1;
  if (scaling_mode == NVTE_MXFP8_1D_SCALING) {
    block_y = 32;
  } else if (scaling_mode == NVTE_NVFP4_1D_SCALING) {
    block_y = 16;
  } else {
    NVTE_ERROR("Unsupported scaling_mode = ", static_cast<int>(scaling_mode));
  }
  if (is_colwise) {
    return {block_y, block_x};
  } else {
    return {block_x, block_y};
  }
}

std::pair<size_t, size_t> get_block_scale_shape(NVTEScalingMode scaling_mode, size_t dim_1,
                                                size_t dim_2, bool is_colwise) {
  auto [block_dim_1, block_dim_2] = get_block_size(scaling_mode, is_colwise);
  auto alignment_dim_1 = is_colwise ? 4 : 128;
  auto alignment_dim_2 = is_colwise ? 128 : 4;

  NVTE_CHECK(dim_1 % block_dim_1 == 0, "dim_1 must be divisble by %zu (got %zu)", block_dim_1,
             dim_1);
  NVTE_CHECK(dim_2 % block_dim_2 == 0, "dim_2 must be divisble by %zu (got %zu)", block_dim_2,
             dim_2);
  size_t scale_dim_1 = DIVUP((dim_1 / block_dim_1), alignment_dim_1) * alignment_dim_1;
  size_t scale_dim_2 = DIVUP((dim_2 / block_dim_2), alignment_dim_2) * alignment_dim_2;

  return {scale_dim_1, scale_dim_2};
}

void CheckGroupedTensorShape(const GroupedTensor &t, const std::string &name) {
  // Check all the non empty fields for the same num_tensors
  const size_t expected_num_tensors = t.num_tensors();
  NVTE_CHECK(expected_num_tensors > 0, "Grouped tensor ", name, " has no tensors!");

  // Validate all allocated fields have matching num_tensors
  if (t.has_data()) {
    NVTE_CHECK(t.data.num_tensors == expected_num_tensors, "Grouped tensor ", name,
               " data.num_tensors (", t.data.num_tensors, ") doesn't match expected num_tensors (",
               expected_num_tensors, ")");
  }
  if (t.has_columnwise_data()) {
    NVTE_CHECK(t.columnwise_data.num_tensors == expected_num_tensors, "Grouped tensor ", name,
               " columnwise_data.num_tensors (", t.columnwise_data.num_tensors,
               ") doesn't match expected num_tensors (", expected_num_tensors, ")");
  }
  if (t.scale_inv.has_data()) {
    NVTE_CHECK(t.scale_inv.num_tensors == expected_num_tensors, "Grouped tensor ", name,
               " scale_inv.num_tensors (", t.scale_inv.num_tensors,
               ") doesn't match expected num_tensors (", expected_num_tensors, ")");
  }
  if (t.columnwise_scale_inv.has_data()) {
    NVTE_CHECK(t.columnwise_scale_inv.num_tensors == expected_num_tensors, "Grouped tensor ", name,
               " columnwise_scale_inv.num_tensors (", t.columnwise_scale_inv.num_tensors,
               ") doesn't match expected num_tensors (", expected_num_tensors, ")");
  }
  if (t.amax.has_data()) {
    NVTE_CHECK(t.amax.num_tensors == expected_num_tensors, "Grouped tensor ", name,
               " amax.num_tensors (", t.amax.num_tensors, ") doesn't match expected num_tensors (",
               expected_num_tensors, ")");
  }
  if (t.columnwise_amax.has_data()) {
    NVTE_CHECK(t.columnwise_amax.num_tensors == expected_num_tensors, "Grouped tensor ", name,
               " columnwise_amax.num_tensors (", t.columnwise_amax.num_tensors,
               ") doesn't match expected num_tensors (", expected_num_tensors, ")");
  }
  if (t.scale.has_data()) {
    NVTE_CHECK(t.scale.num_tensors == expected_num_tensors, "Grouped tensor ", name,
               " scale.num_tensors (", t.scale.num_tensors,
               ") doesn't match expected num_tensors (", expected_num_tensors, ")");
  }

  // Check that shape information is provided and consistent
  NVTE_CHECK(t.first_dims.dptr != nullptr, "Grouped tensor ", name,
             " must have first_dims allocated");
  NVTE_CHECK(t.second_dims.dptr != nullptr, "Grouped tensor ", name,
             " must have second_dims allocated");

  // Ensure shape arrays are 1D
  NVTE_CHECK(t.first_dims.shape.size() == 1, "Grouped tensor ", name,
             " first_dims must be 1D (got ", t.first_dims.shape.size(), "D)");
  NVTE_CHECK(t.second_dims.shape.size() == 1, "Grouped tensor ", name,
             " second_dims must be 1D (got ", t.second_dims.shape.size(), "D)");

  // Ensure shape arrays have correct dtype (size_t = Int64)
  NVTE_CHECK(t.first_dims.dtype == DType::kInt64, "Grouped tensor ", name,
             " first_dims must have dtype Int64 (got ", to_string(t.first_dims.dtype), ")");
  NVTE_CHECK(t.second_dims.dtype == DType::kInt64, "Grouped tensor ", name,
             " second_dims must have dtype Int64 (got ", to_string(t.second_dims.dtype), ")");

  NVTE_CHECK(!t.first_dims.shape.empty() && t.first_dims.shape[0] > 0, "Grouped tensor ", name,
             " first_dims has invalid shape");
  NVTE_CHECK(!t.second_dims.shape.empty() && t.second_dims.shape[0] > 0, "Grouped tensor ", name,
             " second_dims has invalid shape");
  NVTE_CHECK(t.first_dims.shape[0] == t.second_dims.shape[0], "Grouped tensor ", name,
             " first_dims and second_dims must have the same size (got ", t.first_dims.shape[0],
             " vs ", t.second_dims.shape[0], ")");
  NVTE_CHECK(t.first_dims.shape[0] == t.num_tensors(), "Grouped tensor ", name,
             " shape arrays size (", t.first_dims.shape[0], ") doesn't match num_tensors (",
             t.num_tensors(), ")");

  // Check cumulative_tensor_sizes if it's set (optional but must be consistent)
  if (t.has_cumulative_tensor_sizes()) {
    // Ensure cumulative_tensor_sizes is 1D
    NVTE_CHECK(t.cumulative_tensor_sizes.shape.size() == 1, "Grouped tensor ", name,
               " cumulative_tensor_sizes must be 1D (got ", t.cumulative_tensor_sizes.shape.size(),
               "D)");
    // Ensure cumulative_tensor_sizes has correct dtype (size_t = Int64)
    NVTE_CHECK(t.cumulative_tensor_sizes.dtype == DType::kInt64, "Grouped tensor ", name,
               " cumulative_tensor_sizes must have dtype Int64 (got ",
               to_string(t.cumulative_tensor_sizes.dtype), ")");
    NVTE_CHECK(!t.cumulative_tensor_sizes.shape.empty() && t.cumulative_tensor_sizes.shape[0] > 1,
               "Grouped tensor ", name, " cumulative_tensor_sizes has invalid shape");
    NVTE_CHECK(t.cumulative_tensor_sizes.shape[0] == t.first_dims.shape[0] + 1, "Grouped tensor ",
               name, " cumulative_tensor_sizes size (", t.cumulative_tensor_sizes.shape[0],
               ") must be first_dims size + 1 (", t.first_dims.shape[0] + 1, ")");
  }

  // Additional validation for scale_inv based on scaling mode
  if (is_tensor_scaling(t.scaling_mode)) {
    NVTE_CHECK(t.scale.num_elements == expected_num_tensors, "Grouped tensor ", name,
               " scale.num_elements (", t.scale.num_elements,
               ") doesn't match expected num_tensors (", expected_num_tensors, ")");
    NVTE_CHECK(t.amax.num_elements == expected_num_tensors, "Grouped tensor ", name,
               " amax.num_elements (", t.amax.num_elements,
               ") doesn't match expected num_tensors (", expected_num_tensors, ")");
  } else if (t.scaling_mode == NVTE_NVFP4_1D_SCALING || t.scaling_mode == NVTE_MXFP8_1D_SCALING) {
    if (t.has_data()) {
      NVTE_CHECK(
          !t.scaling_mode == NVTE_NVFP4_1D_SCALING || t.amax.num_elements == expected_num_tensors,
          "Grouped tensor ", name, " amax.num_elements (", t.amax.num_elements,
          ") doesn't match expected num_tensors (", expected_num_tensors, ")");
      auto scale_shape = get_block_scale_shape(t.scaling_mode, t.data.sum_first_dims,
                                               t.data.sum_second_dims, false);
      NVTE_CHECK(t.scale_inv.sum_first_dims == scale_shape.first &&
                     t.scale_inv.sum_second_dims == scale_shape.second,
                 "Grouped tensor ", name, " scale_inv.sum_first_dims (", t.scale_inv.sum_first_dims,
                 ") and scale_inv.sum_second_dims (", t.scale_inv.sum_second_dims,
                 ") don't match expected scale_shape.first (", scale_shape.first,
                 ") and scale_shape.second (", scale_shape.second, ")");
    }
    if (t.has_columnwise_data()) {
      NVTE_CHECK(!t.scaling_mode == NVTE_NVFP4_1D_SCALING ||
                     t.columnwise_amax.num_elements == expected_num_tensors,
                 "Grouped tensor ", name, " columnwise_amax.num_elements (",
                 t.columnwise_amax.num_elements, ") doesn't match expected num_tensors (",
                 expected_num_tensors, ")");
      auto scale_shape = get_block_scale_shape(t.scaling_mode, t.columnwise_data.sum_first_dims,
                                               t.columnwise_data.sum_second_dims, true);
      NVTE_CHECK(
          t.columnwise_scale_inv.sum_first_dims == scale_shape.first &&
              t.columnwise_scale_inv.sum_second_dims == scale_shape.second,
          "Grouped tensor ", name, " columnwise_scale_inv.sum_first_dims (",
          t.columnwise_scale_inv.sum_first_dims, ") and columnwise_scale_inv.sum_second_dims (",
          t.columnwise_scale_inv.sum_second_dims, ") don't match expected scale_shape.first (",
          scale_shape.first, ") and scale_shape.second (", scale_shape.second, ")");
    }
  }
}

void CheckInputGroupedTensor(const GroupedTensor &t, const std::string &name) {
  NVTE_CHECK(t.has_data() || t.has_columnwise_data(), "Input grouped tensor ", name,
             " is not allocated!");

  const DType type = t.dtype();
  if (is_fp8_dtype(type)) {
    // FP8 input needs to have scale_inv
    if (t.has_data()) {
      NVTE_CHECK(t.scale_inv.has_data(), "FP8 scaling factor input ", name,
                 "_scale_inverse must be allocated");
      NVTE_CHECK(t.scale_inv.dtype == DType::kFloat32 || t.scale_inv.dtype == DType::kFloat8E8M0,
                 "FP8 scaling factor input ", name,
                 "_scale_inverse has invalid dtype "
                 "(expected Float32 or Byte, got ",
                 to_string(t.scale_inv.dtype), ")");
    }
    if (t.has_columnwise_data()) {
      NVTE_CHECK(t.columnwise_scale_inv.has_data(), "FP8 scaling factor input ", name,
                 "_columnwise_scale_inverse must be allocated");
      NVTE_CHECK(t.columnwise_scale_inv.dtype == DType::kFloat32 ||
                     t.columnwise_scale_inv.dtype == DType::kFloat8E8M0,
                 "FP8 scaling factor input ", name,
                 "_columnwise_scale_inverse has invalid dtype "
                 "(expected Float32 or Byte, got ",
                 to_string(t.columnwise_scale_inv.dtype), ")");
    }
  } else if (is_fp4_dtype(type)) {
    if (t.has_data()) {
      NVTE_CHECK(t.scale_inv.has_data(), "FP4 scaling factor input ", name,
                 "_scale_inverse must be allocated");
      NVTE_CHECK(t.scale_inv.dtype == DType::kFloat8E4M3, "FP4 scaling factor input ", name,
                 "_scale_inverse has invalid dtype "
                 "(expected DType::kFloat8E4M3, got ",
                 to_string(t.scale_inv.dtype), ")");
    }
    if (t.has_columnwise_data()) {
      NVTE_CHECK(t.columnwise_scale_inv.has_data(), "FP4 scaling factor input ", name,
                 "_columnwise_scale_inverse must be allocated");
      NVTE_CHECK(t.columnwise_scale_inv.dtype == DType::kFloat8E4M3, "FP4 scaling factor input ",
                 name,
                 "_columnwise_scale_inverse has invalid dtype "
                 "(expected DType::kFloat8E4M3, got ",
                 to_string(t.columnwise_scale_inv.dtype), ")");
    }
  } else {
    NVTE_CHECK(!t.scale.has_data(), "Scale is not supported for non-FP8 input ", name);
    NVTE_CHECK(!t.amax.has_data(), "Amax is not supported for non-FP8 input ", name);
    NVTE_CHECK(!t.scale_inv.has_data(), "Scale_inv is not supported for non-FP8 input ", name);
    NVTE_CHECK(!t.columnwise_scale_inv.has_data(), "Scale_inv is not supported for non-FP8 input ",
               name);
  }

  CheckGroupedTensorShape(t, name);
}

void CheckOutputGroupedTensor(const GroupedTensor &t, const std::string &name, bool allow_empty) {
  NVTE_CHECK(t.num_tensors() > 0, "Output grouped tensor ", name, " has no tensors!");
  if (!allow_empty) {
    NVTE_CHECK(t.has_data() || t.has_columnwise_data(), "Output grouped tensor ", name,
               " is not allocated!");
  }

  const DType type = t.dtype();
  if (is_fp8_dtype(type)) {
    // FP8 output needs to have scale_inv and (if delayed scaling) amax
    if (t.scaling_mode == NVTE_DELAYED_TENSOR_SCALING && t.amax.has_data()) {
      NVTE_CHECK(t.amax.dtype == DType::kFloat32, "Invalid amax dtype (expected ",
                 to_string(DType::kFloat32), ", got ", to_string(t.amax.dtype), ")");
    }
    if (t.has_data()) {
      NVTE_CHECK(t.scale_inv.has_data(), "FP8 scaling factor output ", name,
                 "_scale_inverse must be allocated");
      NVTE_CHECK(t.scale_inv.dtype == DType::kFloat32 || t.scale_inv.dtype == DType::kFloat8E8M0,
                 "FP8 scaling factor output ", name,
                 "_scale_inverse has invalid dtype "
                 "(expected Float32 or Float8E8M0, got ",
                 to_string(t.scale_inv.dtype), ")");
    }
    if (t.has_columnwise_data()) {
      NVTE_CHECK(t.columnwise_scale_inv.has_data(), "FP8 scaling factor output ", name,
                 "_columnwise_scale_inverse must be allocated");
      NVTE_CHECK(t.columnwise_scale_inv.dtype == DType::kFloat32 ||
                     t.columnwise_scale_inv.dtype == DType::kFloat8E8M0,
                 "FP8 scaling factor output ", name,
                 "_columnwise_scale_inverse has invalid dtype "
                 "(expected Float32 or Float8E8M0, got ",
                 to_string(t.columnwise_scale_inv.dtype), ")");
    }
  } else if (is_fp4_dtype(type)) {
    if (t.has_data()) {
      NVTE_CHECK(t.scale_inv.has_data(), "FP4 scaling factor output ", name,
                 "_scale_inverse must be allocated");
      NVTE_CHECK(t.scale_inv.dtype == DType::kFloat8E4M3, "FP4 scaling factor output ", name,
                 "_scale_inverse has invalid dtype "
                 "(expected Float8E4M3, got ",
                 to_string(t.scale_inv.dtype), ")");
    }
    if (t.has_columnwise_data()) {
      NVTE_CHECK(t.columnwise_scale_inv.has_data(), "FP4 scaling factor output ", name,
                 "_columnwise_scale_inverse must be allocated");
      NVTE_CHECK(t.columnwise_scale_inv.dtype == DType::kFloat8E4M3, "FP4 scaling factor output ",
                 name,
                 "_columnwise_scale_inverse has invalid dtype "
                 "(expected Float8E4M3, got ",
                 to_string(t.columnwise_scale_inv.dtype), ")");
    }
  } else {
    NVTE_CHECK(!t.scale.has_data(), "Scale is not supported for non-FP8 output ", name);
    NVTE_CHECK(!t.scale_inv.has_data(), "Scale_inv is not supported for non-FP8 output ", name);
    NVTE_CHECK(!t.columnwise_scale_inv.has_data(), "Scale_inv is not supported for non-FP8 output ",
               name);
  }

  CheckGroupedTensorShape(t, name);
}

class TensorAllocator {
 public:
  static TensorAllocator &instance() {
    static TensorAllocator allocator;
    return allocator;
  }

  ~TensorAllocator() {}

  NVTETensor Allocate(NVTEScalingMode mode) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!free_list.empty()) {
      uintptr_t index = free_list.back();
      NVTETensor ret = reinterpret_cast<NVTETensor>(index);
      free_list.pop_back();
      if (debug) {
        std::cout << "Allocated " << index
                  << " from free list. Free list size: " << free_list.size() << " and capacity "
                  << free_list.capacity() << std::endl;
      }
      // 1-based indexing
      memory[index - 1].scaling_mode = mode;
      return ret;
    }
    if (memory.size() < memory.capacity()) {
      memory.emplace_back();
      Tensor &t = memory.back();
      size = memory.size();
      // 1-based indexing
      uintptr_t index = memory.size();
      if (debug) {
        std::cout << "Allocated " << index << ". Memory size: " << memory.size() << " and capacity "
                  << memory.capacity() << std::endl;
      }
      t.scaling_mode = mode;
      t.nvte_tensor = reinterpret_cast<NVTETensor>(index);
      return reinterpret_cast<NVTETensor>(index);
    }
    NVTE_ERROR("Cannot allocate a new NVTETensor. Maximum number of tensors reached: ",
               MAX_TENSOR_NUM, ". There is probably a memory leak in your application.");
  }

  void Free(NVTETensor t) {
    std::lock_guard<std::mutex> lock(mutex);
    uintptr_t index = reinterpret_cast<uintptr_t>(t);
    if (index == 0) return;
    NVTE_CHECK(index <= memory.size(), "Invalid tensor.");
    free_list.push_back(index);
    // Clean up
    memory[index - 1].clear();
    if (debug) {
      std::cout << "Freed " << index << ". Free list size: " << free_list.size() << " and capacity "
                << free_list.capacity() << std::endl;
    }
  }

  void Free(NVTETensor *t, size_t N) {
    std::lock_guard<std::mutex> lock(mutex);
    for (size_t i = 0; i < N; ++i) {
      uintptr_t index = reinterpret_cast<uintptr_t>(t[i]);
      if (index == 0) continue;
      NVTE_CHECK(index <= memory.size(), "Invalid tensor.");
      free_list.push_back(index);
      // Clean up
      memory[index - 1].clear();
    }
    if (debug) {
      std::cout << "Freed range of" << N << " tensors. Free list size: " << free_list.size()
                << " and capacity " << free_list.capacity() << std::endl;
    }
  }

  Tensor *convertNVTETensor(NVTETensor t) {
    uintptr_t index = reinterpret_cast<uintptr_t>(t);
    // 1-based indexing to enable 0-initialization of NVTETensor
    // to be invalid tensor
    static_assert(nullptr == 0);
    if (index != 0 && index <= size) {
      return &(memory[index - 1]);
    }
    return nullptr;
  }

  void setDebug(bool debug) {
    std::lock_guard<std::mutex> lock(mutex);
    this->debug = debug;
  }

 private:
  TensorAllocator() {
    std::lock_guard<std::mutex> lock(mutex);
    memory.reserve(MAX_TENSOR_NUM);
  }

  std::mutex mutex;
  std::atomic<size_t> size;
  // Allocate at most 20 MB for tensors
  // Should be replaced by virtual memory allocation
  const size_t MAX_TENSOR_NUM = 20 * 1024 * 1024 / sizeof(Tensor);
  std::vector<uintptr_t> free_list;
  std::vector<Tensor> memory;
  bool debug = false;
};

Tensor *convertNVTETensor(const NVTETensor t) {
  return TensorAllocator::instance().convertNVTETensor(t);
}

Tensor *convertNVTETensorCheck(const NVTETensor t) {
  Tensor *ptr = TensorAllocator::instance().convertNVTETensor(t);
  NVTE_CHECK(ptr != nullptr, "Invalid tensor.");
  return ptr;
}

// GroupedTensor allocator - similar pattern to TensorAllocator
class GroupedTensorAllocator {
 public:
  static GroupedTensorAllocator &instance() {
    static GroupedTensorAllocator allocator;
    return allocator;
  }

  ~GroupedTensorAllocator() {}

  NVTEGroupedTensor Allocate(NVTEScalingMode mode) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!free_list.empty()) {
      uintptr_t index = free_list.back();
      NVTEGroupedTensor ret = reinterpret_cast<NVTEGroupedTensor>(index);
      free_list.pop_back();
      // 1-based indexing
      memory[index - 1].scaling_mode = mode;
      return ret;
    }
    if (memory.size() < memory.capacity()) {
      memory.emplace_back();
      GroupedTensor &t = memory.back();
      size = memory.size();
      // 1-based indexing
      uintptr_t index = memory.size();
      t.scaling_mode = mode;
      t.nvte_tensor = reinterpret_cast<NVTEGroupedTensor>(index);
      return reinterpret_cast<NVTEGroupedTensor>(index);
    }
    NVTE_ERROR(
        "Cannot allocate a new NVTEGroupedTensor. Maximum number of grouped tensors reached: ",
        MAX_GROUPED_TENSOR_NUM, ". There is probably a memory leak in your application.");
  }

  void Free(NVTEGroupedTensor t) {
    std::lock_guard<std::mutex> lock(mutex);
    uintptr_t index = reinterpret_cast<uintptr_t>(t);
    if (index == 0) return;
    NVTE_CHECK(index <= memory.size(), "Invalid grouped tensor.");
    free_list.push_back(index);
    // Clean up
    memory[index - 1].clear();
  }

  GroupedTensor *convertNVTEGroupedTensor(NVTEGroupedTensor t) {
    uintptr_t index = reinterpret_cast<uintptr_t>(t);
    // 1-based indexing to enable 0-initialization of NVTEGroupedTensor
    // to be invalid tensor
    static_assert(nullptr == 0);
    if (index != 0 && index <= size) {
      return &(memory[index - 1]);
    }
    return nullptr;
  }

 private:
  GroupedTensorAllocator() {
    std::lock_guard<std::mutex> lock(mutex);
    memory.reserve(MAX_GROUPED_TENSOR_NUM);
  }

  std::mutex mutex;
  std::atomic<size_t> size;
  // Allocate at most 5 MB for grouped tensors
  const size_t MAX_GROUPED_TENSOR_NUM = 5 * 1024 * 1024 / sizeof(GroupedTensor);
  std::vector<uintptr_t> free_list;
  std::vector<GroupedTensor> memory;
};

GroupedTensor *convertNVTEGroupedTensor(const NVTEGroupedTensor t) {
  return GroupedTensorAllocator::instance().convertNVTEGroupedTensor(t);
}

GroupedTensor *convertNVTEGroupedTensorCheck(const NVTEGroupedTensor t) {
  GroupedTensor *ptr = GroupedTensorAllocator::instance().convertNVTEGroupedTensor(t);
  NVTE_CHECK(ptr != nullptr, "Invalid grouped tensor.");
  return ptr;
}

}  // namespace transformer_engine

NVTETensor nvte_create_tensor(NVTEScalingMode scaling_mode) {
  NVTETensor ret = transformer_engine::TensorAllocator::instance().Allocate(scaling_mode);
  return ret;
}

void nvte_destroy_tensor(NVTETensor tensor) {
  transformer_engine::TensorAllocator::instance().Free(tensor);
}

void nvte_destroy_tensors(NVTETensor *tensors, size_t N) {
  transformer_engine::TensorAllocator::instance().Free(tensors, N);
}

NVTEDType nvte_tensor_type(const NVTETensor tensor) {
  auto *t = transformer_engine::convertNVTETensor(tensor);
  if (t == nullptr) return kNVTEFloat32;
  return static_cast<NVTEDType>(t->dtype());
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
  auto *t = transformer_engine::convertNVTETensor(tensor);
  if (t == nullptr) {
    NVTE_ERROR("Invalid tensor");
  }

  // Determine tensor shape depending on tensor format
  const std::vector<size_t> &shape = t->shape();

  return nvte_make_shape(shape.data(), shape.size());
}

NVTEShape nvte_tensor_columnwise_shape(const NVTETensor tensor) {
  auto *t = transformer_engine::convertNVTETensor(tensor);
  if (t == nullptr) {
    NVTE_ERROR("Invalid tensor");
  }
  const std::vector<size_t> &shape = t->columnwise_data.shape;
  return nvte_make_shape(shape.data(), shape.size());
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

size_t nvte_tensor_element_size_bits(const NVTETensor tensor) {
  auto *t = transformer_engine::convertNVTETensor(tensor);
  if (t == nullptr) return 8 * sizeof(float);
  return transformer_engine::typeToNumBits(t->dtype());
}

size_t nvte_tensor_element_size(const NVTETensor tensor) {
  auto *t = transformer_engine::convertNVTETensor(tensor);
  if (t == nullptr) return sizeof(float);
  NVTE_CHECK(!is_fp4_dtype(t->dtype()),
             "For FP4 type please use the nvte_tensor_element_size_bits.");
  return nvte_tensor_element_size_bits(tensor) / 8;
}

size_t nvte_tensor_size_bytes(const NVTETensor tensor) {
  auto *t = transformer_engine::convertNVTETensor(tensor);
  if (t == nullptr) return 0;
  return (nvte_tensor_numel(tensor) * nvte_tensor_element_size_bits(tensor)) / 8;
}

void *nvte_tensor_data(const NVTETensor tensor) {
  auto *t = transformer_engine::convertNVTETensor(tensor);
  if (t == nullptr) return nullptr;
  return t->data.dptr;
}

void *nvte_tensor_columnwise_data(const NVTETensor tensor) {
  auto *t = transformer_engine::convertNVTETensor(tensor);
  if (t == nullptr) return nullptr;
  return t->columnwise_data.dptr;
}

float *nvte_tensor_amax(const NVTETensor tensor) {
  auto *t = transformer_engine::convertNVTETensor(tensor);
  if (t == nullptr) return nullptr;
  NVTE_CHECK(t->amax.dtype == transformer_engine::DType::kFloat32,
             "Tensor's amax must have Float32 type!");
  return reinterpret_cast<float *>(t->amax.dptr);
}

float *nvte_tensor_scale(const NVTETensor tensor) {
  auto *t = transformer_engine::convertNVTETensor(tensor);
  if (t == nullptr) return nullptr;
  NVTE_CHECK(t->scale.dtype == transformer_engine::DType::kFloat32,
             "Tensor's scale must have Float32 type!");
  return reinterpret_cast<float *>(t->scale.dptr);
}

float *nvte_tensor_scale_inv(const NVTETensor tensor) {
  auto *t = transformer_engine::convertNVTETensor(tensor);
  if (t == nullptr) return nullptr;
  return reinterpret_cast<float *>(t->scale_inv.dptr);
}

void *nvte_tensor_columnwise_scale_inv(const NVTETensor tensor) {
  auto *t = transformer_engine::convertNVTETensor(tensor);
  if (t == nullptr) return nullptr;
  return t->columnwise_scale_inv.dptr;
}

NVTEShape nvte_tensor_scale_inv_shape(const NVTETensor tensor) {
  auto *t = transformer_engine::convertNVTETensor(tensor);
  if (t == nullptr) {
    return nvte_make_shape(nullptr, 0);
  }
  return nvte_make_shape(t->scale_inv.shape.data(), t->scale_inv.shape.size());
}

void nvte_set_tensor_param(NVTETensor *tensor, NVTETensorParam param_name,
                           const NVTEBasicTensor *param) {
  NVTE_CHECK(tensor != nullptr, "Tensor pointer can't be NULL.");
  auto *t = transformer_engine::convertNVTETensor(*tensor);
  NVTE_CHECK(t != nullptr, "Tensor is not allocated.");
  switch (param_name) {
    case kNVTERowwiseData:
      t->data = *param;
      break;
    case kNVTEColumnwiseData:
      t->columnwise_data = *param;
      break;
    case kNVTEScale:
      t->scale = *param;
      break;
    case kNVTEAmax:
      t->amax = *param;
      break;
    case kNVTERowwiseScaleInv:
      t->scale_inv = *param;
      break;
    case kNVTEColumnwiseScaleInv:
      t->columnwise_scale_inv = *param;
      break;
    case kNVTEColumnwiseAmax:
      t->columnwise_amax = *param;
      break;
    default:
      NVTE_ERROR("Unknown tensor parameter!");
  }
}

NVTEBasicTensor nvte_get_tensor_param(const NVTETensor tensor, NVTETensorParam param_name) {
  if (tensor == nullptr) {
    return {nullptr, kNVTEFloat32, nvte_make_shape(nullptr, 0)};
  }
  const auto &t = *transformer_engine::convertNVTETensorCheck(tensor);
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
    case kNVTEColumnwiseAmax:
      return t.columnwise_amax;
    default:
      NVTE_ERROR("Unknown tensor parameter!");
  }
}

NVTEScalingMode nvte_tensor_scaling_mode(const NVTETensor tensor) {
  if (tensor == nullptr) {
    return NVTE_DELAYED_TENSOR_SCALING;
  }
  const auto &t = *transformer_engine::convertNVTETensorCheck(tensor);
  return t.scaling_mode;
}

void nvte_tensor_pack_create(NVTETensorPack *pack) {
  for (int i = 0; i < pack->MAX_SIZE; i++) {
    pack->tensors[i] =
        transformer_engine::TensorAllocator::instance().Allocate(NVTE_DELAYED_TENSOR_SCALING);
  }
}

void nvte_tensor_pack_destroy(NVTETensorPack *pack) {
  transformer_engine::TensorAllocator::instance().Free(pack->tensors, pack->MAX_SIZE);
}

void nvte_zero_tensor(const NVTETensor tensor, cudaStream_t stream) {
  if (tensor == nullptr) return;
  const auto &t = *transformer_engine::convertNVTETensorCheck(tensor);
  // Zero out tensor data if allocated
  if (t.data.dptr != nullptr) {
    const size_t size_in_bytes = nvte_tensor_size_bytes(tensor);
    NVTE_CHECK_CUDA(cudaMemsetAsync(t.data.dptr, 0, size_in_bytes, stream));
  }
  // Set amax to 0 if allocated
  if (t.amax.dptr != nullptr) {
    NVTE_CHECK_CUDA(cudaMemsetAsync(t.amax.dptr, 0, sizeof(float), stream));
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
    case kNVTEQuantizationConfigFloat8BlockScaleTensorFormat:
      std::memcpy(buf, &config_.float8_block_scale_tensor_format, attr_size);
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
    case kNVTEQuantizationConfigFloat8BlockScaleTensorFormat:
      std::memcpy(&config_.float8_block_scale_tensor_format, buf, attr_size);
      break;
    case kNVTEQuantizationConfigRNGState:
      std::memcpy(&config_.rng_state, buf, attr_size);
      break;
    case kNVTEQuantizationConfigNVFP42DQuantization:
      std::memcpy(&config_.nvfp4_2d_quantization, buf, attr_size);
      break;
    case kNVTEQuantizationConfigStochasticRounding:
      std::memcpy(&config_.stochastic_rounding, buf, attr_size);
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

// Grouped Tensor C API implementations
NVTEGroupedTensor nvte_create_grouped_tensor(NVTEScalingMode scaling_mode) {
  NVTEGroupedTensor ret =
      transformer_engine::GroupedTensorAllocator::instance().Allocate(scaling_mode);
  return ret;
}

void nvte_destroy_grouped_tensor(NVTEGroupedTensor tensor) {
  transformer_engine::GroupedTensorAllocator::instance().Free(tensor);
}

void nvte_set_grouped_tensor_param(NVTEGroupedTensor *tensor, NVTEGroupedTensorParam param_name,
                                   const NVTEGroupedTensorInfo *param) {
  NVTE_CHECK(tensor != nullptr, "Grouped tensor pointer can't be NULL.");
  auto *t = transformer_engine::convertNVTEGroupedTensor(*tensor);
  NVTE_CHECK(t != nullptr, "Grouped tensor is not allocated.");
  NVTE_CHECK(param != nullptr, "Grouped tensor info can't be NULL.");

  // Get current num_tensors (may be 0 if this is the first parameter being set)
  const size_t current_num_tensors = t->num_tensors();

  // If num_tensors is already set (> 0), validate consistency
  if (current_num_tensors > 0) {
    NVTE_CHECK(param->num_tensors == current_num_tensors,
               "Number of tensors mismatch: grouped tensor has ", current_num_tensors,
               " but trying to set parameter with ", param->num_tensors);
  }

  // Helper to create SimpleGroupedTensor from NVTEGroupedTensorInfo
  auto create_simple_grouped =
      [](const NVTEGroupedTensorInfo *info) -> transformer_engine::SimpleGroupedTensor {
    if (info->base_dptr != nullptr) {
      return transformer_engine::SimpleGroupedTensor(
          info->base_dptr, static_cast<transformer_engine::DType>(info->dtype), info->num_tensors,
          info->contiguous, info->sum_first_dims, info->sum_second_dims);
    } else if (info->dptr_list != nullptr) {
      std::vector<void *> dptrs(info->dptr_list, info->dptr_list + info->num_tensors);
      return transformer_engine::SimpleGroupedTensor(
          dptrs, static_cast<transformer_engine::DType>(info->dtype), info->num_tensors,
          info->contiguous, info->sum_first_dims, info->sum_second_dims);
    } else {
      return transformer_engine::SimpleGroupedTensor();
    }
  };

  switch (param_name) {
    case kNVTEGroupedRowwiseData:
      t->data = create_simple_grouped(param);
      break;
    case kNVTEGroupedColumnwiseData:
      t->columnwise_data = create_simple_grouped(param);
      break;
    case kNVTEGroupedScale:
      t->scale = create_simple_grouped(param);
      break;
    case kNVTEGroupedAmax:
      t->amax = create_simple_grouped(param);
      break;
    case kNVTEGroupedRowwiseScaleInv:
      t->scale_inv = create_simple_grouped(param);
      break;
    case kNVTEGroupedColumnwiseScaleInv:
      t->columnwise_scale_inv = create_simple_grouped(param);
      break;
    case kNVTEGroupedColumnwiseAmax:
      t->columnwise_amax = create_simple_grouped(param);
      break;
    case kNVTEGroupedFirstDims:
      NVTE_CHECK(param->base_dptr != nullptr, "First dims must have a valid pointer");
      NVTE_CHECK(param->num_tensors > 0, "First dims must have num_tensors > 0");
      // If second_dims is already set, validate consistency
      if (t->second_dims.dptr != nullptr && !t->second_dims.shape.empty()) {
        NVTE_CHECK(t->second_dims.shape[0] == param->num_tensors, "First dims size (",
                   param->num_tensors, ") must match second_dims size (", t->second_dims.shape[0],
                   ")");
      }
      t->first_dims = transformer_engine::SimpleTensor(param->base_dptr, {param->num_tensors},
                                                       transformer_engine::DType::kInt64);
      break;
    case kNVTEGroupedSecondDims:
      NVTE_CHECK(param->base_dptr != nullptr, "Second dims must have a valid pointer");
      NVTE_CHECK(param->num_tensors > 0, "Second dims must have num_tensors > 0");
      // If first_dims is already set, validate consistency
      if (t->first_dims.dptr != nullptr && !t->first_dims.shape.empty()) {
        NVTE_CHECK(t->first_dims.shape[0] == param->num_tensors, "Second dims size (",
                   param->num_tensors, ") must match first_dims size (", t->first_dims.shape[0],
                   ")");
      }
      t->second_dims = transformer_engine::SimpleTensor(param->base_dptr, {param->num_tensors},
                                                        transformer_engine::DType::kInt64);
      break;
    case kNVTEGroupedCumulativeSizes:
      NVTE_CHECK(param->base_dptr != nullptr, "Cumulative sizes must have a valid pointer");
      NVTE_CHECK(param->num_tensors > 0, "Cumulative sizes must have num_tensors > 0");
      // If first_dims or second_dims is already set, validate consistency
      if (t->first_dims.dptr != nullptr && !t->first_dims.shape.empty()) {
        NVTE_CHECK(t->first_dims.shape[0] == param->num_tensors, "Cumulative sizes num_tensors (",
                   param->num_tensors, ") must match first_dims size (", t->first_dims.shape[0],
                   ")");
      }
      if (t->second_dims.dptr != nullptr && !t->second_dims.shape.empty()) {
        NVTE_CHECK(t->second_dims.shape[0] == param->num_tensors, "Cumulative sizes num_tensors (",
                   param->num_tensors, ") must match second_dims size (", t->second_dims.shape[0],
                   ")");
      }
      // cumulative_tensor_sizes has num_tensors+1 elements
      t->cumulative_tensor_sizes = transformer_engine::SimpleTensor(
          param->base_dptr, {param->num_tensors + 1}, transformer_engine::DType::kInt64);
      break;
    default:
      NVTE_ERROR("Unknown grouped tensor parameter!");
  }
}

NVTEGroupedTensorInfo nvte_get_grouped_tensor_param(const NVTEGroupedTensor tensor,
                                                    NVTEGroupedTensorParam param_name) {
  if (tensor == nullptr) {
    return {nullptr, nullptr, kNVTEFloat32, 0, false, 0, 0};
  }
  const auto &t = *transformer_engine::convertNVTEGroupedTensorCheck(tensor);

  // Helper to convert SimpleGroupedTensor to NVTEGroupedTensorInfo
  auto to_info = [](const transformer_engine::SimpleGroupedTensor &sgt) -> NVTEGroupedTensorInfo {
    NVTEGroupedTensorInfo info;
    info.dtype = static_cast<NVTEDType>(sgt.dtype);
    info.num_tensors = sgt.num_tensors;
    info.contiguous = sgt.contiguous;
    info.sum_first_dims = sgt.sum_first_dims;
    info.sum_second_dims = sgt.sum_second_dims;
    if (sgt.has_dptr_list()) {
      info.base_dptr = nullptr;
      info.dptr_list = const_cast<void **>(sgt.get_dptr_list());
    } else {
      info.base_dptr = sgt.get_base_dptr();
      info.dptr_list = nullptr;
    }
    return info;
  };

  switch (param_name) {
    case kNVTEGroupedRowwiseData:
      return to_info(t.data);
    case kNVTEGroupedColumnwiseData:
      return to_info(t.columnwise_data);
    case kNVTEGroupedScale:
      return to_info(t.scale);
    case kNVTEGroupedAmax:
      return to_info(t.amax);
    case kNVTEGroupedRowwiseScaleInv:
      return to_info(t.scale_inv);
    case kNVTEGroupedColumnwiseScaleInv:
      return to_info(t.columnwise_scale_inv);
    case kNVTEGroupedColumnwiseAmax:
      return to_info(t.columnwise_amax);
    case kNVTEGroupedFirstDims:
      return {t.first_dims.dptr,
              nullptr,
              static_cast<NVTEDType>(t.first_dims.dtype),
              t.first_dims.shape.empty() ? 0 : t.first_dims.shape[0],
              true,
              0,
              0};
    case kNVTEGroupedSecondDims:
      return {t.second_dims.dptr,
              nullptr,
              static_cast<NVTEDType>(t.second_dims.dtype),
              t.second_dims.shape.empty() ? 0 : t.second_dims.shape[0],
              true,
              0,
              0};
    case kNVTEGroupedCumulativeSizes:
      return {t.cumulative_tensor_sizes.dptr,
              nullptr,
              static_cast<NVTEDType>(t.cumulative_tensor_sizes.dtype),
              t.cumulative_tensor_sizes.shape.empty() ? 0 : t.cumulative_tensor_sizes.shape[0],
              true,
              0,
              0};
    default:
      NVTE_ERROR("Unknown grouped tensor parameter!");
  }
}

size_t nvte_grouped_tensor_num_tensors(const NVTEGroupedTensor tensor) {
  auto *t = transformer_engine::convertNVTEGroupedTensor(tensor);
  if (t == nullptr) return 0;
  return t->num_tensors();
}

NVTEDType nvte_grouped_tensor_type(const NVTEGroupedTensor tensor) {
  auto *t = transformer_engine::convertNVTEGroupedTensor(tensor);
  if (t == nullptr) return kNVTEFloat32;
  return static_cast<NVTEDType>(t->dtype());
}

NVTEScalingMode nvte_grouped_tensor_scaling_mode(const NVTEGroupedTensor tensor) {
  if (tensor == nullptr) {
    return NVTE_DELAYED_TENSOR_SCALING;
  }
  const auto &t = *transformer_engine::convertNVTEGroupedTensorCheck(tensor);
  return t.scaling_mode;
}
