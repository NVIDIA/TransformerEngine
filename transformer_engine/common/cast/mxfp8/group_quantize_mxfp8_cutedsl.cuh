/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_GROUP_QUANTIZE_MXFP8_CUTEDSL_CUH_
#define TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_GROUP_QUANTIZE_MXFP8_CUTEDSL_CUH_

#include <tvm/ffi/any.h>
#include <tvm/ffi/function.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "../../common.h"
#include "../../tvm_ffi_bridge.h"
#include "../../util/cuda_runtime.h"
#include "../../util/cutedsl_utils.h"
#include "../../utils.cuh"     // ShapeRepresentation
#include "../core/common.cuh"  // MAX_SUPPORTED_TENSOR_DESCRIPTORS, grouped_reduce_dbias

namespace transformer_engine {
namespace cutedsl_backend {

// DLTensorWrapper and TVMFFICentral live in transformer_engine::tvm_ffi_bridge.
using namespace tvm_ffi_bridge;

inline const char *shape_rep_to_str(ShapeRepresentation shape_rep) {
  switch (shape_rep) {
    case ShapeRepresentation::SAME_BOTH_DIMS:
      return "sbd";
    case ShapeRepresentation::VARYING_FIRST_DIM:
      return "vfd";
    case ShapeRepresentation::VARYING_LAST_DIM:
      return "vld";
    default:
      return "vbd";
  }
}

struct MXFP8GroupQuantConfig {
  static constexpr const char *kEntrypointName = "get_mxfp8_group_quantization_function";

  DType dtype;                    // The input format
  DType fp8_dtype;                // The fp8 output format
  bool rowwise;                   // If quantize rowwisely
  bool colwise;                   // If quantize columnwisely
  ShapeRepresentation shape_rep;  // How the member shapes vary across the group
  bool swizzled;                  // If the scales are written in the GEMM-swizzled layout
  bool with_dbias;                // If the partial dbias is computed (via the workspace tensor)
  bool with_dact;                 // If an activation derivative operation is fused
  bool with_act;                  // If an activation operation is fused
  Activation activation = Activation::kNone;
  uint32_t sm_arch = static_cast<uint32_t>(cuda::sm_arch());

  // Bit layout: dtype [3:0] (4 used/reserved), fp8_dtype [7:4] (4 used/reserved),
  // flags [13:8] (6 used), shape_rep [15:14] (2 used), activation [21:16] (6 used/reserved),
  // and SM architecture [30:22] (9 used/reserved). Bit 31 is unused.
  uint32_t to_id() const {
    static_assert(static_cast<uint32_t>(DType::kNumTypes) <= 16,
                  "DType no longer fits in the 4 bits to_id() gives it.");
    static_assert(ShapeRepresentation::VARYING_BOTH_DIMS < 4,
                  "ShapeRepresentation no longer fits in the 2 bits to_id() gives it.");
    static_assert(static_cast<uint32_t>(Activation::kNumTypes) <= 64,
                  "Activation no longer fits in the 6 bits to_id() gives it.");
    NVTE_CHECK(sm_arch < 512, "SM architecture no longer fits in the 9 bits to_id() gives it.");
    return static_cast<uint32_t>(dtype) | (static_cast<uint32_t>(fp8_dtype) << 4) |
           (static_cast<uint32_t>(rowwise) << 8) | (static_cast<uint32_t>(colwise) << 9) |
           (static_cast<uint32_t>(swizzled) << 10) | (static_cast<uint32_t>(with_dbias) << 11) |
           (static_cast<uint32_t>(with_dact) << 12) | (static_cast<uint32_t>(with_act) << 13) |
           (static_cast<uint32_t>(shape_rep) << 14) | (static_cast<uint32_t>(activation) << 16) |
           (sm_arch << 22);
  }

  std::optional<tvm::ffi::Function> get_kernel() const {
    static TVMFFIConfigCache &cache = TVMFFIConfigCache::create();
    return cache.get_or_load(*this);
  }

  // Globally unique TVM-FFI registry key used when the CuTeDSL function is
  // compiled and registered on a cache miss.
  std::string to_key() const {
    std::string key;
    // longest: cutedsl_group_mxfp8_smXXX_BFloat16_Float8E4M3_1_1_vfd_1_1_1_0_dqgelu
    key.reserve(96);
    key.append("cutedsl_group_mxfp8_sm")
        .append(std::to_string(sm_arch))
        .append("_")
        .append(to_string(dtype))
        .append("_")
        .append(to_string(fp8_dtype))
        .append("_")
        .append(rowwise ? "1" : "0")
        .append("_")
        .append(colwise ? "1" : "0")
        .append("_")
        .append(shape_rep_to_str(shape_rep))
        .append("_")
        .append(swizzled ? "1" : "0")
        .append("_")
        .append(with_dbias ? "1" : "0")
        .append("_")
        .append(with_dact ? "1" : "0")
        .append("_")
        .append(with_act ? "1" : "0")
        .append("_")
        .append(activation_to_str(activation));
    return key;
  }

  bool retrieve_func_from_python(const std::string &fn_name) const {
    auto entrypoint = tvm::ffi::Function::GetGlobal(kEntrypointName);
    if (!entrypoint.has_value()) {
      return false;
    }
    tvm::ffi::Any result =
        (*entrypoint)(tvm::ffi::String(fn_name), tvm::ffi::String(to_string(dtype)),
                      tvm::ffi::String(to_string(fp8_dtype)), rowwise, colwise,
                      tvm::ffi::String(shape_rep_to_str(shape_rep)), swizzled, with_dbias,
                      with_dact, with_act, tvm::ffi::String(activation_to_str(activation)));
    return result.try_cast<bool>().value_or(false);
  }
};

// kGroupTensorMapSlots is 5 slots for: input, rowwise output, colwise output, activation input,
// plus one carrying the metadata of tensor -- (rows, cols, base_elts)
constexpr size_t kGroupTensorMapSlots = 5;
constexpr size_t kInt64PerTensorMap = 128 / sizeof(int64_t);
constexpr size_t kMaxGroupTensors =
    static_cast<size_t>(dispatch::common::MAX_SUPPORTED_TENSOR_DESCRIPTORS);

// We need to use int64_t here instead of CUtensorMap so we can pass this through tvm-ffi boundary
struct alignas(128) TensorMapStorage {
  alignas(128) int64_t tensor_maps[kMaxGroupTensors][kGroupTensorMapSlots][kInt64PerTensorMap];
};
static __device__ TensorMapStorage g_group_descriptor_workspace;

static TensorMapStorage *group_descriptor_workspace_ptr() {
  // Each device has its own workspace for descriptors
  static std::vector<TensorMapStorage *> cache(cuda::num_devices(), nullptr);
  static std::vector<std::once_flag> flags(cuda::num_devices());
  const int device_id = cuda::current_device();
  NVTE_CHECK(0 <= device_id && device_id < cuda::num_devices(), "invalid CUDA device ID");
  // Copy the device symbol address on the current device into the cache on its first use only
  std::call_once(flags[device_id], [&]() {
    void *p = nullptr;
    NVTE_CHECK_CUDA(cudaGetSymbolAddress(&p, g_group_descriptor_workspace));
    cache[device_id] = static_cast<TensorMapStorage *>(p);
  });
  // Return the cached device pointer for the current device to the host
  return cache[device_id];
}

// Signature mirrors mxfp8::group_quantize (input, act_input, noop, output, dbias, workspace,
// stream). Returns false to fall back to the CUDA kernel.
inline bool mxfp8_group_quantize_cutedsl(const MXFP8GroupQuantConfig &config,
                                         const GroupedTensor *input_tensor,
                                         const GroupedTensor *act_input_tensor,
                                         const Tensor *noop_tensor, GroupedTensor *output_tensor,
                                         GroupedTensor *dbias_tensor, Tensor *workspace_tensor,
                                         cudaStream_t stream) {
  const size_t num_tensors = input_tensor->num_tensors;
  const size_t first_logical_dim = input_tensor->logical_shape.data[0];
  const size_t last_logical_dim = input_tensor->logical_shape.data[1];

  // Match the symbolic divisibility checks in group_quantize_mxfp8.py.
  if (config.shape_rep != ShapeRepresentation::VARYING_BOTH_DIMS && first_logical_dim % 128 != 0) {
    maybe_warn_cutedsl_not_chosen("the first logical dimension is not divisible by 128 for a non-varying both dimensions tensor.");
    return false;
  }
  if (last_logical_dim % 16 != 0) {
    maybe_warn_cutedsl_not_chosen("the last logical dimension is not divisible by 16.");
    return false;
  }

  // The same extents are sym_int32 in the compiled kernel.
  if (first_logical_dim > static_cast<size_t>(INT32_MAX) ||
      last_logical_dim > static_cast<size_t>(INT32_MAX)) {
    maybe_warn_cutedsl_not_chosen("the grouped logical shape does not fit in int32.");
    return false;
  }

  // How many rows a job processes (see the CuTeDSL kernel)
  constexpr size_t kRowsPerJob = 128;
  // For dbias workspace-size query
  if (config.with_dbias && workspace_tensor->data.dptr == nullptr) {
    workspace_tensor->data.shape = {DIVUP(first_logical_dim, kRowsPerJob), last_logical_dim};
    workspace_tensor->data.dtype = DType::kFloat32;
    return true;
  }

  std::optional<tvm::ffi::Function> group_quant_func_opt = config.get_kernel();
  if (!group_quant_func_opt.has_value()) {
    return false;
  }

  const int32_t device_index = transformer_engine::cuda::current_device();
  TensorMapStorage *const workspace = group_descriptor_workspace_ptr();

  const SimpleTensor &scale_row =
      config.rowwise ? output_tensor->scale_inv : output_tensor->columnwise_scale_inv;
  const SimpleTensor &scale_col =
      config.colwise ? output_tensor->columnwise_scale_inv : output_tensor->scale_inv;

  // The group's payload is stored flat; the kernel wants it as the logical 2D view.
  const std::vector<size_t> logical_shape{first_logical_dim, last_logical_dim};
  DLTensorWrapper mX(
      make_basic_tensor(input_tensor->data.dptr, input_tensor->dtype(), logical_shape), true,
      device_index);

  DLTensorWrapper mO_row, mO_col;
  if (config.rowwise) {
    mO_row = DLTensorWrapper(
        make_basic_tensor(output_tensor->data.dptr, output_tensor->data.dtype, logical_shape), true,
        device_index);
  }
  if (config.colwise) {
    mO_col = DLTensorWrapper(make_basic_tensor(output_tensor->columnwise_data.dptr,
                                               output_tensor->columnwise_data.dtype, logical_shape),
                             true, device_index);
  }

  // The kernel only takes the base address of the scale buffers (per-tensor bases and
  // strides are derived from the member shapes), so these stay 1D.
  DLTensorWrapper mS_row(make_basic_tensor(scale_row.dptr, scale_row.dtype, {scale_row.numel()}),
                         false, device_index);
  DLTensorWrapper mS_col(make_basic_tensor(scale_col.dptr, scale_col.dtype, {scale_col.numel()}),
                         false, device_index);

  // Offsets and member dims are read from the output, as in mxfp8::group_quantize.
  DLTensorWrapper mOffsets, mFirstDims, mLastDims;
  if (config.shape_rep != ShapeRepresentation::SAME_BOTH_DIMS) {
    NVTE_CHECK(output_tensor->tensor_offsets.has_data(), "Grouped MXFP8 quantization with ",
               shape_rep_to_str(config.shape_rep), " requires an allocated tensor_offsets buffer.");
    mOffsets = DLTensorWrapper(
        make_basic_tensor(output_tensor->tensor_offsets.dptr, DType::kInt64, {num_tensors + 1}),
        false, device_index);
  }
  if (config.shape_rep == ShapeRepresentation::VARYING_FIRST_DIM ||
      config.shape_rep == ShapeRepresentation::VARYING_BOTH_DIMS) {
    NVTE_CHECK(output_tensor->first_dims.has_data(), "Grouped MXFP8 quantization with ",
               shape_rep_to_str(config.shape_rep), " requires an allocated first_dims buffer.");
    mFirstDims = DLTensorWrapper(
        make_basic_tensor(output_tensor->first_dims.dptr, DType::kInt64, {num_tensors}), false,
        device_index);
  }
  if (config.shape_rep == ShapeRepresentation::VARYING_LAST_DIM ||
      config.shape_rep == ShapeRepresentation::VARYING_BOTH_DIMS) {
    NVTE_CHECK(output_tensor->last_dims.has_data(), "Grouped MXFP8 quantization with ",
               shape_rep_to_str(config.shape_rep), " requires an allocated last_dims buffer.");
    mLastDims = DLTensorWrapper(
        make_basic_tensor(output_tensor->last_dims.dptr, DType::kInt64, {num_tensors}), false,
        device_index);
  }

  // Pass tensormaps as a 3D tensor of int64_t
  DLTensorWrapper mTensormaps(
      make_basic_tensor(static_cast<void *>(workspace->tensor_maps), DType::kInt64,
                        {num_tensors, kGroupTensorMapSlots, kInt64PerTensorMap}),
      false, device_index);

  // Optional inputs: a wrapper over a null buffer packs as TVM-FFI None.
  DLTensorWrapper mActInput, mWorkspace;
  if (config.with_dact) {
    mActInput = DLTensorWrapper(
        make_basic_tensor(act_input_tensor->data.dptr, act_input_tensor->dtype(), logical_shape),
        true, device_index);
  }
  if (config.with_dbias) {
    mWorkspace = DLTensorWrapper(workspace_tensor->data, true, device_index);
  }

  void *noop_ptr = (noop_tensor != nullptr) ? noop_tensor->data.dptr : nullptr;

  (*group_quant_func_opt)(&mX, &mO_row, &mO_col, &mS_row, &mS_col, &mOffsets, &mFirstDims,
                          &mLastDims, &mTensormaps, noop_ptr, &mActInput, &mWorkspace,
                          static_cast<void *>(stream));

  // Reduce the per-chunk partial dbias per member with the CUDA kernel's reduction.
  if (config.with_dbias) {
    const float *workspace_ptr = reinterpret_cast<const float *>(workspace_tensor->data.dptr);
    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
        input_tensor->dtype(), IType,
        dispatch::common::grouped_reduce_dbias<IType>(
            config.shape_rep, num_tensors, first_logical_dim, last_logical_dim,
            reinterpret_cast<const int64_t *>(output_tensor->tensor_offsets.dptr),
            reinterpret_cast<const int64_t *>(output_tensor->first_dims.dptr),
            reinterpret_cast<const int64_t *>(output_tensor->last_dims.dptr), dbias_tensor,
            workspace_ptr, kChunkDimY, stream);)  // NOLINT(*)
  }
  return true;
}

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &)>
bool mxfp8_group_quantize_cutedsl(const GroupedTensor *input_tensor,
                                  const GroupedTensor *act_input_tensor, const Tensor *noop_tensor,
                                  GroupedTensor *output_tensor, GroupedTensor *dbias_tensor,
                                  Tensor *workspace_tensor, const QuantizationConfig *quant_config,
                                  cudaStream_t stream) {
  if (!tvm_ffi_bridge::TVMFFICentral::getInstance().get_cutedsl_backend_enabled()) {
    maybe_warn_cutedsl_not_chosen("the CuTeDSL backend is disabled.");
    return false;
  }
  // TODO(kainingz): port 2D quantization to CuTeDSL
  if (quant_config != nullptr && quant_config->mxfp8_2d_quantization) {
    maybe_warn_cutedsl_not_chosen("2D quantization is not supported.");
    return false;
  }
  constexpr Activation activation = activation_func_to_enum<ParamOP, OP>();
  if constexpr (activation == Activation::kUnsupported) {
    maybe_warn_cutedsl_not_chosen("the fused activation is not supported.");
    return false;
  } else {
    // Mirrors the shape-representation selection in mxfp8::group_quantize.
    ShapeRepresentation shape_rep = ShapeRepresentation::SAME_BOTH_DIMS;
    if (output_tensor->all_same_shape()) {
      shape_rep = ShapeRepresentation::SAME_BOTH_DIMS;
    } else if (output_tensor->all_same_first_dim()) {
      shape_rep = ShapeRepresentation::VARYING_LAST_DIM;
    } else if (output_tensor->all_same_last_dim()) {
      shape_rep = ShapeRepresentation::VARYING_FIRST_DIM;
    } else if (output_tensor->varying_both_dims()) {
      shape_rep = ShapeRepresentation::VARYING_BOTH_DIMS;
    }
    const bool is_single_tensor = shape_rep == ShapeRepresentation::SAME_BOTH_DIMS ||
                                  shape_rep == ShapeRepresentation::VARYING_FIRST_DIM;

    // Leave invalid group sizes to mxfp8::group_quantize, which raises a proper error.
    // Every member gets a descriptor slot in the fixed-size workspace, so the CUDA
    // kernel's descriptor limit applies to the single-tensor representations here too.
    const size_t num_tensors = input_tensor->num_tensors;
    if (num_tensors == 0 || num_tensors > kMaxGroupTensors) {
      maybe_warn_cutedsl_not_chosen("the group size ", num_tensors, " is not between 1 and ",
                                    kMaxGroupTensors, ".");
      return false;
    }

    if (shape_rep == ShapeRepresentation::SAME_BOTH_DIMS) {
      // The kernel tiles the stacked rows without tensor boundaries, which matches the CUDA
      // kernel's per-tensor tiling only when every member's rows are a multiple of its
      // 128-row chunk. mxfp8::group_quantize raises for a non-integral row count.
      const size_t first_logical_dim = input_tensor->logical_shape.data[0];
      if (first_logical_dim % num_tensors != 0 || (first_logical_dim / num_tensors) % 128 != 0) {
        maybe_warn_cutedsl_not_chosen("the rows of each group member are not a multiple of 128.");
        return false;
      }
    } else if (!output_tensor->tensor_offsets.has_data()) {
      // The varying representations read per-member offsets, as the CUDA kernel does.
      maybe_warn_cutedsl_not_chosen("the grouped tensor has no tensor offsets.");
      return false;
    }

    if (IS_DBIAS && !is_single_tensor) {
      // mxfp8::group_quantize raises a proper error for this.
      maybe_warn_cutedsl_not_chosen("dbias is only supported for a common last dimension.");
      return false;
    }

    const bool rowwise = output_tensor->has_data();
    const bool colwise = output_tensor->has_columnwise_data();
    const bool swizzled = output_tensor->with_gemm_swizzled_scales;

    // Some Sanity checks
    if (!rowwise && !colwise) {
      maybe_warn_cutedsl_not_chosen("the grouped tensor has neither rowwise nor columnwise data.");
      return false;
    }
    checkCuDriverContext(stream);
    CheckNoopTensor(*noop_tensor, "cast_noop");
    if (rowwise) {
      NVTE_CHECK(output_tensor->scale_inv.dptr != nullptr, "Scaling tensor must be allocated");
    }
    if (colwise) {
      NVTE_CHECK(output_tensor->columnwise_scale_inv.dptr != nullptr,
                 "Columnwise scaling tensor must be allocated");
    }
    NVTE_CHECK(input_tensor->num_tensors == output_tensor->num_tensors,
               "Number of input and output tensors must be same.");
    NVTE_CHECK(input_tensor->has_data(), "Cannot quantize tensor without rowwise data.");
    NVTE_CHECK(is_fp8_dtype(output_tensor->dtype()), "Output must have FP8 type.");
    if constexpr (IS_DACT) {
      NVTE_CHECK(act_input_tensor->has_data(), "Activations tensor must have data.");
      NVTE_CHECK(input_tensor->num_tensors == act_input_tensor->num_tensors,
                 "Number of grad and activations tensors must be same.");
      NVTE_CHECK(input_tensor->dtype() == act_input_tensor->dtype(),
                 "Grad and activations tensors must have the same type.");
    }
    if constexpr (IS_DBIAS) {
      NVTE_CHECK(dbias_tensor->data.dtype == input_tensor->dtype(),
                 "DBias must have the same type as input_tensor.");
      const Shape expected_shape_dbias_tensor = {num_tensors, input_tensor->logical_shape.data[1]};
      NVTE_CHECK(dbias_tensor->data.shape == expected_shape_dbias_tensor, "Wrong shape of DBias.");
      NVTE_CHECK(workspace_tensor != nullptr, "Workspace must be a tensor.");
    }

    const MXFP8GroupQuantConfig config{/*dtype=*/input_tensor->dtype(),
                                       /*fp8_dtype=*/output_tensor->dtype(),
                                       /*rowwise=*/rowwise,
                                       /*colwise=*/colwise,
                                       /*shape_rep=*/shape_rep,
                                       /*swizzled=*/swizzled,
                                       /*with_dbias=*/IS_DBIAS,
                                       /*with_dact=*/IS_DACT,
                                       /*with_act=*/IS_ACT,
                                       /*activation=*/activation};
    return mxfp8_group_quantize_cutedsl(config, input_tensor, act_input_tensor, noop_tensor,
                                        output_tensor, dbias_tensor, workspace_tensor, stream);
  }
}

}  // namespace cutedsl_backend
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_GROUP_QUANTIZE_MXFP8_CUTEDSL_CUH_
