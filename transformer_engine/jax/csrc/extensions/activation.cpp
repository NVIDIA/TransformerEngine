/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include "transformer_engine/activation.h"

#include <cuda_runtime.h>

#include "extensions.h"
#include "transformer_engine/cast.h"
#include "xla/ffi/api/c_api.h"

namespace {
bool is_gated(NVTE_Activation_Type act_type) {
  return act_type == NVTE_Activation_Type::GEGLU || act_type == NVTE_Activation_Type::SWIGLU ||
         act_type == NVTE_Activation_Type::REGLU || act_type == NVTE_Activation_Type::QGEGLU ||
         act_type == NVTE_Activation_Type::SREGLU;
}
}  // namespace

namespace transformer_engine {
namespace jax {

Error_Type ActLuFFI(cudaStream_t stream, Buffer_Type input_buf, Buffer_Type scale_buf,
                    Result_Type output_buf, Result_Type colwise_output_buf,
                    Result_Type scale_inv_buf, Result_Type colwise_scale_inv_buf,
                    Result_Type amax_buf, int64_t act_enum, int64_t scaling_mode_enum,
                    bool is_2x_int) {
  auto in_dtype = convert_ffi_datatype_to_te_dtype(input_buf.element_type());
  auto out_dtype = convert_ffi_datatype_to_te_dtype(output_buf->element_type());

  auto *input = input_buf.untyped_data();
  float *scale = reinterpret_cast<float *>(scale_buf.untyped_data());

  auto *output = output_buf->untyped_data();
  auto *colwise_output = colwise_output_buf->untyped_data();
  float *amax = reinterpret_cast<float *>(amax_buf->untyped_data());

  auto input_dims = input_buf.dimensions();
  auto m = product(input_dims, 0, input_dims.size() - 2);
  auto n = input_dims.back();
  auto act_type = static_cast<NVTE_Activation_Type>(act_enum);
  auto act_len = input_dims[input_dims.size() - 2];
  auto scaling_mode = static_cast<NVTEScalingMode>(scaling_mode_enum);
  auto is_2x = static_cast<bool>(is_2x_int);

  auto input_shape = std::vector<size_t>{m, act_len * n};
  auto output_shape = std::vector<size_t>{m, n};
  auto input_tensor = TensorWrapper(input, input_shape, static_cast<DType>(in_dtype));
  auto output_tensor = TensorWrapper(scaling_mode);
  output_tensor.set_rowwise_data(output, static_cast<DType>(out_dtype), output_shape);

  if (is_fp8_dtype(out_dtype)) {
    output_tensor.set_rowwise_scale_inv(
        scale_inv_buf->untyped_data(),
        convert_ffi_datatype_to_te_dtype(scale_inv_buf->element_type()),
        std::vector<size_t>{
            product(scale_inv_buf->dimensions(), 0, scale_inv_buf->dimensions().size() - 1),
            scale_inv_buf->dimensions().back()});
  }

  if (scaling_mode == NVTE_DELAYED_TENSOR_SCALING && is_fp8_dtype(out_dtype)) {
    NVTE_CHECK(scale != nullptr, "scale must be provided for delayed tensor scaling");
    NVTE_CHECK(amax != nullptr, "amax must be provided for delayed tensor scaling");
    cudaMemsetAsync(amax, 0, sizeof(float), stream);
    output_tensor.set_scale(scale, DType::kFloat32, std::vector<size_t>{1});
    output_tensor.set_amax(amax, DType::kFloat32, std::vector<size_t>{1});
  }

  if (is_2x) {
    output_tensor.set_columnwise_data(colwise_output, static_cast<DType>(out_dtype), output_shape);
    output_tensor.set_columnwise_scale_inv(
        colwise_scale_inv_buf->untyped_data(),
        convert_ffi_datatype_to_te_dtype(colwise_scale_inv_buf->element_type()),
        std::vector<size_t>{product(colwise_scale_inv_buf->dimensions(), 0,
                                    colwise_scale_inv_buf->dimensions().size() - 1),
                            colwise_scale_inv_buf->dimensions().back()});
  }

  switch (act_type) {
    case NVTE_Activation_Type::GELU:
      nvte_gelu(input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::GEGLU:
      nvte_geglu(input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::SILU:
      nvte_silu(input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::SWIGLU:
      nvte_swiglu(input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::RELU:
      nvte_relu(input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::REGLU:
      nvte_reglu(input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::QGELU:
      nvte_qgelu(input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::QGEGLU:
      nvte_qgeglu(input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::SRELU:
      nvte_srelu(input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::SREGLU:
      nvte_sreglu(input_tensor.data(), output_tensor.data(), stream);
      break;
    default:
      NVTE_ERROR("Unsupported ActivationEnum");
      break;
  }

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(ActLuHandler, ActLuFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // input
                                  .Arg<Buffer_Type>()      // scale
                                  .Ret<Buffer_Type>()      // output
                                  .Ret<Buffer_Type>()      // colwise output
                                  .Ret<Buffer_Type>()      // scale_inv
                                  .Ret<Buffer_Type>()      // scale_inv colwise
                                  .Ret<Buffer_Type>()      // amax
                                  .Attr<int64_t>("act_enum")
                                  .Attr<int64_t>("scaling_mode")
                                  .Attr<bool>("is_2x"),
                              FFI_CudaGraph_Traits);

pybind11::tuple GetDActDBiasQuantizeWorkspaceSizes(size_t batch_size, size_t hidden_size,
                                                   DType in_dtype, DType out_dtype,
                                                   int scaling_mode, bool is_2x) {
  auto input_shape = std::vector<size_t>{batch_size, hidden_size};
  auto dact_input_shape = std::vector<size_t>{batch_size, hidden_size};
  auto output_shape = std::vector<size_t>{batch_size, hidden_size};
  auto output_trans_shape = std::vector<size_t>{hidden_size, batch_size};
  auto dbias_shape = std::vector<size_t>{hidden_size};

  // Evil hack to specify TE impl
  // Note: nvte_quantize_dbias_dgelu chooses its internal impl based
  // on what pointers are allocated, e.g. whether to output with
  // column-wise data. However, we don't have access to any allocated
  // buffers in this function. We pass a dummy pointer as a
  // workaround.
  int temp = 0;

  auto input_tensor = TensorWrapper(reinterpret_cast<void *>(&temp), input_shape, in_dtype);
  auto dact_input_tensor =
      TensorWrapper(reinterpret_cast<void *>(&temp), dact_input_shape, in_dtype);
  auto dbias_tensor = TensorWrapper(reinterpret_cast<void *>(&temp), dbias_shape, in_dtype);
  auto output_tensor = TensorWrapper(static_cast<NVTEScalingMode>(scaling_mode));
  output_tensor.set_rowwise_data(reinterpret_cast<void *>(&temp), out_dtype, output_shape);
  // Only the pointers will be checked for scale_inv, thus the shapes do not matter
  if (is_fp8_dtype(out_dtype)) {
    output_tensor.set_rowwise_scale_inv(reinterpret_cast<void *>(&temp), DType::kFloat32,
                                        std::vector<size_t>{1});
  }

  if (is_2x) {
    output_tensor.set_columnwise_data(reinterpret_cast<void *>(&temp), out_dtype,
                                      output_trans_shape);

    // Only the pointers will be checked for scale_inv, thus the shapes do not matter
    if (is_fp8_dtype(out_dtype)) {
      output_tensor.set_columnwise_scale_inv(reinterpret_cast<void *>(&temp), DType::kFloat32,
                                             std::vector<size_t>{1});
    }
  }

  if (is_fp8_dtype(out_dtype) && scaling_mode == NVTEScalingMode::NVTE_DELAYED_TENSOR_SCALING) {
    output_tensor.set_amax(reinterpret_cast<void *>(&temp), DType::kFloat32,
                           std::vector<size_t>{1});
    output_tensor.set_scale(reinterpret_cast<void *>(&temp), DType::kFloat32,
                            std::vector<size_t>{1});
  }

  TensorWrapper dummy_workspace;
  // For now, all dbias_dact(-s) have the same workspace size
  nvte_quantize_dbias_dgelu(input_tensor.data(), dact_input_tensor.data(), output_tensor.data(),
                            dbias_tensor.data(), dummy_workspace.data(), nullptr);

  auto work_shape = MakeShapeVector(dummy_workspace.shape());
  return pybind11::make_tuple(std::make_pair(work_shape, dummy_workspace.dtype()));
}

Error_Type DActLuDBiasQuantizeFFI(cudaStream_t stream, Buffer_Type input_buf,
                                  Buffer_Type act_input_buf, Buffer_Type scale_buf,
                                  Result_Type output_buf, Result_Type output_trans_buf,
                                  Result_Type scale_inv_buf, Result_Type trans_scale_inv_buf,
                                  Result_Type amax_out_buf, Result_Type dbias_buf,
                                  Result_Type workspace_buf, int64_t scaling_mode_enum, bool is_2x,
                                  bool is_dbias, int64_t act_enum) {
  auto in_dtype = convert_ffi_datatype_to_te_dtype(input_buf.element_type());
  auto out_dtype = convert_ffi_datatype_to_te_dtype(output_buf->element_type());
  auto workspace_dtype = convert_ffi_datatype_to_te_dtype(workspace_buf->element_type());

  auto *input = input_buf.untyped_data();
  auto *act_input = act_input_buf.untyped_data();

  auto scaling_mode = static_cast<NVTEScalingMode>(scaling_mode_enum);

  auto *output = output_buf->untyped_data();
  auto *output_trans = output_trans_buf->untyped_data();
  auto *dbias = dbias_buf->untyped_data();
  void *workspace = workspace_buf->untyped_data();

  auto input_dims = input_buf.dimensions();
  auto act_input_dims = act_input_buf.dimensions();
  auto workspace_dims = workspace_buf->dimensions();
  // m = x_batch_size = reduce(operator.mul, x_shape[:-2]), x_shape == act_input_dims
  // n = ir_dz_shape[-1], ir_dz_shape == input_dims
  auto input_ranks = input_dims.size();
  auto act_input_ranks = act_input_dims.size();
  auto m = product(act_input_dims, 0, act_input_dims.size() - 1);
  // 'n' will be 2x the size of input_dims.back() if the dactivation is dgated
  auto n = act_input_dims.back();
  auto input_shape = std::vector<size_t>{m, input_dims.back()};
  auto act_input_shape = std::vector<size_t>{m, n};
  auto output_shape = std::vector<size_t>{m, n};
  auto output_trans_shape = std::vector<size_t>{m, n};
  auto dbias_shape = std::vector<size_t>{n};
  std::vector<size_t> workspace_shape(workspace_dims.begin(), workspace_dims.end());

  auto input_tensor = TensorWrapper(input, input_shape, in_dtype);
  auto act_input_tensor = TensorWrapper(act_input, act_input_shape, in_dtype);
  auto output_tensor = TensorWrapper(scaling_mode);
  output_tensor.set_rowwise_data(output, out_dtype, output_shape);
  if (is_fp8_dtype(out_dtype)) {
    output_tensor.set_rowwise_scale_inv(
        scale_inv_buf->untyped_data(),
        convert_ffi_datatype_to_te_dtype(scale_inv_buf->element_type()),
        std::vector<size_t>{
            product(scale_inv_buf->dimensions(), 0, scale_inv_buf->dimensions().size() - 1),
            scale_inv_buf->dimensions().back()});

    if (scaling_mode == NVTE_DELAYED_TENSOR_SCALING) {
      float *scale = reinterpret_cast<float *>(scale_buf.untyped_data());
      float *amax_out = reinterpret_cast<float *>(amax_out_buf->untyped_data());
      NVTE_CHECK(scale != nullptr, "scale must be provided for delayed tensor scaling");
      NVTE_CHECK(amax_out != nullptr, "amax must be provided for delayed tensor scaling");
      cudaMemsetAsync(amax_out, 0, sizeof(float), stream);
      output_tensor.set_scale(scale, DType::kFloat32, std::vector<size_t>{1});
      output_tensor.set_amax(amax_out, DType::kFloat32, std::vector<size_t>{1});
    }
  }

  if (is_2x) {
    output_tensor.set_columnwise_data(output_trans, out_dtype, output_trans_shape);

    if (is_fp8_dtype(out_dtype)) {
      // For 2x delayed scaling, the scale buffer is shared between rowwise and columnwise scaling
      auto &colwise_scale_inv_buf =
          (scaling_mode == NVTE_DELAYED_TENSOR_SCALING) ? scale_inv_buf : trans_scale_inv_buf;
      output_tensor.set_columnwise_scale_inv(
          colwise_scale_inv_buf->untyped_data(),
          convert_ffi_datatype_to_te_dtype(colwise_scale_inv_buf->element_type()),
          std::vector<size_t>{product(colwise_scale_inv_buf->dimensions(), 0,
                                      colwise_scale_inv_buf->dimensions().size() - 1),
                              colwise_scale_inv_buf->dimensions().back()});
    }
  }

  auto dbias_tensor = TensorWrapper(dbias, dbias_shape, in_dtype);
  auto workspace_tensor = TensorWrapper(workspace, workspace_shape, workspace_dtype);

  auto act_type = static_cast<NVTE_Activation_Type>(act_enum);

  // fused_dgated_dbias is not available, so we use dact_lu + quantize_dbias in Python instead
  NVTE_CHECK(!(is_gated(act_type) && is_dbias), "Unsupported DGatedActedDBias Fusion!");
  NVTE_CHECK(!(scaling_mode == NVTEScalingMode::NVTE_DELAYED_TENSOR_SCALING && is_2x &&
               is_gated(act_type)),
             "TE/common does not support delayed scaling for 2x with gated activations.");

  if (is_dbias) {
    switch (act_type) {
      case NVTE_Activation_Type::GELU:
        nvte_quantize_dbias_dgelu(input_tensor.data(), act_input_tensor.data(),
                                  output_tensor.data(), dbias_tensor.data(),
                                  workspace_tensor.data(), stream);
        break;
      case NVTE_Activation_Type::SILU:
        nvte_quantize_dbias_dsilu(input_tensor.data(), act_input_tensor.data(),
                                  output_tensor.data(), dbias_tensor.data(),
                                  workspace_tensor.data(), stream);
        break;
      case NVTE_Activation_Type::RELU:
        nvte_quantize_dbias_drelu(input_tensor.data(), act_input_tensor.data(),
                                  output_tensor.data(), dbias_tensor.data(),
                                  workspace_tensor.data(), stream);
        break;
      case NVTE_Activation_Type::QGELU:
        nvte_quantize_dbias_dqgelu(input_tensor.data(), act_input_tensor.data(),
                                   output_tensor.data(), dbias_tensor.data(),
                                   workspace_tensor.data(), stream);
        break;
      case NVTE_Activation_Type::SRELU:
        nvte_quantize_dbias_dsrelu(input_tensor.data(), act_input_tensor.data(),
                                   output_tensor.data(), dbias_tensor.data(),
                                   workspace_tensor.data(), stream);
        break;
      default:
        NVTE_ERROR("Unsupported ActivationEnum = ", act_enum, "with dbias = True");
        break;
    }
  } else {
    switch (act_type) {
      case NVTE_Activation_Type::GELU:
        nvte_dgelu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
        break;
      case NVTE_Activation_Type::SILU:
        nvte_dsilu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
        break;
      case NVTE_Activation_Type::RELU:
        nvte_drelu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
        break;
      case NVTE_Activation_Type::QGELU:
        nvte_dqgelu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
        break;
      case NVTE_Activation_Type::SRELU:
        nvte_dsrelu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
        break;
      case NVTE_Activation_Type::GEGLU:
        nvte_dgeglu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
        break;
      case NVTE_Activation_Type::SWIGLU:
        nvte_dswiglu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
        break;
      case NVTE_Activation_Type::REGLU:
        nvte_dreglu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
        break;
      case NVTE_Activation_Type::QGEGLU:
        nvte_dqgeglu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
        break;
      case NVTE_Activation_Type::SREGLU:
        nvte_dsreglu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
        break;
      default:
        NVTE_ERROR("Unsupported ActivationEnum");
        break;
    }
  }

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(DActLuDBiasQuantizeHandler, DActLuDBiasQuantizeFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // input
                                  .Arg<Buffer_Type>()      // act input
                                  .Arg<Buffer_Type>()      // scale
                                  .Ret<Buffer_Type>()      // output
                                  .Ret<Buffer_Type>()      // colwise output
                                  .Ret<Buffer_Type>()      // scale_inv
                                  .Ret<Buffer_Type>()      // scale_inv colwise
                                  .Ret<Buffer_Type>()      // amax
                                  .Ret<Buffer_Type>()      // dbias
                                  .Ret<Buffer_Type>()      // wkspace
                                  .Attr<int64_t>("scaling_mode")
                                  .Attr<bool>("is_2x")
                                  .Attr<bool>("is_dbias")
                                  .Attr<int64_t>("act_enum"),
                              FFI_CudaGraph_Traits);
}  // namespace jax
}  // namespace transformer_engine
