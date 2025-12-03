/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include "transformer_engine/activation.h"

#include <cuda_runtime.h>

#include "../extensions.h"
#include "transformer_engine/cast.h"
#include "xla/ffi/api/c_api.h"

namespace transformer_engine {
namespace jax {

Error_Type ActLuFFI(cudaStream_t stream, Buffer_Type input_buf, Buffer_Type scale_buf,
                    Buffer_Type amax_buf, Result_Type output_buf, Result_Type colwise_output_buf,
                    Result_Type scale_inv_buf, Result_Type colwise_scale_inv_buf,
                    Result_Type updated_amax_buf, int64_t act_enum, JAXX_Scaling_Mode scaling_mode,
                    JAXX_Quantize_Layout quantize_layout, ActivationConfig act_params,
                    bool output_amax_when_no_scaling) {
  // parameters for clamped swiglu used in GPT OSS
  auto swiglu_limit = act_params.clamped_swiglu.limit;
  auto swiglu_alpha = act_params.clamped_swiglu.alpha;

  auto in_dtype = convert_ffi_datatype_to_te_dtype(input_buf.element_type());
  auto out_dtype = convert_ffi_datatype_to_te_dtype(output_buf->element_type());

  auto *input = input_buf.untyped_data();
  float *scale = reinterpret_cast<float *>(scale_buf.untyped_data());

  auto *output = output_buf->untyped_data();
  auto *colwise_output = colwise_output_buf->untyped_data();
  float *amax = reinterpret_cast<float *>(amax_buf.untyped_data());
  auto *updated_amax = reinterpret_cast<float *>(updated_amax_buf->untyped_data());
  NVTE_CHECK(amax == updated_amax && amax != nullptr, "amax and updated_amax should be aliased");

  auto input_dims = input_buf.dimensions();
  auto m = product(input_dims, 0, input_dims.size() - 2);
  auto n = input_dims.back();
  auto act_type = static_cast<NVTE_Activation_Type>(act_enum);
  auto act_len = input_dims[input_dims.size() - 2];
  auto flatten_axis = output_buf->dimensions().size() - 1;  // output does not have act axis

  auto input_shape = std::vector<size_t>{m, static_cast<size_t>(act_len * n)};
  auto output_shape = std::vector<size_t>{m, static_cast<size_t>(n)};
  auto output_trans_shape = std::vector<size_t>{static_cast<size_t>(n), m};
  auto input_tensor = TensorWrapper(input, input_shape, static_cast<DType>(in_dtype));
  auto output_tensor = TensorWrapper(get_nvte_scaling_mode(scaling_mode));

  output_tensor.set_rowwise_data(output, static_cast<DType>(out_dtype), output_shape);
  if (scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING ||
      (scaling_mode == JAXX_Scaling_Mode::NO_SCALING && output_amax_when_no_scaling)) {
    output_tensor.set_amax(amax, DType::kFloat32, std::vector<size_t>{1});
  }

  NVTE_CHECK(
      scaling_mode != JAXX_Scaling_Mode::CURRENT_TENSOR_SCALING,
      "Current tensor scaling does not support fused operations yet. Please call this primitive "
      "in higher-precision then quantize with current scaling.");

  if (is_fp8_dtype(out_dtype)) {
    if (scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING) {
      NVTE_CHECK(scale != nullptr, "scale must be provided for delayed tensor scaling");
      output_tensor.set_scale(scale, DType::kFloat32, std::vector<size_t>{1});
      output_tensor.set_rowwise_scale_inv(
          scale_inv_buf->untyped_data(),
          convert_ffi_datatype_to_te_dtype(scale_inv_buf->element_type()), std::vector<size_t>{1});
    } else {
      output_tensor.set_rowwise_scale_inv(
          scale_inv_buf->untyped_data(),
          convert_ffi_datatype_to_te_dtype(scale_inv_buf->element_type()),
          std::vector<size_t>{product(scale_inv_buf->dimensions(), 0, flatten_axis),
                              product(scale_inv_buf->dimensions(), flatten_axis,
                                      scale_inv_buf->dimensions().size())});
    }
  }

  if (is_quantize_2x2x(quantize_layout)) {
    auto &tmp_shape = (scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING)
                          ? output_trans_shape
                          : output_shape;
    output_tensor.set_columnwise_data(colwise_output, out_dtype, tmp_shape);

    if (is_fp8_dtype(out_dtype)) {
      // For 2x delayed scaling, the scale buffer is shared between rowwise and columnwise scaling
      auto &tmp_buf = (scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING)
                          ? scale_inv_buf
                          : colwise_scale_inv_buf;
      if (scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING) {
        output_tensor.set_columnwise_scale_inv(
            tmp_buf->untyped_data(), convert_ffi_datatype_to_te_dtype(tmp_buf->element_type()),
            std::vector<size_t>{1});
      } else {
        output_tensor.set_columnwise_scale_inv(
            tmp_buf->untyped_data(), convert_ffi_datatype_to_te_dtype(tmp_buf->element_type()),
            std::vector<size_t>{
                product(tmp_buf->dimensions(), 0, flatten_axis),
                product(tmp_buf->dimensions(), flatten_axis, tmp_buf->dimensions().size())});
      }
    }
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
    case NVTE_Activation_Type::CLAMPED_SWIGLU:
      nvte_clamped_swiglu(input_tensor.data(), output_tensor.data(), swiglu_limit, swiglu_alpha,
                          stream);
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
                                  .Arg<Buffer_Type>()      // amax
                                  .Ret<Buffer_Type>()      // output
                                  .Ret<Buffer_Type>()      // colwise output
                                  .Ret<Buffer_Type>()      // scale_inv
                                  .Ret<Buffer_Type>()      // scale_inv colwise
                                  .Ret<Buffer_Type>()      // updated_amax
                                  .Attr<int64_t>("act_enum")
                                  .Attr<JAXX_Scaling_Mode>("scaling_mode")
                                  .Attr<JAXX_Quantize_Layout>("quantize_layout")
                                  .Attr<ActivationConfig>("act_params")
                                  .Attr<bool>("output_amax_when_no_scaling"),
                              FFI_CudaGraph_Traits);

Error_Type ActLuInitializeFFI(cudaStream_t stream, Buffer_Type input_buf, Buffer_Type scale_buf,
                              Buffer_Type amax_buf, Result_Type output_buf,
                              Result_Type colwise_output_buf, Result_Type scale_inv_buf,
                              Result_Type colwise_scale_inv_buf, Result_Type updated_amax_buf,
                              int64_t act_enum, JAXX_Scaling_Mode scaling_mode,
                              JAXX_Quantize_Layout quantize_layout, ActivationConfig act_params,
                              bool output_amax_when_no_scaling) {
  return wrapInStreamCapture(std::function(ActLuFFI), stream, input_buf, scale_buf, amax_buf,
                             output_buf, colwise_output_buf, scale_inv_buf, colwise_scale_inv_buf,
                             updated_amax_buf, act_enum, scaling_mode, quantize_layout, act_params,
                             output_amax_when_no_scaling);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(ActLuInitializeHandler, ActLuInitializeFFI,
                              FFI::Bind<FFI_Initialize>()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // input
                                  .Arg<Buffer_Type>()      // scale
                                  .Arg<Buffer_Type>()      // amax
                                  .Ret<Buffer_Type>()      // output
                                  .Ret<Buffer_Type>()      // colwise output
                                  .Ret<Buffer_Type>()      // scale_inv
                                  .Ret<Buffer_Type>()      // scale_inv colwise
                                  .Ret<Buffer_Type>()      // updated_amax
                                  .Attr<int64_t>("act_enum")
                                  .Attr<JAXX_Scaling_Mode>("scaling_mode")
                                  .Attr<JAXX_Quantize_Layout>("quantize_layout")
                                  .Attr<ActivationConfig>("act_params")
                                  .Attr<bool>("output_amax_when_no_scaling"));

pybind11::tuple GetDActDBiasQuantizeWorkspaceSizes(size_t batch_size, size_t hidden_size,
                                                   DType in_dtype, DType out_dtype,
                                                   JAXX_Scaling_Mode scaling_mode,
                                                   JAXX_Quantize_Layout quantize_layout) {
  auto input_shape = std::vector<size_t>{batch_size, hidden_size};
  auto dact_input_shape = std::vector<size_t>{batch_size, hidden_size};
  auto output_shape = std::vector<size_t>{batch_size, hidden_size};
  auto output_trans_shape = std::vector<size_t>{hidden_size, batch_size};
  auto dbias_shape = std::vector<size_t>{hidden_size};

  NVTE_CHECK(
      scaling_mode != JAXX_Scaling_Mode::CURRENT_TENSOR_SCALING,
      "Current tensor scaling does not support fused operations yet. Please call this primitive "
      "in higher-precision then quantize with current scaling.");

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
  auto output_tensor = TensorWrapper(get_nvte_scaling_mode(scaling_mode));
  output_tensor.set_rowwise_data(reinterpret_cast<void *>(&temp), out_dtype, output_shape);
  // Only the pointers will be checked for scale_inv, thus the shapes do not matter
  if (is_fp8_dtype(out_dtype)) {
    output_tensor.set_rowwise_scale_inv(reinterpret_cast<void *>(&temp), DType::kFloat32,
                                        std::vector<size_t>{1});
  }

  if (is_quantize_2x2x(quantize_layout)) {
    auto &tmp_shape = scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING ? output_trans_shape
                                                                                : output_shape;
    output_tensor.set_columnwise_data(reinterpret_cast<void *>(&temp), out_dtype, tmp_shape);

    // Only the pointers will be checked for scale_inv, thus the shapes do not matter
    if (is_fp8_dtype(out_dtype)) {
      output_tensor.set_columnwise_scale_inv(reinterpret_cast<void *>(&temp), DType::kFloat32,
                                             std::vector<size_t>{1});
    }
  }

  if (is_fp8_dtype(out_dtype) && scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING) {
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
                                  Buffer_Type amax_buf, Result_Type output_buf,
                                  Result_Type colwise_output_buf, Result_Type scale_inv_buf,
                                  Result_Type colwise_scale_inv_buf, Result_Type updated_amax_buf,
                                  Result_Type dbias_buf, Result_Type workspace_buf,
                                  JAXX_Scaling_Mode scaling_mode, int64_t act_enum,
                                  JAXX_Quantize_Layout quantize_layout, bool is_dbias,
                                  ActivationConfig act_params, bool output_amax_when_no_scaling) {
  // parameters for clamped swiglu used in GPT OSS
  auto swiglu_limit = act_params.clamped_swiglu.limit;
  auto swiglu_alpha = act_params.clamped_swiglu.alpha;

  auto in_dtype = convert_ffi_datatype_to_te_dtype(input_buf.element_type());
  auto out_dtype = convert_ffi_datatype_to_te_dtype(output_buf->element_type());
  auto workspace_dtype = convert_ffi_datatype_to_te_dtype(workspace_buf->element_type());

  auto *input = input_buf.untyped_data();
  auto *act_input = act_input_buf.untyped_data();
  float *scale = reinterpret_cast<float *>(scale_buf.untyped_data());
  float *amax = reinterpret_cast<float *>(amax_buf.untyped_data());
  auto *updated_amax = reinterpret_cast<float *>(updated_amax_buf->untyped_data());
  NVTE_CHECK(amax == updated_amax && amax != nullptr, "amax and updated_amax should be aliased");

  auto act_type = static_cast<NVTE_Activation_Type>(act_enum);
  auto flatten_axis = output_buf->dimensions().size() - 2;  // output has act axis

  NVTE_CHECK(
      scaling_mode != JAXX_Scaling_Mode::CURRENT_TENSOR_SCALING,
      "Current tensor scaling does not support fused operations yet. Please call this primitive "
      "in higher-precision then quantize with current scaling.");

  auto *output = output_buf->untyped_data();
  auto *colwise_output = colwise_output_buf->untyped_data();
  auto *dbias = dbias_buf->untyped_data();
  void *workspace = workspace_buf->untyped_data();

  auto input_dims = input_buf.dimensions();
  auto act_input_dims = act_input_buf.dimensions();
  auto workspace_dims = workspace_buf->dimensions();
  // m = x_batch_size = reduce(operator.mul, x_shape[:-2]), x_shape == act_input_dims
  // n = ir_dz_shape[-1] * act_len, ir_dz_shape == input_dims
  auto act_len = act_input_dims[act_input_dims.size() - 2];
  NVTE_CHECK(act_len == 1 || act_len == 2,
             "The value of the activation dimension (axis=-2) must be 1 for non-gated or 2 for "
             "gated activation, got ",
             act_len);

  auto m = product(act_input_dims, 0, act_input_dims.size() - 2);
  auto n = input_dims.back();

  auto input_shape = std::vector<size_t>{m, static_cast<size_t>(n)};
  auto act_input_shape = std::vector<size_t>{m, static_cast<size_t>(n * act_len)};
  auto output_shape = std::vector<size_t>{m, static_cast<size_t>(n * act_len)};
  auto output_trans_shape = std::vector<size_t>{static_cast<size_t>(n * act_len), m};
  auto dbias_shape = std::vector<size_t>{static_cast<size_t>(n * act_len)};
  std::vector<size_t> workspace_shape(workspace_dims.begin(), workspace_dims.end());

  auto input_tensor =
      TensorWrapper(input, input_shape, convert_ffi_datatype_to_te_dtype(input_buf.element_type()));
  auto act_input_tensor = TensorWrapper(
      act_input, act_input_shape, convert_ffi_datatype_to_te_dtype(act_input_buf.element_type()));

  auto output_tensor = TensorWrapper(get_nvte_scaling_mode(scaling_mode));
  output_tensor.set_rowwise_data(output, out_dtype, output_shape);
  if (scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING ||
      (scaling_mode == JAXX_Scaling_Mode::NO_SCALING && output_amax_when_no_scaling)) {
    output_tensor.set_amax(amax, DType::kFloat32, std::vector<size_t>{1});
  }
  if (is_fp8_dtype(out_dtype)) {
    if (scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING) {
      NVTE_CHECK(scale != nullptr, "scale must be provided for delayed tensor scaling");
      output_tensor.set_scale(scale, DType::kFloat32, std::vector<size_t>{1});
      output_tensor.set_rowwise_scale_inv(
          scale_inv_buf->untyped_data(),
          convert_ffi_datatype_to_te_dtype(scale_inv_buf->element_type()), std::vector<size_t>{1});
    } else {
      output_tensor.set_rowwise_scale_inv(
          scale_inv_buf->untyped_data(),
          convert_ffi_datatype_to_te_dtype(scale_inv_buf->element_type()),
          std::vector<size_t>{product(scale_inv_buf->dimensions(), 0, flatten_axis),
                              product(scale_inv_buf->dimensions(), flatten_axis,
                                      scale_inv_buf->dimensions().size())});
    }
  }

  if (is_quantize_2x2x(quantize_layout)) {
    auto &tmp_shape = (scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING)
                          ? output_trans_shape
                          : output_shape;
    output_tensor.set_columnwise_data(colwise_output, out_dtype, tmp_shape);

    if (is_fp8_dtype(out_dtype)) {
      // For 2x delayed scaling, the scale buffer is shared between rowwise and columnwise scaling
      auto &tmp_buf = (scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING)
                          ? scale_inv_buf
                          : colwise_scale_inv_buf;
      if (scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING) {
        output_tensor.set_columnwise_scale_inv(
            tmp_buf->untyped_data(), convert_ffi_datatype_to_te_dtype(tmp_buf->element_type()),
            std::vector<size_t>{1});
      } else {
        output_tensor.set_columnwise_scale_inv(
            tmp_buf->untyped_data(), convert_ffi_datatype_to_te_dtype(tmp_buf->element_type()),
            std::vector<size_t>{
                product(tmp_buf->dimensions(), 0, flatten_axis),
                product(tmp_buf->dimensions(), flatten_axis, tmp_buf->dimensions().size())});
      }
    }
  }

  auto dbias_tensor = TensorWrapper(dbias, dbias_shape, in_dtype);
  auto workspace_tensor = TensorWrapper(workspace, workspace_shape, workspace_dtype);

  // fused_dgated_dbias is not available, so we use dact_lu + quantize_dbias in Python instead
  NVTE_CHECK(!(act_len == 2 && is_dbias), "Unsupported DGatedActedDBias Fusion!");
  NVTE_CHECK(!(scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING &&
               is_quantize_2x2x(quantize_layout) && act_len == 2),
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
      case NVTE_Activation_Type::CLAMPED_SWIGLU:
        nvte_clamped_dswiglu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(),
                             swiglu_limit, swiglu_alpha, stream);
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
                                  .Arg<Buffer_Type>()      // amax
                                  .Ret<Buffer_Type>()      // output
                                  .Ret<Buffer_Type>()      // colwise output
                                  .Ret<Buffer_Type>()      // scale_inv
                                  .Ret<Buffer_Type>()      // scale_inv colwise
                                  .Ret<Buffer_Type>()      // amax
                                  .Ret<Buffer_Type>()      // dbias
                                  .Ret<Buffer_Type>()      // wkspace
                                  .Attr<JAXX_Scaling_Mode>("scaling_mode")
                                  .Attr<int64_t>("act_enum")
                                  .Attr<JAXX_Quantize_Layout>("quantize_layout")
                                  .Attr<bool>("is_dbias")
                                  .Attr<ActivationConfig>("act_params")
                                  .Attr<bool>("output_amax_when_no_scaling"),
                              FFI_CudaGraph_Traits);

Error_Type DActLuDBiasQuantizeInitializeFFI(
    cudaStream_t stream, Buffer_Type input_buf, Buffer_Type act_input_buf, Buffer_Type scale_buf,
    Buffer_Type amax_buf, Result_Type output_buf, Result_Type colwise_output_buf,
    Result_Type scale_inv_buf, Result_Type colwise_scale_inv_buf, Result_Type updated_amax_buf,
    Result_Type dbias_buf, Result_Type workspace_buf, JAXX_Scaling_Mode scaling_mode,
    int64_t act_enum, JAXX_Quantize_Layout quantize_layout, bool is_dbias,
    ActivationConfig act_params, bool output_amax_when_no_scaling) {
  return wrapInStreamCapture(std::function(DActLuDBiasQuantizeFFI), stream, input_buf,
                             act_input_buf, scale_buf, amax_buf, output_buf, colwise_output_buf,
                             scale_inv_buf, colwise_scale_inv_buf, updated_amax_buf, dbias_buf,
                             workspace_buf, scaling_mode, act_enum, quantize_layout, is_dbias,
                             act_params, output_amax_when_no_scaling);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(DActLuDBiasQuantizeInitializeHandler,
                              DActLuDBiasQuantizeInitializeFFI,
                              FFI::Bind<FFI_Initialize>()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // input
                                  .Arg<Buffer_Type>()      // act input
                                  .Arg<Buffer_Type>()      // scale
                                  .Arg<Buffer_Type>()      // amax
                                  .Ret<Buffer_Type>()      // output
                                  .Ret<Buffer_Type>()      // colwise output
                                  .Ret<Buffer_Type>()      // scale_inv
                                  .Ret<Buffer_Type>()      // scale_inv colwise
                                  .Ret<Buffer_Type>()      // updated_amax
                                  .Ret<Buffer_Type>()      // dbias
                                  .Ret<Buffer_Type>()      // wkspace
                                  .Attr<JAXX_Scaling_Mode>("scaling_mode")
                                  .Attr<int64_t>("act_enum")
                                  .Attr<JAXX_Quantize_Layout>("quantize_layout")
                                  .Attr<bool>("is_dbias")
                                  .Attr<ActivationConfig>("act_params")
                                  .Attr<bool>("output_amax_when_no_scaling"));

}  // namespace jax
}  // namespace transformer_engine
