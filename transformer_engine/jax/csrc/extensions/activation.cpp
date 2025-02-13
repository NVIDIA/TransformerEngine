/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include "transformer_engine/activation.h"

#include "extensions.h"
#include "transformer_engine/cast.h"
#include "transformer_engine/transpose.h"
#include "xla/ffi/api/c_api.h"

namespace transformer_engine {
namespace jax {

// TODO: We won't need this function anymore when we move to the new XLA custom calls
size_t get_activation_len(NVTE_Activation_Type activation_enum) {
  switch (activation_enum) {
    case NVTE_Activation_Type::GELU:
      return 1;
    case NVTE_Activation_Type::GEGLU:
      return 2;
    case NVTE_Activation_Type::SILU:
      return 1;
    case NVTE_Activation_Type::SWIGLU:
      return 2;
    case NVTE_Activation_Type::RELU:
      return 1;
    case NVTE_Activation_Type::REGLU:
      return 2;
    case NVTE_Activation_Type::QGELU:
      return 1;
    case NVTE_Activation_Type::QGEGLU:
      return 2;
    case NVTE_Activation_Type::SRELU:
      return 1;
    case NVTE_Activation_Type::SREGLU:
      return 2;
    default:
      NVTE_ERROR("Unsupported ActivationEnum");
      break;
      return -1;
  }
}

void ActLuImpl(void *input, size_t m, size_t n, DType in_dtype, DType out_dtype, float *scale,
               cudaStream_t stream, float *scale_inverse, float *amax, void *output,
               NVTE_Activation_Type act_enum, size_t act_len) {
  auto input_shape = std::vector<size_t>{m, n * act_len};
  auto output_shape = std::vector<size_t>{m, n};
  auto input_tensor = TensorWrapper(input, input_shape, static_cast<DType>(in_dtype));
  auto output_tensor = TensorWrapper(output, output_shape, static_cast<DType>(out_dtype), amax,
                                     scale, scale_inverse);
  switch (act_enum) {
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
}

void ActLu(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
  auto *input = buffers[0];
  auto *output = buffers[1];

  const auto &desc = *UnpackOpaque<CustomCallCommonDescriptor>(opaque, opaque_len);
  auto m = desc.shape.dims[0];
  auto n = desc.shape.dims[1];
  auto act_enum = static_cast<NVTE_Activation_Type>(desc.act_enum);
  auto act_len = get_activation_len(act_enum);

  ActLuImpl(input, m, n, desc.in_dtype, desc.out_dtype, nullptr, stream, nullptr, nullptr, output,
            act_enum, act_len);
}

Error_Type ActLuFFI(cudaStream_t stream, Buffer_Type input_buf, Result_Type output_buf,
                    int64_t act_enum) {
  auto in_dtype = convert_ffi_datatype_to_te_dtype(input_buf.element_type());
  auto out_dtype = convert_ffi_datatype_to_te_dtype(output_buf->element_type());

  auto *input = input_buf.untyped_data();
  auto *output = output_buf->untyped_data();

  auto input_dims = input_buf.dimensions();
  auto m = product(input_dims, 0, input_dims.size() - 2);
  auto n = input_dims.back();
  auto act_len = input_dims.end()[-2];
  auto act_type = static_cast<NVTE_Activation_Type>(act_enum);

  ActLuImpl(input, m, n, in_dtype, out_dtype, nullptr, stream, nullptr, nullptr, output, act_type,
            act_len);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(ActLuHandler, ActLuFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // input
                                  .Ret<Buffer_Type>()      // output
                                  .Attr<int64_t>("act_enum"),
                              FFI_CudaGraph_Traits);

void ActLuFP8(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
  auto *input = buffers[0];
  float *amax = reinterpret_cast<float *>(buffers[1]);
  float *scale = reinterpret_cast<float *>(buffers[2]);
  float *scale_inv = reinterpret_cast<float *>(buffers[3]);
  auto *output = buffers[4];
  float *amax_out = reinterpret_cast<float *>(buffers[5]);
  NVTE_CHECK(amax == amax_out, "amax not bound to amax_out in TE/JAX ActLuFP8 primitive.");

  const auto &desc = *UnpackOpaque<CustomCallCommonDescriptor>(opaque, opaque_len);
  if (!use_fp8(desc.out_dtype)) {
    scale = nullptr;
    scale_inv = nullptr;
    amax_out = nullptr;
  }
  auto m = desc.shape.dims[0];
  auto n = desc.shape.dims[1];
  auto act_enum = static_cast<NVTE_Activation_Type>(desc.act_enum);
  auto act_len = get_activation_len(act_enum);

  ActLuImpl(input, m, n, desc.in_dtype, desc.out_dtype, scale, stream, scale_inv, amax_out, output,
            act_enum, act_len);
}

Error_Type ActLuFP8FFI(cudaStream_t stream, Buffer_Type input_buf, Buffer_Type amax_buf,
                       Buffer_Type scale_buf, Buffer_Type scale_inv_buf, Result_Type output_buf,
                       Result_Type amax_out_buf, int64_t act_enum) {
  auto in_dtype = convert_ffi_datatype_to_te_dtype(input_buf.element_type());
  auto out_dtype = convert_ffi_datatype_to_te_dtype(output_buf->element_type());

  auto *input = input_buf.untyped_data();
  float *amax = reinterpret_cast<float *>(amax_buf.untyped_data());
  float *scale = reinterpret_cast<float *>(scale_buf.untyped_data());
  float *scale_inv = reinterpret_cast<float *>(scale_inv_buf.untyped_data());

  auto *output = output_buf->untyped_data();
  float *amax_out = reinterpret_cast<float *>(amax_out_buf->untyped_data());
  NVTE_CHECK(amax == amax_out, "amax not bound to amax_out in TE/JAX ActLuFP8 primitive.");

  if (!use_fp8(out_dtype)) {
    scale = nullptr;
    scale_inv = nullptr;
    amax_out = nullptr;
  }

  auto input_dims = input_buf.dimensions();
  auto m = product(input_dims, 0, input_dims.size() - 2);
  auto n = input_dims.back();
  auto act_len = input_dims.end()[-2];
  auto act_type = static_cast<NVTE_Activation_Type>(act_enum);

  ActLuImpl(input, m, n, in_dtype, out_dtype, scale, stream, scale_inv, amax_out, output, act_type,
            act_len);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(ActLuFP8Handler, ActLuFP8FFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // input
                                  .Arg<Buffer_Type>()      // amax
                                  .Arg<Buffer_Type>()      // scale
                                  .Arg<Buffer_Type>()      // scale_inv
                                  .Ret<Buffer_Type>()      // output
                                  .Ret<Buffer_Type>()      // amax_out
                                  .Attr<int64_t>("act_enum"),
                              FFI_CudaGraph_Traits);

void DActLu(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
  auto *input = buffers[0];
  auto *act_input = buffers[1];
  auto *output = buffers[2];

  const auto &desc = *UnpackOpaque<CustomCallCommonDescriptor>(opaque, opaque_len);
  auto m = desc.shape.dims[0];
  auto n = desc.shape.dims[1];
  auto act_enum = static_cast<NVTE_Activation_Type>(desc.act_enum);

  auto act_len = get_activation_len(act_enum);
  auto input_shape = std::vector<size_t>{m, n};
  auto act_input_shape = std::vector<size_t>{m, n * act_len};
  auto output_shape = std::vector<size_t>{m, n * act_len};

  auto input_tensor = TensorWrapper(input, input_shape, desc.in_dtype);
  auto act_input_tensor = TensorWrapper(act_input, act_input_shape, desc.in_dtype);
  auto output_tensor = TensorWrapper(output, output_shape, desc.out_dtype);

  switch (act_enum) {
    case NVTE_Activation_Type::GELU:
      nvte_dgelu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::GEGLU:
      nvte_dgeglu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::SILU:
      nvte_dsilu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::SWIGLU:
      nvte_dswiglu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::RELU:
      nvte_drelu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::REGLU:
      nvte_dreglu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::QGELU:
      nvte_dqgelu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::QGEGLU:
      nvte_dqgeglu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::SRELU:
      nvte_dsrelu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::SREGLU:
      nvte_dsreglu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
      break;
    default:
      NVTE_ERROR("Unsupported ActivationEnum");
      break;
  }
}

Error_Type DActLuFFI(cudaStream_t stream, Buffer_Type input_buf, Buffer_Type act_input_buf,
                     Result_Type output_buf, int64_t act_enum) {
  auto in_dtype = convert_ffi_datatype_to_te_dtype(input_buf.element_type());
  auto out_dtype = convert_ffi_datatype_to_te_dtype(output_buf->element_type());

  auto *input = input_buf.untyped_data();
  auto *act_input = act_input_buf.untyped_data();
  auto *output = output_buf->untyped_data();

  auto act_input_dims = act_input_buf.dimensions();
  auto m = static_cast<size_t>(product(act_input_dims, 0, act_input_dims.size() - 2));
  auto n = static_cast<size_t>(act_input_dims.back());
  auto act_len = act_input_dims.end()[-2];

  auto input_shape = std::vector<size_t>{m, n};
  auto act_input_shape = std::vector<size_t>{m, n * act_len};
  auto output_shape = std::vector<size_t>{m, n * act_len};

  auto input_tensor = TensorWrapper(input, input_shape, static_cast<DType>(in_dtype));
  auto act_input_tensor = TensorWrapper(act_input, act_input_shape, static_cast<DType>(in_dtype));
  auto output_tensor = TensorWrapper(output, output_shape, static_cast<DType>(out_dtype));

  auto act_type = static_cast<NVTE_Activation_Type>(act_enum);
  switch (act_type) {
    case NVTE_Activation_Type::GELU:
      nvte_dgelu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::GEGLU:
      nvte_dgeglu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::SILU:
      nvte_dsilu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::SWIGLU:
      nvte_dswiglu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::RELU:
      nvte_drelu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::REGLU:
      nvte_dreglu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::QGELU:
      nvte_dqgelu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::QGEGLU:
      nvte_dqgeglu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::SRELU:
      nvte_dsrelu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::SREGLU:
      nvte_dsreglu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(), stream);
      break;
    default:
      NVTE_ERROR("Unsupported ActivationEnum");
      break;
  }
  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(DActLuHandler, DActLuFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // input
                                  .Arg<Buffer_Type>()      // act_input
                                  .Ret<Buffer_Type>()      // output
                                  .Attr<int64_t>("act_enum"),
                              FFI_CudaGraph_Traits);

pybind11::tuple GetDActDBiasCastTransposeWorkspaceSizes(size_t batch_size, size_t hidden_size,
                                                        DType in_dtype, DType out_dtype) {
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
  auto output_tensor = TensorWrapper();
  output_tensor.set_rowwise_data(reinterpret_cast<void *>(&temp), out_dtype, output_shape);
  output_tensor.set_columnwise_data(reinterpret_cast<void *>(&temp), out_dtype, output_trans_shape);
  auto dbias_tensor = TensorWrapper(reinterpret_cast<void *>(&temp), dbias_shape, in_dtype);

  TensorWrapper dummy_workspace;

  // For now, all dbias_dact(-s) have the same workspace size
  nvte_quantize_dbias_dgelu(input_tensor.data(), dact_input_tensor.data(), output_tensor.data(),
                            dbias_tensor.data(), dummy_workspace.data(), nullptr);

  auto work_shape = MakeShapeVector(dummy_workspace.shape());
  return pybind11::make_tuple(std::make_pair(work_shape, dummy_workspace.dtype()));
}

void DActLuDBiasCastTranspose(cudaStream_t stream, void **buffers, const char *opaque,
                              size_t opaque_len) {
  auto *input = buffers[0];
  auto *act_input = buffers[1];
  float *amax = reinterpret_cast<float *>(buffers[2]);
  float *scale = reinterpret_cast<float *>(buffers[3]);
  float *scale_inv = reinterpret_cast<float *>(buffers[4]);
  auto *output = buffers[5];
  auto *output_trans = buffers[6];
  auto *dbias = buffers[7];
  float *amax_out = reinterpret_cast<float *>(buffers[8]);
  void *workspace_ptr = buffers[9];

  const auto &desc = *UnpackOpaque<CustomCallCommonWkDescriptor>(opaque, opaque_len);
  NVTE_CHECK(amax == amax_out,
             "amax not bound to amax_out in TE/JAX DActLuDBiasCastTranspose primitive.");
  if (!use_fp8(desc.out_dtype)) {
    scale = nullptr;
    scale_inv = nullptr;
    amax_out = nullptr;
  }
  auto m = desc.shape.dims[0];
  auto n = desc.shape.dims[1];
  auto act_enum = static_cast<NVTE_Activation_Type>(desc.act_enum);

  auto input_shape = std::vector<size_t>{m, n};
  auto act_input_shape = std::vector<size_t>{m, n};
  auto output_shape = std::vector<size_t>{m, n};
  auto output_trans_shape = std::vector<size_t>{n, m};
  auto dbias_shape = std::vector<size_t>{n};

  auto input_tensor = TensorWrapper(input, input_shape, desc.in_dtype);
  auto act_input_tensor = TensorWrapper(act_input, act_input_shape, desc.in_dtype);
  auto output_tensor =
      TensorWrapper(output, output_shape, desc.out_dtype, amax_out, scale, scale_inv);
  output_tensor.set_columnwise_data(output_trans, desc.out_dtype, output_trans_shape);
  output_tensor.set_columnwise_scale_inv(scale_inv, DType::kFloat32, std::vector<size_t>{1});
  auto dbias_tensor = TensorWrapper(dbias, dbias_shape, desc.in_dtype);

  auto workspace = TensorWrapper(workspace_ptr, desc.wkshape.to_vector(), desc.wk_dtype);

  switch (act_enum) {
    case NVTE_Activation_Type::GELU:
      nvte_quantize_dbias_dgelu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(),
                                dbias_tensor.data(), workspace.data(), stream);
      break;
    case NVTE_Activation_Type::SILU:
      nvte_quantize_dbias_dsilu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(),
                                dbias_tensor.data(), workspace.data(), stream);
      break;
    case NVTE_Activation_Type::RELU:
      nvte_quantize_dbias_drelu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(),
                                dbias_tensor.data(), workspace.data(), stream);
      break;
    case NVTE_Activation_Type::QGELU:
      nvte_quantize_dbias_dqgelu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(),
                                 dbias_tensor.data(), workspace.data(), stream);
      break;
    case NVTE_Activation_Type::SRELU:
      nvte_quantize_dbias_dsrelu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(),
                                 dbias_tensor.data(), workspace.data(), stream);
      break;
    default:
      NVTE_ERROR("Unsupported ActivationEnum");
      break;
  }
}

Error_Type DActLuDBiasCastTransposeFFI(cudaStream_t stream, Buffer_Type input_buf,
                                       Buffer_Type act_input_buf, Buffer_Type amax_buf,
                                       Buffer_Type scale_buf, Buffer_Type scale_inv_buf,
                                       Result_Type output_buf, Result_Type output_trans_buf,
                                       Result_Type dbias_buf, Result_Type amax_out_buf,
                                       Result_Type workspace_buf, int64_t act_enum) {
  auto in_dtype = convert_ffi_datatype_to_te_dtype(input_buf.element_type());
  auto out_dtype = convert_ffi_datatype_to_te_dtype(output_buf->element_type());
  auto workspace_dtype = convert_ffi_datatype_to_te_dtype(workspace_buf->element_type());

  auto *input = input_buf.untyped_data();
  auto *act_input = act_input_buf.untyped_data();
  float *amax = reinterpret_cast<float *>(amax_buf.untyped_data());
  float *scale = reinterpret_cast<float *>(scale_buf.untyped_data());
  float *scale_inv = reinterpret_cast<float *>(scale_inv_buf.untyped_data());
  auto *output = output_buf->untyped_data();
  auto *output_trans = output_trans_buf->untyped_data();
  auto *dbias = dbias_buf->untyped_data();
  float *amax_out = reinterpret_cast<float *>(amax_out_buf->untyped_data());
  void *workspace = workspace_buf->untyped_data();
  NVTE_CHECK(amax == amax_out,
             "amax not bound to amax_out in TE/JAX DActLuDBiasCastTranspose primitive.");
  if (!use_fp8(out_dtype)) {
    scale = nullptr;
    scale_inv = nullptr;
    amax_out = nullptr;
  }

  auto input_dims = input_buf.dimensions();
  auto act_input_dims = act_input_buf.dimensions();
  auto workspace_dims = workspace_buf->dimensions();
  // m = x_batch_size = reduce(operator.mul, x_shape[:-2]), x_shape == act_input_dims
  // n = ir_dz_shape[-1], ir_dz_shape == input_dims
  auto input_ranks = input_dims.size();
  auto m = product(act_input_dims, 0, act_input_dims.size() - 2);
  auto n = product(input_dims, input_ranks - 1, input_ranks);
  auto input_shape = std::vector<size_t>{m, n};
  auto act_input_shape = std::vector<size_t>{m, n};
  auto output_shape = std::vector<size_t>{m, n};
  auto output_trans_shape = std::vector<size_t>{n, m};
  auto dbias_shape = std::vector<size_t>{n};
  std::vector<size_t> workspace_shape(workspace_dims.begin(), workspace_dims.end());

  auto input_tensor = TensorWrapper(input, input_shape, in_dtype);
  auto act_input_tensor = TensorWrapper(act_input, input_shape, in_dtype);
  auto output_tensor = TensorWrapper(output, output_shape, out_dtype, amax_out, scale, scale_inv);
  output_tensor.set_columnwise_data(output_trans, out_dtype, output_trans_shape);
  output_tensor.set_columnwise_scale_inv(scale_inv, DType::kFloat32, std::vector<size_t>{1});
  auto dbias_tensor = TensorWrapper(dbias, dbias_shape, in_dtype);
  auto workspace_tensor = TensorWrapper(workspace, workspace_shape, workspace_dtype);

  auto act_type = static_cast<NVTE_Activation_Type>(act_enum);
  switch (act_type) {
    case NVTE_Activation_Type::GELU:
      nvte_quantize_dbias_dgelu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(),
                                dbias_tensor.data(), workspace_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::SILU:
      nvte_quantize_dbias_dsilu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(),
                                dbias_tensor.data(), workspace_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::RELU:
      nvte_quantize_dbias_drelu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(),
                                dbias_tensor.data(), workspace_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::QGELU:
      nvte_quantize_dbias_dqgelu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(),
                                 dbias_tensor.data(), workspace_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::SRELU:
      nvte_quantize_dbias_dsrelu(input_tensor.data(), act_input_tensor.data(), output_tensor.data(),
                                 dbias_tensor.data(), workspace_tensor.data(), stream);
      break;
    default:
      NVTE_ERROR("Unsupported ActivationEnum");
      break;
  }
  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(DActLuDBiasCastTransposeHandler, DActLuDBiasCastTransposeFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // input
                                  .Arg<Buffer_Type>()      // act_input
                                  .Arg<Buffer_Type>()      // amax
                                  .Arg<Buffer_Type>()      // scale
                                  .Arg<Buffer_Type>()      // scale_inv
                                  .Ret<Buffer_Type>()      // output
                                  .Ret<Buffer_Type>()      // output_trans
                                  .Ret<Buffer_Type>()      // dbias
                                  .Ret<Buffer_Type>()      // amax_out
                                  .Ret<Buffer_Type>()      // workspace
                                  .Attr<int64_t>("act_enum"),
                              FFI_CudaGraph_Traits);

void DGatedActLuCastTranspose(cudaStream_t stream, void **buffers, const char *opaque,
                              size_t opaque_len) {
  auto *input = buffers[0];
  auto *act_input = buffers[1];
  float *amax = reinterpret_cast<float *>(buffers[2]);
  float *scale = reinterpret_cast<float *>(buffers[3]);
  float *scale_inv = reinterpret_cast<float *>(buffers[4]);
  auto *output = buffers[5];
  auto *output_trans = buffers[6];
  float *amax_out = reinterpret_cast<float *>(buffers[7]);

  const auto &desc = *UnpackOpaque<CustomCallCommonDescriptor>(opaque, opaque_len);
  NVTE_CHECK(amax == amax_out,
             "amax not bound to amax_out in TE/JAX DGatedActLuCastTranspose primitive.");
  if (!use_fp8(desc.out_dtype)) {
    scale = nullptr;
    scale_inv = nullptr;
    amax_out = nullptr;
  }
  auto m = desc.shape.dims[0];
  auto n = desc.shape.dims[1];
  auto act_enum = static_cast<NVTE_Activation_Type>(desc.act_enum);

  auto input_shape = desc.shape.to_vector();
  auto act_input_shape = std::vector<size_t>{m, n * 2};
  auto output_shape = std::vector<size_t>{m, n * 2};
  auto output_trans_shape = std::vector<size_t>{n * 2, m};

  auto input_tensor = TensorWrapper(input, input_shape, desc.in_dtype);
  auto act_input_tensor = TensorWrapper(act_input, act_input_shape, desc.in_dtype);
  auto output_tensor =
      TensorWrapper(output, output_shape, desc.out_dtype, amax_out, scale, scale_inv);
  output_tensor.set_columnwise_data(output_trans, desc.out_dtype, output_trans_shape);
  output_tensor.set_columnwise_scale_inv(scale_inv, DType::kFloat32, std::vector<size_t>{1});

  switch (act_enum) {
    case NVTE_Activation_Type::GEGLU:
      nvte_dgeglu_cast_transpose(input_tensor.data(), act_input_tensor.data(), output_tensor.data(),
                                 stream);
      break;
    case NVTE_Activation_Type::SWIGLU:
      nvte_dswiglu_cast_transpose(input_tensor.data(), act_input_tensor.data(),
                                  output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::REGLU:
      nvte_dreglu_cast_transpose(input_tensor.data(), act_input_tensor.data(), output_tensor.data(),
                                 stream);
      break;
    case NVTE_Activation_Type::QGEGLU:
      nvte_dqgeglu_cast_transpose(input_tensor.data(), act_input_tensor.data(),
                                  output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::SREGLU:
      nvte_dsreglu_cast_transpose(input_tensor.data(), act_input_tensor.data(),
                                  output_tensor.data(), stream);
      break;
    default:
      NVTE_ERROR("Unsupported ActivationEnum");
      break;
  }
}

Error_Type DGatedActLuCastTransposeFFI(cudaStream_t stream, Buffer_Type input_buf,
                                       Buffer_Type act_input_buf, Buffer_Type amax_buf,
                                       Buffer_Type scale_buf, Buffer_Type scale_inv_buf,
                                       Result_Type output_buf, Result_Type output_trans_buf,
                                       Result_Type amax_out_buf, int64_t act_enum) {
  auto in_dtype = convert_ffi_datatype_to_te_dtype(input_buf.element_type());
  auto out_dtype = convert_ffi_datatype_to_te_dtype(output_buf->element_type());

  auto *input = input_buf.untyped_data();
  auto *act_input = act_input_buf.untyped_data();
  float *amax = reinterpret_cast<float *>(amax_buf.untyped_data());
  float *scale = reinterpret_cast<float *>(scale_buf.untyped_data());
  float *scale_inv = reinterpret_cast<float *>(scale_inv_buf.untyped_data());
  auto *output = output_buf->untyped_data();
  auto *output_trans = output_trans_buf->untyped_data();
  float *amax_out = reinterpret_cast<float *>(amax_out_buf->untyped_data());
  NVTE_CHECK(amax == amax_out,
             "amax not bound to amax_out in TE/JAX DGatedActLuCastTranspose primitive.");
  if (!use_fp8(out_dtype)) {
    scale = nullptr;
    scale_inv = nullptr;
    amax_out = nullptr;
  }

  auto input_dims = input_buf.dimensions();
  auto act_input_dims = act_input_buf.dimensions();
  auto act_input_ranks = act_input_dims.size();
  auto m = product(act_input_dims, 0, act_input_ranks - 2);
  auto n = product(act_input_dims, act_input_ranks - 1, act_input_ranks);
  auto input_shape = std::vector<size_t>{m, n};
  auto act_input_shape = std::vector<size_t>{m, n * 2};
  auto output_shape = std::vector<size_t>{m, n * 2};
  auto output_trans_shape = std::vector<size_t>{n * 2, m};

  auto input_tensor = TensorWrapper(input, input_shape, in_dtype);
  auto act_input_tensor = TensorWrapper(act_input, act_input_shape, in_dtype);
  auto output_tensor = TensorWrapper(output, output_shape, out_dtype, amax_out, scale, scale_inv);
  output_tensor.set_columnwise_data(output_trans, out_dtype, output_trans_shape);
  output_tensor.set_columnwise_scale_inv(scale_inv, DType::kFloat32, std::vector<size_t>{1});

  auto act_type = static_cast<NVTE_Activation_Type>(act_enum);
  switch (act_type) {
    case NVTE_Activation_Type::GEGLU:
      nvte_dgeglu_cast_transpose(input_tensor.data(), act_input_tensor.data(), output_tensor.data(),
                                 stream);
      break;
    case NVTE_Activation_Type::SWIGLU:
      nvte_dswiglu_cast_transpose(input_tensor.data(), act_input_tensor.data(),
                                  output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::REGLU:
      nvte_dreglu_cast_transpose(input_tensor.data(), act_input_tensor.data(), output_tensor.data(),
                                 stream);
      break;
    case NVTE_Activation_Type::QGEGLU:
      nvte_dqgeglu_cast_transpose(input_tensor.data(), act_input_tensor.data(),
                                  output_tensor.data(), stream);
      break;
    case NVTE_Activation_Type::SREGLU:
      nvte_dsreglu_cast_transpose(input_tensor.data(), act_input_tensor.data(),
                                  output_tensor.data(), stream);
      break;
    default:
      NVTE_ERROR("Unsupported ActivationEnum");
      break;
  }
  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(DGatedActLuCastTransposeHandler, DGatedActLuCastTransposeFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // input
                                  .Arg<Buffer_Type>()      // act_input
                                  .Arg<Buffer_Type>()      // amax
                                  .Arg<Buffer_Type>()      // scale
                                  .Arg<Buffer_Type>()      // scale_inv
                                  .Ret<Buffer_Type>()      // output
                                  .Ret<Buffer_Type>()      // output_trans
                                  .Ret<Buffer_Type>()      // amax_out
                                  .Attr<int64_t>("act_enum"),
                              FFI_CudaGraph_Traits);

}  // namespace jax
}  // namespace transformer_engine
