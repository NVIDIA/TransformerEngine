/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/transpose.h"

#include "extensions.h"
#include "xla/ffi/api/ffi.h"

namespace transformer_engine {
namespace jax {

void TransposeImpl(void *input, size_t rows, size_t cols, DType dtype, cudaStream_t stream,
                   void *output) {
  auto input_shape = std::vector<size_t>{rows, cols};
  auto output_shape = std::vector<size_t>{cols, rows};

  auto input_tensor = TensorWrapper(input, input_shape, dtype);
  auto transposed_tensor = TensorWrapper(output, output_shape, dtype);

  nvte_transpose(input_tensor.data(), transposed_tensor.data(), stream);
}

void Transpose(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
  void *input = buffers[0];
  void *output = buffers[1];

  const auto &desc = *UnpackOpaque<CustomCallCommonDescriptor>(opaque, opaque_len);
  auto rows = desc.shape.dims[0];
  auto cols = desc.shape.dims[1];
  assert(desc.in_dtype == desc.out_dtype);
  auto dtype = desc.out_dtype;

  TransposeImpl(input, rows, cols, dtype, stream, output);
}

Error_Type TransposeFFI(cudaStream_t stream, Buffer_Type input_buf, Result_Type output_buf,
                        int64_t transpose_axis) {
  auto in_dtype = convert_ffi_datatype_to_te_dtype(input_buf.element_type());
  auto out_dtype = convert_ffi_datatype_to_te_dtype(output_buf->element_type());

  void *input = input_buf.untyped_data();
  void *output = output_buf->untyped_data();

  auto input_dims = input_buf.dimensions();
  if (transpose_axis < 0) transpose_axis += input_dims.size();
  auto m = std::accumulate(input_dims.begin(), input_dims.begin() + transpose_axis, 1,
                           std::multiplies<>());
  auto n = std::accumulate(input_dims.begin() + transpose_axis, input_dims.end(), 1,
                           std::multiplies<>());
  auto input_shape = std::vector<size_t>{m, n};
  auto output_shape = std::vector<size_t>{n, m};

  auto input_tensor = TensorWrapper(input, input_shape, in_dtype);
  auto output_tensor = TensorWrapper(output, output_shape, out_dtype);

  nvte_transpose(input_tensor.data(), output_tensor.data(), stream);
  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(TransposeHandler, TransposeFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // input
                                  .Ret<Buffer_Type>()      // output
                                  .Attr<int64_t>("transpose_axis"),
                              FFI_CudaGraph_Traits);

void CastTranspose(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
  auto *input = buffers[0];
  float *amax = reinterpret_cast<float *>(buffers[1]);
  float *scale = reinterpret_cast<float *>(buffers[2]);
  float *scale_inv = reinterpret_cast<float *>(buffers[3]);
  auto *input_cast = buffers[4];
  auto *input_cast_trans = buffers[5];
  float *amax_out = reinterpret_cast<float *>(buffers[6]);
  NVTE_CHECK(amax == amax_out, "amax not bound to amax_out in TE/JAX CastTranspose primitive.");

  const auto &desc = *UnpackOpaque<CustomCallCommonDescriptor>(opaque, opaque_len);
  if (!use_fp8(desc.out_dtype)) {
    scale = nullptr;
    scale_inv = nullptr;
    amax_out = nullptr;
  }
  auto m = desc.shape.dims[0];
  auto n = desc.shape.dims[1];
  auto input_shape = std::vector<size_t>{m, n};
  auto input_trans_shape = std::vector<size_t>{n, m};

  auto input_tensor = TensorWrapper(input, input_shape, desc.in_dtype);
  auto input_cast_tensor =
      TensorWrapper(input_cast, input_shape, desc.out_dtype, amax_out, scale, scale_inv);
  auto input_cast_trans_tensor = TensorWrapper(input_cast_trans, input_trans_shape, desc.out_dtype,
                                               amax_out, scale, scale_inv);

  nvte_cast_transpose(input_tensor.data(), input_cast_tensor.data(), input_cast_trans_tensor.data(),
                      stream);
}

Error_Type CastTransposeFFI(cudaStream_t stream, Buffer_Type input_buf, Buffer_Type amax_buf,
                            Buffer_Type scale_buf, Buffer_Type scale_inv_buf,
                            Result_Type input_cast_buf, Result_Type input_cast_trans_buf,
                            Result_Type amax_out_buf, int64_t transpose_axis) {
  auto in_dtype = convert_ffi_datatype_to_te_dtype(input_buf.element_type());
  auto out_dtype = convert_ffi_datatype_to_te_dtype(input_cast_buf->element_type());

  auto *input = input_buf.untyped_data();
  float *amax = reinterpret_cast<float *>(amax_buf.untyped_data());
  float *scale = reinterpret_cast<float *>(scale_buf.untyped_data());
  float *scale_inv = reinterpret_cast<float *>(scale_inv_buf.untyped_data());

  auto *input_cast = input_cast_buf->untyped_data();
  auto *input_cast_trans = input_cast_trans_buf->untyped_data();
  float *amax_out = reinterpret_cast<float *>(amax_out_buf->untyped_data());
  NVTE_CHECK(amax == amax_out, "amax not bound to amax_out in TE/JAX CastTranspose primitive.");

  if (!use_fp8(out_dtype)) {
    scale = nullptr;
    scale_inv = nullptr;
    amax_out = nullptr;
  }

  auto input_dims = input_buf.dimensions();
  if (transpose_axis < 0) transpose_axis += input_dims.size();
  auto m = std::accumulate(input_dims.begin(), input_dims.begin() + transpose_axis, 1,
                           std::multiplies<>());
  auto n = std::accumulate(input_dims.begin() + transpose_axis, input_dims.end(), 1,
                           std::multiplies<>());
  auto input_shape = std::vector<size_t>{m, n};
  auto input_trans_shape = std::vector<size_t>{n, m};

  auto input_tensor = TensorWrapper(input, input_shape, in_dtype);
  auto input_cast_tensor =
      TensorWrapper(input_cast, input_shape, out_dtype, amax_out, scale, scale_inv);
  auto input_cast_trans_tensor =
      TensorWrapper(input_cast_trans, input_trans_shape, out_dtype, amax_out, scale, scale_inv);

  nvte_cast_transpose(input_tensor.data(), input_cast_tensor.data(), input_cast_trans_tensor.data(),
                      stream);
  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CastTransposeHandler, CastTransposeFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // input
                                  .Arg<Buffer_Type>()      // amax
                                  .Arg<Buffer_Type>()      // scale
                                  .Arg<Buffer_Type>()      // scale_inv
                                  .Ret<Buffer_Type>()      // input_cast
                                  .Ret<Buffer_Type>()      // input_cast_trans
                                  .Ret<Buffer_Type>()      // amax_out
                                  .Attr<int64_t>("transpose_axis"),
                              FFI_CudaGraph_Traits);

pybind11::tuple GetDBiasCastTransposeWorkspaceSizes(size_t batch_size, size_t hidden_size,
                                                    DType in_dtype, DType out_dtype) {
  auto input_shape = std::vector<size_t>{batch_size, hidden_size};
  auto output_shape = std::vector<size_t>{batch_size, hidden_size};
  auto output_trans_shape = std::vector<size_t>{hidden_size, batch_size};
  auto dbias_shape = std::vector<size_t>{hidden_size};

  auto input_tensor = TensorWrapper(nullptr, input_shape, in_dtype);
  auto output_tensor = TensorWrapper(nullptr, output_shape, out_dtype);
  auto output_trans_tensor = TensorWrapper(nullptr, output_trans_shape, out_dtype);
  auto dbias_tensor = TensorWrapper(nullptr, dbias_shape, in_dtype);

  TensorWrapper dummy_workspace;

  nvte_cast_transpose_dbias(input_tensor.data(), output_tensor.data(), output_trans_tensor.data(),
                            dbias_tensor.data(), dummy_workspace.data(), nullptr);

  auto work_shape = MakeShapeVector(dummy_workspace.shape());
  return pybind11::make_tuple(std::make_pair(work_shape, dummy_workspace.dtype()));
}

void DBiasCastTranspose(cudaStream_t stream, void **buffers, const char *opaque,
                        size_t opaque_len) {
  auto *input = buffers[0];
  float *amax = reinterpret_cast<float *>(buffers[1]);
  float *scale = reinterpret_cast<float *>(buffers[2]);
  float *scale_inv = reinterpret_cast<float *>(buffers[3]);
  auto *output = buffers[4];
  auto *output_trans = buffers[5];
  auto *dbias = buffers[6];
  float *amax_out = reinterpret_cast<float *>(buffers[7]);
  void *workspace_ptr = buffers[8];

  const auto &desc = *UnpackOpaque<CustomCallCommonWkDescriptor>(opaque, opaque_len);
  NVTE_CHECK(amax == amax_out,
             "amax not bound to amax_out in TE/JAX DBiasCastTranspose primitive.");
  if (!use_fp8(desc.out_dtype)) {
    scale = nullptr;
    scale_inv = nullptr;
    amax_out = nullptr;
  }
  auto m = desc.shape.dims[0];
  auto n = desc.shape.dims[1];
  auto input_shape = std::vector<size_t>{m, n};
  auto output_shape = std::vector<size_t>{m, n};
  auto output_trans_shape = std::vector<size_t>{n, m};
  auto dbias_shape = std::vector<size_t>{n};

  auto input_tensor = TensorWrapper(input, input_shape, desc.in_dtype);
  auto output_tensor =
      TensorWrapper(output, output_shape, desc.out_dtype, amax_out, scale, scale_inv);
  auto output_trans_tensor =
      TensorWrapper(output_trans, output_trans_shape, desc.out_dtype, amax_out, scale, scale_inv);
  auto dbias_tensor = TensorWrapper(dbias, dbias_shape, desc.in_dtype);

  auto workspace = TensorWrapper(workspace_ptr, desc.wkshape.to_vector(), desc.wk_dtype);

  nvte_cast_transpose_dbias(input_tensor.data(), output_tensor.data(), output_trans_tensor.data(),
                            dbias_tensor.data(), workspace.data(), stream);
}

}  // namespace jax
}  // namespace transformer_engine
