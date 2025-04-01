/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include <cuda_runtime.h>

#include "extensions.h"
#include "transformer_engine/cast.h"
#include "xla/ffi/api/c_api.h"

namespace transformer_engine {
namespace jax {

pybind11::tuple GetDBiasQuantizeWorkspaceSizes(size_t batch_size, size_t hidden_size,
                                               DType in_dtype, DType out_dtype) {
  auto input_shape = std::vector<size_t>{batch_size, hidden_size};
  auto output_shape = std::vector<size_t>{batch_size, hidden_size};
  auto output_trans_shape = std::vector<size_t>{hidden_size, batch_size};
  auto dbias_shape = std::vector<size_t>{hidden_size};

  // Evil hack to specify TE impl
  // Note: nvte_quantize_dbias chooses its internal impl based on what
  // pointers are allocated, e.g. whether to output with column-wise
  // data. However, we don't have access to any allocated buffers in
  // this function. We pass a dummy pointer as a workaround.
  int temp = 0;

  auto input_tensor = TensorWrapper(reinterpret_cast<void *>(&temp), input_shape, in_dtype);
  auto output_tensor = TensorWrapper(reinterpret_cast<void *>(&temp), output_shape, out_dtype);
  output_tensor.set_columnwise_data(reinterpret_cast<void *>(&temp), out_dtype, output_trans_shape);
  auto dbias_tensor = TensorWrapper(reinterpret_cast<void *>(&temp), dbias_shape, in_dtype);

  TensorWrapper dummy_workspace;

  nvte_quantize_dbias(input_tensor.data(), output_tensor.data(), dbias_tensor.data(),
                      dummy_workspace.data(), nullptr);

  auto work_shape = MakeShapeVector(dummy_workspace.shape());
  return pybind11::make_tuple(std::make_pair(work_shape, dummy_workspace.dtype()));
}

Error_Type DBiasQuantizeFFI(cudaStream_t stream, Buffer_Type input_buf, Buffer_Type scale_buf,
                            Result_Type output_buf, Result_Type output_trans_buf,
                            Result_Type scale_inv_buf, Result_Type trans_scale_inv_buf,
                            Result_Type amax_out_buf, Result_Type dbias_buf,
                            Result_Type workspace_buf, int64_t scaling_mode_enum,
                            int64_t quantize_axis_enum, bool is_dbias) {
  auto in_dtype = convert_ffi_datatype_to_te_dtype(input_buf.element_type());
  auto out_dtype = convert_ffi_datatype_to_te_dtype(output_buf->element_type());
  auto workspace_dtype = convert_ffi_datatype_to_te_dtype(workspace_buf->element_type());

  NVTE_CHECK(is_fp8_dtype(out_dtype), "Output datatype must be FP8 for quantization.");

  auto *input = input_buf.untyped_data();

  auto scaling_mode = static_cast<NVTEScalingMode>(scaling_mode_enum);
  auto const quantize_axis = static_cast<QuantizeAxis>(quantize_axis_enum);

  auto *output = output_buf->untyped_data();
  auto *output_trans = output_trans_buf->untyped_data();
  auto *dbias = dbias_buf->untyped_data();
  void *workspace = workspace_buf->untyped_data();

  auto input_dims = input_buf.dimensions();
  auto workspace_dims = workspace_buf->dimensions();
  auto m = product(input_dims, 0, input_dims.size() - 1);
  auto n = input_dims.back();
  auto input_shape = std::vector<size_t>{m, n};
  auto output_shape = std::vector<size_t>{m, n};
  auto output_trans_shape = std::vector<size_t>{n, m};
  auto dbias_shape = std::vector<size_t>{n};
  std::vector<size_t> workspace_shape{workspace_dims.begin(), workspace_dims.end()};

  auto input_tensor = TensorWrapper(input, input_shape, in_dtype);
  auto output_tensor = TensorWrapper(scaling_mode);

  if (quantize_axis == QuantizeAxis::ROWWISE || quantize_axis == QuantizeAxis::ROWWISE_COLWISE) {
    output_tensor.set_rowwise_data(output, out_dtype, output_shape);
    output_tensor.set_rowwise_scale_inv(
        scale_inv_buf->untyped_data(),
        convert_ffi_datatype_to_te_dtype(scale_inv_buf->element_type()),
        std::vector<size_t>{
            product(scale_inv_buf->dimensions(), 0, scale_inv_buf->dimensions().size() - 1),
            scale_inv_buf->dimensions().back()});
  }

  if (scaling_mode == NVTE_DELAYED_TENSOR_SCALING) {
    float *scale = reinterpret_cast<float *>(scale_buf.untyped_data());
    float *amax_out = reinterpret_cast<float *>(amax_out_buf->untyped_data());
    NVTE_CHECK(scale != nullptr, "scale must be provided for delayed tensor scaling");
    NVTE_CHECK(amax_out != nullptr, "amax must be provided for delayed tensor scaling");
    output_tensor.set_scale(scale, DType::kFloat32, std::vector<size_t>{1});
    cudaMemsetAsync(amax_out, 0, sizeof(float), stream);
    output_tensor.set_amax(amax_out, DType::kFloat32, std::vector<size_t>{1});
  }

  if (quantize_axis == QuantizeAxis::COLWISE || quantize_axis == QuantizeAxis::ROWWISE_COLWISE) {
    output_tensor.set_columnwise_data(output_trans, out_dtype, output_trans_shape);
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

  auto dbias_tensor = TensorWrapper(dbias, dbias_shape, in_dtype);
  auto workspace_tensor = TensorWrapper(workspace, workspace_shape, workspace_dtype);

  if (is_dbias) {
    nvte_quantize_dbias(input_tensor.data(), output_tensor.data(), dbias_tensor.data(),
                        workspace_tensor.data(), stream);
  } else {
    nvte_quantize(input_tensor.data(), output_tensor.data(), stream);
  }
  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(DBiasQuantizeHandler, DBiasQuantizeFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // input
                                  .Arg<Buffer_Type>()      // scale
                                  .Ret<Buffer_Type>()      // output
                                  .Ret<Buffer_Type>()      // colwise output
                                  .Ret<Buffer_Type>()      // scale_inv
                                  .Ret<Buffer_Type>()      // scale_inv colwise
                                  .Ret<Buffer_Type>()      // amax
                                  .Ret<Buffer_Type>()      // dbias
                                  .Ret<Buffer_Type>()      // wkspace
                                  .Attr<int64_t>("scaling_mode")
                                  .Attr<int64_t>("q_axis")
                                  .Attr<bool>("is_dbias"),
                              FFI_CudaGraph_Traits);

Error_Type DequantizeFFI(cudaStream_t stream, Buffer_Type input_buf, Buffer_Type amax_buf,
                         Buffer_Type scale_buf, Buffer_Type scale_inv_buf, Result_Type output_buf) {
  auto in_dtype = convert_ffi_datatype_to_te_dtype(input_buf.element_type());
  auto out_dtype = convert_ffi_datatype_to_te_dtype(output_buf->element_type());

  auto *input = input_buf.untyped_data();
  auto *amax = reinterpret_cast<float *>(amax_buf.untyped_data());
  auto *scale = reinterpret_cast<float *>(scale_buf.untyped_data());
  auto *scale_inv = reinterpret_cast<float *>(scale_inv_buf.untyped_data());

  auto *output = output_buf->untyped_data();

  auto input_dims = input_buf.dimensions();
  std::vector<size_t> shape(input_dims.begin(), input_dims.end());
  auto input_tensor = TensorWrapper(input, shape, in_dtype, amax, scale, scale_inv);
  auto output_tensor = TensorWrapper(output, shape, out_dtype);

  nvte_dequantize(input_tensor.data(), output_tensor.data(), stream);
  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(DequantizeHandler, DequantizeFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // input
                                  .Arg<Buffer_Type>()      // amax
                                  .Arg<Buffer_Type>()      // scale
                                  .Arg<Buffer_Type>()      // scale_inv
                                  .Ret<Buffer_Type>(),     // output
                              FFI_CudaGraph_Traits);

}  // namespace jax
}  // namespace transformer_engine
