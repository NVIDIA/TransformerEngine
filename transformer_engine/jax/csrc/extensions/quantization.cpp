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
                                               DType in_dtype, DType out_dtype,
                                               JAXX_Scaling_Mode scaling_mode,
                                               QuantizeLayout q_layout) {
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
  auto dbias_tensor = TensorWrapper(reinterpret_cast<void *>(&temp), dbias_shape, in_dtype);

  auto output_tensor = TensorWrapper(get_nvte_scaling_mode(scaling_mode));
  // Only the pointers will be checked for scale_inv, thus the shapes do not matter
  if (q_layout == QuantizeLayout::ROWWISE_COLWISE || q_layout == QuantizeLayout::ROWWISE) {
    output_tensor.set_rowwise_data(reinterpret_cast<void *>(&temp), out_dtype, output_shape);
    if (is_fp8_dtype(out_dtype)) {
      output_tensor.set_rowwise_scale_inv(reinterpret_cast<void *>(&temp), DType::kFloat32,
                                          std::vector<size_t>{1});
    }
  }

  if (q_layout == QuantizeLayout::ROWWISE_COLWISE || q_layout == QuantizeLayout::COLWISE) {
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

  nvte_quantize_dbias(input_tensor.data(), output_tensor.data(), dbias_tensor.data(),
                      dummy_workspace.data(), nullptr);

  auto work_shape = MakeShapeVector(dummy_workspace.shape());
  return pybind11::make_tuple(std::make_pair(work_shape, dummy_workspace.dtype()));
}

Error_Type DBiasQuantizeFFI(cudaStream_t stream, Buffer_Type input_buf, Buffer_Type scale_buf,
                            Result_Type output_buf, Result_Type output_trans_buf,
                            Result_Type scale_inv_buf, Result_Type colwise_scale_inv_buf,
                            Result_Type amax_buf, Result_Type dbias_buf, Result_Type workspace_buf,
                            JAXX_Scaling_Mode scaling_mode, int64_t quantize_layout_enum,
                            bool is_dbias, int64_t flatten_axis) {
  auto in_dtype = convert_ffi_datatype_to_te_dtype(input_buf.element_type());
  auto out_dtype = convert_ffi_datatype_to_te_dtype(output_buf->element_type());
  auto workspace_dtype = convert_ffi_datatype_to_te_dtype(workspace_buf->element_type());

  NVTE_CHECK(is_fp8_dtype(out_dtype), "Output datatype must be FP8 for quantization.");

  auto *input = input_buf.untyped_data();

  auto const quantize_layout = static_cast<QuantizeLayout>(quantize_layout_enum);

  auto *output = output_buf->untyped_data();
  auto *output_trans = output_trans_buf->untyped_data();
  auto *dbias = dbias_buf->untyped_data();
  void *workspace = workspace_buf->untyped_data();

  auto input_dims = input_buf.dimensions();
  int64_t input_ndim = input_dims.size();
  if (flatten_axis < 0) flatten_axis += input_ndim;
  NVTE_CHECK(flatten_axis < input_ndim && flatten_axis > 0, "flatten_axis is out of bounds!");

  auto workspace_dims = workspace_buf->dimensions();
  auto m = product(input_dims, 0, flatten_axis);
  auto n = product(input_dims, flatten_axis, input_ndim);
  auto input_shape = std::vector<size_t>{m, n};
  auto output_shape = std::vector<size_t>{m, n};
  auto output_trans_shape = std::vector<size_t>{n, m};
  auto dbias_shape = std::vector<size_t>{n};
  std::vector<size_t> workspace_shape{workspace_dims.begin(), workspace_dims.end()};

  auto input_tensor = TensorWrapper(input, input_shape, in_dtype);
  auto output_tensor = TensorWrapper(get_nvte_scaling_mode(scaling_mode));

  if (quantize_layout == QuantizeLayout::ROWWISE ||
      quantize_layout == QuantizeLayout::ROWWISE_COLWISE) {
    output_tensor.set_rowwise_data(output, out_dtype, output_shape);

    if (is_fp8_dtype(out_dtype)) {
      if (scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING) {
        float *scale = reinterpret_cast<float *>(scale_buf.untyped_data());
        float *amax = reinterpret_cast<float *>(amax_buf->untyped_data());
        NVTE_CHECK(scale != nullptr, "scale must be provided for delayed tensor scaling");
        NVTE_CHECK(amax != nullptr, "amax must be provided for delayed tensor scaling");
        output_tensor.set_scale(scale, DType::kFloat32, std::vector<size_t>{1});
        cudaMemsetAsync(amax, 0, sizeof(float), stream);
        output_tensor.set_amax(amax, DType::kFloat32, std::vector<size_t>{1});
        output_tensor.set_rowwise_scale_inv(
            scale_inv_buf->untyped_data(),
            convert_ffi_datatype_to_te_dtype(scale_inv_buf->element_type()),
            std::vector<size_t>{1});
      } else {
        output_tensor.set_rowwise_scale_inv(
            scale_inv_buf->untyped_data(),
            convert_ffi_datatype_to_te_dtype(scale_inv_buf->element_type()),
            std::vector<size_t>{product(scale_inv_buf->dimensions(), 0, flatten_axis),
                                product(scale_inv_buf->dimensions(), flatten_axis,
                                        scale_inv_buf->dimensions().size())});
      }
    }
  }

  if (quantize_layout == QuantizeLayout::COLWISE ||
      quantize_layout == QuantizeLayout::ROWWISE_COLWISE) {
    auto &tmp_shape = (scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING)
                          ? output_trans_shape
                          : output_shape;
    output_tensor.set_columnwise_data(output_trans, out_dtype, tmp_shape);
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
                                  .Attr<JAXX_Scaling_Mode>("scaling_mode")
                                  .Attr<int64_t>("q_layout")
                                  .Attr<bool>("is_dbias")
                                  .Attr<int64_t>("flatten_axis"),
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

Error_Type GroupedQuantizeFFI(cudaStream_t stream, Buffer_Type inputs, Buffer_Type scales,
                            Buffer_Type group_sizes,
                            Result_Type outputs, Result_Type colwise_outputs,
                            Result_Type scale_invs, Result_Type colwise_scale_invs,
                            Result_Type amaxs,
                            JAXX_Scaling_Mode scaling_mode, int64_t quantize_layout_enum,
                            int64_t flatten_axis) {
  auto in_dtype = convert_ffi_datatype_to_te_dtype(inputs.element_type());
  auto out_dtype = convert_ffi_datatype_to_te_dtype(outputs->element_type());
  NVTE_CHECK(is_fp8_dtype(out_dtype), "Output datatype must be FP8 for quantization.");

  auto scale_dtype = convert_ffi_datatype_to_te_dtype(scales.element_type());
  auto sinv_dtype = convert_ffi_datatype_to_te_dtype(scale_invs.element_type());
  auto group_size_dtype = convert_ffi_datatype_to_te_dtype(group_sizes.element_type());
  auto const quantize_layout = static_cast<QuantizeLayout>(quantize_layout_enum);

  auto *input_ptr = reinterpret_cast<uint8_t *>(inputs.untyped_data());
  auto *scale_ptr = reinterpret_cast<uint8_t *>(scales.untyped_data());
  auto *group_size_ptr = reinterpret_cast<uint8_t *>(group_sizes.untyped_data());
  auto *output_ptr = reinterpret_cast<uint8_t*>(outputs->untyped_data());
  auto *colwise_output_ptr = reinterpret_cast<uint8_t*>(colwise_output_ptr->untyped_data());
  auto *sinv_ptr = reinterpret_cast<uint8_t*>(scale_invs->untyped_data());
  auto *colwise_sinv_ptr = reinterpret_cast<uint8_t*>(colwise_scale_invs->untyped_data());
  auto *amaxs_ptr = reinterpret_cast<uint8_t *>(amaxs->untyped_data());

  size_t input_dtype_bytes = te_dtype_bytes(in_dtype);
  size_t output_dtype_bytes = te_dtype_bytes(out_dtype);
  size_t scale_dtype_bytes = te_dtype_bytes(scale_dtype);
  size_t sinv_dtype_bytes = te_dtype_bytes(sinv_dtype);
  size_t group_size_dtype_bytes = te_dtype_bytes(group_size_dtype);

  auto input_dims = inputs.dimensions();
  int64_t input_ndim = input_dims.size();
  if (flatten_axis < 0) flatten_axis += input_ndim;
  NVTE_CHECK(flatten_axis < input_ndim && flatten_axis > 0, "flatten_axis is out of bounds!");

  auto m = product(input_dims, 0, flatten_axis);
  auto n = product(input_dims, flatten_axis, input_ndim);
  auto input_shape = std::vector<size_t>{m, n};
  auto output_shape = std::vector<size_t>{m * n};

  // These lists are to keep the TensorWrapper objects alive
  std::vector<TensorWrapper> input_holders;
  std::vector<TensorWrapper> output_holders;

  // These lists are the actual NVTETensor (void *) lists for multi-stream GEMM
  std::vector<NVTETensor> input_list;
  std::vector<NVTETensor> output_list;

  size_t dim_list_bytes = group_size_dtype * num_gemms;
  std::vector<int32_t> dim_list_host(num_gemms);
  auto dim_list_ptr = reinterpret_cast<int32_t *>(group_sizes.untyped_data());
  cudaMemcpyAsync(dim_list_host.data(), dim_list_ptr, dim_list_bytes, cudaMemcpyDeviceToHost,
                  stream);
  // Note: This may break cudaGraph.
  cudaStreamSynchronize(stream);
  size_t sum_group_sizes = std::accumulate(dim_list_host.begin(), dim_list_host.end(), 0);
  NVTE_CHECK(m == sum_group_sizes, "Unexpected group_sizes! M =", m, ", got sum(group_sizes)=", sum_group_sizes);

  if (quantize_layout == QuantizeLayout::ROWWISE ||
      quantize_layout == QuantizeLayout::ROWWISE_COLWISE) {
    output_tensor.set_rowwise_data(output, out_dtype, output_shape);

    if (is_fp8_dtype(out_dtype)) {
      if (scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING) {
        float *scale = reinterpret_cast<float *>(scales.untyped_data());
        float *amax = reinterpret_cast<float *>(amaxs->untyped_data());
        NVTE_CHECK(scale != nullptr, "scale must be provided for delayed tensor scaling");
        NVTE_CHECK(amax != nullptr, "amax must be provided for delayed tensor scaling");
        output_tensor.set_scale(scale, DType::kFloat32, std::vector<size_t>{1});
        cudaMemsetAsync(amax, 0, sizeof(float), stream);
        output_tensor.set_amax(amax, DType::kFloat32, std::vector<size_t>{1});
        output_tensor.set_rowwise_scale_inv(
            scale_invs->untyped_data(),
            convert_ffi_datatype_to_te_dtype(scale_invs->element_type()),
            std::vector<size_t>{1});
      } else {
        output_tensor.set_rowwise_scale_inv(
            scale_invs->untyped_data(),
            convert_ffi_datatype_to_te_dtype(scale_invs->element_type()),
            std::vector<size_t>{product(scale_invs->dimensions(), 0, flatten_axis),
                                product(scale_invs->dimensions(), flatten_axis,
                                        scale_invs->dimensions().size())});
      }
    }
  }

  if (quantize_layout == QuantizeLayout::COLWISE ||
      quantize_layout == QuantizeLayout::ROWWISE_COLWISE) {
    auto &tmp_shape = (scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING)
                          ? output_trans_shape
                          : output_shape;
    output_tensor.set_columnwise_data(output_trans, out_dtype, tmp_shape);
    // For 2x delayed scaling, the scale buffer is shared between rowwise and columnwise scaling
    auto &tmps = (scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING)
                        ? scale_invs
                        : colwise_scale_invs;

    if (scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING) {
      output_tensor.set_columnwise_scale_inv(
          tmps->untyped_data(), convert_ffi_datatype_to_te_dtype(tmps->element_type()),
          std::vector<size_t>{1});
    } else {
      output_tensor.set_columnwise_scale_inv(
          tmps->untyped_data(), convert_ffi_datatype_to_te_dtype(tmps->element_type()),
          std::vector<size_t>{
              product(tmps->dimensions(), 0, flatten_axis),
              product(tmps->dimensions(), flatten_axis, tmps->dimensions().size())});
    }
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
                                  .Attr<JAXX_Scaling_Mode>("scaling_mode")
                                  .Attr<int64_t>("q_layout")
                                  .Attr<bool>("is_dbias")
                                  .Attr<int64_t>("flatten_axis"),
                              FFI_CudaGraph_Traits);

}  // namespace jax
}  // namespace transformer_engine
