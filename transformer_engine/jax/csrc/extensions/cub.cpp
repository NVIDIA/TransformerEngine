/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"
#include "xla/ffi/api/c_api.h"
#include "transformer_engine/cub.h"

namespace transformer_engine {
namespace jax {

Error_Type CubTopkFFI(cudaStream_t stream, Buffer_Type keys_in_buf, Buffer_Type values_in_buf,
                      Result_Type keys_out_buf, Result_Type values_out_buf, Result_Type workspace_buf,
                      int64_t k_value, int64_t workbuf_bytes) {
  auto keys_in_dtype = convert_ffi_datatype_to_te_dtype(keys_in_buf.element_type());
  auto values_in_dtype = convert_ffi_datatype_to_te_dtype(values_in_buf.element_type());
  auto keys_out_dtype = convert_ffi_datatype_to_te_dtype(keys_out_buf->element_type());
  auto values_out_dtype = convert_ffi_datatype_to_te_dtype(values_out_buf->element_type());
  NVTE_CHECK(keys_in_dtype == keys_out_dtype, "Input and output keys must have the same datatype");
  NVTE_CHECK(values_in_dtype == values_out_dtype, "Input and output values must have the same datatype");
  NVTE_CHECK(values_in_dtype == DType::kInt32, "CubTopkFFI() only supports int32 values for now");

  auto keys_in_shape = keys_in_buf.dimensions();
  auto values_in_shape = values_in_buf.dimensions();
  auto keys_out_shape = keys_out_buf->dimensions();
  auto values_out_shape = values_out_buf->dimensions();
  NVTE_CHECK(keys_in_shape.size() == 1, "Keys input must have 1 dimension");
  NVTE_CHECK(values_in_shape.size() == 1, "Values input must have 1 dimension");
  NVTE_CHECK(keys_out_shape.size() == 1, "Keys output must have 1 dimension");
  NVTE_CHECK(values_out_shape.size() == 1, "Values output must have 1 dimension");
  NVTE_CHECK(keys_in_shape[0] == values_in_shape[0], "Keys and values input must have the same number of items");
  NVTE_CHECK(keys_out_shape[0] == values_out_shape[0], "Keys and values output must have the same number of items");
  int num_items = static_cast<int>(keys_in_shape[0]);
  int k = static_cast<int>(k_value);

  auto input_shape = std::vector<size_t>{keys_in_shape[0]};
  auto output_shape = std::vector<size_t>{keys_out_shape[0]};
  auto workspace_shape = std::vector<size_t>{workbuf_bytes};

  auto keys_in_tensor = TensorWrapper(keys_in_buf.untyped_data(), input_shape, keys_in_dtype);
  auto values_in_tensor = TensorWrapper(values_in_buf.untyped_data(), input_shape, values_in_dtype);
  auto keys_out_tensor = TensorWrapper(keys_out_buf->untyped_data(), output_shape, keys_out_dtype);
  auto values_out_tensor = TensorWrapper(values_out_buf->untyped_data(), output_shape, values_out_dtype);
  auto workspace_tensor = TensorWrapper(workspace_buf->untyped_data(), workspace_shape, DType::kByte);

  nvte_cub_topk(stream, keys_in_tensor.data(), values_in_tensor.data(),
                keys_out_tensor.data(), values_out_tensor.data(), workspace_tensor.data(),
                num_items, k, workbuf_bytes);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CubTopkHandler, CubTopkFFI,
  FFI::Bind()
      .Ctx<FFI_Stream_Type>()  // stream
      .Arg<Buffer_Type>()      // keys_buf
      .Arg<Buffer_Type>()      // values_buf
      .Ret<Buffer_Type>()      // topk_buf
      .Ret<Buffer_Type>()      // indices_buf
      .Ret<Buffer_Type>()      // workspace_buf
      .Attr<int64_t>("k_value")
      .Attr<int64_t>("workbuf_bytes"),
  FFI_CudaGraph_Traits);

}  // namespace jax
}  // namespace transformer_engine
