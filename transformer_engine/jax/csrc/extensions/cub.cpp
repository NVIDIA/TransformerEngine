/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/cub.h"

#include "../extensions.h"
#include "xla/ffi/api/c_api.h"

namespace transformer_engine {
namespace jax {

Error_Type TopkFFI(cudaStream_t stream, Buffer_Type keys_in_buf, Buffer_Type values_in_buf,
                   Result_Type keys_out_buf, Result_Type values_out_buf, Result_Type workspace_buf,
                   int64_t k_value, int64_t workbuf_bytes) {
  auto keys_in_dtype = convert_ffi_datatype_to_te_dtype(keys_in_buf.element_type());
  auto values_in_dtype = convert_ffi_datatype_to_te_dtype(values_in_buf.element_type());
  auto keys_out_dtype = convert_ffi_datatype_to_te_dtype(keys_out_buf->element_type());
  auto values_out_dtype = convert_ffi_datatype_to_te_dtype(values_out_buf->element_type());
  NVTE_CHECK(keys_in_dtype == keys_out_dtype, "Input and output keys must have the same datatype");
  NVTE_CHECK(values_in_dtype == values_out_dtype,
             "Input and output values must have the same datatype");
  NVTE_CHECK(values_in_dtype == DType::kInt32, "CubTopkFFI() only supports int32 values for now");

  auto keys_in_shape = keys_in_buf.dimensions();
  NVTE_CHECK(keys_in_shape.size() == 1 || keys_in_shape.size() == 2,
             "Keys input must have 1 or 2 dimensions");

  // Extract batch_size and num_items from input shape.
  // 1D: (num_items,)          – batch_size = 1
  // 2D: (batch_size, num_items) – each row is one top-k problem
  int batch_size = (keys_in_shape.size() == 2) ? static_cast<int>(keys_in_shape[0]) : 1;
  int num_items  = static_cast<int>(keys_in_shape[keys_in_shape.size() - 1]);
  int k          = static_cast<int>(k_value);

  // Byte stride between rows in each buffer (element size * items-per-row).
  // Only float32/float16/bfloat16 keys and int32 values are supported, so
  // we compute strides explicitly rather than pulling in a separate helper.
  size_t keys_element_bytes;
  switch (keys_in_dtype) {
    case DType::kFloat32:  keys_element_bytes = 4; break;
    case DType::kFloat16:  keys_element_bytes = 2; break;
    case DType::kBFloat16: keys_element_bytes = 2; break;
    default: NVTE_ERROR("Unsupported key dtype for CUB TopK");
  }
  size_t keys_in_row_bytes  = static_cast<size_t>(num_items) * keys_element_bytes;
  size_t keys_out_row_bytes = static_cast<size_t>(k)         * keys_element_bytes;
  size_t vals_in_row_bytes  = static_cast<size_t>(num_items) * sizeof(int32_t);
  size_t vals_out_row_bytes = static_cast<size_t>(k)         * sizeof(int32_t);

  auto row_in_shape  = std::vector<size_t>{static_cast<size_t>(num_items)};
  auto row_out_shape = std::vector<size_t>{static_cast<size_t>(k)};
  auto workspace_shape = std::vector<size_t>{static_cast<size_t>(workbuf_bytes)};

  auto workspace_tensor =
      TensorWrapper(workspace_buf->untyped_data(), workspace_shape, DType::kByte);

  char *keys_in_ptr  = static_cast<char *>(keys_in_buf.untyped_data());
  char *vals_in_ptr  = static_cast<char *>(values_in_buf.untyped_data());
  char *keys_out_ptr = static_cast<char *>(keys_out_buf->untyped_data());
  char *vals_out_ptr = static_cast<char *>(values_out_buf->untyped_data());

  // One nvte_topk call per row.  All calls go to the same CUDA stream so they
  // serialise on the GPU; the workspace is safely reused across rows.
  for (int b = 0; b < batch_size; b++) {
    auto keys_in_tensor =
        TensorWrapper(keys_in_ptr  + b * keys_in_row_bytes,  row_in_shape,  keys_in_dtype);
    auto values_in_tensor =
        TensorWrapper(vals_in_ptr  + b * vals_in_row_bytes,  row_in_shape,  values_in_dtype);
    auto keys_out_tensor =
        TensorWrapper(keys_out_ptr + b * keys_out_row_bytes, row_out_shape, keys_out_dtype);
    auto values_out_tensor =
        TensorWrapper(vals_out_ptr + b * vals_out_row_bytes, row_out_shape, values_out_dtype);

    nvte_topk(stream, keys_in_tensor.data(), values_in_tensor.data(), keys_out_tensor.data(),
              values_out_tensor.data(), workspace_tensor.data(), num_items, k, workbuf_bytes);
  }

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(TopkHandler, TopkFFI,
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
