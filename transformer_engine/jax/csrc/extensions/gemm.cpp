/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include "transformer_engine/gemm.h"

#include <memory>

#include "common/util/cuda_runtime.h"
#include "common/util/system.h"
#include "extensions.h"
#include "transformer_engine/multi_streams.h"
#include "xla/ffi/api/c_api.h"

#define MXFP8_BLOCK_SIZE 32

namespace transformer_engine {
namespace jax {

Error_Type GroupedGemmFFI(cudaStream_t stream, Buffer_Type lhs_data, Buffer_Type lhs_scale,
                          Buffer_Type rhs_data, Buffer_Type rhs_scale, Buffer_Type bias,
                          Buffer_Type group_sizes, Buffer_Type group_offset, Result_Type output,
                          Result_Type workspace, size_t m, size_t n, size_t k, bool lhs_is_trans,
                          bool rhs_is_trans, JAXX_Scaling_Mode scaling_mode, bool has_bias) {
  // Notes on matrix layouts and transpose:
  // Jax uses row-major data_layout, on entering this function, each input matrix pair:
  //   A: row-major [m, k] for N - [k, m] for T
  //   B: row-major [k, n] for N - [n, k] for T
  // on exiting this function, JAX expect:
  //   C: row-major with size [m, n].
  // cuBLAS uses column-major data_layout, in this view, each input matrix pair:
  //   A: column-major with size [k, m] for T - [m, k] for N
  //   B: column-major with size [n, k] for T - [k, n] for N
  //
  // If we call cuBLAS GEMM for A * B, the output will be:
  //   C: column-major with size [m, n] --> row-major with size [n, m].
  // To make the output compatible with JAX, we need to swap A and B in cuBLAS GEMM call.

  // Inputs
  auto lhs_ptr = reinterpret_cast<uint8_t *>(lhs_data.untyped_data());
  auto rhs_ptr = reinterpret_cast<uint8_t *>(rhs_data.untyped_data());
  auto lhs_scale_ptr = reinterpret_cast<uint8_t *>(lhs_scale.untyped_data());
  auto rhs_scale_ptr = reinterpret_cast<uint8_t *>(rhs_scale.untyped_data());
  auto lhs_dtype = convert_ffi_datatype_to_te_dtype(lhs_data.element_type());
  auto rhs_dtype = convert_ffi_datatype_to_te_dtype(rhs_data.element_type());
  auto lhs_scale_dtype = convert_ffi_datatype_to_te_dtype(lhs_scale.element_type());
  auto rhs_scale_dtype = convert_ffi_datatype_to_te_dtype(rhs_scale.element_type());
  auto bias_ptr = has_bias ? reinterpret_cast<uint8_t *>(bias.untyped_data()) : nullptr;
  auto bias_dtype = convert_ffi_datatype_to_te_dtype(bias.element_type());

  NVTE_CHECK(group_sizes.dimensions().size() == 1);
  size_t num_gemms = group_sizes.dimensions()[0];

  // Outputs
  auto out_ptr = reinterpret_cast<uint8_t *>(output->untyped_data());
  auto out_dtype = convert_ffi_datatype_to_te_dtype(output->element_type());
  auto workspace_ptr = reinterpret_cast<uint8_t *>(workspace->untyped_data());
  auto workspace_total_size = product(workspace->dimensions()) / num_streams;

  auto lhs_sinv_size = product(lhs_scale.dimensions());
  auto rhs_sinv_size = product(rhs_scale.dimensions());
  auto workspace_size = (workspace_total_size - lhs_sinv_size - rhs_sinv_size) / num_streams;
  auto swizzled_lhs_sinv_ptr = workspace_ptr;
  auto swizzled_rhs_sinv_ptr = workspace_ptr + lhs_sinv_size;
  workspace_ptr += lhs_sinv_size + rhs_sinv_size;

  size_t lhs_dtype_bytes = te_dtype_bytes(lhs_dtype);
  size_t rhs_dtype_bytes = te_dtype_bytes(rhs_dtype);
  size_t lhs_scale_dtype_bytes = te_dtype_bytes(lhs_scale_dtype);
  size_t rhs_scale_dtype_bytes = te_dtype_bytes(rhs_scale_dtype);
  size_t bias_dtype_bytes = te_dtype_bytes(bias_dtype);
  size_t out_dtype_bytes = te_dtype_bytes(out_dtype);

  NVTE_CHECK(lhs_dtype_bytes == rhs_dtype_bytes, "sizeof(lhs_dtype) != sizeof(rhs_dtype)");
  NVTE_CHECK(lhs_scale_dtype_bytes == rhs_scale_dtype_bytes,
             "sizeof(lhs_scale_dtype) != sizeof(rhs_scale_dtype)");
  NVTE_CHECK(m * k == product(lhs_data.dimensions()), "Unexpected lhs size! Expect ", m * k,
             ", got ", product(lhs_data.dimensions()));
  NVTE_CHECK(n * k * num_gemms == product(rhs_data.dimensions()),
             "Unexpected rhs size! Expect n * k * num_gemms = ", n, " * ", k, " * ", num_gemms,
             " = ", n * k * num_gemms, ", got ", product(rhs_data.dimensions()));
  NVTE_CHECK(n * m == product(output->dimensions()), "Unexpected output size! Expect ", n * m,
             ", got ", product(output->dimensions()));

  size_t dim_list_bytes = sizeof(int32_t) * num_gemms;
  std::vector<int32_t> dim_list_host(num_gemms);
  auto dim_list_ptr = reinterpret_cast<int32_t *>(group_sizes.untyped_data());
  cudaMemcpyAsync(dim_list_host.data(), dim_list_ptr, dim_list_bytes, cudaMemcpyDeviceToHost,
                  stream);
  // Note: This may break cudaGraph.
  cudaStreamSynchronize(stream);
  size_t sum_group_sizes = std::accumulate(dim_list_host.begin(), dim_list_host.end(), 0);
  NVTE_CHECK(m == sum_group_sizes, "Unexpected group_sizes! M =", m,
             ", got sum(group_sizes)=", sum_group_sizes);

  auto num_math_sm = cuda::sm_count() - getenv<int>("NVTE_EXT_MARGIN_SM", 0);
  bool grad = false;
  bool accumulate = false;
  bool use_split_accumulator = false;
  auto bias_shape = std::vector<size_t>{has_bias ? n : 0};
  const int arch = cuda::sm_arch();

  if (arch < 100 && is_fp8_dtype(lhs_dtype))
    NVTE_CHECK(!lhs_is_trans && rhs_is_trans,
               "Only NT (row-major) GEMM is supported, got lhs_is_trans=", lhs_is_trans,
               ", rhs_is_trans=", rhs_is_trans);

  // These lists are to keep the TensorWrapper objects alive
  std::vector<TensorWrapper> lhs_wrapper_list;
  std::vector<TensorWrapper> rhs_wrapper_list;
  std::vector<TensorWrapper> bias_wrapper_list;
  std::vector<TensorWrapper> pre_gelu_wrapper_list;
  std::vector<TensorWrapper> out_wrapper_list;
  std::vector<TensorWrapper> workspace_wrapper_list;

  // These lists are the actual NVTETensor (void *) lists for multi-stream GEMM
  std::vector<NVTETensor> lhs_list;
  std::vector<NVTETensor> rhs_list;
  std::vector<NVTETensor> bias_list;
  std::vector<NVTETensor> pre_gelu_list;
  std::vector<NVTETensor> out_list;
  std::vector<NVTETensor> workspace_list;

  // It is weird that TE/Common GEMM only use colwise for MXFP8
  bool rhs_use_colwise = (scaling_mode == JAXX_Scaling_Mode::MXFP8_1D_SCALING) && !rhs_is_trans;
  bool lhs_use_colwise = (scaling_mode == JAXX_Scaling_Mode::MXFP8_1D_SCALING) && lhs_is_trans;

  for (size_t i = 0; i < num_gemms; i++) {
    size_t m_i = dim_list_host[i];
    auto rhs_shape = std::vector<size_t>{rhs_is_trans ? n : k, rhs_is_trans ? k : n};
    auto lhs_shape = std::vector<size_t>{lhs_is_trans ? k : m_i, lhs_is_trans ? m_i : k};
    // auto lhs_shape = std::vector<size_t>{lhs_is_trans ? m_i : k,
    //                                     lhs_is_trans ? k : m_i};
    // auto rhs_shape = std::vector<size_t>{rhs_is_trans ? k : n,
    //                                      rhs_is_trans ? n : k};

    auto out_shape = std::vector<size_t>{m_i, n};
    auto lhs_scale_shape = std::vector<size_t>{1, 1};
    auto rhs_scale_shape = std::vector<size_t>{1, 1};

    auto lhs_i = TensorWrapper(get_nvte_scaling_mode(scaling_mode));
    auto rhs_i = TensorWrapper(get_nvte_scaling_mode(scaling_mode));

    if (rhs_use_colwise)  // MatA
      rhs_i.set_columnwise_data(static_cast<void *>(rhs_ptr), rhs_dtype, rhs_shape);
    else
      rhs_i.set_rowwise_data(static_cast<void *>(rhs_ptr), rhs_dtype, rhs_shape);

    if (lhs_use_colwise)  // MatB
      lhs_i.set_columnwise_data(static_cast<void *>(lhs_ptr), lhs_dtype, lhs_shape);
    else
      lhs_i.set_rowwise_data(static_cast<void *>(lhs_ptr), lhs_dtype, lhs_shape);

    auto lhs_scale_size = std::vector<size_t>{1};
    auto rhs_scale_size = std::vector<size_t>{1};
    if (scaling_mode == JAXX_Scaling_Mode::MXFP8_1D_SCALING) {
      NVTE_CHECK(k % MXFP8_BLOCK_SIZE == 0, "MXFP8 K-dim being divisble by %d (got %d)",
                 MXFP8_BLOCK_SIZE, k);
      size_t scale_k = k / MXFP8_BLOCK_SIZE;
      lhs_scale_size[0] = m_i * scale_k;
      rhs_scale_size[0] = n * scale_k;
      // Need to add swizzle here
    }
    if (is_fp8_dtype(lhs_dtype)) {
      if (rhs_use_colwise)  // MatA
        rhs_i.set_columnwise_scale_inv(static_cast<void *>(rhs_scale_ptr), rhs_scale_dtype,
                                       rhs_scale_size);
      else
        rhs_i.set_rowwise_scale_inv(static_cast<void *>(rhs_scale_ptr), rhs_scale_dtype,
                                    rhs_scale_size);
      if (lhs_use_colwise)  // MatB
        lhs_i.set_columnwise_scale_inv(static_cast<void *>(lhs_scale_ptr), lhs_scale_dtype,
                                       lhs_scale_size);
      else
        lhs_i.set_rowwise_scale_inv(static_cast<void *>(lhs_scale_ptr), lhs_scale_dtype,
                                    lhs_scale_size);
    } else {
      NVTE_CHECK(scaling_mode == JAXX_Scaling_Mode::NO_SCALING,
                 "Unsupported scaling mode: ", static_cast<int>(scaling_mode));
    }
    lhs_wrapper_list.push_back(std::move(lhs_i));
    rhs_wrapper_list.push_back(std::move(rhs_i));

    auto out_i = TensorWrapper(static_cast<void *>(out_ptr), out_shape, out_dtype);
    lhs_ptr += m_i * k * lhs_dtype_bytes;
    rhs_ptr += n * k * rhs_dtype_bytes;
    out_ptr += m_i * n * out_dtype_bytes;
    if (is_fp8_dtype(lhs_dtype)) {
      lhs_scale_ptr += lhs_scale_size[0] * lhs_scale_dtype_bytes;
      rhs_scale_ptr += rhs_scale_size[0] * rhs_scale_dtype_bytes;
    }

    auto bias_i = TensorWrapper(bias_ptr, bias_shape, bias_dtype);
    if (has_bias) bias_ptr += n * bias_dtype_bytes;
    auto pre_gelu_i = TensorWrapper(nullptr, std::vector<size_t>{0}, out_dtype);

    out_wrapper_list.push_back(std::move(out_i));
    bias_wrapper_list.push_back(std::move(bias_i));
    pre_gelu_wrapper_list.push_back(std::move(pre_gelu_i));

    lhs_list.push_back(lhs_wrapper_list.back().data());
    rhs_list.push_back(rhs_wrapper_list.back().data());
    bias_list.push_back(bias_wrapper_list.back().data());
    pre_gelu_list.push_back(pre_gelu_wrapper_list.back().data());
    out_list.push_back(out_wrapper_list.back().data());
  }

  auto workspace_shape = std::vector<size_t>{workspace_size};
  for (int i = 0; i < num_streams; i++) {
    auto workspace_i =
        TensorWrapper(static_cast<void *>(workspace_ptr), workspace_shape, DType::kByte);
    workspace_wrapper_list.push_back(std::move(workspace_i));
    workspace_list.push_back(workspace_wrapper_list.back().data());
    workspace_ptr += workspace_size;
  }

  nvte_multi_stream_cublas_gemm(rhs_list.data(), lhs_list.data(), out_list.data(), bias_list.data(),
                                pre_gelu_list.data(), num_gemms, rhs_is_trans, lhs_is_trans, grad,
                                workspace_list.data(), accumulate, use_split_accumulator,
                                num_math_sm, stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(GroupedGemmHandler, GroupedGemmFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // lhs_data
                                  .Arg<Buffer_Type>()      // lhs_scale
                                  .Arg<Buffer_Type>()      // rhs_data
                                  .Arg<Buffer_Type>()      // rhs_scale
                                  .Arg<Buffer_Type>()      // bias
                                  .Arg<Buffer_Type>()      // group_sizes
                                  .Arg<Buffer_Type>()      // group_offset
                                  .Ret<Buffer_Type>()      // output
                                  .Ret<Buffer_Type>()      // workspace
                                  .Attr<int64_t>("M")
                                  .Attr<int64_t>("N")
                                  .Attr<int64_t>("K")
                                  .Attr<bool>("lhs_is_trans")
                                  .Attr<bool>("rhs_is_trans")
                                  .Attr<JAXX_Scaling_Mode>("scaling_mode")
                                  .Attr<bool>("has_bias"),
                              FFI_CudaGraph_Traits);

}  // namespace jax
}  // namespace transformer_engine
