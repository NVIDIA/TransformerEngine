/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/gemm.h>
#include <transformer_engine/swizzle.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <tuple>
#include <unordered_map>

#include "common/util/cuda_runtime.h"
#include "common/util/system.h"
#include "extensions.h"
#include "extensions/ffi.h"
#include "extensions/misc.h"
#include "extensions/utils.h"

namespace transformer_engine {

namespace jax {

static std::unordered_map<int64_t, CommOverlapCore *> comm_overlaps;

int64_t CreateCommOverlapBuffer(CommOverlapType comm_type, CommOverlapMethod method,
                                const std::vector<size_t> &buffer_shape, DType buffer_dtype,
                                int tp_size, int num_splits, int num_max_streams, int comm_cga_size,
                                int gemm_priority, int comm_priority, int num_comm_sm,
                                int set_sm_margin, bool use_ce, bool atomic_gemm,
                                bool rs_overlap_first_gemm, bool aggregate_ag) {
  int64_t unique_id = 0;
  hash_combine(unique_id, static_cast<int>(comm_type), static_cast<int>(method), buffer_shape[0],
               buffer_shape[0], static_cast<int>(buffer_dtype), tp_size, num_splits,
               num_max_streams, comm_cga_size, gemm_priority, comm_priority, num_comm_sm,
               set_sm_margin, use_ce, atomic_gemm, rs_overlap_first_gemm, aggregate_ag);

  auto it = comm_overlaps.find(unique_id);
  if (it == comm_overlaps.end()) {
    if (method == CommOverlapMethod::RING_EXCHANGE) {
      comm_overlaps[unique_id] = reinterpret_cast<CommOverlapCore *>(
          new CommOverlapP2PBase(buffer_shape, buffer_dtype, tp_size, comm_type, num_max_streams,
                                 comm_cga_size, gemm_priority, comm_priority, num_comm_sm,
                                 set_sm_margin, use_ce, atomic_gemm, aggregate_ag));
    } else {
      comm_overlaps[unique_id] = reinterpret_cast<CommOverlapCore *>(
          new CommOverlapBase(buffer_shape, buffer_dtype, tp_size, num_splits, num_max_streams,
                              comm_cga_size, gemm_priority, comm_priority, num_comm_sm,
                              set_sm_margin, atomic_gemm, rs_overlap_first_gemm));
    }
  }

  return unique_id;
}

void DestroyCommOverlapBuffer(size_t unique_id) {
  auto it = comm_overlaps.find(unique_id);
  if (it != comm_overlaps.end()) {
    delete it->second;
    comm_overlaps.erase(it);
  }
}

void DestroyAllCommOverlapBuffers() {
  for (auto it = comm_overlaps.begin(); it != comm_overlaps.end();) {
    delete it->second;
    it = comm_overlaps.erase(it);
  }
}

std::optional<NVTEBasicTensor> SwizzleScalingFactors(cudaStream_t stream, TensorWrapper &input,
                                                     bool rowwise) {
  if (input.scaling_mode() == NVTE_INVALID_SCALING) {
    NVTE_ERROR("Invalid scaling mode for swizzle.");
  } else if (input.scaling_mode() != NVTE_MXFP8_1D_SCALING) {
    return std::nullopt;
  }

  NVTE_CHECK(input.element_size() == 1, "8-bit input required for swizzling scaling factors.");

  NVTEBasicTensor scale_inv;
  if (rowwise) {
    scale_inv = input.get_rowwise_scale_inv();
  } else {
    scale_inv = input.get_columnwise_scale_inv();
  }
  auto input_shape = nvte_shape_to_vector(input.shape());
  auto scale_inv_shape = nvte_shape_to_vector(scale_inv.shape);
  void *scale_inv_dptr = scale_inv.data_ptr;

  // Allocate memory for swizzled output.
  void *swizzled_scale_inv_dptr;
  auto scale_inv_size = product(scale_inv_shape);
  NVTE_CHECK_CUDA(cudaMalloc(&swizzled_scale_inv_dptr, scale_inv_size));

  // Reconstruct input only to avoid swizzling both directions if not needed.
  // Use any 8 bit type, it's irrelevant.
  TensorWrapper input_cu(NVTE_MXFP8_1D_SCALING);
  TensorWrapper output_cu(NVTE_MXFP8_1D_SCALING);
  if (rowwise) {
    input_cu.set_rowwise_data(input.dptr(), DType::kFloat8E4M3, input_shape);
    input_cu.set_rowwise_scale_inv(scale_inv_dptr, DType::kFloat8E8M0, scale_inv_shape);
    output_cu.set_rowwise_data(input.dptr(), DType::kFloat8E4M3, input_shape);
    output_cu.set_rowwise_scale_inv(swizzled_scale_inv_dptr, DType::kFloat8E8M0, scale_inv_shape);
  } else {
    input_cu.set_columnwise_data(input.columnwise_dptr(), DType::kFloat8E4M3, input_shape);
    input_cu.set_columnwise_scale_inv(scale_inv_dptr, DType::kFloat8E8M0, scale_inv_shape);
    output_cu.set_columnwise_data(input.columnwise_dptr(), DType::kFloat8E4M3, input_shape);
    output_cu.set_columnwise_scale_inv(swizzled_scale_inv_dptr, DType::kFloat8E8M0,
                                       scale_inv_shape);
  }

  // Launch kernel
  nvte_swizzle_scaling_factors(input_cu.data(), output_cu.data(), stream);

  if (rowwise) {
    input.set_rowwise_scale_inv(swizzled_scale_inv_dptr, DType::kFloat8E8M0, scale_inv_shape);
  } else {
    input.set_columnwise_scale_inv(swizzled_scale_inv_dptr, DType::kFloat8E8M0, scale_inv_shape);
  }

  NVTEBasicTensor swizzled_scale_inv;
  swizzled_scale_inv.data_ptr = swizzled_scale_inv_dptr;
  swizzled_scale_inv.shape = nvte_make_shape(scale_inv_shape.data(), scale_inv_shape.size());
  swizzled_scale_inv.dtype = static_cast<NVTEDType>(DType::kByte);
  return swizzled_scale_inv;
}

Error_Type CollectiveGemmFFI(cudaStream_t stream, Buffer_Type lhs, Buffer_Type lhs_scale_inv,
                             Buffer_Type rhs, Buffer_Type rhs_scale_inv, Buffer_Type bias,
                             Buffer_Type pre_gelu_in, Buffer_Type aux_in, Result_Type out,
                             Result_Type bias_grad, Result_Type pre_gelu_out, Result_Type aux_out,
                             Result_Type workspace, JAXX_Scaling_Mode scaling_mode, bool lhs_trans,
                             bool rhs_trans, bool fuse_bias, bool fuse_gelu, bool grad,
                             bool accumulate, bool use_split_accumulator, int64_t comm_overlap_id,
                             CommOverlapMethod comm_overlap_method, CommOverlapType comm_type) {
  // LHS & RHS operands with collapsed 2D shapes
  auto scaling_mode_ = get_nvte_scaling_mode(scaling_mode);
  TensorWrapper lhs_(scaling_mode_), rhs_(scaling_mode_);

  // LHS flips row/colwise for cuBLAS column-major layout
  auto lhs_shape = std::vector<size_t>{product(lhs.dimensions(), 0, lhs.dimensions().size() - 1),
                                       static_cast<size_t>(lhs.dimensions().back())};
  auto lhs_dtype = convert_ffi_datatype_to_te_dtype(lhs.element_type());
  if (lhs_trans) {
    lhs_.set_columnwise_data(lhs.untyped_data(), lhs_dtype, lhs_shape);
  } else {
    lhs_.set_rowwise_data(lhs.untyped_data(), lhs_dtype, lhs_shape);
  }

  auto rhs_shape = std::vector<size_t>{product(rhs.dimensions(), 0, rhs.dimensions().size() - 1),
                                       static_cast<size_t>(rhs.dimensions().back())};
  auto rhs_dtype = convert_ffi_datatype_to_te_dtype(rhs.element_type());
  if (rhs_trans) {
    rhs_.set_columnwise_data(rhs.untyped_data(), rhs_dtype, rhs_shape);
  } else {
    rhs_.set_rowwise_data(rhs.untyped_data(), rhs_dtype, rhs_shape);
  }

  if (scaling_mode != JAXX_Scaling_Mode::NO_SCALING) {
    DType scale_inv_dtype =
        (scaling_mode_ == NVTE_MXFP8_1D_SCALING) ? DType::kFloat8E8M0 : DType::kFloat32;

    auto lhs_scale_inv_shape =
        std::vector<size_t>(lhs_scale_inv.dimensions().begin(), lhs_scale_inv.dimensions().end());
    if (lhs_trans) {
      lhs_.set_columnwise_scale_inv(lhs_scale_inv.untyped_data(), scale_inv_dtype,
                                    lhs_scale_inv_shape);
    } else {
      lhs_.set_rowwise_scale_inv(lhs_scale_inv.untyped_data(), scale_inv_dtype,
                                 lhs_scale_inv_shape);
    }

    auto rhs_scale_inv_shape =
        std::vector<size_t>(rhs_scale_inv.dimensions().begin(), rhs_scale_inv.dimensions().end());
    if (rhs_trans) {
      rhs_.set_columnwise_scale_inv(rhs_scale_inv.untyped_data(), scale_inv_dtype,
                                    rhs_scale_inv_shape);
    } else {
      rhs_.set_rowwise_scale_inv(rhs_scale_inv.untyped_data(), scale_inv_dtype,
                                 rhs_scale_inv_shape);
    }
  }

  // Output buffer
  auto out_shape = std::vector<size_t>{product(out->dimensions(), 0, out->dimensions().size() - 1),
                                       static_cast<size_t>(out->dimensions().back())};
  auto out_ = TensorWrapper(out->untyped_data(), out_shape,
                            convert_ffi_datatype_to_te_dtype(out->element_type()));

  // Bias tensor
  void *bias_ptr = nullptr;
  auto bias_shape = std::vector<size_t>{0};
  DType bias_dtype = DType::kBFloat16;
  if (fuse_bias) {
    if (grad) {
      bias_ptr = bias_grad->untyped_data();
      bias_shape =
          std::vector<size_t>(bias_grad->dimensions().begin(), bias_grad->dimensions().end());
      bias_dtype = convert_ffi_datatype_to_te_dtype(bias_grad->element_type());
    } else {
      bias_ptr = bias.untyped_data();
      bias_shape = std::vector<size_t>(bias.dimensions().begin(), bias.dimensions().end());
      bias_dtype = convert_ffi_datatype_to_te_dtype(bias.element_type());
    }
  }
  auto bias_ = TensorWrapper(bias_ptr, bias_shape, bias_dtype);

  // Pre-GeLU tensor
  void *pre_gelu_ptr = nullptr;
  auto pre_gelu_shape = std::vector<size_t>{0};
  DType pre_gelu_dtype = DType::kBFloat16;
  if (fuse_gelu) {
    if (grad) {
      pre_gelu_ptr = pre_gelu_in.untyped_data();
      pre_gelu_shape = std::vector<size_t>{
          product(pre_gelu_in.dimensions(), 0, pre_gelu_in.dimensions().size() - 1),
          static_cast<size_t>(pre_gelu_in.dimensions().back())};
      pre_gelu_dtype = convert_ffi_datatype_to_te_dtype(pre_gelu_in.element_type());
    } else {
      pre_gelu_ptr = pre_gelu_out->untyped_data();
      pre_gelu_shape = std::vector<size_t>{
          product(pre_gelu_out->dimensions(), 0, pre_gelu_out->dimensions().size() - 1),
          static_cast<size_t>(pre_gelu_out->dimensions().back())};
      pre_gelu_dtype = convert_ffi_datatype_to_te_dtype(pre_gelu_out->element_type());
    }
  }
  auto pre_gelu_ = TensorWrapper(pre_gelu_ptr, pre_gelu_shape, pre_gelu_dtype);

  // cuBLAS workspace
  auto workspace_ = TensorWrapper(workspace->untyped_data(),
                                  std::vector<size_t>{workspace->element_count()}, DType::kByte);

  // Keep swizzling factors alive here in order to de-allocate the memory later
  std::vector<std::optional<NVTEBasicTensor>> swizzled_scale_inverses_list;
  swizzled_scale_inverses_list.emplace_back(
      std::move(SwizzleScalingFactors(stream, lhs_, lhs_trans)));
  swizzled_scale_inverses_list.emplace_back(
      std::move(SwizzleScalingFactors(stream, rhs_, rhs_trans)));

  if (comm_type == CommOverlapType::NONE) {
    // No comm. overlap, do plain cuBLAS GEMM
    auto num_math_sm = cuda::sm_count() - getenv<int>("NVTE_EXT_MARGIN_SM", 0);
    nvte_cublas_gemm(rhs_.data(), lhs_.data(), bias_.data(), pre_gelu_.data(), out_.data(),
                     rhs_trans, lhs_trans, grad, workspace_.data(), accumulate,
                     use_split_accumulator, num_math_sm, stream);
  } else {
    // Prepare the auxiliary output tensor
    auto aux_out_shape =
        std::vector<size_t>(aux_out->dimensions().begin(), aux_out->dimensions().end());
    auto aux_out_dtype = convert_ffi_datatype_to_te_dtype(aux_out->element_type());
    auto aux_out_ = TensorWrapper(aux_out->untyped_data(), aux_out_shape, aux_out_dtype);

    auto executor = comm_overlaps[comm_overlap_id];
    if (comm_overlap_method == CommOverlapMethod::BULK) {
      // Copy the auxiliary data into the communications buffer
      auto aux_in_shape =
          std::vector<size_t>(aux_in.dimensions().begin(), aux_in.dimensions().end());
      auto aux_in_dtype = convert_ffi_datatype_to_te_dtype(aux_in.element_type());
      auto aux_in_ = TensorWrapper(aux_in.untyped_data(), aux_in_shape, aux_in_dtype);
      executor->copy_into_buffer(aux_in_, (comm_type == CommOverlapType::AG), stream);

      // Launch GEMM w/ bulk overlap
      executor->bulk_overlap(rhs_, rhs_trans, lhs_, lhs_trans, out_, bias_, pre_gelu_, workspace_,
                             grad, accumulate, use_split_accumulator, comm_type, aux_out_, stream);
    } else if (comm_type == CommOverlapType::RS) {
      // Launch GEMM+RS
      executor->split_overlap_rs(rhs_, rhs_trans, lhs_, lhs_trans, out_, bias_, pre_gelu_,
                                 workspace_, grad, accumulate, use_split_accumulator, aux_out_,
                                 stream);
    } else {
      // Copy the distributed LHS operand into the local chunk of the communication buffer
      executor->copy_into_buffer(lhs_, true, stream);

      // Launch AG+GEMM
      executor->split_overlap_ag(rhs_, rhs_trans, lhs_, lhs_trans, out_, bias_, pre_gelu_,
                                 workspace_, grad, accumulate, use_split_accumulator, aux_out_,
                                 stream);
    }
  }

  for (auto &scale_inv : swizzled_scale_inverses_list) {
    if (scale_inv.has_value()) {
      // Free memory we allocated when swizzling
      NVTE_CHECK_CUDA(cudaFree(scale_inv.value().data_ptr));
    }
  }

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CollectiveGemmHandler, CollectiveGemmFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // lhs
                                  .Arg<Buffer_Type>()      // lhs_scale_inv
                                  .Arg<Buffer_Type>()      // rhs
                                  .Arg<Buffer_Type>()      // rhs_scale_inv
                                  .Arg<Buffer_Type>()      // bias
                                  .Arg<Buffer_Type>()      // pre_gelu_in
                                  .Arg<Buffer_Type>()      // aux_in (for bulk overlaps)
                                  .Ret<Buffer_Type>()      // out
                                  .Ret<Buffer_Type>()      // bias_grad
                                  .Ret<Buffer_Type>()      // pre_gelu_out
                                  .Ret<Buffer_Type>()      // aux_out (gathered or scattered)
                                  .Ret<Buffer_Type>()      // workspace
                                  .Attr<JAXX_Scaling_Mode>("scaling_mode")
                                  .Attr<bool>("lhs_trans")
                                  .Attr<bool>("rhs_trans")
                                  .Attr<bool>("fuse_bias")
                                  .Attr<bool>("fuse_gelu")
                                  .Attr<bool>("grad")
                                  .Attr<bool>("accumulate")
                                  .Attr<bool>("use_split_accumulator")
                                  .Attr<int64_t>("comm_overlap_id")
                                  .Attr<CommOverlapMethod>("comm_overlap_method")
                                  .Attr<CommOverlapType>("comm_type"),
                              FFI_CudaGraph_Traits);

Error_Type GroupedGemmFFI(cudaStream_t stream, Variadic_Buffer_Type input_list,
                          Variadic_Result_Type output_list, int64_t num_gemms,
                          JAXX_Scaling_Mode scaling_mode, int64_t has_bias) {
  // Notes on matrix layouts and transpose:
  // Jax uses row-major data_layout, on entering this function, each input matrix pair:
  //   A: row-major with size [m, k],
  //   B: row-major with size [n, k], needs transpose,
  // on exiting this function, JAX expect:
  //   C: row-major with size [m, n].
  // cuBLAS uses column-major data_layout, in this view, each input matrix pair:
  //   A: column-major with size [k, m], needs transpose,
  //   B: column-major with size [k, n].
  // If we call cuBLAS GEMM for A * B, the output will be:
  //   C: column-major with size [m, n] --> row-major with size [n, m].
  // To make the output compatible with JAX, we need to swap A and B in cuBLAS GEMM call.

  if (num_gemms <= 0) {
    return ffi_with_cuda_error_check();
  }
  size_t expected_input_size = has_bias ? 5 * num_gemms : 4 * num_gemms;
  size_t expected_output_size = num_gemms + 1;
  size_t actual_input_size = input_list.size();
  size_t actual_output_size = output_list.size();
  NVTE_CHECK(actual_input_size == expected_input_size, "Expected %zu input tensors, got %zu",
             expected_input_size, actual_input_size);
  NVTE_CHECK(actual_output_size == expected_output_size, "Expected %zu output tensors, got %zu",
             expected_output_size, actual_output_size);

  bool trans_lhs = true;
  bool trans_rhs = false;
  auto num_math_sm = cuda::sm_count() - getenv<int>("NVTE_EXT_MARGIN_SM", 0);
  bool grad = false;
  bool accumulate = false;
  bool use_split_accumulator = false;

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

  int lhs_list_offset = 0;
  int rhs_list_offset = num_gemms;
  int lhs_sinv_list_offset = 2 * num_gemms;
  int rhs_sinv_list_offset = 3 * num_gemms;
  int bias_list_offset = 4 * num_gemms;
  int out_list_offset = 0;
  for (int i = 0; i < num_gemms; i++) {
    Buffer_Type lhs_i = input_list.get<Buffer_Type>(lhs_list_offset + i).value();
    Buffer_Type rhs_i = input_list.get<Buffer_Type>(rhs_list_offset + i).value();
    Buffer_Type lhs_sinv_i = input_list.get<Buffer_Type>(lhs_sinv_list_offset + i).value();
    Buffer_Type rhs_sinv_i = input_list.get<Buffer_Type>(rhs_sinv_list_offset + i).value();
    Result_Type out_i = output_list.get<Buffer_Type>(out_list_offset + i).value();

    DType lhs_dtype = convert_ffi_datatype_to_te_dtype(lhs_i.element_type());
    DType rhs_dtype = convert_ffi_datatype_to_te_dtype(rhs_i.element_type());
    DType out_dtype = convert_ffi_datatype_to_te_dtype(out_i->element_type());

    void *lhs_ptr = lhs_i.untyped_data();
    void *rhs_ptr = rhs_i.untyped_data();
    void *lhs_sinv_ptr = lhs_sinv_i.untyped_data();
    void *rhs_sinv_ptr = rhs_sinv_i.untyped_data();
    void *out_ptr = out_i->untyped_data();

    // Placeholder for bias since it can be empty
    DType bias_dtype = DType::kFloat32;
    void *bias_ptr = nullptr;

    auto lhs_shape_ = lhs_i.dimensions();
    auto rhs_shape_ = rhs_i.dimensions();

    // lhs and rhs has shape [1, m, k] and [1, n, k]
    size_t m = lhs_shape_[1];
    size_t n = rhs_shape_[1];
    size_t k = lhs_shape_[2];

    auto lhs_shape = std::vector<size_t>{m, k};
    auto rhs_shape = std::vector<size_t>{n, k};
    auto out_shape = std::vector<size_t>{n, m};
    auto lhs_sinv_shape = std::vector<size_t>{1, 1};
    auto rhs_sinv_shape = std::vector<size_t>{1, 1};

    if (scaling_mode == JAXX_Scaling_Mode::NO_SCALING ||
        scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING ||
        scaling_mode == JAXX_Scaling_Mode::CURRENT_TENSOR_SCALING) {
      float *amax_dptr = nullptr;
      float *scale_dptr = nullptr;
      auto lhs_i_ = TensorWrapper(lhs_ptr, lhs_shape, lhs_dtype, amax_dptr, scale_dptr,
                                  reinterpret_cast<float *>(lhs_sinv_ptr));
      auto rhs_i_ = TensorWrapper(rhs_ptr, rhs_shape, rhs_dtype, amax_dptr, scale_dptr,
                                  reinterpret_cast<float *>(rhs_sinv_ptr));
      lhs_wrapper_list.push_back(std::move(lhs_i_));
      rhs_wrapper_list.push_back(std::move(rhs_i_));
    } else if (scaling_mode == JAXX_Scaling_Mode::MXFP8_1D_SCALING) {
      // Note: the scale_inv array should have been swizzled in Python before lowering
      auto lhs_sinv_shape_ = lhs_sinv_i.dimensions();
      auto rhs_sinv_shape_ = rhs_sinv_i.dimensions();
      for (int i = 0; i < 2; i++) {
        lhs_sinv_shape[i] = lhs_sinv_shape_[i];
        rhs_sinv_shape[i] = rhs_sinv_shape_[i];
      }

      NVTEScalingMode nvte_scaling_mode = get_nvte_scaling_mode(scaling_mode);
      TensorWrapper lhs_i_(nvte_scaling_mode);
      TensorWrapper rhs_i_(nvte_scaling_mode);
      lhs_i_.set_rowwise_data(lhs_ptr, lhs_dtype, lhs_shape);
      rhs_i_.set_rowwise_data(rhs_ptr, rhs_dtype, rhs_shape);
      lhs_i_.set_rowwise_scale_inv(lhs_sinv_ptr, DType::kFloat8E8M0, lhs_sinv_shape);
      rhs_i_.set_rowwise_scale_inv(rhs_sinv_ptr, DType::kFloat8E8M0, rhs_sinv_shape);

      lhs_wrapper_list.push_back(std::move(lhs_i_));
      rhs_wrapper_list.push_back(std::move(rhs_i_));
    } else {
      NVTE_ERROR("Unsupported scaling mode: ", static_cast<int>(scaling_mode));
    }

    auto out_i_ = TensorWrapper(out_ptr, out_shape, out_dtype);
    void *pre_gelu_ptr = nullptr;
    auto bias_shape = std::vector<size_t>{0};
    auto pre_gelu_shape = std::vector<size_t>{0};
    if (has_bias) {
      auto bias_i_get = input_list.get<Buffer_Type>(bias_list_offset + i);
      Buffer_Type bias_i = bias_i_get.value();
      bias_ptr = bias_i.untyped_data();
      bias_dtype = convert_ffi_datatype_to_te_dtype(bias_i.element_type());
      bias_shape[0] = n;
    }
    auto bias_i = TensorWrapper(bias_ptr, bias_shape, bias_dtype);
    auto pre_gelu_i = TensorWrapper(pre_gelu_ptr, pre_gelu_shape, out_dtype);

    out_wrapper_list.push_back(std::move(out_i_));
    bias_wrapper_list.push_back(std::move(bias_i));
    pre_gelu_wrapper_list.push_back(std::move(pre_gelu_i));

    lhs_list.push_back(lhs_wrapper_list.back().data());
    rhs_list.push_back(rhs_wrapper_list.back().data());
    bias_list.push_back(bias_wrapper_list.back().data());
    pre_gelu_list.push_back(pre_gelu_wrapper_list.back().data());
    out_list.push_back(out_wrapper_list.back().data());
  }

  auto workspace_get = output_list.get<Buffer_Type>(num_gemms);
  Result_Type workspace = workspace_get.value();
  uint8_t *workspace_ptr = reinterpret_cast<uint8_t *>(workspace->untyped_data());
  size_t workspace_size = workspace->dimensions()[0] / num_streams;
  auto workspace_shape = std::vector<size_t>{workspace_size};
  for (int i = 0; i < num_streams; i++) {
    auto workspace_i =
        TensorWrapper(static_cast<void *>(workspace_ptr), workspace_shape, DType::kByte);
    workspace_wrapper_list.push_back(std::move(workspace_i));
    workspace_list.push_back(workspace_wrapper_list.back().data());
    workspace_ptr += workspace_size;
  }

  nvte_multi_stream_cublas_gemm(rhs_list.data(), lhs_list.data(), out_list.data(), bias_list.data(),
                                pre_gelu_list.data(), num_gemms, trans_lhs, trans_rhs, grad,
                                workspace_list.data(), accumulate, use_split_accumulator,
                                num_math_sm, stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(GroupedGemmHandler, GroupedGemmFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .RemainingArgs()         // input list
                                  .RemainingRets()         // output list
                                  .Attr<int64_t>("num_gemms")
                                  .Attr<JAXX_Scaling_Mode>("scaling_mode")
                                  .Attr<int64_t>("has_bias"),
                              FFI_CudaGraph_Traits);

}  // namespace jax
}  // namespace transformer_engine
