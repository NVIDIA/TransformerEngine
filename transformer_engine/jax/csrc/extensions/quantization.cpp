/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include <cuda_runtime.h>

#include <iostream>

#include "../extensions.h"
#include "transformer_engine/cast.h"
#include "transformer_engine/hadamard_transform.h"
#include "transformer_engine/recipe.h"
#include "transformer_engine/transformer_engine.h"
#include "xla/ffi/api/c_api.h"

namespace transformer_engine {
namespace jax {

pybind11::tuple GetDBiasQuantizeWorkspaceSizes(size_t batch_size, size_t hidden_size,
                                               DType in_dtype, DType out_dtype, DType scale_dtype,
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

  bool const is_nvfp4 = scaling_mode == JAXX_Scaling_Mode::NVFP4_1D_SCALING ||
                        scaling_mode == JAXX_Scaling_Mode::NVFP4_2D_SCALING;

  auto input_tensor = TensorWrapper(reinterpret_cast<void *>(&temp), input_shape, in_dtype);
  auto dbias_tensor = TensorWrapper(reinterpret_cast<void *>(&temp), dbias_shape, in_dtype);

  auto output_tensor = TensorWrapper(get_nvte_scaling_mode(scaling_mode));
  auto scale_shape = std::vector<size_t>{1};
  // Only the pointers will be checked for scale_inv, thus the shapes do not matter
  if (q_layout == QuantizeLayout::ROWWISE_COLWISE || q_layout == QuantizeLayout::ROWWISE) {
    output_tensor.set_rowwise_data(reinterpret_cast<void *>(&temp), out_dtype, output_shape);
    if (scaling_mode != JAXX_Scaling_Mode::NO_SCALING) {
      if (is_nvfp4)
        scale_shape = get_block_scale_shape(scaling_mode, batch_size, hidden_size, false);
      output_tensor.set_rowwise_scale_inv(reinterpret_cast<void *>(&temp), scale_dtype,
                                          scale_shape);
    }
  }

  if (q_layout == QuantizeLayout::ROWWISE_COLWISE || q_layout == QuantizeLayout::COLWISE) {
    auto &tmp_shape = scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING ? output_trans_shape
                                                                                : output_shape;
    output_tensor.set_columnwise_data(reinterpret_cast<void *>(&temp), out_dtype, tmp_shape);

    // Only the pointers will be checked for scale_inv, thus the shapes do not matter
    if (scaling_mode != JAXX_Scaling_Mode::NO_SCALING) {
      if (is_nvfp4)
        scale_shape =
            get_block_scale_shape(scaling_mode, hidden_size, batch_size, false);  //Transpose
      output_tensor.set_columnwise_scale_inv(reinterpret_cast<void *>(&temp), scale_dtype,
                                             scale_shape);
    }
  }

  if (scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING || is_nvfp4) {
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
                            Buffer_Type amax_buf, Buffer_Type sr_rng_state,
                            Buffer_Type post_rht_amax_buf, Buffer_Type rht_matrix_buf,
                            Result_Type output_buf, Result_Type output_trans_buf,
                            Result_Type scale_inv_buf, Result_Type colwise_scale_inv_buf,
                            Result_Type updated_amax_buf, Result_Type dbias_buf,
                            Result_Type workspace_buf, JAXX_Scaling_Mode scaling_mode,
                            int64_t quantize_layout_enum, bool is_dbias, int64_t flatten_axis,
                            bool stochastic_rounding, bool use_rht) {
  auto in_dtype = convert_ffi_datatype_to_te_dtype(input_buf.element_type());
  auto out_dtype = convert_ffi_datatype_to_te_dtype(output_buf->element_type());
  auto workspace_dtype = convert_ffi_datatype_to_te_dtype(workspace_buf->element_type());

  NVTE_CHECK(is_fp8_dtype(out_dtype) || is_fp4_dtype(out_dtype),
             "Output datatype must be FP8 or FP4 for quantization.");

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

  bool const is_tensor_scaling = scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING ||
                                 scaling_mode == JAXX_Scaling_Mode::CURRENT_TENSOR_SCALING;
  bool const is_mxfp8 = scaling_mode == JAXX_Scaling_Mode::MXFP8_1D_SCALING;
  bool const is_nvfp4 = scaling_mode == JAXX_Scaling_Mode::NVFP4_1D_SCALING ||
                        scaling_mode == JAXX_Scaling_Mode::NVFP4_2D_SCALING;

  NVTE_CHECK(!stochastic_rounding || is_nvfp4, "Stochastic rounding is only supported for NVFP4.");
  NVTE_CHECK(!use_rht || is_nvfp4, "RHT is only supported for NVFP4 scaling");

  if (quantize_layout == QuantizeLayout::ROWWISE ||
      quantize_layout == QuantizeLayout::ROWWISE_COLWISE) {
    output_tensor.set_rowwise_data(output, out_dtype, output_shape);

    if (is_tensor_scaling) {
      float *scale = reinterpret_cast<float *>(scale_buf.untyped_data());
      float *amax = reinterpret_cast<float *>(updated_amax_buf->untyped_data());
      NVTE_CHECK(scale != nullptr, "scale must be provided for delayed tensor scaling");
      NVTE_CHECK(amax != nullptr, "amax must be provided for delayed tensor scaling");
      output_tensor.set_scale(scale, DType::kFloat32, std::vector<size_t>{1});
      output_tensor.set_amax(amax, DType::kFloat32, std::vector<size_t>{1});
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

  if (is_nvfp4) {
    float *amax = reinterpret_cast<float *>(amax_buf.untyped_data());
    NVTE_CHECK(amax != nullptr, "amax must be provided for NVFP4");
    output_tensor.set_amax(amax, DType::kFloat32, std::vector<size_t>{1});
  }

  QuantizationConfigWrapper quant_config{};
  if (scaling_mode == JAXX_Scaling_Mode::NVFP4_2D_SCALING) {
    quant_config.set_nvfp4_2d_quantization(true);
  }

  // Stochastic rounding
  quant_config.set_stochastic_rounding(stochastic_rounding);
  TensorWrapper sr_rng_state_tensor(sr_rng_state.untyped_data(), std::vector<size_t>{2},
                                    DType::kInt64);
  if (stochastic_rounding) {
    NVTE_CHECK(sr_rng_state.size_bytes() == 2 * sizeof(uint64_t),
               "rng_state must be of type int64[2]");
    NVTE_CHECK(sr_rng_state.untyped_data() != nullptr, "rng_state must be provided for SR");
    quant_config.set_rng_state(sr_rng_state_tensor.data());
  }

  if (quantize_layout == QuantizeLayout::COLWISE ||
      quantize_layout == QuantizeLayout::ROWWISE_COLWISE) {
    if (is_nvfp4 && use_rht) {
      if (quantize_layout == QuantizeLayout::ROWWISE_COLWISE) {
        // Do regular rowwise quantization without RHT
        nvte_quantize_v2(input_tensor.data(), output_tensor.data(), quant_config, stream);
      }

      TensorWrapper out_transpose(get_nvte_scaling_mode(scaling_mode));

      // nvte_hadamard_transform_cast_fusion_columnwise expects the colwise data to be populated in the rowwise buffers on TensorWrapper
      out_transpose.set_rowwise_data(output_trans, out_dtype, output_trans_shape);
      auto const colwise_flatten_axis = output_trans_buf->dimensions().size() - flatten_axis;
      out_transpose.set_rowwise_scale_inv(
          colwise_scale_inv_buf->untyped_data(),
          convert_ffi_datatype_to_te_dtype(colwise_scale_inv_buf->element_type()),
          std::vector<size_t>{product(colwise_scale_inv_buf->dimensions(), 0, colwise_flatten_axis),
                              product(colwise_scale_inv_buf->dimensions(), colwise_flatten_axis,
                                      colwise_scale_inv_buf->dimensions().size())});

      float *post_rht_amax = reinterpret_cast<float *>(post_rht_amax_buf.untyped_data());
      NVTE_CHECK(post_rht_amax != nullptr, "Post-RHT colwise amax must be provided for NVFP4");
      out_transpose.set_amax(post_rht_amax, DType::kFloat32, std::vector<size_t>{1});

      bool const eligible_for_rht_cast_fusion =
          input_tensor.dtype() == DType::kBFloat16 && m % 64 == 0 && n % 128 == 0;
      NVTE_CHECK(eligible_for_rht_cast_fusion, "RHT cast fusion conditions not met");

      NVTE_CHECK(
          convert_ffi_datatype_to_te_dtype(rht_matrix_buf.element_type()) == DType::kBFloat16,
          "RHT matrix must be bf16");
      NVTE_CHECK(rht_matrix_buf.dimensions().size() == 2 && rht_matrix_buf.dimensions()[0] == 16 &&
                     rht_matrix_buf.dimensions()[1] == 16,
                 "RHT matrix must be 16x16");
      TensorWrapper rht_matrix_tensor(rht_matrix_buf.untyped_data(), std::vector<size_t>{16, 16},
                                      DType::kBFloat16);

      nvte_hadamard_transform_cast_fusion_columnwise(input_tensor.data(), out_transpose.data(),
                                                     rht_matrix_tensor.data(), quant_config,
                                                     stream);

      return ffi_with_cuda_error_check();
    }

    bool const is_colwise_transposed =
        scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING || is_nvfp4;
    auto &tmp_shape = is_colwise_transposed ? output_trans_shape : output_shape;
    output_tensor.set_columnwise_data(output_trans, out_dtype, tmp_shape);
    // For 2x delayed scaling, the scale buffer is shared between rowwise and columnwise scaling
    auto &tmp_buf = is_tensor_scaling ? scale_inv_buf : colwise_scale_inv_buf;

    if (is_tensor_scaling) {
      output_tensor.set_columnwise_scale_inv(
          tmp_buf->untyped_data(), convert_ffi_datatype_to_te_dtype(tmp_buf->element_type()),
          std::vector<size_t>{1});
    } else {
      auto colwise_flatten_axis = flatten_axis;
      if (is_colwise_transposed) {
        // convert flatten_axis from N layout to T layout
        colwise_flatten_axis = tmp_buf->dimensions().size() - flatten_axis;
      }
      output_tensor.set_columnwise_scale_inv(
          tmp_buf->untyped_data(), convert_ffi_datatype_to_te_dtype(tmp_buf->element_type()),
          std::vector<size_t>{
              product(tmp_buf->dimensions(), 0, colwise_flatten_axis),
              product(tmp_buf->dimensions(), colwise_flatten_axis, tmp_buf->dimensions().size())});
    }
  }

  auto dbias_tensor = TensorWrapper(dbias, dbias_shape, in_dtype);
  auto workspace_tensor = TensorWrapper(workspace, workspace_shape, workspace_dtype);

  if (is_dbias) {
    NVTE_CHECK(scaling_mode != JAXX_Scaling_Mode::NVFP4_2D_SCALING,
               "DBias quantization is not supported for NVFP4_2D_SCALING as fused dbias API cannot "
               "take quant_config as input.");
    nvte_quantize_dbias(input_tensor.data(), output_tensor.data(), dbias_tensor.data(),
                        workspace_tensor.data(), stream);
  } else {
    nvte_quantize_v2(input_tensor.data(), output_tensor.data(), quant_config, stream);
  }
  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(DBiasQuantizeHandler, DBiasQuantizeFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // input
                                  .Arg<Buffer_Type>()      // scale
                                  .Arg<Buffer_Type>()      // amax
                                  .Arg<Buffer_Type>()      // sr_rng_state
                                  .Arg<Buffer_Type>()      // colwise amax
                                  .Arg<Buffer_Type>()      // rht matrix
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
                                  .Attr<int64_t>("flatten_axis")
                                  .Attr<bool>("stochastic_rounding")
                                  .Attr<bool>("use_rht"),
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
                              Buffer_Type group_sizes, Result_Type outputs,
                              Result_Type colwise_outputs, Result_Type scale_invs,
                              Result_Type colwise_scale_invs, Result_Type amaxs,
                              JAXX_Scaling_Mode scaling_mode, int64_t quantize_layout_enum,
                              int64_t flatten_axis) {
  NVTE_CHECK(scaling_mode != JAXX_Scaling_Mode::NO_SCALING,
             "Unsupported scaling mode: ", static_cast<int>(scaling_mode));

  auto in_dtype = convert_ffi_datatype_to_te_dtype(inputs.element_type());
  auto out_dtype = convert_ffi_datatype_to_te_dtype(outputs->element_type());
  NVTE_CHECK(is_fp8_dtype(out_dtype), "Output datatype must be FP8 for quantization.");

  auto scale_dtype = convert_ffi_datatype_to_te_dtype(scales.element_type());
  auto group_size_dtype = convert_ffi_datatype_to_te_dtype(group_sizes.element_type());
  auto sinv_dtype = convert_ffi_datatype_to_te_dtype(scale_invs->element_type());
  auto amax_dtype = convert_ffi_datatype_to_te_dtype(amaxs->element_type());
  auto const quantize_layout = static_cast<QuantizeLayout>(quantize_layout_enum);

  auto *input_ptr = reinterpret_cast<uint8_t *>(inputs.untyped_data());
  auto *scale_ptr = reinterpret_cast<uint8_t *>(scales.untyped_data());
  auto *output_ptr = reinterpret_cast<uint8_t *>(outputs->untyped_data());
  auto *colwise_output_ptr = reinterpret_cast<uint8_t *>(colwise_outputs->untyped_data());
  auto *sinv_ptr = reinterpret_cast<uint8_t *>(scale_invs->untyped_data());
  auto *colwise_sinv_ptr = reinterpret_cast<uint8_t *>(colwise_scale_invs->untyped_data());
  auto *amax_ptr = reinterpret_cast<uint8_t *>(amaxs->untyped_data());

  bool has_rowwise = quantize_layout == QuantizeLayout::ROWWISE ||
                     quantize_layout == QuantizeLayout::ROWWISE_COLWISE;
  bool has_colwise = quantize_layout == QuantizeLayout::COLWISE ||
                     quantize_layout == QuantizeLayout::ROWWISE_COLWISE;
  bool is_delayed_scaling = scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING;
  bool const is_tensor_scaling = scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING ||
                                 scaling_mode == JAXX_Scaling_Mode::CURRENT_TENSOR_SCALING;
  bool const is_mxfp8_scaling = scaling_mode == JAXX_Scaling_Mode::MXFP8_1D_SCALING;

  size_t input_dtype_bytes = te_dtype_bytes(in_dtype);
  size_t output_dtype_bytes = te_dtype_bytes(out_dtype);
  size_t sinv_dtype_bytes = te_dtype_bytes(sinv_dtype);
  size_t group_size_dtype_bytes = te_dtype_bytes(group_size_dtype);
  size_t colwise_output_dtype_bytes = has_colwise ? output_dtype_bytes : 0;
  size_t colwise_sinv_dtype_bytes = has_colwise ? sinv_dtype_bytes : 0;
  size_t scale_dtype_bytes = is_tensor_scaling ? te_dtype_bytes(scale_dtype) : 0;
  size_t amax_dtype_bytes = is_tensor_scaling ? te_dtype_bytes(amax_dtype) : 0;

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

  size_t num_groups = group_sizes.dimensions()[0];
  size_t dim_list_bytes = group_size_dtype_bytes * num_groups;
  std::vector<int32_t> dim_list_host(num_groups);
  auto *group_size_ptr = reinterpret_cast<int32_t *>(group_sizes.untyped_data());
  cudaMemcpyAsync(dim_list_host.data(), group_size_ptr, dim_list_bytes, cudaMemcpyDeviceToHost,
                  stream);
  // Note: This may break cudaGraph.
  cudaStreamSynchronize(stream);

  size_t sum_group_sizes = std::accumulate(dim_list_host.begin(), dim_list_host.end(), 0);
  NVTE_CHECK(m == sum_group_sizes || input_dims[0] == sum_group_sizes,
             "Unexpected group_sizes! Got %zu (M=%zu, input_dims[0] = %zu)", sum_group_sizes, m,
             input_dims[0]);

  if (is_delayed_scaling) {
    NVTE_CHECK(amaxs->dimensions()[0] == num_groups, "Unexpected amax size, Expected ", num_groups,
               ", got ", amaxs->dimensions()[0]);
    NVTE_CHECK(amax_dtype == DType::kFloat32 && scale_dtype == DType::kFloat32);
    cudaMemsetAsync(amax_ptr, 0, sizeof(float) * num_groups, stream);
  }

  size_t sinv_size = 0;
  size_t colwise_sinv_size = 0;
  size_t non_group_m = flatten_axis > 1 ? product(input_dims, 1, flatten_axis) : 1;
  size_t num_non_empty_groups = 0;
  size_t total_rowwise_sinv_size = 0;
  size_t total_colwise_sinv_size = 0;
  for (size_t i = 0; i < num_groups; i++) {
    size_t m_i = dim_list_host[i] * non_group_m;
    // Skip for zero-size input + shiff the scale ptr
    if (m_i == 0) {
      if (is_tensor_scaling) scale_ptr += scale_dtype_bytes;
      continue;
    }
    num_non_empty_groups++;
    auto shape_i = std::vector<size_t>{m_i, n};
    auto shape_trans_i = std::vector<size_t>{n, m_i};

    auto inp_i = TensorWrapper(static_cast<void *>(input_ptr), shape_i, in_dtype);
    auto out_i = TensorWrapper(get_nvte_scaling_mode(scaling_mode));

    if (has_rowwise) {
      out_i.set_rowwise_data(static_cast<void *>(output_ptr), out_dtype, shape_i);

      if (is_fp8_dtype(out_dtype)) {
        if (is_tensor_scaling) {
          out_i.set_scale(static_cast<void *>(scale_ptr), DType::kFloat32, std::vector<size_t>{1});
          out_i.set_amax(static_cast<void *>(amax_ptr), DType::kFloat32, std::vector<size_t>{1});
          out_i.set_rowwise_scale_inv(static_cast<void *>(sinv_ptr), sinv_dtype,
                                      std::vector<size_t>{1});
          sinv_size = 1;
        } else {
          const bool is_colwise = false;
          auto sinv_shape_i = get_block_scale_shape(scaling_mode, m_i, n, is_colwise);
          out_i.set_rowwise_scale_inv(static_cast<void *>(sinv_ptr), sinv_dtype, sinv_shape_i);
          sinv_size = product(sinv_shape_i);
        }
      }
    }

    if (has_colwise) {
      auto &tmp_shape = is_tensor_scaling ? shape_trans_i : shape_i;
      out_i.set_columnwise_data(static_cast<void *>(colwise_output_ptr), out_dtype, tmp_shape);
      // For 2x delayed scaling, the scale buffer is shared between rowwise and columnwise scaling
      auto &tmp_sinv_ptr = is_tensor_scaling ? sinv_ptr : colwise_sinv_ptr;

      if (is_tensor_scaling) {
        out_i.set_columnwise_scale_inv(static_cast<void *>(tmp_sinv_ptr), sinv_dtype,
                                       std::vector<size_t>{1});
        colwise_sinv_size = 1;
      } else {
        const bool is_colwise = true;
        auto sinv_shape_i = get_block_scale_shape(scaling_mode, m_i, n, is_colwise);
        out_i.set_columnwise_scale_inv(static_cast<void *>(colwise_sinv_ptr), sinv_dtype,
                                       sinv_shape_i);
        colwise_sinv_size = product(sinv_shape_i);
      }
    }

    input_holders.push_back(std::move(inp_i));
    output_holders.push_back(std::move(out_i));

    input_list.push_back(input_holders.back().data());
    output_list.push_back(output_holders.back().data());

    input_ptr += m_i * n * input_dtype_bytes;
    scale_ptr += scale_dtype_bytes;
    output_ptr += m_i * n * output_dtype_bytes;
    colwise_output_ptr += m_i * n * colwise_output_dtype_bytes;
    sinv_ptr += sinv_size * sinv_dtype_bytes;
    colwise_sinv_ptr += colwise_sinv_size * colwise_sinv_dtype_bytes;
    amax_ptr += amax_dtype_bytes;
    total_rowwise_sinv_size += sinv_size;
    total_colwise_sinv_size += colwise_sinv_size;
  }
  if (is_mxfp8_scaling) {
    nvte_memset(scale_invs->untyped_data(), 0, total_rowwise_sinv_size, stream);
    nvte_memset(colwise_scale_invs->untyped_data(), 0, total_colwise_sinv_size, stream);
  }

  QuantizationConfigWrapper quant_config;
  nvte_multi_tensor_quantize(input_list.data(), output_list.data(), quant_config,
                             num_non_empty_groups, stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(GroupedQuantizeHandler, GroupedQuantizeFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // input
                                  .Arg<Buffer_Type>()      // scale
                                  .Arg<Buffer_Type>()      // group_sizes
                                  .Ret<Buffer_Type>()      // output
                                  .Ret<Buffer_Type>()      // colwise output
                                  .Ret<Buffer_Type>()      // scale_inv
                                  .Ret<Buffer_Type>()      // scale_inv colwise
                                  .Ret<Buffer_Type>()      // amax
                                  .Attr<JAXX_Scaling_Mode>("scaling_mode")
                                  .Attr<int64_t>("q_layout")
                                  .Attr<int64_t>("flatten_axis"));

}  // namespace jax
}  // namespace transformer_engine
