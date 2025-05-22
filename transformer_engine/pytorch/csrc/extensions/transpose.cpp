/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <pybind.h>

#include <optional>

#include "extensions.h"
#include "pybind.h"

#include <nvtx3/nvToolsExt.h>

namespace transformer_engine::pytorch {

void _fused_bulk_alloc_outputs(at::Tensor input_view, std::vector<int> &m_splits, 
                                std::vector<std::unique_ptr<Quantizer>> &quantizers,
                                std::vector<py::handle> quantizer_list,
                                std::vector<TensorWrapper> &output_list,
                                std::vector<py::object> &output_list_py) {
  using namespace py::literals;

  nvtxRangePush("_fused_bulk_alloc_outputs");

  nvtxRangePush("prep_fused_bulk_alloc_outputs");

  int num_splits = m_splits.size();

  bool rowwise_usage = quantizers[0]->rowwise_usage;
  bool columnwise_usage = quantizers[0]->columnwise_usage;
  size_t hidden_dim = input_view.size(1);



  std::vector<Float8BlockQuantizer*> blockwise_quantizers;
  for (size_t i = 0; i < quantizers.size(); i++) {
    Quantizer* raw_ptr = quantizers[i].get();
    Float8BlockQuantizer* blockwise_quantizer = static_cast<Float8BlockQuantizer*>(raw_ptr);
    blockwise_quantizers.push_back(blockwise_quantizer);
  }

  NVTEScalingMode scaling_mode = blockwise_quantizers[0]->get_scaling_mode();
  bool is_2D_scaled = scaling_mode == NVTE_BLOCK_SCALING_2D;
  transformer_engine::DType fp8_dtype = blockwise_quantizers[0]->dtype;

  size_t fp8_elem_size = 1;
  size_t scale_elem_size = 4;

  std::vector<std::vector<size_t>> rowwise_data_shapes;
  std::vector<std::vector<size_t>> rowwise_scale_shapes;
  std::vector<size_t> rowwise_data_sizes;
  std::vector<size_t> rowwise_scale_sizes;
  std::vector<std::vector<size_t>> columnwise_data_shapes;
  std::vector<std::vector<size_t>> columnwise_scale_shapes;
  std::vector<size_t> columnwise_data_sizes;
  std::vector<size_t> columnwise_scale_sizes;
  for (int i = 0; i < num_splits; i++) {
    std::vector<size_t> input_view_i_shape =
        std::vector<size_t>{(size_t)m_splits[i], (size_t)hidden_dim};
    if (rowwise_usage) {
      rowwise_data_shapes.emplace_back(input_view_i_shape);
      rowwise_scale_shapes.emplace_back(blockwise_quantizers[i]->get_scale_shape(
          input_view_i_shape, false));
      rowwise_data_sizes.emplace_back(input_view_i_shape[0] * input_view_i_shape[1] *
                                      fp8_elem_size);
      rowwise_scale_sizes.emplace_back(rowwise_scale_shapes.back()[0] *
                                        rowwise_scale_shapes.back()[1] * scale_elem_size);
    }
    if (columnwise_usage) {
      columnwise_data_shapes.emplace_back(
          std::vector<size_t>{input_view_i_shape[1], input_view_i_shape[0]});
      columnwise_scale_shapes.emplace_back(blockwise_quantizers[i]->get_scale_shape(
          input_view_i_shape, true));
      columnwise_data_sizes.emplace_back(input_view_i_shape[0] * input_view_i_shape[1] *
                                          fp8_elem_size);
      columnwise_scale_sizes.emplace_back(columnwise_scale_shapes.back()[0] *
                                          columnwise_scale_shapes.back()[1] *
                                          scale_elem_size);
    }
  }

  size_t total_size_rowwise_data =
      std::accumulate(rowwise_data_sizes.begin(), rowwise_data_sizes.end(), 0);
  size_t total_size_rowwise_scale =
      std::accumulate(rowwise_scale_sizes.begin(), rowwise_scale_sizes.end(), 0);
  size_t total_size_columnwise_data =
      std::accumulate(columnwise_data_sizes.begin(), columnwise_data_sizes.end(), 0);
  size_t total_size_columnwise_scale =
      std::accumulate(columnwise_scale_sizes.begin(), columnwise_scale_sizes.end(), 0);

  size_t total_size_rowwise = total_size_rowwise_data + total_size_rowwise_scale;
  size_t total_size_columnwise = total_size_columnwise_data + total_size_columnwise_scale;

  std::vector<at::Tensor> rowwise_data_list;
  std::vector<at::Tensor> rowwise_scale_list;
  std::vector<at::Tensor> columnwise_data_list;
  std::vector<at::Tensor> columnwise_scale_list;

  at::Tensor rowwise_full_tensor;
  at::Tensor columnwise_full_tensor;

  // each from_blob will hold a reference to the full tensor, since we need to keep the full tensor alive
  // when all the views are gone, the full tensor will be garbage collected
  std::shared_ptr<at::Tensor> rowwise_full_tensor_holder;
  std::shared_ptr<at::Tensor> columnwise_full_tensor_holder;

  nvtxRangePop();

  nvtxRangePush("construct_rowwise_tensors");

  if (rowwise_usage) {
    rowwise_full_tensor =
        at::empty({(int64_t)total_size_rowwise}, at::device(at::kCUDA).dtype(torch::kUInt8));
    rowwise_full_tensor_holder = std::make_shared<at::Tensor>(rowwise_full_tensor);
    // use raw pointer math + from blob, avoid torch slice to reduce cpu overhead
    uint8_t* rowwise_data_ptr = rowwise_full_tensor.data_ptr<uint8_t>();
    uint8_t* rowwise_scale_ptr =
        rowwise_full_tensor.data_ptr<uint8_t>() + total_size_rowwise_data;
    // use from_blob to construct rowwise_data_list and rowwise_scale_list
    for (int i = 0; i < num_splits; i++) {
      if (rowwise_data_sizes[i] == 0) {
        NVTE_CHECK(rowwise_scale_sizes[i] == 0,
                    "Rowwise scale size is not 0 when rowwise data size is 0");
        rowwise_data_list.emplace_back(
            at::empty({static_cast<int64_t>(rowwise_data_shapes[i][0]),
                        static_cast<int64_t>(rowwise_data_shapes[i][1])},
                      at::device(at::kCUDA).dtype(torch::kUInt8)));
        rowwise_scale_list.emplace_back(
            at::empty({static_cast<int64_t>(rowwise_scale_shapes[i][0]),
                        static_cast<int64_t>(rowwise_scale_shapes[i][1])},
                      at::device(at::kCUDA).dtype(torch::kFloat32)));
      } else {
        rowwise_data_list.emplace_back(at::from_blob(
            rowwise_data_ptr,
            {static_cast<int64_t>(rowwise_data_shapes[i][0]),
              static_cast<int64_t>(rowwise_data_shapes[i][1])},
            [rowwise_full_tensor_holder](void*) {}, at::device(at::kCUDA).dtype(torch::kUInt8)));
        rowwise_scale_list.emplace_back(at::from_blob(
            rowwise_scale_ptr,
            {static_cast<int64_t>(rowwise_scale_shapes[i][0]),
              static_cast<int64_t>(rowwise_scale_shapes[i][1])},
            [rowwise_full_tensor_holder](void*) {},
            at::device(at::kCUDA).dtype(torch::kFloat32)));
        rowwise_data_ptr += rowwise_data_sizes[i];
        rowwise_scale_ptr += rowwise_scale_sizes[i];
      }
    }
  }

  nvtxRangePop();

  nvtxRangePush("construct_columnwise_tensors");

  if (columnwise_usage) {
    columnwise_full_tensor =
        at::empty({(int64_t)total_size_columnwise}, at::device(at::kCUDA).dtype(torch::kUInt8));
    columnwise_full_tensor_holder = std::make_shared<at::Tensor>(columnwise_full_tensor);
    uint8_t* columnwise_data_ptr = columnwise_full_tensor.data_ptr<uint8_t>();
    uint8_t* columnwise_scale_ptr =
        columnwise_full_tensor.data_ptr<uint8_t>() + total_size_columnwise_data;
    for (int i = 0; i < num_splits; i++) {
      if (columnwise_data_sizes[i] == 0) {
        NVTE_CHECK(columnwise_scale_sizes[i] == 0,
                    "Columnwise scale size is not 0 when columnwise data size is 0");
        columnwise_data_list.emplace_back(
            at::empty({static_cast<int64_t>(columnwise_data_shapes[i][0]),
                        static_cast<int64_t>(columnwise_data_shapes[i][1])},
                      at::device(at::kCUDA).dtype(torch::kUInt8)));
        columnwise_scale_list.emplace_back(
            at::empty({static_cast<int64_t>(columnwise_scale_shapes[i][0]),
                        static_cast<int64_t>(columnwise_scale_shapes[i][1])},
                      at::device(at::kCUDA).dtype(torch::kFloat32)));
      } else {
        columnwise_data_list.emplace_back(at::from_blob(
            columnwise_data_ptr,
            {static_cast<int64_t>(columnwise_data_shapes[i][0]),
              static_cast<int64_t>(columnwise_data_shapes[i][1])},
            [columnwise_full_tensor_holder](void*) {},
            at::device(at::kCUDA).dtype(torch::kUInt8)));
        columnwise_scale_list.emplace_back(at::from_blob(
            columnwise_scale_ptr,
            {static_cast<int64_t>(columnwise_scale_shapes[i][0]),
              static_cast<int64_t>(columnwise_scale_shapes[i][1])},
            [columnwise_full_tensor_holder](void*) {},
            at::device(at::kCUDA).dtype(torch::kFloat32)));
        columnwise_data_ptr += columnwise_data_sizes[i];
        columnwise_scale_ptr += columnwise_scale_sizes[i];
      }
    }
  }

  nvtxRangePop();

  for (int i = 0; i < num_splits; i++) {
    nvtxRangePush(std::string("assemble_py_output_objects_" + std::to_string(i)).c_str());

    py::handle Float8BlockwiseQTensorClass(
        reinterpret_cast<PyObject*>(Float8BlockwiseQTensorBasePythonClass));

    // Create the tensor object with proper reference counting
    py::object rowwise_data = rowwise_usage ? py::cast(rowwise_data_list[i]) : py::none();
    py::object columnwise_data =
        columnwise_usage ? py::cast(columnwise_data_list[i]) : py::none();
    py::object rowwise_scale = rowwise_usage ? py::cast(rowwise_scale_list[i]) : py::none();
    py::object columnwise_scale =
        columnwise_usage ? py::cast(columnwise_scale_list[i]) : py::none();

    py::object ret = Float8BlockwiseQTensorClass(
        "rowwise_data"_a = rowwise_data, "columnwise_data"_a = columnwise_data,
        "rowwise_scale_inv"_a = rowwise_scale, "columnwise_scale_inv"_a = columnwise_scale,
        "fp8_dtype"_a = fp8_dtype, "quantizer"_a = quantizer_list[i],
        "is_2D_scaled"_a = is_2D_scaled);

    output_list_py.emplace_back(std::move(ret));

    nvtxRangePop();

    nvtxRangePush(std::string("assemble_tensor_wrappers_" + std::to_string(i)).c_str());

    // as for tensor wrappers, these tensor wrappers are going to be quantized, so no need to insert empty tensors here
    if (m_splits[i] > 0) {
      TensorWrapper te_tensor = makeTransformerEngineTensor(
        rowwise_usage ? rowwise_data_list[i].data_ptr() : nullptr,
        columnwise_usage ? columnwise_data_list[i].data_ptr() : nullptr,
        rowwise_data_shapes[i],
        columnwise_data_shapes[i],
        fp8_dtype,
        nullptr, nullptr, 
        rowwise_usage ? rowwise_scale_list[i].data_ptr() : nullptr,
        columnwise_usage ? columnwise_scale_list[i].data_ptr() : nullptr,
        rowwise_scale_shapes[i],
        columnwise_scale_shapes[i],
        scaling_mode
      );

      output_list.emplace_back(std::move(te_tensor));
    }

    nvtxRangePop();

  }
  
  nvtxRangePop();
}

std::vector<py::object> fused_multi_quantize(std::vector<at::Tensor> input_list, at::Tensor input_view, std::vector<int> m_splits,
                                             std::vector<py::handle> quantizer_list, DType otype) {
  init_extension();
  std::vector<NVTETensor> nvte_tensor_input_list;
  std::vector<NVTETensor> nvte_tensor_output_list;
  std::vector<py::object> py_output_objects_list;
  std::vector<TensorWrapper> tensor_wrappers_input;
  std::vector<TensorWrapper> tensor_wrappers_output;

  NVTE_CHECK(input_list.size() == quantizer_list.size(), "Input list and quantizer list must have the same size");
  NVTE_CHECK(input_list.size() == m_splits.size(), "Input list and m_splits must have the same size");

  // Choose implementation
  // Note: Currently only have fused kernel for FP8 cast-transpose
  bool with_fused_kernel = true;
  // currently only fp8 subchannel recipe supports bulk alloc, fall back to for loop if not
  bool use_fused_bulk_alloc = true;

  for (size_t i = 0; i < quantizer_list.size(); i++) {
    if (!detail::IsFloat8Quantizers(quantizer_list[i].ptr())) {
      with_fused_kernel = false;
    }
    if (!detail::IsFloat8BlockwiseQuantizers(quantizer_list[i].ptr())) {
      use_fused_bulk_alloc = false;
    }
  }

  // convert the quantizers to C++ objects
  std::vector<std::unique_ptr<Quantizer>> quantizers;
  for (size_t i = 0; i < quantizer_list.size(); i++) {
    quantizers.push_back(convert_quantizer(quantizer_list[i]));
  }

  // create TE tensors from input
  if (use_fused_bulk_alloc) {
    nvtxRangePush("construct_input_tensor_wrappers");
    for (size_t i = 0; i < input_list.size(); i++) {
      if (m_splits[i] == 0) continue;
      tensor_wrappers_input.emplace_back(makeTransformerEngineTensor(input_list[i]));
    }
    nvtxRangePop();
    _fused_bulk_alloc_outputs(input_view, m_splits, quantizers, quantizer_list, tensor_wrappers_output, py_output_objects_list);
  }else{
    for (size_t i = 0; i < input_list.size(); i++) {
      nvtxRangePush(std::string("for_loop_create_outputs_" + std::to_string(i)).c_str());
      // raise error is each input[i] is not contiguous
      NVTE_CHECK(input_list[i].is_contiguous(), "Input tensor is not contiguous");

      auto input_tensor = makeTransformerEngineTensor(input_list[i]);
      const NVTEShape input_shape = input_tensor.shape();

      TensorWrapper output_tensor;

      auto& quantizer = quantizers[i];
      std::vector<size_t> output_shape(input_shape.data, input_shape.data + input_shape.ndim);
      py::object o;
      std::tie(output_tensor, o) = quantizer->create_tensor(output_shape, otype);
      py_output_objects_list.push_back(o);

      if (input_tensor.numel() == 0) continue;

      nvte_tensor_output_list.emplace_back(output_tensor.data());
      nvte_tensor_input_list.emplace_back(input_tensor.data());
      tensor_wrappers_input.emplace_back(std::move(input_tensor));
      tensor_wrappers_output.emplace_back(std::move(output_tensor));

      nvtxRangePop();
    }
  }

  // Check tensor lists
  NVTE_CHECK(nvte_tensor_output_list.size() == nvte_tensor_input_list.size(),
             "Number of input and output tensors must match");

  for (size_t i = 0; i < nvte_tensor_output_list.size(); i++) {
    if (nvte_tensor_columnwise_data(nvte_tensor_output_list[i]) == nullptr) {
      with_fused_kernel = false;
      break;
    }
  }

  // Launch TE kernel
  if (with_fused_kernel) {
    NVTE_SCOPED_GIL_RELEASE({
      nvte_multi_cast_transpose(nvte_tensor_input_list.size(), nvte_tensor_input_list.data(),
                                nvte_tensor_output_list.data(), at::cuda::getCurrentCUDAStream());
    });
  } else {
    size_t non_empty_tensor_idx = 0;

    NVTE_CHECK(tensor_wrappers_input.size() == tensor_wrappers_output.size(), "Input tensor wrappers and output tensor wrappers must have the same size");

    for (size_t i = 0; i < py_output_objects_list.size(); i++) {
      // quantize(input_list[i], quantizer_list[i], py_output_objects_list[i], std::nullopt);

      if (m_splits[i] == 0) continue;

      quantize_cpp(tensor_wrappers_input[non_empty_tensor_idx], quantizer_list[i], quantizers[i], tensor_wrappers_output[non_empty_tensor_idx], std::nullopt);
      non_empty_tensor_idx++;
    }
  }
  return py_output_objects_list;
}

at::Tensor fp8_transpose(at::Tensor input, DType otype, std::optional<at::Tensor> output) {
  init_extension();

  const auto dim = input.dim();
  NVTE_CHECK(dim >= 2, "Need at least 2D tensor to transpose.");

  if (input.dim() > 2) {
    input = input.view({-1, input.size(dim - 1)});
  }

  size_t M = static_cast<size_t>(input.size(0));
  size_t N = static_cast<size_t>(input.size(1));

  at::Tensor out;
  if (output.has_value()) {
    out = *output;
  } else {
    out = allocateTorchTensor(input.size(1), input.size(0), DType::kByte);
  }
  if (M == 0 || N == 0) return out;

  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), std::vector<size_t>{M, N}, otype);
  auto output_cu = makeTransformerEngineTensor(out.data_ptr(), std::vector<size_t>{N, M}, otype);

  nvte_transpose(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return out;
}

}  // namespace transformer_engine::pytorch
