/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <pybind.h>

#include "../extensions.h"
#include <iostream>
#include <optional>

#include "extensions.h"
#include "pybind.h"

namespace transformer_engine::pytorch {

std::vector<py::object> fused_bulk_alloc_outputs(at::Tensor input_view, std::vector<int> m_splits,
                                                 std::vector<py::handle> quantizer_list) {
  init_extension();
  using namespace pybind11::literals;  // For operator""_a

  int num_splits = m_splits.size();

  // convert all the quantizers
  std::vector<std::unique_ptr<Quantizer>> quantizers;
  for (int i = 0; i < num_splits; i++) {
    quantizers.push_back(convert_quantizer(quantizer_list[i]));
  }

  bool rowwise_usage = quantizers[0]->rowwise_usage;
  bool columnwise_usage = quantizers[0]->columnwise_usage;
  size_t hidden_dim = input_view.size(1);

  std::vector<py::object> output_list;

  if (detail::IsFloat8BlockwiseQuantizers(quantizer_list[0].ptr())) {
    // implement the fuse bulk alloc for blockwise quantizer
    // downcast quantizers, resorces are owned by the unique_ptr, so use raw ptr here just to get the attributes
    std::vector<Float8BlockQuantizer*> blockwise_quantizers;
    for (size_t i = 0; i < quantizers.size(); i++) {
      Quantizer* raw_ptr = quantizers[i].get();
      Float8BlockQuantizer* blockwise_quantizer = static_cast<Float8BlockQuantizer*>(raw_ptr);
      blockwise_quantizers.push_back(blockwise_quantizer);
    }

    bool is_2D_scaled = blockwise_quantizers[0]->get_scaling_mode() == NVTE_BLOCK_SCALING_2D;
    transformer_engine::DType fp8_dtype = blockwise_quantizers[0]->dtype;

    size_t fp8_elem_size = 1;
    size_t scale_elem_size = 4;

    std::vector<std::pair<size_t, size_t>> rowwise_data_shapes;
    std::vector<std::pair<size_t, size_t>> rowwise_scale_shapes;
    std::vector<size_t> rowwise_data_sizes;
    std::vector<size_t> rowwise_scale_sizes;
    std::vector<std::pair<size_t, size_t>> columnwise_data_shapes;
    std::vector<std::pair<size_t, size_t>> columnwise_scale_shapes;
    std::vector<size_t> columnwise_data_sizes;
    std::vector<size_t> columnwise_scale_sizes;
    for (int i = 0; i < num_splits; i++) {
      std::pair<size_t, size_t> input_view_i_shape =
          std::make_pair((size_t)m_splits[i], (size_t)hidden_dim);
      if (rowwise_usage) {
        rowwise_data_shapes.emplace_back(input_view_i_shape);
        rowwise_scale_shapes.emplace_back(blockwise_quantizers[i]->get_scale_shape(
            {input_view_i_shape.first, input_view_i_shape.second}, false));
        rowwise_data_sizes.emplace_back(input_view_i_shape.first * input_view_i_shape.second *
                                        fp8_elem_size);
        rowwise_scale_sizes.emplace_back(rowwise_scale_shapes.back().first *
                                         rowwise_scale_shapes.back().second * scale_elem_size);
      }
      if (columnwise_usage) {
        columnwise_data_shapes.emplace_back(
            std::make_pair(input_view_i_shape.second, input_view_i_shape.first));
        columnwise_scale_shapes.emplace_back(blockwise_quantizers[i]->get_scale_shape(
            {input_view_i_shape.first, input_view_i_shape.second}, true));
        columnwise_data_sizes.emplace_back(input_view_i_shape.first * input_view_i_shape.second *
                                           fp8_elem_size);
        columnwise_scale_sizes.emplace_back(columnwise_scale_shapes.back().first *
                                            columnwise_scale_shapes.back().second *
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

    if (rowwise_usage) {
      rowwise_full_tensor = at::empty({(int64_t)total_size_rowwise},
                                      at::device(input_view.device()).dtype(torch::kUInt8));
      rowwise_full_tensor_holder = std::make_shared<at::Tensor>(rowwise_full_tensor);
      // use raw pointer math + from blob, avoid torch slice to reduce cpu overhead
      uint8_t* rowwise_data_ptr = rowwise_full_tensor.data_ptr<uint8_t>();
      uint8_t* rowwise_scale_ptr =
          rowwise_full_tensor.data_ptr<uint8_t>() + total_size_rowwise_data;
      // use from_blob to construct rowwise_data_list and rowwise_scale_list
      for (int i = 0; i < num_splits; i++) {
        rowwise_data_list.emplace_back(
            at::from_blob(rowwise_data_ptr,
                          {static_cast<int64_t>(rowwise_data_shapes[i].first),
                           static_cast<int64_t>(rowwise_data_shapes[i].second)},
                           [rowwise_full_tensor_holder](void *){},
                          at::device(input_view.device()).dtype(torch::kUInt8)));
        rowwise_scale_list.emplace_back(
            at::from_blob(rowwise_scale_ptr,
                          {static_cast<int64_t>(rowwise_scale_shapes[i].first),
                           static_cast<int64_t>(rowwise_scale_shapes[i].second)},
                          [rowwise_full_tensor_holder](void *){},
                          at::device(input_view.device()).dtype(torch::kFloat32)));
        rowwise_data_ptr += rowwise_data_sizes[i];
        rowwise_scale_ptr += rowwise_scale_sizes[i];
      }
    }

    if (columnwise_usage) {
      columnwise_full_tensor = at::empty({(int64_t)total_size_columnwise},
                                         at::device(input_view.device()).dtype(torch::kUInt8));
      columnwise_full_tensor_holder = std::make_shared<at::Tensor>(columnwise_full_tensor);
      uint8_t* columnwise_data_ptr = columnwise_full_tensor.data_ptr<uint8_t>();
      uint8_t* columnwise_scale_ptr =
          columnwise_full_tensor.data_ptr<uint8_t>() + total_size_columnwise_data;
      for (int i = 0; i < num_splits; i++) {
        columnwise_data_list.emplace_back(
            at::from_blob(columnwise_data_ptr,
                          {static_cast<int64_t>(columnwise_data_shapes[i].first),
                           static_cast<int64_t>(columnwise_data_shapes[i].second)},
                          [columnwise_full_tensor_holder](void *){},
                          at::device(input_view.device()).dtype(torch::kUInt8)));
        columnwise_scale_list.emplace_back(
            at::from_blob(columnwise_scale_ptr,
                          {static_cast<int64_t>(columnwise_scale_shapes[i].first),
                           static_cast<int64_t>(columnwise_scale_shapes[i].second)},
                          [columnwise_full_tensor_holder](void *){},
                          at::device(input_view.device()).dtype(torch::kFloat32)));
        columnwise_data_ptr += columnwise_data_sizes[i];
        columnwise_scale_ptr += columnwise_scale_sizes[i];
      }
    }

    for (int i = 0; i < num_splits; i++) {
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

      output_list.emplace_back(std::move(ret));
    }

  } else {
    NVTE_ERROR("Fused bulk alloc is not supported for this quantizer type");
  }

  return output_list;
}

std::vector<py::object> fused_multi_quantize(std::vector<at::Tensor> input_list,
                                             std::optional<std::vector<py::object>> output_list,
                                             std::vector<py::handle> quantizer_list, DType otype) {
  init_extension();
  std::vector<NVTETensor> nvte_tensor_input_list;
  std::vector<NVTETensor> nvte_tensor_output_list;
  std::vector<py::object> py_output_objects_list;
  std::vector<TensorWrapper> tensor_wrappers;
  if (output_list.has_value()) {
    py_output_objects_list = output_list.value();
  }

  // Choose implementation
  // Note: Currently only have fused kernel for FP8 cast-transpose
  bool with_fused_kernel = true;

  // create TE tensors from input
  for (size_t i = 0; i < input_list.size(); i++) {
    auto input_tensor = makeTransformerEngineTensor(input_list[i]);
    const NVTEShape input_shape = input_tensor.shape();

    TensorWrapper output_tensor;

    if (!detail::IsFloat8Quantizers(quantizer_list[i].ptr())) {
      with_fused_kernel = false;
    }
    if (output_list == std::nullopt) {
      std::unique_ptr<Quantizer> quantizer = convert_quantizer(quantizer_list[i]);
      std::vector<size_t> output_shape(input_shape.data, input_shape.data + input_shape.ndim);
      py::object o;
      std::tie(output_tensor, o) = quantizer->create_tensor(output_shape, otype);
      py_output_objects_list.push_back(o);
    } else {
      output_tensor = makeTransformerEngineTensor((*output_list)[i], quantizer_list[i]);
    }
    if (input_tensor.numel() == 0) continue;

    nvte_tensor_output_list.emplace_back(output_tensor.data());
    nvte_tensor_input_list.emplace_back(input_tensor.data());
    tensor_wrappers.emplace_back(std::move(input_tensor));
    tensor_wrappers.emplace_back(std::move(output_tensor));
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
    for (size_t i = 0; i < py_output_objects_list.size(); i++) {
      quantize(input_list[i], quantizer_list[i], py_output_objects_list[i], std::nullopt);
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
