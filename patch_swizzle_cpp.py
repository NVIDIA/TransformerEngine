import re

with open("transformer_engine/pytorch/csrc/extensions/swizzle.cpp", "r") as f:
    content = f.read()

old_code = """  swizzle_output.set_with_gemm_swizzled_scales(true);
  NVTE_SCOPED_GIL_RELEASE({
    nvte_swizzle_grouped_scaling_factors(swizzle_input.data(), swizzle_output.data(),
                                         at::cuda::getCurrentCUDAStream());
  });"""

new_code = """  swizzle_output.set_with_gemm_swizzled_scales(true);

  size_t num_tensors = input.num_tensors();
  size_t workspace_size = (num_tensors + 2) * sizeof(int) + (num_tensors + 1) * sizeof(size_t);
  workspace_size = roundup(workspace_size, 256);
  auto workspace = allocateSpace(std::vector<size_t>{workspace_size}, transformer_engine::DType::kByte, false);

  NVTE_SCOPED_GIL_RELEASE({
    nvte_swizzle_grouped_scaling_factors(swizzle_input.data(), swizzle_output.data(),
                                         getDataPtr(workspace),
                                         at::cuda::getCurrentCUDAStream());
  });"""

content = content.replace(old_code, new_code)

# Check if first_dims error check exists and remove it
old_check = """  const auto first_dims = input.get_first_dims();
  const auto last_dims = input.get_last_dims();
  if (first_dims.data_ptr != nullptr || last_dims.data_ptr != nullptr) {
    NVTE_ERROR(
        "Grouped GEMM swizzle requires uniform shapes for now (first_dims/last_dims must be "
        "absent).");
  }"""

new_check = """  const auto first_dims = input.get_first_dims();
  const auto last_dims = input.get_last_dims();"""

content = content.replace(old_check, new_check)

with open("transformer_engine/pytorch/csrc/extensions/swizzle.cpp", "w") as f:
    f.write(content)
