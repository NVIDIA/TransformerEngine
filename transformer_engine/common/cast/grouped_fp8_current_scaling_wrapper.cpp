/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/grouped_fp8_current_scaling.h"
#include "../common.h"

namespace transformer_engine {
namespace detail {

// Forward declarations for internal C++ functions
void launch_grouped_fp8_quantize_rowwise(
    const GroupedTensor& input,
    GroupedTensor& output,
    cudaStream_t stream
);

void launch_grouped_fp8_quantize_columnwise(
    const GroupedTensor& input,
    GroupedTensor& output,
    cudaStream_t stream
);

} // namespace detail
} // namespace transformer_engine

/*
 * C API Wrapper Functions
 * 
 * These functions provide the C API that can be called from Python via pybind11.
 * They handle conversion from NVTEGroupedTensor (C opaque pointer) to
 * GroupedTensor (C++ class) and call the appropriate C++ implementation.
 */

extern "C" {

void nvte_grouped_fp8_quantize_rowwise(
    const NVTEGroupedTensor input,
    NVTEGroupedTensor output,
    cudaStream_t stream
) {
    NVTE_API_CALL(nvte_grouped_fp8_quantize_rowwise);
    using namespace transformer_engine;
    using namespace transformer_engine::detail;
    
    // Convert C opaque pointers to C++ objects
    const GroupedTensor* input_tensor = convertNVTEGroupedTensorCheck(input);
    GroupedTensor* output_tensor = convertNVTEGroupedTensorCheck(output);
    
    // Validate inputs
    NVTE_CHECK(input_tensor != nullptr, "Input grouped tensor is null");
    NVTE_CHECK(output_tensor != nullptr, "Output grouped tensor is null");
    NVTE_CHECK(input_tensor->num_tensors == output_tensor->num_tensors,
               "Input and output must have same number of tensors");
    NVTE_CHECK(output_tensor->has_data(), "Output must have rowwise data buffer allocated");
    NVTE_CHECK(output_tensor->scale != nullptr, "Output must have scales computed");
    
    // Launch the C++ kernel
    launch_grouped_fp8_quantize_rowwise(*input_tensor, *output_tensor, stream);
}

void nvte_grouped_fp8_quantize_columnwise(
    const NVTEGroupedTensor input,
    NVTEGroupedTensor output,
    cudaStream_t stream
) {
    NVTE_API_CALL(nvte_grouped_fp8_quantize_columnwise);
    using namespace transformer_engine;
    using namespace transformer_engine::detail;
    
    // Convert C opaque pointers to C++ objects
    const GroupedTensor* input_tensor = convertNVTEGroupedTensorCheck(input);
    GroupedTensor* output_tensor = convertNVTEGroupedTensorCheck(output);
    
    // Validate inputs
    NVTE_CHECK(input_tensor != nullptr, "Input grouped tensor is null");
    NVTE_CHECK(output_tensor != nullptr, "Output grouped tensor is null");
    NVTE_CHECK(input_tensor->num_tensors == output_tensor->num_tensors,
               "Input and output must have same number of tensors");
    NVTE_CHECK(output_tensor->has_columnwise_data(), 
               "Output must have columnwise data buffer allocated");
    NVTE_CHECK(output_tensor->scale != nullptr, "Output must have scales computed");
    
    // Verify all tensors are 2D (required for transpose)
    for (int i = 0; i < input_tensor->num_tensors; i++) {
        NVTE_CHECK(input_tensor->shapes[i].size() == 2,
                   "Columnwise quantization requires 2D tensors, tensor ", i, " has ",
                   input_tensor->shapes[i].size(), " dimensions");
    }
    
    // Launch the C++ kernel
    launch_grouped_fp8_quantize_columnwise(*input_tensor, *output_tensor, stream);
}

void nvte_grouped_fp8_quantize_both(
    const NVTEGroupedTensor input,
    NVTEGroupedTensor output,
    cudaStream_t stream
) {
    NVTE_API_CALL(nvte_grouped_fp8_quantize_both);
    using namespace transformer_engine;
    using namespace transformer_engine::detail;
    
    // Convert C opaque pointers to C++ objects
    const GroupedTensor* input_tensor = convertNVTEGroupedTensorCheck(input);
    GroupedTensor* output_tensor = convertNVTEGroupedTensorCheck(output);
    
    // Validate inputs
    NVTE_CHECK(input_tensor != nullptr, "Input grouped tensor is null");
    NVTE_CHECK(output_tensor != nullptr, "Output grouped tensor is null");
    NVTE_CHECK(input_tensor->num_tensors == output_tensor->num_tensors,
               "Input and output must have same number of tensors");
    NVTE_CHECK(output_tensor->has_data(), "Output must have rowwise data buffer allocated");
    NVTE_CHECK(output_tensor->has_columnwise_data(),
               "Output must have columnwise data buffer allocated");
    NVTE_CHECK(output_tensor->scale != nullptr, "Output must have scales computed");
    
    // Launch both quantization variants
    // Note: In the future, this could be optimized to share computation
    // or launch a fused kernel that produces both outputs
    launch_grouped_fp8_quantize_rowwise(*input_tensor, *output_tensor, stream);
    launch_grouped_fp8_quantize_columnwise(*input_tensor, *output_tensor, stream);
}

} // extern "C"
