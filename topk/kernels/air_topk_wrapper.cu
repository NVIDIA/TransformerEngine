#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Include headers from the GitLab dependency
// Need to include utility headers first before standalone_air_topk
#include "nv_util.h"
#include "standalone_air_topk.cuh"

// Use the namespace where the functions are defined
using namespace nv::air_topk;

// Allocate all buffers needed for the kernel
// Returns (buffer, out_indices)
std::tuple<torch::Tensor, torch::Tensor> allocate_air_topk_buffers(torch::Tensor input, int k, bool is_prefill) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D [batch_size, seq_len]");

    int batch_size = input.size(0);
    int len = input.size(1);

    // Allocate output tensors
    auto out_indices = torch::empty({batch_size, k}, torch::dtype(torch::kInt32).device(input.device()));

    // Calculate buffer size needed
    size_t buf_size = 0;
    nv::standalone_air_topk<float, int>(nullptr, buf_size, input.data_ptr<float>(), batch_size, len, k,
                                        nullptr, out_indices.data_ptr<int>(),
                                        true,  // greater = true (for top-k, not bottom-k)
                                        nullptr, nullptr, is_prefill);

    // Allocate buffer
    auto buffer = torch::empty({(int64_t)buf_size}, torch::dtype(torch::kUInt8).device(input.device()));

    return std::make_tuple(buffer, out_indices);
}

// Call the kernel with pre-allocated buffers
void call_air_topk_kernel(torch::Tensor input, torch::Tensor lengths, int k, torch::Tensor buffer,
                          torch::Tensor out_indices, bool is_prefill) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D [batch_size, seq_len]");
    TORCH_CHECK(buffer.is_cuda(), "Buffer must be a CUDA tensor");
    TORCH_CHECK(buffer.is_contiguous(), "Buffer must be contiguous");
    TORCH_CHECK(out_indices.is_cuda(), "out_indices must be a CUDA tensor");

    int batch_size = input.size(0);
    int len = input.size(1);
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // Calculate buffer size needed
    size_t buf_size = 0;
    nv::standalone_air_topk<float, int>(nullptr, buf_size, input.data_ptr<float>(), batch_size, len, k,
                                        nullptr, out_indices.data_ptr<int>(),
                                        true,  // greater = true (for top-k, not bottom-k)
                                        stream, lengths.data_ptr<int>(), is_prefill);
    TORCH_CHECK(buffer.numel() >= (int64_t)buf_size, "Buffer is too small");

    // Call the actual kernel with current CUDA stream
    nv::standalone_air_topk<float, int>(buffer.data_ptr(), buf_size, input.data_ptr<float>(), batch_size, len,
                                        k, nullptr, out_indices.data_ptr<int>(),
                                        true,  // greater = true (for top-k, not bottom-k)
                                        stream, lengths.data_ptr<int>(), is_prefill);
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("allocate_buffers", &allocate_air_topk_buffers,
          "Allocate all buffers for AIR TopK: returns (buffer, out_values, out_indices)", py::arg("input"),
          py::arg("k"), py::arg("is_prefill"));
    m.def("topk_kernel", &call_air_topk_kernel, "Call AIR TopK kernel with pre-allocated buffers",
          py::arg("input"), py::arg("lengths"), py::arg("k"), py::arg("buffer"), py::arg("out_indices"), py::arg("is_prefill"));
}
