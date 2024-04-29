#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

extern "C"
__global__ void attn_copy(__nv_bfloat16* A, int* seq_len, __nv_bfloat16* B, int max_seq_len, int b, int s) {
    for(int batch_idx = blockIdx.x; batch_idx < b; batch_idx += gridDim.x) {
        int per_block = s / blockDim.x;
        int remainder = s % blockDim.x;
        int copy_block_offset_begin = per_block * threadIdx.x + min(threadIdx.x, remainder);

        int offset = seq_len[batch_idx];

        __nv_bfloat16* begin_A_copy = A + max_seq_len * s * batch_idx + s * offset; 
        __nv_bfloat16* begin_B_copy = B + s * batch_idx;

        int limit = copy_block_offset_begin + per_block + (threadIdx.x < remainder ? 1 : 0);
        
        for(int i = copy_block_offset_begin; i < limit; i++) {
            *(begin_A_copy + i) = *(begin_B_copy + i);
        }
    } 
}

extern "C"
__global__ void gv(float* src, int* seq_len, float* dst,  int d, int b) {
    // src [s, 1, 1, d]
    // dst [b]
    for(int batch_idx = blockIdx.x; batch_idx < b; batch_idx += gridDim.x) {
        int per_block = d / blockDim.x;
        int remainder = d % blockDim.x;
        int copy_block_offset_begin = per_block * threadIdx.x + min(threadIdx.x, remainder);

        int offset = seq_len[batch_idx];

        float* begin_src_copy = src + d * offset; 
        float* begin_dst_copy = dst + d * batch_idx;

        int limit = copy_block_offset_begin + per_block + (threadIdx.x < remainder ? 1 : 0);
        
        for(int i = copy_block_offset_begin; i < limit; i++) {
            *(begin_dst_copy + i) = *(begin_src_copy + i);
        }
    } 
}






void attention_copy(torch::Tensor A, torch::Tensor seq_len, torch::Tensor B, int max_seq_len, int b, int s, void* stream_ptr) {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    attn_copy<<<16, 32, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(A.data_ptr<torch::BFloat16>()),
                          seq_len.data_ptr<int>(),
                          reinterpret_cast<__nv_bfloat16*>(B.data_ptr<torch::BFloat16>()), max_seq_len, b, s);
}


void attention_copy2(torch::Tensor A, torch::Tensor seq_len, torch::Tensor B, int max_seq_len, int b, int s) {
    attn_copy<<<16, 32, 0>>>(reinterpret_cast<__nv_bfloat16*>(A.data_ptr<torch::BFloat16>()),
                          seq_len.data_ptr<int>(),
                          reinterpret_cast<__nv_bfloat16*>(B.data_ptr<torch::BFloat16>()), max_seq_len, b, s);
}


void get_values(torch::Tensor A, torch::Tensor seq_len, torch::Tensor B,  int d, int b, void* stream_ptr) {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    gv<<<16, 32, 0, stream>>>(A.data_ptr<float>(),
                          seq_len.data_ptr<int>(),
                          B.data_ptr<float>(),  d, b);
}


void get_values2(torch::Tensor A, torch::Tensor seq_len, torch::Tensor B,  int d, int b) {
    gv<<<16, 32, 0>>>((A.data_ptr<float>()),
                       seq_len.data_ptr<int>(),
                       (B.data_ptr<float>()), d, b);
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_copy", &attention_copy, "Copy function for attention mechanism",
          py::arg("A"), py::arg("seq_len"), py::arg("B"), py::arg("b"), py::arg("max_seq_len"), py::arg("s"), py::arg("stream_ptr"));

    m.def("attention_copy2", &attention_copy2, "Copy function for attention mechanism",
          py::arg("A"), py::arg("seq_len"), py::arg("B"), py::arg("b"), py::arg("max_seq_len"), py::arg("s"));

    m.def("get_values", &get_values, "1Get values function",
          py::arg("A"), py::arg("seq_len"), py::arg("B"),  py::arg("d"),  py::arg("b"), py::arg("stream_ptr"));

    m.def("get_values2", &get_values2, "2Get values function",
          py::arg("A"), py::arg("seq_len"), py::arg("B"), py::arg("d"),  py::arg("b"));
}