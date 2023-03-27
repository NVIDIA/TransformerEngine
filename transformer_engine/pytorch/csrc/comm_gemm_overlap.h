#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <transformer_engine/userbuffers.h>
#include "gemm.h"
#include <torch/cuda.h>


#define BFLOAT16_BYTES 2

#define CHECK_CUDA(call) do { \
  cudaError_t status_ = call; \
  if( status_ != cudaSuccess ) { \
    fprintf(stderr, "CUDA Error at line %d: %s\n", __LINE__, cudaGetErrorString(status_)); \
    exit(1); \
  } \
} while(0)


namespace ubuf {


enum COMM_TYPE{
  RS = 0,
  AG = 1
};

enum UBOverlapAlgo{
  BULK_OVERLAP_AG = 0,
  BULK_OVERLAP_RS = 1
};


struct UbufCommOverlap : torch::CustomClassHolder {
    int _rank;
    int _tp_size;
    int _num_splits;
    int _math_sms;
    int _sm_all, _sm_margin;
    communicator* _ub_comm;
    at::cuda::CUDAStream _stream_comm = at::cuda::getStreamFromPool(true);
    std::vector<at::cuda::CUDAStream> _stream_compute;
    cudaStream_t* stream_math;
    torch::Tensor _ubuf;
    int _handle;
    cudaEvent_t _start_compute, _stop_compute, _start_d2dcopy, _start_comm, _stop_comm;
    std::vector<torch::Tensor> _lt_workspaces;
    torch::Tensor output_tensor;

    // Initialize userbuf.
    UbufCommOverlap(torch::Tensor& sample,
                    int rank,
                    int pp_size,
                    int tp_size,
                    int num_comm_sm=16,
                    int comm_cga_size=2,
                    int num_splits=1,
                    int use_rr_kernel=0,
                    bool set_sm_margin=false)
    {
        // Initialize userbuf communicator
        create_communicator_grouped2(&_ub_comm, 1, pp_size, tp_size, 1);
        _ub_comm->sms = num_comm_sm;
        _ub_comm->push = 1;
        _ub_comm->use_ce = 0;
        _ub_comm->cga_size = comm_cga_size;
        _ub_comm->use_rr_kernel = use_rr_kernel;

        // Allocate and register extra userbuffers
        _ubuf = sample;
        void* ubuf_ptr = static_cast<void*>(_ubuf.data_ptr());
        _handle = register_user_buffer_collective(
            (void**)&ubuf_ptr, _ubuf.numel() * _ubuf.element_size(), _ub_comm);

        for (int i = 0; i < num_splits; i++) {
            _stream_compute.push_back(at::cuda::getStreamFromPool());
            _lt_workspaces.push_back(torch::empty({1 << 25}, at::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA)));
        }

        _num_splits = num_splits;
        _tp_size = tp_size;
        _rank = rank;

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        _sm_all = prop.multiProcessorCount;
        _math_sms = (set_sm_margin) ? (_sm_all - num_comm_sm) : _sm_all;

        output_tensor = torch::Tensor();
        // CUDA event creation
        cudaEventCreateWithFlags(&_start_compute, 0);
        cudaEventCreateWithFlags(&_stop_compute, 0);
        cudaEventCreateWithFlags(&_start_d2dcopy, 0);
        cudaEventCreateWithFlags(&_start_comm, 0);
        cudaEventCreateWithFlags(&_stop_comm, 0);
    }


    /*
    ** Reduce Scatter
    ** This function assumes the communication input is pre-copied to _ubuf
    */
    torch::Tensor rs()
    {
        at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
        CHECK_CUDA(cudaEventRecord(_start_comm, (cudaStream_t) stream_main));
        CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) _stream_comm, _start_comm, 0));

        torch::Tensor output = torch::empty(
            {_ubuf.size(0) / _tp_size, _ubuf.size(1)},
            _ubuf.options()
        );

        // Communication
        reducescatter2_userbuff(
            output.data_ptr(),
            _handle,
            0,
            _ubuf.numel(),
            _ub_comm,
            (cudaStream_t) _stream_comm
        );

        CHECK_CUDA(cudaEventRecord(_stop_comm, (cudaStream_t) _stream_comm));
        CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) stream_main, _stop_comm, 0));

        return output;
    } // comm

    /*
    ** All Gather
    ** This function assumes the communication input is pre-copied to _ubuf
    */
    torch::Tensor ag()
    {
        at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
        CHECK_CUDA(cudaEventRecord(_start_comm, (cudaStream_t) stream_main));
        CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) _stream_comm, _start_comm, 0));

        // Communication
        allgather2_userbuff_inplace(
            _handle,
            0,
            (_ubuf.numel() / 2) * _ubuf.element_size(), // UBUF uses 2Byte element size
            _ub_comm,
            (cudaStream_t) _stream_comm
        );

        CHECK_CUDA(cudaEventRecord(_stop_comm, (cudaStream_t) _stream_comm));
        CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) stream_main, _stop_comm, 0));

        // Generate output tensor from userbuf data pointer
        torch::Tensor output = torch::from_blob(
            _ubuf.data_ptr(),
            {_ubuf.size(0), _ubuf.size(1)},
            _ubuf.options()
        );

        return output;
    } // comm


    /*
    ** Bulk GEMM + COMM
    ** This function assumes the communication input is pre-copied to _ubuf
    */
    std::vector<at::Tensor> bulk_overlap(at::Tensor A,
                      at::Tensor A_scale_inverse,
                      int64_t A_fp8_tensor,
                      transformer_engine::DType A_type,
                      bool transa,
                      at::Tensor B,
                      at::Tensor B_scale_inverse,
                      int64_t B_fp8_tensor,
                      transformer_engine::DType B_type,
                      bool transb,
                      at::Tensor D,
                      at::Tensor D_scale,
                      transformer_engine::DType D_type,
                      at::Tensor D_amax,
                      at::Tensor bias,
                      transformer_engine::DType bias_type,
                      at::Tensor pre_gelu_out,
                      bool grad,
                      at::Tensor workspace,
                      size_t workspaceSize,
                      bool accumulate,
                      bool use_split_accumulator,
                      int comm_type)
    {
        // Get the current userbuf offset
        char* ubuf_wt_ptr = reinterpret_cast<char*>(_ubuf.data_ptr());
        COMM_TYPE _comm_type = static_cast<COMM_TYPE>(comm_type);
        if (_comm_type == COMM_TYPE::RS) {
            ubuf_wt_ptr += _ubuf.numel() / _tp_size * _rank * _ubuf.element_size();
        }

        // Catch up the default torch stream
        at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
        CHECK_CUDA(cudaEventRecord(_start_comm, (cudaStream_t) stream_main));
        CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) _stream_comm, _start_comm, 0));

        // Communication: AG and RS
        if (_comm_type == COMM_TYPE::AG) {
            allgather2_userbuff_inplace(
                _handle,
                0,
                (_ubuf.numel() / 2) * _ubuf.element_size(), // UBUF uses 2Byte element size
                _ub_comm,
                (cudaStream_t) _stream_comm
            );
        } else if (_comm_type == COMM_TYPE::RS) {
            reducescatter2_userbuff_inplace(
                _handle,
                0,
                _ubuf.numel(),
                _ub_comm,
                (cudaStream_t) _stream_comm
            );
        } else {
            NVTE_ERROR("Not supported communication type.");
        }

        // GEMM
        #if 0
        bool transa, transb;
        if (fp8) {
            transa = true;
            transb = false;
        } else {
            std::tie(transa, transb) = get_gemm_input_layout(
                static_cast<GEMM_INPUT_LAYOUT>(gemm_input_layout));
        }

        const int m = transa ? input_a.size(0) : input_a.size(1);
        const int k = transa ? input_a.size(1) : input_a.size(0);
        const int n = transb ? input_b.size(1) : input_b.size(0);
        auto output_dtype = (gemm_input_layout == 2) ? torch::kFloat32 : torch::kBFloat16;
        torch::Tensor output = torch::empty(
            {n, m}, at::TensorOptions().dtype(output_dtype).device(torch::kCUDA));
        torch::Tensor psum = (gemm_input_layout == 2) ?
            torch::empty({n, m}, at::TensorOptions().dtype(output_dtype).device(torch::kCUDA)) : torch::empty({});

        matmul_cuda(
            input_a,
            input_b,
            output,
            psum,
            m,
            n,
            k,
            transa,
            transb,
            (cudaStream_t) stream_main,
            (void*) _lt_workspaces[0].data_ptr(),
            _math_sms,
            (gemm_input_layout == 0) ? true : false,
            fp8,
            (gemm_input_layout == 2) ? true : false
        );
        #endif
        if (A_scale_inverse.numel())
            A_scale_inverse = A_scale_inverse[A_fp8_tensor];

        if (B_scale_inverse.numel())
            B_scale_inverse = B_scale_inverse[B_fp8_tensor];

        te_gemm(A,
            A_scale_inverse,
            A_type,
            transa,
            B,
            B_scale_inverse,
            B_type,
            transb,
            D,
            D_scale,
            D_type,
            D_amax,
            bias,
            bias_type,
            pre_gelu_out,
            grad,
            workspace,
            workspaceSize,
            accumulate,
            use_split_accumulator,
            _math_sms);

        CHECK_CUDA(cudaEventRecord(_stop_comm, (cudaStream_t) _stream_comm));
        CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) stream_main, _stop_comm, 0));

        // Generate output tensor from userbuf data pointer
        int output_c_dim0 = (_comm_type == COMM_TYPE::AG) ? _ubuf.size(0) : _ubuf.size(0) / _tp_size;
        int output_c_dim1 = _ubuf.size(1);
        output_tensor = torch::from_blob(
            ubuf_wt_ptr,
            {output_c_dim0, output_c_dim1},
            _ubuf.options()
        );

        return {D, output_tensor};
    } // bulk_overlap


    /*
    ** Split FPROP GEMM + ReduceScatter
    */
    torch::Tensor split_overlap_rs(torch::Tensor input_a, torch::Tensor input_b, bool fp8, bool gemm_overlap)
    {
        // FPROP only
        bool transa = true;
        bool transb = false;

        // Get GEMM dimensions
        int m = input_a.size(0);
        int k = input_a.size(1);
        int n = input_b.size(0);
        int n_chunk = n / _num_splits;
        int input_b_chunk_size = n_chunk * k;
        int output_chunk_size = n_chunk * m;

        // Get input and output data pointers
        char* input_b_chunk_ptr = reinterpret_cast<char*>(input_b.data_ptr());
        char* output_chunk_ptr = reinterpret_cast<char*>(_ubuf.data_ptr());
        torch::Tensor rs_output = torch::empty({n / _tp_size, m}, _ubuf.options());
        char* rs_output_ptr = reinterpret_cast<char*>(rs_output.data_ptr());
        int ubuf_offset = 0;
        torch::Tensor dummy = torch::empty({});

        // Catch up the default torch stream
        at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
        CHECK_CUDA(cudaEventRecord(_start_compute, stream_main));
        for (int i = 0; i < _num_splits; i++) {
            CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) _stream_compute[i], _start_compute, 0));
        }

        if (gemm_overlap) {
            // First GEMM
            // Launch the first GEMM out-of-the-loop to schedule the second GEMM before the first communication kernel
            torch::Tensor input_b_chunk = torch::from_blob(
                input_b_chunk_ptr,
                {n_chunk, k},
                input_b.options()
            );
            torch::Tensor output_chunk = torch::from_blob(
                output_chunk_ptr,
                {n_chunk, m},
                at::TensorOptions().dtype(at::kBFloat16).device(torch::kCUDA)
            );
            matmul_cuda(
                input_a,
                input_b_chunk,
                output_chunk,
                dummy,
                m,  
                n_chunk,
                k,  
                transa,
                transb,
                (cudaStream_t) _stream_compute[0],
                (void*) _lt_workspaces[0].data_ptr(),
                _math_sms,
                true,   // fast accum
                fp8,
                false   // no wgrad accum
            );

            for (int i = 1; i < _num_splits; i++) {
                // Update input and output data pointers
                input_b_chunk_ptr += input_b_chunk_size * input_b.element_size();
                output_chunk_ptr += output_chunk_size * _ubuf.element_size();

                // GEMM
                torch::Tensor input_b_chunk = torch::from_blob(
                    input_b_chunk_ptr,
                    {n_chunk, k},
                    input_b.options()
                );
                torch::Tensor output_chunk = torch::from_blob(
                    output_chunk_ptr,
                    {n_chunk, m},
                    at::TensorOptions().dtype(at::kBFloat16).device(torch::kCUDA)
                );
                matmul_cuda(
                    input_a,
                    input_b_chunk,
                    output_chunk,
                    dummy,
                    m,  
                    n_chunk,
                    k,  
                    transa,
                    transb,
                    (cudaStream_t) _stream_compute[i],
                    (void*) _lt_workspaces[i].data_ptr(),
                    _math_sms,
                    true,   // fast accum
                    fp8,
                    false   // no wgrad accum
                );

                // Have communication wait handle hear to perfect overlap of GEMM and COMM
                CHECK_CUDA(cudaEventRecord(_start_comm, (cudaStream_t) _stream_compute[i-1]));
                CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) _stream_comm, _start_comm, 0));

                // Communication
                reducescatter2_userbuff(
                    rs_output_ptr,
                    _handle,
                    ubuf_offset,
                    output_chunk_size,
                    _ub_comm,
                    (cudaStream_t) _stream_comm
                );

                ubuf_offset += output_chunk_size;
                rs_output_ptr += (output_chunk_size / _tp_size) * _ubuf.element_size();
            }

            CHECK_CUDA(cudaEventRecord(_start_comm, (cudaStream_t) _stream_compute[_num_splits-1]));
            CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) _stream_comm, _start_comm, 0));

            // Last communication chunk
            reducescatter2_userbuff(
                rs_output_ptr,
                _handle,
                ubuf_offset,
                output_chunk_size,
                _ub_comm,
                _stream_comm
            );
        } else {
            for (int i = 0; i < _num_splits; i++) {
                torch::Tensor input_b_chunk = torch::from_blob(
                    input_b_chunk_ptr,
                    {n_chunk, k},
                    input_b.options()
                );
                torch::Tensor output_chunk = torch::from_blob(
                    output_chunk_ptr,
                    {n_chunk, m},
                    at::TensorOptions().dtype(at::kBFloat16).device(torch::kCUDA)
                );

                // GEMM chunk
                matmul_cuda(
                    input_a,
                    input_b_chunk,
                    output_chunk,
                    dummy,
                    m,  
                    n_chunk,
                    k,  
                    transa,
                    transb,
                    (cudaStream_t) _stream_compute[i],
                    (void*) _lt_workspaces[i].data_ptr(),
                    _math_sms,
                    true,   // fast accum
                    fp8,
                    false   // no wgrad accum
                );

                CHECK_CUDA(cudaEventRecord(_start_comm, (cudaStream_t) _stream_compute[i]));
                CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) _stream_comm, _start_comm, 0));

                // Communication chunk
                reducescatter2_userbuff(
                    rs_output_ptr,
                    _handle,
                    ubuf_offset,
                    output_chunk_size,
                    _ub_comm,
                    _stream_comm  
                );

                // Update input and output data pointers
                ubuf_offset += output_chunk_size;
                rs_output_ptr += (output_chunk_size / _tp_size) * _ubuf.element_size();
                input_b_chunk_ptr += input_b_chunk_size * input_b.element_size();
                output_chunk_ptr += output_chunk_size * _ubuf.element_size();
            }
        }

        CHECK_CUDA(cudaEventRecord(_stop_compute, (cudaStream_t) _stream_compute[_num_splits-1]));
        CHECK_CUDA(cudaEventRecord(_stop_comm, (cudaStream_t) _stream_comm));
        CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) stream_main, _stop_compute, 0));
        CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) stream_main, _stop_comm, 0));

        return rs_output;
    } // split_overlap_rs


    /*
    ** Split AllGather + GEMM
    */
    std::vector<torch::Tensor> split_overlap_ag(
        torch::Tensor input_a,
        int gemm_input_layout,
        bool get_ag_output,
        bool fp8)
    {
        bool transa, transb;
        if (fp8) {
            transa = true;
            transb = false;
        } else {
            std::tie(transa, transb) = get_gemm_input_layout(
                static_cast<GEMM_INPUT_LAYOUT>(gemm_input_layout));
        }

        // Get GEMM dimensions between TN and NN input layouts
        const int m = (transa) ? input_a.size(0) : input_a.size(1);
        const int k = (transa) ? input_a.size(1) : input_a.size(0);
        const int n = _ubuf.size(0);
        const int n_chunk = n / (_tp_size * _num_splits);
        // Communication bytes per chunk. UBUF considers element size as 2Bytes
        const int input_b_chunk_elements = (_ubuf.numel() / _tp_size / 2) * _ubuf.element_size();
        // GEMM input and output size that includes the whole strided data range
        const int n_valid_size = (n / _tp_size * (_tp_size - 1)) + n_chunk;
        const long long int stride_input_b = n * k / _tp_size;
        const long long int stride_output = n * m / _tp_size;
        // Create GEMM output buffer and a pointer to its current chunk
        torch::Tensor output = torch::empty({n, m}, at::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        char* output_chunk_ptr = reinterpret_cast<char*>(output.data_ptr());
        char* input_b_chunk_ptr = reinterpret_cast<char*>(_ubuf.data_ptr());

        // Catch up the default torch stream
        at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
        CHECK_CUDA(cudaEventRecord(_start_comm, (cudaStream_t) stream_main));
        CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) _stream_comm, _start_comm, 0));

        torch::Tensor dummy = torch::empty({});
        for (int i = 0; i < _num_splits; i++) {
            // Communication
            allgather2_userbuff_inplace_sliced(
                _handle,
                0,
                input_b_chunk_elements,
                _ub_comm,
                i,
                _num_splits,
                _stream_comm
            );

            CHECK_CUDA(cudaEventRecord(_start_compute, (cudaStream_t) _stream_comm));
            CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) _stream_compute[i], _start_compute, 0));

            // GEMM
            torch::Tensor input_b_chunk = torch::from_blob(
                input_b_chunk_ptr,
                {n_valid_size, k},
                _ubuf.options()
            );
            torch::Tensor output_chunk = torch::from_blob(
                output_chunk_ptr,
                {n_valid_size, m},
                at::TensorOptions().dtype(at::kBFloat16).device(torch::kCUDA)
            );
            strided_gemm_cuda(
                input_a,
                input_b_chunk,
                output_chunk,
                m,
                n_chunk,
                k,
                stride_input_b,
                stride_output,
                _tp_size,   // batch
                transa,
                transb,
                (cudaStream_t) _stream_compute[i],
                (void*) _lt_workspaces[i].data_ptr(),
                fp8
            );
            // Update the data pointers
            input_b_chunk_ptr += n_chunk * k * _ubuf.element_size();
            output_chunk_ptr += n_chunk * m * output.element_size();
        }

        CHECK_CUDA(cudaEventRecord(_stop_compute, (cudaStream_t) _stream_compute[_num_splits-1]));
        CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) stream_main, _stop_compute, 0));

        // Return the AG output to provide WGRAD GEMM input
        torch::Tensor ag_output = (get_ag_output) ? _ubuf : torch::empty({});

        return {output, ag_output};
    }


    /*
    ** Helper function to copy input to _ubuf
    */
    void copy_input_to_ubuf(torch::Tensor input, int comm_type)
    {
        char* ubuf_ptr = reinterpret_cast<char*>(_ubuf.data_ptr());
        COMM_TYPE _comm_type = static_cast<COMM_TYPE>(comm_type);
        if (_comm_type == COMM_TYPE::AG) {
            if ((input.numel() * _tp_size) != _ubuf.numel() || input.element_size() != _ubuf.element_size()) {
                NVTE_ERROR("input and ubuf size do not match!");
            }
            ubuf_ptr += _ubuf.numel() / _tp_size * _rank * _ubuf.element_size();
        } else {
            if (input.numel() != _ubuf.numel() || input.element_size() != _ubuf.element_size()) {
                NVTE_ERROR("input and ubuf size do not match!");
            }
        }

        at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
        CHECK_CUDA(cudaEventRecord(_start_d2dcopy, (cudaStream_t) stream_main));
        CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) _stream_comm, _start_d2dcopy, 0));
        CHECK_CUDA(cudaMemcpyAsync(
            ubuf_ptr,
            input.data_ptr(),
            input.numel() * input.element_size(),
            cudaMemcpyDeviceToDevice,
            (cudaStream_t) _stream_comm)
        );
    }
    torch::Tensor get_output()
    {
        if (output_tensor.numel() == 0)
            NVTE_ERROR("Empty output");
        return output_tensor;
    }
    torch::Tensor & get_ubuf_output(int comm_type)
    {
        char* ubuf_wt_ptr = reinterpret_cast<char*>(_ubuf.data_ptr());
        COMM_TYPE _comm_type = static_cast<COMM_TYPE>(comm_type);
        if (_comm_type != COMM_TYPE::AG && _comm_type != COMM_TYPE::RS)
            NVTE_ERROR("Invalid comm_type");
        if (_comm_type == COMM_TYPE::RS) 
            ubuf_wt_ptr += _ubuf.numel() / _tp_size * _rank * _ubuf.element_size();
        int output_c_dim0 = (_comm_type == COMM_TYPE::AG) ? _ubuf.size(0) : _ubuf.size(0) / _tp_size;
        int output_c_dim1 = _ubuf.size(1);
        output_tensor = torch::from_blob(
            ubuf_wt_ptr,
            {output_c_dim0, output_c_dim1},
            _ubuf.options()
        );
        return output_tensor;
    }

}; // UbufCommOverlap



struct UbufP2PCommOverlap : torch::CustomClassHolder {
    int _rank;
    int _tp_size;
    int _reg;
    int _next_rank, _prev_rank;
    int _math_sms;
    communicator* _ub_comm;
    at::cuda::CUDAStream _stream_comm = at::cuda::getStreamFromPool(true);
    at::cuda::CUDAStream _stream_accum = at::cuda::getStreamFromPool();
    std::vector<at::cuda::CUDAStream> _stream_compute;
    torch::Tensor _ubuf;
    std::vector<torch::Tensor> _ubufs;
    cudaEvent_t _start_compute, _stop_compute, _start_comm, _stop_comm, _start_accum, _stop_accum;
    std::vector<torch::Tensor> _lt_workspaces;
    void* _ubuf_ptr;

    UbufP2PCommOverlap(
        torch::Tensor sample,
        int rank,
        int pp_size,
        int tp_size,
        int comm_sm,
        bool set_sm_margin)
    {
        // Initialize userbuf communicator
        create_communicator_grouped2(&_ub_comm, 1, pp_size, tp_size, 1);
        _ub_comm->use_ce = 1;
        _ub_comm->push = 1;
        _ub_comm->sms = comm_sm;
        _ub_comm->cga_size = 1;

        // Create workspace tensor with userbuffer
        int ubuf_bytes = sample.numel() * sample.element_size();
        int ubuf_chunk_bytes = ubuf_bytes / tp_size;
        cudaMalloc((void**)&_ubuf_ptr, ubuf_bytes);
        _reg = register_user_buffer_collective((void**)&_ubuf_ptr, ubuf_bytes, _ub_comm);
        _ubuf = torch::from_blob(_ubuf_ptr, {sample.size(0), sample.size(1)}, sample.options());

        // Create tensor chunks for easy management
        char* ubuf_byte_ptr = reinterpret_cast<char*>(_ubuf.data_ptr());
        for (int i = 0; i < tp_size; i++) {
            torch::Tensor ubuf_chunk = torch::from_blob(
                ubuf_byte_ptr, {sample.size(0) / tp_size, sample.size(1)}, sample.options()
            );
            _ubufs.push_back(ubuf_chunk);
            ubuf_byte_ptr += ubuf_chunk_bytes;
        }

        for (int i = 0; i < tp_size; i++) {
            _stream_compute.push_back(at::cuda::getStreamFromPool());
            _lt_workspaces.push_back(torch::empty({1 << 25}, at::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA)));
        }

        _tp_size = tp_size;
        _rank = rank;

        _next_rank = (tp_size + rank + 1) % tp_size;
        _prev_rank = (tp_size + rank + -1) % tp_size;

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        _math_sms = (set_sm_margin) ? prop.multiProcessorCount - comm_sm : prop.multiProcessorCount;

        // CUDA event creation
        cudaEventCreateWithFlags(&_start_compute, 0);
        cudaEventCreateWithFlags(&_stop_compute, 0);
        cudaEventCreateWithFlags(&_start_comm, 0);
        cudaEventCreateWithFlags(&_stop_comm, 0);
        cudaEventCreateWithFlags(&_start_accum, 0);
        cudaEventCreateWithFlags(&_stop_accum, 0);
    }

    ~UbufP2PCommOverlap()
    {
        cudaFree(_ubuf_ptr);
    }

    /*
    ** Split AllGather + GEMM using P2P communication
    ** This function assumes the input_b is pre-copied to _ubufs[rank_id]. This is needed to have AG outputs 
    ** in each rank to be in the contiguous memory space after all ring exchange phases.
    */
    torch::Tensor split_overlap_ag(
        torch::Tensor input_a,
        int gemm_input_layout,
        bool get_ag_output,
        bool fp8)
    {
        bool transa, transb;
        if (fp8) {
            transa = true;
            transb = false;
        } else {
            std::tie(transa, transb) = get_gemm_input_layout(
                static_cast<GEMM_INPUT_LAYOUT>(gemm_input_layout));
        }

        // Get GEMM dimensions between TN and NN input layouts
        const int m = (transa) ? input_a.size(0) : input_a.size(1);
        const int k = (transa) ? input_a.size(1) : input_a.size(0);
        const int n_chunk = _ubufs[0].size(0);
        // Get communication and GEMM output chunk sizes
        const int comm_bytes = _ubufs[0].numel() * _ubufs[0].element_size();
        int output_chunk_bytes = (n_chunk * m) * BFLOAT16_BYTES;
        // Create GEMM output buffer and a pointer to its current chunk
        torch::Tensor output = torch::empty({n_chunk * _tp_size, m}, at::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        char* output_ptr = reinterpret_cast<char*>(output.data_ptr());
        int cur_ouput_chunk_id = _rank;

        // Catch up the default torch stream
        at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
        CHECK_CUDA(cudaEventRecord(_start_compute, (cudaStream_t) stream_main));
        CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) _stream_compute[0], _start_compute, 0));
        CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) _stream_comm, _start_compute, 0));

        torch::Tensor dummy = torch::empty({});
        for (int i = 0; i < _tp_size; i++) {
            // Set the userbuffer id. Buffer under send is the input for the current GEMM chunk
            // The initial input chunk is stored _ubuf[rank]. This is to have the AG output in all ranks to 
            // be contiguous after the ring exchanges
            int send_chunk_id = (_tp_size + _rank - i) % _tp_size;
            int recv_chunk_id = (_tp_size + _rank - i - 1) % _tp_size;
            int send_offset = comm_bytes * send_chunk_id;
            int recv_offset = comm_bytes * recv_chunk_id;

            // GEMM
            torch::Tensor output_chunk = torch::from_blob(
                output_ptr + (cur_ouput_chunk_id * output_chunk_bytes),
                {n_chunk, m},
                output.options()
            );
            matmul_cuda(
                input_a,
                _ubufs[send_chunk_id],
                output_chunk,
                dummy,
                m,
                n_chunk,
                k,
                transa,
                transb,
                _stream_compute[i],
                (void*) _lt_workspaces[i].data_ptr(),
                _math_sms,
                (gemm_input_layout == 0) ? true : false,
                fp8,
                false // no grad accum
            );

            if (i < _tp_size - 1) {
                // P2P communication
                userbuffers_send(_reg, send_offset, _reg, send_offset, comm_bytes, _ub_comm, _next_rank, (cudaStream_t) _stream_comm);
                userbuffers_recv(_reg, recv_offset, _reg, recv_offset, comm_bytes, _ub_comm, _prev_rank, (cudaStream_t) _stream_comm);
                CHECK_CUDA(cudaEventRecord(_stop_comm, (cudaStream_t) _stream_comm));
                CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) _stream_compute[i+1], _stop_comm, 0));
            }
            cur_ouput_chunk_id = (_tp_size + cur_ouput_chunk_id - 1) % _tp_size;
        }

        CHECK_CUDA(cudaEventRecord(_stop_compute, (cudaStream_t) _stream_compute[_tp_size-1]));
        CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) stream_main, _stop_compute, 0));

        return output;
    } // split_overlap_ag


    /*
    ** Split GEMM + ReduceScatter using P2P communication
    ** This function assumes the input_b is pre-copied to _ubuf
    */
    torch::Tensor split_overlap_rs(
        torch::Tensor input_a,
        torch::Tensor input_b,
        bool fp8)
    {
        // FPROP only
        bool transa = true;
        bool transb = false;

        // Get the dimensions of TN GEMM
        int m = input_a.size(0);
        int k = input_a.size(1);
        int n = input_b.size(0);
        int n_chunk = input_b.size(0) / _tp_size;
        const int comm_bytes = _ubufs[0].numel() * _ubufs[0].element_size();
        const int input_b_chunk_bytes = n_chunk * k * input_b.element_size();
        char* input_b_ptr = reinterpret_cast<char*>(input_b.data_ptr());

        // Init send/recv chunk id
        int send_buf_id = (_tp_size + _rank - 1) % _tp_size;
        int recv_buf_id = (_tp_size + _rank - 2) % _tp_size;
        int input_b_chunk_id = send_buf_id;

        // Triple buffer GEMM chunk outputs to overlap GEMMs with multi streams
        std::vector<torch::Tensor> output_bufs;
        output_bufs.push_back(torch::empty({n_chunk, m}, at::TensorOptions().dtype(at::kBFloat16).device(torch::kCUDA)));
        output_bufs.push_back(torch::empty({n_chunk, m}, at::TensorOptions().dtype(at::kBFloat16).device(torch::kCUDA)));
        output_bufs.push_back(torch::empty({n_chunk, m}, at::TensorOptions().dtype(at::kBFloat16).device(torch::kCUDA)));

        // Catch up the default torch stream
        at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
        CHECK_CUDA(cudaEventRecord(_start_compute, stream_main));
        for (int i = 0; i < _tp_size; i++) {
            CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) _stream_compute[i], _start_compute, 0));
        }

        torch::Tensor dummy = torch::empty({});
        for (int i = 0; i < _tp_size; i++) {
            char* input_b_chunk_ptr = input_b_ptr + input_b_chunk_id * input_b_chunk_bytes;
            // GEMM
            torch::Tensor input_b_chunk = torch::from_blob(
                input_b_chunk_ptr,
                {n_chunk, k},
                input_b.options()
            );
            matmul_cuda(
                input_a,
                input_b_chunk,
                (i == 0) ? _ubufs[send_buf_id] : output_bufs[i % 3],
                dummy,
                m,
                n_chunk,
                k,
                transa,
                transb,
                (cudaStream_t) _stream_compute[i],
                (void*) _lt_workspaces[i].data_ptr(),
                _math_sms,
                true,   // fast accum
                fp8,
                false   // no grad accum
            );

            if (i > 0) {
                int send_offset = comm_bytes * send_buf_id;
                int recv_offset = comm_bytes * recv_buf_id;

                // P2P communication
                CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) _stream_comm, _start_comm, 0));

                userbuffers_send(_reg, send_offset, _reg, send_offset, comm_bytes, _ub_comm, _next_rank, (cudaStream_t) _stream_comm);
                userbuffers_recv(_reg, recv_offset, _reg, recv_offset, comm_bytes, _ub_comm, _prev_rank, (cudaStream_t) _stream_comm);

                CHECK_CUDA(cudaEventRecord(_stop_comm, (cudaStream_t) _stream_comm));
                CHECK_CUDA(cudaEventRecord(_stop_compute, (cudaStream_t) _stream_compute[i]));
                CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) _stream_accum, _stop_comm, 0));
                CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) _stream_accum, _stop_compute, 0));
                
                // Accumulate the current GEMM output to the partial sum
                {
                    c10::cuda::CUDAStreamGuard guard(_stream_accum);
                    torch::add(_ubufs[recv_buf_id], output_bufs[i % 3]);
                }
                CHECK_CUDA(cudaEventRecord(_start_comm, (cudaStream_t) _stream_accum));
                // Update send recv buffer id
                send_buf_id = (_tp_size + send_buf_id - 1) % _tp_size;
                recv_buf_id = (_tp_size + recv_buf_id - 1) % _tp_size;
            } else {
                CHECK_CUDA(cudaEventRecord(_start_comm, (cudaStream_t) _stream_compute[i]));
            }
            input_b_chunk_id = (_tp_size + input_b_chunk_id -1) % _tp_size;
        }
        CHECK_CUDA(cudaEventRecord(_stop_compute, (cudaStream_t) _stream_compute[_tp_size-1]));
        CHECK_CUDA(cudaEventRecord(_stop_accum, (cudaStream_t) _stream_accum));
        CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) stream_main, _stop_compute, 0));
        CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) stream_main, _stop_accum, 0));
        return _ubufs[_rank];
    }


    /*
    ** Get ubuf-registered tensor
    */
    torch::Tensor& get_ubuf_tensor()
    {
        return _ubuf;
    }

    /*
    ** Helper function to test ring exchange
    */
    void test_p2p_exchange() {
        const int comm_bytes = _ubufs[0].numel() * _ubufs[0].element_size();
        at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
        for (int i = 0; i < _tp_size; i++) {
            int send_chunk_id = (_tp_size + _rank - i) % _tp_size;
            int recv_chunk_id = (_tp_size + _rank - i - 1) % _tp_size;
            int send_offset = comm_bytes * send_chunk_id;
            int recv_offset = comm_bytes * recv_chunk_id;

            userbuffers_send(_reg, send_offset, _reg, send_offset, comm_bytes, _ub_comm, _next_rank, (cudaStream_t) stream_main);
            userbuffers_recv(_reg, recv_offset, _reg, recv_offset, comm_bytes, _ub_comm, _prev_rank, (cudaStream_t) stream_main);
        }
    }


    /*
    ** Helper function to test inter-rank send/recv
    */
    void test_send_recv(int send_rank, int recv_rank, int send_chunk_id, int recv_chunk_id) {
        int comm_bytes = _ubufs[0].numel() * _ubufs[0].element_size();
        int send_offset = comm_bytes * send_chunk_id;
        int recv_offset = comm_bytes * recv_chunk_id;
        at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();

        if (_rank == send_rank) {
            userbuffers_send(_reg, send_offset, _reg, recv_offset, comm_bytes, _ub_comm, recv_rank, (cudaStream_t) stream_main);
        }
        else if (_rank == recv_rank) {
            userbuffers_recv(_reg, send_offset, _reg, recv_offset, comm_bytes, _ub_comm, send_rank, (cudaStream_t) stream_main);
        }
    }

    /*
    ** Copy input to _ubufs[0]
    */
    void copy_input_to_ubuf(torch::Tensor input, bool chunk)
    {
        at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
        if (chunk) {
            // Copy input to the target ubuf chunk by rank offset
            if (input.numel() != _ubufs[0].numel() || input.element_size() != _ubufs[0].element_size()) {
                NVTE_ERROR("input and ubuf size do not match!");
            }
            CHECK_CUDA(cudaMemcpyAsync(
                _ubufs[_rank].data_ptr(),
                input.data_ptr(),
                input.numel() * input.element_size(),
                cudaMemcpyDeviceToDevice,
                (cudaStream_t) stream_main)
            );
        } else {
            if (input.numel() != _ubuf.numel() || input.element_size() != _ubuf.element_size()) {
                NVTE_ERROR("input and ubuf size do not match!");
            }
            CHECK_CUDA(cudaMemcpyAsync(
                _ubuf.data_ptr(),
                input.data_ptr(),
                input.numel() * input.element_size(),
                cudaMemcpyDeviceToDevice,
                (cudaStream_t) stream_main)
            );
        }
    }

}; // UbufP2PCommOverlap


/*
** Helper function to test split-pipelining using strided GEMM
*/
torch::Tensor gemm_strided(
    torch::Tensor input_a,
    torch::Tensor input_b,
    int gemm_input_layout,
    int tp_size,
    int num_splits,
    bool fp8)
{
    bool transa, transb;
    if (fp8) {
        transa = true;
        transb = false;
    } else {
        std::tie(transa, transb) = get_gemm_input_layout(
            static_cast<GEMM_INPUT_LAYOUT>(gemm_input_layout));
    }

    const int m = transa ? input_a.size(0) : input_a.size(1);
    const int k = transa ? input_a.size(1) : input_a.size(0);
    const int n = transb ? input_b.size(1) : input_b.size(0);
    const int n_chunk = n / (tp_size * num_splits);
    const int n_valid_size = (n / tp_size * (tp_size - 1)) + n_chunk;
    const long long int stride_input_b = n * k / tp_size;
    const long long int stride_output = n * m / tp_size;

    torch::Tensor output = torch::empty(
        {n, m}, at::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
    char* input_b_chunk_ptr = reinterpret_cast<char*>(input_b.data_ptr());
    char* output_chunk_ptr = reinterpret_cast<char*>(output.data_ptr());
    
    torch::Tensor lt_workspace = torch::empty({1 << 25}, at::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
    at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
    for (int i = 0; i < num_splits; i++) {
        torch::Tensor input_b_chunk = torch::from_blob(
            input_b_chunk_ptr,
            {n_valid_size, k},
            input_b.options()
        );
        torch::Tensor output_chunk = torch::from_blob(
            output_chunk_ptr,
            {n_valid_size, m},
            at::TensorOptions().dtype(at::kBFloat16).device(torch::kCUDA)
        );
        strided_gemm_cuda(
            input_a,
            input_b_chunk,
            output_chunk,
            m,
            n_chunk,
            k,
            stride_input_b,
            stride_output,
            tp_size,   // batch
            transa,
            transb,
            (cudaStream_t) stream_main,
            (void*) lt_workspace.data_ptr(),
            fp8
        );
        input_b_chunk_ptr += n_chunk * k * input_b.element_size();
        output_chunk_ptr += n_chunk * m * output.element_size();
    }

    return output;
} // gemm


/*
** Helper function to run GEMM
*/
torch::Tensor gemm(
    torch::Tensor input_a,
    torch::Tensor input_b,
    int gemm_input_layout,
    bool fp8)
{
    bool transa, transb;
    if (fp8) {
        transa = true;
        transb = false;
    } else {
        std::tie(transa, transb) = get_gemm_input_layout(
            static_cast<GEMM_INPUT_LAYOUT>(gemm_input_layout));
    }

    const int m = transa ? input_a.size(0) : input_a.size(1);
    const int k = transa ? input_a.size(1) : input_a.size(0);
    const int n = transb ? input_b.size(1) : input_b.size(0);
    auto output_dtype = (gemm_input_layout == 2) ? torch::kFloat32 : torch::kBFloat16;
    torch::Tensor output = torch::empty(
        {n, m}, at::TensorOptions().dtype(output_dtype).device(torch::kCUDA));
    torch::Tensor psum = (gemm_input_layout == 2) ? torch::empty_like(output) : torch::empty({});
    torch::Tensor lt_workspace = torch::empty({1 << 25}, at::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
    at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
    matmul_cuda(
        input_a,
        input_b,
        output,
        psum,
        m,
        n,
        k,
        transa,
        transb,
        (cudaStream_t) stream_main,
        (void*) lt_workspace.data_ptr(),
        0,
        (gemm_input_layout == 0) ? true : false,
        fp8,
        (gemm_input_layout == 2) ? true : false
    );

    return output;
}


//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//    py::class_<UbufCommOverlap>(m, "UbufCommOverlap")
//        .def(py::init<torch::Tensor&, int, int, int, int, int, int, int, bool>())
//        .def("rs", &UbufCommOverlap::rs)
//        .def("ag", &UbufCommOverlap::ag)
//        .def("bulk_overlap", &UbufCommOverlap::bulk_overlap)
//        .def("split_overlap_rs", &UbufCommOverlap::split_overlap_rs)
//        .def("split_overlap_ag", &UbufCommOverlap::split_overlap_ag)
//        .def("copy_input_to_ubuf", &UbufCommOverlap::copy_input_to_ubuf);
//    py::class_<UbufP2PCommOverlap>(m, "UbufP2PCommOverlap")
//        .def(py::init<torch::Tensor, int, int, int, int, bool>())
//        .def("split_overlap_ag", &UbufP2PCommOverlap::split_overlap_ag)
//        .def("split_overlap_rs", &UbufP2PCommOverlap::split_overlap_rs)
//        .def("copy_input_to_ubuf", &UbufP2PCommOverlap::copy_input_to_ubuf)
//        .def("test_p2p_exchange", &UbufP2PCommOverlap::test_p2p_exchange)
//        .def("test_send_recv", &UbufP2PCommOverlap::test_send_recv)
//        .def("get_ubuf_tensor", &UbufP2PCommOverlap::get_ubuf_tensor);
//    m.def("gemm_strided", &gemm_strided);
//    m.def("gemm", &gemm);
//}

} // end of namespace ubuf_comm_gemm_overlap
