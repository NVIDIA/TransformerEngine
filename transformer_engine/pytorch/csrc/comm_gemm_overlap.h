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


struct UbufCommOverlap : torch::CustomClassHolder {
    int _rank;
    int _tp_size;
    int _num_splits;
    int _ubuf_offset_size;
    int _num_buffers;
    int _cur_ubuf_id = 0;
    int _math_sms;
    int _sm_all, _sm_margin;
    communicator* _ub_comm;
    at::cuda::CUDAStream _stream_comm = at::cuda::getStreamFromPool(true);
    std::vector<at::cuda::CUDAStream> _stream_compute;
    cudaStream_t* stream_math;
    std::vector<torch::Tensor> _ubufs;
    std::vector<int> _handles;
    cudaEvent_t _start_compute, _stop_compute, _start_comm, _stop_comm;
    std::vector<torch::Tensor> _lt_workspaces;

    // Initialize userbuf.
    UbufCommOverlap(torch::Tensor& ubuf0,
                    int rank,
                    int pp_size,
                    int tp_size,
                    int num_comm_sm=16,
                    int comm_cga_size=2,
                    int num_splits=1,
                    int num_buffers=2,
                    bool set_sm_margin=false)
    {
        // Initialize userbuf communicator
        create_communicator_grouped2(&_ub_comm, 1, pp_size, tp_size, 1);
        _ub_comm->sms = num_comm_sm;
        _ub_comm->push = 1;
        _ub_comm->use_ce = 0;
        _ub_comm->cga_size = comm_cga_size;
        _ub_comm->use_rr_kernel = 0;

        // Allocate and register extra userbuffers
        _num_buffers = num_buffers;
        _ubuf_offset_size = ubuf0.numel() / num_buffers;
        _ubufs.push_back(ubuf0);
        void* ubuf0_ptr = static_cast<void*>(ubuf0.data_ptr());
        _handles.push_back(register_user_buffer_collective(
            (void**)&ubuf0_ptr, ubuf0.numel() * ubuf0.element_size(), _ub_comm));
        for (int i = 1; i < num_buffers; i++) {
            torch::Tensor ubuf = torch::empty_like(ubuf0);
            _ubufs.push_back(ubuf);
            void* ubuf_ptr = static_cast<void*>(ubuf.data_ptr());
            _handles.push_back(register_user_buffer_collective(
                (void**)&ubuf_ptr, ubuf0.numel() * ubuf0.element_size(), _ub_comm));
        }

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
        _sm_margin = _sm_all - num_comm_sm;
        _math_sms = _sm_margin;

        // CUDA event creation
        cudaEventCreateWithFlags(&_start_compute, 0);
        cudaEventCreateWithFlags(&_stop_compute, 0);
        cudaEventCreateWithFlags(&_start_comm, 0);
        cudaEventCreateWithFlags(&_stop_comm, 0);
    }


    /*
    ** Reduce Scatter
    ** This function assumes the communication input is pre-copied to _ubufs[_cur_ubuf_id]
    */
    torch::Tensor rs()
    {
        // Get the current userbuf
        torch::Tensor* ubuf = &_ubufs[_cur_ubuf_id];

        at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
        CHECK_CUDA(cudaEventRecord(_start_comm, (cudaStream_t) stream_main));
        CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) _stream_comm, _start_comm, 0));

        torch::Tensor output = torch::empty(
            {ubuf->size(0) / _tp_size, ubuf->size(1)},
            ubuf->options()
        );

        // Communication
        reducescatter2_userbuff(
            output.data_ptr(),
            _handles[_cur_ubuf_id],
            0,
            ubuf->numel(),
            _ub_comm,
            (cudaStream_t) _stream_comm
        );

        CHECK_CUDA(cudaEventRecord(_stop_comm, (cudaStream_t) _stream_comm));
        CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) stream_main, _stop_comm, 0));

        _cur_ubuf_id = (_cur_ubuf_id + 1) % _num_buffers;

        return output;
    } // comm

    /*
    ** All Gather
    ** This function assumes the communication input is pre-copied to _ubufs[_cur_ubuf_id]
    */
    torch::Tensor ag()
    {
        // Get the current userbuf
        torch::Tensor* ubuf = &_ubufs[_cur_ubuf_id];

        at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
        CHECK_CUDA(cudaEventRecord(_start_comm, (cudaStream_t) stream_main));
        CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) _stream_comm, _start_comm, 0));

        // Communication
        allgather2_userbuff_inplace(
            _handles[_cur_ubuf_id],
            0,
            (ubuf->numel() / 2) * ubuf->element_size(), // UBUF uses 2Byte element size
            _ub_comm,
            (cudaStream_t) _stream_comm
        );

        CHECK_CUDA(cudaEventRecord(_stop_comm, (cudaStream_t) _stream_comm));
        CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) stream_main, _stop_comm, 0));

        // Generate output tensor from userbuf data pointer
        torch::Tensor output = torch::from_blob(
            ubuf->data_ptr(),
            {ubuf->size(0), ubuf->size(1)},
            ubuf->options()
        );
        _cur_ubuf_id = (_cur_ubuf_id + 1) % _num_buffers;

        return output;
    } // comm


    /*
    ** Bulk GEMM + COMM
    ** This function assumes the communication input is pre-copied to _ubufs[_cur_ubuf_id]
    */
    std::vector<torch::Tensor> bulk_overlap(
        torch::Tensor input_a,
        torch::Tensor input_b,
        int comm_type,
        int gemm_input_layout,
        bool fp8)
    {
        // Get the current userbuf offset
        torch::Tensor* ubuf = &_ubufs[_cur_ubuf_id];
        char* ubuf_wt_ptr = reinterpret_cast<char*>(ubuf->data_ptr());
        COMM_TYPE _comm_type = static_cast<COMM_TYPE>(comm_type);
        if (_comm_type == COMM_TYPE::RS) {
            ubuf_wt_ptr += ubuf->numel() / _tp_size * _rank * ubuf->element_size();
        }

        // Catch up the default torch stream
        at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
        CHECK_CUDA(cudaEventRecord(_start_comm, (cudaStream_t) stream_main));
        CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) _stream_comm, _start_comm, 0));

        // Communication: AG and RS
        if (_comm_type == COMM_TYPE::AG) {
            allgather2_userbuff_inplace(
                _handles[_cur_ubuf_id],
                0,
                (ubuf->numel() / 2) * ubuf->element_size(), // UBUF uses 2Byte element size
                _ub_comm,
                (cudaStream_t) _stream_comm
            );
        } else if (_comm_type == COMM_TYPE::RS) {
            reducescatter2_userbuff_inplace(
                _handles[_cur_ubuf_id],
                0,
                ubuf->numel(),
                _ub_comm,
                (cudaStream_t) _stream_comm
            );
        } else {
            NVTE_ERROR("Not supported communication type.");
        }

        // GEMM
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

        CHECK_CUDA(cudaEventRecord(_stop_comm, (cudaStream_t) _stream_comm));
        CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) stream_main, _stop_comm, 0));

        // Generate output tensor from userbuf data pointer
        int output_c_dim0 = (_comm_type == COMM_TYPE::AG) ? ubuf->size(0) : ubuf->size(0) / _tp_size;
        int output_c_dim1 = ubuf->size(1);
        torch::Tensor output_c = torch::from_blob(
            ubuf_wt_ptr,
            {output_c_dim0, output_c_dim1},
            ubuf->options()
        );
        _cur_ubuf_id = (_cur_ubuf_id + 1) % _num_buffers;

        return {output, output_c};
    } // bulk_overlap


    /*
    ** Split FPROP GEMM + ReduceScatter
    */
    torch::Tensor split_overlap_rs(torch::Tensor input_a, torch::Tensor input_b, bool fp8, bool gemm_overlap)
    {
        // FPROP only
        bool transa = true;
        bool transb = false;

        // Get the current userbuf offset
        torch::Tensor* ubuf = &_ubufs[_cur_ubuf_id];

        // Get GEMM dimensions
        int m = input_a.size(0);
        int k = input_a.size(1);
        int n = input_b.size(0);
        int n_chunk = n / _num_splits;
        int input_b_chunk_size = n_chunk * k;
        int output_chunk_size = n_chunk * m;

        // Get input and output data pointers
        char* input_b_chunk_ptr = reinterpret_cast<char*>(input_b.data_ptr());
        char* output_chunk_ptr = reinterpret_cast<char*>(ubuf->data_ptr());
        torch::Tensor rs_output = torch::empty({n / _tp_size, m}, ubuf->options());
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
                output_chunk_ptr += output_chunk_size * ubuf->element_size();

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
                    _handles[_cur_ubuf_id],
                    ubuf_offset,
                    output_chunk_size,
                    _ub_comm,
                    (cudaStream_t) _stream_comm
                );

                ubuf_offset += output_chunk_size;
                rs_output_ptr += (output_chunk_size / _tp_size) * ubuf->element_size();
            }

            CHECK_CUDA(cudaEventRecord(_start_comm, (cudaStream_t) _stream_compute[_num_splits-1]));
            CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) _stream_comm, _start_comm, 0));

            // Last communication chunk
            reducescatter2_userbuff(
                rs_output_ptr,
                _handles[_cur_ubuf_id],
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
                    _handles[_cur_ubuf_id],
                    ubuf_offset,
                    output_chunk_size,
                    _ub_comm,
                    _stream_comm  
                );

                // Update input and output data pointers
                ubuf_offset += output_chunk_size;
                rs_output_ptr += (output_chunk_size / _tp_size) * ubuf->element_size();
                input_b_chunk_ptr += input_b_chunk_size * input_b.element_size();
                output_chunk_ptr += output_chunk_size * ubuf->element_size();
            }
        }

        CHECK_CUDA(cudaEventRecord(_stop_compute, (cudaStream_t) _stream_compute[_num_splits-1]));
        CHECK_CUDA(cudaEventRecord(_stop_comm, (cudaStream_t) _stream_comm));
        CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) stream_main, _stop_compute, 0));
        CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) stream_main, _stop_comm, 0));

        _cur_ubuf_id = (_cur_ubuf_id + 1) % _num_buffers;

        return rs_output;
    } // split_overlap_rs


    /*
    ** Split AllGather + GEMM
    */
    torch::Tensor split_overlap_ag(
        torch::Tensor input_a,
        torch::Tensor input_b,
        int gemm_input_layout,
        bool fp8)
    {
        return input_a;
    }


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
            (void*) _lt_workspaces[0].data_ptr(),
            _math_sms,
            (gemm_input_layout == 0) ? true : false,
            fp8,
            (gemm_input_layout == 2) ? true : false
        );

        return output;
    } // gemm


    /*
    ** Helper function to copy input to _ubufs[cur_ubuf_id]
    */
    void copy_input_to_ubuf(torch::Tensor input, int comm_type)
    {
        torch::Tensor* ubuf = &_ubufs[_cur_ubuf_id];
        char* ubuf_ptr = reinterpret_cast<char*>(ubuf->data_ptr());
        COMM_TYPE _comm_type = static_cast<COMM_TYPE>(comm_type);
        if (_comm_type == COMM_TYPE::AG) {
            if ((input.numel() * _tp_size) != ubuf->numel() || input.element_size() != ubuf->element_size()) {
                NVTE_ERROR("input and ubuf size do not match!");
            }
            ubuf_ptr += ubuf->numel() / _tp_size * _rank * ubuf->element_size();
        } else {
            if (input.numel() != ubuf->numel() || input.element_size() != ubuf->element_size()) {
                NVTE_ERROR("input and ubuf size do not match!");
            }
        }

        at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
        CHECK_CUDA(cudaMemcpyAsync(
            ubuf_ptr,
            input.data_ptr(),
            input.numel() * input.element_size(),
            cudaMemcpyDeviceToDevice,
            (cudaStream_t) stream_main)
        );
    }

}; // UbufCommOverlap



struct UbufP2PCommOverlap : torch::CustomClassHolder {
    int _rank;
    int _tp_size;
    std::vector<int> _handles;
    int _next_rank, _prev_rank;
    int _math_sms;
    communicator* _ub_comm;
    at::cuda::CUDAStream _stream_comm = at::cuda::getStreamFromPool(true);
    at::cuda::CUDAStream _stream_accum = at::cuda::getStreamFromPool();
    std::vector<at::cuda::CUDAStream> _stream_compute;
    torch::Tensor _ubuf;    // To return the full contiguous AG output
    std::vector<torch::Tensor> _ubufs;
    cudaEvent_t _start_compute, _stop_compute, _start_comm, _stop_comm, _start_accum, _stop_accum;
    std::vector<torch::Tensor> _lt_workspaces;

    UbufP2PCommOverlap(
        torch::Tensor& ubuf,
        int rank,
        int pp_size,
        int tp_size,
        int sm_margin,
        bool use_ce=true)
    {
        // Initialize userbuf communicator
        create_communicator_grouped2(&_ub_comm, 1, pp_size, tp_size, 1);
        _ub_comm->use_ce = (use_ce) ? 1 : 0;
        _ub_comm->push = 1;

        // Allocate ubuf chunks in a contiguous memory space
        // This is needed to return the whole chunks of AG outputs for WGRAD execution
        _ubuf = ubuf;
        int ubuf_chunk_bytes = (_ubuf.numel() * _ubuf.element_size()) / tp_size;
        for (int i = 0; i < tp_size; i++) {
            char* ubuf_ptr = reinterpret_cast<char*>(_ubuf.data_ptr());
            ubuf_ptr += ubuf_chunk_bytes * i;
            torch::Tensor ubuf_chunk = torch::from_blob(
                ubuf_ptr, {_ubuf.size(0) / tp_size, _ubuf.size(1)}, _ubuf.options()
            );
            _ubufs.push_back(ubuf_chunk);
            void* ubuf_chunk_ptr = static_cast<void*>(ubuf_chunk.data_ptr());
            _handles.push_back(register_user_buffer_collective(
                (void**)&ubuf_chunk_ptr, ubuf_chunk_bytes, _ub_comm));
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
        _math_sms = prop.multiProcessorCount - sm_margin;

        // CUDA event creation
        cudaEventCreateWithFlags(&_start_compute, 0);
        cudaEventCreateWithFlags(&_stop_compute, 0);
        cudaEventCreateWithFlags(&_start_comm, 0);
        cudaEventCreateWithFlags(&_stop_comm, 0);
        cudaEventCreateWithFlags(&_start_accum, 0);
        cudaEventCreateWithFlags(&_stop_accum, 0);
    }


    /*
    ** Split AllGather + GEMM using P2P communication
    ** This function assumes the input_b is pre-copied to _ubufs[rank_id]. This is needed to have AG outputs 
    ** in each rank to be in the contiguous memory space after all ring exchange phases.
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
        for (int i = 0; i < _tp_size; i++) {
            CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) _stream_compute[i], _start_compute, 0));
        }
        CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) _stream_comm, _start_compute, 0));


        torch::Tensor dummy = torch::empty({});
        for (int i = 0; i < _tp_size; i++) {
            // Set the userbuffer id. Buffer under send is the input for the current GEMM chunk
            // The initial input chunk is stored ubuf[rank]. This is to have the AG output in all ranks to 
            // be contiguous after all ring exchanges
            int send_id = (_tp_size + _rank - i) % _tp_size;
            int recv_id = (_tp_size + _rank - i - 1) % _tp_size;
            // FIXME: Add offset to the receiver's buffer pointer as WAR. Need to be removed once the userbuffer bug is fixed
            size_t push_offset = send_id * comm_bytes;

            // GEMM
            torch::Tensor output_chunk = torch::from_blob(
                output_ptr + (cur_ouput_chunk_id * output_chunk_bytes),
                {n_chunk, m},
                output.options()
            );
            matmul_cuda(
                input_a,
                _ubufs[send_id],
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
                userbuffers_send(_handles[send_id], 0, _handles[send_id], push_offset, comm_bytes, _ub_comm, _next_rank, (cudaStream_t) _stream_comm);
                userbuffers_recv(_handles[recv_id], 0, _handles[recv_id], 0,           comm_bytes, _ub_comm, _prev_rank, (cudaStream_t) _stream_comm);

                CHECK_CUDA(cudaEventRecord(_stop_comm, (cudaStream_t) _stream_comm));
                CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) _stream_compute[i+1], _stop_comm, 0));
            }
            cur_ouput_chunk_id = (_tp_size + cur_ouput_chunk_id - 1) % _tp_size;
        }

        CHECK_CUDA(cudaEventRecord(_stop_compute, (cudaStream_t) _stream_compute[_tp_size-1]));
        CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) stream_main, _stop_compute, 0));

        // Return the AG output to provide WGRAD GEMM input
        torch::Tensor ag_output = (get_ag_output) ? _ubuf : torch::empty({});

        return {output, ag_output};
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
        const int input_b_chunk_size = n_chunk * k;
        char* input_b_chunk_ptr = reinterpret_cast<char*>(input_b.data_ptr());

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
            // Set the userbuffer id. Start from -1 back to match the phase with GEMM input chunk id
            int send_id = (_tp_size + i - 1) % _tp_size;
            int recv_id = (_tp_size + i) % _tp_size;
            size_t recv_offset = recv_id * comm_bytes;

            // GEMM
            torch::Tensor input_b_chunk = torch::from_blob(
                input_b_chunk_ptr,
                {n_chunk, k},
                input_b.options()
            );
            matmul_cuda(
                input_a,
                input_b_chunk,
                output_bufs[i % 3],
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
                // P2P communication
                CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) _stream_comm, _start_comm, 0));

                userbuffers_send(_handles[send_id], 0, _handles[recv_id], recv_offset, comm_bytes, _ub_comm, _next_rank, (cudaStream_t) _stream_comm);
                userbuffers_recv(_handles[send_id], 0, _handles[recv_id], 0,           comm_bytes, _ub_comm, _prev_rank, (cudaStream_t) _stream_comm);

                CHECK_CUDA(cudaEventRecord(_stop_comm, (cudaStream_t) _stream_comm));
                CHECK_CUDA(cudaEventRecord(_stop_compute, (cudaStream_t) _stream_compute[i]));
                CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) _stream_accum, _stop_comm, 0));
                CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) _stream_accum, _stop_compute, 0));
                
                // Accumulate the current GEMM output to the partial sum
                {
                    c10::cuda::CUDAStreamGuard guard(_stream_accum);
                    torch::add(_ubufs[recv_id], output_bufs[i % 3]);
                }
                CHECK_CUDA(cudaEventRecord(_start_comm, (cudaStream_t) _stream_accum));
                // Update input_b pointer
                input_b_chunk_ptr += input_b_chunk_size * input_b.element_size();
            } else {
                CHECK_CUDA(cudaEventRecord(_start_comm, (cudaStream_t) _stream_compute[i]));
            }
        }
        CHECK_CUDA(cudaEventRecord(_stop_compute, (cudaStream_t) _stream_compute[_tp_size-1]));
        CHECK_CUDA(cudaEventRecord(_stop_accum, (cudaStream_t) _stream_accum));
        CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) stream_main, _stop_compute, 0));
        CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) stream_main, _stop_accum, 0));
        return _ubufs[_rank];
    }


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
        torch::Tensor psum = (gemm_input_layout == 2) ?
            torch::empty({n, m}, at::TensorOptions().dtype(output_dtype).device(torch::kCUDA)) : torch::empty({});

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
            (void*) _lt_workspaces[0].data_ptr(),
            _math_sms,
            (gemm_input_layout == 0) ? true : false,
            fp8,
            (gemm_input_layout == 2) ? true : false
        );

        return output;
    } // gemm


    /*
    ** Helper function to test ring exchange
    */
    void test_p2p_exchange() {
        const int comm_bytes = _ubufs[0].numel() * _ubufs[0].element_size();

        at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
        for (int i = 0; i < _tp_size; i++) {
//            int send_id = (_tp_size + i) % _tp_size;
//            int recv_id = (_tp_size + i + 1) % _tp_size;
//            size_t recv_offset = recv_id * comm_bytes;
//            userbuffers_send(_handles[send_id], 0, _handles[recv_id], recv_offset, comm_bytes, _ub_comm, _next_rank, (cudaStream_t) stream_main);
//            userbuffers_recv(_handles[send_id], 0, _handles[recv_id], 0, comm_bytes, _ub_comm, _prev_rank, (cudaStream_t) stream_main);

            int send_id = (_tp_size + _rank - i) % _tp_size;
            int recv_id = (_tp_size + _rank - i - 1) % _tp_size;
            size_t push_offset = send_id * comm_bytes;

            userbuffers_send(_handles[send_id], 0, _handles[send_id], push_offset, comm_bytes, _ub_comm, _next_rank, (cudaStream_t) stream_main);
            userbuffers_recv(_handles[recv_id], 0, _handles[recv_id], 0, comm_bytes, _ub_comm, _prev_rank, (cudaStream_t) stream_main);
        }
    }


    /*
    ** Helper function to test inter-rank send/recv
    */
    void test_send_recv(int send_rank, int send_id, int recv_rank, int recv_id) {
        const int comm_bytes = _ubufs[0].numel() * _ubufs[0].element_size();
        const size_t recv_offset = comm_bytes * recv_id;
        at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();

        if (_rank == send_rank) {
            userbuffers_send(_handles[send_id], 0, _handles[recv_id], recv_offset, comm_bytes, _ub_comm, 5, (cudaStream_t) stream_main);
        }
        else if (_rank == recv_rank) {
            userbuffers_recv(_handles[send_id], 0, _handles[recv_id], 0, comm_bytes, _ub_comm, 2, (cudaStream_t) stream_main);
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


//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//    py::class_<UbufCommOverlap>(m, "UbufCommOverlap")
//        .def(py::init<torch::Tensor&, int, int, int, int, int, int, int, bool>())
//        .def("rs", &UbufCommOverlap::rs)
//        .def("ag", &UbufCommOverlap::ag)
//        .def("bulk_overlap", &UbufCommOverlap::bulk_overlap)
//        .def("split_overlap_rs", &UbufCommOverlap::split_overlap_rs)
//        .def("split_overlap_ag", &UbufCommOverlap::split_overlap_ag)
//        .def("copy_input_to_ubuf", &UbufCommOverlap::copy_input_to_ubuf)
//        .def("gemm", &UbufCommOverlap::gemm);
//    py::class_<UbufP2PCommOverlap>(m, "UbufP2PCommOverlap")
//        .def(py::init<torch::Tensor&, int, int, int, int, bool>())
//        .def("split_overlap_ag", &UbufP2PCommOverlap::split_overlap_ag)
//        .def("split_overlap_rs", &UbufP2PCommOverlap::split_overlap_rs)
//        .def("gemm", &UbufP2PCommOverlap::gemm)
//        .def("copy_input_to_ubuf", &UbufP2PCommOverlap::copy_input_to_ubuf)
//        .def("test_p2p_exchange", &UbufP2PCommOverlap::test_p2p_exchange)
//        .def("test_send_recv", &UbufP2PCommOverlap::test_send_recv);
//}

} // end of namespace ubuf_comm_gemm_overlap
