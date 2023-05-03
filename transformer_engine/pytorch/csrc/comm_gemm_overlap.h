/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/cuda.h>
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/types.h>
#include "userbuffers/userbuffers.h"

#define HALF_BYTES 2
#define UB_MAX_SM 32

#define CHECK_CUDA(call)                                                                     \
  do {                                                                                       \
    cudaError_t status_ = call;                                                              \
    if (status_ != cudaSuccess) {                                                            \
      fprintf(stderr, "CUDA Error at line %d: %s\n", __LINE__, cudaGetErrorString(status_)); \
      exit(1);                                                                               \
    }                                                                                        \
  } while (0)

namespace ubuf {

enum class COMM_TYPE { RS = 0, AG = 1 };

enum class UBOverlapAlgo {
  BULK_OVERLAP_AG = 0,
  BULK_OVERLAP_RS = 1,
  SPLIT_PIPELINED_AG = 2,
  SPLIT_PIPELINED_RS = 3
};

struct UbufCommOverlap : torch::CustomClassHolder {
  communicator *_ub_comm;
  int _tp_id;
  int _tp_size;
  int _num_splits;
  int _math_sms;
  int _ub_reg;
  void *_ubuf_ptr;
  torch::Tensor _ubuf;
  torch::Tensor output_tensor;
  at::cuda::CUDAStream _stream_comm = at::cuda::getStreamFromPool(true);
  std::vector<at::cuda::CUDAStream> _stream_compute;
  cudaEvent_t _start_compute, _stop_compute, _start_d2dcopy, _start_comm, _stop_comm;

  UbufCommOverlap(torch::Tensor sample, int rank, int tp_size, int num_comm_sm, int comm_cga_size,
                  int num_splits, bool set_sm_margin, int num_max_streams) {
    // Initialize userbuf communicator
    create_communicator_grouped2(&_ub_comm, 1, 1, tp_size, 1);
    _ub_comm->use_ce = 0;
    _ub_comm->sms = num_comm_sm;
    _ub_comm->cga_size = comm_cga_size;

    // Allocate and register extra userbuffers
    int ubuf_bytes = sample.numel() * sample.element_size();
    _ub_reg = register_user_buffer_collective(reinterpret_cast<void **>(&_ubuf_ptr), ubuf_bytes,
                                              _ub_comm, true);
    _ubuf = torch::from_blob(_ubuf_ptr, {sample.size(0), sample.size(1)}, sample.options());

    at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
    for (int i = 0; i < std::min(num_max_streams, num_splits); i++) {
      cudaStream_t stream;
      cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, -1);
      _stream_compute.push_back(
          at::cuda::getStreamFromExternal(stream, stream_main.device_index()));
    }

    _num_splits = num_splits;
    _tp_size = tp_size;
    _tp_id = (rank % tp_size);

    // Set the number of SMs for GEMM with margin
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    _math_sms = (set_sm_margin) ? prop.multiProcessorCount - num_comm_sm : prop.multiProcessorCount;

    output_tensor = torch::Tensor();
    // CUDA event creation
    cudaEventCreateWithFlags(&_start_compute, 0);
    cudaEventCreateWithFlags(&_stop_compute, 0);
    cudaEventCreateWithFlags(&_start_d2dcopy, 0);
    cudaEventCreateWithFlags(&_start_comm, 0);
    cudaEventCreateWithFlags(&_stop_comm, 0);
  }

  /*
  ** Bulk GEMM + COMM
  ** This function assumes the communication input is pre-copied to _ubuf
  */
  std::vector<at::Tensor> bulk_overlap(
      at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
      transformer_engine::DType A_type, bool transa, at::Tensor B, at::Tensor B_scale_inverse,
      int64_t B_fp8_tensor, transformer_engine::DType B_type, bool transb, at::Tensor D,
      at::Tensor D_scale, transformer_engine::DType D_type, at::Tensor D_amax, at::Tensor bias,
      transformer_engine::DType bias_type, at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
      size_t workspaceSize, bool accumulate, bool use_split_accumulator, int comm_type) {
    // Get the current userbuf offset
    char *ubuf_wt_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());
    int comm_elements = (_ubuf.numel() / 2) * _ubuf.element_size();  // UBUF uses 2Byte element size
    COMM_TYPE _comm_type = static_cast<COMM_TYPE>(comm_type);
    if (_comm_type == COMM_TYPE::RS) {
      ubuf_wt_ptr += _ubuf.numel() / _tp_size * _tp_id * _ubuf.element_size();
    }

    // Catch up the default torch stream
    at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
    CHECK_CUDA(cudaEventRecord(_start_comm, (cudaStream_t)stream_main));
    CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_comm, _start_comm, 0));

    // Communication: AG and RS
    if (_comm_type == COMM_TYPE::AG) {
      allgather2_userbuff_inplace(_ub_reg, 0, comm_elements, _ub_comm, (cudaStream_t)_stream_comm);
    } else if (_comm_type == COMM_TYPE::RS) {
      reducescatter2_userbuff_inplace(_ub_reg, 0, comm_elements, _ub_comm,
                                      (cudaStream_t)_stream_comm);
    } else {
      NVTE_ERROR("Not supported communication type.");
    }

    if (A_scale_inverse.numel()) A_scale_inverse = A_scale_inverse[A_fp8_tensor];

    if (B_scale_inverse.numel()) B_scale_inverse = B_scale_inverse[B_fp8_tensor];

    assert(pre_gelu_out.numel() == 0);
    te_gemm(A, A_scale_inverse, A_type, transa, B, B_scale_inverse, B_type, transb, D, D_scale,
            D_type, D_amax, bias, bias_type, pre_gelu_out, grad, workspace, workspaceSize,
            accumulate, use_split_accumulator, _math_sms);

    CHECK_CUDA(cudaEventRecord(_stop_comm, (cudaStream_t)_stream_comm));
    CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_comm, 0));

    // Generate output tensor from userbuf data pointer
    int output_c_dim0 = (_comm_type == COMM_TYPE::AG) ? _ubuf.size(0) : _ubuf.size(0) / _tp_size;
    int output_c_dim1 = _ubuf.size(1);
    output_tensor = torch::from_blob(ubuf_wt_ptr, {output_c_dim0, output_c_dim1}, _ubuf.options());

    return {D, output_tensor};
  }  // bulk_overlap

  /*
  ** Split FPROP GEMM + ReduceScatter
  */
  void split_overlap_rs(at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
                        transformer_engine::DType A_type, bool transa, at::Tensor B,
                        at::Tensor B_scale_inverse, int64_t B_fp8_tensor,
                        transformer_engine::DType B_type, bool transb, at::Tensor D,
                        at::Tensor D_scale, transformer_engine::DType D_type, at::Tensor D_amax,
                        at::Tensor bias, transformer_engine::DType bias_type,
                        at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
                        size_t workspaceSize, bool accumulate, bool use_split_accumulator,
                        bool gemm_overlap, at::Tensor rs_output) {
    // Get GEMM dimensions
    int m = A.size(0);
    int k = A.size(1);
    int n = B.size(0);
    int m_chunk = m / _num_splits;
    int input_a_chunk_size = m_chunk * k;
    int output_chunk_size = n * m_chunk;
    int workspace_size_chunk = workspaceSize / _stream_compute.size();

    // Get input, output, and workspace data pointers
    char *input_a_chunk_ptr = reinterpret_cast<char *>(A.data_ptr());
    char *output_buf_chunk_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());
    char *workspace_ptr = reinterpret_cast<char *>(workspace.data_ptr());

    char *rs_output_ptr = reinterpret_cast<char *>(rs_output.data_ptr());
    int ubuf_offset = 0;
    int ori_sms = _ub_comm->sms;

    // Catch up the default torch stream
    at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
    CHECK_CUDA(cudaEventRecord(_start_compute, stream_main));
    for (int i = 0; i < _stream_compute.size(); i++) {
      CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_compute[i], _start_compute, 0));
    }

    if (A_scale_inverse.numel()) A_scale_inverse = A_scale_inverse[A_fp8_tensor];

    if (B_scale_inverse.numel()) B_scale_inverse = B_scale_inverse[B_fp8_tensor];

    assert(pre_gelu_out.numel() == 0);

    if (gemm_overlap) {
      torch::Tensor input_a_chunk = torch::from_blob(input_a_chunk_ptr, {m_chunk, k}, A.options());
      torch::Tensor output_chunk =
          torch::from_blob(output_buf_chunk_ptr, {n, m_chunk}, _ubuf.options());
      torch::Tensor workspace_chunk =
          torch::from_blob(workspace_ptr, {workspace_size_chunk}, workspace.options());
      at::cuda::setCurrentCUDAStream(_stream_compute[0]);
      te_gemm(input_a_chunk, A_scale_inverse, A_type, transa, B, B_scale_inverse, B_type, transb,
              output_chunk, D_scale, D_type, D_amax, bias, bias_type, pre_gelu_out, grad,
              workspace_chunk, workspace_size_chunk, accumulate, use_split_accumulator, _math_sms);

      for (int i = 1; i < _num_splits; i++) {
        input_a_chunk_ptr += input_a_chunk_size * B.element_size();
        output_buf_chunk_ptr += output_chunk_size * _ubuf.element_size();

        torch::Tensor input_a_chunk =
            torch::from_blob(input_a_chunk_ptr, {m_chunk, k}, A.options());
        torch::Tensor output_chunk =
            torch::from_blob(output_buf_chunk_ptr, {n, m_chunk}, _ubuf.options());
        torch::Tensor workspace_chunk =
            torch::from_blob(workspace_ptr + (i % _stream_compute.size()) * workspace_size_chunk,
                             {workspace_size_chunk}, workspace.options());
        at::cuda::setCurrentCUDAStream(_stream_compute[i % _stream_compute.size()]);
        te_gemm(input_a_chunk, A_scale_inverse, A_type, transa, B, B_scale_inverse, B_type, transb,
                output_chunk, D_scale, D_type, D_amax, bias, bias_type, pre_gelu_out, grad,
                workspace_chunk, workspace_size_chunk, accumulate, use_split_accumulator,
                _math_sms);

        CHECK_CUDA(cudaEventRecord(
            _start_comm, (cudaStream_t)_stream_compute[(i - 1) % _stream_compute.size()]));
        CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_comm, _start_comm, 0));

        // Communication chunk
        reducescatter2_userbuff_stridedoutput(rs_output_ptr, _ub_reg, (i - 1) * output_chunk_size,
                                              m_chunk, n, m, _ub_comm, (cudaStream_t)_stream_comm);

        rs_output_ptr += m_chunk * _ubuf.element_size();
      }
      int last_compute_stream_id =
          (_num_splits + _stream_compute.size() - 1) % _stream_compute.size();
      CHECK_CUDA(
          cudaEventRecord(_start_comm, (cudaStream_t)_stream_compute[last_compute_stream_id]));
      CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_comm, _start_comm, 0));

      // Last communication chunk with max SM
      _ub_comm->sms = UB_MAX_SM;
      reducescatter2_userbuff_stridedoutput(rs_output_ptr, _ub_reg,
                                            (_num_splits - 1) * output_chunk_size, m_chunk, n, m,
                                            _ub_comm, (cudaStream_t)_stream_comm);
    } else {
      for (int i = 0; i < _num_splits; i++) {
        torch::Tensor input_a_chunk =
            torch::from_blob(input_a_chunk_ptr, {m_chunk, k}, A.options());
        torch::Tensor output_chunk =
            torch::from_blob(output_buf_chunk_ptr, {n, m_chunk}, _ubuf.options());
        torch::Tensor workspace_chunk =
            torch::from_blob(workspace_ptr + (i % _stream_compute.size()) * workspace_size_chunk,
                             {workspace_size_chunk}, workspace.options());
        at::cuda::setCurrentCUDAStream(_stream_compute[i % _stream_compute.size()]);
        te_gemm(input_a_chunk, A_scale_inverse, A_type, transa, B, B_scale_inverse, B_type, transb,
                output_chunk, D_scale, D_type, D_amax, bias, bias_type, pre_gelu_out, grad,
                workspace_chunk, workspace_size_chunk, accumulate, use_split_accumulator,
                _math_sms);

        CHECK_CUDA(cudaEventRecord(_start_comm,
                                   (cudaStream_t)_stream_compute[i % _stream_compute.size()]));
        CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_comm, _start_comm, 0));

        // Communication chunk. Uses MAX_SM at the last chunk
        if (i == _num_splits-1) {
          _ub_comm->sms = UB_MAX_SM;
        }
        reducescatter2_userbuff_stridedoutput(rs_output_ptr, _ub_reg, i * output_chunk_size,
                                              m_chunk, n, m, _ub_comm, (cudaStream_t)_stream_comm);

        rs_output_ptr += m_chunk * _ubuf.element_size();
        input_a_chunk_ptr += input_a_chunk_size * B.element_size();
        output_buf_chunk_ptr += output_chunk_size * _ubuf.element_size();
      }
    }
    _ub_comm->sms = ori_sms;
    int last_compute_stream_id =
        (_num_splits + _stream_compute.size() - 1) % _stream_compute.size();
    CHECK_CUDA(
        cudaEventRecord(_stop_compute, (cudaStream_t)_stream_compute[last_compute_stream_id]));
    CHECK_CUDA(cudaEventRecord(_stop_comm, (cudaStream_t)_stream_comm));
    CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_compute, 0));
    CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_comm, 0));
    at::cuda::setCurrentCUDAStream(stream_main);

    return;
  }  // split_overlap_rs

  /*
  ** Helper function to copy input to _ubuf
  */
  void copy_input_to_ubuf(torch::Tensor input, int comm_type) {
    char *ubuf_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());
    COMM_TYPE _comm_type = static_cast<COMM_TYPE>(comm_type);
    if (_comm_type == COMM_TYPE::AG) {
      if ((input.numel() * _tp_size) != _ubuf.numel() ||
          input.element_size() != _ubuf.element_size()) {
        NVTE_ERROR("input and ubuf size do not match!");
      }
      ubuf_ptr += _ubuf.numel() / _tp_size * _tp_id * _ubuf.element_size();
    } else {
      if (input.numel() != _ubuf.numel() || input.element_size() != _ubuf.element_size()) {
        NVTE_ERROR("input and ubuf size do not match!");
      }
    }

    at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
    CHECK_CUDA(cudaEventRecord(_start_d2dcopy, (cudaStream_t)stream_main));
    CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_comm, _start_d2dcopy, 0));
    CHECK_CUDA(cudaMemcpyAsync(ubuf_ptr, input.data_ptr(), input.numel() * input.element_size(),
                               cudaMemcpyDeviceToDevice, (cudaStream_t)_stream_comm));
  }

  torch::Tensor &get_ubuf_output(int comm_type) {
    char *ubuf_wt_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());
    COMM_TYPE _comm_type = static_cast<COMM_TYPE>(comm_type);
    if (_comm_type != COMM_TYPE::AG && _comm_type != COMM_TYPE::RS) NVTE_ERROR("Invalid comm_type");
    if (_comm_type == COMM_TYPE::RS)
      ubuf_wt_ptr += _ubuf.numel() / _tp_size * _tp_id * _ubuf.element_size();
    int output_c_dim0 = (_comm_type == COMM_TYPE::AG) ? _ubuf.size(0) : _ubuf.size(0) / _tp_size;
    int output_c_dim1 = _ubuf.size(1);
    output_tensor = torch::from_blob(ubuf_wt_ptr, {output_c_dim0, output_c_dim1}, _ubuf.options());
    return output_tensor;
  }
};  // UbufCommOverlap

struct UbufP2PCommOverlap : torch::CustomClassHolder {
  communicator *_ub_comm;
  int _tp_id;
  int _tp_size;
  int _ub_reg;
  int _next_rank, _prev_rank, _rank, _rank_round_tp;
  int _aggregate2;
  int _math_sms;
  void *_ubuf_ptr;
  torch::Tensor _ubuf;
  std::vector<torch::Tensor> _ubufs;
  at::cuda::CUDAStream _stream_send = at::cuda::getStreamFromPool(true);
  at::cuda::CUDAStream _stream_recv = at::cuda::getStreamFromPool(true);
  std::vector<at::cuda::CUDAStream> _stream_compute;
  cudaEvent_t _start_compute, _stop_compute, _stop_send, _stop_recv;

  UbufP2PCommOverlap(torch::Tensor sample, int rank, int tp_size, bool aggregate2,
                     int num_max_streams) {
    // Initialize userbuf communicator
    create_communicator_grouped2(&_ub_comm, 1, 1, tp_size, 1);
    _ub_comm->use_ce = 1;
    _ub_comm->sms = 1;
    _ub_comm->cga_size = 1;

    // Create workspace tensor with userbuffer
    int ubuf_bytes = sample.numel() * sample.element_size();
    int ubuf_chunk_bytes = ubuf_bytes / tp_size;
    _ub_reg = register_user_buffer_collective(reinterpret_cast<void **>(&_ubuf_ptr), ubuf_bytes,
                                              _ub_comm, true);
    _ubuf = torch::from_blob(_ubuf_ptr, {sample.size(0), sample.size(1)}, sample.options());

    // Create tensor chunks for easy management
    char *ubuf_byte_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());
    for (int i = 0; i < tp_size; i++) {
      torch::Tensor ubuf_chunk = torch::from_blob(
          ubuf_byte_ptr, {sample.size(0) / tp_size, sample.size(1)}, sample.options());
      _ubufs.push_back(ubuf_chunk);
      ubuf_byte_ptr += ubuf_chunk_bytes;
    }

    at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
    for (int i = 0; i < std::min(num_max_streams, tp_size); i++) {
      cudaStream_t stream;
      cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, -1);
      _stream_compute.push_back(
          at::cuda::getStreamFromExternal(stream, stream_main.device_index()));
    }

    // Set the number of SMs for GEMM with margin
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    _math_sms = prop.multiProcessorCount;

    _tp_size = tp_size;
    _aggregate2 = aggregate2;

    _rank = rank;
    _tp_id = (rank % tp_size);
    _rank_round_tp = (rank / tp_size) * tp_size;
    _next_rank = (tp_size + rank + 1) % tp_size + _rank_round_tp;
    _prev_rank = (tp_size + rank + -1) % tp_size + _rank_round_tp;

    // CUDA event creation
    cudaEventCreateWithFlags(&_start_compute, 0);
    cudaEventCreateWithFlags(&_stop_compute, 0);
    cudaEventCreateWithFlags(&_stop_send, 0);
    cudaEventCreateWithFlags(&_stop_recv, 0);
  }

  /*
  ** Split AllGather + GEMM using P2P communication
  ** This function assumes the input_b is pre-copied to _ubufs[rank_id]. This is needed to have AG
  *outputs
  ** in each rank to be in the contiguous memory space after all ring exchange phases.
  */
  torch::Tensor split_overlap_ag(at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
                                 transformer_engine::DType A_type, bool transa, at::Tensor B,
                                 at::Tensor B_scale_inverse, int64_t B_fp8_tensor,
                                 transformer_engine::DType B_type, bool transb, at::Tensor D,
                                 at::Tensor D_scale, transformer_engine::DType D_type,
                                 at::Tensor D_amax, at::Tensor bias,
                                 transformer_engine::DType bias_type, at::Tensor pre_gelu_out,
                                 bool grad, at::Tensor workspace, size_t workspaceSize,
                                 bool accumulate, bool use_split_accumulator, at::Tensor B_copy) {
    // Get GEMM dimensions between TN and NN input layouts
    const int m = (transa) ? A.size(0) : A.size(1);
    const int k = (transa) ? A.size(1) : A.size(0);
    const int n_chunk = _ubufs[0].size(0);

    // Get communication and GEMM output chunk sizes
    const int comm_bytes = _ubufs[0].numel() * _ubufs[0].element_size();
    const int output_chunk_bytes = (n_chunk * m) * HALF_BYTES;

    // Get output and workspace data pointers
    char *output_ptr = reinterpret_cast<char *>(D.data_ptr());
    char *workspace_ptr = reinterpret_cast<char *>(workspace.data_ptr());
    int workspace_size_chunk = workspaceSize / _stream_compute.size();

    if (A_scale_inverse.numel()) A_scale_inverse = A_scale_inverse[A_fp8_tensor];

    if (B_scale_inverse.numel()) B_scale_inverse = B_scale_inverse[B_fp8_tensor];

    at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
    CHECK_CUDA(cudaEventRecord(_start_compute, (cudaStream_t)stream_main));

    assert(pre_gelu_out.numel() == 0);
    if (_aggregate2) {
      // Catch up the default torch stream
      CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_send, _start_compute, 0));
      CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_recv, _start_compute, 0));

      const int num_steps = _tp_size / 2;
      char *input_b_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());

      // Initial 1X input chunk exchange between neighboring peers
      int send_chunk_id = _tp_id;
      int recv_chunk_id = (_tp_id % 2 == 0) ? _tp_id + 1 : _tp_id - 1;
      int send_offset = comm_bytes * send_chunk_id;
      int recv_offset = comm_bytes * recv_chunk_id;
      int peer_rank = (_tp_id % 2 == 0) ? _next_rank : _prev_rank;
      userbuffers_send(_ub_reg, send_offset, _ub_reg, send_offset, comm_bytes, _ub_comm, peer_rank,
                       (cudaStream_t)_stream_send);
      userbuffers_recv(_ub_reg, recv_offset, _ub_reg, recv_offset, comm_bytes, _ub_comm, peer_rank,
                       (cudaStream_t)_stream_recv);
      CHECK_CUDA(cudaEventRecord(_stop_recv, (cudaStream_t)_stream_recv));
      CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_send, _stop_recv, 0));
      CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_compute[0], _stop_recv, 0));

      int local_rank_round2 = (_tp_id % 2 == 0) ? _tp_id : _tp_id - 1;
      const int next_rank = (_tp_size + _tp_id + 2) % _tp_size + _rank_round_tp;
      const int prev_rank = (_tp_size + _tp_id - 2) % _tp_size + _rank_round_tp;

      // Ring exchange of 2X inputs chunks
      for (int i = 0; i < num_steps; i++) {
        send_chunk_id = (_tp_size + local_rank_round2 - i * 2) % _tp_size;
        recv_chunk_id = (_tp_size + local_rank_round2 - i * 2 - 2) % _tp_size;
        send_offset = comm_bytes * send_chunk_id;
        recv_offset = comm_bytes * recv_chunk_id;

        // GEMM
        torch::Tensor input_b_chunk =
            torch::from_blob(input_b_ptr + send_offset, {n_chunk * 2, k}, _ubuf.options());
        torch::Tensor output_chunk = torch::from_blob(
            output_ptr + (send_chunk_id * output_chunk_bytes), {n_chunk * 2, m}, D.options());
        torch::Tensor workspace_chunk =
            torch::from_blob(workspace_ptr + (i % _stream_compute.size()) * workspace_size_chunk,
                             {workspace_size_chunk}, workspace.options());
        at::cuda::setCurrentCUDAStream(_stream_compute[i % _stream_compute.size()]);
        te_gemm(A, A_scale_inverse, A_type, transa, input_b_chunk, B_scale_inverse, B_type, transb,
                output_chunk, D_scale, D_type, D_amax, bias, bias_type, pre_gelu_out, grad,
                workspace_chunk, workspace_size_chunk, accumulate, use_split_accumulator,
                _math_sms);

        if (i < num_steps - 1) {
          // P2P communication
          userbuffers_send(_ub_reg, send_offset, _ub_reg, send_offset, comm_bytes * 2, _ub_comm,
                           next_rank, (cudaStream_t)_stream_send);
          userbuffers_recv(_ub_reg, recv_offset, _ub_reg, recv_offset, comm_bytes * 2, _ub_comm,
                           prev_rank, (cudaStream_t)_stream_recv);
          CHECK_CUDA(cudaEventRecord(_stop_recv, (cudaStream_t)_stream_recv));
          CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_send, _stop_recv, 0));
          CHECK_CUDA(cudaStreamWaitEvent(
              (cudaStream_t)_stream_compute[(i + 1) % _stream_compute.size()], _stop_recv, 0));
        } else if (B_copy.numel() > 0) {
          assert(B_copy.numel() == _ubufs[_tp_id].numel());
          assert(B_copy.element_size() == _ubufs[_tp_id].element_size());
          CHECK_CUDA(cudaMemcpyAsync(B_copy.data_ptr(), _ubufs[_tp_id].data_ptr(),
                                     _ubufs[_tp_id].numel() * _ubufs[_tp_id].element_size(),
                                     cudaMemcpyDeviceToDevice, (cudaStream_t)_stream_send));
          CHECK_CUDA(cudaEventRecord(_stop_send, (cudaStream_t)_stream_send));
          CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_send, 0));
        }
      }
      at::cuda::setCurrentCUDAStream(stream_main);
      int last_compute_stream_id =
          (num_steps + _stream_compute.size() - 1) % _stream_compute.size();
      CHECK_CUDA(
          cudaEventRecord(_stop_compute, (cudaStream_t)_stream_compute[last_compute_stream_id]));
    } else {
      // Catch up the default torch stream
      CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_send, _start_compute, 0));
      CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_recv, _start_compute, 0));
      CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_compute[0], _start_compute, 0));

      for (int i = 0; i < _tp_size; i++) {
        // Set the userbuffer id. Buffer under send is the input for the current GEMM chunk
        // The initial input chunk is stored _ubuf[rank]. This is to have the AG output in all ranks
        // to be contiguous after the ring exchanges
        int send_chunk_id = (_tp_size + _tp_id - i) % _tp_size;
        int recv_chunk_id = (_tp_size + _tp_id - i - 1) % _tp_size;
        int send_offset = comm_bytes * send_chunk_id;
        int recv_offset = comm_bytes * recv_chunk_id;

        // GEMM
        torch::Tensor output_chunk = torch::from_blob(
            output_ptr + (send_chunk_id * output_chunk_bytes), {n_chunk, m}, D.options());
        torch::Tensor workspace_chunk =
            torch::from_blob(workspace_ptr + (i % _stream_compute.size()) * workspace_size_chunk,
                             {workspace_size_chunk}, workspace.options());
        at::cuda::setCurrentCUDAStream(_stream_compute[i % _stream_compute.size()]);
        te_gemm(A, A_scale_inverse, A_type, transa, _ubufs[send_chunk_id], B_scale_inverse, B_type,
                transb, output_chunk, D_scale, D_type, D_amax, bias, bias_type, pre_gelu_out, grad,
                workspace_chunk, workspace_size_chunk, accumulate, use_split_accumulator,
                _math_sms);

        if (i < _tp_size - 1) {
          // P2P communication
          userbuffers_send(_ub_reg, send_offset, _ub_reg, send_offset, comm_bytes, _ub_comm,
                           _next_rank, (cudaStream_t)_stream_send);
          userbuffers_recv(_ub_reg, recv_offset, _ub_reg, recv_offset, comm_bytes, _ub_comm,
                           _prev_rank, (cudaStream_t)_stream_recv);
          CHECK_CUDA(cudaEventRecord(_stop_recv, (cudaStream_t)_stream_recv));
          CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_send, _stop_recv, 0));
          CHECK_CUDA(cudaStreamWaitEvent(
              (cudaStream_t)_stream_compute[(i + 1) % _stream_compute.size()], _stop_recv, 0));
        } else if (B_copy.numel() > 0) {
          assert(B_copy.numel() == _ubufs[_tp_id].numel());
          assert(B_copy.element_size() == _ubufs[_tp_id].element_size());
          CHECK_CUDA(cudaMemcpyAsync(B_copy.data_ptr(), _ubufs[_tp_id].data_ptr(),
                                     _ubufs[_tp_id].numel() * _ubufs[_tp_id].element_size(),
                                     cudaMemcpyDeviceToDevice, (cudaStream_t)_stream_send));
          CHECK_CUDA(cudaEventRecord(_stop_send, (cudaStream_t)_stream_send));
          CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_send, 0));
        }
      }
      at::cuda::setCurrentCUDAStream(stream_main);
      int last_compute_stream_id = (_tp_size + _stream_compute.size() - 1) % _stream_compute.size();
      CHECK_CUDA(
          cudaEventRecord(_stop_compute, (cudaStream_t)_stream_compute[last_compute_stream_id]));
    }
    CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_compute, 0));

    return D;
  }  // split_overlap_ag

  /*
  ** Copy input to _ubufs[0]
  */
  void copy_input_to_ubuf(torch::Tensor input, bool chunk) {
    at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
    if (chunk) {
      // Copy input to the target ubuf chunk by rank offset
      if (input.numel() != _ubufs[0].numel() || input.element_size() != _ubufs[0].element_size()) {
        NVTE_ERROR("input and ubuf size do not match!");
      }
      CHECK_CUDA(cudaMemcpyAsync(_ubufs[_tp_id].data_ptr(), input.data_ptr(),
                                 input.numel() * input.element_size(), cudaMemcpyDeviceToDevice,
                                 (cudaStream_t)stream_main));
    } else {
      if (input.numel() != _ubuf.numel() || input.element_size() != _ubuf.element_size()) {
        NVTE_ERROR("input and ubuf size do not match!");
      }
      CHECK_CUDA(cudaMemcpyAsync(_ubuf.data_ptr(), input.data_ptr(),
                                 input.numel() * input.element_size(), cudaMemcpyDeviceToDevice,
                                 (cudaStream_t)stream_main));
    }
  }
  torch::Tensor get_ubuf_output(int comm_type) {
    char *ubuf_wt_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());
    COMM_TYPE _comm_type = static_cast<COMM_TYPE>(comm_type);
    if (_comm_type != COMM_TYPE::AG && _comm_type != COMM_TYPE::RS) NVTE_ERROR("Invalid comm_type");
    if (_comm_type == COMM_TYPE::RS)
      ubuf_wt_ptr += _ubuf.numel() / _tp_size * _tp_id * _ubuf.element_size();
    int output_c_dim0 = (_comm_type == COMM_TYPE::AG) ? _ubuf.size(0) : _ubuf.size(0) / _tp_size;
    int output_c_dim1 = _ubuf.size(1);
    return torch::from_blob(ubuf_wt_ptr, {output_c_dim0, output_c_dim1}, _ubuf.options());
  }
};  // UbufP2PCommOverlap

}  // namespace ubuf
