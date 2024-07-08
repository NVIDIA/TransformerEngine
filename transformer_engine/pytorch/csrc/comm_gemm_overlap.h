/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_COMM_GEMM_OVERLAP_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_COMM_GEMM_OVERLAP_H_

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp8.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/cuda.h>
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/types.h>

#include "common/util/logging.h"
#include "common/util/system.h"
#include "extensions.h"
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

using namespace torch::indexing;
namespace ubuf {

/*
** Static container for Python callbacks to torch.distributed collectives
*/
static struct TorchCallbacks : torch::CustomClassHolder {
  bool initialized{false};
  std::unordered_map<void *, at::Tensor> gathered_tensors;
  std::function<at::Tensor(at::Tensor &, const std::string &)> allgather;
  std::function<void(const std::string &)> barrier;
  std::function<void(at::Tensor &)> free;
} torch_callbacks;

/*
** Helper function for setting Python callbacks to torch.distributed collectives.
*/
void set_ubuf_bootstrap_callbacks(
    std::function<at::Tensor(at::Tensor &, const std::string &)> allgather,
    std::function<void(const std::string &)> barrier, std::function<void(at::Tensor &)> free) {
  torch_callbacks.allgather = allgather;
  torch_callbacks.barrier = barrier;
  torch_callbacks.free = free;
  torch_callbacks.initialized = true;
}

/*
** Python callback for globaldata = torch.distributed.all_gather(localdata, tp_group).
** This *creates* a new tensor, which Userbuffers later frees with a separate callback.
*/
void ub_alloc_copy_allgather(void **globaldata, void *localdata, size_t localbytes, char *group) {
  assert(torch_callbacks.initialized);
  auto localtensor =
      torch::from_blob(localdata, {static_cast<int64_t>(localbytes / sizeof(uint8_t))},
                       at::device(torch::kCPU).dtype(torch::kUInt8));
  auto globaltensor = torch_callbacks.allgather(localtensor, group);
  *globaldata = globaltensor.data_ptr();
  torch_callbacks.gathered_tensors[*globaldata] = globaltensor;
}

/*
** Python callback for torch.distributed.barrier(tp_group).
*/
void ub_barrier(char *group) {
  assert(torch_callbacks.initialized);
  torch_callbacks.barrier(group);
}

/*
** Python callback for freeing up tensors created in the ub_alloc_copy_allgather(...) callback.
*/
void ub_free(void *ptr) {
  assert(torch_callbacks.initialized);
  auto i = torch_callbacks.gathered_tensors.find(ptr);
  if (i == torch_callbacks.gathered_tensors.end()) return;
  auto tensor = std::move(i->second);
  torch_callbacks.gathered_tensors.erase(i);
  torch_callbacks.free(tensor);
}

enum class COMM_TYPE { RS = 0, AG = 1 };

enum class UBOverlapAlgo {
  BULK_OVERLAP_AG = 0,
  BULK_OVERLAP_RS = 1,
  SPLIT_PIPELINED_AG_P2P = 2,
  SPLIT_PIPELINED_RS = 3,
  SPLIT_PIPELINED_RS_P2P = 4,
  ATOMIC_GEMM_RS = 5,
  ATOMIC_GEMM_AG_P2P = 6,
  ATOMIC_GEMM_RS_P2P = 7
};

struct UbufBase {
  static inline communicator *_ub_comm{nullptr};
  static inline bool comm_created{false};
};
struct UbufCommOverlap : torch::CustomClassHolder, UbufBase {
  int _tp_id;
  int _tp_size;
  int _num_splits;
  int _math_sms;
  int _ub_reg;
  void *_ubuf_ptr;
  torch::Tensor _ubuf;
  torch::Tensor output_tensor;
  torch::Tensor _ubuf_scale_inv;
  bool _ubuf_scale_inv_initialized;
  torch::Tensor counter;
  torch::Tensor _empty_tensor;
  at::cuda::CUDAStream _stream_comm = at::cuda::getStreamFromPool(true);
  std::vector<at::cuda::CUDAStream> _stream_compute;
  cudaEvent_t _start_compute, _stop_compute, _start_d2dcopy, _start_comm, _stop_comm;
  int _num_comm_sm;
  int _cga_size;
  int _use_ce;
  bool _atomic_gemm;

  UbufCommOverlap(torch::Tensor sample, int rank, int world_size, int tp_rank, int tp_size,
                  int num_comm_sm, int comm_cga_size, int num_splits, bool set_sm_margin,
                  int num_max_streams, bool atomic_gemm, torch::Tensor empty_tensor) {
    // Initialize userbuf communicator
    if (!comm_created) {
      if (rank == 0) {
        printf("!!! [UB] Create UbufCommOverlap Communicator\n");
      }
      if (transformer_engine::getenv<bool>("UB_MPI_BOOTSTRAP")) {
        create_communicator_grouped2_mpi(&_ub_comm, 1, 1, tp_size, 1);
      } else {
        create_communicator_grouped2(&_ub_comm, rank, world_size, tp_rank, tp_size, 1, 1,
                                     &ub_alloc_copy_allgather, &ub_barrier, &ub_free, 1, 1, tp_size,
                                     1);
      }
      comm_created = true;
    }
    _use_ce = 0;
    _num_comm_sm = num_comm_sm;
    _cga_size = comm_cga_size;
    _empty_tensor = empty_tensor;

    // Allocate and register extra userbuffers
    int ubuf_bytes = sample.numel() * sample.element_size();
    _ub_reg = register_user_buffer_collective(reinterpret_cast<void **>(&_ubuf_ptr), ubuf_bytes,
                                              _ub_comm, true);
    if (rank == 0) {
      printf("!!! [UB] Register UBuf %d\n", _ub_reg);
    }
    _ubuf = torch::from_blob(_ubuf_ptr, {sample.size(0), sample.size(1)}, sample.options());

    at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
    for (int i = 0; i < std::min(num_max_streams, num_splits); i++) {
      cudaStream_t stream;
      cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, -1);
      _stream_compute.push_back(
          at::cuda::getStreamFromExternal(stream, stream_main.device_index()));
    }

    _num_splits = num_splits;
    _tp_size = tp_size;
    _tp_id = (rank % tp_size);
    _ubuf_scale_inv_initialized = false;

    // Set the number of SMs for GEMM with margin
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    _math_sms = (set_sm_margin) ? prop.multiProcessorCount - num_comm_sm : prop.multiProcessorCount;
    _math_sms -= transformer_engine::getenv<int>("NVTE_EXT_MARGIN_SM", 0);

    output_tensor = torch::Tensor();
    _atomic_gemm = atomic_gemm;
    if (_atomic_gemm) {
      auto counter_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
      counter = torch::zeros({num_splits * 2}, counter_options);
      counter.index_put_({Slice(None, num_splits)}, 1);
    }
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
      size_t workspaceSize, bool accumulate, bool use_split_accumulator, int comm_type,
      at::Tensor rs_output) {
    _ub_comm->use_ce = _use_ce;
    _ub_comm->sms = _num_comm_sm;
    _ub_comm->cga_size = _cga_size;
    // Get the current userbuf offset
    char *ubuf_wt_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());
    int comm_elements = (_ubuf.numel() / 2) * _ubuf.element_size();  // UBUF uses 2Byte element size
    COMM_TYPE _comm_type = static_cast<COMM_TYPE>(comm_type);
    if (_comm_type == COMM_TYPE::RS) {
      ubuf_wt_ptr += _ubuf.numel() / _tp_size * _tp_id * _ubuf.element_size();
    }

    // Catch up the default torch stream
    at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
    CHECK_CUDA(cudaEventRecord(_start_comm, (cudaStream_t)stream_main));
    CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_comm, _start_comm, 0));

    // Communication: AG and RS
    if (_comm_type == COMM_TYPE::AG) {
      allgather2_userbuff_inplace(_ub_reg, 0, comm_elements, _ub_comm, (cudaStream_t)_stream_comm);
    } else if (_comm_type == COMM_TYPE::RS) {
      if (_ubuf.element_size() == 1) {
        assert(_ubuf_scale_inv_initialized);
        comm_elements *= 2;
        float *scale_inv_ptr = reinterpret_cast<float *>(_ubuf_scale_inv.data_ptr());
        assert(rs_output.numel() == _ubuf.numel() / _tp_size);
        assert(rs_output.size(0) == _ubuf.size(0) / _tp_size);
        assert(rs_output.element_size() == 2);
        char *rs_output_ptr = reinterpret_cast<char *>(rs_output.data_ptr());
        reducescatter2_userbuff_fp8<__nv_fp8_e5m2>(rs_output_ptr, scale_inv_ptr, _ub_reg, 0,
                                                   comm_elements, _ub_comm,
                                                   (cudaStream_t)_stream_comm);
      } else {
        reducescatter2_userbuff_inplace(_ub_reg, 0, comm_elements, _ub_comm,
                                        (cudaStream_t)_stream_comm);
      }
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
  void atomic_gemm_overlap_rs(at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
                              transformer_engine::DType A_type, bool transa, at::Tensor B,
                              at::Tensor B_scale_inverse, int64_t B_fp8_tensor,
                              transformer_engine::DType B_type, bool transb, at::Tensor D,
                              at::Tensor D_scale, transformer_engine::DType D_type,
                              at::Tensor D_amax, at::Tensor bias,
                              transformer_engine::DType bias_type, at::Tensor pre_gelu_out,
                              bool grad, at::Tensor workspace, size_t workspaceSize,
                              bool accumulate, bool use_split_accumulator, bool gemm_overlap,
                              at::Tensor rs_output) {
    _ub_comm->use_ce = _use_ce;
    _ub_comm->sms = _num_comm_sm;
    _ub_comm->cga_size = _cga_size;
    // Get GEMM dimensions
    int m = A.size(0);
    int k = A.size(1);
    int n = B.size(0);
    int m_chunk = m / _num_splits;
    int workspace_size_chunk = workspaceSize / _stream_compute.size();

    // Get input, output, and workspace data pointers
    char *input_a_chunk_ptr = reinterpret_cast<char *>(A.data_ptr());
    char *output_buf_chunk_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());
    char *workspace_ptr = reinterpret_cast<char *>(workspace.data_ptr());
    int *counter_ptr = reinterpret_cast<int *>(counter.data_ptr());
    char *rs_output_ptr = reinterpret_cast<char *>(rs_output.data_ptr());
    int ori_sms = _ub_comm->sms;

    // Catch up the default torch stream
    at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
    CHECK_CUDA(cudaEventRecord(_start_compute, stream_main));
    CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_compute[0], _start_compute, 0));
    CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_comm, _start_compute, 0));

    if (A_scale_inverse.numel()) A_scale_inverse = A_scale_inverse[A_fp8_tensor];

    if (B_scale_inverse.numel()) B_scale_inverse = B_scale_inverse[B_fp8_tensor];

    assert(pre_gelu_out.numel() == 0);

    torch::Tensor input_a = torch::from_blob(input_a_chunk_ptr, {m, k}, A.options());
    torch::Tensor output_d = torch::from_blob(output_buf_chunk_ptr, {n, m}, _ubuf.options());
    //    torch::zeros({n, m}, _ubuf.options());
    torch::Tensor workspace_chunk =
        torch::from_blob(workspace_ptr, {workspace_size_chunk}, workspace.options());
    at::cuda::setCurrentCUDAStream(_stream_compute[0]);
    te_atomic_gemm(input_a, A_scale_inverse, A_type, transa, B, B_scale_inverse, B_type, transb,
                   output_d, D_scale, D_type, D_amax, bias, bias_type, pre_gelu_out, grad,
                   workspace_chunk, workspace_size_chunk, accumulate, use_split_accumulator,
                   _math_sms, _num_splits /*m_split*/, 0 /*n_split*/, true /*gemm_producer*/,
                   counter);
    for (int i = 0; i < _num_splits; i++) {
      const char *env_p = std::getenv("NVTE_RS_STRIDED_ATOMIC");
      if (env_p != nullptr && env_p[0] == '1') {
        if (i == _num_splits - 1) {
          _ub_comm->sms = UB_MAX_SM;
        }
        if (_ubuf.element_size() == 1) {
          assert(_ubuf_scale_inv_initialized);
          float *d_scale_inv_ptr = reinterpret_cast<float *>(_ubuf_scale_inv.data_ptr());
          reducescatter2_userbuff_strided_atomic_fp8<__nv_fp8_e4m3>(
              rs_output_ptr, d_scale_inv_ptr, _ub_reg, i * m_chunk, m_chunk, n, m, m, _num_splits,
              &counter_ptr[i], _ub_comm, (cudaStream_t)_stream_comm);
        } else {
          reducescatter2_userbuff_strided_atomic(rs_output_ptr, _ub_reg, i * m_chunk, m_chunk, n, m,
                                                 _num_splits, &counter_ptr[i], _ub_comm,
                                                 (cudaStream_t)_stream_comm);
        }
      } else if (env_p != nullptr && env_p[0] == '2') {
        if (_ubuf.element_size() == 1) {
          assert(_ubuf_scale_inv_initialized);
          float *d_scale_inv_ptr = reinterpret_cast<float *>(_ubuf_scale_inv.data_ptr());
          reducescatter2_userbuff_strided_multiatomic_fp8<__nv_fp8_e4m3>(
              rs_output_ptr, d_scale_inv_ptr, _ub_reg, m_chunk, m_chunk, n, m, m, _num_splits,
              counter_ptr, _ub_comm, (cudaStream_t)_stream_comm);
        } else {
          reducescatter2_userbuff_strided_multiatomic(rs_output_ptr, _ub_reg, m_chunk, m_chunk, n,
                                                      m, _num_splits, counter_ptr, _ub_comm,
                                                      (cudaStream_t)_stream_comm);
        }
        break;
      } else {
        consumer(counter_ptr, i, (cudaStream_t)_stream_comm);
        //        if (i == _num_splits-1) {
        //           _ub_comm->sms = UB_MAX_SM;
        //        }
        reducescatter2_userbuff_strided(rs_output_ptr, _ub_reg, i * m_chunk, m_chunk, n, m,
                                        _ub_comm, (cudaStream_t)_stream_comm);
      }

      rs_output_ptr += m_chunk * rs_output.element_size();
    }

    _ub_comm->sms = ori_sms;
    CHECK_CUDA(cudaEventRecord(_stop_compute, (cudaStream_t)_stream_compute[0]));
    CHECK_CUDA(cudaEventRecord(_stop_comm, (cudaStream_t)_stream_comm));
    CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_compute, 0));
    CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_comm, 0));
    at::cuda::setCurrentCUDAStream(stream_main);

    return;
  }  // split_overlap_rs

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
    _ub_comm->use_ce = _use_ce;
    _ub_comm->sms = _num_comm_sm;
    _ub_comm->cga_size = _cga_size;
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
    int ori_sms = _ub_comm->sms;

    // Catch up the default torch stream
    at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
    CHECK_CUDA(cudaEventRecord(_start_compute, stream_main));
    for (size_t i = 0; i < _stream_compute.size(); i++) {
      CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_compute[i], _start_compute, 0));
    }
    CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_comm, _start_compute, 0));

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
        if (_ubuf.element_size() == 1) {
          assert(_ubuf_scale_inv_initialized);
          float *d_scale_inv_ptr = reinterpret_cast<float *>(_ubuf_scale_inv.data_ptr());
          reducescatter2_userbuff_stridedoutput_fp8<__nv_fp8_e4m3>(
              rs_output_ptr, d_scale_inv_ptr, _ub_reg, (i - 1) * output_chunk_size, m_chunk, n, m,
              _ub_comm, (cudaStream_t)_stream_comm);
        } else {
          reducescatter2_userbuff_stridedoutput(rs_output_ptr, _ub_reg, (i - 1) * output_chunk_size,
                                                m_chunk, n, m, _ub_comm,
                                                (cudaStream_t)_stream_comm);
        }

        rs_output_ptr += m_chunk * rs_output.element_size();
      }
      int last_compute_stream_id =
          (_num_splits + _stream_compute.size() - 1) % _stream_compute.size();
      CHECK_CUDA(
          cudaEventRecord(_start_comm, (cudaStream_t)_stream_compute[last_compute_stream_id]));
      CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_comm, _start_comm, 0));

      // Last communication chunk with max SM
      _ub_comm->sms = UB_MAX_SM;
      if (_ubuf.element_size() == 1) {
        assert(_ubuf_scale_inv_initialized);
        float *d_scale_inv_ptr = reinterpret_cast<float *>(_ubuf_scale_inv.data_ptr());
        reducescatter2_userbuff_stridedoutput_fp8<__nv_fp8_e4m3>(
            rs_output_ptr, d_scale_inv_ptr, _ub_reg, (_num_splits - 1) * output_chunk_size, m_chunk,
            n, m, _ub_comm, (cudaStream_t)_stream_comm);
      } else {
        reducescatter2_userbuff_stridedoutput(rs_output_ptr, _ub_reg,
                                              (_num_splits - 1) * output_chunk_size, m_chunk, n, m,
                                              _ub_comm, (cudaStream_t)_stream_comm);
      }
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
        if (i == _num_splits - 1) {
          _ub_comm->sms = UB_MAX_SM;
        }
        if (_ubuf.element_size() == 1) {
          assert(_ubuf_scale_inv_initialized);
          float *d_scale_inv_ptr = reinterpret_cast<float *>(_ubuf_scale_inv.data_ptr());
          reducescatter2_userbuff_stridedoutput_fp8<__nv_fp8_e4m3>(
              rs_output_ptr, d_scale_inv_ptr, _ub_reg, i * output_chunk_size, m_chunk, n, m,
              _ub_comm, (cudaStream_t)_stream_comm);
        } else {
          reducescatter2_userbuff_stridedoutput(rs_output_ptr, _ub_reg, i * output_chunk_size,
                                                m_chunk, n, m, _ub_comm,
                                                (cudaStream_t)_stream_comm);
        }
        rs_output_ptr += m_chunk * rs_output.element_size();
        input_a_chunk_ptr += input_a_chunk_size * B.element_size();
        output_buf_chunk_ptr += output_chunk_size * _ubuf.element_size();
      }
    }
    for (size_t i = 0; i < _stream_compute.size(); i++) {
      CHECK_CUDA(cudaEventRecord(_stop_compute, (cudaStream_t)_stream_compute[i]));
      CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_compute, 0));
    }
    _ub_comm->sms = ori_sms;
    CHECK_CUDA(cudaEventRecord(_stop_comm, (cudaStream_t)_stream_comm));
    CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_comm, 0));
    at::cuda::setCurrentCUDAStream(stream_main);

    return;
  }  // split_overlap_rs

  void set_ubuf_scale_inv(const torch::Tensor &scale_inv) {
    _ubuf_scale_inv = scale_inv;
    _ubuf_scale_inv_initialized = true;
  }

  bool is_fp8_ubuf() { return (_ubuf.element_size() == 1); }
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

    at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
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

  bool is_atomic_gemm() { return _atomic_gemm; }
  bool is_p2p_overlap() { return false; }
};  // UbufCommOverlap

struct UbufP2PCommOverlap : torch::CustomClassHolder, UbufBase {
  int _tp_id;
  int _tp_size;
  int _ub_reg, _ub_reg2;
  int _next_rank, _prev_rank, _rank, _rank_round_tp;
  int _aggregate2;
  int _math_sms;
  int _self_chunk_id;
  void *_ubuf_ptr;
  torch::Tensor _ubuf;
  torch::Tensor counter;
  torch::Tensor _empty_tensor;
  torch::Tensor _ubuf_scale_inv;
  bool _ubuf_scale_inv_initialized;
  std::vector<torch::Tensor> _ubufs;
  at::cuda::CUDAStream _stream_send = at::cuda::getStreamFromPool(true);
  at::cuda::CUDAStream _stream_recv = at::cuda::getStreamFromPool(true);
  std::vector<at::cuda::CUDAStream> _stream_compute;
  cudaEvent_t _start_compute, _stop_compute, _start_comm, _stop_send, _stop_recv;
  int _use_ce;
  int _num_comm_sm;
  int _cga_size;
  bool _atomic_gemm;

  UbufP2PCommOverlap(torch::Tensor sample, int rank, int world_size, int tp_rank, int tp_size,
                     int num_comm_sm, int comm_cga_size, bool set_sm_margin, bool aggregate2,
                     int num_max_streams, bool is_reduce_scatter, bool atomic_gemm, bool use_ce,
                     torch::Tensor empty_tensor) {
    // Initialize userbuf communicator
    if (!comm_created) {
      if (rank == 0) {
        printf("!!! [UB] Create UbufP2PCommOverlap Communicator\n");
      }
      if (transformer_engine::getenv<bool>("UB_MPI_BOOTSTRAP")) {
        create_communicator_grouped2_mpi(&_ub_comm, 1, 1, tp_size, 1);
      } else {
        create_communicator_grouped2(&_ub_comm, rank, world_size, tp_rank, tp_size, 1, 1,
                                     &ub_alloc_copy_allgather, &ub_barrier, &ub_free, 1, 1, tp_size,
                                     1);
      }
      comm_created = true;
    }
    _use_ce = use_ce;
    _num_comm_sm = num_comm_sm;
    _cga_size = comm_cga_size;

    _empty_tensor = empty_tensor;
    // Create workspace tensor with userbuffer
    int ubuf_bytes = sample.numel() * sample.element_size();
    int ubuf_chunk_bytes = ubuf_bytes / tp_size;
    int num_ubuf_chunks = tp_size;
    if (is_reduce_scatter) {
      // GEMM + RS overlap: Allocate `2 x tp_size - 1` buffers to hold recieved GEMM chunk
      // outputs for reduction at the end of the pipelining.
      ubuf_bytes = static_cast<int>(ubuf_bytes / tp_size * (tp_size * 2 - 1));
      num_ubuf_chunks = static_cast<int>(tp_size * 2 - 1);
    }
    _ub_reg = register_user_buffer_collective(reinterpret_cast<void **>(&_ubuf_ptr), ubuf_bytes,
                                              _ub_comm, true);
    if (rank == 0) {
      printf("!!! [UBP2P] Register UBuf %d\n", _ub_reg);
    }

    _ubuf = torch::from_blob(
        _ubuf_ptr, {sample.size(0) / tp_size * num_ubuf_chunks, sample.size(1)}, sample.options());

    // Create tensor chunks for easy management
    char *ubuf_byte_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());
    for (int i = 0; i < num_ubuf_chunks; i++) {
      torch::Tensor ubuf_chunk = torch::from_blob(
          ubuf_byte_ptr, {sample.size(0) / tp_size, sample.size(1)}, sample.options());
      _ubufs.push_back(ubuf_chunk);
      ubuf_byte_ptr += ubuf_chunk_bytes;
    }

    at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
    for (int i = 0; i < std::min(num_max_streams, tp_size); i++) {
      cudaStream_t stream;
      cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, -1);
      _stream_compute.push_back(
          at::cuda::getStreamFromExternal(stream, stream_main.device_index()));
    }

    // Set the number of SMs for GEMM with margin
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    _math_sms = (set_sm_margin) ? prop.multiProcessorCount - num_comm_sm : prop.multiProcessorCount;
    _math_sms -= transformer_engine::getenv<int>("NVTE_EXT_MARGIN_SM", 0);

    _tp_size = tp_size;
    _aggregate2 = aggregate2;

    _rank = rank;
    _tp_id = (rank % tp_size);
    _rank_round_tp = (rank / tp_size) * tp_size;
    _next_rank = (tp_size + rank + 1) % tp_size + _rank_round_tp;
    _prev_rank = (tp_size + rank + -1) % tp_size + _rank_round_tp;
    _ubuf_scale_inv_initialized = false;

    _atomic_gemm = atomic_gemm;
    _self_chunk_id = _tp_id;
    if (_atomic_gemm) {
      auto counter_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
      counter = torch::zeros({tp_size * 2}, counter_options);
      counter.index_put_({Slice(None, tp_size)}, 1);

      if (!is_reduce_scatter) {
        const char *env_p = std::getenv("NVTE_AG_P2P_MULTI_ATOMIC");
        if (rank == 0 && env_p != nullptr) {
          if (env_p[0] == '1') {
            printf("!!userbuffers_sendrecv_multi_atomic_shuffle\n");
          }
        }
        _self_chunk_id = 0;
        counter.index_put_({_self_chunk_id}, 0);
      }
    }

    // CUDA event creation
    cudaEventCreateWithFlags(&_start_compute, 0);
    cudaEventCreateWithFlags(&_stop_compute, 0);
    cudaEventCreateWithFlags(&_start_comm, 0);
    cudaEventCreateWithFlags(&_stop_send, 0);
    cudaEventCreateWithFlags(&_stop_recv, 0);
  }

  /*
  ** Split AllGather + AtomicGEMM using P2P communication
  ** This function assumes the input_b is pre-copied to _ubufs[rank_id]. This is
  *needed to have AG outputs
  ** in each rank to be in the contiguous memory space after all ring exchange
  *phases.
  */
  torch::Tensor atomic_gemm_overlap_ag(
      at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
      transformer_engine::DType A_type, bool transa, at::Tensor B, at::Tensor B_scale_inverse,
      int64_t B_fp8_tensor, transformer_engine::DType B_type, bool transb, at::Tensor D,
      at::Tensor D_scale, transformer_engine::DType D_type, at::Tensor D_amax, at::Tensor bias,
      transformer_engine::DType bias_type, at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
      size_t workspaceSize, bool accumulate, bool use_split_accumulator, at::Tensor B_copy) {
    _ub_comm->use_ce = _use_ce;
    _ub_comm->sms = _num_comm_sm;
    _ub_comm->cga_size = _cga_size;
    // Get GEMM dimensions between TN and NN input layouts
    const int m = (transa) ? A.size(0) : A.size(1);
    const int n = _ubuf.size(0);
    const int n_chunk = n / _tp_size;

    // Get communication and GEMM output chunk sizes
    const int comm_bytes = _ubufs[0].numel() * _ubufs[0].element_size();

    // Create an GEMM output buffer with N+1 chunks in a contiguous memory
    torch::Tensor D_buffer = torch::empty({n_chunk * (_tp_size + 1), m}, D.options());
    D = torch::from_blob(D_buffer.data_ptr(), {D.size(0), D.size(1)}, D.options());

    // Get output and workspace data pointers
    char *workspace_ptr = reinterpret_cast<char *>(workspace.data_ptr());
    int *counter_ptr = reinterpret_cast<int *>(counter.data_ptr());
    int workspace_size_chunk = workspaceSize / _stream_compute.size();

    if (A_scale_inverse.numel()) A_scale_inverse = A_scale_inverse[A_fp8_tensor];

    if (B_scale_inverse.numel()) B_scale_inverse = B_scale_inverse[B_fp8_tensor];

    assert(pre_gelu_out.numel() == 0);

    // Catch up the default torch stream
    at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
    CHECK_CUDA(cudaEventRecord(_start_compute, (cudaStream_t)stream_main));
    CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_send, _start_compute, 0));
    CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_recv, _start_compute, 0));

    torch::Tensor workspace_chunk =
        torch::from_blob(workspace_ptr, {workspace_size_chunk}, workspace.options());

    for (int i = 0; i < _tp_size - 1; i++) {
      // Set the userbuffer id. Buffer under send is the input for the current
      // GEMM chunk The initial input chunk is stored _ubuf[rank]. This is to
      // have the AG output in all ranks to be contiguous after the ring
      // exchanges
      int send_chunk_id = i;
      int recv_chunk_id = i + 1;
      int send_offset = comm_bytes * send_chunk_id;
      int recv_offset = comm_bytes * recv_chunk_id;

      const char *env_p = std::getenv("NVTE_AG_P2P_MULTI_ATOMIC");
      if (env_p != nullptr && env_p[0] == '1') {
        if (i == 0) {
          userbuffers_sendrecv_multiatomic(_ub_reg, _ub_reg, comm_bytes, comm_bytes, comm_bytes,
                                           _ub_comm, _next_rank, _prev_rank, _tp_size, counter_ptr,
                                           true, (cudaStream_t)_stream_recv);
        }
      } else {
        userbuffers_send(_ub_reg, send_offset, _ub_reg, recv_offset, comm_bytes, _ub_comm,
                         _next_rank, (cudaStream_t)_stream_recv);
        userbuffers_recv(_ub_reg, send_offset, _ub_reg, recv_offset, comm_bytes, _ub_comm,
                         _prev_rank, (cudaStream_t)_stream_recv);
        producer(counter_ptr, recv_chunk_id, (cudaStream_t)_stream_recv);
      }
      if (i == 0) {
        te_atomic_gemm(A, A_scale_inverse, A_type, transa, _ubuf, B_scale_inverse, B_type, transb,
                       D, D_scale, D_type, D_amax, bias, bias_type, pre_gelu_out, grad,
                       workspace_chunk, workspace_size_chunk, accumulate, use_split_accumulator,
                       _math_sms, 0, _tp_size, false, counter);
      }
    }

    // Store the input activation for backprop
    if (B_copy.numel() > 0) {
      assert(B_copy.numel() == _ubufs[_self_chunk_id].numel());
      assert(B_copy.element_size() == _ubufs[_self_chunk_id].element_size());
      CHECK_CUDA(
          cudaMemcpyAsync(B_copy.data_ptr(), _ubufs[_self_chunk_id].data_ptr(),
                          _ubufs[_self_chunk_id].numel() * _ubufs[_self_chunk_id].element_size(),
                          cudaMemcpyDeviceToDevice, (cudaStream_t)_stream_send));
      CHECK_CUDA(cudaEventRecord(_stop_send, (cudaStream_t)_stream_send));
      CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_send, 0));
    }

    // Reset atomic counters
    consumer_batch(counter_ptr, 1, _tp_size, (cudaStream_t)stream_main);

    // Copy the first GEMM output chunk to the end chunk position of D_buffer
    char *src_ptr = reinterpret_cast<char *>(D_buffer.data_ptr());
    CHECK_CUDA(cudaMemcpyAsync(src_ptr + (D.numel() * D.element_size()), src_ptr,
                               n_chunk * m * D.element_size(), cudaMemcpyDeviceToDevice,
                               (cudaStream_t)stream_main));
    // Return the last N rows of D_buffer
    torch::Tensor D_return = D_buffer.narrow(0, n_chunk, n);
    return D_return;
  }  // atomic_gemm_overlap_ag

  /*
  ** Split AllGather + GEMM using P2P communication
  ** This function assumes the input_b is pre-copied to _ubufs[rank_id]. This is
  *needed to have AG outputs
  ** in each rank to be in the contiguous memory space after all ring exchange
  *phases.
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
    _ub_comm->use_ce = _use_ce;
    _ub_comm->sms = _num_comm_sm;
    _ub_comm->cga_size = _cga_size;
    // Get GEMM dimensions between TN and NN input layouts
    const int m = (transa) ? A.size(0) : A.size(1);
    const int k = (transa) ? A.size(1) : A.size(0);
    const int n_chunk = _ubufs[0].size(0);

    // Get communication and GEMM output chunk sizes
    const int comm_bytes = _ubufs[0].numel() * _ubufs[0].element_size();
    const bool do_gelu = pre_gelu_out.numel() > 0;
    const int output_chunk_bytes = (n_chunk * m) * D.element_size();
    const int aux_chunk_bytes = do_gelu ? (n_chunk * m) * pre_gelu_out.element_size() : 0;

    // Get output and workspace data pointers
    char *output_ptr = reinterpret_cast<char *>(D.data_ptr());
    char *pre_gelu_out_ptr = reinterpret_cast<char *>(pre_gelu_out.data_ptr());
    char *workspace_ptr = reinterpret_cast<char *>(workspace.data_ptr());
    int workspace_size_chunk = workspaceSize / _stream_compute.size();

    if (A_scale_inverse.numel()) A_scale_inverse = A_scale_inverse[A_fp8_tensor];

    if (B_scale_inverse.numel()) B_scale_inverse = B_scale_inverse[B_fp8_tensor];

    at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
    CHECK_CUDA(cudaEventRecord(_start_compute, (cudaStream_t)stream_main));

    CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_send, _start_compute, 0));
    CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_recv, _start_compute, 0));
    for (size_t i = 0; i < _stream_compute.size(); i++) {
      CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_compute[i], _start_compute, 0));
    }
    if (_aggregate2) {
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
        if (do_gelu) {
          pre_gelu_out = torch::from_blob(pre_gelu_out_ptr + (send_chunk_id * aux_chunk_bytes),
                                          {n_chunk * 2, m}, pre_gelu_out.options());
        }
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
        }
      }
    } else {
      for (int i = 0; i < _tp_size; i++) {
        // Set the userbuffer id. Buffer under send is the input for the current
        // GEMM chunk The initial input chunk is stored _ubuf[rank]. This is to
        // have the AG output in all ranks to be contiguous after the ring
        // exchanges
        int send_chunk_id = (_tp_size + _tp_id - i) % _tp_size;
        int recv_chunk_id = (_tp_size + _tp_id - i - 1) % _tp_size;
        int send_offset = comm_bytes * send_chunk_id;
        int recv_offset = comm_bytes * recv_chunk_id;

        // GEMM
        torch::Tensor output_chunk = torch::from_blob(
            output_ptr + (send_chunk_id * output_chunk_bytes), {n_chunk, m}, D.options());
        if (do_gelu) {
          pre_gelu_out = torch::from_blob(pre_gelu_out_ptr + (send_chunk_id * aux_chunk_bytes),
                                          {n_chunk, m}, pre_gelu_out.options());
        }
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
        }
      }
    }
    for (size_t i = 0; i < _stream_compute.size(); i++) {
      CHECK_CUDA(cudaEventRecord(_stop_compute, (cudaStream_t)_stream_compute[i]));
      CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_compute, 0));
    }
    CHECK_CUDA(cudaEventRecord(_stop_send, (cudaStream_t)_stream_send));
    CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_send, 0));
    CHECK_CUDA(cudaEventRecord(_stop_recv, (cudaStream_t)_stream_recv));
    CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_recv, 0));
    at::cuda::setCurrentCUDAStream(stream_main);

    return D;
  }  // split_overlap_ag

  /*
  ** Split ReduceScatter + GEMM using P2P communication
  */
  void atomic_gemm_overlap_rs(at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
                              transformer_engine::DType A_type, bool transa, at::Tensor B,
                              at::Tensor B_scale_inverse, int64_t B_fp8_tensor,
                              transformer_engine::DType B_type, bool transb, at::Tensor D,
                              at::Tensor D_scale, transformer_engine::DType D_type,
                              at::Tensor D_amax, at::Tensor bias,
                              transformer_engine::DType bias_type, at::Tensor pre_gelu_out,
                              bool grad, at::Tensor workspace, size_t workspaceSize,
                              bool accumulate, bool use_split_accumulator, at::Tensor rs_output) {
    _ub_comm->use_ce = _use_ce;
    _ub_comm->sms = _num_comm_sm;
    _ub_comm->cga_size = _cga_size;

    // Get communication and GEMM input chunk sizes
    const int comm_bytes = _ubufs[0].numel() * _ubufs[0].element_size();

    // Get input and workspace data pointers
    char *workspace_ptr = reinterpret_cast<char *>(workspace.data_ptr());
    int *counter_ptr = reinterpret_cast<int *>(counter.data_ptr());
    int workspace_size_chunk = workspaceSize / _stream_compute.size();

    if (A_scale_inverse.numel()) A_scale_inverse = A_scale_inverse[A_fp8_tensor];

    if (B_scale_inverse.numel()) B_scale_inverse = B_scale_inverse[B_fp8_tensor];

    // Catch up the main stream
    at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
    CHECK_CUDA(cudaEventRecord(_start_compute, (cudaStream_t)stream_main));
    CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_recv, _start_compute, 0));

    // Atomic GEMM
    // Process GEMM chunks in the order that AG+GEMM places the output chunks.
    torch::Tensor workspace_chunk =
        torch::from_blob(workspace_ptr, {workspace_size_chunk}, workspace.options());
    te_atomic_gemm(A, A_scale_inverse, A_type, transa, B, B_scale_inverse, B_type, transb, _ubuf,
                   D_scale, D_type, D_amax, bias, bias_type, pre_gelu_out, grad, workspace_chunk,
                   workspace_size_chunk, accumulate, use_split_accumulator, _math_sms, 0, _tp_size,
                   true, counter);

    // P2P communication chunk
    for (int i = 1; i < _tp_size; i++) {
      int send_chunk_id = i - 1;
      int recv_chunk_id = send_chunk_id + _tp_size;
      int send_offset = comm_bytes * send_chunk_id;
      int recv_offset = comm_bytes * recv_chunk_id;
      int send_rank = (_tp_size + _tp_id - i) % _tp_size + _rank_round_tp;
      int recv_rank = (_tp_id + i) % _tp_size + _rank_round_tp;

      consumer(counter_ptr, send_chunk_id, (cudaStream_t)_stream_recv);
      userbuffers_send(_ub_reg, send_offset, _ub_reg, recv_offset, comm_bytes, _ub_comm, send_rank,
                       (cudaStream_t)_stream_recv);
      userbuffers_recv(_ub_reg, send_offset, _ub_reg, recv_offset, comm_bytes, _ub_comm, recv_rank,
                       (cudaStream_t)_stream_recv);
    }
    CHECK_CUDA(cudaEventRecord(_stop_recv, (cudaStream_t)_stream_recv));
    CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_recv, 0));

    // Reduce GEMM output chunks
    char *reduce_buf_ptr = reinterpret_cast<char *>(_ubufs[_tp_size - 1].data_ptr());
    if (_ubuf.element_size() == 1 && rs_output.element_size() == 2) {
      assert(_ubuf_scale_inv_initialized);
      float *d_scale_inv_ptr = reinterpret_cast<float *>(_ubuf_scale_inv.data_ptr());
      char *rs_output_ptr = reinterpret_cast<char *>(rs_output.data_ptr());
      reduce_fp8_in_bf16_out<__nv_fp8_e4m3>(reduce_buf_ptr, rs_output_ptr, d_scale_inv_ptr,
                                            _tp_size, _ubufs[0].numel(), (cudaStream_t)stream_main);
    } else {
      torch::Tensor reduce_buf = torch::from_blob(
          reduce_buf_ptr, {_tp_size, _ubufs[0].size(0), _ubufs[0].size(1)}, _ubuf.options());
      torch::sum_out(rs_output, reduce_buf, 0);
    }
  }

  /*
  ** Split ReduceScatter + GEMM using P2P communication
  */
  void split_overlap_rs(at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
                        transformer_engine::DType A_type, bool transa, at::Tensor B,
                        at::Tensor B_scale_inverse, int64_t B_fp8_tensor,
                        transformer_engine::DType B_type, bool transb, at::Tensor D,
                        at::Tensor D_scale, transformer_engine::DType D_type, at::Tensor D_amax,
                        at::Tensor bias, transformer_engine::DType bias_type,
                        at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
                        size_t workspaceSize, bool accumulate, bool use_split_accumulator,
                        at::Tensor rs_output) {
    _ub_comm->use_ce = _use_ce;
    _ub_comm->sms = _num_comm_sm;
    _ub_comm->cga_size = _cga_size;
    int k = A.size(1);
    int n = B.size(0);

    // Get communication and GEMM input chunk sizes
    int n_chunk = n / _tp_size;
    const int comm_bytes = _ubufs[0].numel() * _ubufs[0].element_size();
    const int input_b_chunk_bytes = n_chunk * k * B.element_size();

    // Get input and workspace data pointers
    char *input_b_ptr = reinterpret_cast<char *>(B.data_ptr());
    char *workspace_ptr = reinterpret_cast<char *>(workspace.data_ptr());
    int workspace_size_chunk = workspaceSize / _stream_compute.size();

    if (A_scale_inverse.numel()) A_scale_inverse = A_scale_inverse[A_fp8_tensor];

    if (B_scale_inverse.numel()) B_scale_inverse = B_scale_inverse[B_fp8_tensor];

    // Catch up the main stream
    at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
    CHECK_CUDA(cudaEventRecord(_start_compute, (cudaStream_t)stream_main));
    CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_send, _start_compute, 0));
    CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_recv, _start_compute, 0));
    for (size_t i = 0; i < _stream_compute.size(); i++) {
      CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_compute[i], _start_compute, 0));
    }

    // GEMM and send/recv chunks
    for (int i = 0; i < _tp_size; i++) {
      // GEMM chunk
      int input_b_chunk_id = (_tp_id + i + 1) % _tp_size;
      char *input_b_chunk_ptr = input_b_ptr + (input_b_chunk_id * input_b_chunk_bytes);
      torch::Tensor input_b_chunk = torch::from_blob(input_b_chunk_ptr, {n_chunk, k}, B.options());
      // Store the last GEMM chunk output to the recieve buffer.
      torch::Tensor workspace_chunk =
          torch::from_blob(workspace_ptr + (i % _stream_compute.size()) * workspace_size_chunk,
                           {workspace_size_chunk}, workspace.options());
      if (i == _tp_size - 1) {
        at::cuda::setCurrentCUDAStream(stream_main);
      } else {
        at::cuda::setCurrentCUDAStream(_stream_compute[i % _stream_compute.size()]);
      }
      te_gemm(A, A_scale_inverse, A_type, transa, input_b_chunk, B_scale_inverse, B_type, transb,
              _ubufs[i], D_scale, D_type, D_amax, bias, bias_type, pre_gelu_out, grad,
              workspace_chunk, workspace_size_chunk, accumulate, use_split_accumulator, _math_sms);

      if (i > 0) {
        // P2P communication chunk
        int send_offset = comm_bytes * (i - 1);
        int recv_offset = comm_bytes * (i - 1 + _tp_size);
        int send_rank = (_tp_id + i) % _tp_size + _rank_round_tp;
        int recv_rank = (_tp_size + _tp_id - i) % _tp_size + _rank_round_tp;
        CHECK_CUDA(cudaEventRecord(
            _start_comm, (cudaStream_t)_stream_compute[(i - 1) % _stream_compute.size()]));
        CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_send, _start_comm, 0));
        CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_recv, _start_comm, 0));
        userbuffers_send(_ub_reg, send_offset, _ub_reg, recv_offset, comm_bytes, _ub_comm,
                         send_rank, (cudaStream_t)_stream_send);
        userbuffers_recv(_ub_reg, send_offset, _ub_reg, recv_offset, comm_bytes, _ub_comm,
                         recv_rank, (cudaStream_t)_stream_recv);
      }
    }
    CHECK_CUDA(cudaEventRecord(_stop_recv, (cudaStream_t)_stream_recv));
    CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_recv, 0));

    // Reduce GEMM output chunks
    char *reduce_buf_ptr = reinterpret_cast<char *>(_ubufs[_tp_size - 1].data_ptr());
    if (_ubuf.element_size() == 1 && rs_output.element_size() == 2) {
      assert(_ubuf_scale_inv_initialized);
      float *d_scale_inv_ptr = reinterpret_cast<float *>(_ubuf_scale_inv.data_ptr());
      char *rs_output_ptr = reinterpret_cast<char *>(rs_output.data_ptr());
      reduce_fp8_in_bf16_out<__nv_fp8_e4m3>(reduce_buf_ptr, rs_output_ptr, d_scale_inv_ptr,
                                            _tp_size, _ubufs[0].numel(), (cudaStream_t)stream_main);
    } else {
      torch::Tensor reduce_buf = torch::from_blob(
          reduce_buf_ptr, {_tp_size, _ubufs[0].size(0), _ubufs[0].size(1)}, _ubuf.options());
      torch::sum_out(rs_output, reduce_buf, 0);
    }
    for (size_t i = 0; i < _stream_compute.size(); i++) {
      CHECK_CUDA(cudaEventRecord(_stop_compute, (cudaStream_t)_stream_compute[i]));
      CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_compute, 0));
    }
    CHECK_CUDA(cudaEventRecord(_stop_send, (cudaStream_t)_stream_send));
    CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_send, 0));
  }

  /*
  ** Copy input to _ubufs[0]
  */
  void copy_input_to_ubuf(torch::Tensor input, bool chunk) {
    at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
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
      ubuf_wt_ptr += _ubuf.numel() / _tp_size * _self_chunk_id * _ubuf.element_size();
    int output_c_dim0 = (_comm_type == COMM_TYPE::AG) ? _ubuf.size(0) : _ubuf.size(0) / _tp_size;
    int output_c_dim1 = _ubuf.size(1);
    return torch::from_blob(ubuf_wt_ptr, {output_c_dim0, output_c_dim1}, _ubuf.options());
  }

  void set_ubuf_scale_inv(const torch::Tensor &scale_inv) {
    _ubuf_scale_inv = scale_inv;
    _ubuf_scale_inv_initialized = true;
  }

  bool is_fp8_ubuf() { return (_ubuf.element_size() == 1); }
  bool is_atomic_gemm() { return _atomic_gemm; }
  bool is_p2p_overlap() { return true; }
};  // UbufP2PCommOverlap

}  // namespace ubuf

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_COMM_GEMM_OVERLAP_H_
