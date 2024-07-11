/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// Standard library includes
#include <cstring>
#include <cassert>

// TE/common includes
#include <transformer_engine/gemm.h>
#include <transformer_engine/comm_gemm_overlap.h>
#include "../util/logging.h"
#include "../util/cuda_driver.h"
#include "../util/system.h"

#define HALF_BYTES 2
#define UB_MAX_SM 32

namespace transformer_engine {

bool device_supports_multicast() {
  int dev, supports_multicast;
  CUdevice cudev;

  NVTE_CHECK_CUDA(cudaGetDevice(&dev));
  NVTE_CALL_CHECK_CUDA_DRIVER(cuDeviceGet, &cudev, dev);
  NVTE_CALL_CHECK_CUDA_DRIVER(
      cuDeviceGetAttribute, &supports_multicast, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, cudev);

  return static_cast<bool>(supports_multicast);
}

namespace common {

/***************************************************************************************************
** CommGemmOverlapBase -- Base class for comm+GEMM overlap algorithms
***************************************************************************************************/

CommGemmOverlapBase::CommGemmOverlapBase(
    int worldrank, int worldsize, int localrank, int localsize, int nodeid, int numnodes,
    int tp_size, int num_splits, int num_max_streams, int cga_size, int num_comm_sms,
    bool set_sm_margin, bool use_ce, bool atomic_gemm, const char *name,
    std::function<void(void *, size_t, void *, size_t, char *)> allgather_handle,
    std::function<void(void *, size_t, int, char *)> bcast_handle,
    std::function<void(char *)> barrier_handle) {
  // Initialize the UB communicator
  snprintf(_name, sizeof(_name), "%s", name);
  if (!_comm_created) {
#ifdef UB_MPI_BOOTSTRAP
    create_communicator_grouped2_mpi(&_ub_comm, 1, 1, tp_size, 1);
#else
    create_communicator_grouped2(&_ub_comm, worldrank, worldsize, localrank, localsize, nodeid,
                                  numnodes, allgather_handle, barrier_handle, 1, 1, tp_size, 1);
#endif
    if (worldrank == 0) {
      printf("[%s] Communicator initialized\n", _name);
    }
    _comm_created = true;
  }

  _atomic_gemm = atomic_gemm;
  _tp_size = tp_size;
  _tp_id = worldrank % _tp_size;
  _num_splits = num_splits;
  _cga_size = cga_size;
  _use_ce = static_cast<int>(use_ce);

  for (int i = 0; i < std::min(_num_splits, num_max_streams); i++) {
    cudaStream_t new_stream;
    NVTE_CHECK_CUDA(cudaStreamCreateWithPriority(&new_stream, cudaStreamNonBlocking, -1));
    _stream_compute.push_back(new_stream);
  }

  cudaDeviceProp prop;
  NVTE_CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  _comm_sms = (set_sm_margin) ? num_comm_sms : 0;
  _math_sms = prop.multiProcessorCount - _comm_sms - getenv<int>("NVTE_EXT_MARGIN_SM", 0);

  NVTE_CHECK_CUDA(cudaEventCreateWithFlags(&_start_compute, 0));
  NVTE_CHECK_CUDA(cudaEventCreateWithFlags(&_stop_compute, 0));
  NVTE_CHECK_CUDA(cudaEventCreateWithFlags(&_start_comm, 0));
  NVTE_CHECK_CUDA(cudaEventCreateWithFlags(&_stop_comm, 0));
}

CommGemmOverlapBase::~CommGemmOverlapBase() {
  cudaEventDestroy(_stop_comm);
  cudaEventDestroy(_start_comm);
  cudaEventDestroy(_stop_compute);
  cudaEventDestroy(_start_compute);
  for (size_t i = 0; i < _stream_compute.size(); i++) {
    cudaStreamDestroy(_stream_compute[i]);
  }
  if (_comm_created) {
#ifdef UB_MPI_BOOTSTRAP
    destroy_communicator_mpi(_ub_comm);
#else
    destroy_communicator(_ub_comm);
#endif
    _comm_created = false;
  }
}

void CommGemmOverlapBase::register_gpu_buffer(void **gpuptr, size_t bytes, bool alloc) {
  NVTE_CHECK(_comm_created,
              "[%s] Communicator must be initialized before buffer registration.", _name);
  NVTE_CHECK(!_buffer_registered, "[%s] GPU buffer is already registered.", _name);
  _ub_reg = register_user_buffer_collective(gpuptr, bytes, _ub_comm, alloc);
  _buffer_registered = true;
  if (_tp_id == 0) {
    printf("[%s] Registered buffer %d\n", _name, _ub_reg);
  }
}

/***************************************************************************************************
** CommGemmOverlap -- Collective (pipelined) comm+GEMM overlap algorithms
***************************************************************************************************/

CommGemmOverlap::CommGemmOverlap(
    int worldrank, int worldsize, int localrank, int localsize, int nodeid,
    int numnodes, int tp_size, int num_splits, int num_max_streams, int cga_size,
    int num_comm_sms, bool set_sm_margin, bool use_ce, bool atomic_gemm,
    std::function<void(void *, size_t, void *, size_t, char *)> allgather_handle,
    std::function<void(void *, size_t, int, char *)> bcast_handle,
    std::function<void(char *)> barrier_handle)
    : CommGemmOverlapBase(worldrank, worldsize, localrank, localsize, nodeid, numnodes, tp_size,
                          num_splits, num_max_streams, cga_size, num_comm_sms, set_sm_margin,
                          use_ce, atomic_gemm, "CommOverlap", allgather_handle, bcast_handle,
                          barrier_handle) {
  if (_atomic_gemm) {
    _rs_kernel_type = getenv<int>("NVTE_RS_STRIDED_ATOMIC", 1);
    NVTE_CHECK(0 <= _rs_kernel_type && _rs_kernel_type < 3,
                "[%s] Invalid choice for NVTE_RS_STRIDED_ATOMIC", _name);
    if (worldrank == 0 && _rs_kernel_type == 0) {
      NVTE_ERROR("[%s] Non-atomic RS overlap (BF16) is incompatible with Atomic GEMM (FP8).",
                 _name);
    } else if (worldrank == 0 && _rs_kernel_type == 1) {
      printf("[%s] NVTE_RS_STRIDED_ATOMIC=1 (atomic GEMM + atomic RS)\n", _name);
    } else if (worldrank == 0) {
      printf("[%s] NVTE_RS_STRIDED_ATOMIC=2 (atomic GEMM + multi-atomic RS)\n", _name);
    }
  }

  NVTE_CHECK_CUDA(cudaStreamCreateWithPriority(&_stream_comm, cudaStreamNonBlocking, -1));
  NVTE_CHECK_CUDA(cudaEventCreateWithFlags(&_start_d2dcopy, 0));
}

CommGemmOverlap::~CommGemmOverlap() {
  cudaEventDestroy(_start_d2dcopy);
  cudaStreamDestroy(_stream_comm);
}

/*
  Bulk GEMM + All-Gather/Reduce-Scatter
*/
void CommGemmOverlap::bulk_gemm_overlap(
    cudaStream_t stream_main, const TensorWrapper &A, bool A_trans, const TensorWrapper &B,
    bool B_trans, const TensorWrapper &bias, const TensorWrapper &D,
    const TensorWrapper &pre_gelu_out, const TensorWrapper &ubuf, const TensorWrapper &rs_output,
    const TensorWrapper &workspace, bool grad, bool accumulate, bool use_split_accumulator,
    NVTE_Comm_Overlap_Type comm_type) {
  int ori_sm = _ub_comm->sms;
  _ub_comm->use_ce = _use_ce;
  _ub_comm->sms = _comm_sms;
  _ub_comm->cga_size = _cga_size;

  // Get the current userbuf offset
  char *ubuf_wt_ptr = reinterpret_cast<char *>(ubuf.dptr());
  int comm_elements = (ubuf.numel() / 2) * ubuf.element_size();
  if (comm_type == NVTE_Comm_Overlap_Type::REDUCE_SCATTER) {
    ubuf_wt_ptr += (ubuf.numel() * ubuf.element_size() / _tp_size) * _tp_id;
  }

  // Catch up the default torch stream
  NVTE_CHECK_CUDA(cudaEventRecord(_start_comm, stream_main));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_comm, _start_comm, 0));

  // Communication: AG and RS
  if (comm_type == NVTE_Comm_Overlap_Type::ALL_GATHER) {
    allgather2_userbuff_inplace(_ub_reg, 0, comm_elements, _ub_comm, _stream_comm);
  } else if (comm_type == NVTE_Comm_Overlap_Type::REDUCE_SCATTER) {
    if (ubuf.element_size() == 1) {
      assert(rs_output.numel() == ubuf.numel() / _tp_size);
      assert(rs_output.size(0) == ubuf.size(0) / _tp_size);
      assert(rs_output.element_size() == 2);
      char *rs_output_ptr = reinterpret_cast<char *>(rs_output.dptr());
      reducescatter2_userbuff_fp8<__nv_fp8_e5m2>(rs_output_ptr, ubuf.scale_inv(), _ub_reg, 0,
                                                  comm_elements, _ub_comm, _stream_comm);
    } else {
      reducescatter2_userbuff_inplace(_ub_reg, 0, comm_elements, _ub_comm, _stream_comm);
    }
  } else {
    NVTE_ERROR("[%s] Unsupported communication type.", _name);
  }

  assert(pre_gelu_out.numel() == 0);
  nvte_cublas_gemm(A.data(), B.data(), D.data(), bias.data(), pre_gelu_out.data(), A_trans,
                    B_trans, grad, workspace.data(), accumulate, use_split_accumulator, _math_sms,
                    stream_main);

  _ub_comm->sms = ori_sm;
  NVTE_CHECK_CUDA(cudaEventRecord(_stop_comm, _stream_comm));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_comm, 0));
}  // CommGemmOverlap::bulk_gemm_overlap

/*
  Atomic FPROP GEMM + ReduceScatter
*/
void CommGemmOverlap::atomic_gemm_overlap_rs(
    cudaStream_t stream_main, const TensorWrapper &A, bool A_trans, const TensorWrapper &B,
    bool B_trans, const TensorWrapper &bias, const TensorWrapper &D,
    const TensorWrapper &pre_gelu_out, const TensorWrapper &ubuf, const TensorWrapper &counters,
    const TensorWrapper &rs_output, const TensorWrapper &workspace, bool grad, bool accumulate,
    bool use_split_accumulator) {
  int ori_sms = _ub_comm->sms;
  _ub_comm->use_ce = _use_ce;
  _ub_comm->sms = _comm_sms;
  _ub_comm->cga_size = _cga_size;

  // Get GEMM dimensions
  size_t m = A.size(0);
  size_t n = B.size(0);
  size_t m_chunk = m / _num_splits;
  size_t workspace_size_chunk = workspace.numel() / _stream_compute.size();

  char *workspace_ptr = reinterpret_cast<char *>(workspace.dptr());
  int *counter_ptr = reinterpret_cast<int *>(counters.dptr());
  char *rs_output_ptr = reinterpret_cast<char *>(rs_output.dptr());

  assert(pre_gelu_out.numel() == 0);
  if (_rs_kernel_type == 0)
    assert(ubuf.element_size() == 2);

  // Reset counters to "not ready" (1)
  reset_counters(counter_ptr, _num_splits, -1, stream_main);

  // Catch up the default torch stream
  NVTE_CHECK_CUDA(cudaEventRecord(_start_compute, stream_main));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_comm, _start_compute, 0));

  TensorWrapper workspace_chunk = TensorWrapper(reinterpret_cast<void *>(workspace_ptr),
                                                {workspace_size_chunk}, workspace.dtype());
  nvte_cublas_atomic_gemm(A.data(), B.data(), ubuf.data(), bias.data(), pre_gelu_out.data(),
                          A_trans, B_trans, grad, workspace_chunk.data(), accumulate,
                          use_split_accumulator, _math_sms, _num_splits, 0, true, counters.data(),
                          stream_main);

  // NOTE: Atomic GEMM always requires FP8 inputs but it can output BF16 result. The RS output is
  //       always BF16 regardless of what data type the userbuffer tensor has.
  for (int i = 0; i < _num_splits; i++) {
    if (_rs_kernel_type == 1) {
      if (i == _num_splits - 1) {
        _ub_comm->sms = UB_MAX_SM;
      }
      if (ubuf.element_size() == 1) {
        // TODO (Alp): RS kernels for FP8 GEMM output is numerically incorrect.
        float *d_scale_inv_ptr = reinterpret_cast<float *>(ubuf.scale_inv());
        reducescatter2_userbuff_strided_atomic_fp8<__nv_fp8_e4m3>(
            rs_output_ptr, d_scale_inv_ptr, _ub_reg, i * m_chunk, m_chunk, n, m, m, _num_splits,
            &counter_ptr[i], _ub_comm, (cudaStream_t)_stream_comm);
      } else {
        reducescatter2_userbuff_strided_atomic(rs_output_ptr, _ub_reg, i * m_chunk, m_chunk, n, m,
                                                _num_splits, &counter_ptr[i], _ub_comm,
                                                (cudaStream_t)_stream_comm);
      }
    } else if (_rs_kernel_type == 2) {
      if (ubuf.element_size() == 1) {
        // TODO (Alp): RS kernels for FP8 GEMM output is numerically incorrect.
        float *d_scale_inv_ptr = reinterpret_cast<float *>(ubuf.scale_inv());
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
      reducescatter2_userbuff_strided(rs_output_ptr, _ub_reg, i * m_chunk, m_chunk, n, m,
                                      _ub_comm, (cudaStream_t)_stream_comm);
    }

    rs_output_ptr += m_chunk * rs_output.element_size();
  }

  _ub_comm->sms = ori_sms;
  NVTE_CHECK_CUDA(cudaEventRecord(_stop_comm, _stream_comm));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_comm, 0));
}  // CommGemmOverlap::atomic_gemm_overlap_rs

/*
  Split FPROP GEMM + ReduceScatter
*/
void CommGemmOverlap::split_gemm_overlap_rs(
    cudaStream_t stream_main, const TensorWrapper &A, bool A_trans, const TensorWrapper &B,
    bool B_trans, const TensorWrapper &bias, const TensorWrapper &D,
    const TensorWrapper &pre_gelu_out, const TensorWrapper &ubuf, const TensorWrapper &rs_output,
    const TensorWrapper &workspace, bool grad, bool accumulate, bool use_split_accumulator,
    bool gemm_overlap) {
  int ori_sms = _ub_comm->sms;
  _ub_comm->use_ce = _use_ce;
  _ub_comm->sms = _comm_sms;
  _ub_comm->cga_size = _cga_size;

  size_t m = A.size(0);
  size_t k = A.size(1);
  size_t n = B.size(0);
  size_t m_chunk = m / _num_splits;
  size_t input_a_chunk_size = m_chunk * k;
  size_t output_chunk_size = n * m_chunk;
  size_t workspace_size_chunk = workspace.numel() / _stream_compute.size();

  // Get input, output, and workspace data pointers
  char *input_a_chunk_ptr = reinterpret_cast<char *>(A.dptr());
  char *output_buf_chunk_ptr = reinterpret_cast<char *>(ubuf.dptr());
  char *workspace_ptr = reinterpret_cast<char *>(workspace.dptr());

  char *rs_output_ptr = reinterpret_cast<char *>(rs_output.dptr());

  // Catch up the default torch stream
  NVTE_CHECK_CUDA(cudaEventRecord(_start_compute, stream_main));
  for (size_t i = 0; i < _stream_compute.size(); i++) {
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_compute[i], _start_compute, 0));
  }
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_comm, _start_compute, 0));

  assert(pre_gelu_out.numel() == 0);

  if (gemm_overlap) {
    TensorWrapper input_a_chunk =
        TensorWrapper(reinterpret_cast<void *>(input_a_chunk_ptr), {m_chunk, k}, A.dtype(),
                      A.amax(), A.scale(), A.scale_inv());
    TensorWrapper output_chunk =
        TensorWrapper(reinterpret_cast<void *>(output_buf_chunk_ptr), {n, m_chunk}, ubuf.dtype(),
                      ubuf.amax(), ubuf.scale(), ubuf.scale_inv());
    TensorWrapper workspace_chunk = TensorWrapper(reinterpret_cast<void *>(workspace_ptr),
                                                  {workspace_size_chunk}, workspace.dtype());

    nvte_cublas_gemm(input_a_chunk.data(), B.data(), output_chunk.data(), bias.data(),
                      pre_gelu_out.data(), A_trans, B_trans, grad, workspace_chunk.data(),
                      accumulate, use_split_accumulator, _math_sms, _stream_compute[0]);

    for (int i = 1; i < _num_splits; i++) {
      input_a_chunk_ptr += input_a_chunk_size * B.element_size();
      output_buf_chunk_ptr += output_chunk_size * ubuf.element_size();

      TensorWrapper input_a_chunk =
          TensorWrapper(reinterpret_cast<void *>(input_a_chunk_ptr), {m_chunk, k}, A.dtype(),
                        A.amax(), A.scale(), A.scale_inv());
      TensorWrapper output_chunk =
          TensorWrapper(reinterpret_cast<void *>(output_buf_chunk_ptr), {n, m_chunk},
                        ubuf.dtype(), ubuf.amax(), ubuf.scale(), ubuf.scale_inv());
      TensorWrapper workspace_chunk =
          TensorWrapper(reinterpret_cast<void *>(workspace_ptr + (i % _stream_compute.size()) *
                                                                      workspace_size_chunk),
                        {workspace_size_chunk}, workspace.dtype());

      nvte_cublas_gemm(input_a_chunk.data(), B.data(), output_chunk.data(), bias.data(),
                        pre_gelu_out.data(), A_trans, B_trans, grad, workspace_chunk.data(),
                        accumulate, use_split_accumulator, _math_sms,
                        _stream_compute[i % _stream_compute.size()]);

      NVTE_CHECK_CUDA(
          cudaEventRecord(_start_comm, _stream_compute[(i - 1) % _stream_compute.size()]));
      NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_comm, _start_comm, 0));

      // Communication chunk
      if (ubuf.element_size() == 1) {
        reducescatter2_userbuff_stridedoutput_fp8<__nv_fp8_e4m3>(
            rs_output_ptr, ubuf.scale_inv(), _ub_reg, (i - 1) * output_chunk_size, m_chunk, n, m,
            _ub_comm, _stream_comm);
      } else {
        reducescatter2_userbuff_stridedoutput(rs_output_ptr, _ub_reg, (i - 1) * output_chunk_size,
                                              m_chunk, n, m, _ub_comm, _stream_comm);
      }

      rs_output_ptr += m_chunk * rs_output.element_size();
    }
    int last_compute_stream_id =
        (_num_splits + _stream_compute.size() - 1) % _stream_compute.size();
    NVTE_CHECK_CUDA(cudaEventRecord(_start_comm, _stream_compute[last_compute_stream_id]));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_comm, _start_comm, 0));

    // Last communication chunk with max SM
    _ub_comm->sms = UB_MAX_SM;
    if (ubuf.element_size() == 1) {
      reducescatter2_userbuff_stridedoutput_fp8<__nv_fp8_e4m3>(
          rs_output_ptr, ubuf.scale_inv(), _ub_reg, (_num_splits - 1) * output_chunk_size,
          m_chunk, n, m, _ub_comm, _stream_comm);
    } else {
      reducescatter2_userbuff_stridedoutput(rs_output_ptr, _ub_reg,
                                            (_num_splits - 1) * output_chunk_size, m_chunk, n, m,
                                            _ub_comm, _stream_comm);
    }
  } else {
    for (int i = 0; i < _num_splits; i++) {
      TensorWrapper input_a_chunk =
          TensorWrapper(reinterpret_cast<void *>(input_a_chunk_ptr), {m_chunk, k}, A.dtype(),
                        A.amax(), A.scale(), A.scale_inv());
      TensorWrapper output_chunk =
          TensorWrapper(reinterpret_cast<void *>(output_buf_chunk_ptr), {n, m_chunk},
                        ubuf.dtype(), ubuf.amax(), ubuf.scale(), ubuf.scale_inv());
      TensorWrapper workspace_chunk =
          TensorWrapper(reinterpret_cast<void *>(workspace_ptr + (i % _stream_compute.size()) *
                                                                      workspace_size_chunk),
                        {workspace_size_chunk}, workspace.dtype());

      nvte_cublas_gemm(input_a_chunk.data(), B.data(), output_chunk.data(), bias.data(),
                        pre_gelu_out.data(), A_trans, B_trans, grad, workspace_chunk.data(),
                        accumulate, use_split_accumulator, _math_sms,
                        _stream_compute[i % _stream_compute.size()]);

      NVTE_CHECK_CUDA(cudaEventRecord(_start_comm, _stream_compute[i % _stream_compute.size()]));
      NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_comm, _start_comm, 0));

      // Communication chunk. Uses MAX_SM at the last chunk
      if (i == _num_splits - 1) {
        _ub_comm->sms = UB_MAX_SM;
      }
      if (ubuf.element_size() == 1) {
        reducescatter2_userbuff_stridedoutput_fp8<__nv_fp8_e4m3>(
            rs_output_ptr, ubuf.scale_inv(), _ub_reg, i * output_chunk_size, m_chunk, n, m,
            _ub_comm, _stream_comm);
      } else {
        reducescatter2_userbuff_stridedoutput(rs_output_ptr, _ub_reg, i * output_chunk_size,
                                              m_chunk, n, m, _ub_comm, _stream_comm);
      }
      rs_output_ptr += m_chunk * rs_output.element_size();
      input_a_chunk_ptr += input_a_chunk_size * B.element_size();
      output_buf_chunk_ptr += output_chunk_size * ubuf.element_size();
    }
  }

  for (size_t i = 0; i < _stream_compute.size(); i++) {
    NVTE_CHECK_CUDA(cudaEventRecord(_stop_compute, _stream_compute[i]));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_compute, 0));
  }

  _ub_comm->sms = ori_sms;
  NVTE_CHECK_CUDA(cudaEventRecord(_stop_comm, _stream_comm));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_comm, 0));
}  // CommGemmOverlap::split_gemm_overlap_rs

/***************************************************************************************************
** CommGemmOverlapBaseP2P -- Point-2-Point (ring-exchange) comm+GEMM overlap algorithms
***************************************************************************************************/

CommGemmOverlapP2P::CommGemmOverlapP2P(
    int worldrank, int worldsize, int localrank, int localsize, int nodeid, int numnodes,
    int tp_size, int num_max_streams, int cga_size, int num_comm_sms, bool set_sm_margin,
    bool use_ce, bool atomic_gemm, bool aggregate, bool is_reduce_scatter,
    std::function<void(void *, size_t, void *, size_t, char *)> allgather_handle,
    std::function<void(void *, size_t, int, char *)> bcast_handle,
    std::function<void(char *)> barrier_handle)
    : CommGemmOverlapBase(worldrank, worldsize, localrank, localsize, nodeid, numnodes, tp_size,
                          tp_size, num_max_streams, cga_size, num_comm_sms, set_sm_margin, use_ce,
                          atomic_gemm, "CommOverlapP2P", allgather_handle, bcast_handle,
                          barrier_handle) {
  _is_p2p = true;
  _aggregate = aggregate;
  _is_reduce_scatter = is_reduce_scatter;
  _rank_round_tp = (worldrank / localsize) * localsize;
  _next_rank = (localsize + worldrank + 1) % localsize + _rank_round_tp;
  _prev_rank = (localsize + worldrank - 1) % localsize + _rank_round_tp;
  _self_chunk_id = (_atomic_gemm && !_is_reduce_scatter) ? 0 : _tp_id;
  _num_ubuf_chunks = (_is_reduce_scatter) ? static_cast<int>(localsize * 2 - 1) : localsize;

  if (_atomic_gemm) {
    if (!_is_reduce_scatter) {
      _ag_sendrecv_multiatomic = getenv<bool>("NVTE_AG_P2P_MULTI_ATOMIC");
      if (worldrank == 0 && _ag_sendrecv_multiatomic) {
        printf("[%s] NVTE_AG_P2P_MULTI_ATOMIC=1 (multi-atomic AG + atomic GEMM)\n", _name);
      }
    }
  }

  NVTE_CHECK_CUDA(cudaStreamCreateWithPriority(&_stream_send, cudaStreamNonBlocking, -1));
  NVTE_CHECK_CUDA(cudaStreamCreateWithPriority(&_stream_recv, cudaStreamNonBlocking, -1));
  NVTE_CHECK_CUDA(cudaEventCreateWithFlags(&_stop_send, 0));
  NVTE_CHECK_CUDA(cudaEventCreateWithFlags(&_stop_recv, 0));
}

CommGemmOverlapP2P::~CommGemmOverlapP2P() {
  cudaEventDestroy(_stop_recv);
  cudaEventDestroy(_stop_send);
  cudaStreamDestroy(_stream_recv);
  cudaStreamDestroy(_stream_send);
}

/*
  Split AllGather + AtomicGEMM using P2P communication
*/
void CommGemmOverlapP2P::atomic_gemm_overlap_ag(
    cudaStream_t stream_main, const TensorWrapper &A, bool A_trans, const TensorWrapper &B,
    bool B_trans, const TensorWrapper &bias, const TensorWrapper &D,
    const TensorWrapper &pre_gelu_out, const TensorWrapper &ubuf,
    const std::vector<TensorWrapper> &ubufs, const TensorWrapper &counters,
    const TensorWrapper &B_copy, const TensorWrapper &D_buffer, const TensorWrapper &workspace,
    bool grad, bool accumulate, bool use_split_accumulator) {
  int ori_sm = _ub_comm->sms;
  _ub_comm->use_ce = _use_ce;
  _ub_comm->sms = _comm_sms;
  _ub_comm->cga_size = _cga_size;

  // Get GEMM dimensions between TN and NN input layouts
  const size_t m = (A_trans) ? A.size(0) : A.size(1);
  const size_t n = ubuf.size(0);
  const size_t n_chunk = n / _tp_size;

  // Get communication and GEMM output chunk sizes
  const int comm_bytes = ubufs[0].numel() * ubufs[0].element_size();

  // Get output and workspace data pointers
  char *workspace_ptr = reinterpret_cast<char *>(workspace.dptr());
  int *counter_ptr = reinterpret_cast<int *>(counters.dptr());
  size_t workspace_size_chunk = workspace.numel() / _stream_compute.size();

  // Reset atomic counters
  reset_counters(counter_ptr, _tp_size, _self_chunk_id, stream_main);

  assert(pre_gelu_out.numel() == 0);

  // Catch up the default torch stream
  NVTE_CHECK_CUDA(cudaEventRecord(_start_compute, stream_main));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_send, _start_compute, 0));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_recv, _start_compute, 0));

  TensorWrapper workspace_chunk = TensorWrapper(reinterpret_cast<void *>(workspace_ptr),
                                                {workspace_size_chunk}, workspace.dtype());

  for (int i = 0; i < _tp_size - 1; i++) {
    if (_ag_sendrecv_multiatomic) {
      if (i == 0) {
        userbuffers_sendrecv_multiatomic(_ub_reg, _ub_reg, comm_bytes, comm_bytes, comm_bytes,
                                          _ub_comm, _next_rank, _prev_rank, _tp_size, counter_ptr,
                                          true, _stream_recv);
      }
    } else {
      // Set the userbuffer id. Buffer under send is the input for the current GEMM chunk The
      // initial input chunk is stored ubuf[rank]. This is to have the AG output in all ranks to
      // be contiguous after the ring exchanges.
      int send_chunk_id = i;
      int recv_chunk_id = i + 1;
      int send_offset = comm_bytes * send_chunk_id;
      int recv_offset = comm_bytes * recv_chunk_id;

      userbuffers_send(_ub_reg, send_offset, _ub_reg, recv_offset, comm_bytes, _ub_comm,
                        _next_rank, _stream_recv);
      userbuffers_recv(_ub_reg, send_offset, _ub_reg, recv_offset, comm_bytes, _ub_comm,
                        _prev_rank, _stream_recv);
      producer(counter_ptr, recv_chunk_id, _stream_recv);
    }

    if (i == 0) {
      nvte_cublas_atomic_gemm(A.data(), ubuf.data(), D.data(), bias.data(), pre_gelu_out.data(),
                              A_trans, B_trans, grad, workspace_chunk.data(), accumulate,
                              use_split_accumulator, _math_sms, 0, _tp_size, false,
                              counters.data(), stream_main);
    }
  }

  // Store the input activation for backprop
  if (B_copy.numel() > 0) {
    assert(B_copy.numel() == ubufs[_self_chunk_id].numel());
    assert(B_copy.element_size() == ubufs[_self_chunk_id].element_size());
    NVTE_CHECK_CUDA(
        cudaMemcpyAsync(B_copy.dptr(), ubufs[_self_chunk_id].dptr(),
                        ubufs[_self_chunk_id].numel() * ubufs[_self_chunk_id].element_size(),
                        cudaMemcpyDeviceToDevice, _stream_send));
    NVTE_CHECK_CUDA(cudaEventRecord(_stop_send, _stream_send));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_send, 0));
  }

  // Copy the first GEMM output chunk to the end chunk position of D_buffer
  char *src_ptr = reinterpret_cast<char *>(D_buffer.dptr());
  NVTE_CHECK_CUDA(cudaMemcpyAsync(src_ptr + (D.numel() * D.element_size()), src_ptr,
                                  n_chunk * m * D.element_size(), cudaMemcpyDeviceToDevice,
                                  stream_main));

  _ub_comm->sms = ori_sm;
}  // CommGemmOverlapP2P::atomic_gemm_overlap_ag

/*
  Split AllGather + GEMM using P2P communication
*/
void CommGemmOverlapP2P::split_gemm_overlap_ag(
    cudaStream_t stream_main, const TensorWrapper &A, bool A_trans, const TensorWrapper &B,
    bool B_trans, const TensorWrapper &bias, const TensorWrapper &D,
    const TensorWrapper &pre_gelu_out, const std::vector<TensorWrapper> &ubufs,
    const TensorWrapper &B_copy, const TensorWrapper &workspace, bool grad, bool accumulate,
    bool use_split_accumulator) {
  int ori_sm = _ub_comm->sms;
  _ub_comm->use_ce = _use_ce;
  _ub_comm->sms = _comm_sms;
  _ub_comm->cga_size = _cga_size;

  // Get GEMM dimensions between TN and NN input layouts
  const size_t m = (A_trans) ? A.size(0) : A.size(1);
  const size_t k = (A_trans) ? A.size(1) : A.size(0);
  const size_t n_chunk = ubufs[0].size(0);

  // Get communication and GEMM output chunk sizes
  const int comm_bytes = ubufs[0].numel() * ubufs[0].element_size();
  const bool do_gelu = pre_gelu_out.numel() > 0;
  const int output_chunk_bytes = (n_chunk * m) * D.element_size();
  const int aux_chunk_bytes = do_gelu ? (n_chunk * m) * pre_gelu_out.element_size() : 0;

  // Get output and workspace data pointers
  char *output_ptr = reinterpret_cast<char *>(D.dptr());
  char *pre_gelu_out_ptr = reinterpret_cast<char *>(pre_gelu_out.dptr());
  char *workspace_ptr = reinterpret_cast<char *>(workspace.dptr());
  size_t workspace_size_chunk = workspace.numel() / _stream_compute.size();

  NVTE_CHECK_CUDA(cudaEventRecord(_start_compute, stream_main));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_send, _start_compute, 0));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_recv, _start_compute, 0));
  for (size_t i = 0; i < _stream_compute.size(); i++) {
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_compute[i], _start_compute, 0));
  }

  if (_aggregate) {
    const int num_steps = _tp_size / 2;
    char *input_b_ptr = reinterpret_cast<char *>(ubufs[0].dptr());

    // Initial 1X input chunk exchange between neighboring peers
    int send_chunk_id = _tp_id;
    int recv_chunk_id = (_tp_id % 2 == 0) ? _tp_id + 1 : _tp_id - 1;
    int send_offset = comm_bytes * send_chunk_id;
    int recv_offset = comm_bytes * recv_chunk_id;
    int peer_rank = (_tp_id % 2 == 0) ? _next_rank : _prev_rank;
    userbuffers_send(_ub_reg, send_offset, _ub_reg, send_offset, comm_bytes, _ub_comm, peer_rank,
                      _stream_send);
    userbuffers_recv(_ub_reg, recv_offset, _ub_reg, recv_offset, comm_bytes, _ub_comm, peer_rank,
                      _stream_recv);

    NVTE_CHECK_CUDA(cudaEventRecord(_stop_recv, _stream_recv));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_send, _stop_recv, 0));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_compute[0], _stop_recv, 0));

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
      TensorWrapper input_b_chunk = TensorWrapper(
          reinterpret_cast<void *>(input_b_ptr + send_offset), {n_chunk * 2, k}, ubufs[0].dtype(),
          ubufs[0].amax(), ubufs[0].scale(), ubufs[0].scale_inv());
      TensorWrapper output_chunk = TensorWrapper(
          reinterpret_cast<void *>(output_ptr + (send_chunk_id * output_chunk_bytes)),
          {n_chunk * 2, m}, D.dtype(), D.amax(), D.scale(), D.scale_inv());
      TensorWrapper pre_gelu_out_chunk =
          TensorWrapper(nullptr, std::vector<size_t>{0}, pre_gelu_out.dtype());
      if (do_gelu) {
        pre_gelu_out_chunk = TensorWrapper(
            reinterpret_cast<void *>(pre_gelu_out_ptr + (send_chunk_id * aux_chunk_bytes)),
            {n_chunk * 2, m}, pre_gelu_out.dtype());
      }
      TensorWrapper workspace_chunk =
          TensorWrapper(reinterpret_cast<void *>(workspace_ptr + (i % _stream_compute.size()) *
                                                                      workspace_size_chunk),
                        {workspace_size_chunk}, workspace.dtype());

      nvte_cublas_gemm(A.data(), input_b_chunk.data(), output_chunk.data(), bias.data(),
                        pre_gelu_out_chunk.data(), A_trans, B_trans, grad, workspace_chunk.data(),
                        accumulate, use_split_accumulator, _math_sms,
                        _stream_compute[i % _stream_compute.size()]);

      if (i < num_steps - 1) {
        // P2P communication
        userbuffers_send(_ub_reg, send_offset, _ub_reg, send_offset, comm_bytes * 2, _ub_comm,
                          next_rank, _stream_send);
        userbuffers_recv(_ub_reg, recv_offset, _ub_reg, recv_offset, comm_bytes * 2, _ub_comm,
                          prev_rank, _stream_recv);

        NVTE_CHECK_CUDA(cudaEventRecord(_stop_recv, _stream_recv));
        NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_send, _stop_recv, 0));
        NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_compute[(i + 1) % _stream_compute.size()],
                                            _stop_recv, 0));
      } else if (B_copy.numel() > 0) {
        assert(B_copy.numel() == ubufs[_tp_id].numel());
        assert(B_copy.element_size() == ubufs[_tp_id].element_size());
        NVTE_CHECK_CUDA(cudaMemcpyAsync(B_copy.dptr(), ubufs[_tp_id].dptr(),
                                        ubufs[_tp_id].numel() * ubufs[_tp_id].element_size(),
                                        cudaMemcpyDeviceToDevice, _stream_send));
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
      TensorWrapper output_chunk = TensorWrapper(
          reinterpret_cast<void *>(output_ptr + (send_chunk_id * output_chunk_bytes)),
          {n_chunk, m}, D.dtype(), D.amax(), D.scale(), D.scale_inv());
      TensorWrapper pre_gelu_out_chunk =
          TensorWrapper(nullptr, std::vector<size_t>{0}, pre_gelu_out.dtype());
      if (do_gelu) {
        pre_gelu_out_chunk = TensorWrapper(
            reinterpret_cast<void *>(pre_gelu_out_ptr + (send_chunk_id * aux_chunk_bytes)),
            {n_chunk, m}, pre_gelu_out.dtype());
      }
      TensorWrapper workspace_chunk =
          TensorWrapper(reinterpret_cast<void *>(workspace_ptr + (i % _stream_compute.size()) *
                                                                      workspace_size_chunk),
                        {workspace_size_chunk}, workspace.dtype());

      nvte_cublas_gemm(A.data(), ubufs[send_chunk_id].data(), output_chunk.data(), bias.data(),
                        pre_gelu_out_chunk.data(), A_trans, B_trans, grad, workspace_chunk.data(),
                        accumulate, use_split_accumulator, _math_sms,
                        _stream_compute[i % _stream_compute.size()]);

      if (i < _tp_size - 1) {
        // P2P communication
        userbuffers_send(_ub_reg, send_offset, _ub_reg, send_offset, comm_bytes, _ub_comm,
                          _next_rank, _stream_send);
        userbuffers_recv(_ub_reg, recv_offset, _ub_reg, recv_offset, comm_bytes, _ub_comm,
                          _prev_rank, _stream_recv);

        NVTE_CHECK_CUDA(cudaEventRecord(_stop_recv, _stream_recv));
        NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_send, _stop_recv, 0));
        NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_compute[(i + 1) % _stream_compute.size()],
                                            _stop_recv, 0));
      } else if (B_copy.numel() > 0) {
        assert(B_copy.numel() == ubufs[_tp_id].numel());
        assert(B_copy.element_size() == ubufs[_tp_id].element_size());
        NVTE_CHECK_CUDA(cudaMemcpyAsync(B_copy.dptr(), ubufs[_tp_id].dptr(),
                                        ubufs[_tp_id].numel() * ubufs[_tp_id].element_size(),
                                        cudaMemcpyDeviceToDevice, _stream_send));
      }
    }
  }

  for (size_t i = 0; i < _stream_compute.size(); i++) {
    NVTE_CHECK_CUDA(cudaEventRecord(_stop_compute, _stream_compute[i]));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_compute, 0));
  }

  _ub_comm->sms = ori_sm;
  NVTE_CHECK_CUDA(cudaEventRecord(_stop_send, _stream_send));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_send, 0));
  NVTE_CHECK_CUDA(cudaEventRecord(_stop_recv, _stream_recv));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_recv, 0));
}  // CommGemmOverlapP2P::split_gemm_overlap_ag

/*
  Split ReduceScatter + Atomic GEMM using P2P communication
*/
void CommGemmOverlapP2P::atomic_gemm_overlap_rs(
    cudaStream_t stream_main, const TensorWrapper &A, bool A_trans, const TensorWrapper &B,
    bool B_trans, const TensorWrapper &bias, const TensorWrapper &D,
    const TensorWrapper &pre_gelu_out, const TensorWrapper &ubuf,
    const std::vector<TensorWrapper> &ubufs, const TensorWrapper &counters,
    const TensorWrapper &workspace, bool grad, bool accumulate, bool use_split_accumulator) {
  int ori_sm = _ub_comm->sms;
  _ub_comm->use_ce = _use_ce;
  _ub_comm->sms = _comm_sms;
  _ub_comm->cga_size = _cga_size;

  // Get communication and GEMM input chunk sizes
  const int comm_bytes = ubufs[0].numel() * ubufs[0].element_size();

  // Get input and workspace data pointers
  char *workspace_ptr = reinterpret_cast<char *>(workspace.dptr());
  int *counter_ptr = reinterpret_cast<int *>(counters.dptr());
  size_t workspace_size_chunk = workspace.numel() / _stream_compute.size();

  // Reset counters to "not ready" (1)
  reset_counters(counter_ptr, _tp_size, -1, stream_main);

  // Catch up the main stream
  NVTE_CHECK_CUDA(cudaEventRecord(_start_compute, stream_main));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_recv, _start_compute, 0));

  // Atomic GEMM
  // Process GEMM chunks in the order that AG+GEMM places the output chunks.
  TensorWrapper workspace_chunk = TensorWrapper(reinterpret_cast<void *>(workspace_ptr),
                                                {workspace_size_chunk}, workspace.dtype());
  nvte_cublas_atomic_gemm(A.data(), B.data(), ubuf.data(), bias.data(), pre_gelu_out.data(),
                          A_trans, B_trans, grad, workspace_chunk.data(), accumulate,
                          use_split_accumulator, _math_sms, 0, _tp_size, true, counters.data(),
                          stream_main);

  // P2P communication chunk
  for (int i = 1; i < _tp_size; i++) {
    int send_chunk_id = i - 1;
    int recv_chunk_id = send_chunk_id + _tp_size;
    int send_offset = comm_bytes * send_chunk_id;
    int recv_offset = comm_bytes * recv_chunk_id;
    int send_rank = (_tp_size + _tp_id - i) % _tp_size + _rank_round_tp;
    int recv_rank = (_tp_id + i) % _tp_size + _rank_round_tp;

    consumer(counter_ptr, send_chunk_id, _stream_recv);
    userbuffers_send(_ub_reg, send_offset, _ub_reg, recv_offset, comm_bytes, _ub_comm, send_rank,
                      _stream_recv);
    userbuffers_recv(_ub_reg, send_offset, _ub_reg, recv_offset, comm_bytes, _ub_comm, recv_rank,
                      _stream_recv);
  }

  _ub_comm->sms = ori_sm;
  NVTE_CHECK_CUDA(cudaEventRecord(_stop_recv, _stream_recv));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_recv, 0));
}  // CommGemmOverlapP2P::atomic_gemm_overlap_rs

/*
  Split ReduceScatter + Pipelined GEMM using P2P communication
*/
void CommGemmOverlapP2P::split_gemm_overlap_rs(
    cudaStream_t stream_main, const TensorWrapper &A, bool A_trans, const TensorWrapper &B,
    bool B_trans, const TensorWrapper &bias, const TensorWrapper &D,
    const TensorWrapper &pre_gelu_out, const std::vector<TensorWrapper> &ubufs,
    const TensorWrapper &workspace, bool grad, bool accumulate, bool use_split_accumulator) {
  int ori_sm = _ub_comm->sms;
  _ub_comm->use_ce = _use_ce;
  _ub_comm->sms = _comm_sms;
  _ub_comm->cga_size = _cga_size;

  // Get communication and GEMM input chunk sizes
  size_t k = (A_trans) ? A.size(1) : A.size(0);
  size_t n = (B_trans) ? B.size(1) : B.size(0);
  size_t n_chunk = n / _tp_size;
  const int comm_bytes = ubufs[0].numel() * ubufs[0].element_size();
  const int input_b_chunk_bytes = n_chunk * k * B.element_size();

  // Get input and workspace data pointers
  char *input_b_ptr = reinterpret_cast<char *>(B.dptr());
  char *workspace_ptr = reinterpret_cast<char *>(workspace.dptr());
  size_t workspace_size_chunk = workspace.numel() / _stream_compute.size();

  // Catch up the main stream
  NVTE_CHECK_CUDA(cudaEventRecord(_start_compute, stream_main));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_send, _start_compute, 0));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_recv, _start_compute, 0));
  for (size_t i = 0; i < _stream_compute.size(); i++) {
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_compute[i], _start_compute, 0));
  }

  // GEMM and send/recv chunks
  for (int i = 0; i < _tp_size; i++) {
    // GEMM chunk
    int input_b_chunk_id = (_tp_id + i + 1) % _tp_size;
    char *input_b_chunk_ptr = input_b_ptr + (input_b_chunk_id * input_b_chunk_bytes);
    TensorWrapper input_b_chunk = TensorWrapper(
      reinterpret_cast<void *>(input_b_chunk_ptr), {n_chunk, k}, B.dtype(), nullptr, nullptr,
      B.scale_inv());
    TensorWrapper workspace_chunk = TensorWrapper(
      reinterpret_cast<void *>(workspace_ptr + (i % _stream_compute.size()) * workspace_size_chunk),
      {workspace_size_chunk}, workspace.dtype());

    cudaStream_t gemm_stream = (i == _tp_size - 1) ? stream_main
                                                   : _stream_compute[i % _stream_compute.size()];
    nvte_cublas_gemm(A.data(), input_b_chunk.data(), ubufs[i].data(), bias.data(),
                      pre_gelu_out.data(), A_trans, B_trans, grad, workspace_chunk.data(),
                      accumulate, use_split_accumulator, _math_sms, gemm_stream);

    if (i > 0) {
      // P2P communication chunk
      int send_offset = comm_bytes * (i - 1);
      int recv_offset = comm_bytes * (i - 1 + _tp_size);
      int send_rank = (_tp_id + i) % _tp_size + _rank_round_tp;
      int recv_rank = (_tp_size + _tp_id - i) % _tp_size + _rank_round_tp;
      NVTE_CHECK_CUDA(cudaEventRecord(
          _start_comm, _stream_compute[(i - 1) % _stream_compute.size()]));
      NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_send, _start_comm, 0));
      NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_recv, _start_comm, 0));
      userbuffers_send(_ub_reg, send_offset, _ub_reg, recv_offset, comm_bytes, _ub_comm,
                        send_rank, _stream_send);
      userbuffers_recv(_ub_reg, send_offset, _ub_reg, recv_offset, comm_bytes, _ub_comm,
                        recv_rank, _stream_recv);
    }
  }

  _ub_comm->sms = ori_sm;
  NVTE_CHECK_CUDA(cudaEventRecord(_stop_recv, _stream_recv));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_recv, 0));
}  // CommGemmOverlapP2P::split_gemm_overlap_rs

}  // namespace common

}  // namespace transformer_engine
