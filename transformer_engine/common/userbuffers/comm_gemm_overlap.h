/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_USERBUFFERS_COMM_GEMM_OVERLAP_H_
#define TRANSFORMER_ENGINE_USERBUFFERS_COMM_GEMM_OVERLAP_H_

#include <cstring>
#include <typeinfo>
#include <variant>
#include <any>
#include <dlpack/dlpack.h>
#include <pybind11/pybind11.h>
#include <transformer_engine/transformer_engine.h>
#include <transformer_engine/gemm.h>

#include "../util/logging.h"
#include "../util/system.h"
#include "../util/dlpack_helper.h"

#include "userbuffers.h"

#define HALF_BYTES 2
#define UB_MAX_SM 32

// Hacky type restriction to comply with userbuffers
#if __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#define half nv_bfloat16
#else
#include <cuda_fp16.h>
#endif

namespace py = pybind11;

static const size_t NVTE_MAX_USERBUFFER_STREAMS = 3;

enum class NVTE_Comm_Overlap_Type {
  REDUCE_SCATTER = 0,
  ALL_GATHER = 1
};

enum class NVTE_Comm_Overlap_Algo {
  BULK_OVERLAP_AG = 0,
  BULK_OVERLAP_RS = 1,
  SPLIT_PIPELINED_AG_P2P = 2,
  SPLIT_PIPELINED_RS = 3,
  SPLIT_PIPELINED_RS_P2P = 4,
  ATOMIC_GEMM_RS = 5,
  ATOMIC_GEMM_AG_P2P = 6,
  ATOMIC_GEMM_RS_P2P = 7
};

namespace transformer_engine {

namespace userbuffers {

struct PYBIND11_EXPORT CommGemmOverlapBase {
  static inline communicator *_ub_comm{nullptr};
  static inline bool _comm_created{false};

  int _tp_id, _tp_size;
  int _comm_sms, _math_sms;
  int _ub_reg;
  int _num_splits;
  int _cga_size;
  int _use_ce;
  bool _atomic_gemm{false};
  bool _buffer_registered{false};
  bool _is_p2p{false};

  std::function<py::capsule(py::capsule &, const std::string &)> _alloc_copy_allgather;
  std::function<void(py::capsule &)> _free;
  std::function<void(const std::string &)> _barrier;

  cudaEvent_t _start_compute, _stop_compute, _start_comm, _stop_comm;
  std::vector<cudaStream_t> _stream_compute;

  CommGemmOverlapBase(
    int worldrank, int worldsize, int localrank, int localsize, int nodeid, int numnodes,
    int num_splits, int num_max_streams, int num_comm_cga, int num_comm_sms,
    bool set_sm_margin, bool atomic_gemm
  ) {
    // Initialize the UB communicator
    if (!_comm_created) {
      if (worldrank == 0) {
        printf("!!! [UB] Create UB communicator\n");
      }
#ifndef UB_MPI_BOOTSTRAP
      create_communicator_grouped2(&_ub_comm,
        worldrank, worldsize, localrank, localsize, nodeid, numnodes,
        [this](void **globalbuf, void *localbuf, size_t localbytes, const char *group) {
          _ub_alloc_copy_allgather(globalbuf, localbuf, localbytes, group); },
        [this](void *ptr, size_t bytes) { _ub_free(ptr, bytes); },
        [this](const char *group) { _ub_barrier(group); },
        1, 1, localsize, 1);
#else
      create_communicator_grouped2_mpi(&_ub_comm, 1, 1, localsize, 1);
#endif
      _comm_created = true;
    }

    _tp_id = _ub_comm->myrank % _tp_size;
    _num_splits = num_splits;
    _cga_size = num_comm_cga;
    _use_ce = 0;

    for (int i = 0; i < std::min(_num_splits, num_max_streams); i++) {
      cudaStream_t new_stream;
      NVTE_CHECK_CUDA(cudaStreamCreateWithPriority(&new_stream, cudaStreamNonBlocking, -1));
      _stream_compute.push_back(new_stream);
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    _comm_sms = num_comm_sms;
    _math_sms = (set_sm_margin) ? prop.multiProcessorCount - num_comm_sms \
                                : prop.multiProcessorCount;
    _math_sms -= getenv<int>("NVTE_EXT_MARGIN_SM", 0);

    NVTE_CHECK_CUDA(cudaEventCreateWithFlags(&_start_compute, 0));
    NVTE_CHECK_CUDA(cudaEventCreateWithFlags(&_stop_compute, 0));
    NVTE_CHECK_CUDA(cudaEventCreateWithFlags(&_start_comm, 0));
    NVTE_CHECK_CUDA(cudaEventCreateWithFlags(&_stop_comm, 0));
  }

  ~CommGemmOverlapBase() {
    destroy_communicator(_ub_comm);
    _comm_created = false;
    for (size_t i = 0; i < _stream_compute.size(); i++) {
      cudaStreamDestroy(_stream_compute[i]);
    }
    cudaEventDestroy(_start_compute);
    cudaEventDestroy(_stop_compute);
    cudaEventDestroy(_start_comm);
    cudaEventDestroy(_stop_comm);
  }

  CommGemmOverlapBase(const CommGemmOverlapBase &other) = delete;
  CommGemmOverlapBase& operator=(const CommGemmOverlapBase &other) = delete;

  void _ub_alloc_copy_allgather(
    void **globalbuf, void *localbuf, size_t localbytes, const char *group
  ) {
    int64_t group_size;
    if (strcmp(group, "world")) {
      group_size = _ub_comm->nranks;
    } else if (strcmp(group, "inter")) {
      group_size = _ub_comm->num_nodes;
    } else if (strcmp(group, "intra")) {
      group_size = _ub_comm->nvsize;
    } else {
      NVTE_ERROR("Invalid group name: must be 'world', 'inter', or 'intra'.");
    }
    auto localdata = buffer_to_capsule<uint8_t>(localbuf, static_cast<int64_t>(localbytes));
    auto globaldata = _alloc_copy_allgather(localdata, group);
    int64_t globalbytes = capsule_to_buffer(globaldata, globalbuf);
    NVTE_CHECK(globalbytes == static_cast<int64_t>(localbytes) * group_size,
               "Incorrect size for allgathered data.");
  }

  void _ub_free(void *ptr, size_t bytes) {
    auto data = buffer_to_capsule<uint8_t>(ptr, static_cast<int64_t>(bytes));
    _free(data);
  }

  void _ub_barrier(const char *group) { _barrier(group); }

  void register_gpu_buffer(void **gpuptr, size_t bytes, bool alloc = false) {
    NVTE_CHECK(_comm_created, "[UB] Communicator must be initialized before buffer registration.");
    NVTE_CHECK(!_buffer_registered, "[UB] GPU buffer is already registered.");
    _ub_reg = register_user_buffer_collective(gpuptr, bytes, _ub_comm, alloc);
    _buffer_registered = true;
  }

  void register_gpu_buffer(py::capsule *gpubuf, bool alloc = false) {
    void *gpuptr;
    size_t bytes = capsule_to_buffer(gpubuf, &gpuptr);
    register_gpu_buffer(&gpuptr, bytes, alloc);
  }

  void set_collective_callbacks(
    std::function<py::capsule(py::capsule&, const std::string&)> alloc_copy_allgather_handle,
    std::function<void(py::capsule&)> free_handle,
    std::function<void(const std::string&)> barrier_handle
  ) {
    _alloc_copy_allgather = alloc_copy_allgather_handle;
    _free = free_handle;
    _barrier = barrier_handle;
  }

  bool is_atomic_gemm() { return _atomic_gemm; }

  bool is_p2p_overlap() { return _is_p2p; }
};  // CommGemmOverlapBase

struct PYBIND11_EXPORT CommGemmOverlap : CommGemmOverlapBase {
  cudaStream_t _stream_comm;
  cudaEvent_t _start_d2dcopy;

  CommGemmOverlap(
    int worldrank, int worldsize, int localrank, int localsize, int nodeid, int numnodes,
    int num_splits, int num_max_streams, int num_comm_cga, int num_comm_sms,
    bool set_sm_margin, bool atomic_gemm)
  : CommGemmOverlapBase(worldrank, worldsize, localrank, localsize, nodeid, numnodes,
                        num_splits, num_max_streams, num_comm_cga, num_comm_sms,
                        set_sm_margin, atomic_gemm) {
    NVTE_CHECK_CUDA(cudaStreamCreate(&_stream_comm));
    NVTE_CHECK_CUDA(cudaEventCreateWithFlags(&_start_d2dcopy, 0));
  }

  ~CommGemmOverlap() {
    cudaStreamDestroy(_stream_comm);
    cudaEventDestroy(_start_d2dcopy);
  }

  /*
  ** Bulk GEMM + AllGather
  ** This function assumes that input (B) is pre-copied to ubuf
  */
  void bulk_gemm_overlap_ag(cudaStream_t stream_main,
    TensorWrapper *A, bool A_trans, TensorWrapper *B, bool B_trans, TensorWrapper *bias,
    TensorWrapper *D, TensorWrapper *pre_gelu_out,
    TensorWrapper *ubuf, TensorWrapper *workspace,
    bool grad, bool accumulate, bool use_split_accumulator
  ) {
    _ub_comm->use_ce = _use_ce;
    _ub_comm->sms = _comm_sms;
    _ub_comm->cga_size = _cga_size;

    NVTE_CHECK_CUDA(cudaEventRecord(_start_comm, cudaStreamDefault));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _start_comm, 0));

    int comm_elements = (ubuf->numel() / 2) * ubuf->element_size();
    allgather2_userbuff_inplace(_ub_reg, 0, comm_elements, _ub_comm, stream_main);

    assert(pre_gelu_out->numel() == 0);
    nvte_cublas_gemm(A->data(), B->data(), D->data(), bias->data(), pre_gelu_out->data(),
                    A_trans, B_trans, grad, workspace->data(), accumulate, use_split_accumulator,
                    _math_sms, stream_main);

    NVTE_CHECK_CUDA(cudaEventRecord(_stop_comm, cudaStreamDefault));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_comm, 0));
  }  // bulk_gemm_overlap_ag

  /*
  ** Bulk GEMM + ReduceScatter
  ** This function assumes that input (B) is pre-copied to ubuf
  */
  void bulk_gemm_overlap_rs(cudaStream_t stream_main,
    TensorWrapper *A, bool A_trans, TensorWrapper *B, bool B_trans, TensorWrapper *bias,
    TensorWrapper *D, TensorWrapper *pre_gelu_out,
    TensorWrapper *ubuf, TensorWrapper *rs_out, TensorWrapper *workspace,
    bool grad, bool accumulate, bool use_split_accumulator
  ) {
    _ub_comm->use_ce = _use_ce;
    _ub_comm->sms = _comm_sms;
    _ub_comm->cga_size = _cga_size;

    NVTE_CHECK_CUDA(cudaEventRecord(_start_comm, cudaStreamDefault));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _start_comm, 0));

    int comm_elements = (ubuf->numel() / 2) * ubuf->element_size();

    if (is_fp8_dtype(ubuf->dtype())) {
      comm_elements *= 2;
      assert(rs_out->numel() == ubuf->numel() / _tp_size);
      assert(rs_out->size(0) == ubuf->size(0) / _tp_size);
      assert(!is_fp8_dtype(rs_out->dtype()));
      char *rs_out_ptr = reinterpret_cast<char *>(rs_out->dptr());
      reducescatter2_userbuff_fp8<__nv_fp8_e5m2>(rs_out_ptr, ubuf->scale_inv(), _ub_reg, 0,
                                                  comm_elements, _ub_comm, _stream_comm);
    } else {
      reducescatter2_userbuff_inplace(_ub_reg, 0, comm_elements, _ub_comm, _stream_comm);
    }

    assert(pre_gelu_out->numel() == 0);
    nvte_cublas_gemm(A->data(), B->data(), D->data(), bias->data(), pre_gelu_out->data(),
                    A_trans, B_trans, grad, workspace->data(), accumulate, use_split_accumulator,
                    _math_sms, stream_main);

    NVTE_CHECK_CUDA(cudaEventRecord(_stop_comm, cudaStreamDefault));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_comm, 0));
  }  // bulk_gemm_overlap_rs

  /*
  ** Atomic FPROP GEMM + ReduceScatter
  */
  void atomic_gemm_overlap_rs(cudaStream_t stream_main,
    TensorWrapper *A, bool A_trans, TensorWrapper *B, bool B_trans, TensorWrapper *bias,
    TensorWrapper *D, TensorWrapper *pre_gelu_out,
    TensorWrapper *ubuf, TensorWrapper *counters, TensorWrapper *rs_out,
    TensorWrapper *workspace, bool grad, bool accumulate, bool use_split_accumulator
  ) {
    _ub_comm->use_ce = _use_ce;
    _ub_comm->sms = _comm_sms;
    _ub_comm->cga_size = _cga_size;

    size_t m = (A_trans) ? A->size(1) : A->size(0);
    size_t n = (B_trans) ? B->size(1) : B->size(0);
    size_t m_chunk = m / _num_splits;
    size_t workspace_size_chunk = workspace->numel() / _stream_compute.size();

    char *rs_out_ptr = reinterpret_cast<char *>(rs_out->dptr());
    int *counter_ptr = reinterpret_cast<int *>(counters->dptr());
    int ori_sms = _ub_comm->sms;

    NVTE_CHECK_CUDA(cudaEventRecord(_start_compute, cudaStreamDefault));
    NVTE_CHECK_CUDA(cudaEventRecord(_stop_comm, stream_main));
    for (size_t i = 0; i < _stream_compute.size(); i++) {
      NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_compute[i], _start_compute, 0));
      NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_compute[i], _stop_comm, 0));
    }

    assert(pre_gelu_out->numel() == 0);
    TensorWrapper output_d = TensorWrapper(
      ubuf->dptr(), {n, m}, ubuf->dtype(), D->amax(), D->scale(), nullptr);
    TensorWrapper workspace_chunk = TensorWrapper(
      workspace->dptr(), {workspace_size_chunk}, workspace->dtype());
    nvte_cublas_atomic_gemm(A->data(), B->data(), output_d.data(), bias->data(),
                            pre_gelu_out->data(), A_trans, B_trans, grad, workspace_chunk.data(),
                            accumulate, use_split_accumulator, _math_sms, _num_splits, 0, true,
                            counters->dptr(), _stream_compute[0]);

    for (int i = 0; i < _num_splits; i++) {
      const char *env_p = std::getenv("NVTE_RS_STRIDED_ATOMIC");
      if (env_p != nullptr && env_p[0] == '1') {
        if (i == _num_splits - 1) {
          _ub_comm->sms = UB_MAX_SM;
        }
        if (is_fp8_dtype(ubuf->dtype())) {
          reducescatter2_userbuff_strided_atomic_fp8<__nv_fp8_e4m3>(
              rs_out_ptr, ubuf->scale_inv(), _ub_reg, i * m_chunk, m_chunk, n, m, m, _num_splits,
              &counter_ptr[i], _ub_comm, _stream_comm);
        } else {
          reducescatter2_userbuff_strided_atomic(rs_out_ptr, _ub_reg, i * m_chunk, m_chunk, n, m,
                                                _num_splits, &counter_ptr[i], _ub_comm,
                                                _stream_comm);
        }
      } else if (env_p != nullptr && env_p[0] == '2') {
        if (is_fp8_dtype(ubuf->dtype())) {
          reducescatter2_userbuff_strided_multiatomic_fp8<__nv_fp8_e4m3>(
              rs_out_ptr, ubuf->scale_inv(), _ub_reg, m_chunk, m_chunk, n, m, m, _num_splits,
              counter_ptr, _ub_comm, _stream_comm);
        } else {
          reducescatter2_userbuff_strided_multiatomic(rs_out_ptr, _ub_reg, m_chunk, m_chunk, n,
                                                      m, _num_splits, counter_ptr, _ub_comm,
                                                      _stream_comm);
        }
        break;
      } else {
        consumer(counter_ptr, i, _stream_comm);
        //        if (i == _num_splits-1) {
        //           _ub_comm->sms = UB_MAX_SM;
        //        }
        reducescatter2_userbuff_strided(rs_out_ptr, _ub_reg, i * m_chunk, m_chunk, n, m,
                                        _ub_comm, _stream_comm);
      }

      rs_out_ptr += m_chunk * rs_out->element_size();
    }

    _ub_comm->sms = ori_sms;
    NVTE_CHECK_CUDA(cudaEventRecord(_stop_compute, _stream_compute[0]));
    NVTE_CHECK_CUDA(cudaEventRecord(_stop_comm, stream_main));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(cudaStreamDefault, _stop_compute, 0));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(cudaStreamDefault, _stop_comm, 0));
  }  // atomic_gemm_overlap_rs

  /*
  ** Split FPROP GEMM + ReduceScatter
  */
  void split_gemm_overlap_rs(cudaStream_t stream_main,
    TensorWrapper *A, bool A_trans, TensorWrapper *B, bool B_trans, TensorWrapper *bias,
    TensorWrapper *D, TensorWrapper *pre_gelu_out,
    TensorWrapper *ubuf, TensorWrapper *rs_out, TensorWrapper *workspace,
    bool grad, bool accumulate, bool use_split_accumulator, bool gemm_overlap
  ) {
    _ub_comm->use_ce = _use_ce;
    _ub_comm->sms = _comm_sms;
    _ub_comm->cga_size = _cga_size;

    size_t m = (A_trans) ? A->size(1) : A->size(0);
    size_t k = (A_trans) ? A->size(0) : A->size(1);
    size_t n = (B_trans) ? B->size(1) : B->size(0);
    size_t m_chunk = m / _num_splits;
    size_t input_a_chunk_size = m_chunk * k;
    size_t output_chunk_size = n * m_chunk;
    size_t workspace_size_chunk = workspace->numel() / _stream_compute.size();

    char *input_a_chunk_ptr = reinterpret_cast<char*>(A->dptr());
    char *output_buf_chunk_ptr = reinterpret_cast<char *>(ubuf->dptr());
    char *workspace_chunk_ptr = reinterpret_cast<char *>(workspace->dptr());
    char *rs_out_ptr = reinterpret_cast<char *>(rs_out->dptr());
    int ori_sms = _ub_comm->sms;

    NVTE_CHECK_CUDA(cudaEventRecord(_start_compute, cudaStreamDefault));
    NVTE_CHECK_CUDA(cudaEventRecord(_stop_comm, stream_main));
    for (size_t i = 0; i < _stream_compute.size(); i++) {
      NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_compute[i], _start_compute, 0));
      NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_compute[i], _stop_comm, 0));
    }

    assert(pre_gelu_out->numel() == 0);
    if (gemm_overlap) {
      TensorWrapper input_a_chunk = TensorWrapper(
        reinterpret_cast<void *>(input_a_chunk_ptr), {m_chunk, k}, A->dtype(),
        nullptr, nullptr, A->scale_inv());
      TensorWrapper output_chunk = TensorWrapper(
        reinterpret_cast<void *>(output_buf_chunk_ptr), {n, m_chunk}, D->dtype(),
        D->amax(), D->scale(), nullptr);
      TensorWrapper workspace_chunk = TensorWrapper(
        reinterpret_cast<void *>(workspace_chunk_ptr), {workspace_size_chunk}, workspace->dtype());

      nvte_cublas_gemm(
        input_a_chunk.data(), B->data(), output_chunk.data(), bias->data(), pre_gelu_out->data(),
        A_trans, B_trans, grad, workspace_chunk.data(), accumulate, use_split_accumulator,
        _math_sms, _stream_compute[0]);

      for (int i = 1; i < _num_splits; i++) {
        input_a_chunk_ptr += input_a_chunk_size * B->element_size();
        output_buf_chunk_ptr += output_chunk_size * ubuf->element_size();

        TensorWrapper input_a_chunk = TensorWrapper(
          reinterpret_cast<void *>(input_a_chunk_ptr), {m_chunk, k}, A->dtype(),
          nullptr, A->scale_inv(), nullptr);
        TensorWrapper output_chunk = TensorWrapper(
          reinterpret_cast<void *>(output_buf_chunk_ptr), {n, m_chunk}, D->dtype(),
          D->amax(), D->scale(), nullptr);

        nvte_cublas_gemm(
          input_a_chunk.data(), B->data(), output_chunk.data(), bias->data(), pre_gelu_out->data(),
          A_trans, B_trans, grad, workspace_chunk.data(), accumulate, use_split_accumulator,
          _math_sms, _stream_compute[i % _stream_compute.size()]);

        NVTE_CHECK_CUDA(cudaEventRecord(_start_comm,
                                        _stream_compute[(i - 1) % _stream_compute.size()]));
        NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_comm, _start_comm, 0));

        // Communication chunk
        if (is_fp8_dtype(ubuf->dtype())) {
          reducescatter2_userbuff_stridedoutput_fp8<__nv_fp8_e4m3>(
              rs_out_ptr, ubuf->scale_inv(), _ub_reg, (i - 1) * output_chunk_size, m_chunk, n, m,
              _ub_comm, _stream_comm);
        } else {
          reducescatter2_userbuff_stridedoutput(rs_out_ptr, _ub_reg, (i - 1) * output_chunk_size,
                                                m_chunk, n, m, _ub_comm, _stream_comm);
        }

        rs_out_ptr += m_chunk * rs_out->element_size();
      }
      int last_compute_stream_id =
          (_num_splits + _stream_compute.size() - 1) % _stream_compute.size();
      NVTE_CHECK_CUDA(cudaEventRecord(_start_comm, _stream_compute[last_compute_stream_id]));
      NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_comm, _start_comm, 0));

      // Last communication chunk with max SM
      _ub_comm->sms = UB_MAX_SM;
      if (is_fp8_dtype(ubuf->dtype())) {
        reducescatter2_userbuff_stridedoutput_fp8<__nv_fp8_e4m3>(
            rs_out_ptr, ubuf->scale_inv(), _ub_reg, (_num_splits - 1) * output_chunk_size, m_chunk,
            n, m, _ub_comm, _stream_comm);
      } else {
        reducescatter2_userbuff_stridedoutput(rs_out_ptr, _ub_reg,
                                              (_num_splits - 1) * output_chunk_size, m_chunk, n, m,
                                              _ub_comm, _stream_comm);
      }
    } else {
      for (int i = 0; i < _num_splits; i++) {
        TensorWrapper input_a_chunk = TensorWrapper(
          reinterpret_cast<void *>(input_a_chunk_ptr), {m_chunk, k}, A->dtype(),
          nullptr, nullptr, A->scale_inv());
        TensorWrapper output_chunk = TensorWrapper(
          reinterpret_cast<void *>(output_buf_chunk_ptr), {n, m_chunk}, D->dtype(),
          D->amax(), D->scale(), nullptr);
        TensorWrapper workspace_chunk = TensorWrapper(
          reinterpret_cast<void *>(workspace_chunk_ptr), {workspace_size_chunk},
          workspace->dtype());

        nvte_cublas_gemm(
          input_a_chunk.data(), B->data(), output_chunk.data(), bias->data(), pre_gelu_out->data(),
          A_trans, B_trans, grad, workspace_chunk.data(), accumulate, use_split_accumulator,
          _math_sms, _stream_compute[i % _stream_compute.size()]);

        NVTE_CHECK_CUDA(cudaEventRecord(_start_comm,
                                        _stream_compute[i % _stream_compute.size()]));
        NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_comm, _start_comm, 0));

        // Communication chunk. Uses MAX_SM at the last chunk
        if (i == _num_splits - 1) {
          _ub_comm->sms = UB_MAX_SM;
        }
        if (is_fp8_dtype(ubuf->dtype())) {
          reducescatter2_userbuff_stridedoutput_fp8<__nv_fp8_e4m3>(
              rs_out_ptr, ubuf->scale_inv(), _ub_reg, i * output_chunk_size, m_chunk, n, m,
              _ub_comm, _stream_comm);
        } else {
          reducescatter2_userbuff_stridedoutput(rs_out_ptr, _ub_reg, i * output_chunk_size,
                                                m_chunk, n, m, _ub_comm, _stream_comm);
        }
        rs_out_ptr += m_chunk * rs_out->element_size();
        input_a_chunk_ptr += input_a_chunk_size * B->element_size();
        output_buf_chunk_ptr += output_chunk_size * ubuf->element_size();
      }
    }

    _ub_comm->sms = ori_sms;
    int last_compute_stream_id =
      (_num_splits + _stream_compute.size() - 1) % _stream_compute.size();
    NVTE_CHECK_CUDA(cudaEventRecord(_stop_compute, _stream_compute[last_compute_stream_id]));
    NVTE_CHECK_CUDA(cudaEventRecord(_stop_comm, stream_main));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(cudaStreamDefault, _stop_compute, 0));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(cudaStreamDefault, _stop_comm, 0));
  }  // split_gemm_overlap_rs
};  //  CommGemmOverlap

struct PYBIND11_EXPORT CommGemmOverlapP2P : CommGemmOverlapBase {
  bool _reduce_scatter{false};
  bool _aggregate{false};
  bool _is_reduce_scatter{false};
  int _next_rank, _prev_rank, _rank, _rank_round_tp;

  int _num_ubuf_chunks;
  cudaStream_t _stream_send, _stream_recv;
  cudaEvent_t _stop_send, _stop_recv;

  CommGemmOverlapP2P(
    int worldrank, int worldsize, int localrank, int localsize, int nodeid, int numnodes,
    int num_splits, int num_max_streams, int num_comm_cga, int num_comm_sms,
    bool set_sm_margin, bool atomic_gemm, bool aggregate, bool is_reduce_scatter)
  : CommGemmOverlapBase(worldrank, worldsize, localrank, localsize, nodeid, numnodes,
                        num_splits, num_max_streams, num_comm_cga, num_comm_sms,
                        set_sm_margin, atomic_gemm) {
    _is_p2p = true;
    _aggregate = aggregate;
    _rank_round_tp = (_rank / _tp_size) * _tp_size;
    _next_rank = (_tp_size + _rank + 1) % _rank_round_tp;
    _prev_rank = (_tp_size + _rank - 1) % _rank_round_tp;

    _is_reduce_scatter = is_reduce_scatter;
    _num_ubuf_chunks = (_is_reduce_scatter) ? static_cast<int>(localsize * 2 - 1)
                                            : localsize;

    NVTE_CHECK_CUDA(cudaStreamCreate(&_stream_send));
    NVTE_CHECK_CUDA(cudaStreamCreate(&_stream_recv));
    NVTE_CHECK_CUDA(cudaEventCreateWithFlags(&_stop_send, 0));
    NVTE_CHECK_CUDA(cudaEventCreateWithFlags(&_stop_recv, 0));
  }

  ~CommGemmOverlapP2P() {
    cudaStreamDestroy(_stream_send);
    cudaStreamDestroy(_stream_recv);
    cudaEventDestroy(_stop_send);
    cudaEventDestroy(_stop_recv);
  }

  /*
  ** Get ubuf chunks
  ** This helper function returns a vector of TensorWrapper handles that correspond to
  ** chunks of the ubuf tensor for each rank in the userbuffer communicator.
  */
  std::vector<TensorWrapper> get_ubuf_chunks(TensorWrapper *ubuf) {
    size_t ubuf_chunk_bytes = ubuf->numel() * ubuf->element_size() / _num_ubuf_chunks;
    std::vector<size_t> ubuf_chunk_shape = {ubuf->size(0) / _num_ubuf_chunks, ubuf->size(1)};
    std::vector<TensorWrapper> ubuf_chunks;
    for (int i = 0; i < _num_ubuf_chunks; i++) {
      char *ubuf_byte_ptr = reinterpret_cast<char *>(ubuf->dptr()) + (i * ubuf_chunk_bytes);
      ubuf_chunks.push_back(
        TensorWrapper(
          reinterpret_cast<void *>(ubuf_byte_ptr), ubuf_chunk_shape,
          ubuf->dtype(), nullptr, nullptr, ubuf->scale_inv()));
    }
    return ubuf_chunks;
  }

  /*
  ** Split AllGather + AtomicGEMM using P2P communication
  ** This function assumes the input (B) is pre-copied to ubuf_chunks[rank_id]. This is
  ** necessary to have AG outputs in each rank to be in the contiguous memory space
  ** after all ring exchange phases.
  */
  void atomic_gemm_overlap_ag(cudaStream_t stream_main,
    TensorWrapper *A, bool A_trans, TensorWrapper *B, bool B_trans, TensorWrapper *bias,
    TensorWrapper *D, TensorWrapper *pre_gelu_out,
    TensorWrapper *ubuf, TensorWrapper *counters,
    TensorWrapper *B_copy, TensorWrapper *D_buffer,
    TensorWrapper *workspace, bool grad, bool accumulate, bool use_split_accumulator
  ) {
    _ub_comm->use_ce = _use_ce;
    _ub_comm->sms = _comm_sms;
    _ub_comm->cga_size = _cga_size;

    // Get GEMM dimensions between TN and NN input layouts
    const size_t m = (A_trans) ? A->size(0) : A->size(1);
    auto ubuf_chunks = get_ubuf_chunks(ubuf);
    const size_t n_chunk = ubuf_chunks[0].size(0);

    // Get communication and GEMM output chunk sizes
    const int comm_bytes = ubuf_chunks[0].numel() * ubuf_chunks[0].element_size();
    int *counter_ptr = reinterpret_cast<int *>(counters->dptr());
    size_t workspace_size_chunk = workspace->numel() / _stream_compute.size();
    assert(pre_gelu_out->numel() == 0);

    // Catch up the default stream
    NVTE_CHECK_CUDA(cudaEventRecord(_start_compute, stream_main));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_send, _start_compute, 0));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_recv, _start_compute, 0));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_compute[0], _start_compute, 0));

    TensorWrapper out_chunk = TensorWrapper(
      D->dptr(), {ubuf->size(0), m}, D->dtype(), D->amax(), D->scale(), nullptr);
    TensorWrapper work_chunk = TensorWrapper(
      workspace->dptr(), {workspace_size_chunk}, workspace->dtype());
    for (int i=0; i < _tp_size - 1; i++) {
      // Set the userbuffer id Buffer under send is the input for the current
      // GEMM chunk The initial input chunk is stored _ubuf[rank]. This is to
      // have the AG output in all ranks to be contiguous after the ring
      // exchanges
      int send_chunk_id = (_tp_size + _tp_id - i) % _tp_size;
      int recv_chunk_id = (_tp_size + _tp_id - i - 1) % _tp_size;
      int send_offset = comm_bytes * send_chunk_id;
      int recv_offset = comm_bytes * recv_chunk_id;

      const char *env_p = std::getenv("NVTE_AG_P2P_MULTI_ATOMIC");
      if (env_p != nullptr && env_p[0] == '1') {
        if (i == 0) {
          userbuffers_sendrecv_multiatomic(_ub_reg, _ub_reg, comm_bytes, comm_bytes, comm_bytes,
                                          _ub_comm, _next_rank, _prev_rank, _tp_size,
                                          counter_ptr, true, _stream_recv);
        }
      } else {
        userbuffers_send(_ub_reg, send_offset, _ub_reg, recv_offset, comm_bytes,
                        _ub_comm, _next_rank,  _stream_recv);
        userbuffers_recv(_ub_reg, send_offset, _ub_reg, recv_offset, comm_bytes,
                        _ub_comm, _prev_rank,  _stream_recv);
        producer(counter_ptr, recv_chunk_id, _stream_recv);
      }
      if (i == 0) {
        nvte_cublas_atomic_gemm(
          A->data(), ubuf->data(), D_buffer->data(), bias->data(), pre_gelu_out->data(),
          A_trans, B_trans, grad, work_chunk.data(), accumulate, use_split_accumulator,
          _math_sms, 0, _tp_size, false, counters->data(), _stream_compute[0]);
      }
    }

    // Store the input activation for backprop
    if (B_copy->numel() > 0) {
      assert(B_copy->numel() == ubuf_chunks[_tp_id].numel());
      assert(B_copy->element_size() == ubuf_chunks[_tp_id].element_size());
      NVTE_CHECK_CUDA(
        cudaMemcpyAsync(
          B_copy->dptr(), ubuf_chunks[_tp_id].dptr(),
          ubuf_chunks[_tp_id].numel() * ubuf_chunks[_tp_id].element_size(),
          cudaMemcpyDeviceToDevice, _stream_send));
      NVTE_CHECK_CUDA(cudaEventRecord(_stop_send, _stream_send));
      NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_send, 0));
    }

    // Reset atomic counters
    consumer_batch(counter_ptr, 1, _tp_size, stream_main);

    // Copy the first GEMM output chunk to the end chunk position of the output
    char *src_ptr = reinterpret_cast<char *>(D_buffer->dptr());
    NVTE_CHECK_CUDA(
        cudaMemcpyAsync(
        src_ptr + (D->numel() * D->element_size()),
        src_ptr,
        n_chunk * m * D->element_size(),
        cudaMemcpyDeviceToDevice,
        stream_main));
  }  // atomic_gemm_overlap_ag

  /*
  ** Split AllGather + GEMM using P2P communication
  ** This function assumes the input_b is pre-copied to _ubufs[rank_id]. This is
  ** needed to have AG outputs
  ** in each rank to be in the contiguous memory space after all ring exchange
  ** phases.
  */
  void split_gemm_overlap_ag(cudaStream_t stream_main,
    TensorWrapper *A, bool A_trans, TensorWrapper *B, bool B_trans, TensorWrapper *bias,
    TensorWrapper *D, TensorWrapper *pre_gelu_out,
    TensorWrapper *ubuf,  TensorWrapper *B_copy,
    TensorWrapper *workspace, bool grad, bool accumulate, bool use_split_accumulator
  ) {
    _ub_comm->use_ce = _use_ce;
    _ub_comm->sms = _comm_sms;
    _ub_comm->cga_size = _cga_size;

    // Get GEMM dimensions between TN and NN input layouts
    const size_t m = (A_trans) ? A->size(0) : A->size(1);
    const size_t k = (A_trans) ? A->size(1) : A->size(0);
    auto ubuf_chunks = get_ubuf_chunks(ubuf);
    const size_t n_chunk = ubuf_chunks[0].size(0);

    // Get communication and GEMM output chunk sizes
    const int comm_bytes = ubuf_chunks[0].numel() * ubuf_chunks[0].element_size();
    const bool do_gelu = pre_gelu_out->numel() > 0;
    const int output_chunk_bytes = (do_gelu
                                    ? (n_chunk * m) * D->element_size()
                                    : (n_chunk * m) * HALF_BYTES);
    const int aux_chunk_bytes = do_gelu ? (n_chunk * m) * pre_gelu_out->element_size() : 0;

    // Get output and workspace data pointers
    char *output_ptr = reinterpret_cast<char *>(D->dptr());
    char *pre_gelu_out_ptr = reinterpret_cast<char *>(pre_gelu_out->dptr());
    char *workspace_ptr = reinterpret_cast<char *>(workspace->dptr());
    size_t workspace_size_chunk = workspace->numel() / _stream_compute.size();

    NVTE_CHECK_CUDA(cudaEventRecord(_start_compute, stream_main));

    if (_aggregate) {
      // Catch up the default torch stream
      NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_send, _start_compute, 0));
      NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_recv, _start_compute, 0));

      const int num_steps = _tp_size / 2;
      char *input_b_ptr = reinterpret_cast<char *>(ubuf->dptr());

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
          reinterpret_cast<void *>(input_b_ptr + send_offset),
          {n_chunk * 2, k}, ubuf->dtype(), nullptr, nullptr, ubuf->scale_inv());
        TensorWrapper output_chunk = TensorWrapper(
          reinterpret_cast<void *>(output_ptr + (send_chunk_id * output_chunk_bytes)),
          {n_chunk * 2, m}, D->dtype(), D->amax(), D->scale(), nullptr);
        TensorWrapper pre_gelu_chunk = (do_gelu)
          ? TensorWrapper(reinterpret_cast<void *>(
              pre_gelu_out_ptr + (send_chunk_id * aux_chunk_bytes)),
              {n_chunk * 2, m}, pre_gelu_out->dtype())
          : TensorWrapper(nullptr, std::vector<size_t>{0}, DType::kByte);
        TensorWrapper work_chunk = TensorWrapper(
          reinterpret_cast<void *>(
            workspace_ptr + ((i % _stream_compute.size()) * workspace_size_chunk)),
            {workspace_size_chunk}, workspace->dtype());
        nvte_cublas_gemm(A->data(), input_b_chunk.data(), output_chunk.data(),
                        bias->data(), pre_gelu_chunk.data(), A_trans, B_trans, grad,
                        work_chunk.data(), accumulate, use_split_accumulator, _math_sms,
                        _stream_compute[i % _stream_compute.size()]);

        if (i < num_steps - 1) {
          // P2P communication
          userbuffers_send(_ub_reg, send_offset, _ub_reg, send_offset, comm_bytes * 2, _ub_comm,
                          next_rank, _stream_send);
          userbuffers_recv(_ub_reg, recv_offset, _ub_reg, recv_offset, comm_bytes * 2, _ub_comm,
                          prev_rank, _stream_recv);
          NVTE_CHECK_CUDA(cudaEventRecord(_stop_recv, _stream_recv));
          NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_send, _stop_recv, 0));
          NVTE_CHECK_CUDA(cudaStreamWaitEvent(
              _stream_compute[(i + 1) % _stream_compute.size()], _stop_recv, 0));
        } else if (B_copy->numel() > 0) {
          assert(B_copy->numel() == ubuf_chunks[_tp_id].numel());
          assert(B_copy->element_size() == ubuf_chunks[_tp_id].element_size());
          NVTE_CHECK_CUDA(
            cudaMemcpyAsync(B_copy->dptr(), ubuf_chunks[_tp_id].dptr(),
                            ubuf_chunks[_tp_id].numel() * ubuf_chunks[_tp_id].element_size(),
                            cudaMemcpyDeviceToDevice, _stream_send));
          NVTE_CHECK_CUDA(cudaEventRecord(_stop_send, _stream_send));
          NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_send, 0));
        }
      }

      int last_compute_stream_id =
          (num_steps + _stream_compute.size() - 1) % _stream_compute.size();
      NVTE_CHECK_CUDA(
          cudaEventRecord(_stop_compute, _stream_compute[last_compute_stream_id]));
    } else {
      // Catch up the default torch stream
      NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_send, _start_compute, 0));
      NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_recv, _start_compute, 0));
      NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_compute[0], _start_compute, 0));

      for (int i = 0; i < _tp_size; i++) {
        // Set the userbuffer iD-> Buffer under send is the input for the current
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
          {n_chunk, m}, D->dtype(), D->amax(), D->scale(), nullptr);
        TensorWrapper pre_gelu_chunk = (do_gelu)
          ? TensorWrapper(
            reinterpret_cast<void *>(pre_gelu_out_ptr + (send_chunk_id * aux_chunk_bytes)),
            {n_chunk, m}, pre_gelu_out->dtype())
          : TensorWrapper(nullptr, std::vector<size_t>{0}, DType::kByte);
        TensorWrapper work_chunk = TensorWrapper(
          reinterpret_cast<void *>(
            workspace_ptr + ((i % _stream_compute.size()) * workspace_size_chunk)),
            {workspace_size_chunk}, workspace->dtype());
        nvte_cublas_gemm(A->data(), ubuf_chunks[send_chunk_id].data(), output_chunk.data(),
                        bias->data(), pre_gelu_chunk.data(), A_trans, B_trans, grad,
                        work_chunk.data(), accumulate, use_split_accumulator, _math_sms,
                        _stream_compute[i % _stream_compute.size()]);

        if (i < _tp_size - 1) {
          // P2P communication
          userbuffers_send(_ub_reg, send_offset, _ub_reg, send_offset, comm_bytes, _ub_comm,
                          _next_rank, _stream_send);
          userbuffers_recv(_ub_reg, recv_offset, _ub_reg, recv_offset, comm_bytes, _ub_comm,
                          _prev_rank, _stream_recv);
          NVTE_CHECK_CUDA(cudaEventRecord(_stop_recv, _stream_recv));
          NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_send, _stop_recv, 0));
          NVTE_CHECK_CUDA(cudaStreamWaitEvent(
              _stream_compute[(i + 1) % _stream_compute.size()], _stop_recv, 0));
        } else if (B_copy->numel() > 0) {
          assert(B_copy->numel() == ubuf_chunks[_tp_id].numel());
          assert(B_copy->element_size() == ubuf_chunks[_tp_id].element_size());
          NVTE_CHECK_CUDA(
            cudaMemcpyAsync(B_copy->dptr(), ubuf_chunks[_tp_id].dptr(),
                            ubuf_chunks[_tp_id].numel() * ubuf_chunks[_tp_id].element_size(),
                            cudaMemcpyDeviceToDevice, _stream_send));
          NVTE_CHECK_CUDA(cudaEventRecord(_stop_send, _stream_send));
          NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_send, 0));
        }
      }

      int last_compute_stream_id = (_tp_size + _stream_compute.size() - 1) % _stream_compute.size();
      NVTE_CHECK_CUDA(
          cudaEventRecord(_stop_compute, _stream_compute[last_compute_stream_id]));
    }
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_compute, 0));
  }  // split_gemm_overlap_ag

  /*
  ** Split ReduceScatter + GEMM using P2P communication
  */
  void atomic_gemm_overlap_rs(cudaStream_t stream_main,
    TensorWrapper *A, bool A_trans, TensorWrapper *B, bool B_trans, TensorWrapper *bias,
    TensorWrapper *D, TensorWrapper *pre_gelu_out,
    TensorWrapper *ubuf, TensorWrapper *counters, TensorWrapper *rs_out,
    TensorWrapper *workspace, bool grad, bool accumulate, bool use_split_accumulator
  ) {
    _ub_comm->use_ce = _use_ce;
    _ub_comm->sms = _comm_sms;
    _ub_comm->cga_size = _cga_size;

    // Get communication and GEMM input chunk sizes
    auto ubuf_chunks = get_ubuf_chunks(ubuf);
    const int comm_bytes = ubuf_chunks[0].numel() * ubuf_chunks[0].element_size();

    // Get input and workspace data pointers
    char *workspace_ptr = reinterpret_cast<char *>(workspace->dptr());
    int *counter_ptr = reinterpret_cast<int *>(counters->dptr());
    size_t workspace_size_chunk = workspace->numel() / _stream_compute.size();

    // Catch up the main stream
    NVTE_CHECK_CUDA(cudaEventRecord(_start_compute, stream_main));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_recv, _start_compute, 0));

    // Atomic GEMM
    // Process GEMM chunks in the order that AG+GEMM places the output chunks.
    TensorWrapper workspace_chunk = TensorWrapper(
      reinterpret_cast<void *>(workspace_ptr), {workspace_size_chunk}, workspace->dtype());
    nvte_cublas_atomic_gemm(A->data(), B->data(), ubuf->data(), bias->data(), pre_gelu_out->data(),
                            A_trans, B_trans, grad, workspace_chunk.data(), accumulate,
                            use_split_accumulator, _math_sms, 0, _tp_size, true, counters->data(),
                            stream_main);

    // P2P communication chunk
    char *rs_out_ptr = reinterpret_cast<char *>(rs_out->dptr());
    for (int i = 1; i < _tp_size; i++) {
      int send_chunk_id = i - 1;
      int recv_chunk_id = send_chunk_id + _tp_size;
      int send_offset = comm_bytes * send_chunk_id;
      int recv_offset = comm_bytes * recv_chunk_id;
      int send_rank = (_tp_size + _tp_id - i) % _tp_size + _rank_round_tp;
      int recv_rank = (_tp_id + i) % _tp_size + _rank_round_tp;

      consumer(counter_ptr, send_chunk_id, _stream_recv);
      userbuffers_send(_ub_reg, send_offset, _ub_reg, recv_offset, comm_bytes,
                      _ub_comm, send_rank, _stream_recv);
      userbuffers_recv(_ub_reg, send_offset, _ub_reg, recv_offset, comm_bytes,
                      _ub_comm, recv_rank, _stream_recv);
    }
    NVTE_CHECK_CUDA(cudaEventRecord(_stop_recv, _stream_recv));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_recv, 0));

    // Reduce GEMM output chunks
    char *reduce_buf_ptr = reinterpret_cast<char *>(ubuf_chunks[_tp_size - 1].dptr());
    if (is_fp8_dtype(ubuf->dtype())) {
      assert(!is_fp8_dtype(rs_out->dtype()));
      reduce_fp8_in_bf16_out<__nv_fp8_e4m3>(reduce_buf_ptr, rs_out_ptr, ubuf->scale_inv(),
                                            _tp_size, ubuf_chunks[_tp_size - 1].numel(),
                                            stream_main);
    } else {
      if (ubuf->dtype() == DType::kFloat32) {
        reduce_bf16_out<float>(
          reinterpret_cast<void *>(reduce_buf_ptr), reinterpret_cast<void *>(rs_out_ptr),
          _tp_size, static_cast<int>(ubuf_chunks[_tp_size - 1].numel()), stream_main);
      } else {
        reduce_bf16_out<half>(
            reinterpret_cast<void *>(reduce_buf_ptr), reinterpret_cast<void *>(rs_out_ptr),
            _tp_size, static_cast<int>(ubuf_chunks[_tp_size - 1].numel()), stream_main);
      }
    }
  }  // atomic_gemm_overlap_rs

  /*
  ** Split ReduceScatter + GEMM using P2P communication
  */
  void split_gemm_overlap_rs(cudaStream_t stream_main,
    TensorWrapper *A, bool A_trans, TensorWrapper *B, bool B_trans, TensorWrapper *bias,
    TensorWrapper *D, TensorWrapper *pre_gelu_out,
    TensorWrapper *ubuf,  TensorWrapper *rs_out,
    TensorWrapper *workspace, bool grad, bool accumulate, bool use_split_accumulator
  ) {
    _ub_comm->use_ce = _use_ce;
    _ub_comm->sms = _comm_sms;
    _ub_comm->cga_size = _cga_size;
    size_t k = A->size(1);
    size_t n = B->size(0);

    // Get communication and GEMM input chunk sizes
    auto ubuf_chunks = get_ubuf_chunks(ubuf);
    size_t n_chunk = n / _tp_size;
    const int comm_bytes = ubuf_chunks[0].numel() * ubuf_chunks[0].element_size();
    const int input_b_chunk_bytes = n_chunk * k * B->element_size();

    // Get input and workspace data pointers
    char *input_b_ptr = reinterpret_cast<char *>(B->dptr());
    char *workspace_ptr = reinterpret_cast<char *>(workspace->dptr());
    size_t workspace_size_chunk = workspace->numel() / _stream_compute.size();

    // Catch up the main stream
    NVTE_CHECK_CUDA(cudaEventRecord(_start_compute, stream_main));
    for (size_t i = 0; i < _stream_compute.size(); i++) {
        NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_compute[i], _start_compute, 0));
    }

    // GEMM and send/recv chunks
    char *rs_out_ptr = reinterpret_cast<char *>(rs_out->dptr());
    for (int i = 0; i < _tp_size; i++) {
      // GEMM chunk
      int input_b_chunk_id = (_tp_id + i + 1) % _tp_size;
      char* input_b_chunk_ptr = input_b_ptr + (input_b_chunk_id * input_b_chunk_bytes);
      TensorWrapper input_b_chunk = TensorWrapper(
        reinterpret_cast<void *>(input_b_chunk_ptr), {n_chunk, k},
        B->dtype(), B->scale(), B->scale_inv(), B->amax());
      // Store the last GEMM chunk output to the recieve buffer.
      TensorWrapper workspace_chunk = TensorWrapper(
        reinterpret_cast<void*>(
          workspace_ptr + ((i % _stream_compute.size()) * workspace_size_chunk)),
          {workspace_size_chunk}, workspace->dtype());
      cudaStream_t gemm_stream = (i == _tp_size - 1) ? stream_main
                                                    : _stream_compute[i % _stream_compute.size()];
      nvte_cublas_gemm(A->data(), input_b_chunk.data(), ubuf_chunks[i].data(),
                      bias->data(), pre_gelu_out->data(), A_trans, B_trans, grad,
                      workspace_chunk.data(), accumulate, use_split_accumulator, _math_sms,
                      gemm_stream);

      if (i > 0) {
          // P2P communication chunk
          int send_offset = comm_bytes * (i - 1);
          int recv_offset = comm_bytes * (i - 1 + _tp_size);
          int send_rank = (_tp_id + i) % _tp_size + _rank_round_tp;
          int recv_rank = (_tp_size + _tp_id - i) % _tp_size + _rank_round_tp;
          NVTE_CHECK_CUDA(cudaEventRecord(
              _start_comm,  _stream_compute[(i - 1) % _stream_compute.size()]));
          NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_send, _start_comm, 0));
          NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_recv, _start_comm, 0));
          userbuffers_send(_ub_reg, send_offset, _ub_reg, recv_offset, comm_bytes,
                          _ub_comm, send_rank, _stream_send);
          userbuffers_recv(_ub_reg, send_offset, _ub_reg, recv_offset, comm_bytes,
                          _ub_comm, recv_rank, _stream_recv);
      }
    }
    NVTE_CHECK_CUDA(cudaEventRecord(_stop_recv,  _stream_recv));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_recv, 0));

    // Reduce GEMM output chunks
    char *reduce_buf_ptr = reinterpret_cast<char *>(ubuf_chunks[_tp_size - 1].dptr());
    if (is_fp8_dtype(ubuf->dtype())) {
      assert(!is_fp8_dtype(rs_out->dtype()));
      reduce_fp8_in_bf16_out<__nv_fp8_e4m3>(reduce_buf_ptr, rs_out_ptr, ubuf->scale_inv(),
                                            _tp_size, ubuf_chunks[_tp_size - 1].numel(),
                                            stream_main);
    } else {
      if (ubuf->dtype() == DType::kFloat32) {
        reduce_bf16_out<float>(
          reinterpret_cast<void *>(reduce_buf_ptr), reinterpret_cast<void *>(rs_out_ptr),
          _tp_size, static_cast<int>(ubuf_chunks[_tp_size - 1].numel()), stream_main);
      } else {
        reduce_bf16_out<half>(
            reinterpret_cast<void *>(reduce_buf_ptr), reinterpret_cast<void *>(rs_out_ptr),
            _tp_size, static_cast<int>(ubuf_chunks[_tp_size - 1].numel()), stream_main);
      }
    }

    for (size_t i = 0; i < _stream_compute.size(); i++) {
      NVTE_CHECK_CUDA(cudaEventRecord(_stop_compute, _stream_compute[i]));
      NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_compute, 0));
    }
    NVTE_CHECK_CUDA(cudaEventRecord(_stop_send, _stream_send));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_send, 0));
  }  // split_gemm_overlap_rs
};  // CommGemmOverlapP2P

}  // namespace userbuffers

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_USERBUFFERS_COMM_GEMM_OVERLAP_H_
