/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_COMM_GEMM_OVERLAP_H_
#define TRANSFORMER_ENGINE_COMMON_COMM_GEMM_OVERLAP_H_

#include <cuda.h>
#include <cuda_fp8.h>
#include <transformer_engine/transformer_engine.h>

#include <functional>

#include "common/comm_gemm_overlap/userbuffers/userbuffers.h"

#define NVTE_COMM_OVERLAP_MAX_STREAMS 3

namespace transformer_engine {

/* \brief Check if Userbufers bootstraps with direct calls to MPI collectives.
 *        This can turned on by building Transformer Engine with the `NVTE_UB_WITH_MPI=1` option.
 *
 * \return True if Userbuffers is built with MPI
 */
bool ubuf_built_with_mpi();

enum class CommOverlapType { RS = 0, AG = 1 };

enum class CommOverlapAlgo {
  BULK_OVERLAP_AG = 0,
  BULK_OVERLAP_RS = 1,
  SPLIT_PIPELINED_AG_P2P = 2,
  SPLIT_PIPELINED_RS = 3,
  SPLIT_PIPELINED_RS_P2P = 4,
  ATOMIC_GEMM_RS = 5,
  ATOMIC_GEMM_AG_P2P = 6,
  ATOMIC_GEMM_RS_P2P = 7
};

class CommOverlapCore {
 protected:
  static inline communicator *_ub_comm{nullptr};
  static inline bool _comm_created{false};

  int _rank;
  int _tp_id;
  int _tp_size;
  int _num_splits;
  int _math_sms;
  int _num_comm_sm;
  int _cga_size;
  int _use_ce;
  int _ub_reg;
  int _gemm_priority;
  int _comm_priority;
  bool _atomic_gemm{false};
  bool _is_p2p{false};

  TensorWrapper _ubuf;
  TensorWrapper _counter;
  float *_ubuf_scale_inv;
  bool _ubuf_scale_inv_initialized{false};

  std::vector<cudaStream_t> _stream_compute;
  cudaEvent_t _start_compute, _stop_compute, _start_comm, _stop_comm, _comm_launch_event;

 public:
  CommOverlapCore() {}  // dummy constructor for exposing type to Python

  CommOverlapCore(int myrank, int numranks, int mylocal, int numlocal, int mynode, int numnodes,
                  int tp_size, ExtAllgatherOp allgather_handle, ExtBarrierOp barrier_handle,
                  int num_splits, int num_max_streams, int comm_cga_size, int gemm_priority,
                  int comm_priority, int num_comm_sm, bool set_sm_margin, bool use_ce,
                  bool atomic_gemm);

  virtual ~CommOverlapCore();

  void set_ubuf_scale_inv(float *scale_inv) {
    _ubuf_scale_inv = scale_inv;
    _ubuf_scale_inv_initialized = true;
  }

  TensorWrapper get_tensor_chunk(const TensorWrapper &source, size_t offset,
                                 const std::vector<size_t> &shape);

  TensorWrapper get_buffer_chunk_like(const TensorWrapper &source, size_t offset,
                                      const std::vector<size_t> &shape);

  bool is_atomic_gemm() { return _atomic_gemm; }

  bool is_p2p_overlap() { return _is_p2p; }

  bool is_fp8_ubuf() { return _ubuf.element_size() == 1; }

  virtual void bulk_overlap(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                            bool transb, TensorWrapper &D, TensorWrapper &bias,
                            TensorWrapper &pre_gelu_out, TensorWrapper &workspace, bool grad,
                            bool accumulate, bool use_split_accumulator, CommOverlapType comm_type,
                            TensorWrapper &rs_output, cudaStream_t stream_main) {
    NVTE_ERROR("Operation is not implemented.");
  }

  virtual void atomic_gemm_overlap_rs(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                                      bool transb, TensorWrapper &D, TensorWrapper &bias,
                                      TensorWrapper &pre_gelu_out, TensorWrapper &workspace,
                                      bool grad, bool accumulate, bool use_split_accumulator,
                                      TensorWrapper &rs_output, cudaStream_t stream_main) {
    NVTE_ERROR("Operation is not implemented.");
  }

  virtual void split_overlap_rs(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                                bool transb, TensorWrapper &D, TensorWrapper &bias,
                                TensorWrapper &pre_gelu_out, TensorWrapper &workspace, bool grad,
                                bool accumulate, bool use_split_accumulator,
                                TensorWrapper &rs_output, cudaStream_t stream_main) {
    NVTE_ERROR("Operation is not implemented.");
  }

  virtual void atomic_gemm_overlap_ag(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                                      bool transb, TensorWrapper &D, TensorWrapper &bias,
                                      TensorWrapper &pre_gelu_out, TensorWrapper &workspace,
                                      bool grad, bool accumulate, bool use_split_accumulator,
                                      TensorWrapper &B_copy, cudaStream_t stream_main) {
    NVTE_ERROR("Operation is not implemented.");
  }

  virtual void split_overlap_ag(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                                bool transb, TensorWrapper &D, TensorWrapper &bias,
                                TensorWrapper &pre_gelu_out, TensorWrapper &workspace, bool grad,
                                bool accumulate, bool use_split_accumulator, TensorWrapper &B_copy,
                                cudaStream_t stream_main) {
    NVTE_ERROR("Operation is not implemented.");
  }
};  // CommOverlapCore

class CommOverlapBase : public CommOverlapCore {
 protected:
  int _rs_kernel_type;
  bool _rs_overlap_first_gemm;
  cudaStream_t _stream_comm;
  cudaEvent_t _start_d2dcopy;

 public:
  CommOverlapBase() {}  // dummy constructor for exposing type to Python

  CommOverlapBase(const std::vector<size_t> &buffer_shape, DType buffer_dtype, int myrank,
                  int numranks, int mylocal, int numlocal, int mynode, int numnodes, int tp_size,
                  ExtAllgatherOp allgather_handle, ExtBarrierOp barrier_handle, int num_splits = 3,
                  int num_max_streams = NVTE_COMM_OVERLAP_MAX_STREAMS, int comm_cga_size = 2,
                  int gemm_priority = 0, int comm_priority = 0, int num_comm_sm = 16,
                  bool set_sm_margin = true, bool atomic_gemm = false,
                  bool rs_overlap_first_gemm = false);

  virtual ~CommOverlapBase();

  /*
  ** Bulk GEMM + COMM
  ** This function assumes the communication input is pre-copied to _ubuf
  */
  void bulk_overlap(const TensorWrapper &A, bool transa, const TensorWrapper &B, bool transb,
                    TensorWrapper &D, TensorWrapper &bias, TensorWrapper &pre_gelu_out,
                    TensorWrapper &workspace, bool grad, bool accumulate,
                    bool use_split_accumulator, CommOverlapType comm_type, TensorWrapper &rs_output,
                    cudaStream_t stream_main) override;

  void atomic_gemm_overlap_ag(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                              bool transb, TensorWrapper &D, TensorWrapper &bias,
                              TensorWrapper &pre_gelu_out, TensorWrapper &workspace, bool grad,
                              bool accumulate, bool use_split_accumulator, TensorWrapper &B_copy,
                              cudaStream_t stream_main) override {
    NVTE_ERROR("Operation not supported.");
  }

  void split_overlap_ag(const TensorWrapper &A, bool transa, const TensorWrapper &B, bool transb,
                        TensorWrapper &D, TensorWrapper &bias, TensorWrapper &pre_gelu_out,
                        TensorWrapper &workspace, bool grad, bool accumulate,
                        bool use_split_accumulator, TensorWrapper &B_copy,
                        cudaStream_t stream_main) override {
    NVTE_ERROR("Operation not supported.");
  }

  /*
  ** Split FPROP GEMM + ReduceScatter
  */
  void atomic_gemm_overlap_rs(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                              bool transb, TensorWrapper &D, TensorWrapper &bias,
                              TensorWrapper &pre_gelu_out, TensorWrapper &workspace, bool grad,
                              bool accumulate, bool use_split_accumulator, TensorWrapper &rs_output,
                              cudaStream_t stream_main) override;

  /*
  ** Split FPROP GEMM + ReduceScatter
  */
  void split_overlap_rs(const TensorWrapper &A, bool transa, const TensorWrapper &B, bool transb,
                        TensorWrapper &D, TensorWrapper &bias, TensorWrapper &pre_gelu_out,
                        TensorWrapper &workspace, bool grad, bool accumulate,
                        bool use_split_accumulator, TensorWrapper &rs_output,
                        cudaStream_t stream_main) override;
};  // CommOverlapBase

class CommOverlapP2PBase : public CommOverlapCore {
 protected:
  bool _is_reduce_scatter{false};
  bool _use_multiatomic_ag{false};
  bool _aggregate;
  int _next_rank;
  int _prev_rank;
  int _rank_round_tp;
  int _num_ubuf_chunks;
  int _self_chunk_id;
  std::vector<TensorWrapper> _ubufs;
  std::vector<cudaStream_t> _stream_send;
  cudaStream_t _stream_recv;
  cudaEvent_t _stop_send, _stop_recv;

 public:
  CommOverlapP2PBase() {}  // dummy constructor for exposing type to Python

  CommOverlapP2PBase(const std::vector<size_t> &buffer_shape, DType buffer_dtype, int myrank,
                     int numranks, int mylocal, int numlocal, int mynode, int numnodes, int tp_size,
                     ExtAllgatherOp allgather_handle, ExtBarrierOp barrier_handle,
                     CommOverlapType comm_type, int num_max_streams = NVTE_COMM_OVERLAP_MAX_STREAMS,
                     int comm_cga_size = 1, int gemm_priority = 0, int comm_priority = 0,
                     int num_comm_sm = 1, bool set_sm_margin = false, bool use_ce = true,
                     bool atomic_gemm = false, bool aggregate = false);

  virtual ~CommOverlapP2PBase();

  TensorWrapper get_buffer_chunk_by_id(const TensorWrapper &source, size_t buffer_id);

  void bulk_overlap(const TensorWrapper &A, bool transa, const TensorWrapper &B, bool transb,
                    TensorWrapper &D, TensorWrapper &bias, TensorWrapper &pre_gelu_out,
                    TensorWrapper &workspace, bool grad, bool accumulate,
                    bool use_split_accumulator, CommOverlapType comm_type, TensorWrapper &rs_output,
                    cudaStream_t stream_main) override {
    NVTE_ERROR("Operation not supported.");
  }

  /*
  ** Split AllGather + AtomicGEMM using P2P communication
  ** This function assumes the input_b is pre-copied to _ubufs[rank_id]. This is needed to have AG
  ** outputs in each rank to be in the contiguous memory space after all ring exchange phases.
  */
  void atomic_gemm_overlap_ag(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                              bool transb, TensorWrapper &D, TensorWrapper &bias,
                              TensorWrapper &pre_gelu_out, TensorWrapper &workspace, bool grad,
                              bool accumulate, bool use_split_accumulator, TensorWrapper &B_copy,
                              cudaStream_t stream_main) override;

  /*
  ** Split AllGather + GEMM using P2P communication
  ** This function assumes the input_b is pre-copied to _ubufs[rank_id]. This is needed to have AG
  ** outputs in each rank to be in the contiguous memory space after all ring exchange phases.
  */
  void split_overlap_ag(const TensorWrapper &A, bool transa, const TensorWrapper &B, bool transb,
                        TensorWrapper &D, TensorWrapper &bias, TensorWrapper &pre_gelu_out,
                        TensorWrapper &workspace, bool grad, bool accumulate,
                        bool use_split_accumulator, TensorWrapper &B_copy,
                        cudaStream_t stream_main) override;

  /*
  ** Split ReduceScatter + GEMM using P2P communication
  */
  void atomic_gemm_overlap_rs(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                              bool transb, TensorWrapper &D, TensorWrapper &bias,
                              TensorWrapper &pre_gelu_out, TensorWrapper &workspace, bool grad,
                              bool accumulate, bool use_split_accumulator, TensorWrapper &rs_output,
                              cudaStream_t stream_main) override;

  /*
  ** Split ReduceScatter + GEMM using P2P communication
  */
  void split_overlap_rs(const TensorWrapper &A, bool transa, const TensorWrapper &B, bool transb,
                        TensorWrapper &D, TensorWrapper &bias, TensorWrapper &pre_gelu_out,
                        TensorWrapper &workspace, bool grad, bool accumulate,
                        bool use_split_accumulator, TensorWrapper &rs_output,
                        cudaStream_t stream_main) override;
};  // CommOverlapP2PBase

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_COMM_GEMM_OVERLAP_H_
