/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_COMM_GEMM_OVERLAP_H_
#define TRANSFORMER_ENGINE_COMMON_COMM_GEMM_OVERLAP_H_

// Standard library
#include <functional>

// External includes
#include <cuda_runtime_api.h>

// TE includes
#include <transformer_engine/transformer_engine.h>

#include "common/comm_gemm_overlap/userbuffers/userbuffers.h"

#define NVTE_COMM_OVERLAP_MAX_STREAMS 3

#ifdef __cplusplus

namespace transformer_engine {

/*! \brief Check if the platform has CUDA Multicast support.
 *
 *  CUDA Multicast requires:
 *  - Device compute capability 9.0+
 *  - CUDA driver major version 535+
 *  - CUDA Toolkit version 12.2+
 */
bool device_supports_multicast();

/*! \brief Check if Userbuffers has been built with MPI.
 */
bool userbuffers_built_with_mpi();

enum class CommOverlapType {
  ALL_GATHER = 0,
  REDUCE_SCATTER = 1
};

enum class CommOverlapBuffer {
  GLOBAL = 0,
  LOCAL = 1
};

enum class CommOverlapAlgo {
  // bulk overlaps (no dependence between comm and compute)
  BULK_AG = 0,  // GEMM + all-gather
  BULK_RS = 1,  // GEMM + reduce-scatter

  // producer-consumer overlaps
  // ................ // producer                 | consumer
  // ===========================================================================
  SPLIT_AG_P2P = 2,   // point-2-point all-gather | split GEMM
  SPLIT_RS = 3,       // split GEMM               | collective reduce-scatter
  SPLIT_RS_P2P = 4,   // split GEMM               | point-2-point reduce-scatter
  ATOMIC_AG_P2P = 5,  // point-2-point all-gather | atomic GEMM
  ATOMIC_RS = 6,      // atomic GEMM              | collective reduce-scatter
  ATOMIC_RS_P2P = 7   // atomic GEMM              | point-2-point reduce-scatter
};

class CommOverlapBase {
 protected:
  static inline userbuffers::communicator *_ub_comm{nullptr};
  static inline bool _comm_created{false};

  int _rank;
  int _tp_id, _tp_size;
  int _comm_sms, _math_sms;
  int _ub_reg;
  int _num_splits;
  int _cga_size;
  int _use_ce;
  bool _atomic_gemm{false};
  bool _buffer_registered{false};
  bool _is_p2p{false};
  char _name[32];

  cudaEvent_t _start_compute, _stop_compute, _start_comm, _stop_comm, _start_d2dcopy;
  std::vector<cudaStream_t> _stream_compute;

 public:
  CommOverlapBase(int world_rank, int world_size, int local_rank, int local_size, int node_id,
                  int num_nodes, int tp_size,
                  std::function<void(void *, size_t, void *, size_t, char *)> allgather_handle,
                  std::function<void(char *)> barrier_handle, int num_splits, int num_max_streams,
                  int cga_size, int num_comm_sms, bool set_sm_margin, bool use_ce, bool atomic_gemm,
                  const char *name);

  virtual ~CommOverlapBase();

  // Disallow copy-constructor and copy-assignment
  CommOverlapBase(const CommOverlapBase &other) = delete;
  CommOverlapBase &operator=(const CommOverlapBase &other) = delete;

  void register_gpu_buffer(void **gpuptr, size_t bytes, bool alloc);

  bool is_atomic_gemm() { return _atomic_gemm; }

  bool is_p2p_overlap() { return _is_p2p; }
};  // CommOverlapBase

/*! \struct CommOverlap
 *  \brief Structure to manage and execute collective comm+GEMM overlap algorithms.
 */
class CommOverlap : public CommOverlapBase {
 protected:
  int _rs_kernel_type = 1;
  cudaStream_t _stream_comm;

 public:
  /*! \brief Constructs new CommOverlap object.
   *
   * Create a structure to manage and execute collective (pipelined) comm+GEMM overlap algorithms.
   *
   *  \param[in]  world_rank        Global rank of the process.
   *  \param[in]  world_size        Global number of processes.
   *  \param[in]  local_rank        Local rank of the process on its physical node.
   *  \param[in]  local_size        Local number of processes in this physical node.
   *  \param[in]  node_id           ID of the physical node that hosts this process.
   *  \param[in]  num_nodes         Number of physical nodes.
   *  \param[in]  tp_size           Size of the tensor-parallel group (may be less than localsize).
   *  \param[in]  allgather_handle  Function pointer for external allgather op (e.g. DL framework).
   *  \param[in]  barrier_handle    Function pointer for external barrier op (e.g. DL framework).
   *  \param[in]  num_splits        Number of chunks to split the work into.
   *  \param[in]  num_max_streams   Maximum number of concurrent CUDA streams.
   *  \param[in]  cga_size          cuBlasLt GEMM heuristic cluster size.
   *  \param[in]  num_comm_sms      Number of SMs to use for communication.
   *  \param[in]  set_sm_margin     Flag for reserving communication SMs (subtracts from math SMs).
   *  \param[in]  atomic_gemm       Use atomic GEMM.
   */
  CommOverlap(int world_rank, int world_size, int local_rank, int local_size, int node_id,
              int num_nodes, int tp_size,
              std::function<void(void *, size_t, void *, size_t, char *)> allgather_handle,
              std::function<void(char *)> barrier_handle, int num_splits = 4,
              int num_max_streams = NVTE_COMM_OVERLAP_MAX_STREAMS, int cga_size = 2,
              int num_comm_sms = 16, bool set_sm_margin = true, bool atomic_gemm = false);

  ~CommOverlap();

  /*! \brief Overlap GEMM compute with an independent collective on a tensor not involved in GEMM.
   *
   * This algorithm assumes that the B matrix is pre-copied to the `ubuf` workspace.
   *
   *  \param[in]      stream                 CUDA stream used for the operation.
   *  \param[in]      A                      The A matrix.
   *  \param[in]      A_trans                Whether A matrix is transposed.
   *  \param[in]      B                      The B matrix.
   *  \param[in]      transb                 Whether B matrix is transposed.
   *  \param[in]      bias                   Bias tensor.
   *  \param[out]     D                      Output matrix.
   *  \param[out]     pre_gelu_out           Output matrix before GELU activation.
   *  \param[in,out]  ubuf                   Userbuffers workspace.
   *  \param[out]     rs_output              Output tensor for the reduce-scattered result.
   *  \param[in]      workspace              GEMM Workspace tensor.
   *  \param[in]      grad                   Whether this operation is part of the
   *                                         gradient computation.
   *  \param[in]      accumulate             Whether to accumulate the result into the D matrix.
   *  \param[in]      use_split_accumulator  Whether to use split accumulator in the FP8 GEMM.
   *  \param[in]      comm_type              Whether to overlap All-Gather or Reduce-Scatter.
   */
  void bulk_overlap(cudaStream_t stream_main, const TensorWrapper &A, bool A_trans,
                    const TensorWrapper &B, bool B_trans, const TensorWrapper &bias,
                    const TensorWrapper &D, const TensorWrapper &pre_gelu_out,
                    const TensorWrapper &ubuf, const TensorWrapper &rs_output,
                    const TensorWrapper &workspace, bool grad, bool accumulate,
                    bool use_split_accumulator, CommOverlapType comm_type);

  /*! \brief Overlap atomic GEMM (producer) with reduce-scatter (consumer).
   *
   *  \param[in]   stream                 CUDA stream used for the operation.
   *  \param[in]   A                      The A matrix.
   *  \param[in]   A_trans                Whether A matrix is transposed.
   *  \param[in]   B                      The B matrix.
   *  \param[in]   transb                 Whether B matrix is transposed.
   *  \param[in]   bias                   Bias tensor.
   *  \param[out]  D                      Output matrix.
   *  \param[out]  pre_gelu_out           Output matrix before GELU activation.
   *  \param[out]  ubuf                   Userbuffers workspace.
   *  \param[in]   counters               Atomic flags.
   *  \param[out]  rs_output              Output tensor for the reduce-scattered result.
   *  \param[in]   workspace              GEMM Workspace tensor.
   *  \param[in]   grad                   Whether this operation is part of the
   *                                      gradient computation.
   *  \param[in]   accumulate             Whether to accumulate the result into the D matrix.
   *  \param[in]   use_split_accumulator  Whether to use split accumulator in the FP8 GEMM.
   */
  void atomic_gemm_overlap_rs(cudaStream_t stream_main, const TensorWrapper &A, bool A_trans,
                              const TensorWrapper &B, bool B_trans, const TensorWrapper &bias,
                              const TensorWrapper &D, const TensorWrapper &pre_gelu_out,
                              const TensorWrapper &ubuf, const TensorWrapper &counters,
                              const TensorWrapper &rs_output, const TensorWrapper &workspace,
                              bool grad, bool accumulate, bool use_split_accumulator);

  /*! \brief Overlap split GEMM (producer) with reduce-scatter (consumer).
   *
   *  \param[in]   stream                 CUDA stream used for the operation.
   *  \param[in]   A                      The A matrix.
   *  \param[in]   A_trans                Whether A matrix is transposed.
   *  \param[in]   B                      The B matrix.
   *  \param[in]   transb                 Whether B matrix is transposed.
   *  \param[in]   bias                   Bias tensor.
   *  \param[out]  D                      Output matrix.
   *  \param[out]  pre_gelu_out           Output matrix before GELU activation.
   *  \param[out]  ubuf                   Userbuffers workspace.
   *  \param[out]  rs_output              Output tensor for the reduce-scattered result.
   *  \param[in]   workspace              GEMM Workspace tensor.
   *  \param[in]   grad                   Whether this operation is part of the
   *                                      gradient computation.
   *  \param[in]   accumulate             Whether to accumulate the result into the D matrix.
   *  \param[in]   use_split_accumulator  Whether to use split accumulator in the FP8 GEMM.
   */
  void split_gemm_overlap_rs(cudaStream_t stream_main, const TensorWrapper &A, bool A_trans,
                             const TensorWrapper &B, bool B_trans, const TensorWrapper &bias,
                             const TensorWrapper &D, const TensorWrapper &pre_gelu_out,
                             const TensorWrapper &ubuf, const TensorWrapper &rs_output,
                             const TensorWrapper &workspace, bool grad, bool accumulate,
                             bool use_split_accumulator, bool gemm_overlap);

};  // CommOverlap

/*! \struct CommOverlapP2P
 *  \brief Structure to manage and execute point-to-point comm+GEMM overlap algorithms.
 */
class CommOverlapP2P : public CommOverlapBase {
 protected:
  bool _aggregate{false};
  bool _is_reduce_scatter{false};
  bool _ag_sendrecv_multiatomic{false};
  int _next_rank, _prev_rank, _rank, _rank_round_tp;

  int _num_ubuf_chunks, _self_chunk_id;
  cudaStream_t _stream_send, _stream_recv;
  cudaEvent_t _stop_send, _stop_recv;

 public:
  /*! \brief Constructs new CommOverlapP2P object.
   *
   * Create a structure to manage and execute point-to-point (ring-exchange) comm+GEMM overlap
   * algorithms.
   *
   *  \param[in]  world_rank         Global rank of the process.
   *  \param[in]  world_size         Global number of processes.
   *  \param[in]  local_rank         Local rank of the process on its physical node.
   *  \param[in]  local_size         Local number of processes in this physical node.
   *  \param[in]  node_id            ID of the physical node that hosts this process.
   *  \param[in]  num_nodes          Number of physical nodes.
   *  \param[in]  tp_size            Size of the tensor-parallel group (may be less than localsize).
   *  \param[in]  comm_type          Type of collective op to overlap with GEMM.
   *  \param[in]  num_max_streams    Maximum number of concurrent CUDA streams.
   *  \param[in]  cga_size           cuBlasLt GEMM heuristic cluster size.
   *  \param[in]  num_comm_sms       Number of SMs to use for communication.
   *  \param[in]  set_sm_margin      Flag for reserving communication SMs (subtracts from math SMs).
   *  \param[in]  use_ce             Use copy engine for comm. kernels instead of SMs.
   *  \param[in]  atomic_gemm        Use atomic GEMM.
   *  \param[in]  aggregate          Whether to aggregate 2X work chunks in AG+split GEMM overlap.
   *  \param[in]  allgather_handle   Function pointer for external allgather op (e.g. DL framework).
   *  \param[in]  barrier_handle     Function pointer for external barrier op (e.g. DL framework).
   */
  CommOverlapP2P(int world_rank, int world_size, int local_rank, int local_size, int node_id,
                 int num_nodes, int tp_size, CommOverlapType comm_type,
                 std::function<void(void *, size_t, void *, size_t, char *)> allgather_handle,
                 std::function<void(char *)> barrier_handle,
                 int num_max_streams = NVTE_COMM_OVERLAP_MAX_STREAMS, int cga_size = 1,
                 int num_comm_sms = 1, bool set_sm_margin = false, bool use_ce = true,
                 bool atomic_gemm = false, bool aggregate = false);

  ~CommOverlapP2P();

  /*! \brief Overlap all-gather (producer) with atomic GEMM (consumer).
   *
   * This algorithm assumes the B matrix is pre-copied to ubufs[rank_id]. This is
   * needed to have AG outputs in each rank to be in the contiguous memory space
   * after all ring exchange phases.
   *
   * WARNING: Cannot be invoked on its own for a stand-alone all-gather overlap. The output matrix
   * D will have permuted work chunks that must to be unscrambled by a paired reduce-scatter
   * overlap.
   *
   *  \param[in]   stream                 CUDA stream used for the operation.
   *  \param[in]   A                      The A matrix.
   *  \param[in]   A_trans                Whether A matrix is transposed.
   *  \param[in]   B                      The B matrix.
   *  \param[in]   transb                 Whether B matrix is transposed.
   *  \param[in]   bias                   Bias tensor.
   *  \param[out]  D                      Output matrix.
   *  \param[out]  pre_gelu_out           Output matrix before GELU activation.
   *  \param[in]   ubuf                   Combined Userbuffers workspace.
   *  \param[in]   ubufs                  Vector of Userbuffer workspace chunks.
   *  \param[in]   counters               Atomic flags.
   *  \param[out]  B_copy                 All-gathered copy of the B matrix.
   *  \param[out]  D_buffer               Buffer space for the permuted output matrix chunks.
   *  \param[in]   workspace              GEMM Workspace tensor.
   *  \param[in]   grad                   Whether this operation is part of the
   *                                      gradient computation.
   *  \param[in]   accumulate             Whether to accumulate the result into the D matrix.
   *  \param[in]   use_split_accumulator  Whether to use split accumulator in the FP8 GEMM.
   */
  void atomic_gemm_overlap_ag(cudaStream_t stream_main, const TensorWrapper &A, bool A_trans,
                              const TensorWrapper &B, bool B_trans, const TensorWrapper &bias,
                              const TensorWrapper &D, const TensorWrapper &pre_gelu_out,
                              const TensorWrapper &ubuf, const std::vector<TensorWrapper> &ubufs,
                              const TensorWrapper &counters, const TensorWrapper &B_copy,
                              const TensorWrapper &D_buffer, const TensorWrapper &workspace,
                              bool grad, bool accumulate, bool use_split_accumulator);

  /*! \brief Overlap all-gather (producer) with split GEMM (consumer).
   *
   * This algorithm assumes the B matrix is pre-copied to ubufs[rank_id]. This is
   * needed to have AG outputs in each rank to be in the contiguous memory space
   * after all ring exchange phases.
   *
   *  \param[in]   stream                 CUDA stream used for the operation.
   *  \param[in]   A                      The A matrix.
   *  \param[in]   A_trans                Whether A matrix is transposed.
   *  \param[in]   B                      The B matrix.
   *  \param[in]   transb                 Whether B matrix is transposed.
   *  \param[in]   bias                   Bias tensor.
   *  \param[out]  D                      Output matrix.
   *  \param[out]  pre_gelu_out           Output matrix before GELU activation.
   *  \param[in]   ubuf                   Userbuffers workspace.
   *  \param[in]   ubufs                  Vector of Userbuffer workspace chunks.
   *  \param[out]  rs_output              Output tensor for the reduce-scattered result.
   *  \param[in]   workspace              GEMM Workspace tensor.
   *  \param[in]   grad                   Whether this operation is part of the
   *                                      gradient computation.
   *  \param[in]   accumulate             Whether to accumulate the result into the D matrix.
   *  \param[in]   use_split_accumulator  Whether to use split accumulator in the FP8 GEMM.
   */
  void split_gemm_overlap_ag(cudaStream_t stream_main, const TensorWrapper &A, bool A_trans,
                             const TensorWrapper &B, bool B_trans, const TensorWrapper &bias,
                             const TensorWrapper &D, const TensorWrapper &pre_gelu_out,
                             const std::vector<TensorWrapper> &ubufs, const TensorWrapper &B_copy,
                             const TensorWrapper &workspace, bool grad, bool accumulate,
                             bool use_split_accumulator);

  /*! \brief Overlap atomic GEMM (producer) with reduce-scatter (consumer).
   *
   *  \param[in]   stream                 CUDA stream used for the operation.
   *  \param[in]   A                      The A matrix.
   *  \param[in]   A_trans                Whether A matrix is transposed.
   *  \param[in]   B                      The B matrix.
   *  \param[in]   transb                 Whether B matrix is transposed.
   *  \param[in]   bias                   Bias tensor.
   *  \param[out]  D                      Output matrix.
   *  \param[out]  pre_gelu_out           Output matrix before GELU activation.
   *  \param[out]  ubuf                   Userbuffers workspace.
   *  \param[out]  ubufs                  Vector of Userbuffer workspace chunks.
   *  \param[in]   counters               Atomic flags.
   *  \param[in]   workspace              GEMM Workspace tensor.
   *  \param[in]   grad                   Whether this operation is part of the
   *                                      gradient computation.
   *  \param[in]   accumulate             Whether to accumulate the result into the D matrix.
   *  \param[in]   use_split_accumulator  Whether to use split accumulator in the FP8 GEMM.
   *  \param[out]  rs_output              Reduce-scattered output tensor.
   */
  void atomic_gemm_overlap_rs(cudaStream_t stream_main, const TensorWrapper &A, bool A_trans,
                              const TensorWrapper &B, bool B_trans, const TensorWrapper &bias,
                              const TensorWrapper &D, const TensorWrapper &pre_gelu_out,
                              const TensorWrapper &ubuf, const std::vector<TensorWrapper> &ubufs,
                              const TensorWrapper &counters, const TensorWrapper &workspace,
                              bool grad, bool accumulate, bool use_split_accumulator,
                              const TensorWrapper &rs_output);

  /*! \brief Overlap split GEMM (producer) with reduce-scatter (consumer).
   *
   *  \param[in]   stream                 CUDA stream used for the operation.
   *  \param[in]   A                      The A matrix.
   *  \param[in]   A_trans                Whether A matrix is transposed.
   *  \param[in]   B                      The B matrix.
   *  \param[in]   transb                 Whether B matrix is transposed.
   *  \param[in]   bias                   Bias tensor.
   *  \param[out]  D                      Output matrix.
   *  \param[out]  pre_gelu_out           Output matrix before GELU activation.
   *  \param[in]   ubufs                  Vector of Userbuffer workspace chunks.
   *  \param[in]   workspace              GEMM Workspace tensor.
   *  \param[in]   grad                   Whether this operation is part of the
   *                                      gradient computation.
   *  \param[in]   accumulate             Whether to accumulate the result into the D matrix.
   *  \param[in]   use_split_accumulator  Whether to use split accumulator in the FP8 GEMM.
   *  \param[out]  rs_output              Reduce-scattered output tensor.
   */
  void split_gemm_overlap_rs(cudaStream_t stream_main, const TensorWrapper &A, bool A_trans,
                             const TensorWrapper &B, bool B_trans, const TensorWrapper &bias,
                             const TensorWrapper &D, const TensorWrapper &pre_gelu_out,
                             const std::vector<TensorWrapper> &ubufs,
                             const TensorWrapper &workspace, bool grad, bool accumulate,
                             bool use_split_accumulator, const TensorWrapper &rs_output);
};  // CommOverlapP2P

}  // namespace transformer_engine

#endif  // __cplusplus

#endif  // TRANSFORMER_ENGINE_COMMON_COMM_GEMM_OVERLAP_H_
