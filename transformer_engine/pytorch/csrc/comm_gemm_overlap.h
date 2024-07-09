/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_COMM_GEMM_OVERLAP_H_
#define TRANSFORMER_ENGINE_PYTORCH_COMM_GEMM_OVERLAP_H_

#include <ATen/ATen.h>
#include <torch/custom_class.h>

#include <transformer_engine/transformer_engine.h>
#include <transformer_engine/comm_gemm_overlap.h>

using namespace torch::indexing;

namespace transformer_engine {

namespace comm_gemm_overlap {

/*
** Helper function for setting Python callbacks to torch.distributed collectives.
*/
void set_bootstrap_callbacks(
  std::function<void(at::Tensor &, at::Tensor &, const std::string &)> allgather,
  std::function<void(const std::string &)> barrier);

/*
** Python callback for torch.distributed.all_gather_into_tensor(global_data, localdata, tp_group).
*/
void ub_torch_allgather(void *globaldata, size_t globalbytes, void *localdata, size_t localbytes,
                        char *group);

/*
** Python callback for torch.distributed.barrier(tp_group).
*/
void ub_torch_barrier(char *group);

struct PYBIND11_EXPORT UbufCommOverlap : torch::CustomClassHolder, CommGemmOverlap {
  torch::Tensor _counters;
  torch::Tensor _ubuf;
  int _ubuf_bytes;
  DType _ubuf_dtype;
  void *_ubuf_scale_inv_ptr;
  bool _ubuf_scale_inv_initialized{false};

  UbufCommOverlap(torch::Tensor sample, int world_rank, int world_size, int local_rank,
                  int local_size, int node_id, int num_nodes, int tp_size, int num_splits,
                  int num_max_streams, int cga_size, int num_comm_sm, bool set_sm_margin,
                  bool use_ce, bool atomic_gemm);

  /*
  ** Bulk GEMM + COMM
  ** This function assumes the communication input is pre-copied to _ubuf.
  */
  std::vector<at::Tensor> bulk_overlap(
      at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor, DType A_type, bool transa,
      at::Tensor B, at::Tensor B_scale_inverse, int64_t B_fp8_tensor, DType B_type, bool transb,
      at::Tensor D, at::Tensor D_scale, DType D_type, at::Tensor D_amax, at::Tensor bias,
      DType bias_type, at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
      size_t workspaceSize, bool accumulate, bool use_split_accumulator,
      NVTE_Comm_Overlap_Type comm_type, at::Tensor rs_output);

  /*
  ** Atomic GEMM + Split Reduce-Scatter
  */
  void atomic_gemm_overlap_rs(at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
                              DType A_type, bool transa, at::Tensor B, at::Tensor B_scale_inverse,
                              int64_t B_fp8_tensor, DType B_type, bool transb, at::Tensor D,
                              at::Tensor D_scale, DType D_type, at::Tensor D_amax, at::Tensor bias,
                              DType bias_type, at::Tensor pre_gelu_out, bool grad,
                              at::Tensor workspace, size_t workspaceSize, bool accumulate,
                              bool use_split_accumulator, bool gemm_overlap, at::Tensor rs_output);

  /*
  ** Pipelined GEMM + Split Reduce-Scatter
  */
  void split_overlap_rs(at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
                        DType A_type, bool transa, at::Tensor B, at::Tensor B_scale_inverse,
                        int64_t B_fp8_tensor, DType B_type, bool transb, at::Tensor D,
                        at::Tensor D_scale, DType D_type, at::Tensor D_amax, at::Tensor bias,
                        DType bias_type, at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
                        size_t workspaceSize, bool accumulate, bool use_split_accumulator,
                        bool gemm_overlap, at::Tensor rs_output);

  /*
  ** Helper function to copy input to _ubuf.
  */
  void copy_input_to_ubuf(torch::Tensor input, bool chunk);

  /*
  ** Helper function to export _ubuf output.
  */
  torch::Tensor get_ubuf_output(NVTE_Comm_Overlap_Type comm_type);

  /*
  ** Helper function to set the inverse Fp8 scale for the _ubuf tensor.
  */
  void set_ubuf_scale_inv(const torch::Tensor &scale_inv);

  bool is_fp8_ubuf() { return (_ubuf.element_size() == 1); }
};  // UbufCommOverlap

struct PYBIND11_EXPORT UbufP2PCommOverlap : torch::CustomClassHolder, CommGemmOverlapP2P {
  torch::Tensor _counters;
  torch::Tensor _ubuf;
  std::vector<torch::Tensor> _ubufs;
  DType _ubuf_dtype;

  void *_ubuf_scale_inv_ptr;
  bool _ubuf_scale_inv_initialized{false};
  int _ubuf_bytes, _ubuf_chunk_bytes;

  UbufP2PCommOverlap(torch::Tensor sample, int world_rank, int world_size, int local_rank,
                     int local_size, int node_id, int num_nodes, int tp_size, int num_max_streams,
                     int cga_size, int num_comm_sms, bool set_sm_margin, bool use_ce,
                     bool atomic_gemm, bool aggregate, bool is_reduce_scatter);

  /*
  ** Split AllGather + Atomic GEMM using P2P communication
  ** This function assumes the input_b is pre-copied to _ubufs[rank_id]. This is needed to have AG
  ** outputs in each rank to be in the contiguous memory space after all ring exchange phases.
  */
  torch::Tensor atomic_gemm_overlap_ag(
      at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor, DType A_type, bool transa,
      at::Tensor B, at::Tensor B_scale_inverse, int64_t B_fp8_tensor, DType B_type, bool transb,
      at::Tensor D, at::Tensor D_scale, DType D_type, at::Tensor D_amax, at::Tensor bias,
      DType bias_type, at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
      size_t workspaceSize, bool accumulate, bool use_split_accumulator, at::Tensor B_copy);

  /*
  ** Split AllGather + Pipelined GEMM using P2P communication
  ** This function assumes the input_b is pre-copied to _ubufs[rank_id]. This is needed to have AG
  ** outputs in each rank to be in the contiguous memory space after all ring exchange phases.
  */
  torch::Tensor split_overlap_ag(at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
                                 DType A_type, bool transa, at::Tensor B,
                                 at::Tensor B_scale_inverse, int64_t B_fp8_tensor, DType B_type,
                                 bool transb, at::Tensor D, at::Tensor D_scale, DType D_type,
                                 at::Tensor D_amax, at::Tensor bias, DType bias_type,
                                 at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
                                 size_t workspaceSize, bool accumulate, bool use_split_accumulator,
                                 at::Tensor B_copy);

  /*
  ** Atomic GEMM + Split Reduce-Scatter using P2P communication
  */
  void atomic_gemm_overlap_rs(at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
                              DType A_type, bool transa, at::Tensor B, at::Tensor B_scale_inverse,
                              int64_t B_fp8_tensor, DType B_type, bool transb, at::Tensor D,
                              at::Tensor D_scale, DType D_type, at::Tensor D_amax, at::Tensor bias,
                              DType bias_type, at::Tensor pre_gelu_out, bool grad,
                              at::Tensor workspace, size_t workspaceSize, bool accumulate,
                              bool use_split_accumulator, at::Tensor rs_output);

  /*
  ** Pipelined GEMM + Split Reduce+Scatter using P2P communication
  */
  void split_overlap_rs(at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
                        DType A_type, bool transa, at::Tensor B, at::Tensor B_scale_inverse,
                        int64_t B_fp8_tensor, DType B_type, bool transb, at::Tensor D,
                        at::Tensor D_scale, DType D_type, at::Tensor D_amax, at::Tensor bias,
                        DType bias_type, at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
                        size_t workspaceSize, bool accumulate, bool use_split_accumulator,
                        at::Tensor rs_output);

  /*
  ** Helper function to copy input to _ubuf or _ubufs chunks.
  */
  void copy_input_to_ubuf(torch::Tensor input, bool chunk);

  /*
  ** Helper function to export _ubuf output.
  */
  torch::Tensor get_ubuf_output(NVTE_Comm_Overlap_Type comm_type);

  /*
  ** Helper function to set the inverse Fp8 scale for the _ubuf tensor.
  */
  void set_ubuf_scale_inv(const torch::Tensor &scale_inv);

  bool is_fp8_ubuf() { return (_ubuf.element_size() == 1); }
};  // UbufP2PCommOverlap

}  // namespace comm_gemm_overlap

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_PYTORCH_COMM_GEMM_OVERLAP_H_
