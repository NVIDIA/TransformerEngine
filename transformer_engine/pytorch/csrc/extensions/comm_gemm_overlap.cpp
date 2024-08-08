/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_fp8.h>

#include <algorithm>

#include "../extensions.h"
#include "common/util/system.h"

#define HALF_BYTES 2
#define UB_MAX_SM 32

using namespace torch::indexing;
using namespace std::placeholders;
namespace te = transformer_engine;

namespace transformer_engine_torch {

CommOverlapHelper::CommOverlapHelper(c10d::ProcessGroup *world_group,
                                     std::optional<c10d::ProcessGroup *> intra_node_group_holder) {
  c10d::ProcessGroup *intra_node_group;
  if (intra_node_group_holder.has_value()) {
    intra_node_group = intra_node_group_holder.value();
    NVTE_CHECK(intra_node_group->getBackendType() == world_group->getBackendType(),
               "Intra-node group backend (", intra_node_group->getBackendName(), ") must match ",
               "the world group backend (", world_group->getBackendName(), ").");
  } else {
    // If no intra-node group is provided, assume it is equal to the world group
    intra_node_group = world_group;
  }

  pgs.insert({"world", world_group});
  world_rank = world_group->getRank();
  world_size = world_group->getSize();
  backend_is_nccl = (world_group->getBackendType() == c10d::ProcessGroup::BackendType::NCCL);

  pgs.insert({"intra", intra_node_group});
  local_rank = intra_node_group->getRank();
  local_size = intra_node_group->getSize();

  NVTE_CHECK(world_size % local_size == 0, "Size of the world_group (", world_size,
             ") must be divisible by the size of the ", "intra_node_group (", local_size, ").");
  node_id = world_rank / local_size;
  num_nodes = world_size / local_size;

  initialized = true;
}

CommOverlapHelper::~CommOverlapHelper() {
  for (auto &pg : pgs) pg.second = nullptr;
  backend_is_nccl = false;
  initialized = false;
}

void CommOverlapHelper::allgather(void *globaldata, size_t globalbytes, void *localdata,
                                  size_t localbytes, char *group) {
  NVTE_CHECK(initialized, "Internal TE error: tex.CommOverlapHelper() not initialized!");

  auto localtensor =
      torch::from_blob(localdata, {static_cast<int64_t>(localbytes / sizeof(uint8_t))},
                       at::device(torch::kCPU).dtype(torch::kUInt8));
  auto localtmp = (backend_is_nccl) ? localtensor.cuda() : localtensor;
  auto globaltensor =
      torch::from_blob(globaldata, {static_cast<int64_t>(globalbytes / sizeof(uint8_t))},
                       at::device(torch::kCPU).dtype(torch::kUInt8));
  auto globaltmp = (backend_is_nccl) ? globaltensor.cuda() : globaltensor;

  std::vector<std::vector<torch::Tensor>> globalchunks = {globaltmp.chunk(pgs[group]->getSize())};
  std::vector<torch::Tensor> localchunk = {localtmp};
  auto work = pgs[group]->allgather(globalchunks, localchunk);
  work->wait();

  if (backend_is_nccl) {
    globaltensor.copy_(globaltmp.cpu());
    globaltmp = torch::Tensor();
    localtmp = torch::Tensor();
  }
}

void CommOverlapHelper::broadcast(void *data, size_t bytes, int src, char *group) {
  NVTE_CHECK(initialized, "Internal TE error: tex.CommOverlapHelper() not initialized!");

  auto datatensor = torch::from_blob(data, {static_cast<int64_t>(bytes / sizeof(uint8_t))},
                                     at::device(torch::kCPU).dtype(torch::kUInt8));
  auto datatmp = (backend_is_nccl) ? datatensor.cuda() : datatensor;

  c10d::BroadcastOptions bcast_opts;
  bcast_opts.rootRank = src;
  std::vector<torch::Tensor> datachunk = {datatmp};
  auto work = pgs[group]->broadcast(datachunk, bcast_opts);
  work->wait();

  if (backend_is_nccl) {
    datatensor.copy_(datatmp.cpu());
    datatmp = torch::Tensor();
  }
}

void CommOverlapHelper::barrier(char *group) {
  NVTE_CHECK(initialized, "Internal TE error: tex.CommOverlapHelper() not initialized!");
  auto work = pgs[group]->barrier();
  work->wait();
}

/***************************************************************************************************
** CommOverlap -- Collective (pipelined) comm+GEMM wrappers for PyTorch
***************************************************************************************************/

CommOverlap::CommOverlap(torch::Tensor sample, CommOverlapHelper &helper, int tp_size,
                         int num_splits, int num_max_streams, int cga_size, int num_comm_sm,
                         bool set_sm_margin, bool atomic_gemm)
    : te::CommOverlap(helper.world_rank, helper.world_size, helper.local_rank, helper.local_size,
                      helper.node_id, helper.num_nodes, tp_size,
                      std::bind(&CommOverlapHelper::allgather, helper, _1, _2, _3, _4, _5),
                      std::bind(&CommOverlapHelper::barrier, helper, _1), num_splits,
                      num_max_streams, cga_size, num_comm_sm, set_sm_margin, atomic_gemm) {
  void *ubuf_ptr;
  _ubuf_bytes = sample.numel() * sample.element_size();
  register_gpu_buffer(&ubuf_ptr, _ubuf_bytes, true);
  _ubuf = torch::from_blob(ubuf_ptr, {sample.size(0), sample.size(1)}, sample.options());

  if (_atomic_gemm) {
    auto counter_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    _counters = torch::zeros({_num_splits * 2}, counter_options);
    _counters = _counters.index_put_({Slice(None, _num_splits)}, 1);
  }
}

/*
** Bulk GEMM + COMM
** This function assumes the communication input is pre-copied to _ubuf.
*/
std::vector<at::Tensor> CommOverlap::bulk_overlap(
    at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor, te::DType A_type, bool transa,
    at::Tensor B, at::Tensor B_scale_inverse, int64_t B_fp8_tensor, te::DType B_type, bool transb,
    at::Tensor D, at::Tensor D_scale, te::DType D_type, at::Tensor D_amax, at::Tensor bias,
    te::DType bias_type, at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
    size_t workspaceSize, bool accumulate, bool use_split_accumulator,
    te::CommOverlapType comm_type, te::DType bulk_fp8_dtype, at::Tensor rs_output) {
  void *A_scale_inv_ptr = nullptr;
  if (A_scale_inverse.numel()) A_scale_inv_ptr = A_scale_inverse[A_fp8_tensor].data_ptr();
  auto A_ = makeTransformerEngineTensor(
      A.data_ptr(), {static_cast<size_t>(A.size(0)), static_cast<size_t>(A.size(1))}, A_type,
      nullptr, nullptr, A_scale_inv_ptr);

  void *B_scale_inv_ptr = nullptr;
  if (B_scale_inverse.numel()) B_scale_inv_ptr = B_scale_inverse[B_fp8_tensor].data_ptr();
  auto B_ = makeTransformerEngineTensor(
      B.data_ptr(), {static_cast<size_t>(B.size(0)), static_cast<size_t>(B.size(1))}, B_type,
      nullptr, nullptr, B_scale_inv_ptr);

  void *D_amax_ptr = nullptr;
  void *D_scale_ptr = nullptr;
  if (D_amax.numel()) D_amax_ptr = D_amax.data_ptr();
  if (D_scale.numel()) D_scale_ptr = D_scale.data_ptr();
  auto D_ = makeTransformerEngineTensor(
      D.data_ptr(), {static_cast<size_t>(D.size(0)), static_cast<size_t>(D.size(1))}, D_type,
      D_amax_ptr, D_scale_ptr, nullptr);

  auto bias_ =
      makeTransformerEngineTensor(bias.data_ptr(), {static_cast<size_t>(bias.size(0))}, bias_type);

  const auto gelu_shape = (pre_gelu_out.data_ptr() == nullptr)
                              ? std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0))}
                              : std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0)),
                                                    static_cast<size_t>(pre_gelu_out.size(1))};
  auto pre_gelu_out_ = makeTransformerEngineTensor(
      pre_gelu_out.data_ptr(), gelu_shape, GetTransformerEngineDType(pre_gelu_out.scalar_type()));

  te::DType ubuf_dtype = GetTransformerEngineDType(_ubuf.scalar_type());
  void *_ubuf_scale_inv_ptr = nullptr;
  if (_ubuf.element_size() == 1) {
    assert(bulk_fp8_dtype == te::DType::kFloat8E4M3 || bulk_fp8_dtype == te::DType::kFloat8E5M2);
    ubuf_dtype = bulk_fp8_dtype;
    if (comm_type == te::CommOverlapType::REDUCE_SCATTER) {
      assert(_ubuf_scale_inv_initialized);
      _ubuf_scale_inv_ptr = _ubuf_scale_inv.data_ptr();
    }
  }
  auto ubuf_ = makeTransformerEngineTensor(
      _ubuf.data_ptr(), {static_cast<size_t>(_ubuf.size(0)), static_cast<size_t>(_ubuf.size(1))},
      ubuf_dtype, nullptr, nullptr, _ubuf_scale_inv_ptr);

  void *rs_out_ptr = nullptr;
  auto rs_out_shape = std::vector<size_t>{static_cast<size_t>(rs_output.size(0))};
  if (comm_type == te::CommOverlapType::REDUCE_SCATTER && _ubuf.element_size() == 1) {
    rs_out_ptr = rs_output.data_ptr();
    rs_out_shape.push_back(static_cast<size_t>(rs_output.size(1)));
  }
  auto rs_out_ = makeTransformerEngineTensor(rs_out_ptr, rs_out_shape, te::DType::kBFloat16);

  auto workspace_ =
      makeTransformerEngineTensor(workspace.data_ptr(), {workspaceSize}, te::DType::kByte);

  at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
  te::CommOverlap::bulk_overlap((cudaStream_t)stream_main, A_, transa, B_, transb, bias_, D_,
                                pre_gelu_out_, ubuf_, rs_out_, workspace_, grad, accumulate,
                                use_split_accumulator, comm_type);

  // Generate output tensor from userbuf data pointer
  char *ubuf_wt_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());
  if (comm_type == te::CommOverlapType::REDUCE_SCATTER) {
    ubuf_wt_ptr += (_ubuf.numel() * _ubuf.element_size() / _tp_size) * _tp_id;
  }
  int output_c_dim0 =
      (comm_type == te::CommOverlapType::ALL_GATHER) ? _ubuf.size(0) : _ubuf.size(0) / _tp_size;
  int output_c_dim1 = _ubuf.size(1);
  torch::Tensor output_tensor =
      torch::from_blob(ubuf_wt_ptr, {output_c_dim0, output_c_dim1}, _ubuf.options());

  return {D, output_tensor};
}  // CommOverlap::bulk_overlap

/*
** Atomic GEMM + Split Reduce-Scatter
*/
void CommOverlap::atomic_gemm_overlap_rs(
    at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor, te::DType A_type, bool transa,
    at::Tensor B, at::Tensor B_scale_inverse, int64_t B_fp8_tensor, te::DType B_type, bool transb,
    at::Tensor D, at::Tensor D_scale, te::DType D_type, at::Tensor D_amax, at::Tensor bias,
    te::DType bias_type, at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
    size_t workspaceSize, bool accumulate, bool use_split_accumulator, bool gemm_overlap,
    at::Tensor rs_output) {
  void *A_scale_inv_ptr = nullptr;
  if (A_scale_inverse.numel()) A_scale_inv_ptr = A_scale_inverse[A_fp8_tensor].data_ptr();
  auto A_ = makeTransformerEngineTensor(
      A.data_ptr(), {static_cast<size_t>(A.size(0)), static_cast<size_t>(A.size(1))}, A_type,
      nullptr, nullptr, A_scale_inv_ptr);

  void *B_scale_inv_ptr = nullptr;
  if (B_scale_inverse.numel()) B_scale_inv_ptr = B_scale_inverse[B_fp8_tensor].data_ptr();
  auto B_ = makeTransformerEngineTensor(
      B.data_ptr(), {static_cast<size_t>(B.size(0)), static_cast<size_t>(B.size(1))}, B_type,
      nullptr, nullptr, B_scale_inv_ptr);

  void *D_amax_ptr = nullptr;
  void *D_scale_ptr = nullptr;
  if (D_amax.numel()) D_amax_ptr = D_amax.data_ptr();
  if (D_scale.numel()) D_scale_ptr = D_scale.data_ptr();
  auto D_ = makeTransformerEngineTensor(
      D.data_ptr(), {static_cast<size_t>(D.size(0)), static_cast<size_t>(D.size(1))}, D_type,
      D_amax_ptr, D_scale_ptr, nullptr);

  auto bias_ =
      makeTransformerEngineTensor(bias.data_ptr(), {static_cast<size_t>(bias.size(0))}, bias_type);

  const auto gelu_shape = (pre_gelu_out.data_ptr() == nullptr)
                              ? std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0))}
                              : std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0)),
                                                    static_cast<size_t>(pre_gelu_out.size(1))};
  auto pre_gelu_out_ = makeTransformerEngineTensor(
      pre_gelu_out.data_ptr(), gelu_shape, GetTransformerEngineDType(pre_gelu_out.scalar_type()));

  void *_ubuf_scale_inv_ptr = nullptr;
  if (_ubuf.element_size() == 1) {
    assert(_ubuf_scale_inv_initialized);
    _ubuf_scale_inv_ptr = _ubuf_scale_inv.data_ptr();
  }
  auto ubuf_ = makeTransformerEngineTensor(
      _ubuf.data_ptr(), {static_cast<size_t>(_ubuf.size(0)), static_cast<size_t>(_ubuf.size(1))},
      D_type, D_amax_ptr, D_scale_ptr, _ubuf_scale_inv_ptr);

  auto rs_out_ = makeTransformerEngineTensor(
      rs_output.data_ptr(),
      {static_cast<size_t>(rs_output.size(0)), static_cast<size_t>(rs_output.size(1))},
      te::DType::kBFloat16);

  auto counters_ = makeTransformerEngineTensor(
      _counters.data_ptr(), {static_cast<size_t>(_counters.size(0))}, te::DType::kInt32);

  auto workspace_ =
      makeTransformerEngineTensor(workspace.data_ptr(), {workspaceSize}, te::DType::kByte);

  at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
  te::CommOverlap::atomic_gemm_overlap_rs((cudaStream_t)stream_main, A_, transa, B_, transb, bias_,
                                          D_, pre_gelu_out_, ubuf_, rs_out_, counters_, workspace_,
                                          grad, accumulate, use_split_accumulator);
}  // CommOverlap::atomic_gemm_overlap_rs

/*
** Pipelined GEMM + Split Reduce-Scatter
*/
void CommOverlap::split_gemm_overlap_rs(
    at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor, te::DType A_type, bool transa,
    at::Tensor B, at::Tensor B_scale_inverse, int64_t B_fp8_tensor, te::DType B_type, bool transb,
    at::Tensor D, at::Tensor D_scale, te::DType D_type, at::Tensor D_amax, at::Tensor bias,
    te::DType bias_type, at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
    size_t workspaceSize, bool accumulate, bool use_split_accumulator, bool gemm_overlap,
    at::Tensor rs_output) {
  void *A_scale_inv_ptr = nullptr;
  if (A_scale_inverse.numel()) A_scale_inv_ptr = A_scale_inverse[A_fp8_tensor].data_ptr();
  auto A_ = makeTransformerEngineTensor(
      A.data_ptr(), {static_cast<size_t>(A.size(0)), static_cast<size_t>(A.size(1))}, A_type,
      nullptr, nullptr, A_scale_inv_ptr);

  void *B_scale_inv_ptr = nullptr;
  if (B_scale_inverse.numel()) B_scale_inv_ptr = B_scale_inverse[B_fp8_tensor].data_ptr();
  auto B_ = makeTransformerEngineTensor(
      B.data_ptr(), {static_cast<size_t>(B.size(0)), static_cast<size_t>(B.size(1))}, B_type,
      nullptr, nullptr, B_scale_inv_ptr);
  void *D_amax_ptr = nullptr;
  void *D_scale_ptr = nullptr;
  if (D_amax.numel()) D_amax_ptr = D_amax.data_ptr();
  if (D_scale.numel()) D_scale_ptr = D_scale.data_ptr();
  auto D_ = makeTransformerEngineTensor(
      D.data_ptr(), {static_cast<size_t>(D.size(0)), static_cast<size_t>(D.size(1))}, D_type,
      D_amax_ptr, D_scale_ptr, nullptr);

  auto bias_ =
      makeTransformerEngineTensor(bias.data_ptr(), {static_cast<size_t>(bias.size(0))}, bias_type);

  const auto gelu_shape = (pre_gelu_out.data_ptr() == nullptr)
                              ? std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0))}
                              : std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0)),
                                                    static_cast<size_t>(pre_gelu_out.size(1))};
  auto pre_gelu_out_ = makeTransformerEngineTensor(
      pre_gelu_out.data_ptr(), gelu_shape, GetTransformerEngineDType(pre_gelu_out.scalar_type()));

  void *_ubuf_scale_inv_ptr = nullptr;
  if (_ubuf.element_size() == 1) {
    assert(_ubuf_scale_inv_initialized);
    _ubuf_scale_inv_ptr = _ubuf_scale_inv.data_ptr();
  }
  auto ubuf_ = makeTransformerEngineTensor(
      _ubuf.data_ptr(), {static_cast<size_t>(_ubuf.size(0)), static_cast<size_t>(_ubuf.size(1))},
      D_type, D_amax_ptr, D_scale_ptr, _ubuf_scale_inv_ptr);

  auto rs_out_ = makeTransformerEngineTensor(
      rs_output.data_ptr(),
      {static_cast<size_t>(rs_output.size(0)), static_cast<size_t>(rs_output.size(1))},
      te::DType::kBFloat16);

  auto workspace_ =
      makeTransformerEngineTensor(workspace.data_ptr(), {workspaceSize}, te::DType::kByte);

  at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
  te::CommOverlap::split_gemm_overlap_rs((cudaStream_t)stream_main, A_, transa, B_, transb, bias_,
                                         D_, pre_gelu_out_, ubuf_, rs_out_, workspace_, grad,
                                         accumulate, use_split_accumulator, gemm_overlap);
}  // CommOverlap::split_overlap_rs

/*
** Helper function to copy input to _ubuf.
*/
void CommOverlap::copy_input_to_ubuf(torch::Tensor input, te::CommOverlapBuffer buffer_type) {
  char *ubuf_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());
  if (buffer_type == te::CommOverlapBuffer::LOCAL) {
    if ((input.numel() * _tp_size) != _ubuf.numel() ||
        input.element_size() != _ubuf.element_size()) {
      NVTE_ERROR("input and ubuf size do not match!");
    }
    ubuf_ptr += (_ubuf.numel() * _ubuf.element_size() / _tp_size) * _tp_id;
  } else {
    if (input.numel() != _ubuf.numel() || input.element_size() != _ubuf.element_size()) {
      NVTE_ERROR("input and ubuf size do not match!");
    }
  }

  at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
  NVTE_CHECK_CUDA(cudaEventRecord(_start_d2dcopy, (cudaStream_t)stream_main));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_comm, _start_d2dcopy, 0));
  NVTE_CHECK_CUDA(cudaMemcpyAsync(ubuf_ptr, input.data_ptr(), input.numel() * input.element_size(),
                                  cudaMemcpyDeviceToDevice, _stream_comm));
}

/*
** Helper function to export _ubuf output.
*/
torch::Tensor CommOverlap::get_ubuf_output(te::CommOverlapBuffer buffer_type) {
  char *ubuf_wt_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());
  int output_c_dim0 = _ubuf.size(0);
  if (buffer_type == te::CommOverlapBuffer::LOCAL) {
    ubuf_wt_ptr += (_ubuf.numel() * _ubuf.element_size() / _tp_size) * _tp_id;
    output_c_dim0 /= _tp_size;
  }
  auto output_tensor =
      torch::from_blob(ubuf_wt_ptr, {output_c_dim0, _ubuf.size(1)}, _ubuf.options());
  return output_tensor;
}

/*
** Helper function to set the inverse Fp8 scale for the _ubuf tensor.
*/
void CommOverlap::set_ubuf_scale_inv(const torch::Tensor &scale_inv) {
  _ubuf_scale_inv = scale_inv;
  _ubuf_scale_inv_initialized = true;
}

/***************************************************************************************************
** CommOverlapP2P -- Point-2-Point (ring-exchange) comm+GEMM wrappers for PyTorch
***************************************************************************************************/

CommOverlapP2P::CommOverlapP2P(torch::Tensor sample, CommOverlapHelper &helper, int tp_size,
                               te::CommOverlapType comm_type, int num_max_streams, int cga_size,
                               int num_comm_sms, bool set_sm_margin, bool use_ce, bool atomic_gemm,
                               bool aggregate)
    : te::CommOverlapP2P(helper.world_rank, helper.world_size, helper.local_rank, helper.local_size,
                         helper.node_id, helper.num_nodes, tp_size, comm_type,
                         std::bind(&CommOverlapHelper::allgather, helper, _1, _2, _3, _4, _5),
                         std::bind(&CommOverlapHelper::barrier, helper, _1), num_max_streams,
                         cga_size, num_comm_sms, set_sm_margin, use_ce, atomic_gemm, aggregate) {
  _ubuf_bytes = sample.numel() * sample.element_size();
  _ubuf_chunk_bytes = _ubuf_bytes / _tp_size;
  if (_is_reduce_scatter) {
    // GEMM + RS overlap: Allocate `2 x tp_size - 1` buffers to hold recieved GEMM chunk
    // outputs for reduction at the end of the pipelining.
    _ubuf_bytes = static_cast<int>((_ubuf_bytes / _tp_size) * (_tp_size * 2 - 1));
  }

  void *ubuf_ptr;
  register_gpu_buffer(&ubuf_ptr, _ubuf_bytes, true);
  _ubuf = torch::from_blob(
      ubuf_ptr, {(sample.size(0) / _tp_size) * _num_ubuf_chunks, sample.size(1)}, sample.options());

  // Create tensor chunks for easy management
  char *ubuf_byte_ptr = reinterpret_cast<char *>(ubuf_ptr);
  for (int i = 0; i < _num_ubuf_chunks; i++) {
    torch::Tensor ubuf_chunk = torch::from_blob(
        ubuf_byte_ptr, {sample.size(0) / _tp_size, sample.size(1)}, sample.options());
    _ubufs.push_back(ubuf_chunk);
    ubuf_byte_ptr += _ubuf_chunk_bytes;
  }

  if (_atomic_gemm) {
    auto counter_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    _counters = torch::zeros({_tp_size * 2}, counter_options);
    _counters = _counters.index_put_({Slice(None, _tp_size)}, 1);
    if (!_is_reduce_scatter) {
      _counters = _counters.index_put_({_self_chunk_id /* = 0 for AG + atomic GEMM */}, 0);
    }
  }
}

/*
** Split AllGather + Atomic GEMM using P2P communication
** This function assumes the input_b is pre-copied to _ubufs[rank_id]. This is needed to have AG
** outputs in each rank to be in the contiguous memory space after all ring exchange phases.
*/
torch::Tensor CommOverlapP2P::atomic_gemm_overlap_ag(
    at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor, te::DType A_type, bool transa,
    at::Tensor B, at::Tensor B_scale_inverse, int64_t B_fp8_tensor, te::DType B_type, bool transb,
    at::Tensor D, at::Tensor D_scale, te::DType D_type, at::Tensor D_amax, at::Tensor bias,
    te::DType bias_type, at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
    size_t workspaceSize, bool accumulate, bool use_split_accumulator, at::Tensor B_copy) {
  void *A_scale_inv_ptr = nullptr;
  if (A_scale_inverse.numel()) A_scale_inv_ptr = A_scale_inverse[A_fp8_tensor].data_ptr();
  auto A_ = makeTransformerEngineTensor(
      A.data_ptr(), {static_cast<size_t>(A.size(0)), static_cast<size_t>(A.size(1))}, A_type,
      nullptr, nullptr, A_scale_inv_ptr);

  void *B_scale_inv_ptr = nullptr;
  if (B_scale_inverse.numel()) B_scale_inv_ptr = B_scale_inverse[B_fp8_tensor].data_ptr();
  auto B_ = makeTransformerEngineTensor(
      B.data_ptr(), {static_cast<size_t>(B.size(0)), static_cast<size_t>(B.size(1))}, B_type,
      nullptr, nullptr, B_scale_inv_ptr);

  // Create an GEMM output buffer with N+1 chunks in a contiguous memory
  int m = (transa) ? A.size(0) : A.size(1);
  int n = _ubuf.size(0);
  int n_chunk = n / _tp_size;
  torch::Tensor D_buffer = torch::zeros({n_chunk * (_tp_size + 1), m}, D.options());
  D = torch::from_blob(D_buffer.data_ptr(), {D.size(0), D.size(1)}, D.options());

  void *D_amax_ptr = nullptr;
  void *D_scale_ptr = nullptr;
  if (D_amax.numel()) D_amax_ptr = D_amax.data_ptr();
  if (D_scale.numel()) D_scale_ptr = D_scale.data_ptr();
  auto D_ = makeTransformerEngineTensor(
      D.data_ptr(), {static_cast<size_t>(D.size(0)), static_cast<size_t>(D.size(1))}, D_type,
      D_amax_ptr, D_scale_ptr, nullptr);

  auto bias_ =
      makeTransformerEngineTensor(bias.data_ptr(), {static_cast<size_t>(bias.size(0))}, bias_type);

  const auto gelu_shape = (pre_gelu_out.data_ptr() == nullptr)
                              ? std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0))}
                              : std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0)),
                                                    static_cast<size_t>(pre_gelu_out.size(1))};
  auto pre_gelu_out_ = makeTransformerEngineTensor(
      pre_gelu_out.data_ptr(), gelu_shape, GetTransformerEngineDType(pre_gelu_out.scalar_type()));

  auto ubuf_ = makeTransformerEngineTensor(
      _ubuf.data_ptr(), {static_cast<size_t>(_ubuf.size(0)), static_cast<size_t>(_ubuf.size(1))},
      B_type, nullptr, nullptr, B_scale_inv_ptr);

  std::vector<te::TensorWrapper> ubufs_;
  for (int i = 0; i < _num_ubuf_chunks; i++)
    ubufs_.push_back(makeTransformerEngineTensor(
        _ubufs[i].data_ptr(),
        {static_cast<size_t>(_ubufs[i].size(0)), static_cast<size_t>(_ubufs[i].size(1))}, B_type,
        nullptr, nullptr, B_scale_inv_ptr));

  auto counters_ = makeTransformerEngineTensor(
      _counters.data_ptr(), {static_cast<size_t>(_counters.size(0))}, te::DType::kInt32);

  const auto B_copy_shape = (B_copy.data_ptr() == nullptr)
                                ? std::vector<size_t>{static_cast<size_t>(B_copy.size(0))}
                                : std::vector<size_t>{static_cast<size_t>(B_copy.size(0)),
                                                      static_cast<size_t>(B_copy.size(1))};
  auto B_copy_ = makeTransformerEngineTensor(B_copy.data_ptr(), B_copy_shape, B_type, nullptr,
                                             nullptr, B_scale_inv_ptr);

  auto D_buffer_ = makeTransformerEngineTensor(
      D_buffer.data_ptr(),
      {static_cast<size_t>(D_buffer.size(0)), static_cast<size_t>(D_buffer.size(1))}, D_type,
      D_amax_ptr, D_scale_ptr, nullptr);

  auto workspace_ =
      makeTransformerEngineTensor(workspace.data_ptr(), {workspaceSize}, te::DType::kByte);

  at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
  te::CommOverlapP2P::atomic_gemm_overlap_ag(
      (cudaStream_t)stream_main, A_, transa, B_, transb, bias_, D_, pre_gelu_out_, ubuf_, ubufs_,
      counters_, B_copy_, D_buffer_, workspace_, grad, accumulate, use_split_accumulator);

  // Return the last N rows of D_buffer
  torch::Tensor D_return = D_buffer.narrow(0, n_chunk, n);
  return D_return;
}  // CommOverlapP2P::atomic_gemm_overlap_ag

/*
** Split AllGather + Pipelined GEMM using P2P communication
** This function assumes the input_b is pre-copied to _ubufs[rank_id]. This is needed to have AG
** outputs in each rank to be in the contiguous memory space after all ring exchange phases.
*/
torch::Tensor CommOverlapP2P::split_gemm_overlap_ag(
    at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor, te::DType A_type, bool transa,
    at::Tensor B, at::Tensor B_scale_inverse, int64_t B_fp8_tensor, te::DType B_type, bool transb,
    at::Tensor D, at::Tensor D_scale, te::DType D_type, at::Tensor D_amax, at::Tensor bias,
    te::DType bias_type, at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
    size_t workspaceSize, bool accumulate, bool use_split_accumulator, at::Tensor B_copy) {
  void *A_scale_inv_ptr = nullptr;
  if (A_scale_inverse.numel()) A_scale_inv_ptr = A_scale_inverse[A_fp8_tensor].data_ptr();
  auto A_ = makeTransformerEngineTensor(
      A.data_ptr(), {static_cast<size_t>(A.size(0)), static_cast<size_t>(A.size(1))}, A_type,
      nullptr, nullptr, A_scale_inv_ptr);

  void *B_scale_inv_ptr = nullptr;
  if (B_scale_inverse.numel()) B_scale_inv_ptr = B_scale_inverse[B_fp8_tensor].data_ptr();
  auto B_ = makeTransformerEngineTensor(
      B.data_ptr(), {static_cast<size_t>(B.size(0)), static_cast<size_t>(B.size(1))}, B_type,
      nullptr, nullptr, B_scale_inv_ptr);

  void *D_amax_ptr = nullptr;
  void *D_scale_ptr = nullptr;
  if (D_amax.numel()) D_amax_ptr = D_amax.data_ptr();
  if (D_scale.numel()) D_scale_ptr = D_scale.data_ptr();
  auto D_ = makeTransformerEngineTensor(
      D.data_ptr(), {static_cast<size_t>(D.size(0)), static_cast<size_t>(D.size(1))}, D_type,
      D_amax_ptr, D_scale_ptr, nullptr);

  auto bias_ =
      makeTransformerEngineTensor(bias.data_ptr(), {static_cast<size_t>(bias.size(0))}, bias_type);

  const auto gelu_shape = (pre_gelu_out.data_ptr() == nullptr)
                              ? std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0))}
                              : std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0)),
                                                    static_cast<size_t>(pre_gelu_out.size(1))};
  auto pre_gelu_out_ = makeTransformerEngineTensor(
      pre_gelu_out.data_ptr(), gelu_shape, GetTransformerEngineDType(pre_gelu_out.scalar_type()));

  std::vector<te::TensorWrapper> ubufs_;
  for (int i = 0; i < _num_ubuf_chunks; i++)
    ubufs_.push_back(makeTransformerEngineTensor(
        _ubufs[i].data_ptr(),
        {static_cast<size_t>(_ubufs[i].size(0)), static_cast<size_t>(_ubufs[i].size(1))}, B_type,
        nullptr, nullptr, B_scale_inv_ptr));

  const auto B_copy_shape = (B_copy.data_ptr() == nullptr)
                                ? std::vector<size_t>{static_cast<size_t>(B_copy.size(0))}
                                : std::vector<size_t>{static_cast<size_t>(B_copy.size(0)),
                                                      static_cast<size_t>(B_copy.size(1))};
  auto B_copy_ = makeTransformerEngineTensor(B_copy.data_ptr(), B_copy_shape, B_type, nullptr,
                                             nullptr, B_scale_inv_ptr);

  auto workspace_ =
      makeTransformerEngineTensor(workspace.data_ptr(), {workspaceSize}, te::DType::kByte);

  at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
  te::CommOverlapP2P::split_gemm_overlap_ag((cudaStream_t)stream_main, A_, transa, B_, transb,
                                            bias_, D_, pre_gelu_out_, ubufs_, B_copy_, workspace_,
                                            grad, accumulate, use_split_accumulator);

  return D;
}  // CommOverlapP2P::split_overlap_ag

/*
** Atomic GEMM + Split Reduce-Scatter using P2P communication
*/
void CommOverlapP2P::atomic_gemm_overlap_rs(
    at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor, te::DType A_type, bool transa,
    at::Tensor B, at::Tensor B_scale_inverse, int64_t B_fp8_tensor, te::DType B_type, bool transb,
    at::Tensor D, at::Tensor D_scale, te::DType D_type, at::Tensor D_amax, at::Tensor bias,
    te::DType bias_type, at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
    size_t workspaceSize, bool accumulate, bool use_split_accumulator, at::Tensor rs_output) {
  void *A_scale_inv_ptr = nullptr;
  if (A_scale_inverse.numel()) A_scale_inv_ptr = A_scale_inverse[A_fp8_tensor].data_ptr();
  auto A_ = makeTransformerEngineTensor(
      A.data_ptr(), {static_cast<size_t>(A.size(0)), static_cast<size_t>(A.size(1))}, A_type,
      nullptr, nullptr, A_scale_inv_ptr);

  void *B_scale_inv_ptr = nullptr;
  if (B_scale_inverse.numel()) B_scale_inv_ptr = B_scale_inverse[B_fp8_tensor].data_ptr();
  auto B_ = makeTransformerEngineTensor(
      B.data_ptr(), {static_cast<size_t>(B.size(0)), static_cast<size_t>(B.size(1))}, B_type,
      nullptr, nullptr, B_scale_inv_ptr);

  void *D_amax_ptr = nullptr;
  void *D_scale_ptr = nullptr;
  if (D_amax.numel()) D_amax_ptr = D_amax.data_ptr();
  if (D_scale.numel()) D_scale_ptr = D_scale.data_ptr();
  auto D_ = makeTransformerEngineTensor(
      D.data_ptr(), {static_cast<size_t>(D.size(0)), static_cast<size_t>(D.size(1))}, D_type,
      D_amax_ptr, D_scale_ptr, nullptr);

  auto bias_ =
      makeTransformerEngineTensor(bias.data_ptr(), {static_cast<size_t>(bias.size(0))}, bias_type);

  const auto gelu_shape = (pre_gelu_out.data_ptr() == nullptr)
                              ? std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0))}
                              : std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0)),
                                                    static_cast<size_t>(pre_gelu_out.size(1))};
  auto pre_gelu_out_ = makeTransformerEngineTensor(
      pre_gelu_out.data_ptr(), gelu_shape, GetTransformerEngineDType(pre_gelu_out.scalar_type()));

  void *_ubuf_scale_inv_ptr = nullptr;
  if (_ubuf.element_size() == 1) {
    assert(_ubuf_scale_inv_initialized);
    _ubuf_scale_inv_ptr = _ubuf_scale_inv.data_ptr();
  }
  auto ubuf_ = makeTransformerEngineTensor(
      _ubuf.data_ptr(), {static_cast<size_t>(_ubuf.size(0)), static_cast<size_t>(_ubuf.size(1))},
      D_type, D_amax_ptr, D_scale_ptr, _ubuf_scale_inv_ptr);

  std::vector<te::TensorWrapper> ubufs_;
  for (int i = 0; i < _num_ubuf_chunks; i++)
    ubufs_.push_back(makeTransformerEngineTensor(
        _ubufs[i].data_ptr(),
        {static_cast<size_t>(_ubufs[i].size(0)), static_cast<size_t>(_ubufs[i].size(1))}, D_type,
        D_amax_ptr, D_scale_ptr, _ubuf_scale_inv_ptr));

  auto workspace_ =
      makeTransformerEngineTensor(workspace.data_ptr(), {workspaceSize}, te::DType::kByte);

  auto counters_ = makeTransformerEngineTensor(
      _counters.data_ptr(), {static_cast<size_t>(_counters.size(0))}, te::DType::kInt32);

  auto rs_out_ = makeTransformerEngineTensor(rs_output);

  at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
  te::CommOverlapP2P::atomic_gemm_overlap_rs(
      (cudaStream_t)stream_main, A_, transa, B_, transb, bias_, D_, pre_gelu_out_, ubuf_, ubufs_,
      counters_, workspace_, grad, accumulate, use_split_accumulator, rs_out_);
}  // CommOverlapP2P::atomic_gemm_overlap_rs

/*
** Pipelined GEMM + Split Reduce+Scatter using P2P communication
*/
void CommOverlapP2P::split_gemm_overlap_rs(
    at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor, te::DType A_type, bool transa,
    at::Tensor B, at::Tensor B_scale_inverse, int64_t B_fp8_tensor, te::DType B_type, bool transb,
    at::Tensor D, at::Tensor D_scale, te::DType D_type, at::Tensor D_amax, at::Tensor bias,
    te::DType bias_type, at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
    size_t workspaceSize, bool accumulate, bool use_split_accumulator, at::Tensor rs_output) {
  void *A_scale_inv_ptr = nullptr;
  if (A_scale_inverse.numel()) A_scale_inv_ptr = A_scale_inverse[A_fp8_tensor].data_ptr();
  auto A_ = makeTransformerEngineTensor(
      A.data_ptr(), {static_cast<size_t>(A.size(0)), static_cast<size_t>(A.size(1))}, A_type,
      nullptr, nullptr, A_scale_inv_ptr);

  void *B_scale_inv_ptr = nullptr;
  if (B_scale_inverse.numel()) B_scale_inv_ptr = B_scale_inverse[B_fp8_tensor].data_ptr();
  auto B_ = makeTransformerEngineTensor(
      B.data_ptr(), {static_cast<size_t>(B.size(0)), static_cast<size_t>(B.size(1))}, B_type,
      nullptr, nullptr, B_scale_inv_ptr);
  void *D_amax_ptr = nullptr;
  void *D_scale_ptr = nullptr;
  if (D_amax.numel()) D_amax_ptr = D_amax.data_ptr();
  if (D_scale.numel()) D_scale_ptr = D_scale.data_ptr();
  auto D_ = makeTransformerEngineTensor(
      D.data_ptr(), {static_cast<size_t>(D.size(0)), static_cast<size_t>(D.size(1))}, D_type,
      D_amax_ptr, D_scale_ptr, nullptr);

  auto bias_ =
      makeTransformerEngineTensor(bias.data_ptr(), {static_cast<size_t>(bias.size(0))}, bias_type);

  const auto gelu_shape = (pre_gelu_out.data_ptr() == nullptr)
                              ? std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0))}
                              : std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0)),
                                                    static_cast<size_t>(pre_gelu_out.size(1))};
  auto pre_gelu_out_ = makeTransformerEngineTensor(
      pre_gelu_out.data_ptr(), gelu_shape, GetTransformerEngineDType(pre_gelu_out.scalar_type()));

  void *_ubuf_scale_inv_ptr = nullptr;
  if (_ubuf.element_size() == 1) {
    assert(_ubuf_scale_inv_initialized);
    _ubuf_scale_inv_ptr = _ubuf_scale_inv.data_ptr();
  }
  std::vector<te::TensorWrapper> ubufs_;
  for (int i = 0; i < _num_ubuf_chunks; i++)
    ubufs_.push_back(makeTransformerEngineTensor(
        _ubufs[i].data_ptr(),
        {static_cast<size_t>(_ubufs[i].size(0)), static_cast<size_t>(_ubufs[i].size(1))}, D_type,
        D_amax_ptr, D_scale_ptr, _ubuf_scale_inv_ptr));

  auto workspace_ =
      makeTransformerEngineTensor(workspace.data_ptr(), {workspaceSize}, te::DType::kByte);

  auto rs_out_ = makeTransformerEngineTensor(rs_output);

  at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
  te::CommOverlapP2P::split_gemm_overlap_rs((cudaStream_t)stream_main, A_, transa, B_, transb,
                                            bias_, D_, pre_gelu_out_, ubufs_, workspace_, grad,
                                            accumulate, use_split_accumulator, rs_out_);
}  // CommOverlapP2P::split_overlap_rs

/*
** Helper function to copy input to _ubuf or _ubufs chunks.
*/
void CommOverlapP2P::copy_input_to_ubuf(torch::Tensor input, te::CommOverlapBuffer buffer_type) {
  at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
  if (buffer_type == te::CommOverlapBuffer::LOCAL) {
    // Copy input to the target ubuf chunk by rank offset
    if (input.numel() != _ubufs[0].numel() || input.element_size() != _ubufs[0].element_size()) {
      NVTE_ERROR("input and ubuf size do not match!");
    }
    NVTE_CHECK_CUDA(cudaMemcpyAsync(_ubufs[_tp_id].data_ptr(), input.data_ptr(),
                                    input.numel() * input.element_size(), cudaMemcpyDeviceToDevice,
                                    (cudaStream_t)stream_main));
  } else {
    if (input.numel() != _ubuf.numel() || input.element_size() != _ubuf.element_size()) {
      NVTE_ERROR("input and ubuf size do not match!");
    }
    NVTE_CHECK_CUDA(cudaMemcpyAsync(_ubuf.data_ptr(), input.data_ptr(),
                                    input.numel() * input.element_size(), cudaMemcpyDeviceToDevice,
                                    (cudaStream_t)stream_main));
  }
}

/*
** Helper function to export _ubuf output.
*/
torch::Tensor CommOverlapP2P::get_ubuf_output(te::CommOverlapBuffer buffer_type) {
  char *ubuf_wt_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());
  int output_c_dim0 = _ubuf.size(0);
  if (buffer_type == te::CommOverlapBuffer::LOCAL) {
    ubuf_wt_ptr += (_ubuf.numel() * _ubuf.element_size() / _tp_size) * _self_chunk_id;
    output_c_dim0 /= _tp_size;
  }
  auto output_tensor =
      torch::from_blob(ubuf_wt_ptr, {output_c_dim0, _ubuf.size(1)}, _ubuf.options());
  return output_tensor;
}

/*
** Helper function to set the inverse Fp8 scale for the _ubuf tensor.
*/
void CommOverlapP2P::set_ubuf_scale_inv(const torch::Tensor &scale_inv) {
  _ubuf_scale_inv = scale_inv;
  _ubuf_scale_inv_initialized = true;
}

}  // namespace transformer_engine_torch
