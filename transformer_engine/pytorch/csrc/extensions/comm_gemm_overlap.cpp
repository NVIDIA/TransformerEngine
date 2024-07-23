/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#if __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#define half nv_bfloat16  // TODO(Alp): Compatibility with userbuffers.cu, will be fixed w/ NVSHMEM.
#else
#include <cuda_fp16.h>
#endif

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <common/common.h>
#include <common/util/logging.h>
#include <common/util/system.h>
#include <torch/cuda.h>
#include <torch/types.h>

#include "../common.h"
#include "../extensions.h"

#define HALF_BYTES 2
#define UB_MAX_SM 32

using namespace torch::indexing;
namespace te = transformer_engine;
namespace te_torch = transformer_engine_torch;

/*
** Static container for Python callbacks to torch.distributed collectives
*/
static struct TorchDistCallbacks : torch::CustomClassHolder {
  bool initialized{false};
  std::function<void(at::Tensor &, at::Tensor &, const std::string &)> allgather;
  std::function<void(at::Tensor &, int64_t, const std::string &)> bcast;
  std::function<void(const std::string &)> barrier;
} torch_dist_callbacks;

void set_comm_overlap_callbacks(
    std::function<void(at::Tensor &, at::Tensor &, const std::string &)> allgather_callback,
    std::function<void(at::Tensor &, int64_t, const std::string &)> bcast_callback,
    std::function<void(const std::string &)> barrier_callback) {
  torch_dist_callbacks.allgather = allgather_callback;
  torch_dist_callbacks.bcast = bcast_callback;
  torch_dist_callbacks.barrier = barrier_callback;
  torch_dist_callbacks.initialized = true;
}

/*
** Python callback for torch.distributed.all_gather_into_tensor(global_data, localdata, tp_group).
*/
void ub_torch_allgather(void *globaldata, size_t globalbytes, void *localdata, size_t localbytes,
                        char *group) {
  NVTE_CHECK(
      torch_dist_callbacks.initialized,
      "tex.set_comm_overlap_callbacks() must be called before initializing overlap communciator.");
  auto localtensor =
      torch::from_blob(localdata, {static_cast<int64_t>(localbytes / sizeof(uint8_t))},
                       at::device(torch::kCPU).dtype(torch::kUInt8));
  auto globaltensor =
      torch::from_blob(globaldata, {static_cast<int64_t>(globalbytes / sizeof(uint8_t))},
                       at::device(torch::kCPU).dtype(torch::kUInt8));
  torch_dist_callbacks.allgather(globaltensor, localtensor, group);
  if (globaltensor.data_ptr() != globaldata) {
    memcpy(globaldata, globaltensor.data_ptr(), globalbytes);
  }
}

/*
** Python callback for torch.distributed.broadcast(data, src, tp_group).
*/
void ub_torch_bcast(void *data, size_t bytes, int64_t src, char *group) {
  NVTE_CHECK(
      torch_dist_callbacks.initialized,
      "tex.set_comm_overlap_callbacks() must be called before initializing overlap communciator.");
  auto datatensor = torch::from_blob(data, {static_cast<int64_t>(bytes / sizeof(uint8_t))},
                                     at::device(torch::kCPU).dtype(torch::kUInt8));
  torch_dist_callbacks.bcast(datatensor, src, group);
  if (datatensor.data_ptr() != data) {
    memcpy(data, datatensor.data_ptr(), bytes);
  }
}

/*
** Python callback for torch.distributed.barrier(tp_group).
*/
void ub_torch_barrier(char *group) {
  NVTE_CHECK(
      torch_dist_callbacks.initialized,
      "tex.set_comm_overlap_callbacks() must be called before initializing overlap communciator.");
  torch_dist_callbacks.barrier(group);
}

/***************************************************************************************************
** CommGemmOverlap -- Collective (pipelined) comm+GEMM wrappers for PyTorch
***************************************************************************************************/

te_torch::CommGemmOverlap::CommGemmOverlap(torch::Tensor sample, int world_rank, int world_size,
                                           int local_rank, int local_size, int node_id,
                                           int num_nodes, int tp_size, int num_splits,
                                           int num_max_streams, int cga_size, int num_comm_sm,
                                           bool set_sm_margin, bool use_ce, bool atomic_gemm)
    : te::common::CommGemmOverlap(world_rank, world_size, local_rank, local_size, node_id,
                                  num_nodes, tp_size, num_splits, num_max_streams, cga_size,
                                  num_comm_sm, set_sm_margin, use_ce, atomic_gemm,
                                  &ub_torch_allgather, &ub_torch_bcast, &ub_torch_barrier) {
  _ubuf_bytes = sample.numel() * sample.element_size();
  _ubuf_dtype = (sample.element_size() == 1) ? te::DType::kFloat8E4M3
                                             : GetTransformerEngineDType(sample.scalar_type());
  void *ubuf_ptr;
  if (te::getenv<bool>("UB_SKIPMC")) {
    // Multicast is disabled so we have to pre-allocate the buffer here.
    _ubuf = torch::empty({sample.size(0), sample.size(1)}, sample.options());
    ubuf_ptr = _ubuf.data_ptr();
    this->register_gpu_buffer(&ubuf_ptr, _ubuf_bytes, false);
  } else {
    // Multicast requires UB to allocate the buffer with specific memory options
    // that PyTorch allocator does not support.
    this->register_gpu_buffer(&ubuf_ptr, _ubuf_bytes, true);
    _ubuf = torch::from_blob(ubuf_ptr, {sample.size(0), sample.size(1)}, sample.options());
  }

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
std::vector<at::Tensor> te_torch::CommGemmOverlap::bulk_overlap(
    at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor, te::DType A_type, bool transa,
    at::Tensor B, at::Tensor B_scale_inverse, int64_t B_fp8_tensor, te::DType B_type, bool transb,
    at::Tensor D, at::Tensor D_scale, te::DType D_type, at::Tensor D_amax, at::Tensor bias,
    te::DType bias_type, at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
    size_t workspaceSize, bool accumulate, bool use_split_accumulator,
    NVTE_Comm_Overlap_Type comm_type, at::Tensor rs_output) {
  if (_ubuf.element_size() == 1) {
    NVTE_CHECK(_ubuf_scale_inv_initialized, "Missing userbuffers FP8 inverse scale!");
  }
  auto ubuf_ = makeTransformerEngineTensor(
      _ubuf.data_ptr(), {static_cast<size_t>(_ubuf.size(0)), static_cast<size_t>(_ubuf.size(1))},
      _ubuf_dtype, nullptr, nullptr, _ubuf_scale_inv_ptr);

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

  auto workspace_ =
      makeTransformerEngineTensor(workspace.data_ptr(), {workspaceSize}, te::DType::kByte);

  void *rs_out_ptr = nullptr;
  auto rs_out_shape = std::vector<size_t>{static_cast<size_t>(rs_output.size(0))};
  if (comm_type == NVTE_Comm_Overlap_Type::REDUCE_SCATTER && _ubuf.element_size() == 1) {
    rs_out_ptr = rs_output.data_ptr();
    rs_out_shape.push_back(static_cast<size_t>(rs_output.size(1)));
  }
  auto rs_out_ = makeTransformerEngineTensor(rs_out_ptr, rs_out_shape, te::DType::kBFloat16);

  at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
  te::common::CommGemmOverlap::bulk_gemm_overlap(
      (cudaStream_t)stream_main, A_, transa, B_, transb, bias_, D_, pre_gelu_out_, ubuf_, rs_out_,
      workspace_, grad, accumulate, use_split_accumulator, comm_type);

  // Generate output tensor from userbuf data pointer
  char *ubuf_wt_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());
  if (comm_type == NVTE_Comm_Overlap_Type::REDUCE_SCATTER) {
    ubuf_wt_ptr += (_ubuf.numel() * _ubuf.element_size() / _tp_size) * _tp_id;
  }
  int output_c_dim0 =
      (comm_type == NVTE_Comm_Overlap_Type::ALL_GATHER) ? _ubuf.size(0) : _ubuf.size(0) / _tp_size;
  int output_c_dim1 = _ubuf.size(1);
  torch::Tensor output_tensor =
      torch::from_blob(ubuf_wt_ptr, {output_c_dim0, output_c_dim1}, _ubuf.options());

  return {D, output_tensor};
}  // CommGemmOverlap::bulk_overlap

/*
** Atomic GEMM + Split Reduce-Scatter
*/
void te_torch::CommGemmOverlap::atomic_gemm_overlap_rs(
    at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor, te::DType A_type, bool transa,
    at::Tensor B, at::Tensor B_scale_inverse, int64_t B_fp8_tensor, te::DType B_type, bool transb,
    at::Tensor D, at::Tensor D_scale, te::DType D_type, at::Tensor D_amax, at::Tensor bias,
    te::DType bias_type, at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
    size_t workspaceSize, bool accumulate, bool use_split_accumulator, bool gemm_overlap,
    at::Tensor rs_output) {
  if (_ubuf.element_size() == 1) {
    NVTE_CHECK(_ubuf_scale_inv_initialized, "Missing userbuffers FP8 inverse scale!");
  }
  auto ubuf_ = makeTransformerEngineTensor(
      _ubuf.data_ptr(), {static_cast<size_t>(_ubuf.size(0)), static_cast<size_t>(_ubuf.size(1))},
      _ubuf_dtype, nullptr, nullptr, _ubuf_scale_inv_ptr);

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

  auto workspace_ =
      makeTransformerEngineTensor(workspace.data_ptr(), {workspaceSize}, te::DType::kByte);

  auto counters_ = makeTransformerEngineTensor(
      _counters.data_ptr(), {static_cast<size_t>(_counters.size(0))}, te::DType::kInt32);

  auto rs_out_ = makeTransformerEngineTensor(
      rs_output.data_ptr(),
      {static_cast<size_t>(rs_output.size(0)), static_cast<size_t>(rs_output.size(1))},
      te::DType::kBFloat16);

  at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
  te::common::CommGemmOverlap::atomic_gemm_overlap_rs(
      (cudaStream_t)stream_main, A_, transa, B_, transb, bias_, D_, pre_gelu_out_, ubuf_, rs_out_,
      counters_, workspace_, grad, accumulate, use_split_accumulator);
}  // CommGemmOverlap::atomic_gemm_overlap_rs

/*
** Pipelined GEMM + Split Reduce-Scatter
*/
void te_torch::CommGemmOverlap::split_overlap_rs(
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

  if (_ubuf.element_size() == 1) {
    NVTE_CHECK(_ubuf_scale_inv_initialized, "Missing userbuffers FP8 inverse scale!");
  }
  auto ubuf_ = makeTransformerEngineTensor(
      _ubuf.data_ptr(), {static_cast<size_t>(_ubuf.size(0)), static_cast<size_t>(_ubuf.size(1))},
      _ubuf_dtype, D_amax_ptr, D_scale_ptr, _ubuf_scale_inv_ptr);

  auto workspace_ =
      makeTransformerEngineTensor(workspace.data_ptr(), {workspaceSize}, te::DType::kByte);

  auto rs_out_ = makeTransformerEngineTensor(
      rs_output.data_ptr(),
      {static_cast<size_t>(rs_output.size(0)), static_cast<size_t>(rs_output.size(1))},
      te::DType::kBFloat16);

  at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
  te::common::CommGemmOverlap::split_gemm_overlap_rs(
      (cudaStream_t)stream_main, A_, transa, B_, transb, bias_, D_, pre_gelu_out_, ubuf_, rs_out_,
      workspace_, grad, accumulate, use_split_accumulator, gemm_overlap);
}  // CommGemmOverlap::split_overlap_rs

/*
** Helper function to copy input to _ubuf.
*/
void te_torch::CommGemmOverlap::copy_input_to_ubuf(torch::Tensor input, bool chunk) {
  char *ubuf_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());
  if (chunk) {
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
};

/*
** Helper function to export _ubuf output.
*/
torch::Tensor te_torch::CommGemmOverlap::get_ubuf_output(NVTE_Comm_Overlap_Type comm_type) {
  char *ubuf_wt_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());
  int output_c_dim0 = _ubuf.size(0);
  if (comm_type == NVTE_Comm_Overlap_Type::REDUCE_SCATTER) {
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
void te_torch::CommGemmOverlap::set_ubuf_scale_inv(const torch::Tensor &scale_inv) {
  _ubuf_scale_inv_ptr = scale_inv.data_ptr();
  _ubuf_scale_inv_initialized = true;
}

/***************************************************************************************************
** CommGemmOverlapP2P -- Point-2-Point (ring-exchange) comm+GEMM wrappers for PyTorch
***************************************************************************************************/

te_torch::CommGemmOverlapP2P::CommGemmOverlapP2P(
    torch::Tensor sample, int world_rank, int world_size, int local_rank, int local_size,
    int node_id, int num_nodes, int tp_size, int num_max_streams, int cga_size, int num_comm_sms,
    bool set_sm_margin, bool use_ce, bool atomic_gemm, bool aggregate, bool is_reduce_scatter)
    : te::common::CommGemmOverlapP2P(
          world_rank, world_size, local_rank, local_size, node_id, num_nodes, tp_size,
          num_max_streams, cga_size, num_comm_sms, set_sm_margin, use_ce, atomic_gemm, aggregate,
          is_reduce_scatter, &ub_torch_allgather, &ub_torch_bcast, &ub_torch_barrier) {
  _ubuf_bytes = sample.numel() * sample.element_size();
  _ubuf_chunk_bytes = _ubuf_bytes / _tp_size;
  if (_is_reduce_scatter) {
    // GEMM + RS overlap: Allocate `2 x tp_size - 1` buffers to hold recieved GEMM chunk
    // outputs for reduction at the end of the pipelining.
    _ubuf_bytes = static_cast<int>((_ubuf_bytes / _tp_size) * (_tp_size * 2 - 1));
  }
  _ubuf_dtype = (sample.element_size() == 1) ? te::DType::kFloat8E4M3
                                             : GetTransformerEngineDType(sample.scalar_type());

  void *ubuf_ptr;
  if (te::getenv<bool>("UB_SKIPMC")) {
    // Multicast is disabled so we have to pre-allocate the buffer here.
    _ubuf = torch::empty({(sample.size(0) / _tp_size) * _num_ubuf_chunks, sample.size(1)},
                         sample.options());
    ubuf_ptr = _ubuf.data_ptr();
    this->register_gpu_buffer(&ubuf_ptr, _ubuf_bytes, false);
  } else {
    // Multicast requires UB to allocate the buffer with specific memory options
    // that PyTorch allocator does not support.
    this->register_gpu_buffer(&ubuf_ptr, _ubuf_bytes, true);
    _ubuf =
        torch::from_blob(ubuf_ptr, {(sample.size(0) / _tp_size) * _num_ubuf_chunks, sample.size(1)},
                         sample.options());
  }

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
torch::Tensor te_torch::CommGemmOverlapP2P::atomic_gemm_overlap_ag(
    at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor, te::DType A_type, bool transa,
    at::Tensor B, at::Tensor B_scale_inverse, int64_t B_fp8_tensor, te::DType B_type, bool transb,
    at::Tensor D, at::Tensor D_scale, te::DType D_type, at::Tensor D_amax, at::Tensor bias,
    te::DType bias_type, at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
    size_t workspaceSize, bool accumulate, bool use_split_accumulator, at::Tensor B_copy) {
  if (_ubuf.element_size() == 1) {
    NVTE_CHECK(_ubuf_scale_inv_initialized, "Missing userbuffers FP8 inverse scale!");
  }
  auto ubuf_ = makeTransformerEngineTensor(
      _ubuf.data_ptr(), {static_cast<size_t>(_ubuf.size(0)), static_cast<size_t>(_ubuf.size(1))},
      _ubuf_dtype, nullptr, nullptr, _ubuf_scale_inv_ptr);

  std::vector<te::TensorWrapper> ubufs_;
  for (int i = 0; i < _num_ubuf_chunks; i++)
    ubufs_.push_back(makeTransformerEngineTensor(
        _ubufs[i].data_ptr(),
        {static_cast<size_t>(_ubufs[i].size(0)), static_cast<size_t>(_ubufs[i].size(1))},
        _ubuf_dtype, nullptr, nullptr, _ubuf_scale_inv_ptr));

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

  auto workspace_ =
      makeTransformerEngineTensor(workspace.data_ptr(), {workspaceSize}, te::DType::kByte);

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

  at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
  te::common::CommGemmOverlapP2P::atomic_gemm_overlap_ag(
      (cudaStream_t)stream_main, A_, transa, B_, transb, bias_, D_, pre_gelu_out_, ubuf_, ubufs_,
      counters_, B_copy_, D_buffer_, workspace_, grad, accumulate, use_split_accumulator);

  // Return the last N rows of D_buffer
  torch::Tensor D_return = D_buffer.narrow(0, n_chunk, n);
  return D_return;
}  // CommGemmOverlapP2P::atomic_gemm_overlap_ag

/*
** Split AllGather + Pipelined GEMM using P2P communication
** This function assumes the input_b is pre-copied to _ubufs[rank_id]. This is needed to have AG
** outputs in each rank to be in the contiguous memory space after all ring exchange phases.
*/
torch::Tensor te_torch::CommGemmOverlapP2P::split_overlap_ag(
    at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor, te::DType A_type, bool transa,
    at::Tensor B, at::Tensor B_scale_inverse, int64_t B_fp8_tensor, te::DType B_type, bool transb,
    at::Tensor D, at::Tensor D_scale, te::DType D_type, at::Tensor D_amax, at::Tensor bias,
    te::DType bias_type, at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
    size_t workspaceSize, bool accumulate, bool use_split_accumulator, at::Tensor B_copy) {
  if (_ubuf.element_size() == 1) {
    NVTE_CHECK(_ubuf_scale_inv_initialized, "Missing userbuffers FP8 inverse scale!");
  }
  std::vector<te::TensorWrapper> ubufs_;
  for (int i = 0; i < _num_ubuf_chunks; i++)
    ubufs_.push_back(makeTransformerEngineTensor(
        _ubufs[i].data_ptr(),
        {static_cast<size_t>(_ubufs[i].size(0)), static_cast<size_t>(_ubufs[i].size(1))},
        _ubuf_dtype, nullptr, nullptr, _ubuf_scale_inv_ptr));

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

  auto workspace_ =
      makeTransformerEngineTensor(workspace.data_ptr(), {workspaceSize}, te::DType::kByte);

  const auto B_copy_shape = (B_copy.data_ptr() == nullptr)
                                ? std::vector<size_t>{static_cast<size_t>(B_copy.size(0))}
                                : std::vector<size_t>{static_cast<size_t>(B_copy.size(0)),
                                                      static_cast<size_t>(B_copy.size(1))};
  auto B_copy_ = makeTransformerEngineTensor(B_copy.data_ptr(), B_copy_shape, B_type, nullptr,
                                             nullptr, B_scale_inv_ptr);

  at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
  te::common::CommGemmOverlapP2P::split_gemm_overlap_ag(
      (cudaStream_t)stream_main, A_, transa, B_, transb, bias_, D_, pre_gelu_out_, ubufs_, B_copy_,
      workspace_, grad, accumulate, use_split_accumulator);

  return D;
}  // CommGemmOverlapP2P::split_overlap_ag

/*
** Atomic GEMM + Split Reduce-Scatter using P2P communication
*/
void te_torch::CommGemmOverlapP2P::atomic_gemm_overlap_rs(
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

  if (_ubuf.element_size() == 1) {
    NVTE_CHECK(_ubuf_scale_inv_initialized, "Missing userbuffers FP8 inverse scale!");
  }
  auto ubuf_ = makeTransformerEngineTensor(
      _ubuf.data_ptr(), {static_cast<size_t>(_ubuf.size(0)), static_cast<size_t>(_ubuf.size(1))},
      _ubuf_dtype, D_amax_ptr, D_scale_ptr, _ubuf_scale_inv_ptr);

  std::vector<te::TensorWrapper> ubufs_;
  for (int i = 0; i < _num_ubuf_chunks; i++)
    ubufs_.push_back(makeTransformerEngineTensor(
        _ubufs[i].data_ptr(),
        {static_cast<size_t>(_ubufs[i].size(0)), static_cast<size_t>(_ubufs[i].size(1))},
        _ubuf_dtype, D_amax_ptr, D_scale_ptr, _ubuf_scale_inv_ptr));

  auto workspace_ =
      makeTransformerEngineTensor(workspace.data_ptr(), {workspaceSize}, te::DType::kByte);

  auto counters_ = makeTransformerEngineTensor(
      _counters.data_ptr(), {static_cast<size_t>(_counters.size(0))}, te::DType::kInt32);

  at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
  te::common::CommGemmOverlapP2P::atomic_gemm_overlap_rs(
      (cudaStream_t)stream_main, A_, transa, B_, transb, bias_, D_, pre_gelu_out_, ubuf_, ubufs_,
      counters_, workspace_, grad, accumulate, use_split_accumulator);

  // Reduce GEMM output chunks
  char *reduce_buf_ptr = reinterpret_cast<char *>(_ubufs[_tp_size - 1].data_ptr());
  if (_ubuf.element_size() == 1) {
    assert(rs_output.element_size() == 2);
    float *d_scale_inv_ptr = reinterpret_cast<float *>(_ubuf_scale_inv_ptr);
    char *rs_output_ptr = reinterpret_cast<char *>(rs_output.data_ptr());
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
        D_type, fp8_type,
        reduce_fp8_in_bf16_out<fp8_type>(reduce_buf_ptr, rs_output_ptr, d_scale_inv_ptr, _tp_size,
                                         _ubufs[0].numel(), (cudaStream_t)stream_main););
  } else {
    torch::Tensor reduce_buf = torch::from_blob(
        reduce_buf_ptr, {_tp_size, _ubufs[0].size(0), _ubufs[0].size(1)}, _ubuf.options());
    rs_output = torch::sum_out(rs_output, reduce_buf, 0);
  }
}  // CommGemmOverlapP2P::atomic_gemm_overlap_rs

/*
** Pipelined GEMM + Split Reduce+Scatter using P2P communication
*/
void te_torch::CommGemmOverlapP2P::split_overlap_rs(
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

  if (_ubuf.element_size() == 1) {
    NVTE_CHECK(_ubuf_scale_inv_initialized, "Missing userbuffers FP8 inverse scale!");
  }
  std::vector<te::TensorWrapper> ubufs_;
  for (int i = 0; i < _num_ubuf_chunks; i++)
    ubufs_.push_back(makeTransformerEngineTensor(
        _ubufs[i].data_ptr(),
        {static_cast<size_t>(_ubufs[i].size(0)), static_cast<size_t>(_ubufs[i].size(1))},
        _ubuf_dtype, D_amax_ptr, D_scale_ptr, _ubuf_scale_inv_ptr));

  auto workspace_ =
      makeTransformerEngineTensor(workspace.data_ptr(), {workspaceSize}, te::DType::kByte);

  at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
  te::common::CommGemmOverlapP2P::split_gemm_overlap_rs(
      (cudaStream_t)stream_main, A_, transa, B_, transb, bias_, D_, pre_gelu_out_, ubufs_,
      workspace_, grad, accumulate, use_split_accumulator);

  // Reduce GEMM output chunks
  char *reduce_buf_ptr = reinterpret_cast<char *>(_ubufs[_tp_size - 1].data_ptr());
  if (_ubuf.element_size() == 1) {
    assert(rs_output.element_size() == 2);
    float *d_scale_inv_ptr = reinterpret_cast<float *>(_ubuf_scale_inv_ptr);
    char *rs_output_ptr = reinterpret_cast<char *>(rs_output.data_ptr());
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
        D_type, fp8_type,
        reduce_fp8_in_bf16_out<fp8_type>(reduce_buf_ptr, rs_output_ptr, d_scale_inv_ptr, _tp_size,
                                         _ubufs[0].numel(), (cudaStream_t)stream_main););
  } else {
    torch::Tensor reduce_buf = torch::from_blob(
        reduce_buf_ptr, {_tp_size, _ubufs[0].size(0), _ubufs[0].size(1)}, _ubuf.options());
    rs_output = torch::sum_out(rs_output, reduce_buf, 0);
  }
  for (size_t i = 0; i < _stream_compute.size(); i++) {
    NVTE_CHECK_CUDA(cudaEventRecord(_stop_compute, _stream_compute[i]));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_compute, 0));
  }

  NVTE_CHECK_CUDA(cudaEventRecord(_stop_send, _stream_send));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_send, 0));
}  // CommGemmOverlapP2P::split_overlap_rs

/*
** Helper function to copy input to _ubuf or _ubufs chunks.
*/
void te_torch::CommGemmOverlapP2P::copy_input_to_ubuf(torch::Tensor input, bool chunk) {
  at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
  if (chunk) {
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
torch::Tensor te_torch::CommGemmOverlapP2P::get_ubuf_output(NVTE_Comm_Overlap_Type comm_type) {
  char *ubuf_wt_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());
  int output_c_dim0 = _ubuf.size(0);
  if (comm_type == NVTE_Comm_Overlap_Type::REDUCE_SCATTER) {
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
void te_torch::CommGemmOverlapP2P::set_ubuf_scale_inv(const torch::Tensor &scale_inv) {
  _ubuf_scale_inv_ptr = scale_inv.data_ptr();
  _ubuf_scale_inv_initialized = true;
}
