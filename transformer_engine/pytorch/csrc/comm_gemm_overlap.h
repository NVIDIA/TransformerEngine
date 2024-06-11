/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_COMM_GEMM_OVERLAP_H_
#define TRANSFORMER_ENGINE_PYTORCH_COMM_GEMM_OVERLAP_H_

#include <stdio.h>
#include <stdlib.h>
#include <map>

#include <cuda.h>
#include <cuda_fp8.h>
#if __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#define half nv_bfloat16  // TODO(Alp): Compatibility with userbuffers.cu, will be fixed w/ NVSHMEM.
#else
#include <cuda_fp16.h>
#endif

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/cuda.h>
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/types.h>

#include "transformer_engine/transformer_engine.h"
#include "common/util/logging.h"
#include "common/util/system.h"
#include "common/userbuffers/comm_gemm_overlap.h"
#include "common/userbuffers/userbuffers.h"

#include "common.h"

#define HALF_BYTES 2
#define UB_MAX_SM 32

using namespace torch::indexing;

namespace transformer_engine {

namespace comm_gemm_overlap {

/*
** Static container for Python callbacks to torch.distributed collectives
*/
static struct TorchCollectiveCallbacks : torch::CustomClassHolder {
  bool initialized{false};
  std::unordered_map<void *, at::Tensor> gathered_tensors;
  std::function<at::Tensor(at::Tensor&, const std::string &)> allgather;
  std::function<at::Tensor(at::Tensor &, int, const std::string &)> bcast;
  std::function<void(const std::string &)> barrier;
  std::function<void(at::Tensor &)> free;
} torch_callbacks;

/*
** Helper function for setting Python callbacks to torch.distributed collectives.
*/
void set_bootstrap_callbacks(
  std::function<at::Tensor(at::Tensor&, const std::string &)> allgather,
  std::function<at::Tensor(at::Tensor &, int, const std::string &)> bcast,
  std::function<void(const std::string &)> barrier,
  std::function<void(at::Tensor &)> free
) {
  torch_callbacks.allgather = allgather;
  torch_callbacks.bcast = bcast;
  torch_callbacks.barrier = barrier;
  torch_callbacks.free = free;
  torch_callbacks.initialized = true;
}

/*
** Python callback for globaldata = torch.distributed.all_gather(localdata, tp_group).
** This *creates* a new tensor, which Userbuffers later frees with a separate callback.
*/
void torch_alloc_copy_allgather(void **globaldata, void *localdata, size_t localbytes, char *group) {
  assert(torch_callbacks.initialized);
  auto localtensor = torch::from_blob(
    localdata, {static_cast<int64_t>(localbytes / sizeof(uint8_t))},
    at::device(torch::kCPU).dtype(torch::kUInt8));
  auto globaltensor = torch_callbacks.allgather(localtensor, group);
  *globaldata = globaltensor.data_ptr();
  torch_callbacks.gathered_tensors[*globaldata] = globaltensor;
}

/*
** Python callback for torch.distributed.broadcast(data, src, tp_group).
** If broadcast is via NCCL, casting the datatensor to CUDA device and back to host CPU will
** create a new tensor and leave the original data pointer dangling. In this case, we copy the
** broadcasted data from the new tensor into the original data pointer and leave the new tensor
** to be garbage collected in Python.
*/
void torch_bcast(void *data, size_t bytes, int src, char *group) {
  assert(torch_callbacks.initialized);
  auto datatensor = torch::from_blob(
    data, {static_cast<int64_t>(bytes / sizeof(uint8_t))},
    at::device(torch::kCPU).dtype(torch::kUInt8));
  datatensor = torch_callbacks.bcast(datatensor, src, group);
  if (datatensor.data_ptr() != data)
    memcpy(data, datatensor.data_ptr(), bytes);
}

/*
** Python callback for torch.distributed.barrier(tp_group).
*/
void torch_barrier(char *group) {
  assert(torch_callbacks.initialized);
  torch_callbacks.barrier(group);
}

/*
** Python callback for freeing up tensors created in the torch_alloc_copy_allgather(...) callback.
*/
void torch_free(void *ptr) {
  assert(torch_callbacks.initialized);
  auto i = torch_callbacks.gathered_tensors.find(ptr);
  if (i == torch_callbacks.gathered_tensors.end())
    return;
  auto tensor = std::move(i->second);
  torch_callbacks.gathered_tensors.erase(i);
  torch_callbacks.free(tensor);
}

struct PYBIND11_EXPORT UbufCommOverlap : torch::CustomClassHolder, CommGemmOverlap {
  torch::Tensor _counters;
  torch::Tensor _ubuf;
  int _ubuf_bytes;
  DType _ubuf_dtype;
  void *_ubuf_scale_inv_ptr;
  bool _ubuf_scale_inv_initialized{false};

  UbufCommOverlap(
    torch::Tensor sample, int world_rank, int world_size, int tp_rank, int tp_size,
    int num_splits, int num_max_streams, int comm_cga_size, int num_comm_sm,
    bool set_sm_margin, bool atomic_gemm)
  : CommGemmOverlap(world_rank, world_size, tp_rank, tp_size, 0, 1, num_splits, num_max_streams,
                    comm_cga_size, num_comm_sm, set_sm_margin, atomic_gemm,
                    &torch_alloc_copy_allgather, &torch_bcast, &torch_barrier, &torch_free) {
    _ubuf_bytes = sample.numel() * sample.element_size();
    _ubuf_dtype = (sample.element_size() == 1) ? DType::kFloat8E4M3
                                               : GetTransformerEngineDType(sample.scalar_type());

    void *ubuf_ptr;
    if (transformer_engine::getenv<bool>("UB_SKIPMC")) {
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
    }
  }

  /*
  ** Bulk GEMM + COMM
  ** This function assumes the communication input is pre-copied to _ubuf.
  */
  std::vector<at::Tensor> bulk_overlap(
    at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
    transformer_engine::DType A_type, bool transa,
    at::Tensor B, at::Tensor B_scale_inverse, int64_t B_fp8_tensor,
    transformer_engine::DType B_type, bool transb,
    at::Tensor D, at::Tensor D_scale, transformer_engine::DType D_type, at::Tensor D_amax,
    at::Tensor bias, transformer_engine::DType bias_type, at::Tensor pre_gelu_out, bool grad,
    at::Tensor workspace, size_t workspaceSize, bool accumulate, bool use_split_accumulator,
    NVTE_Comm_Overlap_Type comm_type, at::Tensor rs_output
  ) {
    if (_ubuf.element_size() == 1) {
      NVTE_CHECK(_ubuf_scale_inv_initialized, "Missing userbuffers FP8 inverse scale!");
    }
    auto ubuf_ = makeTransformerEngineTensor(_ubuf.data_ptr(),
                                             {static_cast<size_t>(_ubuf.size(0)),
                                              static_cast<size_t>(_ubuf.size(1))},
                                             _ubuf_dtype, nullptr, nullptr,
                                             _ubuf_scale_inv_ptr);

    void *A_scale_inv_ptr = nullptr;
    if (A_scale_inverse.numel()) A_scale_inv_ptr = A_scale_inverse[A_fp8_tensor].data_ptr();
    auto A_ = makeTransformerEngineTensor(A.data_ptr(),
                                          {static_cast<size_t>(A.size(0)),
                                           static_cast<size_t>(A.size(1))},
                                          A_type, nullptr, nullptr,
                                          A_scale_inv_ptr);

    void *B_scale_inv_ptr = nullptr;
    if (B_scale_inverse.numel()) B_scale_inv_ptr = B_scale_inverse[B_fp8_tensor].data_ptr();
    auto B_ = makeTransformerEngineTensor(B.data_ptr(),
                                          {static_cast<size_t>(B.size(0)),
                                           static_cast<size_t>(B.size(1))},
                                          B_type, nullptr, nullptr,
                                          B_scale_inv_ptr);
    void *D_amax_ptr = nullptr;
    void *D_scale_ptr = nullptr;
    if (D_amax.numel()) D_amax_ptr = D_amax.data_ptr();
    if (D_scale.numel()) D_scale_ptr = D_scale.data_ptr();
    auto D_ = makeTransformerEngineTensor(D.data_ptr(),
                                          {static_cast<size_t>(D.size(0)),
                                           static_cast<size_t>(D.size(1))},
                                          D_type, D_amax_ptr, D_scale_ptr, nullptr);

    auto bias_ = makeTransformerEngineTensor(bias.data_ptr(),
                                             {static_cast<size_t>(bias.size(0))},
                                             bias_type);

    const auto gelu_shape = (pre_gelu_out.data_ptr() == nullptr)
                            ? std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0))}
                            : std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0)),
                                                  static_cast<size_t>(pre_gelu_out.size(1))};
    auto pre_gelu_out_ = makeTransformerEngineTensor(pre_gelu_out.data_ptr(),
                                                     gelu_shape,
                                                     GetTransformerEngineDType(
                                                       pre_gelu_out.scalar_type()));

    auto workspace_ = makeTransformerEngineTensor(workspace.data_ptr(),
                                                  {workspaceSize},
                                                  DType::kByte);

    auto rs_out_ = makeTransformerEngineTensor(rs_output.data_ptr(),
                                               {static_cast<size_t>(rs_output.size(0)),
                                                static_cast<size_t>(rs_output.size(1))},
                                               DType::kBFloat16);

    at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
    CommGemmOverlap::bulk_gemm_overlap((cudaStream_t)stream_main, A_, transa, B_, transb, bias_,
                                       D_, pre_gelu_out_, ubuf_, rs_out_, workspace_,
                                       grad, accumulate, use_split_accumulator, comm_type);

    // Generate output tensor from userbuf data pointer
    char *ubuf_wt_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());
    if (comm_type == NVTE_Comm_Overlap_Type::REDUCE_SCATTER) {
      ubuf_wt_ptr += (_ubuf.numel() * _ubuf.element_size() / _tp_size) * _tp_id;
    }
    int output_c_dim0 = (comm_type == NVTE_Comm_Overlap_Type::ALL_GATHER)
                        ? _ubuf.size(0) : _ubuf.size(0) / _tp_size;
    int output_c_dim1 = _ubuf.size(1);
    torch::Tensor output_tensor = torch::from_blob(
      ubuf_wt_ptr, {output_c_dim0, output_c_dim1}, _ubuf.options());

    return {D, output_tensor};
  }  // UbufCommOverlap::bulk_overlap

  /*
  ** Atomic GEMM + Split Reduce-Scatter
  */
  void atomic_gemm_overlap_rs(
    at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
    transformer_engine::DType A_type, bool transa,
    at::Tensor B, at::Tensor B_scale_inverse, int64_t B_fp8_tensor,
    transformer_engine::DType B_type, bool transb,
    at::Tensor D, at::Tensor D_scale, transformer_engine::DType D_type, at::Tensor D_amax,
    at::Tensor bias, transformer_engine::DType bias_type, at::Tensor pre_gelu_out, bool grad,
    at::Tensor workspace, size_t workspaceSize, bool accumulate, bool use_split_accumulator,
    bool gemm_overlap, at::Tensor rs_output
  ) {
    if (_ubuf.element_size() == 1) {
      NVTE_CHECK(_ubuf_scale_inv_initialized, "Missing userbuffers FP8 inverse scale!");
    }
    auto ubuf_ = makeTransformerEngineTensor(_ubuf.data_ptr(),
                                             {static_cast<size_t>(_ubuf.size(0)),
                                              static_cast<size_t>(_ubuf.size(1))},
                                             _ubuf_dtype, nullptr, nullptr,
                                             _ubuf_scale_inv_ptr);

    void *A_scale_inv_ptr = nullptr;
    if (A_scale_inverse.numel()) A_scale_inv_ptr = A_scale_inverse[A_fp8_tensor].data_ptr();
    auto A_ = makeTransformerEngineTensor(A.data_ptr(),
                                          {static_cast<size_t>(A.size(0)),
                                           static_cast<size_t>(A.size(1))},
                                          A_type, nullptr, nullptr,
                                          A_scale_inv_ptr);

    void *B_scale_inv_ptr = nullptr;
    if (B_scale_inverse.numel()) B_scale_inv_ptr = B_scale_inverse[B_fp8_tensor].data_ptr();
    auto B_ = makeTransformerEngineTensor(B.data_ptr(),
                                          {static_cast<size_t>(B.size(0)),
                                           static_cast<size_t>(B.size(1))},
                                          B_type, nullptr, nullptr,
                                          B_scale_inv_ptr);
    void *D_amax_ptr = nullptr;
    void *D_scale_ptr = nullptr;
    if (D_amax.numel()) D_amax_ptr = D_amax.data_ptr();
    if (D_scale.numel()) D_scale_ptr = D_scale.data_ptr();
    auto D_ = makeTransformerEngineTensor(D.data_ptr(),
                                          {static_cast<size_t>(D.size(0)),
                                           static_cast<size_t>(D.size(1))},
                                          D_type, D_amax_ptr, D_scale_ptr, nullptr);

    auto bias_ = makeTransformerEngineTensor(bias.data_ptr(),
                                             {static_cast<size_t>(bias.size(0))},
                                             bias_type);

    const auto gelu_shape = (pre_gelu_out.data_ptr() == nullptr)
                            ? std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0))}
                            : std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0)),
                                                  static_cast<size_t>(pre_gelu_out.size(1))};
    auto pre_gelu_out_ = makeTransformerEngineTensor(pre_gelu_out.data_ptr(),
                                                     gelu_shape,
                                                     GetTransformerEngineDType(
                                                       pre_gelu_out.scalar_type()));

    auto workspace_ = makeTransformerEngineTensor(workspace.data_ptr(),
                                                  {workspaceSize},
                                                  DType::kByte);

    auto counters_ = makeTransformerEngineTensor(_counters.data_ptr(),
                                                 {static_cast<size_t>(_counters.size(0))},
                                                 DType::kInt32);

    auto rs_out_ = makeTransformerEngineTensor(rs_output.data_ptr(),
                                               {static_cast<size_t>(rs_output.size(0)),
                                                static_cast<size_t>(rs_output.size(1))},
                                               DType::kBFloat16);

    at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
    CommGemmOverlap::atomic_gemm_overlap_rs(
      (cudaStream_t)stream_main, A_, transa, B_, transb, bias_,
      D_, pre_gelu_out_, ubuf_, rs_out_, counters_, workspace_,
      grad, accumulate, use_split_accumulator);
  }  // UbufCommOverlap::atomic_gemm_overlap_rs

  /*
  ** Pipelined GEMM + Split Reduce-Scatter
  */
  void split_overlap_rs(
    at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
    transformer_engine::DType A_type, bool transa,
    at::Tensor B, at::Tensor B_scale_inverse, int64_t B_fp8_tensor,
    transformer_engine::DType B_type, bool transb,
    at::Tensor D, at::Tensor D_scale, transformer_engine::DType D_type, at::Tensor D_amax,
    at::Tensor bias, transformer_engine::DType bias_type, at::Tensor pre_gelu_out, bool grad,
    at::Tensor workspace, size_t workspaceSize, bool accumulate, bool use_split_accumulator,
    bool gemm_overlap, at::Tensor rs_output
  ) {
    void *A_scale_inv_ptr = nullptr;
    if (A_scale_inverse.numel()) A_scale_inv_ptr = A_scale_inverse[A_fp8_tensor].data_ptr();
    auto A_ = makeTransformerEngineTensor(A.data_ptr(),
                                          {static_cast<size_t>(A.size(0)),
                                           static_cast<size_t>(A.size(1))},
                                          A_type, nullptr, nullptr,
                                          A_scale_inv_ptr);

    void *B_scale_inv_ptr = nullptr;
    if (B_scale_inverse.numel()) B_scale_inv_ptr = B_scale_inverse[B_fp8_tensor].data_ptr();
    auto B_ = makeTransformerEngineTensor(B.data_ptr(),
                                          {static_cast<size_t>(B.size(0)),
                                           static_cast<size_t>(B.size(1))},
                                          B_type, nullptr, nullptr,
                                          B_scale_inv_ptr);
    void *D_amax_ptr = nullptr;
    void *D_scale_ptr = nullptr;
    if (D_amax.numel()) D_amax_ptr = D_amax.data_ptr();
    if (D_scale.numel()) D_scale_ptr = D_scale.data_ptr();
    auto D_ = makeTransformerEngineTensor(D.data_ptr(),
                                          {static_cast<size_t>(D.size(0)),
                                           static_cast<size_t>(D.size(1))},
                                          D_type, D_amax_ptr, D_scale_ptr, nullptr);

    auto bias_ = makeTransformerEngineTensor(bias.data_ptr(),
                                             {static_cast<size_t>(bias.size(0))},
                                             bias_type);

    const auto gelu_shape = (pre_gelu_out.data_ptr() == nullptr)
                            ? std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0))}
                            : std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0)),
                                                  static_cast<size_t>(pre_gelu_out.size(1))};
    auto pre_gelu_out_ = makeTransformerEngineTensor(pre_gelu_out.data_ptr(),
                                                     gelu_shape,
                                                     GetTransformerEngineDType(
                                                       pre_gelu_out.scalar_type()));

    if (_ubuf.element_size() == 1) {
      NVTE_CHECK(_ubuf_scale_inv_initialized, "Missing userbuffers FP8 inverse scale!");
    }
    auto ubuf_ = makeTransformerEngineTensor(_ubuf.data_ptr(),
                                             {static_cast<size_t>(_ubuf.size(0)),
                                              static_cast<size_t>(_ubuf.size(1))},
                                             _ubuf_dtype, D_amax_ptr, D_scale_ptr,
                                             _ubuf_scale_inv_ptr);

    auto workspace_ = makeTransformerEngineTensor(workspace.data_ptr(),
                                                  {workspaceSize},
                                                  DType::kByte);

    auto rs_out_ = makeTransformerEngineTensor(rs_output.data_ptr(),
                                               {static_cast<size_t>(rs_output.size(0)),
                                                static_cast<size_t>(rs_output.size(1))},
                                               DType::kBFloat16);

    at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
    CommGemmOverlap::split_gemm_overlap_rs(
      (cudaStream_t)stream_main, A_, transa, B_, transb, bias_,
      D_, pre_gelu_out_, ubuf_, rs_out_, workspace_,
      grad, accumulate, use_split_accumulator, gemm_overlap);
  }  // UbufCommOverlap::split_overlap_rs

  /*
  ** Helper function to copy input to _ubuf.
  */
  void copy_input_to_ubuf(torch::Tensor input, bool chunk) {
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

    at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
    NVTE_CHECK_CUDA(cudaEventRecord(_start_d2dcopy, (cudaStream_t)stream_main));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_comm, _start_d2dcopy, 0));
    NVTE_CHECK_CUDA(
      cudaMemcpyAsync(ubuf_ptr, input.data_ptr(), input.numel() * input.element_size(),
                      cudaMemcpyDeviceToDevice, _stream_comm));
  }

  /*
  ** Helper function to export _ubuf output.
  */
  torch::Tensor get_ubuf_output(NVTE_Comm_Overlap_Type comm_type) {
    char *ubuf_wt_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());
    int output_c_dim0 = _ubuf.size(0);
    if (comm_type == NVTE_Comm_Overlap_Type::REDUCE_SCATTER) {
      ubuf_wt_ptr += (_ubuf.numel() * _ubuf.element_size() / _tp_size) * _tp_id;
      output_c_dim0 /= _tp_size;
    }
    auto output_tensor = torch::from_blob(ubuf_wt_ptr, {output_c_dim0, _ubuf.size(1)},
                                          _ubuf.options());
    return output_tensor;
  }

  /*
  ** Helper function to set the inverse Fp8 scale for the _ubuf tensor.
  */
  void set_ubuf_scale_inv(const torch::Tensor &scale_inv) {
    _ubuf_scale_inv_ptr = scale_inv.data_ptr();
    _ubuf_scale_inv_initialized = true;
  }

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

  UbufP2PCommOverlap(torch::Tensor sample,
    int world_rank, int world_size, int tp_rank, int tp_size, int num_max_streams,
    bool set_sm_margin, bool atomic_gemm, bool aggregate, bool is_reduce_scatter)
  : CommGemmOverlapP2P(world_rank, world_size, tp_rank, tp_size, /* node_id */ 0, /* num_nodes */ 1,
                       num_max_streams, set_sm_margin, atomic_gemm, aggregate, is_reduce_scatter,
                       &torch_alloc_copy_allgather, &torch_bcast, &torch_barrier, &torch_free) {
    _ubuf_bytes = sample.numel() * sample.element_size();
    _ubuf_chunk_bytes = _ubuf_bytes / _tp_size;
    if (_is_reduce_scatter) {
      // GEMM + RS overlap: Allocate `2 x tp_size - 1` buffers to hold recieved GEMM chunk
      // outputs for reduction at the end of the pipelining.
      _ubuf_bytes = static_cast<int>((_ubuf_bytes / _tp_size) * (_tp_size * 2 - 1));
    }
    _ubuf_dtype = (sample.element_size() == 1) ? DType::kFloat8E4M3
                                               : GetTransformerEngineDType(sample.scalar_type());

    void *ubuf_ptr;
    if (transformer_engine::getenv<bool>("UB_SKIPMC")) {
      // Multicast is disabled so we have to pre-allocate the buffer here.
      _ubuf = torch::empty({(sample.size(0) / _tp_size) * _num_ubuf_chunks, sample.size(1)},
                           sample.options());
      ubuf_ptr = _ubuf.data_ptr();
      this->register_gpu_buffer(&ubuf_ptr, _ubuf_bytes, false);
    } else {
      // Multicast requires UB to allocate the buffer with specific memory options
      // that PyTorch allocator does not support.
      this->register_gpu_buffer(&ubuf_ptr, _ubuf_bytes, true);
      _ubuf = torch::from_blob(ubuf_ptr,
                               {(sample.size(0) / _tp_size) * _num_ubuf_chunks, sample.size(1)},
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
      _counters.index_put_({Slice(None, _tp_size)}, 1);
      if (!_is_reduce_scatter) {
        _counters.index_put_({_self_chunk_id /* = 0 for AG + atomic GEMM */}, 0);
      }
    }
  }

  /*
  ** Split AllGather + Atomic GEMM using P2P communication
  ** This function assumes the input_b is pre-copied to _ubufs[rank_id]. This is needed to have AG
  ** outputs in each rank to be in the contiguous memory space after all ring exchange phases.
  */
  torch::Tensor atomic_gemm_overlap_ag(
    at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
    transformer_engine::DType A_type, bool transa,
    at::Tensor B, at::Tensor B_scale_inverse, int64_t B_fp8_tensor,
    transformer_engine::DType B_type, bool transb,
    at::Tensor D, at::Tensor D_scale, transformer_engine::DType D_type, at::Tensor D_amax,
    at::Tensor bias, transformer_engine::DType bias_type, at::Tensor pre_gelu_out, bool grad,
    at::Tensor workspace, size_t workspaceSize, bool accumulate, bool use_split_accumulator,
    at::Tensor B_copy
  ) {
    if (_ubuf.element_size() == 1) {
      NVTE_CHECK(_ubuf_scale_inv_initialized, "Missing userbuffers FP8 inverse scale!");
    }
    auto ubuf_ = makeTransformerEngineTensor(_ubuf.data_ptr(),
                                             {static_cast<size_t>(_ubuf.size(0)),
                                              static_cast<size_t>(_ubuf.size(1))},
                                             _ubuf_dtype, nullptr, nullptr,
                                             _ubuf_scale_inv_ptr);

    std::vector<TensorWrapper> ubufs_;
    for (int i = 0; i < _num_ubuf_chunks; i++)
      ubufs_.push_back(makeTransformerEngineTensor(_ubufs[i].data_ptr(),
                                                   {static_cast<size_t>(_ubufs[i].size(0)),
                                                    static_cast<size_t>(_ubufs[i].size(1))},
                                                   _ubuf_dtype, nullptr, nullptr,
                                                   _ubuf_scale_inv_ptr));

    void *A_scale_inv_ptr = nullptr;
    if (A_scale_inverse.numel()) A_scale_inv_ptr = A_scale_inverse[A_fp8_tensor].data_ptr();
    auto A_ = makeTransformerEngineTensor(A.data_ptr(),
                                          {static_cast<size_t>(A.size(0)),
                                           static_cast<size_t>(A.size(1))},
                                          A_type, nullptr, nullptr,
                                          A_scale_inv_ptr);

    void *B_scale_inv_ptr = nullptr;
    if (B_scale_inverse.numel()) B_scale_inv_ptr = B_scale_inverse[B_fp8_tensor].data_ptr();
    auto B_ = makeTransformerEngineTensor(B.data_ptr(),
                                          {static_cast<size_t>(B.size(0)),
                                           static_cast<size_t>(B.size(1))},
                                          B_type, nullptr, nullptr,
                                          B_scale_inv_ptr);

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
    auto D_ = makeTransformerEngineTensor(D.data_ptr(),
                                          {static_cast<size_t>(D.size(0)),
                                           static_cast<size_t>(D.size(1))},
                                          D_type, D_amax_ptr, D_scale_ptr, nullptr);

    auto bias_ = makeTransformerEngineTensor(bias.data_ptr(),
                                             {static_cast<size_t>(bias.size(0))},
                                             bias_type);

    const auto gelu_shape = (pre_gelu_out.data_ptr() == nullptr)
                            ? std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0))}
                            : std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0)),
                                                  static_cast<size_t>(pre_gelu_out.size(1))};
    auto pre_gelu_out_ = makeTransformerEngineTensor(pre_gelu_out.data_ptr(),
                                                     gelu_shape,
                                                     GetTransformerEngineDType(
                                                       pre_gelu_out.scalar_type()));

    auto workspace_ = makeTransformerEngineTensor(workspace.data_ptr(),
                                                  {workspaceSize},
                                                  DType::kByte);

    auto counters_ = makeTransformerEngineTensor(_counters.data_ptr(),
                                                 {static_cast<size_t>(_counters.size(0))},
                                                 DType::kInt32);

    const auto B_copy_shape = (B_copy.data_ptr() == nullptr)
                              ? std::vector<size_t>{static_cast<size_t>(B_copy.size(0))}
                              : std::vector<size_t>{static_cast<size_t>(B_copy.size(0)),
                                                    static_cast<size_t>(B_copy.size(1))};
    auto B_copy_ = makeTransformerEngineTensor(B_copy.data_ptr(),
                                               B_copy_shape, B_type,
                                               nullptr, nullptr, B_scale_inv_ptr);

    auto D_buffer_ = makeTransformerEngineTensor(D_buffer.data_ptr(),
                                                 {static_cast<size_t>(D_buffer.size(0)),
                                                  static_cast<size_t>(D_buffer.size(1))},
                                                 D_type, D_amax_ptr, D_scale_ptr, nullptr);

    at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
    CommGemmOverlapP2P::atomic_gemm_overlap_ag(
      (cudaStream_t)stream_main, A_, transa, B_, transb, bias_,
      D_, pre_gelu_out_, ubuf_, ubufs_, counters_, B_copy_, D_buffer_, workspace_,
      grad, accumulate, use_split_accumulator);

    // Return the last N rows of D_buffer
    torch::Tensor D_return = D_buffer.narrow(0, n_chunk, n);
    return D_return;
  }  // UbufP2PCommOverlap::atomic_gemm_overlap_ag

  /*
  ** Split AllGather + Pipelined GEMM using P2P communication
  ** This function assumes the input_b is pre-copied to _ubufs[rank_id]. This is needed to have AG
  ** outputs in each rank to be in the contiguous memory space after all ring exchange phases.
  */
  torch::Tensor split_overlap_ag(
    at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
    transformer_engine::DType A_type, bool transa,
    at::Tensor B, at::Tensor B_scale_inverse, int64_t B_fp8_tensor,
    transformer_engine::DType B_type, bool transb,
    at::Tensor D, at::Tensor D_scale, transformer_engine::DType D_type, at::Tensor D_amax,
    at::Tensor bias, transformer_engine::DType bias_type, at::Tensor pre_gelu_out, bool grad,
    at::Tensor workspace, size_t workspaceSize, bool accumulate, bool use_split_accumulator,
    at::Tensor B_copy
  ) {
    if (_ubuf.element_size() == 1) {
      NVTE_CHECK(_ubuf_scale_inv_initialized, "Missing userbuffers FP8 inverse scale!");
    }
    std::vector<TensorWrapper> ubufs_;
    for (int i = 0; i < _num_ubuf_chunks; i++)
      ubufs_.push_back(makeTransformerEngineTensor(_ubufs[i].data_ptr(),
                                                   {static_cast<size_t>(_ubufs[i].size(0)),
                                                    static_cast<size_t>(_ubufs[i].size(1))},
                                                   _ubuf_dtype, nullptr, nullptr,
                                                   _ubuf_scale_inv_ptr));

    void *A_scale_inv_ptr = nullptr;
    if (A_scale_inverse.numel()) A_scale_inv_ptr = A_scale_inverse[A_fp8_tensor].data_ptr();
    auto A_ = makeTransformerEngineTensor(A.data_ptr(),
                                          {static_cast<size_t>(A.size(0)),
                                           static_cast<size_t>(A.size(1))},
                                          A_type, nullptr, nullptr,
                                          A_scale_inv_ptr);

    void *B_scale_inv_ptr = nullptr;
    if (B_scale_inverse.numel()) B_scale_inv_ptr = B_scale_inverse[B_fp8_tensor].data_ptr();
    auto B_ = makeTransformerEngineTensor(B.data_ptr(),
                                          {static_cast<size_t>(B.size(0)),
                                           static_cast<size_t>(B.size(1))},
                                          B_type, nullptr, nullptr,
                                          B_scale_inv_ptr);

    void *D_amax_ptr = nullptr;
    void *D_scale_ptr = nullptr;
    if (D_amax.numel()) D_amax_ptr = D_amax.data_ptr();
    if (D_scale.numel()) D_scale_ptr = D_scale.data_ptr();
    auto D_ = makeTransformerEngineTensor(D.data_ptr(),
                                          {static_cast<size_t>(D.size(0)),
                                           static_cast<size_t>(D.size(1))},
                                          D_type, D_amax_ptr, D_scale_ptr, nullptr);

    auto bias_ = makeTransformerEngineTensor(bias.data_ptr(),
                                             {static_cast<size_t>(bias.size(0))},
                                             bias_type);

    const auto gelu_shape = (pre_gelu_out.data_ptr() == nullptr)
                            ? std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0))}
                            : std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0)),
                                                  static_cast<size_t>(pre_gelu_out.size(1))};
    auto pre_gelu_out_ = makeTransformerEngineTensor(pre_gelu_out.data_ptr(),
                                                     gelu_shape,
                                                     GetTransformerEngineDType(
                                                       pre_gelu_out.scalar_type()));

    auto workspace_ = makeTransformerEngineTensor(workspace.data_ptr(),
                                                  {workspaceSize},
                                                  DType::kByte);

    const auto B_copy_shape = (B_copy.data_ptr() == nullptr)
                              ? std::vector<size_t>{static_cast<size_t>(B_copy.size(0))}
                              : std::vector<size_t>{static_cast<size_t>(B_copy.size(0)),
                                                    static_cast<size_t>(B_copy.size(1))};
    auto B_copy_ = makeTransformerEngineTensor(B_copy.data_ptr(),
                                               B_copy_shape, B_type,
                                               nullptr, nullptr, B_scale_inv_ptr);

    at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
    CommGemmOverlapP2P::split_gemm_overlap_ag(
      (cudaStream_t)stream_main, A_, transa, B_, transb, bias_,
      D_, pre_gelu_out_, ubufs_, B_copy_, workspace_,
      grad, accumulate, use_split_accumulator);
    at::cuda::setCurrentCUDAStream(stream_main);

    return D;
  }  // UbufP2PCommOverlap::split_overlap_ag

  /*
  ** Atomic GEMM + Split Reduce-Scatter using P2P communication
  */
  void atomic_gemm_overlap_rs(
    at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
    transformer_engine::DType A_type, bool transa,
    at::Tensor B, at::Tensor B_scale_inverse, int64_t B_fp8_tensor,
    transformer_engine::DType B_type, bool transb,
    at::Tensor D, at::Tensor D_scale, transformer_engine::DType D_type, at::Tensor D_amax,
    at::Tensor bias, transformer_engine::DType bias_type, at::Tensor pre_gelu_out, bool grad,
    at::Tensor workspace, size_t workspaceSize, bool accumulate, bool use_split_accumulator,
    at::Tensor rs_output
  ) {
    void *A_scale_inv_ptr = nullptr;
    if (A_scale_inverse.numel()) A_scale_inv_ptr = A_scale_inverse[A_fp8_tensor].data_ptr();
    auto A_ = makeTransformerEngineTensor(A.data_ptr(),
                                          {static_cast<size_t>(A.size(0)),
                                           static_cast<size_t>(A.size(1))},
                                          A_type, nullptr, nullptr,
                                          A_scale_inv_ptr);

    void *B_scale_inv_ptr = nullptr;
    if (B_scale_inverse.numel()) B_scale_inv_ptr = B_scale_inverse[B_fp8_tensor].data_ptr();
    auto B_ = makeTransformerEngineTensor(B.data_ptr(),
                                          {static_cast<size_t>(B.size(0)),
                                           static_cast<size_t>(B.size(1))},
                                          B_type, nullptr, nullptr,
                                          B_scale_inv_ptr);
    void *D_amax_ptr = nullptr;
    void *D_scale_ptr = nullptr;
    if (D_amax.numel()) D_amax_ptr = D_amax.data_ptr();
    if (D_scale.numel()) D_scale_ptr = D_scale.data_ptr();
    auto D_ = makeTransformerEngineTensor(D.data_ptr(),
                                          {static_cast<size_t>(D.size(0)),
                                           static_cast<size_t>(D.size(1))},
                                          D_type, D_amax_ptr, D_scale_ptr, nullptr);

    auto bias_ = makeTransformerEngineTensor(bias.data_ptr(),
                                             {static_cast<size_t>(bias.size(0))},
                                             bias_type);

    const auto gelu_shape = (pre_gelu_out.data_ptr() == nullptr)
                            ? std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0))}
                            : std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0)),
                                                  static_cast<size_t>(pre_gelu_out.size(1))};
    auto pre_gelu_out_ = makeTransformerEngineTensor(pre_gelu_out.data_ptr(),
                                                     gelu_shape,
                                                     GetTransformerEngineDType(
                                                       pre_gelu_out.scalar_type()));

    if (_ubuf.element_size() == 1) {
      NVTE_CHECK(_ubuf_scale_inv_initialized, "Missing userbuffers FP8 inverse scale!");
    }
    auto ubuf_ = makeTransformerEngineTensor(_ubuf.data_ptr(),
                                             {static_cast<size_t>(_ubuf.size(0)),
                                              static_cast<size_t>(_ubuf.size(1))},
                                             _ubuf_dtype, D_amax_ptr, D_scale_ptr,
                                             _ubuf_scale_inv_ptr);

    std::vector<TensorWrapper> ubufs_;
    for (int i = 0; i < _num_ubuf_chunks; i++)
      ubufs_.push_back(makeTransformerEngineTensor(_ubufs[i].data_ptr(),
                                                   {static_cast<size_t>(_ubufs[i].size(0)),
                                                    static_cast<size_t>(_ubufs[i].size(1))},
                                                   _ubuf_dtype, D_amax_ptr, D_scale_ptr,
                                                   _ubuf_scale_inv_ptr));

    auto workspace_ = makeTransformerEngineTensor(workspace.data_ptr(),
                                                  {workspaceSize},
                                                  DType::kByte);

    auto counters_ = makeTransformerEngineTensor(_counters.data_ptr(),
                                                 {static_cast<size_t>(_counters.size(0))},
                                                 DType::kInt32);

    at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
    CommGemmOverlapP2P::atomic_gemm_overlap_rs(
      (cudaStream_t)stream_main, A_, transa, B_, transb, bias_,
      D_, pre_gelu_out_, ubuf_, ubufs_, counters_, workspace_,
      grad, accumulate, use_split_accumulator);

    // Reduce GEMM output chunks
    char *reduce_buf_ptr = reinterpret_cast<char *>(_ubufs[_tp_size - 1].data_ptr());
    if (_ubuf.element_size() == 1) {
      assert(rs_output.element_size() == 2);
      float *d_scale_inv_ptr = reinterpret_cast<float *>(_ubuf_scale_inv_ptr);
      char *rs_output_ptr = reinterpret_cast<char *>(rs_output.data_ptr());
      reduce_fp8_in_bf16_out<__nv_fp8_e4m3>(reduce_buf_ptr, rs_output_ptr, d_scale_inv_ptr,
                                            _tp_size, _ubufs[0].numel(), (cudaStream_t)stream_main);
    } else {
      torch::Tensor reduce_buf = torch::from_blob(
        reduce_buf_ptr, {_tp_size, _ubufs[0].size(0), _ubufs[0].size(1)}, _ubuf.options());
      rs_output = torch::sum_out(rs_output, reduce_buf, 0);
    }
  }  // UbufP2PCommOverlap::atomic_gemm_overlap_rs

  /*
  ** Pipelined GEMM + Split Reduce+Scatter using P2P communication
  */
  void split_overlap_rs(
    at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
    transformer_engine::DType A_type, bool transa, at::Tensor B,
    at::Tensor B_scale_inverse, int64_t B_fp8_tensor,
    transformer_engine::DType B_type, bool transb,
    at::Tensor D, at::Tensor D_scale, transformer_engine::DType D_type, at::Tensor D_amax,
    at::Tensor bias, transformer_engine::DType bias_type, at::Tensor pre_gelu_out, bool grad,
    at::Tensor workspace, size_t workspaceSize, bool accumulate, bool use_split_accumulator,
    at::Tensor rs_output
  ) {
    void *A_scale_inv_ptr = nullptr;
    if (A_scale_inverse.numel()) A_scale_inv_ptr = A_scale_inverse[A_fp8_tensor].data_ptr();
    auto A_ = makeTransformerEngineTensor(A.data_ptr(),
                                          {static_cast<size_t>(A.size(0)),
                                           static_cast<size_t>(A.size(1))},
                                          A_type, nullptr, nullptr,
                                          A_scale_inv_ptr);

    void *B_scale_inv_ptr = nullptr;
    if (B_scale_inverse.numel()) B_scale_inv_ptr = B_scale_inverse[B_fp8_tensor].data_ptr();
    auto B_ = makeTransformerEngineTensor(B.data_ptr(),
                                          {static_cast<size_t>(B.size(0)),
                                           static_cast<size_t>(B.size(1))},
                                          B_type, nullptr, nullptr,
                                          B_scale_inv_ptr);
    void *D_amax_ptr = nullptr;
    void *D_scale_ptr = nullptr;
    if (D_amax.numel()) D_amax_ptr = D_amax.data_ptr();
    if (D_scale.numel()) D_scale_ptr = D_scale.data_ptr();
    auto D_ = makeTransformerEngineTensor(D.data_ptr(),
                                          {static_cast<size_t>(D.size(0)),
                                           static_cast<size_t>(D.size(1))},
                                          D_type, D_amax_ptr, D_scale_ptr, nullptr);

    auto bias_ = makeTransformerEngineTensor(bias.data_ptr(),
                                             {static_cast<size_t>(bias.size(0))},
                                             bias_type);

    const auto gelu_shape = (pre_gelu_out.data_ptr() == nullptr)
                            ? std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0))}
                            : std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0)),
                                                  static_cast<size_t>(pre_gelu_out.size(1))};
    auto pre_gelu_out_ = makeTransformerEngineTensor(pre_gelu_out.data_ptr(),
                                                     gelu_shape,
                                                     GetTransformerEngineDType(
                                                       pre_gelu_out.scalar_type()));

    if (_ubuf.element_size() == 1) {
      NVTE_CHECK(_ubuf_scale_inv_initialized, "Missing userbuffers FP8 inverse scale!");
    }
    std::vector<TensorWrapper> ubufs_;
    for (int i = 0; i < _num_ubuf_chunks; i++)
      ubufs_.push_back(makeTransformerEngineTensor(_ubufs[i].data_ptr(),
                                                   {static_cast<size_t>(_ubufs[i].size(0)),
                                                    static_cast<size_t>(_ubufs[i].size(1))},
                                                   _ubuf_dtype, D_amax_ptr, D_scale_ptr,
                                                   _ubuf_scale_inv_ptr));

    auto workspace_ = makeTransformerEngineTensor(workspace.data_ptr(),
                                                  {workspaceSize},
                                                  DType::kByte);

    at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
    CommGemmOverlapP2P::split_gemm_overlap_rs(
      (cudaStream_t)stream_main, A_, transa, B_, transb, bias_,
      D_, pre_gelu_out_, ubufs_, workspace_,
      grad, accumulate, use_split_accumulator);

    // Reduce GEMM output chunks
    char *reduce_buf_ptr = reinterpret_cast<char *>(_ubufs[_tp_size - 1].data_ptr());
    if (_ubuf.element_size() == 1) {
      assert(rs_output.element_size() == 2);
      float *d_scale_inv_ptr = reinterpret_cast<float *>(_ubuf_scale_inv_ptr);
      char *rs_output_ptr = reinterpret_cast<char *>(rs_output.data_ptr());
      reduce_fp8_in_bf16_out<__nv_fp8_e4m3>(reduce_buf_ptr, rs_output_ptr, d_scale_inv_ptr,
                                            _tp_size, _ubufs[0].numel(), (cudaStream_t)stream_main);
    } else {
      torch::Tensor reduce_buf = torch::from_blob(
        reduce_buf_ptr, {_tp_size, _ubufs[0].size(0), _ubufs[0].size(1)}, _ubuf.options());
      rs_output = torch::sum_out(rs_output, reduce_buf, 0);
    }
    for (size_t i = 0; i < _stream_compute.size(); i++) {
      NVTE_CHECK_CUDA(
          cudaEventRecord(_stop_compute, _stream_compute[i]));
      NVTE_CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_compute, 0));
    }

    NVTE_CHECK_CUDA(cudaEventRecord(_stop_send, _stream_send));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_send, 0));
  }  // UbufP2PCommOverlap::split_overlap_rs


  /*
  ** Helper function to copy input to _ubuf or _ubufs chunks.
  */
  void copy_input_to_ubuf(torch::Tensor input, bool chunk) {
    at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
    if (chunk) {
      // Copy input to the target ubuf chunk by rank offset
      if (input.numel() != _ubufs[0].numel() || input.element_size() != _ubufs[0].element_size()) {
        NVTE_ERROR("input and ubuf size do not match!");
      }
      NVTE_CHECK_CUDA(cudaMemcpyAsync(_ubufs[_tp_id].data_ptr(), input.data_ptr(),
                                      input.numel() * input.element_size(),
                                      cudaMemcpyDeviceToDevice, (cudaStream_t)stream_main));
    } else {
      if (input.numel() != _ubuf.numel() || input.element_size() != _ubuf.element_size()) {
        NVTE_ERROR("input and ubuf size do not match!");
      }
      NVTE_CHECK_CUDA(cudaMemcpyAsync(_ubuf.data_ptr(), input.data_ptr(),
                                      input.numel() * input.element_size(),
                                      cudaMemcpyDeviceToDevice, (cudaStream_t)stream_main));
    }
  }

  /*
  ** Helper function to export _ubuf output.
  */
  torch::Tensor get_ubuf_output(NVTE_Comm_Overlap_Type comm_type) {
    char *ubuf_wt_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());
    int output_c_dim0 = _ubuf.size(0);
    if (comm_type == NVTE_Comm_Overlap_Type::REDUCE_SCATTER) {
      ubuf_wt_ptr += (_ubuf.numel() * _ubuf.element_size() / _tp_size) * _self_chunk_id;
      output_c_dim0 /= _tp_size;
    }
    auto output_tensor = torch::from_blob(ubuf_wt_ptr, {output_c_dim0, _ubuf.size(1)},
                                          _ubuf.options());
    return output_tensor;
  }

  /*
  ** Helper function to set the inverse Fp8 scale for the _ubuf tensor.
  */
  void set_ubuf_scale_inv(const torch::Tensor &scale_inv) {
    _ubuf_scale_inv_ptr = scale_inv.data_ptr();
    _ubuf_scale_inv_initialized = true;
  }

  bool is_fp8_ubuf() { return (_ubuf.element_size() == 1); }
};  // UbufP2PCommOverlap

}  // namespace userbuffers

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_PYTORCH_COMM_GEMM_OVERLAP_H_
