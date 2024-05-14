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

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp8.h>
#include <torch/cuda.h>
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/types.h>

#include "transformer_engine/transformer_engine.h"
#include "common/util/logging.h"
#include "common/util/system.h"
#include "common/userbuffers/comm_gemm_overlap.h"

#define HALF_BYTES 2
#define UB_MAX_SM 32

using namespace torch::indexing;

namespace transformer_engine {

namespace userbuffers {

static struct TorchCallbacks : torch::CustomClassHolder {
  bool initialized{false};
  std::unordered_map<void *, at::Tensor> gathered_tensors;
  std::function<at::Tensor(at::Tensor&, const std::string &)> allgather;
  std::function<void(at::Tensor &, int, const std::string &)> bcast_int;
  std::function<void(const std::string &)> barrier;
  std::function<void(at::Tensor &)> free;
} torch_callbacks;

void set_collective_callbacks(
  std::function<at::Tensor(at::Tensor&, const std::string &)> allgather,
  std::function<void(at::Tensor &, int, const std::string &)> bcast_int,
  std::function<void(const std::string &)> barrier,
  std::function<void(at::Tensor &)> free
) {
  torch_callbacks.allgather = allgather;
  torch_callbacks.bcast_int = bcast_int;
  torch_callbacks.barrier = barrier;
  torch_callbacks.free = free;
  torch_callbacks.initialized = true;
}

void ub_alloc_copy_allgather(void **globaldata, void *localdata, size_t localbytes, char *group) {
  assert(torch_callbacks.initialized);
  auto localtensor = torch::from_blob(
    localdata, {static_cast<int64_t>(localbytes / sizeof(uint8_t))},
    at::device(torch::kCPU).dtype(torch::kUInt8));
  auto globaltensor = torch_callbacks.allgather(localtensor, group);
  *globaldata = globaltensor.data_ptr();
  torch_callbacks.gathered_tensors[*globaldata] = globaltensor;
}

void ub_bcast_int(void *data, int src, char *group) {
  assert(torch_callbacks.initialized);
  auto datatensor = torch::from_blob(data, {1}, at::device(torch::kCPU).dtype(torch::kUInt8));
  torch_callbacks.bcast_int(datatensor, src, group);
}

void ub_barrier(char *group) {
  assert(torch_callbacks.initialized);
  torch_callbacks.barrier(group);
}

void ub_free(void *ptr) {
  assert(torch_callbacks.initialized);
  auto i = torch_callbacks.gathered_tensors.find(ptr);
  if (i == torch_callbacks.gathered_tensors.end())
    return;
  auto tensor = std::move(i->second);
  torch_callbacks.gathered_tensors.erase(i);
  torch_callbacks.free(tensor);
}

DType torch_dtype_to_te(torch::Dtype torch_type) {
  switch (torch_type) {
    case torch::kInt32:
      return DType::kInt32;
    case torch::kInt64:
      return DType::kInt64;
    case torch::kFloat16:
      return DType::kFloat16;
    case torch::kBFloat16:
      return DType::kBFloat16;
    default:
      return DType::kByte;
  }
}

TensorWrapper torch_tensor_to_te(torch::Tensor inp,
                                 torch::Tensor amax = torch::Tensor(),
                                 torch::Tensor scale = torch::Tensor(),
                                 torch::Tensor scale_inv = torch::Tensor(),
                                 int64_t fp8_idx = -1) {
  float *amax_ptr = nullptr;
  if (amax.numel())
    amax_ptr = reinterpret_cast<float *>(amax.data_ptr());

  float *scale_ptr = nullptr;
  if(scale.numel())
    scale_ptr = reinterpret_cast<float *>(scale.data_ptr());

  float *scale_inv_ptr = nullptr;
  if(scale_inv.numel()) {
    if (fp8_idx >= 0)
      scale_inv = scale_inv[fp8_idx];
    scale_inv_ptr = reinterpret_cast<float *>(scale_inv[fp8_idx].data_ptr());
  }

  std::vector<size_t> inp_shape;
  for (int64_t i=0; i<inp.ndimension(); i++)
    inp_shape.push_back(static_cast<size_t>(inp.size(i)));

  return TensorWrapper(inp.data_ptr(), inp_shape, torch_dtype_to_te(inp.scalar_type()),
                       amax_ptr, scale_ptr, scale_inv_ptr);
}

struct PYBIND11_EXPORT UbufCommOverlap : torch::CustomClassHolder, CommGemmOverlap {
  torch::Tensor _counters;
  torch::Tensor _ubuf;
  torch::Tensor _ubuf_scale_inv;
  bool _ubuf_scale_inv_initialized{false};

  UbufCommOverlap(
    torch::Tensor sample, int world_rank, int world_size, int tp_rank, int tp_size,
    int num_splits, int num_max_streams, int comm_cga_size, int num_comm_sm,
    bool set_sm_margin, bool atomic_gemm)
  : CommGemmOverlap(world_rank, world_size, tp_rank, tp_size, 0, 1, num_splits, num_max_streams,
                    comm_cga_size, num_comm_sm, set_sm_margin, atomic_gemm,
                    &ub_alloc_copy_allgather, &ub_bcast_int, &ub_barrier, &ub_free) {
    // Allocate and register extra userbuffers
    void *ubuf_ptr;
    size_t ubuf_bytes = sample.numel() * sample.element_size();
    bool alloc = true;
    if (transformer_engine::getenv<bool>("UB_SKIPMC")) {
      alloc = false;
      NVTE_CHECK_CUDA(cudaMalloc(&ubuf_ptr, ubuf_bytes));
    }
    this->register_gpu_buffer(&ubuf_ptr, ubuf_bytes, alloc);
    _ubuf = torch::from_blob(ubuf_ptr, {sample.size(0), sample.size(1)}, sample.options());

    if (_atomic_gemm) {
      auto counter_options = at::device(torch::kCUDA).dtype(torch::kInt32);
      _counters = torch::zeros({num_splits * 2}, counter_options);
      _counters.index_put_({Slice(None, num_splits)}, 1);
    }
  }

  ~UbufCommOverlap() {
    if (transformer_engine::getenv<bool>("UB_SKIPMC")) {
      cudaFree(_ubuf.data_ptr());
    }
  }

  /*
  ** Bulk GEMM + COMM
  ** This function assumes the communication input is pre-copied to _ubuf
  */
  std::vector<at::Tensor>
  bulk_overlap(at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
               transformer_engine::DType A_type, bool transa, at::Tensor B,
               at::Tensor B_scale_inverse, int64_t B_fp8_tensor, transformer_engine::DType B_type,
               bool transb, at::Tensor D, at::Tensor D_scale, transformer_engine::DType D_type,
               at::Tensor D_amax, at::Tensor bias, transformer_engine::DType bias_type,
               at::Tensor pre_gelu_out, bool grad, at::Tensor workspace, size_t workspaceSize,
               bool accumulate, bool use_split_accumulator, int comm_type, at::Tensor rs_output) {
    auto empty = torch::Tensor();
    TensorWrapper A_ = torch_tensor_to_te(A, empty, empty, A_scale_inverse, A_fp8_tensor);
    TensorWrapper B_ = torch_tensor_to_te(B, empty, empty, B_scale_inverse, B_fp8_tensor);
    TensorWrapper bias_ = torch_tensor_to_te(bias);
    TensorWrapper D_ = torch_tensor_to_te(D, D_amax, D_scale, empty);
    TensorWrapper pre_gelu_out_ = torch_tensor_to_te(pre_gelu_out);
    TensorWrapper workspace_ = torch_tensor_to_te(workspace);
    TensorWrapper ubuf_ = torch_tensor_to_te(_ubuf, empty, empty, _ubuf_scale_inv);

    at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
    NVTE_Comm_Overlap_Type comm_type_ = static_cast<NVTE_Comm_Overlap_Type>(comm_type);
    if (comm_type_ == NVTE_Comm_Overlap_Type::ALL_GATHER) {
      CommGemmOverlap::bulk_gemm_overlap_ag(
        (cudaStream_t)stream_main, &A_, transa, &B_, transb, &bias_,
        &D_, &pre_gelu_out_, &ubuf_, &workspace_,
        grad, accumulate, use_split_accumulator);
    } else {
      TensorWrapper rs_out_ = torch_tensor_to_te(rs_output);
      CommGemmOverlap::bulk_gemm_overlap_rs(
        (cudaStream_t)stream_main, &A_, transa, &B_, transb, &bias_,
        &D_, &pre_gelu_out_, &ubuf_, &rs_out_, &workspace_,
        grad, accumulate, use_split_accumulator);
    }

    // Get the current userbuf offset
    char *ubuf_wt_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());
    if (comm_type_ == NVTE_Comm_Overlap_Type::REDUCE_SCATTER) {
      ubuf_wt_ptr += _ubuf.numel() / _tp_size * _tp_id * _ubuf.element_size();
    }

    // Generate output tensor from userbuf data pointer
    int output_c_dim0 = (comm_type_ == NVTE_Comm_Overlap_Type::ALL_GATHER) ? _ubuf.size(0)
                                                         : _ubuf.size(0) / _tp_size;
    int output_c_dim1 = _ubuf.size(1);
    torch::Tensor output_tensor = torch::from_blob(
      reinterpret_cast<void *>(ubuf_wt_ptr), {output_c_dim0, output_c_dim1}, _ubuf.options());

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
    auto empty = torch::Tensor();
    TensorWrapper A_ = torch_tensor_to_te(A, empty, empty, A_scale_inverse, A_fp8_tensor);
    TensorWrapper B_ = torch_tensor_to_te(B, empty, empty, B_scale_inverse, B_fp8_tensor);
    TensorWrapper bias_ = torch_tensor_to_te(bias);
    TensorWrapper D_ = torch_tensor_to_te(D, D_amax, D_scale, empty);
    TensorWrapper pre_gelu_out_ = torch_tensor_to_te(pre_gelu_out);
    TensorWrapper workspace_ = torch_tensor_to_te(workspace);
    TensorWrapper ubuf_ = torch_tensor_to_te(_ubuf, empty, empty, _ubuf_scale_inv);
    TensorWrapper rs_out_ = torch_tensor_to_te(rs_output);
    TensorWrapper counters_ = torch_tensor_to_te(_counters);

    at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
    CommGemmOverlap::atomic_gemm_overlap_rs(
      (cudaStream_t)stream_main, &A_, transa, &B_, transb, &bias_,
      &D_, &pre_gelu_out_, &ubuf_, &rs_out_, &counters_, &workspace_,
      grad, accumulate, use_split_accumulator);
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
    auto empty = torch::Tensor();
    TensorWrapper A_ = torch_tensor_to_te(A, empty, empty, A_scale_inverse, A_fp8_tensor);
    TensorWrapper B_ = torch_tensor_to_te(B, empty, empty, B_scale_inverse, B_fp8_tensor);
    TensorWrapper bias_ = torch_tensor_to_te(bias);
    TensorWrapper D_ = torch_tensor_to_te(D, D_amax, D_scale, empty);
    TensorWrapper pre_gelu_out_ = torch_tensor_to_te(pre_gelu_out);
    TensorWrapper workspace_ = torch_tensor_to_te(workspace);
    TensorWrapper ubuf_ = torch_tensor_to_te(_ubuf, empty, empty, _ubuf_scale_inv);
    TensorWrapper rs_out_ = torch_tensor_to_te(rs_output);

    at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
    CommGemmOverlap::split_gemm_overlap_rs(
      (cudaStream_t)stream_main, &A_, transa, &B_, transb, &bias_,
      &D_, &pre_gelu_out_, &ubuf_, &rs_out_, &workspace_,
      grad, accumulate, use_split_accumulator, gemm_overlap);

    return;
  }  // split_overlap_rs

  /*
  ** Helper function to copy input to _ubuf
  */
  void copy_input_to_ubuf(torch::Tensor input, int comm_type) {
    char *ubuf_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());
    NVTE_Comm_Overlap_Type comm_type_ = static_cast<NVTE_Comm_Overlap_Type>(comm_type);
    if (comm_type_ == NVTE_Comm_Overlap_Type::ALL_GATHER) {
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
    NVTE_CHECK_CUDA(cudaEventRecord(_start_d2dcopy, (cudaStream_t)stream_main));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_comm, _start_d2dcopy, 0));
    NVTE_CHECK_CUDA(
      cudaMemcpyAsync(ubuf_ptr, input.data_ptr(), input.numel() * input.element_size(),
                      cudaMemcpyDeviceToDevice, _stream_comm));
  }

  /*
  ** Helper function to export _ubuf output
  */
  torch::Tensor& get_ubuf_output(int comm_type) {
    char *ubuf_wt_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());
    NVTE_Comm_Overlap_Type comm_type_ = static_cast<NVTE_Comm_Overlap_Type>(comm_type);
    if ((comm_type_ != NVTE_Comm_Overlap_Type::ALL_GATHER) &&
        (comm_type_ != NVTE_Comm_Overlap_Type::REDUCE_SCATTER))
      NVTE_ERROR("Invalid comm_type");
    if (comm_type_ == NVTE_Comm_Overlap_Type::REDUCE_SCATTER)
      ubuf_wt_ptr += _ubuf.numel() / _tp_size * _tp_id * _ubuf.element_size();
    int output_c_dim0 = (comm_type_ == NVTE_Comm_Overlap_Type::ALL_GATHER) ? _ubuf.size(0)
                                                                : _ubuf.size(0) / _tp_size;
    int output_c_dim1 = _ubuf.size(1);
    torch::Tensor output_tensor = torch::from_blob(
      ubuf_wt_ptr, {output_c_dim0, output_c_dim1}, _ubuf.options());
    return output_tensor;
  }

  /*
  ** Helper function to set the inverse Fp8 scale for the _ubuf tensor
  */
  void set_ubuf_scale_inv(const torch::Tensor &scale_inv) {
    _ubuf_scale_inv = scale_inv;
    _ubuf_scale_inv_initialized = true;
  }

  bool is_fp8_ubuf() { return (_ubuf.element_size() == 1); }
};  // UbufCommOverlap

struct PYBIND11_EXPORT UbufP2PCommOverlap : torch::CustomClassHolder, CommGemmOverlapP2P {
  torch::Tensor _counters;
  torch::Tensor _ubuf_scale_inv;
  torch::Tensor _ubuf;
  std::vector<torch::Tensor> _ubufs;
  bool _ubuf_scale_inv_initialized;
  int _self_chunk_id;

  UbufP2PCommOverlap(
    torch::Tensor sample, int world_rank, int world_size, int tp_rank, int tp_size,
    int num_splits, int num_max_streams, int comm_cga_size, int num_comm_sm,
    bool set_sm_margin, bool atomic_gemm, bool aggregate, bool is_reduce_scatter)
  : CommGemmOverlapP2P(world_rank, world_size, tp_rank, tp_size, 0, 1,
                      num_splits, num_max_streams, comm_cga_size, num_comm_sm,
                      set_sm_margin, atomic_gemm, aggregate, is_reduce_scatter,
                      &ub_alloc_copy_allgather, &ub_bcast_int, &ub_barrier, &ub_free) {
    size_t ubuf_bytes = sample.numel() * sample.element_size();
    size_t ubuf_chunk_bytes = ubuf_bytes / tp_size;
    if (is_reduce_scatter)
      ubuf_bytes = static_cast<int>(ubuf_bytes / tp_size * (tp_size * 2 - 1));

    void *ubuf_ptr;
    bool alloc = true;
    if (transformer_engine::getenv<bool>("UB_SKIPMC")) {
      alloc = false;
      NVTE_CHECK_CUDA(cudaMalloc(&ubuf_ptr, ubuf_bytes));
    }
    this->register_gpu_buffer(&ubuf_ptr, ubuf_bytes, alloc);
    _ubuf = torch::from_blob(
      ubuf_ptr, {sample.size(0) / tp_size * _num_ubuf_chunks, sample.size(1)}, sample.options());

    // Create tensor chunks for easy management
    char *ubuf_byte_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());
    for (int i = 0; i < _num_ubuf_chunks; i++) {
      torch::Tensor ubuf_chunk = torch::from_blob(
          ubuf_byte_ptr, {sample.size(0) / tp_size, sample.size(1)}, sample.options());
      _ubufs.push_back(ubuf_chunk);
      ubuf_byte_ptr += ubuf_chunk_bytes;
    }

    _self_chunk_id = _tp_id;
    if (_atomic_gemm) {
      auto counter_options = at::device(torch::kCUDA).dtype(torch::kInt32);
      _counters = torch::zeros({tp_size * 2}, counter_options);
      _counters.index_put_({Slice(None, tp_size)}, 1);

      if (!is_reduce_scatter) {
        const char *env_p = std::getenv("NVTE_AG_P2P_MULTI_ATOMIC");
        if (world_rank == 0 && env_p != nullptr) {
          if (env_p[0] == '1' && world_rank == 0) {
            printf("!!! [UB][PyTorch] userbuffers_sendrecv_multi_atomic_shuffle\n");
          }
        }
        _self_chunk_id = 0;
        _counters.index_put_({_self_chunk_id}, 0);
      }
    }
  }

  ~UbufP2PCommOverlap() {
    if (transformer_engine::getenv<bool>("UB_SKIPMC")) {
      cudaFree(_ubuf.data_ptr());
    }
  }

  /*
  ** Split AllGather + AtomicGEMM using P2P communication
  ** This function assumes the input_b is pre-copied to _ubufs[rank_id]. This is needed to have AG\
  ** outputs in each rank to be in the contiguous memory space after all ring exchange phases.
  */
  torch::Tensor atomic_gemm_overlap_ag(
      at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
      transformer_engine::DType A_type, bool transa, at::Tensor B, at::Tensor B_scale_inverse,
      int64_t B_fp8_tensor, transformer_engine::DType B_type, bool transb, at::Tensor D,
      at::Tensor D_scale, transformer_engine::DType D_type, at::Tensor D_amax, at::Tensor bias,
      transformer_engine::DType bias_type, at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
      size_t workspaceSize, bool accumulate, bool use_split_accumulator, at::Tensor B_copy) {
    // Create an GEMM output buffer with N+1 chunks in a contiguous memory
    int m = (transa) ? A.size(0) : A.size(1);
    int n = _ubuf.size(0);
    int n_chunk = n / _tp_size;
    torch::Tensor D_buffer = torch::empty({n_chunk * (_tp_size + 1), m}, D.options());
    D = torch::from_blob(D_buffer.data_ptr(), {D.size(0), D.size(1)}, D.options());

    auto empty = torch::Tensor();
    TensorWrapper A_ = torch_tensor_to_te(A, empty, empty, A_scale_inverse, A_fp8_tensor);
    TensorWrapper B_ = torch_tensor_to_te(B, empty, empty, B_scale_inverse, B_fp8_tensor);
    TensorWrapper bias_ = torch_tensor_to_te(bias);
    TensorWrapper D_ = torch_tensor_to_te(D, D_amax, D_scale, empty);
    TensorWrapper pre_gelu_out_ = torch_tensor_to_te(pre_gelu_out);
    TensorWrapper workspace_ = torch_tensor_to_te(workspace);
    std::vector<TensorWrapper*> ubufs_(_num_ubuf_chunks);
    for (int i=0; i < _num_ubuf_chunks; i++)
      *(ubufs_[i]) = torch_tensor_to_te(_ubufs[i], empty, empty, _ubuf_scale_inv);
    TensorWrapper counters_ = torch_tensor_to_te(_counters);
    TensorWrapper B_copy_ = torch_tensor_to_te(B_copy);
    TensorWrapper D_buffer_ = torch_tensor_to_te(D_buffer);

    at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
    CommGemmOverlapP2P::atomic_gemm_overlap_ag(
      (cudaStream_t)stream_main, &A_, transa, &B_, transb, &bias_,
      &D_, &pre_gelu_out_, ubufs_, &counters_, &B_copy_, &D_buffer_, &workspace_,
      grad, accumulate, use_split_accumulator);

    // Return the last N rows of D_buffer
    torch::Tensor D_return = D_buffer.narrow(0, n_chunk, n);
    return D_return;
  }  // atomic_gemm_overlap_ag

  /*
  ** Split AllGather + GEMM using P2P communication
  ** This function assumes the input_b is pre-copied to _ubufs[rank_id]. This is needed to have AG
  ** outputs in each rank to be in the contiguous memory space after all ring exchange phases.
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
    auto empty = torch::Tensor();
    TensorWrapper A_ = torch_tensor_to_te(A, empty, empty, A_scale_inverse, A_fp8_tensor);
    TensorWrapper B_ = torch_tensor_to_te(B, empty, empty, B_scale_inverse, B_fp8_tensor);
    TensorWrapper bias_ = torch_tensor_to_te(bias);
    TensorWrapper D_ = torch_tensor_to_te(D, D_amax, D_scale, empty);
    TensorWrapper pre_gelu_out_ = torch_tensor_to_te(pre_gelu_out);
    TensorWrapper workspace_ = torch_tensor_to_te(workspace);
    std::vector<TensorWrapper*> ubufs_(_num_ubuf_chunks);
    for (int i=0; i < _num_ubuf_chunks; i++)
      *(ubufs_[i]) = torch_tensor_to_te(_ubufs[i], empty, empty, _ubuf_scale_inv);
    TensorWrapper B_copy_ = torch_tensor_to_te(B_copy);

    at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
    CommGemmOverlapP2P::split_gemm_overlap_ag(
      (cudaStream_t)stream_main, &A_, transa, &B_, transb, &bias_,
      &D_, &pre_gelu_out_, ubufs_, &B_copy_, &workspace_,
      grad, accumulate, use_split_accumulator);

    return D;
  }  // split_overlap_ag

  /*
  ** Split ReduceScatter + GEMM using P2P communication
  */
  void atomic_gemm_overlap_rs(at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
                        transformer_engine::DType A_type, bool transa, at::Tensor B,
                        at::Tensor B_scale_inverse, int64_t B_fp8_tensor,
                        transformer_engine::DType B_type, bool transb, at::Tensor D,
                        at::Tensor D_scale, transformer_engine::DType D_type, at::Tensor D_amax,
                        at::Tensor bias, transformer_engine::DType bias_type,
                        at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
                        size_t workspaceSize, bool accumulate, bool use_split_accumulator,
                        at::Tensor rs_output) {
    auto empty = torch::Tensor();
    TensorWrapper A_ = torch_tensor_to_te(A, empty, empty, A_scale_inverse, A_fp8_tensor);
    TensorWrapper B_ = torch_tensor_to_te(B, empty, empty, B_scale_inverse, B_fp8_tensor);
    TensorWrapper bias_ = torch_tensor_to_te(bias);
    TensorWrapper D_ = torch_tensor_to_te(D, D_amax, D_scale, empty);
    TensorWrapper pre_gelu_out_ = torch_tensor_to_te(pre_gelu_out);
    TensorWrapper workspace_ = torch_tensor_to_te(workspace);
    std::vector<TensorWrapper*> ubufs_(_num_ubuf_chunks);
    for (int i=0; i < _num_ubuf_chunks; i++)
      *(ubufs_[i]) = torch_tensor_to_te(_ubufs[i], empty, empty, _ubuf_scale_inv);
    TensorWrapper counters_ = torch_tensor_to_te(_counters);
    TensorWrapper rs_out_ = torch_tensor_to_te(rs_output);

    at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
    CommGemmOverlapP2P::atomic_gemm_overlap_rs(
      (cudaStream_t)stream_main, &A_, transa, &B_, transb, &bias_,
      &D_, &pre_gelu_out_, ubufs_, &counters_, &rs_out_, &workspace_,
      grad, accumulate, use_split_accumulator);
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
    auto empty = torch::Tensor();
    TensorWrapper A_ = torch_tensor_to_te(A, empty, empty, A_scale_inverse, A_fp8_tensor);
    TensorWrapper B_ = torch_tensor_to_te(B, empty, empty, B_scale_inverse, B_fp8_tensor);
    TensorWrapper bias_ = torch_tensor_to_te(bias);
    TensorWrapper D_ = torch_tensor_to_te(D, D_amax, D_scale, empty);
    TensorWrapper pre_gelu_out_ = torch_tensor_to_te(pre_gelu_out);
    TensorWrapper workspace_ = torch_tensor_to_te(workspace);
    std::vector<TensorWrapper*> ubufs_(_num_ubuf_chunks);
    for (int i=0; i < _num_ubuf_chunks; i++)
      *(ubufs_[i]) = torch_tensor_to_te(_ubufs[i], empty, empty, _ubuf_scale_inv);
    TensorWrapper rs_out_ = torch_tensor_to_te(rs_output);

    at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
    CommGemmOverlapP2P::split_gemm_overlap_rs(
      (cudaStream_t)stream_main, &A_, transa, &B_, transb, &bias_,
      &D_, &pre_gelu_out_, ubufs_, &rs_out_, &workspace_,
      grad, accumulate, use_split_accumulator);
  }


  /*
  ** Helper function to copy input to _ubuf or _ubufs chunks
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
  ** Helper function to export _ubuf output
  */
  torch::Tensor get_ubuf_output(int comm_type) {
    char *ubuf_wt_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());
    NVTE_Comm_Overlap_Type comm_type_ = static_cast<NVTE_Comm_Overlap_Type>(comm_type);
    if ((comm_type_ != NVTE_Comm_Overlap_Type::ALL_GATHER) &&
        (comm_type_ != NVTE_Comm_Overlap_Type::REDUCE_SCATTER))
      NVTE_ERROR("Invalid comm_type");
    if (comm_type_ == NVTE_Comm_Overlap_Type::REDUCE_SCATTER)
      ubuf_wt_ptr += _ubuf.numel() / _tp_size * _self_chunk_id * _ubuf.element_size();
    int output_c_dim0 = (comm_type_ == NVTE_Comm_Overlap_Type::ALL_GATHER) ? _ubuf.size(0)
                                                                : _ubuf.size(0) / _tp_size;
    int output_c_dim1 = _ubuf.size(1);
    return torch::from_blob(ubuf_wt_ptr, {output_c_dim0, output_c_dim1}, _ubuf.options());
  }

  /*
  ** Helper function to set the inverse Fp8 scale for the _ubuf tensor
  */
  void set_ubuf_scale_inv(const torch::Tensor &scale_inv) {
    _ubuf_scale_inv = scale_inv;
    _ubuf_scale_inv_initialized = true;
  }

  bool is_fp8_ubuf() { return (_ubuf.element_size() == 1); }
};  // UbufP2PCommOverlap

}  // namespace userbuffers

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_PYTORCH_COMM_GEMM_OVERLAP_H_
