/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_COMM_GEMM_OVERLAP_H_
#define TRANSFORMER_ENGINE_PYTORCH_COMM_GEMM_OVERLAP_H_

#include <stdio.h>
#include <stdlib.h>

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

#define CHECK_CUDA(call)                                                                           \
  do {                                                                                             \
    cudaError_t status_ = call;                                                                    \
    if (status_ != cudaSuccess) {                                                                  \
      fprintf(stderr, "CUDA Error at line %d: %s\n", __LINE__, cudaGetErrorString(status_));       \
      exit(1);                                                                                     \
    }                                                                                              \
  } while (0)

using namespace torch::indexing;

namespace transformer_engine {

namespace userbuffers {

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

TensorWrapper torch_tensor_to_te(
  torch::Tensor inp, void *amax = nullptr, void *scale = nullptr, void *scale_inv = nullptr
) {
  return TensorWrapper(inp.data_ptr(), {inp.sizes().begin(), inp.sizes().end()},
                       torch_dtype_to_te(inp.scalar_type()),
                       reinterpret_cast<float *>(amax),
                       reinterpret_cast<float *>(scale),
                       reinterpret_cast<float *>(scale_inv));
}

struct UbufCommOverlap : torch::CustomClassHolder, CommGemmOverlap {
  torch::Tensor _counters;
  torch::Tensor _ubuf;
  torch::Tensor _ubuf_scale_inv;
  bool _ubuf_scale_inv_initialized{false};

  UbufCommOverlap(
    torch::Tensor sample, int world_rank, int world_size, int tp_rank, int tp_size,
    int num_splits, int num_max_streams, int comm_cga_size, int num_comm_sm,
    bool set_sm_margin, bool atomic_gemm)
  : CommGemmOverlap(world_rank, world_size, tp_rank, tp_size, 0, 1, num_splits, num_max_streams,
                 comm_cga_size, num_comm_sm, set_sm_margin, atomic_gemm) {
    // Allocate and register extra userbuffers
    void *ubuf_ptr;
    size_t ubuf_bytes = sample.numel() * sample.element_size();
    register_gpu_buffer(&ubuf_ptr, ubuf_bytes, true);
    _ubuf = torch::from_blob(ubuf_ptr, {sample.size(0), sample.size(1)}, sample.options());

    // Set the number of SMs for GEMM with margin
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    _math_sms = (set_sm_margin) ? prop.multiProcessorCount - num_comm_sm : prop.multiProcessorCount;
    _math_sms -= transformer_engine::getenv<int>("NVTE_EXT_MARGIN_SM", 0);

    if (_atomic_gemm) {
      auto counter_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
      _counters = torch::zeros({num_splits * 2}, counter_options);
      _counters.index_put_({Slice(None, num_splits)}, 1);
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
    if (A_scale_inverse.numel())
      A_scale_inverse = A_scale_inverse[A_fp8_tensor];

    if (B_scale_inverse.numel())
      B_scale_inverse = B_scale_inverse[B_fp8_tensor];

    TensorWrapper A_ = torch_tensor_to_te(A, nullptr, nullptr, A_scale_inverse.data_ptr());
    TensorWrapper B_ = torch_tensor_to_te(B, nullptr, nullptr, B_scale_inverse.data_ptr());
    TensorWrapper bias_ = torch_tensor_to_te(bias);
    TensorWrapper D_ = torch_tensor_to_te(D, D_amax.data_ptr(), D_scale.data_ptr(), nullptr);
    TensorWrapper pre_gelu_out_ = torch_tensor_to_te(pre_gelu_out);
    TensorWrapper workspace_ = torch_tensor_to_te(workspace);
    TensorWrapper ubuf_ = torch_tensor_to_te(_ubuf, nullptr, _ubuf_scale_inv.data_ptr());

    at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
    CommGemmOverlapType comm_type_ = static_cast<CommGemmOverlapType>(comm_type);
    if (comm_type_ == CommGemmOverlapType::AG) {
      CommGemmOverlap::bulk_gemm_overlap_ag(
      (cudaStream_t)stream_main, &A_, transa, &B_, transb, &bias_,
      &D_, &pre_gelu_out_, &workspace_, &ubuf_,
      grad, accumulate, use_split_accumulator);
    } else {
      TensorWrapper rs_out_ = torch_tensor_to_te(rs_output);
      CommGemmOverlap::bulk_gemm_overlap_rs(
      (cudaStream_t)stream_main, &A_, transa, &B_, transb, &bias_,
      &D_, &pre_gelu_out_, &workspace_, &ubuf_, &rs_out_,
      grad, accumulate, use_split_accumulator);
    }

    // Get the current userbuf offset
    char *ubuf_wt_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());
    if (comm_type_ == CommGemmOverlapType::RS) {
      ubuf_wt_ptr += _ubuf.numel() / _tp_size * _tp_id * _ubuf.element_size();
    }

    // Generate output tensor from userbuf data pointer
    int output_c_dim0 = (comm_type_ == CommGemmOverlapType::AG) ? _ubuf.size(0)
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
    if (A_scale_inverse.numel())
      A_scale_inverse = A_scale_inverse[A_fp8_tensor];

    if (B_scale_inverse.numel())
      B_scale_inverse = B_scale_inverse[B_fp8_tensor];

    TensorWrapper A_ = torch_tensor_to_te(A, nullptr, nullptr, A_scale_inverse.data_ptr());
    TensorWrapper B_ = torch_tensor_to_te(B, nullptr, nullptr, B_scale_inverse.data_ptr());
    TensorWrapper bias_ = torch_tensor_to_te(bias);
    TensorWrapper D_ = torch_tensor_to_te(D, D_amax.data_ptr(), D_scale.data_ptr(), nullptr);
    TensorWrapper pre_gelu_out_ = torch_tensor_to_te(pre_gelu_out);
    TensorWrapper workspace_ = torch_tensor_to_te(workspace);
    TensorWrapper ubuf_ = torch_tensor_to_te(_ubuf, nullptr, _ubuf_scale_inv.data_ptr());
    TensorWrapper rs_out_ = torch_tensor_to_te(rs_output);
    TensorWrapper counters_ = torch_tensor_to_te(_counters);

    at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
    CommGemmOverlap::atomic_gemm_overlap_rs(
      (cudaStream_t)stream_main, &A_, transa, &B_, transb, &bias_,
      &D_, &pre_gelu_out_, &workspace_, &ubuf_, &rs_out_, &counters_,
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
    if (A_scale_inverse.numel())
      A_scale_inverse = A_scale_inverse[A_fp8_tensor];

    if (B_scale_inverse.numel())
      B_scale_inverse = B_scale_inverse[B_fp8_tensor];

    TensorWrapper A_ = torch_tensor_to_te(A, nullptr, nullptr, A_scale_inverse.data_ptr());
    TensorWrapper B_ = torch_tensor_to_te(B, nullptr, nullptr, B_scale_inverse.data_ptr());
    TensorWrapper bias_ = torch_tensor_to_te(bias);
    TensorWrapper D_ = torch_tensor_to_te(D, D_amax.data_ptr(), D_scale.data_ptr(), nullptr);
    TensorWrapper pre_gelu_out_ = torch_tensor_to_te(pre_gelu_out);
    TensorWrapper workspace_ = torch_tensor_to_te(workspace);
    TensorWrapper ubuf_ = torch_tensor_to_te(_ubuf, nullptr, _ubuf_scale_inv.data_ptr());
    TensorWrapper rs_out_ = torch_tensor_to_te(rs_output);

    at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
    CommGemmOverlap::split_gemm_overlap_rs(
      (cudaStream_t)stream_main, &A_, transa, &B_, transb, &bias_,
      &D_, &pre_gelu_out_, &workspace_, &ubuf_, &rs_out_,
      grad, accumulate, use_split_accumulator, gemm_overlap);

    return;
  }  // split_overlap_rs

  /*
  ** Helper function to copy input to _ubuf
  */
  void copy_input_to_ubuf(torch::Tensor input, int comm_type) {
    char *ubuf_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());
    CommGemmOverlapType comm_type_ = static_cast<CommGemmOverlapType>(comm_type);
    if (comm_type_ == CommGemmOverlapType::AG) {
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

  torch::Tensor& get_ubuf_output(int comm_type) {
    char *ubuf_wt_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());
    CommGemmOverlapType comm_type_ = static_cast<CommGemmOverlapType>(comm_type);
    if (comm_type_ != CommGemmOverlapType::AG && comm_type_ != CommGemmOverlapType::RS)
      NVTE_ERROR("Invalid comm_type");
    if (comm_type_ == CommGemmOverlapType::RS)
      ubuf_wt_ptr += _ubuf.numel() / _tp_size * _tp_id * _ubuf.element_size();
    int output_c_dim0 = (comm_type_ == CommGemmOverlapType::AG) ? _ubuf.size(0)
                                                                : _ubuf.size(0) / _tp_size;
    int output_c_dim1 = _ubuf.size(1);
    torch::Tensor output_tensor = torch::from_blob(
      ubuf_wt_ptr, {output_c_dim0, output_c_dim1}, _ubuf.options());
    return output_tensor;
  }

  void set_ubuf_scale_inv(const torch::Tensor &scale_inv) {
    _ubuf_scale_inv = scale_inv;
    _ubuf_scale_inv_initialized = true;
  }

  bool is_fp8_ubuf() { return (_ubuf.element_size() == 1); }
};  // UbufCommOverlap

struct UbufP2PCommOverlap : torch::CustomClassHolder, CommGemmOverlapP2P {
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
                    set_sm_margin, atomic_gemm, aggregate, is_reduce_scatter) {
    size_t ubuf_bytes = sample.numel() * sample.element_size();
    size_t ubuf_chunk_bytes = ubuf_bytes / tp_size;
    if (is_reduce_scatter)
      ubuf_bytes = static_cast<size_t>(ubuf_chunk_bytes * _num_ubuf_chunks);

    void *ubuf_ptr;
    register_gpu_buffer(&ubuf_ptr, ubuf_bytes, true);
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
      auto counter_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
      _counters = torch::zeros({tp_size * 2}, counter_options);
      _counters.index_put_({Slice(None, tp_size)}, 1);

      if (!is_reduce_scatter) {
        const char *env_p = std::getenv("NVTE_AG_P2P_MULTI_ATOMIC");
        if (world_rank == 0 && env_p != nullptr) {
          if (env_p[0] == '1') {
            printf("!!userbuffers_sendrecv_multi_atomic_shuffle\n");
          }
        }
        _self_chunk_id = 0;
        _counters.index_put_({_self_chunk_id}, 0);
      }
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
    if (A_scale_inverse.numel())
      A_scale_inverse = A_scale_inverse[A_fp8_tensor];

    if (B_scale_inverse.numel())
      B_scale_inverse = B_scale_inverse[B_fp8_tensor];

    // Create an GEMM output buffer with N+1 chunks in a contiguous memory
    int m = (transa) ? A.size(0) : A.size(1);
    int n = _ubuf.size(0);
    int n_chunk = n / _tp_size;
    torch::Tensor D_buffer = torch::empty({n_chunk * (_tp_size + 1), m}, D.options());
    D = torch::from_blob(D_buffer.data_ptr(), {D.size(0), D.size(1)}, D.options());

    TensorWrapper A_ = torch_tensor_to_te(A, nullptr, nullptr, A_scale_inverse.data_ptr());
    TensorWrapper B_ = torch_tensor_to_te(B, nullptr, nullptr, B_scale_inverse.data_ptr());
    TensorWrapper bias_ = torch_tensor_to_te(bias);
    TensorWrapper D_ = torch_tensor_to_te(D, D_amax.data_ptr(), D_scale.data_ptr(), nullptr);
    TensorWrapper pre_gelu_out_ = torch_tensor_to_te(pre_gelu_out);
    TensorWrapper workspace_ = torch_tensor_to_te(workspace);
    TensorWrapper ubuf_ = torch_tensor_to_te(_ubuf, nullptr, _ubuf_scale_inv.data_ptr());
    TensorWrapper counters_ = torch_tensor_to_te(_counters);
    TensorWrapper B_copy_ = torch_tensor_to_te(B_copy);
    TensorWrapper D_buffer_ = torch_tensor_to_te(D_buffer);

    at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
    CommGemmOverlapP2P::atomic_gemm_overlap_ag(
      (cudaStream_t)stream_main, &A_, transa, &B_, transb, &bias_,
      &D_, &pre_gelu_out_, &workspace_, &ubuf_, &counters_, &B_copy_, &D_buffer_,
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
    if (A_scale_inverse.numel())
      A_scale_inverse = A_scale_inverse[A_fp8_tensor];

    if (B_scale_inverse.numel())
      B_scale_inverse = B_scale_inverse[B_fp8_tensor];

    TensorWrapper A_ = torch_tensor_to_te(A, nullptr, nullptr, A_scale_inverse.data_ptr());
    TensorWrapper B_ = torch_tensor_to_te(B, nullptr, nullptr, B_scale_inverse.data_ptr());
    TensorWrapper bias_ = torch_tensor_to_te(bias);
    TensorWrapper D_ = torch_tensor_to_te(D, D_amax.data_ptr(), D_scale.data_ptr(), nullptr);
    TensorWrapper pre_gelu_out_ = torch_tensor_to_te(pre_gelu_out);
    TensorWrapper workspace_ = torch_tensor_to_te(workspace);
    TensorWrapper ubuf_ = torch_tensor_to_te(_ubuf, nullptr, _ubuf_scale_inv.data_ptr());
    TensorWrapper B_copy_ = torch_tensor_to_te(B_copy);

    at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
    CommGemmOverlapP2P::split_gemm_overlap_ag(
      (cudaStream_t)stream_main, &A_, transa, &B_, transb, &bias_,
      &D_, &pre_gelu_out_, &workspace_, &ubuf_, &B_copy_,
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
    if (A_scale_inverse.numel())
      A_scale_inverse = A_scale_inverse[A_fp8_tensor];

    if (B_scale_inverse.numel())
      B_scale_inverse = B_scale_inverse[B_fp8_tensor];

    TensorWrapper A_ = torch_tensor_to_te(A, nullptr, nullptr, A_scale_inverse.data_ptr());
    TensorWrapper B_ = torch_tensor_to_te(B, nullptr, nullptr, B_scale_inverse.data_ptr());
    TensorWrapper bias_ = torch_tensor_to_te(bias);
    TensorWrapper D_ = torch_tensor_to_te(D, D_amax.data_ptr(), D_scale.data_ptr(), nullptr);
    TensorWrapper pre_gelu_out_ = torch_tensor_to_te(pre_gelu_out);
    TensorWrapper workspace_ = torch_tensor_to_te(workspace);
    TensorWrapper ubuf_ = torch_tensor_to_te(_ubuf, nullptr, _ubuf_scale_inv.data_ptr());
    TensorWrapper counters_ = torch_tensor_to_te(_counters);
    TensorWrapper rs_out_ = torch_tensor_to_te(rs_output);

    at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
    CommGemmOverlapP2P::atomic_gemm_overlap_rs(
      (cudaStream_t)stream_main, &A_, transa, &B_, transb, &bias_,
      &D_, &pre_gelu_out_, &workspace_, &ubuf_, &counters_, &rs_out_,
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
    if (A_scale_inverse.numel())
      A_scale_inverse = A_scale_inverse[A_fp8_tensor];

    if (B_scale_inverse.numel())
      B_scale_inverse = B_scale_inverse[B_fp8_tensor];

    TensorWrapper A_ = torch_tensor_to_te(A, nullptr, nullptr, A_scale_inverse.data_ptr());
    TensorWrapper B_ = torch_tensor_to_te(B, nullptr, nullptr, B_scale_inverse.data_ptr());
    TensorWrapper bias_ = torch_tensor_to_te(bias);
    TensorWrapper D_ = torch_tensor_to_te(D, D_amax.data_ptr(), D_scale.data_ptr(), nullptr);
    TensorWrapper pre_gelu_out_ = torch_tensor_to_te(pre_gelu_out);
    TensorWrapper workspace_ = torch_tensor_to_te(workspace);
    TensorWrapper ubuf_ = torch_tensor_to_te(_ubuf, nullptr, _ubuf_scale_inv.data_ptr());
    TensorWrapper rs_out_ = torch_tensor_to_te(rs_output);

    at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
    CommGemmOverlapP2P::split_gemm_overlap_rs(
      (cudaStream_t)stream_main, &A_, transa, &B_, transb, &bias_,
      &D_, &pre_gelu_out_, &workspace_, &ubuf_, &rs_out_,
      grad, accumulate, use_split_accumulator);
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
    CommGemmOverlapType _comm_type = static_cast<CommGemmOverlapType>(comm_type);
    if (_comm_type != CommGemmOverlapType::AG && _comm_type != CommGemmOverlapType::RS)
      NVTE_ERROR("Invalid comm_type");
    if (_comm_type == CommGemmOverlapType::RS)
      ubuf_wt_ptr += _ubuf.numel() / _tp_size * _self_chunk_id * _ubuf.element_size();
    int output_c_dim0 = (_comm_type == CommGemmOverlapType::AG) ? _ubuf.size(0)
                                                                : _ubuf.size(0) / _tp_size;
    int output_c_dim1 = _ubuf.size(1);
    return torch::from_blob(ubuf_wt_ptr, {output_c_dim0, output_c_dim1}, _ubuf.options());
  }

  void set_ubuf_scale_inv(const torch::Tensor &scale_inv) {
    _ubuf_scale_inv = scale_inv;
    _ubuf_scale_inv_initialized = true;
  }

  bool is_fp8_ubuf() { return (_ubuf.element_size() == 1); }
};  // UbufP2PCommOverlap

}  // namespace userbuffers

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_PYTORCH_COMM_GEMM_OVERLAP_H_
