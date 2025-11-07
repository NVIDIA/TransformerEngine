/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"
#include "transformer_engine/transformer_engine.h"

#define HALF_BYTES 2
#define UB_MAX_SM 32

using namespace torch::indexing;
using namespace std::placeholders;

namespace te = transformer_engine;

/***************************************************************************************************
 * CommOverlapHelper
 **************************************************************************************************/

CommOverlapHelper::CommOverlapHelper() {
#ifndef NVTE_UB_WITH_MPI
  NVTE_ERROR("Internal TE error: CommOverlapHelper() requires NVTE_UB_WITH_MPI=1!");
#endif
}  // empty constructor for NVTE_UB_WITH_MPI=1

CommOverlapHelper::CommOverlapHelper(c10d::ProcessGroup *tp_group) {
#ifndef NVTE_WITH_CUBLASMP
  NVTE_ERROR("Internal TE error: CommOverlapHelper(tp_group) requires NVTE_WITH_CUBLASMP=1!");
#endif
  c10d::ProcessGroup::BackendType backend = tp_group->getBackendType();
  backend_is_nccl = (backend == c10d::ProcessGroup::BackendType::NCCL);
  NVTE_CHECK(backend_is_nccl, "Comm+GEMM overlap with cuBlasMp requires bootstrapping with NCCL.");

  myrank = tp_group->getRank();
  numranks = tp_group->getSize();
  pgs.insert({"tp", tp_group});
  initialized = true;
}
.  // TP group constructor for NVTE_WITH_CUBLASMP=1

    CommOverlapHelper::CommOverlapHelper(c10d::ProcessGroup *world_group,
                                         c10d::ProcessGroup *intra_domain_group) {
#if defined(NVTE_UB_WITH_MPI)
  NVTE_ERROR("Internal TE error: CommOverlapHelper(world, intra_domain) is not supported with ",
             "NVTE_UB_WITH_MPI=1!");
#elif defined(NVTE_WITH_CUBLASMP)
  NVTE_ERROR("Internal TE error: CommOverlapHelper(world, intra_domain) is not supported with ",
             "NVTE_WITH_CUBLASMP=1!");
#endif
  pgs.insert({"world", world_group});
  myrank = pgs["world"]->getRank();
  numranks = pgs["world"]->getSize();
  c10d::ProcessGroup::BackendType backend = pgs["world"]->getBackendType();
  backend_is_nccl = (backend == c10d::ProcessGroup::BackendType::NCCL);

  // Get local rank on node and number of local ranks
  NVTE_CHECK(intra_domain_group.value()->getBackendType() == backend,
             "Internal TE error: Intra-node group must be on the same backend (%s) as the world ",
             "group!", pgs["world"]->getBackendName());
  pgs.insert({"intra", intra_domain_group.value()});
  mylocal = pgs["intra"]->getRank();
  numlocal = pgs["intra"]->getSize();

  if (numlocal == numranks) {
    // Intra-node group is same as the world group so there can only be 1 node
    NVTE_CHECK(
        mylocal == myrank,
        "Internal TE error: Local rank must be equal to global rank when intra-node group size ",
        "is equal to the world group size!");
    mynode = 0;
    numnodes = 1;
  } else {
    // Get node ID and number of nodes
    mynode = myrank / numlocal;
    numnodes = numranks / numlocal;
  }

  initialized = true;
#endif
}  // world + intra-node constructor for Userbuffers w/ PyTorch Distributed bootstrapping

CommOverlapHelper::~CommOverlapHelper() {
#ifndef NVTE_UB_WITH_MPI
  for (auto &pg : pgs) pg.second = nullptr;
  backend_is_nccl = false;
  initialized = false;
#endif
}

void CommOverlapHelper::ub_allgather(void *globaldata, size_t globalbytes, void *localdata,
                                     size_t localbytes, ExtComm group) {
#ifndef NVTE_UB_WITH_MPI
  NVTE_CHECK(initialized, "Internal TE error: tex.CommOverlapHelper() is not initialized ",
             "with valid process groups!");

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
#else
  NVTE_ERROR("Internal TE error: CommOverlapHelper::ub_allgather is a no-op when TE is compiled ",
             "with NVTE_UB_WITH_MPI=1!");
#endif
}

void CommOverlapHelper::ub_barrier(ExtComm group) {
#ifndef NVTE_UB_WITH_MPI
  NVTE_CHECK(initialized, "Internal TE error: tex.CommOverlapHelper() is not initialized ",
             "with valid process groups!");
  auto work = pgs[group]->barrier();
  work->wait();
#else
  NVTE_ERROR("Internal TE error: CommOverlapHelper::ub_barrier is a no-op when TE is compiled ",
             "with NVTE_UB_WITH_MPI=1!");
#endif
}

/***************************************************************************************************
 * CommOverlap
 **************************************************************************************************/

CommOverlapManager::CommOverlapManager(transformer_engine::CommOverlapMethod method,
                                       transformer_engine::CommOverlapType comm_type,
                                       const std::vector<size_t> &buffer_shape,
                                       at::ScalarType buffer_dtype, CommOverlapHelper *helper,
                                       int tp_size, int num_splits, int num_max_streams,
                                       int comm_cga_size, int gemm_priority, int comm_priority,
                                       int num_comm_sm, bool set_sm_margin, bool atomic_gemm,
                                       bool aggregate_ag, bool rs_overlap_first_gemm) {
#ifdef NVTE_WITH_CUBLASMP
  _ctx = nvte_comm_gemm_ctx_create(reinterpret_cast<ncclComm_t>(
      helper->get_comm_ptr("tp"), helper->numranks, helper->myrank, te::cuda::current_device()));
#else
  if (method == te::CommOverlapMethod::RING_EXCHANGE) {
    _ctx = reinterpret_cast<te::CommOverlapCore *>(new te::CommOverlapP2PBase(
        buffer_shape, te::pytorch::GetTransformerEngineDType(buffer_dtype), helper->myrank,
        helper->numranks, helper->mylocal, helper->numlocal, helper->mynode, helper->numnodes,
        tp_size, std::bind(&CommOverlapHelper::ub_allgather, helper, _1, _2, _3, _4, _5),
        std::bind(&CommOverlapHelper::ub_barrier, helper, _1), comm_type, num_max_streams,
        comm_cga_size, gemm_priority, comm_priority, num_comm_sm, set_sm_margin, use_ce,
        atomic_gemm, aggregate));
  } else {
    _ctx = reinterpret_cast<te::CommOverlapCore *>(new te::CommOverlapBase(
        buffer_shape, te::pytorch::GetTransformerEngineDType(buffer_dtype), helper->myrank,
        helper->numranks, helper->mylocal, helper->numlocal, helper->mynode, helper->numnodes,
        tp_size, std::bind(&CommOverlapHelper::ub_allgather, helper, _1, _2, _3, _4, _5),
        std::bind(&CommOverlapHelper::ub_barrier, helper, _1), num_splits, num_max_streams,
        comm_cga_size, gemm_priority, comm_priority, num_comm_sm, set_sm_margin, atomic_gemm,
        rs_overlap_first_gemm))
  }
#endif
}

/*
** Helper function to copy input to _ubuf
*/
void CommOverlapManager::copy_into_buffer(const at::Tensor &input, bool local_chunk) {
#ifndef NVTE_WITH_CUBLASMP
  const auto &input_ = input.contiguous();

  // Check element size
  const size_t element_size = input.element_size();
  NVTE_CHECK(_ubuf.element_size() == element_size,
             "Tried to copy data into a Userbuffers buffer but dtypes are not compatible ",
             "(input dtype has ", element_size, " bytes, UB dtype has ", _ubuf.element_size(),
             " bytes)");

  // Input data
  const size_t input_size = input_.numel();
  const void *src_ptr = input_.data_ptr();

  // Userbuffers data
  if (_method == te::CommOverlapMethod::RING_EXCHANGE) {
    void *dst_ptr;
    if (local_chunk) {
      NVTE_CHECK(_ubufs[_tp_id].numel() == input_size,
                 "Tried to copy an invalid tensor into a local chunk of a Userbuffers buffer ",
                 "(input_size=", input_size, ", local_ubuf_size=", _ubufs[_tp_id].numel(), ")");
      dst_ptr = _ubufs[_tp_id].dptr();
    } else {
      NVTE_CHECK(_ubuf.numel() == input_size,
                 "Tried to copy an invalid tensor into a Userbuffers buffer ",
                 "(input_size=", input_size, ", ubuf_size=", _ubuf.numel(), ")");
      dst_ptr = _ubuf.dptr();
    }

    // Copy data
    NVTE_CHECK_CUDA(cudaMemcpyAsync(dst_ptr, src_ptr, input_size * element_size,
                                    cudaMemcpyDeviceToDevice,
                                    (cudaStream_t)at::cuda::getCurrentCUDAStream()));
  } else {
    const size_t ubuf_size = _ubuf.numel();
    void *dst_ptr = _ubuf.dptr();
    if (local_chunk) {
      NVTE_CHECK(input_size * _tp_size == ubuf_size,
                 "Tried to copy an invalid tensor into a local chunk of a Userbuffers buffer ",
                 "(input_size=", input_size, ", tensor_parallel_size=", _tp_size,
                 ", ubuf_size=", ubuf_size, ")");
      dst_ptr =
          (reinterpret_cast<char *>(dst_ptr) + (ubuf_size / _tp_size) * _tp_id * element_size);
    } else {
      NVTE_CHECK(input_size == ubuf_size,
                 "Tried to copy an invalid tensor into a Userbuffers buffer ",
                 "(input_size=", input_size, ", ubuf_size=", ubuf_size, ")");
    }

    // Copy data
    auto stream_main = at::cuda::getCurrentCUDAStream();
    NVTE_CHECK_CUDA(cudaEventRecord(_start_d2dcopy, (cudaStream_t)stream_main));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_comm, _start_d2dcopy, 0));
    NVTE_CHECK_CUDA(cudaMemcpyAsync(dst_ptr, src_ptr, input_size * element_size,
                                    cudaMemcpyDeviceToDevice, (cudaStream_t)_stream_comm));
  }
#endif
}

at::Tensor CommOverlapManager::get_buffer(bool local_chunk,
                                          std::optional<std::vector<int64_t>> shape) {
#ifndef NVTE_WITH_CUBLASMP
  at::Tensor buffer_tensor;
  if (_method == te::CommOverlapMethod::RING_EXCHANGE) {
    // Check buffer shape
    if (shape) {
      const size_t requested_size = transformer_engine::pytorch::product(*shape);
      if (local_chunk) {
        NVTE_CHECK(requested_size == _ubufs[_tp_id].numel(),

                   "Invalid shape for local chunk of a Userbuffers buffer (requested shape=",
                   *shape, ", local_ubuf_size=", _ubufs[_tp_id].numel(), ")");
      } else {
        NVTE_CHECK(requested_size == _ubuf.numel(),
                   "Invalid shape for a Userbuffers buffer (requested shape=", *shape,
                   ", ubuf_size=", _ubuf.numel(), ")");
      }
    } else {
      int64_t dim0 = _ubuf.size(0);
      int64_t dim1 = _ubuf.size(1);
      if (local_chunk) {
        dim0 /= _tp_size;
      }
      shape = {dim0, dim1};
    }

    // Data pointer
    void *ubuf_ptr = local_chunk ? _ubufs[_tp_id].dptr() : _ubuf.dptr();

    // Construct PyTorch tensor
    const auto dtype = transformer_engine::pytorch::GetATenDType(_ubuf.dtype());
    buffer_tensor = torch::from_blob(ubuf_ptr, *shape, at::dtype(dtype).device(torch::kCUDA));
  } else {
    // Check buffer shape
    const size_t ubuf_size = _ubuf.numel();
    if (shape) {
      const size_t requested_size = transformer_engine::pytorch::product(*shape);
      if (local_chunk) {
        NVTE_CHECK(requested_size * _tp_size == ubuf_size,
                   "Invalid shape for local chunk of a Userbuffers buffer (requested shape=",
                   *shape, ", tensor_parallel_size=", _tp_size, ", ubuf_size=", ubuf_size, ")");
      } else {
        NVTE_CHECK(requested_size == ubuf_size,
                   "Invalid shape for a Userbuffers buffer (requested shape=", *shape,
                   ", ubuf_size=", ubuf_size, ")");
      }
    } else {
      int64_t dim0 = _ubuf.size(0);
      int64_t dim1 = _ubuf.size(1);
      if (local_chunk) {
        dim0 /= _tp_size;
      }
      shape = {dim0, dim1};
    }

    // Data pointer
    void *ubuf_ptr = _ubuf.dptr();
    if (local_chunk) {
      ubuf_ptr = (reinterpret_cast<char *>(ubuf_ptr) +
                  (ubuf_size / _tp_size) * _tp_id * _ubuf.element_size());
    }

    // Construct PyTorch tensor
    const auto dtype = transformer_engine::pytorch::GetATenDType(_ubuf.dtype());
    buffer_tensor = torch::from_blob(ubuf_ptr, *shape, at::dtype(dtype).device(torch::kCUDA));
  }

  return buffer_tensor;
#else
  // Return dummy tensor, will not be used with cuBlasMp
  const auto dtype = transformer_engine::pytorch::GetATenDType(DType::kByte);
  return torch::from_blob(nullptr, std::vector<int64_t>{0}, at::dtype(dtype).device(torch::kCUDA));
#endif
}

at::Stream CommOverlapManager::get_communication_stream() {
  return at::cuda::getStreamFromExternal(_stream_comm, at::cuda::current_device());
}

void CommOverlapManager::execute(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                                 bool transb, TensorWrapper &D, TensorWrapper &bias,
                                 TensorWrapper &pre_gelu_out, TensorWrapper &workspace, bool grad,
                                 bool accumulate, bool use_split_accumulator,
                                 te::CommOverlapType comm_type, TensorWrapper &aux_out,
                                 cudaStream_t stream) {
#ifdef NVTE_WITH_CUBLASMP
  if (_method == te::CommOverlapMethod::BULK) {
    NVTE_ERROR("Bulk overlap is not supported with cuBlasMp.");
  } else {
    cublasMpMatmulAlgoType_t algo = CUBLASMP_MATMUL_ALGO_TYPE_DEFAULT;
    if (_method == te::CommOverlapMethod::RING_EXCHANGE) {
      algo = (_use_atomic_gemm) ? CUBLASMP_MATMUL_ALGO_ATOMIC_P2P : CUBLASMP_MATMUL_ALGO_SPLIT_P2P;
    } else if (_method == te::CommOverlapMethod::PIPELINE) {
      algo = (_use_atomic_gemm) ? CUBLASMP_MATMUL_ALGO_ATOMIC_MULTICAST
                                : CUBLASMP_MATMUL_ALGO_SPLIT_MULTICAST;
    }

    // Tensor dimms in row-major order
    auto A_shape = A.shape();
    const int A0 = product(A_shape, 0, A_shape.ndim - 1);
    const int A1 = A_shape.data[A_shape.ndim - 1];
    auto B_shape = B.shape();
    const int B0 = product(B_shape, 0, B_shape.ndim - 1);
    const int B1 = B_shape.data[B_shape.ndim - 1];

    // GEMM dims in column-major order
    const int m = (transa) ? A0 : A1;
    const int n = (transb) ? B1 : B0;
    const int k = (transa) ? A1 : A0;

    if (comm_type == te::CommOverlapType::AG) {
      n *= _ctx->nranks;  // convert all-gathered dimension to global size
      NVTE_CHECK_CUBLASMP(nvte_all_gather_gemm(_ctx, m, n, k, A.data(), B.data(), D.data(),
                                               bias.data(), pre_gelu_out.data(), transa, transb,
                                               grad, accumulate, _num_comm_sm, stream, algo));
    } else {
      k *= _ctx->nranks;  // convert contracting dimension to global size
      NVTE_CHECK_CUBLASMP(nvte_gemm_reduce_scatter(_ctx, m, n, k, A.data(), B.data(), D.data(),
                                                   bias.data(), pre_gelu_out.data(), transa, transb,
                                                   grad, accumulate, _num_comm_sm, stream, algo));
    }
  }
#else
  if (_method == te::CommOverlapMethod::BULK) {
    _ctx->bulk_overlap(A.data(), transa, B.data(), transb, D.data(), bias.data(),
                       pre_gelu_out.data(), workspace.data(), grad, accumulate,
                       use_split_accumulator, comm_type, aux_out.data(), stream);
  } else if (comm_type == te::CommOverlapType::AG) {
    if (_use_atomic_gemm) {
      _ctx->atomic_gemm_overlap_ag(A.data(), transa, B.data(), transb, D.data(), bias.data(),
                                   pre_gelu_out.data(), workspace.data(), grad, accumulate,
                                   use_split_accumulator, aux_out.data(), stream);
    } else {
      _ctx->split_overlap_ag(A.data(), transa, B.data(), transb, D.data(), bias.data(),
                             pre_gelu_out.data(), workspace.data(), grad, accumulate,
                             use_split_accumulator, aux_out.data(), stream);
    }
  } else {
    if (_use_atomic_gemm) {
      _ctx->atomic_gemm_overlap_rs(A.data(), transa, B.data(), transb, D.data(), bias.data(),
                                   pre_gelu_out.data(), workspace.data(), grad, accumulate,
                                   use_split_accumulator, aux_out.data(), stream);
    } else {
      _ctx->split_overlap_rs(A.data(), transa, B.data(), transb, D.data(), bias.data(),
                             pre_gelu_out.data(), workspace.data(), grad, accumulate,
                             use_split_accumulator, aux_out.data(), stream);
    }
  }
#endif
}
