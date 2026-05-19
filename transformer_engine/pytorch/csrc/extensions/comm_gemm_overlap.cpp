/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#ifdef NVTE_WITH_CUBLASMP
#include <nccl.h>
#endif

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
  NVTE_ERROR("Internal TE error: Dummy CommOverlapHelper init without NVTE_UB_WITH_MPI=1!");
#endif
}  // empty constructor for NVTE_UB_WITH_MPI=1

CommOverlapHelper::CommOverlapHelper(c10d::ProcessGroup *world_group,
                                     std::optional<c10d::ProcessGroup *> intra_domain_group) {
#ifndef NVTE_UB_WITH_MPI
  torch_pgs.insert({"world", world_group});
  myrank = torch_pgs["world"]->getRank();
  numranks = torch_pgs["world"]->getSize();
  c10d::ProcessGroup::BackendType backend = torch_pgs["world"]->getBackendType();
  backend_is_nccl = (backend == c10d::ProcessGroup::BackendType::NCCL);

  if (intra_domain_group.has_value()) {
    // Get local rank on node and number of local ranks
    NVTE_CHECK(intra_domain_group.value()->getBackendType() == backend,
               "Internal TE error: Intra-node group must be on the same backend (%s) as the world ",
               "group!", torch_pgs["world"]->getBackendName());
    torch_pgs.insert({"intra", intra_domain_group.value()});
    mylocal = torch_pgs["intra"]->getRank();
    numlocal = torch_pgs["intra"]->getSize();

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
  } else {
    // Intra-node group is not set so we assume there is only 1 node
    mylocal = myrank;
    numlocal = numranks;
    torch_pgs.insert({"intra", world_group});

    mynode = 0;
    numnodes = 1;
  }

  initialized = true;

#ifdef NVTE_WITH_CUBLASMP
  // Initialize world NCCL communicator via ncclCommInitRank (one GPU per process under torchrun)
  ncclUniqueId nccl_world_id;
  if (myrank == 0) {
    NVTE_CHECK_NCCL(ncclGetUniqueId(&nccl_world_id));
  }
  auto nccl_world_id_tensor =
      torch::from_blob(reinterpret_cast<uint8_t *>(&nccl_world_id), {sizeof(ncclUniqueId)},
                       at::device(torch::kCPU).dtype(torch::kUInt8));
  nccl_world_id_tensor = (backend_is_nccl) ? nccl_world_id_tensor.cuda() : nccl_world_id_tensor;
  {
    c10d::BroadcastOptions bcast_opts;
    bcast_opts.rootRank = 0;
    std::vector<at::Tensor> bcast_tensors = {nccl_world_id_tensor};
    auto work = torch_pgs["world"]->broadcast(bcast_tensors, bcast_opts);
    work->wait();
  }
  nccl_world_id_tensor = (backend_is_nccl) ? nccl_world_id_tensor.cpu() : nccl_world_id_tensor;
  nccl_world_id = *reinterpret_cast<ncclUniqueId *>(nccl_world_id_tensor.data_ptr());

  ncclComm_t nccl_world;
  NVTE_CHECK_NCCL(ncclCommInitRank(&nccl_world, numranks, nccl_world_id, myrank));
  nccl_comms.insert({"world", NcclCommSharedPtr(nccl_world, ncclCommDestroy)});

  if (intra_domain_group.has_value()) {
    // Generate a separate unique ID for the intra-node communicator
    ncclUniqueId nccl_intra_id;
    if (mylocal == 0) {
      NVTE_CHECK_NCCL(ncclGetUniqueId(&nccl_intra_id));
    }

    // Broadcast the intra-node unique ID from the local root to all local ranks
    auto nccl_intra_id_tensor =
        torch::from_blob(reinterpret_cast<uint8_t *>(&nccl_intra_id), {sizeof(ncclUniqueId)},
                         at::device(torch::kCPU).dtype(torch::kUInt8));
    nccl_intra_id_tensor = (backend_is_nccl) ? nccl_intra_id_tensor.cuda() : nccl_intra_id_tensor;
    {
      c10d::BroadcastOptions bcast_opts;
      bcast_opts.rootRank = 0;
      std::vector<at::Tensor> bcast_tensors = {nccl_intra_id_tensor};
      auto work = torch_pgs["intra"]->broadcast(bcast_tensors, bcast_opts);
      work->wait();
    }
    nccl_intra_id_tensor = (backend_is_nccl) ? nccl_intra_id_tensor.cpu() : nccl_intra_id_tensor;
    nccl_intra_id = *reinterpret_cast<ncclUniqueId *>(nccl_intra_id_tensor.data_ptr());

    // Initialize intra-node communicator
    ncclComm_t nccl_intra;
    NVTE_CHECK_NCCL(ncclCommInitRank(&nccl_intra, numlocal, nccl_intra_id, mylocal));
    nccl_comms.insert({"intra", NcclCommSharedPtr(nccl_intra, ncclCommDestroy)});
  }
#endif
#else
  NVTE_ERROR("Internal TE error: CommOverlapHelper cannot be initialized with valid PyTorch ",
             "distributed process groups when TE is compiled with NVTE_UB_WITH_MPI=1!");
#endif
}

CommOverlapHelper::~CommOverlapHelper() {
#ifndef NVTE_UB_WITH_MPI
  for (auto &pg : torch_pgs) {
    pg.second = nullptr;
  }
  torch_pgs.clear();
  backend_is_nccl = false;
  initialized = false;
#ifdef NVTE_WITH_CUBLASMP
  // Releasing the helper's references is enough: each shared_ptr's deleter
  // calls ncclCommDestroy once the last owner (helper or any consuming
  // CommOverlap/CommOverlapP2P) drops it.
  nccl_comms.clear();
#endif
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

  std::vector<std::vector<torch::Tensor>> globalchunks = {
      globaltmp.chunk(torch_pgs[group]->getSize())};
  std::vector<torch::Tensor> localchunk = {localtmp};
  auto work = torch_pgs[group]->allgather(globalchunks, localchunk);
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
  auto work = torch_pgs[group]->barrier();
  work->wait();
#else
  NVTE_ERROR("Internal TE error: CommOverlapHelper::ub_barrier is a no-op when TE is compiled ",
             "with NVTE_UB_WITH_MPI=1!");
#endif
}

CommOverlapHelper::NcclCommSharedPtr CommOverlapHelper::get_nccl_comm(std::string comm_name) {
#ifdef NVTE_WITH_CUBLASMP
  NVTE_CHECK(initialized, "Internal TE error: tex.CommOverlapHelper() is not initialized ",
             "with valid process groups!");
  NVTE_CHECK(backend_is_nccl,
             "Internal TE error: tex.CommOverlapHelper() was not initialized with an NCCL backend, "
             "so no NCCL communicators are available!");
  auto it = nccl_comms.find(comm_name);
  if (it != nccl_comms.end()) {
    return it->second;
  } else {
    NVTE_ERROR("Internal TE error: No NCCL communicator found with name ", comm_name, "!");
  }
#else
  NVTE_ERROR(
      "Internal TE error: CommOverlapHelper::get_nccl_comm() is an internal API that requires TE "
      "to be built with NVTE_WITH_CUBLASMP=1!");
#endif
}

/***************************************************************************************************
 * CommOverlap
 **************************************************************************************************/

CommOverlap::CommOverlap(const std::vector<size_t> &buffer_shape, at::ScalarType buffer_dtype,
                         CommOverlapHelper *helper, int tp_size, int num_splits,
                         int num_max_streams, int comm_cga_size, int gemm_priority,
                         int comm_priority, int num_comm_sm, bool set_sm_margin, bool atomic_gemm,
                         bool rs_overlap_first_gemm)
    : te::CommOverlapBase(buffer_shape, te::pytorch::GetTransformerEngineDType(buffer_dtype),
                          helper->myrank, helper->numranks, helper->mylocal, helper->numlocal,
                          helper->mynode, helper->numnodes, tp_size,
                          std::bind(&CommOverlapHelper::ub_allgather, helper, _1, _2, _3, _4, _5),
                          std::bind(&CommOverlapHelper::ub_barrier, helper, _1), num_splits,
                          num_max_streams, comm_cga_size, gemm_priority, comm_priority, num_comm_sm,
                          set_sm_margin, atomic_gemm, rs_overlap_first_gemm) {}

namespace {

// Run a dummy cuBLASMp matmul during construction so its lazy NCCL window
// registration and workspace allocation happen outside any CUDA-graph
// capture. The warmup is sized from the comm buffer so the cached
// workspace covers any matmul the caller will later run with the same
// descriptor. BF16 is used unconditionally; its workspace is at least as
// large as the FP8 workspace for the same m/n/k.
void cublasmp_capture_warmup(te::CommOverlapCore *core, int tp_size, te::CommOverlapType comm_type,
                             const std::vector<size_t> &buffer_shape, void *warmup_workspace) {
  NVTE_CHECK(buffer_shape.size() == 2, "cuBLASMp warmup expects a 2-D buffer shape, got rank ",
             buffer_shape.size());
  // Treat the matmul as square in the weight dim so workspace is sized
  // for the wider of the two cases.
  const int64_t N_global = static_cast<int64_t>(buffer_shape[0]);
  const int64_t hidden = static_cast<int64_t>(buffer_shape[1]);
  auto ceil_div = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
  const int64_t M_local = ceil_div(hidden, tp_size);
  const int64_t N_local = ceil_div(N_global, tp_size);
  const int64_t K_local = ceil_div(hidden, tp_size);
  const int64_t bf16_bytes = 2;

  std::vector<size_t> a_shape, b_shape, d_shape;
  if (comm_type == te::CommOverlapType::AG) {
    // A = (M_local, K), B = (N_local, K), D = (N_global, M_local)
    a_shape = {static_cast<size_t>(M_local), static_cast<size_t>(hidden)};
    b_shape = {static_cast<size_t>(N_local), static_cast<size_t>(hidden)};
    d_shape = {static_cast<size_t>(N_global), static_cast<size_t>(M_local)};
  } else {  // RS (or AR -- same descriptor-level dims)
    // A = (M_global, K_local), B = (N_global, K_local), D = (N_local, M_global)
    a_shape = {static_cast<size_t>(hidden), static_cast<size_t>(K_local)};
    b_shape = {static_cast<size_t>(N_global), static_cast<size_t>(K_local)};
    d_shape = {static_cast<size_t>(N_local), static_cast<size_t>(hidden)};
  }

  const size_t a_bytes = a_shape[0] * a_shape[1] * bf16_bytes;
  const size_t b_bytes = b_shape[0] * b_shape[1] * bf16_bytes;
  const size_t d_bytes = d_shape[0] * d_shape[1] * bf16_bytes;

  NVTE_CHECK_CUDA(cudaMalloc(&warmup_workspace, a_bytes + b_bytes + d_bytes));
  void *a_ptr = warmup_workspace;
  void *b_ptr = (reinterpret_cast<char *>(warmup_workspace) + a_bytes);
  void *d_ptr = (reinterpret_cast<char *>(warmup_workspace) + a_bytes + b_bytes);
  NVTE_CHECK_CUDA(cudaMemset(a_ptr, 0, a_bytes));
  NVTE_CHECK_CUDA(cudaMemset(b_ptr, 0, b_bytes));
  NVTE_CHECK_CUDA(cudaMemset(d_ptr, 0, d_bytes));

  te::TensorWrapper A_tw, B_tw, D_tw, bias_tw, pre_gelu_tw;
  A_tw.set_rowwise_data(a_ptr, te::DType::kBFloat16, a_shape);
  B_tw.set_rowwise_data(b_ptr, te::DType::kBFloat16, b_shape);
  D_tw.set_rowwise_data(d_ptr, te::DType::kBFloat16, d_shape);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (comm_type == te::CommOverlapType::AG) {
    core->cublasmp_ag_gemm(A_tw, /*transa=*/true, B_tw, /*transb=*/false, D_tw, bias_tw,
                           pre_gelu_tw, /*grad=*/false, /*accumulate=*/false, stream);
  } else {
    core->cublasmp_gemm_rs(A_tw, /*transa=*/true, B_tw, /*transb=*/false, D_tw, bias_tw,
                           pre_gelu_tw, /*grad=*/false, /*accumulate=*/false, stream);
  }
  NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));
  cudaFree(warmup_workspace);
}

}  // namespace

CommOverlap::CommOverlap(CommOverlapHelper *helper, int tp_rank, int tp_size,
                         te::CommOverlapType comm_type, const std::vector<size_t> &buffer_shape,
                         at::ScalarType buffer_dtype, int num_comm_sm, bool atomic_gemm)
    : te::CommOverlapBase(helper->get_nccl_comm("intra").get(), tp_rank, tp_size, num_comm_sm,
                          atomic_gemm),
      _nccl_comm(helper->get_nccl_comm("intra")) {
  // buffer_dtype is unused on this path (the warmup runs in BF16); kept in
  // the signature for API symmetry with the non-cuBLASMp ctor.
  (void)buffer_dtype;
  cublasmp_capture_warmup(this, tp_size, comm_type, buffer_shape, _warmup_workspace);
}

/*
** Helper function to copy input to _ubuf
*/
void CommOverlap::copy_into_buffer(const at::Tensor &input, bool local_chunk) {
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
  const size_t ubuf_size = _ubuf.numel();
  void *dst_ptr = _ubuf.dptr();
  if (local_chunk) {
    NVTE_CHECK(input_size * _tp_size == ubuf_size,
               "Tried to copy an invalid tensor into a local chunk of a Userbuffers buffer ",
               "(input_size=", input_size, ", tensor_parallel_size=", _tp_size,
               ", ubuf_size=", ubuf_size, ")");
    dst_ptr = (reinterpret_cast<char *>(dst_ptr) + (ubuf_size / _tp_size) * _tp_id * element_size);
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

at::Tensor CommOverlap::get_buffer(bool local_chunk, std::optional<std::vector<int64_t>> shape) {
  // Check buffer shape
  const size_t ubuf_size = _ubuf.numel();
  if (shape) {
    const size_t requested_size = transformer_engine::pytorch::product(*shape);
    if (local_chunk) {
      NVTE_CHECK(requested_size * _tp_size == ubuf_size,
                 "Invalid shape for local chunk of a Userbuffers buffer (requested shape=", *shape,
                 ", tensor_parallel_size=", _tp_size, ", ubuf_size=", ubuf_size, ")");
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
  return torch::from_blob(ubuf_ptr, *shape, at::dtype(dtype).device(torch::kCUDA));
}

std::pair<at::Stream, at::Stream> CommOverlap::get_communication_stream() {
  // Return the same stream for both send and recv
  return {at::cuda::getStreamFromExternal(_stream_comm, at::cuda::current_device()),
          at::cuda::getStreamFromExternal(_stream_comm, at::cuda::current_device())};
}

/***************************************************************************************************
 * CommOverlapP2P
 **************************************************************************************************/

CommOverlapP2P::CommOverlapP2P(const std::vector<size_t> &buffer_shape, at::ScalarType buffer_dtype,
                               CommOverlapHelper *helper, int tp_size,
                               te::CommOverlapType comm_type, int num_max_streams,
                               int comm_cga_size, int gemm_priority, int comm_priority,
                               int num_comm_sm, bool set_sm_margin, bool atomic_gemm, bool use_ce,
                               bool aggregate)
    : te::CommOverlapP2PBase(
          buffer_shape, te::pytorch::GetTransformerEngineDType(buffer_dtype), helper->myrank,
          helper->numranks, helper->mylocal, helper->numlocal, helper->mynode, helper->numnodes,
          tp_size, std::bind(&CommOverlapHelper::ub_allgather, helper, _1, _2, _3, _4, _5),
          std::bind(&CommOverlapHelper::ub_barrier, helper, _1), comm_type, num_max_streams,
          comm_cga_size, gemm_priority, comm_priority, num_comm_sm, set_sm_margin, use_ce,
          atomic_gemm, aggregate) {}

CommOverlapP2P::CommOverlapP2P(CommOverlapHelper *helper, int tp_rank, int tp_size,
                               te::CommOverlapType comm_type,
                               const std::vector<size_t> &buffer_shape, at::ScalarType buffer_dtype,
                               int num_comm_sm, bool atomic_gemm)
    : te::CommOverlapP2PBase(helper->get_nccl_comm("intra").get(), tp_rank, tp_size, num_comm_sm,
                             atomic_gemm),
      _nccl_comm(helper->get_nccl_comm("intra")) {
  // See CommOverlap constructor for the buffer_dtype rationale.
  (void)buffer_dtype;
  cublasmp_capture_warmup(this, tp_size, comm_type, buffer_shape, _warmup_workspace);
}

/*
** Copy input to _ubufs[0]
*/
void CommOverlapP2P::copy_into_buffer(const at::Tensor &input, bool local_chunk) {
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
}

at::Tensor CommOverlapP2P::get_buffer(bool local_chunk, std::optional<std::vector<int64_t>> shape) {
  // Check buffer shape
  if (shape) {
    const size_t requested_size = transformer_engine::pytorch::product(*shape);
    if (local_chunk) {
      NVTE_CHECK(requested_size == _ubufs[_tp_id].numel(),
                 "Invalid shape for local chunk of a Userbuffers buffer (requested shape=", *shape,
                 ", local_ubuf_size=", _ubufs[_tp_id].numel(), ")");
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
  return torch::from_blob(ubuf_ptr, *shape, at::dtype(dtype).device(torch::kCUDA));
}

std::pair<at::Stream, at::Stream> CommOverlapP2P::get_communication_stream() {
  return {at::cuda::getStreamFromExternal(_stream_send[0], at::cuda::current_device()),
          at::cuda::getStreamFromExternal(_stream_recv, at::cuda::current_device())};
}

void transformer_engine::pytorch::bulk_overlap_ag_with_external_gemm(
    CommOverlap &allgather_communicator, at::Stream send_stream, at::Stream recv_stream) {
  auto main_stream = at::cuda::getCurrentCUDAStream();
  allgather_communicator.bulk_overlap_external_ag(at::cuda::CUDAStream(send_stream),
                                                  at::cuda::CUDAStream(recv_stream), main_stream);
}
