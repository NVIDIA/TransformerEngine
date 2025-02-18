/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"

#define HALF_BYTES 2
#define UB_MAX_SM 32

using namespace torch::indexing;
using namespace std::placeholders;

namespace te = transformer_engine;

#define MAKE_TRANSFORMER_ENGINE_TENSORS(A, A_scale_inv, A_scaling_mode, A_type, B, B_scale_inv,    \
                                        B_scaling_mode, B_type, D, D_amax, D_scale, D_type, bias,  \
                                        bias_type, pre_gelu_out, workspace)                        \
  A = A.contiguous();                                                                              \
  auto dimA = A_scaling_mode.size();                                                               \
  NVTE_CHECK(dimA == 3, "Incorrect size ", dimA, " for scaling mode.");                            \
  NVTEScalingMode nvte_scaling_modeA = {static_cast<int>(A_scaling_mode[0]),                       \
                                        static_cast<int>(A_scaling_mode[1]),                       \
                                        static_cast<int>(A_scaling_mode[2])};                      \
  auto A_ = makeTransformerEngineTensor(                                                           \
      A.data_ptr(), {static_cast<size_t>(A.size(0)), static_cast<size_t>(A.size(1))}, A_type,      \
      nullptr, nullptr, A_scale_inv.data_ptr(), getTensorShape(A_scale_inv), nvte_scaling_modeA);  \
  B = B.contiguous();                                                                              \
  auto dimB = B_scaling_mode.size();                                                               \
  NVTE_CHECK(dimB == 3, "Incorrect size ", dimB, " for scaling mode.");                            \
  NVTEScalingMode nvte_scaling_modeB = {static_cast<int>(B_scaling_mode[0]),                       \
                                        static_cast<int>(B_scaling_mode[1]),                       \
                                        static_cast<int>(B_scaling_mode[2])};                      \
  auto B_ = makeTransformerEngineTensor(                                                           \
      B.data_ptr(), {static_cast<size_t>(B.size(0)), static_cast<size_t>(B.size(1))}, B_type,      \
      nullptr, nullptr, B_scale_inv.data_ptr(), getTensorShape(B_scale_inv), nvte_scaling_modeB);  \
  auto D_ = makeTransformerEngineTensor(                                                           \
      D.data_ptr(), {static_cast<size_t>(D.size(0)), static_cast<size_t>(D.size(1))}, D_type,      \
      D_amax.data_ptr(), D_scale.data_ptr(), nullptr);                                             \
  auto bias_ = makeTransformerEngineTensor(                                                        \
      bias.data_ptr(), std::vector<size_t>{static_cast<size_t>(bias.size(0))}, bias_type);         \
  const auto gelu_shape = (pre_gelu_out.data_ptr() == nullptr)                                     \
                              ? std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0))}     \
                              : std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0)),     \
                                                    static_cast<size_t>(pre_gelu_out.size(1))};    \
  auto pre_gelu_out_ = makeTransformerEngineTensor(                                                \
      pre_gelu_out.data_ptr(), gelu_shape, GetTransformerEngineDType(pre_gelu_out.scalar_type())); \
  auto workspace_ = makeTransformerEngineTensor(                                                   \
      workspace.data_ptr(), std::vector<size_t>{static_cast<size_t>(workspace.size(0))},           \
      te::DType::kByte);

/***************************************************************************************************
 * CommOverlapHelper
 **************************************************************************************************/

CommOverlapHelper::CommOverlapHelper() {
#ifndef NVTE_UB_WITH_MPI
  NVTE_ERROR("Internal TE error: Dummy CommOverlapHelper init without NVTE_UB_WITH_MPI=1!");
#endif
}  // empty constructor for NVTE_UB_WITH_MPI=1

CommOverlapHelper::CommOverlapHelper(c10d::ProcessGroup *world_group,
                                     std::optional<c10d::ProcessGroup *> intra_domain_group,
                                     std::optional<c10d::ProcessGroup *> inter_domain_group) {
#ifndef NVTE_UB_WITH_MPI
  pgs.insert({"world", world_group});
  myrank = pgs["world"]->getRank();
  numranks = pgs["world"]->getSize();
  c10d::ProcessGroup::BackendType backend = pgs["world"]->getBackendType();
  backend_is_nccl = (backend == c10d::ProcessGroup::BackendType::NCCL);

  if (intra_domain_group.has_value()) {
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
      // Intra-node group is different than the world group so there must be multiple nodes
      NVTE_CHECK(
          inter_domain_group.has_value(),
          "Internal TE error: Inter-node group cannot be `None` when intra-node group is not ",
          "identical to the world_group!");

      // Get node ID and number of nodes
      NVTE_CHECK(
          inter_domain_group.value()->getBackendType() == backend,
          "Internal TE error: Inter-node group must be on the same backend (%s) as the world ",
          "group!", pgs["world"]->getBackendName());
      pgs.insert({"inter", inter_domain_group.value()});
      mynode = pgs["inter"]->getRank();
      numnodes = pgs["inter"]->getSize();
    }
  } else {
    // Intra-node group is not set so we assume there is only 1 node
    mylocal = myrank;
    numlocal = numranks;
    pgs.insert({"intra", world_group});

    mynode = 0;
    numnodes = 1;
  }

  initialized = true;
#else
  NVTE_ERROR("Internal TE error: CommOverlapHelper cannot be initialized with valid PyTorch ",
             "distributed process groups when TE is compiled with NVTE_UB_WITH_MPI=1!");
#endif
}

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

CommOverlap::CommOverlap(const std::vector<size_t> &buffer_shape, at::ScalarType buffer_dtype,
                         CommOverlapHelper *helper, int tp_size, int num_splits,
                         int num_max_streams, int comm_cga_size, int gemm_priority,
                         int comm_priority, int num_comm_sm, bool set_sm_margin, bool atomic_gemm)
    : te::CommOverlapBase(
          buffer_shape, GetTransformerEngineDType(buffer_dtype), helper->myrank, helper->numranks,
          helper->mylocal, helper->numlocal, helper->mynode, helper->numnodes, tp_size,
          std::bind(&CommOverlapHelper::ub_allgather, helper, _1, _2, _3, _4, _5),
          std::bind(&CommOverlapHelper::ub_barrier, helper, _1), num_splits, num_max_streams,
          comm_cga_size, gemm_priority, comm_priority, num_comm_sm, set_sm_margin, atomic_gemm) {
  // Even though we never use these PyTorch tensor wrappers directly, they're still necessary to
  // for PyTorch to factor externally allocated memory into its memory pool and garbage collection
  // threshold calculation.
  _ubuf_torch = torch::from_blob(
      _ubuf.dptr(), {static_cast<int64_t>(_ubuf.size(0)), static_cast<int64_t>(_ubuf.size(1))},
      at::device(torch::kCUDA).dtype(buffer_dtype));
  if (_atomic_gemm) {
    _ubuf_counter = torch::from_blob(_counter.dptr(), {static_cast<int64_t>(_num_splits * 2)},
                                     at::device(torch::kCUDA).dtype(torch::kInt32));
  }
}

/*
** Bulk GEMM + COMM
** This function assumes the communication input is pre-copied to _ubuf
*/
std::vector<at::Tensor> CommOverlap::bulk_overlap(
    at::Tensor A, at::Tensor A_scale_inverse, te::DType A_type, std::vector<int64_t> A_scaling_mode,
    bool transa, at::Tensor B, at::Tensor B_scale_inverse, te::DType B_type,
    std::vector<int64_t> B_scaling_mode, bool transb, at::Tensor D, at::Tensor D_scale,
    te::DType D_type, at::Tensor D_amax, at::Tensor bias, te::DType bias_type,
    at::Tensor pre_gelu_out, bool grad, at::Tensor workspace, size_t workspaceSize, bool accumulate,
    bool use_split_accumulator, te::CommOverlapType comm_type, at::Tensor rs_output) {
  MAKE_TRANSFORMER_ENGINE_TENSORS(A, A_scale_inverse, A_scaling_mode, A_type, B, B_scale_inverse,
                                  B_scaling_mode, B_type, D, D_amax, D_scale, D_type, bias,
                                  bias_type, pre_gelu_out, workspace)

  auto rs_out_ = makeTransformerEngineTensor(rs_output);
  cudaStream_t stream_main = static_cast<cudaStream_t>(at::cuda::getCurrentCUDAStream());
  te::CommOverlapBase::bulk_overlap(A_, transa, B_, transb, D_, bias_, pre_gelu_out_, workspace_,
                                    grad, accumulate, use_split_accumulator, comm_type, rs_out_,
                                    stream_main);

  // Get the current userbuf offset
  char *ubuf_wt_ptr = reinterpret_cast<char *>(_ubuf.dptr());
  if (comm_type == te::CommOverlapType::RS) {
    ubuf_wt_ptr += _ubuf.numel() / _tp_size * _tp_id * _ubuf.element_size();
  }

  // Generate output tensor from userbuf data pointer
  int output_c_dim0 =
      (comm_type == te::CommOverlapType::AG) ? _ubuf.size(0) : _ubuf.size(0) / _tp_size;
  int output_c_dim1 = _ubuf.size(1);
  auto output_tensor =
      torch::from_blob(ubuf_wt_ptr, {output_c_dim0, output_c_dim1}, _ubuf_torch.options());

  return {D, output_tensor};
}  // CommOverlap::bulk_overlap

/*
** Split FPROP GEMM + ReduceScatter
*/
void CommOverlap::atomic_gemm_overlap_rs(
    at::Tensor A, at::Tensor A_scale_inverse, te::DType A_type, std::vector<int64_t> A_scaling_mode,
    bool transa, at::Tensor B, at::Tensor B_scale_inverse, te::DType B_type,
    std::vector<int64_t> B_scaling_mode, bool transb, at::Tensor D, at::Tensor D_scale,
    te::DType D_type, at::Tensor D_amax, at::Tensor bias, te::DType bias_type,
    at::Tensor pre_gelu_out, bool grad, at::Tensor workspace, size_t workspaceSize, bool accumulate,
    bool use_split_accumulator, bool gemm_overlap, at::Tensor rs_output) {
  MAKE_TRANSFORMER_ENGINE_TENSORS(A, A_scale_inverse, A_scaling_mode, A_type, B, B_scale_inverse,
                                  B_scaling_mode, B_type, D, D_amax, D_scale, D_type, bias,
                                  bias_type, pre_gelu_out, workspace)

  auto rs_out_ = makeTransformerEngineTensor(rs_output);
  cudaStream_t stream_main = static_cast<cudaStream_t>(at::cuda::getCurrentCUDAStream());
  te::CommOverlapBase::atomic_gemm_overlap_rs(A_, transa, B_, transb, D_, bias_, pre_gelu_out_,
                                              workspace_, grad, accumulate, use_split_accumulator,
                                              gemm_overlap, rs_out_, stream_main);
}  // CommOverlap::split_overlap_rs

/*
** Split FPROP GEMM + ReduceScatter
*/
void CommOverlap::split_overlap_rs(at::Tensor A, at::Tensor A_scale_inverse, te::DType A_type,
                                   std::vector<int64_t> A_scaling_mode, bool transa, at::Tensor B,
                                   at::Tensor B_scale_inverse, te::DType B_type,
                                   std::vector<int64_t> B_scaling_mode, bool transb, at::Tensor D,
                                   at::Tensor D_scale, te::DType D_type, at::Tensor D_amax,
                                   at::Tensor bias, te::DType bias_type, at::Tensor pre_gelu_out,
                                   bool grad, at::Tensor workspace, size_t workspaceSize,
                                   bool accumulate, bool use_split_accumulator, bool gemm_overlap,
                                   at::Tensor rs_output) {
  MAKE_TRANSFORMER_ENGINE_TENSORS(A, A_scale_inverse, A_scaling_mode, A_type, B, B_scale_inverse,
                                  B_scaling_mode, B_type, D, D_amax, D_scale, D_type, bias,
                                  bias_type, pre_gelu_out, workspace)

  auto rs_out_ = makeTransformerEngineTensor(rs_output);
  cudaStream_t stream_main = static_cast<cudaStream_t>(at::cuda::getCurrentCUDAStream());
  te::CommOverlapBase::split_overlap_rs(A_, transa, B_, transb, D_, bias_, pre_gelu_out_,
                                        workspace_, grad, accumulate, use_split_accumulator,
                                        gemm_overlap, rs_out_, stream_main);
}  // CommOverlap::split_overlap_rs

/*
** Helper function to copy input to _ubuf
*/
void CommOverlap::copy_input_to_ubuf(torch::Tensor input, int comm_type) {
  char *ubuf_ptr = reinterpret_cast<char *>(_ubuf.dptr());
  te::CommOverlapType _comm_type = static_cast<te::CommOverlapType>(comm_type);
  if (_comm_type == te::CommOverlapType::AG) {
    if ((input.numel() * _tp_size) != (int64_t)_ubuf.numel() ||
        input.element_size() != (int64_t)_ubuf.element_size()) {
      NVTE_ERROR("input and ubuf size do not match!");
    }
    ubuf_ptr += _ubuf.numel() / _tp_size * _tp_id * _ubuf.element_size();
  } else {
    if (input.numel() != (int64_t)_ubuf.numel() ||
        input.element_size() != (int64_t)_ubuf.element_size()) {
      NVTE_ERROR("input and ubuf size do not match!");
    }
  }

  at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
  NVTE_CHECK_CUDA(cudaEventRecord(_start_d2dcopy, (cudaStream_t)stream_main));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_comm, _start_d2dcopy, 0));
  NVTE_CHECK_CUDA(cudaMemcpyAsync(ubuf_ptr, input.data_ptr(), input.numel() * input.element_size(),
                                  cudaMemcpyDeviceToDevice, (cudaStream_t)_stream_comm));
}

torch::Tensor CommOverlap::get_ubuf_output(int comm_type) {
  char *ubuf_wt_ptr = reinterpret_cast<char *>(_ubuf.dptr());
  te::CommOverlapType _comm_type = static_cast<te::CommOverlapType>(comm_type);
  if (_comm_type != te::CommOverlapType::AG && _comm_type != te::CommOverlapType::RS)
    NVTE_ERROR("Invalid comm_type");
  if (_comm_type == te::CommOverlapType::RS)
    ubuf_wt_ptr += _ubuf.numel() / _tp_size * _tp_id * _ubuf.element_size();
  int output_c_dim0 =
      (_comm_type == te::CommOverlapType::AG) ? _ubuf.size(0) : _ubuf.size(0) / _tp_size;
  int output_c_dim1 = _ubuf.size(1);
  return torch::from_blob(ubuf_wt_ptr, {output_c_dim0, output_c_dim1},
                          torch::device(torch::kCUDA).dtype(GetATenDType(_ubuf.dtype())));
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
          buffer_shape, GetTransformerEngineDType(buffer_dtype), helper->myrank, helper->numranks,
          helper->mylocal, helper->numlocal, helper->mynode, helper->numnodes, tp_size,
          std::bind(&CommOverlapHelper::ub_allgather, helper, _1, _2, _3, _4, _5),
          std::bind(&CommOverlapHelper::ub_barrier, helper, _1), comm_type, num_max_streams,
          comm_cga_size, gemm_priority, comm_priority, num_comm_sm, set_sm_margin, use_ce,
          atomic_gemm, aggregate) {
  // Even though we never use these PyTorch tensor wrappers directly, they're still necessary to
  // for PyTorch to factor externally allocated memory into its memory pool and garbage collection
  // threshold calculation.
  _ubuf_torch = torch::from_blob(
      _ubuf.dptr(), {static_cast<int64_t>(_ubuf.size(0)), static_cast<int64_t>(_ubuf.size(1))},
      at::device(torch::kCUDA).dtype(buffer_dtype));
  if (_atomic_gemm) {
    _ubuf_counter = torch::from_blob(_counter.dptr(), {static_cast<int64_t>(_num_splits * 2)},
                                     at::device(torch::kCUDA).dtype(torch::kInt32));
  }
}

/*
** Split AllGather + AtomicGEMM using P2P communication
** This function assumes the input_b is pre-copied to _ubufs[rank_id]. This is
*needed to have AG outputs
** in each rank to be in the contiguous memory space after all ring exchange
*phases.
*/
void CommOverlapP2P::atomic_gemm_overlap_ag(
    at::Tensor A, at::Tensor A_scale_inverse, te::DType A_type, std::vector<int64_t> A_scaling_mode,
    bool transa, at::Tensor B, at::Tensor B_scale_inverse, te::DType B_type,
    std::vector<int64_t> B_scaling_mode, bool transb, at::Tensor D, at::Tensor D_scale,
    te::DType D_type, at::Tensor D_amax, at::Tensor bias, te::DType bias_type,
    at::Tensor pre_gelu_out, bool grad, at::Tensor workspace, size_t workspaceSize, bool accumulate,
    bool use_split_accumulator, at::Tensor B_copy) {
  MAKE_TRANSFORMER_ENGINE_TENSORS(A, A_scale_inverse, A_scaling_mode, A_type, B, B_scale_inverse,
                                  B_scaling_mode, B_type, D, D_amax, D_scale, D_type, bias,
                                  bias_type, pre_gelu_out, workspace)

  auto B_copy_ = makeTransformerEngineTensor(B_copy);
  cudaStream_t stream_main = static_cast<cudaStream_t>(at::cuda::getCurrentCUDAStream());
  te::CommOverlapP2PBase::atomic_gemm_overlap_ag(A_, transa, B_, transb, D_, bias_, pre_gelu_out_,
                                                 workspace_, grad, accumulate,
                                                 use_split_accumulator, B_copy_, stream_main);
}  // atomic_gemm_overlap_ag

/*
** Split AllGather + GEMM using P2P communication
** This function assumes the input_b is pre-copied to _ubufs[rank_id]. This is
*needed to have AG outputs
** in each rank to be in the contiguous memory space after all ring exchange
*phases.
*/
void CommOverlapP2P::split_overlap_ag(at::Tensor A, at::Tensor A_scale_inverse, te::DType A_type,
                                      std::vector<int64_t> A_scaling_mode, bool transa,
                                      at::Tensor B, at::Tensor B_scale_inverse, te::DType B_type,
                                      std::vector<int64_t> B_scaling_mode, bool transb,
                                      at::Tensor D, at::Tensor D_scale, te::DType D_type,
                                      at::Tensor D_amax, at::Tensor bias, te::DType bias_type,
                                      at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
                                      size_t workspaceSize, bool accumulate,
                                      bool use_split_accumulator, at::Tensor B_copy) {
  MAKE_TRANSFORMER_ENGINE_TENSORS(A, A_scale_inverse, A_scaling_mode, A_type, B, B_scale_inverse,
                                  B_scaling_mode, B_type, D, D_amax, D_scale, D_type, bias,
                                  bias_type, pre_gelu_out, workspace)

  auto B_copy_ = makeTransformerEngineTensor(B_copy);
  cudaStream_t stream_main = static_cast<cudaStream_t>(at::cuda::getCurrentCUDAStream());
  te::CommOverlapP2PBase::split_overlap_ag(A_, transa, B_, transb, D_, bias_, pre_gelu_out_,
                                           workspace_, grad, accumulate, use_split_accumulator,
                                           B_copy_, stream_main);
}  // split_overlap_ag

/*
** Split ReduceScatter + GEMM using P2P communication
*/
void CommOverlapP2P::atomic_gemm_overlap_rs(
    at::Tensor A, at::Tensor A_scale_inverse, te::DType A_type, std::vector<int64_t> A_scaling_mode,
    bool transa, at::Tensor B, at::Tensor B_scale_inverse, te::DType B_type,
    std::vector<int64_t> B_scaling_mode, bool transb, at::Tensor D, at::Tensor D_scale,
    te::DType D_type, at::Tensor D_amax, at::Tensor bias, te::DType bias_type,
    at::Tensor pre_gelu_out, bool grad, at::Tensor workspace, size_t workspaceSize, bool accumulate,
    bool use_split_accumulator, at::Tensor rs_output) {
  MAKE_TRANSFORMER_ENGINE_TENSORS(A, A_scale_inverse, A_scaling_mode, A_type, B, B_scale_inverse,
                                  B_scaling_mode, B_type, D, D_amax, D_scale, D_type, bias,
                                  bias_type, pre_gelu_out, workspace)

  auto rs_out_ = makeTransformerEngineTensor(rs_output);
  cudaStream_t stream_main = static_cast<cudaStream_t>(at::cuda::getCurrentCUDAStream());
  te::CommOverlapP2PBase::atomic_gemm_overlap_rs(A_, transa, B_, transb, D_, bias_, pre_gelu_out_,
                                                 workspace_, grad, accumulate,
                                                 use_split_accumulator, rs_out_, stream_main);
}

/*
** Split ReduceScatter + GEMM using P2P communication
*/
void CommOverlapP2P::split_overlap_rs(at::Tensor A, at::Tensor A_scale_inverse, te::DType A_type,
                                      std::vector<int64_t> A_scaling_mode, bool transa,
                                      at::Tensor B, at::Tensor B_scale_inverse, te::DType B_type,
                                      std::vector<int64_t> B_scaling_mode, bool transb,
                                      at::Tensor D, at::Tensor D_scale, te::DType D_type,
                                      at::Tensor D_amax, at::Tensor bias, te::DType bias_type,
                                      at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
                                      size_t workspaceSize, bool accumulate,
                                      bool use_split_accumulator, at::Tensor rs_output) {
  MAKE_TRANSFORMER_ENGINE_TENSORS(A, A_scale_inverse, A_scaling_mode, A_type, B, B_scale_inverse,
                                  B_scaling_mode, B_type, D, D_amax, D_scale, D_type, bias,
                                  bias_type, pre_gelu_out, workspace)

  auto rs_out_ = makeTransformerEngineTensor(rs_output);
  cudaStream_t stream_main = static_cast<cudaStream_t>(at::cuda::getCurrentCUDAStream());
  te::CommOverlapP2PBase::split_overlap_rs(A_, transa, B_, transb, D_, bias_, pre_gelu_out_,
                                           workspace_, grad, accumulate, use_split_accumulator,
                                           rs_out_, stream_main);
}

/*
** Copy input to _ubufs[0]
*/
void CommOverlapP2P::copy_input_to_ubuf(torch::Tensor input, bool chunk) {
  at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
  if (chunk) {
    // Copy input to the target ubuf chunk by rank offset
    if (input.numel() != (int64_t)_ubufs[0].numel() ||
        input.element_size() != (int64_t)_ubufs[0].element_size()) {
      NVTE_ERROR("input and ubuf size do not match!");
    }
    NVTE_CHECK_CUDA(cudaMemcpyAsync(_ubufs[_tp_id].dptr(), input.data_ptr(),
                                    input.numel() * input.element_size(), cudaMemcpyDeviceToDevice,
                                    (cudaStream_t)stream_main));
  } else {
    if (input.numel() != (int64_t)_ubuf.numel() ||
        input.element_size() != (int64_t)_ubuf.element_size()) {
      NVTE_ERROR("input and ubuf size do not match!");
    }
    NVTE_CHECK_CUDA(cudaMemcpyAsync(_ubuf.dptr(), input.data_ptr(),
                                    input.numel() * input.element_size(), cudaMemcpyDeviceToDevice,
                                    (cudaStream_t)stream_main));
  }
}

torch::Tensor CommOverlapP2P::get_ubuf_output(int comm_type) {
  char *ubuf_wt_ptr = reinterpret_cast<char *>(_ubuf.dptr());
  te::CommOverlapType _comm_type = static_cast<te::CommOverlapType>(comm_type);
  if (_comm_type != te::CommOverlapType::AG && _comm_type != te::CommOverlapType::RS)
    NVTE_ERROR("Invalid comm_type");
  if (_comm_type == te::CommOverlapType::RS)
    ubuf_wt_ptr += _ubuf.numel() / _tp_size * _self_chunk_id * _ubuf.element_size();
  int output_c_dim0 =
      (_comm_type == te::CommOverlapType::AG) ? _ubuf.size(0) : _ubuf.size(0) / _tp_size;
  int output_c_dim1 = _ubuf.size(1);
  return torch::from_blob(ubuf_wt_ptr, {output_c_dim0, output_c_dim1}, _ubuf_torch.options());
}
