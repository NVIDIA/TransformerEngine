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
                         int comm_priority, int num_comm_sm, bool set_sm_margin, bool atomic_gemm,
                         bool rs_overlap_first_gemm)
    : te::CommOverlapBase(buffer_shape, te::pytorch::GetTransformerEngineDType(buffer_dtype),
                          helper->myrank, helper->numranks, helper->mylocal, helper->numlocal,
                          helper->mynode, helper->numnodes, tp_size,
                          std::bind(&CommOverlapHelper::ub_allgather, helper, _1, _2, _3, _4, _5),
                          std::bind(&CommOverlapHelper::ub_barrier, helper, _1), num_splits,
                          num_max_streams, comm_cga_size, gemm_priority, comm_priority, num_comm_sm,
                          set_sm_margin, atomic_gemm, rs_overlap_first_gemm) {}

void CommOverlap::set_buffer_params(py::handle quantizer) {
  std::unique_ptr<te::pytorch::Quantizer> my_quantizer = te::pytorch::convert_quantizer(quantizer);
  my_quantizer->set_quantization_params(&_ubuf);
  _ubuf_scale_inv_initialized = true;
}

/*
** Helper function to copy input to _ubuf
*/
void CommOverlap::copy_into_buffer(py::handle input, py::handle quantizer, bool local_chunk) {
  auto input_tensor = te::pytorch::makeTransformerEngineTensor(input, quantizer);
  auto input_ptr = input_tensor.dptr() ? input_tensor.dptr() : input_tensor.columnwise_dptr();
  NVTE_CHECK(input_ptr, "Input tensor does not have rowwise or columnwise data!");

  char *ubuf_ptr = reinterpret_cast<char *>(_ubuf.dptr());
  if (local_chunk) {
    if (input_tensor.numel() * _tp_size > (int64_t)_ubuf.numel())
      NVTE_ERROR("input is larger than the local communication buffer!");
    if (input_tensor.element_size() != (int64_t)_ubuf.element_size())
      NVTE_ERROR("input data type does not match communication buffer!");
    ubuf_ptr += (_ubuf.numel() / _tp_size) * _tp_id * _ubuf.element_size();
  } else {
    if (input_tensor.numel() > (int64_t)_ubuf.numel())
      NVTE_ERROR("input is larger than the global communication buffer!");
    if (input_tensor.element_size() != (int64_t)_ubuf.element_size())
      NVTE_ERROR("input data type does not match communication buffer!");
  }

  // Copy either row or columnwise data into the communication buffer's columnwise data
  // NOTE: _ubuf.columnwise_dptr() is not a valid copy target because it is not registered with
  //       the Userbuffers communicator.
  at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
  NVTE_CHECK_CUDA(cudaEventRecord(_start_d2dcopy, (cudaStream_t)stream_main));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_comm, _start_d2dcopy, 0));
  NVTE_CHECK_CUDA(cudaMemcpyAsync(ubuf_ptr, input_tensor.dptr(),
                                  input_tensor.numel() * input_tensor.element_size(),
                                  cudaMemcpyDeviceToDevice, (cudaStream_t)_stream_comm));
}

py::object CommOverlap::get_buffer(py::handle quantizer, bool local_chunk,
                                   std::optional<const std::vector<int64_t>> shape) {
  using namespace te::pytorch;
  char *ubuf_wt_ptr = reinterpret_cast<char *>(_ubuf.dptr());
  if (local_chunk) ubuf_wt_ptr += _ubuf.numel() / _tp_size * _tp_id * _ubuf.element_size();

  std::vector<int64_t> torch_shape;
  if (shape.has_value()) {
    torch_shape = shape.value();
    auto requested = product(torch_shape);
    auto expected = local_chunk ? _ubuf.numel() / _tp_size : _ubuf.numel();
    NVTE_CHECK(requested == expected, "Number of elements in the requested shape (", requested,
               ") does not match allocated buffer size (", expected, ")!");
  } else {
    int64_t output_c_dim0 = (local_chunk) ? _ubuf.size(0) / _tp_size : _ubuf.size(0);
    int64_t output_c_dim1 = _ubuf.size(1);
    torch_shape = {output_c_dim0, output_c_dim1};
  }

  auto ubuf_tensor = torch::from_blob(reinterpret_cast<void *>(ubuf_wt_ptr), torch_shape,
                                      at::dtype(GetATenDType(_ubuf.dtype())).device(torch::kCUDA));

  std::unique_ptr<Quantizer> my_quantizer = convert_quantizer(quantizer);
  std::vector<size_t> te_shape;
  for (auto s : torch_shape) te_shape.emplace_back(static_cast<size_t>(s));

  // Always output a rowwise-only QuantizedTensor
  // TODO (Alp): This needs to produce an un-interleaved transpose when required.
  auto is_internal = my_quantizer->internal;
  auto uses_columnwise = my_quantizer->columnwise_usage;
  my_quantizer->internal = false;
  my_quantizer->columnwise_usage = false;
  auto [te_tensor, py_tensor] = my_quantizer->create_tensor(te_shape, _ubuf.dtype(), ubuf_tensor);
  my_quantizer->internal = is_internal;
  my_quantizer->columnwise_usage = uses_columnwise;
  return py_tensor;
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

void CommOverlapP2P::set_buffer_params(py::handle quantizer) {
  std::unique_ptr<te::pytorch::Quantizer> my_quantizer = te::pytorch::convert_quantizer(quantizer);
  my_quantizer->set_quantization_params(&_ubuf);
  for (size_t i = 0; i < _ubufs.size(); i++) my_quantizer->set_quantization_params(&_ubufs[i]);
}

/*
** Copy input to _ubufs[0]
*/
void CommOverlapP2P::copy_into_buffer(py::handle input, py::handle quantizer, bool local_chunk) {
  auto input_tensor = te::pytorch::makeTransformerEngineTensor(input, quantizer);
  auto input_ptr = input_tensor.dptr() ? input_tensor.dptr() : input_tensor.columnwise_dptr();
  NVTE_CHECK(input_ptr, "Input tensor does not have rowwise or columnwise data!");

  at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
  if (local_chunk) {
    // Copy input to the target ubuf chunk by rank offset
    if (input_tensor.numel() * _tp_size > (int64_t)_ubuf.numel())
      NVTE_ERROR("input is larger than the local communication buffer!");
    if (input_tensor.element_size() != (int64_t)_ubuf.element_size())
      NVTE_ERROR("input data type does not match communication buffer!");
    NVTE_CHECK_CUDA(cudaMemcpyAsync(_ubufs[_tp_id].dptr(), input_ptr,
                                    input_tensor.numel() * input_tensor.element_size(),
                                    cudaMemcpyDeviceToDevice, (cudaStream_t)stream_main));

  } else {
    if (input_tensor.numel() > (int64_t)_ubuf.numel())
      NVTE_ERROR("input is larger than the global communication buffer!");
    if (input_tensor.element_size() != (int64_t)_ubuf.element_size())
      NVTE_ERROR("input data type does not match communication buffer!");
    NVTE_CHECK_CUDA(cudaMemcpyAsync(_ubuf.dptr(), input_ptr,
                                    input_tensor.numel() * input_tensor.element_size(),
                                    cudaMemcpyDeviceToDevice, (cudaStream_t)stream_main));
  }
}

py::object CommOverlapP2P::get_buffer(py::handle quantizer, bool local_chunk,
                                      std::optional<const std::vector<int64_t>> shape) {
  using namespace te::pytorch;
  char *ubuf_wt_ptr = reinterpret_cast<char *>(local_chunk ? _ubufs[_tp_id].dptr() : _ubuf.dptr());

  std::vector<int64_t> torch_shape;
  if (shape.has_value()) {
    torch_shape = shape.value();
    auto requested = product(torch_shape);
    auto expected = local_chunk ? _ubufs[_tp_id].numel() : _ubuf.numel();
    NVTE_CHECK(requested == expected, "Number of elements in the requested shape (", requested,
               ") does not match allocated buffer size (", expected, ")!");
  } else {
    int64_t output_c_dim0 = (local_chunk) ? _ubuf.size(0) / _tp_size : _ubuf.size(0);
    int64_t output_c_dim1 = _ubuf.size(1);
    torch_shape = {output_c_dim0, output_c_dim1};
  }
  auto ubuf_tensor = torch::from_blob(reinterpret_cast<void *>(ubuf_wt_ptr), torch_shape,
                                      at::dtype(GetATenDType(_ubuf.dtype())).device(torch::kCUDA));

  std::unique_ptr<Quantizer> my_quantizer = convert_quantizer(quantizer);
  std::vector<size_t> te_shape;
  for (auto s : torch_shape) te_shape.emplace_back(static_cast<size_t>(s));

  // Always output a rowwise-only QuantizedTensor
  // TODO (Alp): This needs to produce an un-interleaved transpose when required.
  auto is_internal = my_quantizer->internal;
  auto uses_columnwise = my_quantizer->columnwise_usage;
  my_quantizer->internal = false;
  my_quantizer->columnwise_usage = false;
  auto [te_tensor, py_tensor] = my_quantizer->create_tensor(te_shape, _ubuf.dtype(), ubuf_tensor);
  my_quantizer->internal = is_internal;
  my_quantizer->columnwise_usage = uses_columnwise;
  return py_tensor;
}
