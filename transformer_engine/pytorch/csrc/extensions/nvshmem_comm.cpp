/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/


#include "../extensions.h"
#include <nvshmem.h>
#include <nvshmemx.h>
#include <nvshmem_api/nvshmem_waitkernel.h>
#include <torch/cuda.h>
#include <cuda.h>
#include <torch/extension.h>
#include <cuda_fp8.h>

namespace nvshmem_api {
 void init_nvshmem_backend(c10d::ProcessGroup *process_group) {
  nvshmemx_init_attr_t attr = {};
  nvshmemx_uniqueid_t id = {};

  int my_rank = process_group->getRank();
  int num_ranks = process_group->getSize();
  if (my_rank == 0) {
       nvshmemx_get_uniqueid(&id);
  }

  auto backend_is_nccl = (process_group->getBackendType() == c10d::ProcessGroup::BackendType::NCCL);
  NVTE_CHECK(backend_is_nccl, "Currently only support NCCL boostrap for NVSHMEM");
  auto datatensor = torch::from_blob((void*)&id, {static_cast<int64_t>(sizeof(nvshmemx_uniqueid_t) / sizeof(uint8_t))},
                                     at::device(torch::kCPU).dtype(torch::kUInt8));
  auto datatmp = (backend_is_nccl) ? datatensor.cuda() : datatensor;

  c10d::BroadcastOptions bcast_opts;
  bcast_opts.rootRank = 0;
  std::vector<torch::Tensor> datachunk = {datatmp};
  auto work = process_group->broadcast(datachunk, bcast_opts);
  work->wait();

  if (backend_is_nccl) {
    datatensor.copy_(datatmp.cpu());
    datatmp = torch::Tensor();
  }

  nvshmemx_set_attr_uniqueid_args(my_rank, num_ranks, &id, &attr);
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);

  assert(my_rank == nvshmem_my_pe());
  assert(num_ranks == nvshmem_n_pes());
}

void nvshmem_wait_on_stream(torch::Tensor signal, int wait_kind){
  uint64_t* sig_addr = (uint64_t*) signal.data_ptr();
  cudaStream_t cur_stream = (cudaStream_t)at::cuda::getCurrentCUDAStream();

  transformer_engine::nvshmem_wait_on_stream(sig_addr, wait_kind, cur_stream);
}


torch::Tensor create_nvshmem_tensor(const std::vector<int64_t> &shape, c10::ScalarType dtype){
  auto option_gpu =
      at::TensorOptions().dtype(dtype).device(at::kCUDA).device_index(c10::cuda::current_device());
  auto size = torch::elementSize(dtype) *
              std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
  return at::from_blob(
      nvshmem_malloc(size), shape, [](void *ptr) { nvshmem_free(ptr); }, option_gpu);

}

void nvshmem_send_on_stream(torch::Tensor src, torch::Tensor dst, int peer, torch::Tensor signal){
  void* src_ptr = (void *) src.data_ptr();
  void* dst_ptr = (void *) dst.data_ptr();
  uint64_t* sig_addr = (uint64_t*) signal.data_ptr();
  auto nelement = src.numel() * src.element_size();
  uint64_t sigval = 1;
  at::cuda::CUDAStream cur_stream = at::cuda::getCurrentCUDAStream();

  nvshmemx_putmem_signal_on_stream(dst_ptr, src_ptr, nelement, sig_addr, sigval, NVSHMEM_SIGNAL_SET, peer, (cudaStream_t)cur_stream);
}
void nvshmem_finalize(){
  nvshmem_finalize();
}

void nvshmem_quiet(){
  nvshmem_quiet();
}

void nvshmem_ag_from_p2p_on_stream(torch::Tensor buf, torch::Tensor singals, int my_rank, const std::vector<int> &global_ranks){
  // Draft implementation of CE based AG using ring-exchange; This function assumes that the gathered data on local rank is already in buf tensor
  // global_ranks: global ranks for current process group
  // my_rank: the local rank in current process group
  // size of signals must be equal to the size of current process group
  int num_ranks = global_ranks.size();
  if (num_ranks == 1) return ;

  char* buf_start_ptr = (char *) buf.data_ptr();
  uint64_t* signal_start_addr = (uint64_t*) singals.data_ptr();
  size_t perchunksize = buf.numel() * buf.element_size()/num_ranks;
  at::cuda::CUDAStream cur_stream = at::cuda::getCurrentCUDAStream();
  int cur_rank, recv_rank, dst_PE;
  char* src_ptr;
  uint64_t* send_sig_addr, *recv_sig_addr;
  uint64_t signal_reset = 0;
  uint64_t sigval = 1;


  dst_PE =  global_ranks.at((my_rank + 1) % num_ranks);

  for (int idx =0 ; idx < num_ranks-1; idx ++ ){
    cur_rank = (my_rank - idx + num_ranks) % num_ranks;
    recv_rank = (my_rank - idx - 1 + num_ranks)% num_ranks;
    src_ptr = buf_start_ptr + cur_rank*perchunksize;
    send_sig_addr = signal_start_addr + cur_rank;
    nvshmemx_putmem_signal_on_stream((void *)src_ptr, (void *)src_ptr, perchunksize, send_sig_addr, sigval, NVSHMEM_SIGNAL_SET, dst_PE, (cudaStream_t)cur_stream);
    recv_sig_addr = signal_start_addr + recv_rank;
    cuStreamWaitValue64((CUstream)cur_stream, (CUdeviceptr)recv_sig_addr, (cuuint64_t)sigval, CU_STREAM_WAIT_VALUE_GEQ);
    cuStreamWriteValue64((CUstream)cur_stream, (CUdeviceptr)recv_sig_addr, (cuuint64_t)signal_reset, CU_STREAM_WRITE_VALUE_DEFAULT);
  }
}

}