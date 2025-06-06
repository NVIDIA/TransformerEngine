/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_USERBUFFERS_H_
#define TRANSFORMER_ENGINE_USERBUFFERS_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>

#include <chrono>
#include <functional>
#include <stdexcept>

#include "common/util/logging.h"

#ifdef NVTE_UB_WITH_MPI
#include <mpi.h>
#define ExtComm MPI_Comm
#else
#define ExtComm const char *
#endif

using ExtAllgatherOp = std::function<void(void *, size_t, void *, size_t, ExtComm)>;
using ExtBarrierOp = std::function<void(ExtComm)>;

#define NVTE_MAX_REGIONS 16
#define NVTE_MAX_SMS 32
#define NVTE_MAX_OPS 32
#define NVTE_MAX_PEERS 8192
#define NVTE_MAX_REQUESTS 1024
#define NVTE_LAUNCH_GPU 1
#define NVTE_LAUNCH_CPU 2
#define NVTE_MAX_NVLINK 32

#define NVTE_UB_MEM_UC_CONTIG 1
#define NVTE_UB_MEM_MC_CREATED 2
#define NVTE_UB_MEM_ALLOCATED 4

// region 0 flag offsets
#define NVTE_REG0_OPFLAGS 1024
#define NVTE_REG0_RECV (NVTE_REG0_OPFLAGS * userbuffers_op_types)
#define NVTE_REG0_SINGLENODE (2 * NVTE_MAX_NVLINK * NVTE_MAX_SMS + NVTE_MAX_OPS)
#define NVTE_REG0_OFFSET(comm) \
  ((2 * NVTE_MAX_REGIONS) * NVTE_MAX_NVLINK + NVTE_REG0_SINGLENODE * 2 + NVTE_MAX_PEERS)
#define NVTE_REG0_COMMBUFFER 0
// x3 for [flagptr, ce_start_ptr, ce_end_ptr]
#define NVTE_REG0_FLAGS (NVTE_REG0_RECV + NVTE_MAX_PEERS * NVTE_MAX_REGIONS * 3)
#define NVTE_REG0_IBRS 32
#define NVTE_REG0_IBAG 512

#if defined(UCP) || !defined(NOSHARP)
#undef REG0_COMMBUFFER
#define REG0_COMMBUFFER (1024 * 1024 * 16)
#endif
// gpuflags map offsets
#define NVTE_GF_STATE 16000
#define NVTE_GF_IBSHARPDONE 0
#define NVTE_HF_NVRSDONE (userbuffers_op_types + 1)
#define NVTE_HF_NVREDUCEDONE (userbuffers_op_types + 3)
#define NVTE_MAX_SHARP 16

typedef struct ub_request {
  int optype;
  int blocksize;
  int basecounter;
  int elements;
  int handler;
  int handler2;
  size_t offset;
  size_t offset2;
  int peer;
  // ----execution states
  int active, maxcredit;
  int nblock, numblocks, unconfirmed_ib_in_flight;
} ub_request;

enum req_type {
  userbuffers_allreduceop_sharp,
  userbuffers_sendop,
  userbuffers_allreduceop_nonsharp,
  userbuffers_allreduceop_nonsharp2,
  userbuffers_alltoall,
  userbuffers_op_types
};

struct communicator {
  int myrank, nranks;  // global job communicator
  int nvrank, nvsize;  // single node comm_intra
  int free_region;

  int launch_mode;

  void *gpu_ptrs;
  int sms, threads;
  int use_rr_kernel;  // Whether to use RR (or RW) for NVLink-only kernel
  int cga_size;
  int push, use_ce;

  void *mem_ptr[NVTE_MAX_REGIONS];
  void **peer_ptr[NVTE_MAX_REGIONS];

  int memflags[NVTE_MAX_REGIONS];  // UC,MC, user/lib allocated

  CUmemGenericAllocationHandle *uchandles[NVTE_MAX_REGIONS];
  void *ucbase_ptr[NVTE_MAX_REGIONS];  // only for cuMem allocated memory
  size_t uc_offsets[NVTE_MAX_REGIONS];
  size_t mem_size[NVTE_MAX_REGIONS];
  bool mem_dealloc[NVTE_MAX_REGIONS];

  void *mc_ptr[NVTE_MAX_REGIONS];
  void *mc_baseptr;
  CUmemGenericAllocationHandle mc_handle;
  size_t mc_offset, mc_maxsize;
  int use_mc;  // 1: use MC if available, 0: override not to use MC

  int ar_nvsize, ar_firstgpu,
      ar_nvrank;  // number of gpus(and first gpu in a group) of gpus per node in reduction subgroup
                  // (_splitar init used) would be equal to (nvsize,0) for regular comm_create
  int ar2_nvsize, ar2_firstgpu, ar2_nvrank;  // with ar_nvsize as a step
  int sm_arch;
  int num_nodes, my_node;
  // max value for running block counters in hostflags
  int basecounter[userbuffers_op_types];  // NOLINT(*)

  int *flags_baseptr, *flags, *map_flags;

  void *mem_mr[NVTE_MAX_REGIONS];

  // Abstract communication callbacks to support external bootstrapping (e.g. DL frameworks)
  ExtAllgatherOp _allgather;
  ExtBarrierOp _barrier;

  ExtComm comm_world;
  ExtComm comm_intra;  // full intranode (all ndev GPUS)
#ifdef NVTE_UB_WITH_MPI
  MPI_Request mpihndl[NVTE_MAX_SHARP];
#endif

  int *send_id, *recv_id;
  int mydev;
  uint64_t ub_timeout;
};
typedef struct communicator communicator;

void producer(void *atomic_ptr, int chunk_i, cudaStream_t stream);
void consumer(void *atomic_ptr, int chunk_i, cudaStream_t stream);
void consumer_batch(void *atomic_ptr, int first_chunk_i, int num_chunks, cudaStream_t stream);
void reset_counters(void *atomic_ptr, int num_chunks, bool allgather, cudaStream_t stream);

/*  creates communicator, allocates all internal buffers if necessary */
int create_communicator_grouped2(communicator **comm, int myrank, int numranks, int mylocal,
                                 int numlocal, int mynode, int numnodes,
                                 ExtAllgatherOp ext_allgather, ExtBarrierOp ext_barrier,
                                 int pipegpus, int pipenodes, int tensorgpus, int tensornodes);

int create_communicator_grouped(communicator **comm, int myrank, int numranks, int mylocal,
                                int numlocal, int mynode, int numnodes,
                                ExtAllgatherOp ext_allgather, ExtBarrierOp ext_barrier,
                                int pipegpus, int pipenodes);

int create_communicator(communicator **comm, int myrank, int numranks, int mylocal, int numlocal,
                        int mynode, int numnodes, ExtAllgatherOp ext_allgather,
                        ExtBarrierOp ext_barrier);

int create_communicator_grouped2_mpi(communicator **comm, int pipegpus, int pipenodes,
                                     int tensorgpus, int tensornodes);

int create_communicator_grouped_mpi(communicator **comm, int pipegpus, int pipenodes);

int create_communicator_mpi(communicator **comm);

void destroy_communicator(communicator *comm);

void destroy_communicator_mpi(communicator *comm);

// int check_user_buffer_registration(void* gpubuff, int bytes, communicator* comm, size_t* offset);
/*
    local calls, doesnt communicate between peers
    returns handler if buffer is registered already, or -1 if not.
    returned offset is offset of gpubuff relative to buffer registered
*/

int register_user_buffer_collective(void **gpubuff, size_t bytes, communicator *comm, bool alloc);
/*  returns handler and registers buffers. assumed to be collective i.e. you use same groups and
   dont mix buffers for different operations returns -1 if cant register (too many preregistered
   regions already) if alloc==true will allocate memory and fill the pointers (required for NVL
   SHARP and NSO/MNNVL)
*/

// for TP-parallelism, only single node is implemented
void allgather2_userbuff_inplace(const int handler, const int offset, const int elements,
                                 communicator *comm, cudaStream_t stream = 0,
                                 cudaEvent_t comm_launch_event = 0);
/*
each Rank input is
allgather2_userbuff_inplace: offset+myrank*elements
allgather2_userbuff_inplace_sliced: offset+myrank*elements*nslices+slice_id*elements

equivalent codes would be:
for(int slice=0;slice<ncslices;slice++)
 allgather2_userbuff_inplace_sliced(hndl,offset, elements,comm,slice,nslices,stream);

 and

 allgather2_userbuff_inplace(hndl,offset, elements*nslices,comm,stream);
*/
void reducescatter2_userbuff_inplace(const int handler, const int offset, const int elements,
                                     communicator *comm, cudaStream_t stream = 0,
                                     cudaEvent_t comm_launch_event = 0);
void reducescatter2_userbuff(void *output, const int handler, const int offset, const int elements,
                             communicator *comm, cudaStream_t stream = 0,
                             cudaEvent_t comm_launch_event = 0);
void reducescatter2_userbuff_stridedoutput(void *output, const int handler, const int offset,
                                           const int rowelements, const int colelements,
                                           const int strideelements, communicator *comm,
                                           cudaStream_t stream = 0,
                                           cudaEvent_t comm_launch_event = 0);
template <typename fp8type>
void reducescatter2_userbuff_stridedoutput_fp8(void *output, float *scale, const int handler,
                                               const int offset, const int rowelements,
                                               const int colelements, const int strideelements,
                                               communicator *comm, cudaStream_t stream = 0,
                                               cudaEvent_t comm_launch_event = 0);
template <typename fp8type>
void reducescatter2_userbuff_fp8(void *output, float *scale, const int handler, const int offset,
                                 const int elements, communicator *comm, cudaStream_t stream = 0,
                                 cudaEvent_t comm_launch_event = 0);
template <typename fp8type>
void reducescatter2_userbuff_strided_atomic_fp8(void *output, float *scale, const int handler,
                                                const int offset, const int rowelements,
                                                const int colelements, const int strideelements_out,
                                                const int strideelements_in, const int numchunks,
                                                void *counters, communicator *comm,
                                                cudaStream_t stream = 0);
template <typename fp8type>
void reducescatter2_userbuff_strided_multiatomic_fp8(
    void *output, float *scale, const int handler, const int offset, const int rowelements,
    const int colelements, const int strideelements_out, const int strideelements_in,
    const int numchunks, void *counters, communicator *comm, cudaStream_t stream = 0);
void reducescatter2_userbuff_strided(void *output, const int handler, const int offset,
                                     const int rowelements, const int colelements,
                                     const int strideelements, communicator *comm,
                                     cudaStream_t stream = 0);
void reducescatter2_userbuff_strided_atomic(void *output, const int handler, const int offset,
                                            const int rowelements, const int colelements,
                                            const int strideelements, const int numchunks,
                                            void *counters, communicator *comm,
                                            cudaStream_t stream = 0);
void reducescatter2_userbuff_strided_multiatomic(void *output, const int handler, const int offset,
                                                 const int rowelements, const int colelements,
                                                 const int strideelements, const int numchunks,
                                                 void *counters, communicator *comm,
                                                 cudaStream_t stream = 0);
/* everything should be 16byte aligned = 8 elts aligned
output is strided: row starts separated by stride elements*/

/*  inplace allreduce: works only with buffers registered by previous call. offset should be same
 * for all peers */

// two matching pairs, intended to work as push from sender or pull by receiver
// either way signal is a write by sender meaning
// push model: data arrived and visible at receiver(barrier enforced)
// pull model: data ready to be pulled by receiver(no barrier needed)

void userbuffers_send(const int srchandler, const size_t srcoffset, const int dsthandler,
                      const size_t dstoffset, const size_t bytes, communicator *comm,
                      const int peer, cudaStream_t stream = 0);
void userbuffers_recv(const int srchandler, const size_t srcoffset, const int dsthandler,
                      const size_t dstoffset, const size_t bytes, communicator *comm,
                      const int peer, cudaStream_t stream = 0);
void userbuffers_sendrecv(const int srchandler, const int dsthandler, const size_t send_offset,
                          const size_t recv_offset, const size_t bytes, communicator *comm,
                          const int send_peer, const int recv_peer, cudaStream_t stream = 0);
void userbuffers_sendrecv_atomic(const int srchandler, const int dsthandler,
                                 const size_t send_offset, const size_t recv_offset,
                                 const size_t bytes, communicator *comm, const int send_peer,
                                 const int recv_peer, void *counters, cudaStream_t stream = 0);
void userbuffers_sendrecv_multiatomic(const int srchandler, const int dsthandler,
                                      const size_t send_offset, const size_t recv_offset,
                                      const size_t bytes, communicator *comm, const int send_peer,
                                      const int recv_peer, const int nchunks, void *counters,
                                      bool shuffle, cudaStream_t stream = 0);

// alltoall split send and recv to allow for overlap
// send kicks in sending data to the destination - invoke on same stream as data generation
// recv returns once data has received
// send and recv can be on different streams
void userbuffers_alltoall_send(const int srchandler, const size_t srcoffset, const int dsthandler,
                               const size_t dstoffset, const size_t bytes, communicator *comm,
                               cudaStream_t stream = 0);
void userbuffers_alltoall_recv(communicator *comm, cudaStream_t stream = 0);

// void unregister_user_buffer(int handler);

void destroy_communicator(communicator *comm);

template <typename fp8type>
void reduce_fp8_in_bf16_out(void *input, void *output, float *scale, int num_inputs, int input_size,
                            cudaStream_t stream);

void reduce_bf16(void *input, void *output, int num_inputs, int input_size, cudaStream_t stream);

#endif  // TRANSFORMER_ENGINE_USERBUFFERS_H_
