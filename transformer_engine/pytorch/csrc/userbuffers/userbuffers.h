/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_USERBUFFERS_H_
#define TRANSFORMER_ENGINE_USERBUFFERS_H_

#include <cuda.h>
#include <mpi.h>  // TODO (tym): Removing will remove PyT extension dependence on MPI
#include "cuda_runtime.h"
#include <pthread.h>
#include <chrono>
#include "gdrapi.h"
#include <stdexcept>

#define NVTE_MAX_REGIONS 16
#define NVTE_MAX_SMS 32
#define NVTE_MAX_OPS 32
#define NVTE_MAX_PEERS 8192
#define NVTE_MAX_REQUESTS 1024
#define NVTE_LAUNCH_GPU 1
#define NVTE_LAUNCH_CPU 2
#define NVTE_MAX_NVLINK 8

#define UB_MEM_UC_CONTIG 1
#define UB_MEM_MC_CREATED 2
#define UB_MEM_ALLOCATED 4

#define NVTE_UB_MEM_UC_CONTIG 1
#define NVTE_UB_MEM_MC_CREATED 2
#define NVTE_UB_MEM_ALLOCATED 4

#ifdef UCP
#include <ucp/api/ucp.h>
#endif

// region 0 flag offsets
#define NVTE_REG0_OPFLAGS 1024
#define NVTE_REG0_RECV (NVTE_REG0_OPFLAGS * userbuffers_op_types)
#define NVTE_REG0_SINGLENODE (2 * NVTE_MAX_NVLINK * NVTE_MAX_SMS + NVTE_MAX_OPS)
#define NVTE_REG0_OFFSET(comm) ((2 * NVTE_MAX_REGIONS) * NVTE_MAX_NVLINK \
                                 + NVTE_REG0_SINGLENODE * 2 + NVTE_MAX_PEERS)
#define NVTE_REG0_COMMBUFFER 0
#define NVTE_REG0_FLAGS (NVTE_REG0_RECV + NVTE_MAX_PEERS * NVTE_MAX_REGIONS)
#define NVTE_REG0_IBRS 32
#define NVTE_REG0_IBAG 512

#if defined(UCP) || !defined(NOSHARP)
#undef REG0_COMMBUFFER
#define REG0_COMMBUFFER (1024*1024*16)
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
  void* ucbase_ptr[NVTE_MAX_REGIONS];  // only for cuMem allocated memory
  size_t mem_size[NVTE_MAX_REGIONS];

  void* mc_ptr[NVTE_MAX_REGIONS];
  void* mc_baseptr;
  CUmemGenericAllocationHandle mc_handle;
  size_t mc_offset, mc_maxsize;
  int use_mc;  // 1: use MC if available, 0: override not to use MC

  int ar_nvsize, ar_firstgpu,
      ar_nvrank;  // number of gpus(and first gpu in a group) of gpus per node in reduction subgroup
                  // (_splitar init used) would be equal to (nvsize,0) for regular comm_create
  int ar2_nvsize, ar2_firstgpu, ar2_nvrank;  // with ar_nvsize as a step
  int pipe_id;  // which allreduce set of groups (pipeline rank in range of 0..pipeline_size)
  int sm_arch;
  int num_nodes, my_node,
      first_node;  // comm_inter communicator, per-rail allreduce (might have subset of nodes)
  int num2_nodes, my2_node, first2_node;  // with num_nodes as a stride
  // max value for running block counters in hostflags
  int basecounter[userbuffers_op_types];  // NOLINT(*)

  int *hostflags;
  int *flags, *map_flags;
  gdr_t g;

  struct sharp_coll_context *sharp_coll_context;
  struct sharp_coll_comm *sharp_coll_comm;
  void *mem_mr[NVTE_MAX_REGIONS];

  ub_request *fifo;
  volatile int activeproxy;
  int nblocks, alignblock, minblock, asyncblocks, active_nreqs;
  ub_request active_req[userbuffers_op_types];  // NOLINT(*)
  int padding[7];
  volatile int head;
  int padding2[15];
  volatile int tail;

  MPI_Request mpihndl[NVTE_MAX_SHARP];
  MPI_Comm comm_inter,  // reduction group communicator (subset of the nodes) along GPU rail
      comm_intra;       // full intranode (all ndev GPUS)
  int ibnvsize;  // can be used to fake smaller or larger nvlink domain to use ib instead of nvlink
                 // or force MNNVL
  int *send_id, *recv_id;
  int mydev;
};
typedef struct communicator communicator;

void producer(void *atomic_ptr, int chunk_i, cudaStream_t stream);
void consumer(void *atomic_ptr, int chunk_i, cudaStream_t stream);
void consumer_batch(void *atomic_ptr, int first_chunk_i, int num_chunks, cudaStream_t stream);
int create_communicator(communicator **comm);
/*  creates communicator, allocates all internal buffers if necessary */

int create_communicator_grouped(communicator **comm, int pipegpus, int pipenodes);
int create_communicator_grouped2(communicator **comm, int pipegpus, int pipenodes, int tensorgpus,
                                 int tensornodes);
/*  creates communicator with
    allreduce1 to happen in datagpus x datanodes groups,
    allreduce2 to happen in tensorgpus x tensor nodes,
        where num_nodes = pipenodes x tensornodes x datanodes
            nvlink_size = pipegpus x tensorgpus x datagpus
 */

// int check_user_buffer_registration(void* gpubuff, int bytes, communicator* comm, size_t* offset);
/*
    local calls, doesnt communicate between peers
    returns handler if buffer is registered already, or -1 if not.
    returned offset is offset of gpubuff relative to buffer registered
*/

int pipe_rank(communicator *comm,
              int step);  // helper function to help walk across allreduce1 x allreduce2 groups
                          // data-parallel and tensor-parallel position within data and tensor
                          // groups would be preserved

int register_user_buffer_collective(void **gpubuff, size_t bytes, communicator *comm,
                                    bool alloc = false);
/*  returns handler and registers buffers. assumed to be collective i.e. you use same groups and
   dont mix buffers for different operations returns -1 if cant register (too many preregistered
   regions already) if alloc==true will allocate memory and fill the pointers (required for NVL
   SHARP and NSO/MNNVL)
*/

void allreduce_userbuff_inplace(const int handler, const int offset, const int elements,
                                communicator *comm, cudaStream_t stream = 0);
// for DP distributed optimizer, only nonSHARP multinode is implemented & calls must come in pairs
// ordered
void allgather_userbuff_inplace(const int handler, const int offset, const int elements,
                                communicator *comm, cudaStream_t stream = 0);
void reducescatter_userbuff_inplace(const int handler, const int offset, const int elements,
                                    communicator *comm, cudaStream_t stream = 0);

void allreduce2_userbuff_inplace(const int handler, const int offset, const int elements,
                                 communicator *comm, cudaStream_t stream = 0);
// for TP-parallelism, only single node is implemented
void allgather2_userbuff_inplace(const int handler, const int offset, const int elements,
                                 communicator *comm, cudaStream_t stream = 0);
void allgather2_userbuff_inplace_sliced(const int handler, const int offset, const int elements,
                                        communicator *comm, const int slice_id, const int nslices,
                                        cudaStream_t stream = 0);
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
                                     communicator *comm, cudaStream_t stream = 0);
void reducescatter2_userbuff(void *output, const int handler, const int offset, const int elements,
                             communicator *comm, cudaStream_t stream = 0);
void reducescatter2_userbuff_stridedoutput(void *output, const int handler, const int offset,
                                           const int rowelements, const int colelements,
                                           const int strideelements, communicator *comm,
                                           cudaStream_t stream = 0);
template<typename fp8type>
void reducescatter2_userbuff_stridedoutput_fp8(void* output, float* scale, const int handler,
                                               const int offset, const int rowelements,
                                               const int colelements, const int strideelements,
                                               communicator* comm, cudaStream_t stream = 0);
template<typename fp8type>
void reducescatter2_userbuff_fp8(void* output, float* scale, const int handler, const int offset,
                                 const int elements, communicator* comm, cudaStream_t stream = 0);
#if 0
template<typename fp8type>
void reducescatter2_userbuff_strided_atomic_fp8(void* output, float *scale, const int handler,
                                                const int offset, const int rowelements,
                                                const int colelements, const int strideelements,
                                                const int numchunks, void *counters,
                                                communicator* comm, cudaStream_t stream = 0);
#endif
template<typename fp8type>
void reducescatter2_userbuff_strided_atomic_fp8(void* output, float *scale, const int handler,
                                                const int offset, const int rowelements,
                                                const int colelements, const int strideelements_out,
                                                const int strideelements_in, const int numchunks,
                                                void *counters, communicator* comm,
                                                cudaStream_t stream = 0);
template<typename fp8type>
void reducescatter2_userbuff_strided_multiatomic_fp8(
  void* output, float *scale, const int handler, const int offset, const int rowelements,
  const int colelements, const int strideelements_out, const int strideelements_in,
  const int numchunks, void *counters, communicator* comm, cudaStream_t stream = 0);
void reducescatter2_userbuff_strided(
  void* output, const int handler, const int offset, const int rowelements, const int colelements,
  const int strideelements, communicator* comm, cudaStream_t stream = 0);
void reducescatter2_userbuff_strided_atomic(
  void* output, const int handler , const int offset, const int rowelements, const int colelements,
  const int strideelements, const int numchunks, void *counters, communicator* comm,
  cudaStream_t stream = 0);
void reducescatter2_userbuff_strided_multiatomic(
  void* output, const int handler, const int offset, const int rowelements, const int colelements,
  const int strideelements, const int numchunks, void *counters, communicator* comm,
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
void userbuffers_sendrecv(
  const int srchandler, const int dsthandler, const size_t send_offset, const size_t recv_offset,
  const size_t bytes, communicator* comm, const int send_peer, const int recv_peer,
  cudaStream_t stream = 0);
void userbuffers_sendrecv_atomic(
  const int srchandler, const int dsthandler, const size_t send_offset, const size_t recv_offset,
  const size_t bytes, communicator* comm, const int send_peer, const int recv_peer, void *counters,
  cudaStream_t stream = 0);
void userbuffers_sendrecv_multiatomic(
  const int srchandler, const int dsthandler, const size_t send_offset, const size_t recv_offset,
  const size_t bytes, communicator* comm, const int send_peer, const int recv_peer,
  const int nchunks, void *counters, bool shuffle, cudaStream_t stream = 0);


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
void reduce_fp8_in_bf16_out(void *input, void *output, float *scale, int num_inputs,
                            int input_size, cudaStream_t stream);

#endif  // TRANSFORMER_ENGINE_USERBUFFERS_H_
