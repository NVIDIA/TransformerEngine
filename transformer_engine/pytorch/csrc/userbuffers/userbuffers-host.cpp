/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <immintrin.h>
#include <math.h>
#include <mpi.h>
#include <sched.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <x86intrin.h>
#include <chrono>
#include <iostream>
#include "userbuffers.h"

static int oob_bcast(void *comm_context, void *buf, int size, int root) {
  MPI_Bcast(buf, size, MPI_BYTE, root,
            (reinterpret_cast<communicator *>(comm_context))->comm_inter);
  return 0;
}

static int oob_barrier(void *comm_context) {
  MPI_Barrier((reinterpret_cast<communicator *>(comm_context))->comm_inter);
  return 0;
}

static int oob_gather(void *comm_context, int root, void *sbuf, void *rbuf, int len) {
  MPI_Gather(sbuf, len, MPI_BYTE, rbuf, len, MPI_BYTE, root,
             (reinterpret_cast<communicator *>(comm_context))->comm_inter);
  return 0;
}

int stringCmp(const void *a, const void *b) { return strcmp((const char *)a, (const char *)b); }

#define CUDACHECK(cmd)                                                                      \
  do {                                                                                      \
    cudaError_t e = cmd;                                                                    \
    if (e != cudaSuccess) {                                                                 \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(EXIT_FAILURE);                                                                   \
    }                                                                                       \
  } while (0)

#define NVTE_UB_ERROR(x) \
    do { \
        throw std::runtime_error(std::string(__FILE__ ":") + std::to_string(__LINE__) +            \
                                 " in function " + __func__ + ": " + x);                           \
    } while (false)

int pipe_rank(communicator *comm, int step) {
  int mynode = comm->myrank / comm->nvsize;
  int mylocal = comm->nvrank;
  int numlocal = comm->nvsize;

  int newlocal1 = mylocal + step * comm->ar_nvsize * comm->ar2_nvsize;
  int newlocal = (numlocal + (newlocal1 % numlocal)) % numlocal;
  int newnode = mynode;
  newnode += (newlocal1 - newlocal) / numlocal * comm->num_nodes * comm->num2_nodes;
  int allnodes = comm->nranks / comm->nvsize;
  newnode = (allnodes + (newnode % allnodes)) % allnodes;
  return newnode * numlocal + newlocal;
}

int create_communicator_grouped2(communicator **comm, int pipegpus, int pipenodes, int tensorgpus,
                                 int tensornodes) {
  *comm = reinterpret_cast<communicator *>(malloc(sizeof(communicator)));

  int myrank, nranks, cur_dev, ndev;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  (*comm)->nranks = nranks;
  (*comm)->myrank = myrank;
  (*comm)->free_region = 0;
  (*comm)->launch_mode = NVTE_LAUNCH_GPU | NVTE_LAUNCH_CPU;

  cudaDeviceProp device_prop;
  CUDACHECK(cudaGetDevice(&cur_dev));
  CUDACHECK(cudaGetDeviceCount(&ndev));
  CUDACHECK(cudaGetDeviceProperties(&device_prop, cur_dev));
  (*comm)->sm_arch = device_prop.major;
  // (*comm)->use_rr_kernel = device_prop.major == 8;
  (*comm)->use_rr_kernel = 0;
  (*comm)->push = 1;
  (*comm)->use_ce = 0;
  (*comm)->cga_size = 2;
  for (int i = 0; i < userbuffers_op_types; i++) (*comm)->basecounter[i] = 0;
  (*comm)->head = 0;
  (*comm)->tail = 0;
  (*comm)->activeproxy = 1;
  (*comm)->active_nreqs = 0;
  for (int i = 0; i < userbuffers_op_types; i++) (*comm)->active_req[i].active = -1;

  int ret = 0;
  // split communicator
  char host_name[MPI_MAX_PROCESSOR_NAME];
  char(*host_names)[MPI_MAX_PROCESSOR_NAME];
  int namelen, bytes, color, my_node, mylocal, numlocal, num_nodes;
  int rank = (*comm)->myrank, size = (*comm)->nranks;
  MPI_Get_processor_name(host_name, &namelen);
  bytes = size * sizeof(char[MPI_MAX_PROCESSOR_NAME]);
  host_names = (char(*)[MPI_MAX_PROCESSOR_NAME])malloc(bytes);
  strcpy(host_names[rank], host_name);  // NOLINT(*)
  for (int n = 0; n < size; n++)
    MPI_Bcast(&(host_names[n]), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, n, MPI_COMM_WORLD);
  qsort(host_names, size, sizeof(char[MPI_MAX_PROCESSOR_NAME]), stringCmp);

  color = 0;
  for (int n = 0; n < size; n++) {
    if (n > 0 && strcmp(host_names[n - 1], host_names[n])) color++;
    if (strcmp(host_name, host_names[n]) == 0) break;
  }
  free(host_names);

  MPI_Comm_split(MPI_COMM_WORLD, color, rank, &(*comm)->comm_intra);
  // find intranode numbers and make internode communicator
  // figure out mylocal
  MPI_Comm_rank((*comm)->comm_intra, &mylocal);
  MPI_Comm_size((*comm)->comm_intra, &numlocal);
  (*comm)->nvrank = mylocal;
  (*comm)->nvsize = numlocal;

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  int core;
  if (mylocal == 0) core = 50;
  if (mylocal == 1) core = 58;
  if (mylocal == 2) core = 18;
  if (mylocal == 3) core = 26;
  if (mylocal == 4) core = 114;
  if (mylocal == 5) core = 122;
  if (mylocal == 6) core = 82;
  if (mylocal == 7) core = 90;

  CPU_SET(core, &cpuset);
  if (!getenv("NVTE_NODOUBLE")) {
    if (core > 128)
      CPU_SET(core - 128, &cpuset);
    else
      CPU_SET(core + 128, &cpuset);
  }
  if (getenv("NVTE_DOPIN")) pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

  if (ndev == numlocal) {  // all visible devices
    if (cur_dev != mylocal)
      printf("%d: device used %d[%d] ,resetting device to %d\n", rank, cur_dev, ndev, mylocal);
    CUDACHECK(cudaSetDevice(mylocal));
  }
  (*comm)->mydev = cur_dev;
  // FIXME need to check that numlocal is multiple of pipegpus x tensorgpus
  // ar1 is data
  int divgpus = pipegpus * tensorgpus;
  int datagpus = numlocal / divgpus;
  (*comm)->ar_nvsize = datagpus;
  (*comm)->ar_firstgpu = mylocal - ((mylocal / tensorgpus) % datagpus) * tensorgpus;
  (*comm)->ar_nvrank = (mylocal - (*comm)->ar_firstgpu) / tensorgpus;
  // ar2 is tensor
  (*comm)->ar2_nvsize = tensorgpus;
  (*comm)->ar2_firstgpu = mylocal - mylocal % tensorgpus;
  (*comm)->ar2_nvrank = mylocal - (*comm)->ar2_firstgpu;
  // ar2 has step equal to ar_nvsize
  int allnodes = nranks / numlocal;
  int mynode = myrank / numlocal;
  int datanodes = allnodes / pipenodes / tensornodes;
  int pipenodegroup_id = myrank / numlocal / (datanodes * tensornodes);

  (*comm)->pipe_id = pipegpus * pipenodegroup_id + mylocal / (datagpus * tensorgpus);

  CUDACHECK(cudaFree(0));
  int datanodegroup_id =
      myrank / numlocal / datanodes;  // data reduction group node belongs, equals 0 for all if both
                                      // pipenodes=1 and tensornodes=1
  // mpi communicator only needed for SHARP which is always allreduce1/data-parallel
  MPI_Comm_split(MPI_COMM_WORLD, mylocal + numlocal * datanodegroup_id, rank, &(*comm)->comm_inter);
  // different rails from same group are in different subcommunicators

  MPI_Comm_size((*comm)->comm_inter, &num_nodes);
  MPI_Comm_rank((*comm)->comm_inter, &my_node);
  (*comm)->first_node = mynode - my_node;
  (*comm)->num_nodes = num_nodes;
  (*comm)->my_node = my_node;

  (*comm)->num2_nodes = tensornodes;
  (*comm)->my2_node = (mynode / datanodes) % tensornodes;
  (*comm)->first2_node = mynode - (*comm)->my2_node * datanodes;

  char *ib_dev_list;
  int ZIONROCE = getenv("NVTE_ZIONROCE") ? atoi(getenv("NVTE_ZIONROCE")) : 0;
  int ROCE = getenv("NVTE_ROCE") ? atoi(getenv("NVTE_ROCE")) : 0;
  if (ZIONROCE) ROCE = 1;
  int DGX_H100 = device_prop.major == 9;

  switch (mylocal) {
    case 0:ib_dev_list = "mlx5_0:1"; break;  // NOLINT(*)
    case 1:ib_dev_list = (char*)(DGX_H100?"mlx5_3:1":"mlx5_1:1"); break;  // NOLINT(*)
    case 2:ib_dev_list = (char*)(ZIONROCE?"mlx5_4:1":DGX_H100?"mlx5_4:1":"mlx5_2:1"); break;  // NOLINT(*)
    case 3:ib_dev_list = (char*)(DGX_H100?"mlx5_5:1":"mlx5_3:1"); break;  // NOLINT(*)
    case 4:ib_dev_list = (char*)(DGX_H100?"mlx5_6:1":"mlx5_6:1"); break;  // NOLINT(*)
    case 5:ib_dev_list = (char*)(DGX_H100?"mlx5_9:1":"mlx5_7:1"); break;  // NOLINT(*)
    case 6:ib_dev_list = (char*)(ZIONROCE?"mlx5_10:1":DGX_H100?"mlx5_10:1":"mlx5_8:1"); break;  // NOLINT(*)
    case 7:ib_dev_list = (char*)(DGX_H100?"mlx5_11:1":"mlx5_9:1"); break;  // NOLINT(*)
    default: break;
  }

  (*comm)->fifo = reinterpret_cast<ub_request *>(malloc(sizeof(ub_request) * NVTE_MAX_REQUESTS));
  (*comm)->nblocks = 8;
  (*comm)->alignblock = 1024 * 512;
  (*comm)->minblock = 1024 * 2 * 1024;
  (*comm)->asyncblocks = 16;

  CUDACHECK(cudaMallocHost((void **)&(*comm)->hostflags,  // NOLINT(*)
                           (NVTE_MAX_SMS + 100) * sizeof(int)));
  for (int i = 0; i < 100 + NVTE_MAX_SMS; i++) (*comm)->hostflags[i] = 0;
  _mm_mfence();
  sleep(1);

  // init_p2p_transport();
  (*comm)->ibnvsize = (*comm)->nvsize;

#define NBUF 2
#define LOCALSIZE 4 * (NVTE_REG0_OFFSET(*comm) + NVTE_REG0_FLAGS + NVTE_REG0_COMMBUFFER * NBUF)
  // peer pointers + op flags + comm buffer

  CUDACHECK(cudaMalloc(&(*comm)->gpu_ptrs, LOCALSIZE));  // flags and pointers, no block data yet
  CUDACHECK(cudaMemset((*comm)->gpu_ptrs, 0, LOCALSIZE));
  CUDACHECK(cudaDeviceSynchronize());
  register_user_buffer_collective(&((*comm)->gpu_ptrs), LOCALSIZE, *comm);  // will use handler 0
  CUDACHECK(cudaMalloc(&(*comm)->send_id, (*comm)->nranks * sizeof(int)));
  CUDACHECK(cudaMalloc(&(*comm)->recv_id, NVTE_MAX_REGIONS * (*comm)->nranks * sizeof(int)));
  CUDACHECK(cudaMemset((*comm)->send_id, 0, (*comm)->nranks * sizeof(int)));
  CUDACHECK(cudaMemset((*comm)->recv_id, 0, NVTE_MAX_REGIONS * (*comm)->nranks * sizeof(int)));
  (*comm)->sms = 16;
  (*comm)->threads = 1024;

#define GPU_PAGE_SHIFT 16
#define GPU_PAGE_SIZE (1UL << GPU_PAGE_SHIFT)
#define GPU_PAGE_OFFSET (GPU_PAGE_SIZE - 1)
#define GPU_PAGE_MASK (~GPU_PAGE_OFFSET)
  CUDACHECK(cudaMalloc(&(*comm)->flags, 2 * GPU_PAGE_SIZE));
  unsigned int flag = 1;
  // cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr)(*comm)->flags);
  CUDACHECK(cudaMemset((*comm)->flags, 0, 2 * GPU_PAGE_SIZE));
  (*comm)->flags =
      reinterpret_cast<int *>(((CUdeviceptr)(*comm)->flags + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK);

  using namespace std;
  (*comm)->g = gdr_open();
  if ((*comm)->g == NULL) {
    fprintf(stderr, "gdrcopy open failed\n");
    return -1;
  }
  gdr_mh_t mh;
  ret = gdr_pin_buffer((*comm)->g, (CUdeviceptr)(*comm)->flags, GPU_PAGE_SIZE, 0, 0, &mh);
  if (ret) {
    fprintf(stderr, "gdr_pin_buffer failed\n");
    return -1;
  }
  ret = gdr_map((*comm)->g, mh, (void **)&((*comm)->map_flags), GPU_PAGE_SIZE);  // NOLINT(*)

  if (ret) {
    fprintf(stderr, "gdr_map failed\n");
    return -1;
  }
  sched_param param;
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_getschedparam(&attr, &param);
  param.sched_priority = sched_get_priority_max(SCHED_FIFO);

  pthread_attr_setschedparam(&attr, &param);

  if (getenv("NVTE_UBDEBUG"))
    printf("%d/%d:(%d x %d): DP %d x %d TP %d x %d, DPGROUP %dx%d TPGROUP %dx%d PIPE_ID %d/%d\n",
           myrank, nranks, myrank / numlocal, myrank % numlocal, (*comm)->my_node,
           (*comm)->ar_nvrank, (*comm)->my2_node, (*comm)->ar2_nvrank, (*comm)->num_nodes,
           (*comm)->ar_nvsize, (*comm)->num2_nodes, (*comm)->ar2_nvsize, (*comm)->pipe_id,
           pipegpus * pipenodes);
  fflush(NULL);

  return 0;
}
int create_communicator_grouped(communicator **comm, int pipegpus, int pipenodes) {
  return create_communicator_grouped2(comm, pipegpus, pipenodes, 1, 1);
}

int create_communicator(communicator **comm) {
  return create_communicator_grouped2(comm, 1, 1, 1, 1);
}

void destroy_communicator(communicator *comm) {
  comm->activeproxy = 0;
  if (!comm->myrank && getenv("NVTE_UBDEBUG"))
    printf("waiting for userbuffers proxy thread to exit()\n");
  gdr_close(comm->g);
}

int register_user_buffer_collective(void **gpubuff, size_t bytes, communicator *comm, bool alloc) {
  if (comm->free_region > NVTE_MAX_REGIONS) return -1;
  int hndl = comm->free_region;
  // printf("%d register %d size %lld\n",comm->myrank,hndl,bytes);fflush(NULL);
  comm->peer_ptr[hndl] = reinterpret_cast<void **>(malloc(sizeof(void *) * (comm->nvsize)));

  if (alloc) {
    CUDACHECK(cudaMalloc(gpubuff, bytes));
  }
  assert(comm->nvsize <= 8);
  cudaIpcMemHandle_t *memhndl =
      reinterpret_cast<cudaIpcMemHandle_t *>(malloc(sizeof(cudaIpcMemHandle_t) * (comm->nvsize)));

  CUDACHECK(cudaIpcGetMemHandle(&memhndl[comm->nvrank], *gpubuff));

  MPI_Allgather(&memhndl[comm->nvrank], sizeof(cudaIpcMemHandle_t), MPI_BYTE, memhndl,
                sizeof(cudaIpcMemHandle_t), MPI_BYTE, comm->comm_intra);

  for (int i = 0; i < comm->nvsize; i++)
    if (i != comm->nvrank)
      CUDACHECK(cudaIpcOpenMemHandle((void **)&(comm->peer_ptr[hndl][i]),  // NOLINT(*)
                                     memhndl[i], cudaIpcMemLazyEnablePeerAccess));
  comm->peer_ptr[hndl][comm->nvrank] = *gpubuff;
  CUDACHECK(cudaDeviceSynchronize());

  CUDACHECK(
      cudaMemcpy(reinterpret_cast<char *>(comm->gpu_ptrs) + (hndl * comm->nvsize * sizeof(void *)),
                 comm->peer_ptr[hndl], comm->nvsize * sizeof(void *), cudaMemcpyHostToDevice));

  CUDACHECK(cudaDeviceSynchronize());
  free(memhndl);

  comm->mem_ptr[hndl] = *gpubuff;
  return comm->free_region++;
}

int allreduce_userbuff_inplace_gpu(const int handler, const int offset, const int elements,
                                   const int blocksize, communicator *comm, cudaStream_t stream);

int allreduce2_userbuff_inplace_gpu(const int maxcredit, const int handler, const int offset,
                                    const int elements, const int blocksize, communicator *comm,
                                    cudaStream_t stream, int op);

int reducescatter2_userbuff_inplace_gpu(const int maxcredit, const int handler, const int offset,
                                        const int elements, const int blocksize, communicator *comm,
                                        cudaStream_t stream, int op);

int allgather2_userbuff_inplace_gpu(const int maxcredit, const int handler, const int offset,
                                    const int elements, const int blocksize, communicator *comm,
                                    cudaStream_t stream, int op);

void allreduce_nonsharp_inplace(const int handler, const int offset, const int elements,
                                communicator *comm, cudaStream_t stream, int op) {
  if (elements < 64) NVTE_UB_ERROR("Userbuffer comm for given config not implemented.");
  // if(comm->myrank==0) fprintf(stderr,"AR2(%d) user call launch_mode=%d\n",op,comm->launch_mode);
  const int ar_nvsize = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvsize : comm->ar2_nvsize;
  int blocksize = elements * 2;
  int maxcredit = 0;
  const int num_nodes = op == userbuffers_allreduceop_nonsharp ? comm->num_nodes : comm->num2_nodes;
  blocksize = (comm->nblocks - 1 + (comm->alignblock - 1 + elements * 2) / comm->alignblock) /
              comm->nblocks;  // FIXME TUNING
  blocksize *= comm->alignblock;
  if (blocksize < comm->minblock) blocksize = comm->minblock;

  maxcredit = (elements * 2 + blocksize - 1) / blocksize;
  // if(maxcredit>4) maxcredit=4;
  // if(maxcredit>4 && ar_nvsize==1) maxcredit=4;
  size_t peerblock = sizeof(int) * NVTE_REG0_COMMBUFFER / maxcredit;  // max size we can fit
  if (blocksize > peerblock * ar_nvsize) blocksize = peerblock * ar_nvsize;
  // blocksize=elements*2;
  int sms = allreduce2_userbuff_inplace_gpu(maxcredit, handler, offset, elements, blocksize, comm,
                                            stream, op);

  if (num_nodes > 1 && comm->launch_mode & NVTE_LAUNCH_CPU) {
    if (!sms) return;
    comm->fifo[comm->head].optype = op;
    comm->fifo[comm->head].basecounter = comm->basecounter[op];
    comm->fifo[comm->head].blocksize = blocksize;
    comm->fifo[comm->head].maxcredit = maxcredit;
    comm->fifo[comm->head].handler = handler;
    comm->fifo[comm->head].offset = offset;
    comm->fifo[comm->head].elements = elements;

    int newhead = (comm->head + 1) & (NVTE_MAX_REQUESTS - 1);
    while (newhead == comm->tail) {
    }
    comm->head = newhead;

    comm->basecounter[op] += (elements * 2 + blocksize - 1) / blocksize;
  }
}

void allreduce2_userbuff_inplace(const int handler, const int offset, const int elements,
                                 communicator *comm, cudaStream_t stream) {
  allreduce_nonsharp_inplace(handler, offset, elements, comm, stream,
                             userbuffers_allreduceop_nonsharp2);
}

void allreduce_userbuff_inplace(const int handler, const int offset, const int elements,
                                communicator *comm, cudaStream_t stream) {
  if (elements < 64) NVTE_UB_ERROR("Userbuffer comm for given config not implemented.");
  allreduce_nonsharp_inplace(handler, offset, elements, comm, stream,
                             userbuffers_allreduceop_nonsharp);
  return;
}

void reducescatter_userbuff_inplace(const int handler, const int offset, const int elements,
                                    communicator *comm, cudaStream_t stream) {
  if (elements < 64) NVTE_UB_ERROR("Userbuffer comm for given config not implemented.");

  int op = userbuffers_allreduceop_nonsharp;
  const int ar_nvsize = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvsize : comm->ar2_nvsize;
  int blocksize = elements * 2;
  int maxcredit = 0;

  const int num_nodes = op == userbuffers_allreduceop_nonsharp ? comm->num_nodes : comm->num2_nodes;
  blocksize = (comm->nblocks - 1 + (comm->alignblock - 1 + elements * 2) / comm->alignblock) /
              comm->nblocks;  // FIXME TUNING
  blocksize *= comm->alignblock;
  if (blocksize < comm->minblock) blocksize = comm->minblock;

  maxcredit = (elements * 2 + blocksize - 1) / blocksize;
  size_t peerblock = sizeof(int) * NVTE_REG0_COMMBUFFER / maxcredit;  // max size we can fit
  if (blocksize > peerblock * ar_nvsize) blocksize = peerblock * ar_nvsize;

  int sms = reducescatter2_userbuff_inplace_gpu(maxcredit, handler, offset, elements, blocksize,
                                                comm, stream, op);

  if (num_nodes > 1 && comm->launch_mode & NVTE_LAUNCH_CPU) {
    if (!sms) return;
    comm->fifo[comm->head].optype = op;
    comm->fifo[comm->head].basecounter = comm->basecounter[op];
    comm->fifo[comm->head].blocksize = blocksize;
    comm->fifo[comm->head].maxcredit = maxcredit;
    comm->fifo[comm->head].handler = handler;
    comm->fifo[comm->head].offset = offset;
    comm->fifo[comm->head].elements = elements;

    int newhead = (comm->head + 1) & (NVTE_MAX_REQUESTS - 1);
    while (newhead == comm->tail) {
    }
    comm->head = newhead;

    comm->basecounter[op] += (elements * 2 + blocksize - 1) / blocksize;
  }
}

void allgather_userbuff_inplace(const int handler, const int offset, const int elements,
                                communicator *comm, cudaStream_t stream) {
  if (elements < 64) NVTE_UB_ERROR("Userbuffer comm for given config not implemented.");
  int op = userbuffers_allreduceop_nonsharp;
  const int ar_nvsize = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvsize : comm->ar2_nvsize;
  int blocksize = elements * 2;
  int maxcredit = 0;

  const int num_nodes = op == userbuffers_allreduceop_nonsharp ? comm->num_nodes : comm->num2_nodes;
  blocksize = (comm->nblocks - 1 + (comm->alignblock - 1 + elements * 2) / comm->alignblock) /
              comm->nblocks;  // FIXME TUNING
  blocksize *= comm->alignblock;
  if (blocksize < comm->minblock) blocksize = comm->minblock;

  maxcredit = (elements * 2 + blocksize - 1) / blocksize;
  size_t peerblock = sizeof(int) * NVTE_REG0_COMMBUFFER / maxcredit;  // max size we can fit
  if (blocksize > peerblock * ar_nvsize) blocksize = peerblock * ar_nvsize;

  int sms = allgather2_userbuff_inplace_gpu(maxcredit, handler, offset, elements, blocksize, comm,
                                            stream, op);
}
