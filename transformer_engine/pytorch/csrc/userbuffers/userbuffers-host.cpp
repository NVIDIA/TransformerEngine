/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <inttypes.h>
#include <math.h>
#include <sched.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <chrono>
#include <iostream>
#include <map>
#include <utility>

#include "../util/cuda_driver.h"
#include "ipcsocket.h"
#include "userbuffers.h"

#ifdef UB_MPI_BOOTSTRAP
#include <mpi.h>
static MPI_Comm EXT_COMM_WORLD = MPI_COMM_WORLD;
static MPI_Comm EXT_COMM_INTRA;
static MPI_Comm EXT_COMM_INTER;
#else
static char EXT_COMM_WORLD[] = "world";
static char EXT_COMM_INTRA[] = "intra";
static char EXT_COMM_INTER[] = "inter";
#endif

#define MULTICAST_GB_TOTAL 512

int stringCmp(const void *a, const void *b) { return strcmp((const char *)a, (const char *)b); }

#define CUDACHECK(cmd)                                                                      \
  do {                                                                                      \
    cudaError_t e = cmd;                                                                    \
    if (e != cudaSuccess) {                                                                 \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(EXIT_FAILURE);                                                                   \
    }                                                                                       \
  } while (0)

#define NVTE_UB_ERROR(x)                                                            \
  do {                                                                              \
    throw std::runtime_error(std::string(__FILE__ ":") + std::to_string(__LINE__) + \
                             " in function " + __func__ + ": " + x);                \
  } while (false)
#define NCCLCHECK(cmd)                                                                        \
  do {                                                                                        \
    ncclResult_t r = cmd;                                                                     \
    if (r != ncclSuccess) {                                                                   \
      printf("Failed, NCCL error %s:%d ''\n", __FILE__, __LINE__ /*,ncclGetErrorString(r)*/); \
      exit(EXIT_FAILURE);                                                                     \
    }                                                                                         \
  } while (0)

#define NCCLCHECKGOTO(call, RES, label)                \
  do {                                                 \
    RES = call;                                        \
    if (RES != ncclSuccess && RES != ncclInProgress) { \
      goto label;                                      \
    }                                                  \
  } while (0);

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

int create_communicator_grouped2(
    communicator **comm, int myrank, int numranks, int mylocal, int numlocal, int mynode,
    int numnodes, std::function<void(void **, void *, size_t, ExtComm)> ext_alloc_copy_allgather,
    std::function<void(ExtComm)> ext_barrier, std::function<void(void *)> ext_free, int pipegpus,
    int pipenodes, int tensorgpus, int tensornodes) {
  *comm = new communicator();

  (*comm)->comm_world = EXT_COMM_WORLD;
  (*comm)->_alloc_copy_allgather = ext_alloc_copy_allgather;
  (*comm)->_barrier = ext_barrier;
  (*comm)->_free = ext_free;
  (*comm)->nranks = numranks;
  (*comm)->myrank = myrank;
  (*comm)->free_region = 0;
  (*comm)->launch_mode = NVTE_LAUNCH_GPU | NVTE_LAUNCH_CPU;

  int cur_dev, ndev;
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
  (*comm)->active_nreqs = 0;
  for (int i = 0; i < userbuffers_op_types; i++) (*comm)->active_req[i].active = -1;

  int device_clock = 0;
  // 110 sec wait time by default
  int sec_timeout = getenv("UB_TIMEOUT") ? atoi(getenv("UB_TIMEOUT")) : 110;
  CUDACHECK(cudaDeviceGetAttribute(&device_clock, cudaDevAttrClockRate, cur_dev));
  (*comm)->ub_timeout = 1000ull * device_clock * sec_timeout;
  if ((*comm)->myrank == 0) {
    printf("UB_TIMEOUT is set to %d sec, %" PRIu64 " cycles, freq: %dkhz\n", sec_timeout,
           (*comm)->ub_timeout, device_clock);
  }

  (*comm)->comm_intra = EXT_COMM_INTRA;
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
      printf("%d: device used %d[%d] ,resetting device to %d\n", myrank, cur_dev, ndev, mylocal);
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
  int allnodes = numranks / numlocal;
  int nodeid = myrank / numlocal;
  int datanodes = allnodes / pipenodes / tensornodes;
  int pipenodegroup_id = myrank / numlocal / (datanodes * tensornodes);

  (*comm)->pipe_id = pipegpus * pipenodegroup_id + mylocal / (datagpus * tensorgpus);

  (*comm)->comm_inter = EXT_COMM_INTER;
  (*comm)->first_node = nodeid - mynode;
  (*comm)->num_nodes = numnodes;
  (*comm)->my_node = mynode;

  (*comm)->num2_nodes = tensornodes;
  (*comm)->my2_node = (mynode / datanodes) % tensornodes;
  (*comm)->first2_node = mynode - (*comm)->my2_node * datanodes;

  (*comm)->fifo = reinterpret_cast<ub_request *>(malloc(sizeof(ub_request) * NVTE_MAX_REQUESTS));
  (*comm)->nblocks = 8;
  (*comm)->alignblock = 1024 * 512;
  (*comm)->minblock = 1024 * 2 * 1024;
  (*comm)->asyncblocks = 16;

#define NBUF 2
  if ((*comm)->sm_arch >= 9 && (*comm)->ar2_nvsize > 1 &&
      !getenv("UB_SKIPMC")) {  // multicast init only for TP ops (____2 operations)
    size_t mc_maxsize = MULTICAST_GB_TOTAL * (1ull << 30);
    (*comm)->mc_offset = 0;
    (*comm)->use_mc = 1;
    size_t gran;
    CUmulticastObjectProp mcProp = {};
    mcProp.numDevices = (*comm)->ar2_nvsize;
    mcProp.size = (*comm)->mc_maxsize;
    mcProp.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    NVTE_CALL_CHECK_CUDA_DRIVER(
        cuMulticastGetGranularity, &gran, &mcProp,
        static_cast<CUmemAllocationGranularity_flags>(CU_MULTICAST_GRANULARITY_RECOMMENDED));
    mc_maxsize = ((mc_maxsize + gran - 1) / gran) * gran;
    mcProp.size = mc_maxsize;
    (*comm)->mc_maxsize = mc_maxsize;

    // Broadcast the a POSIX file descriptor from the local root rank to other local ranks.
    // NOTE: This cannot be done via MPI_Bcast or other external comm libraries. They mangle the
    //       file descriptor and prevent cuMemImportFromShareableHandle() from correctly
    //       interpreting the file. Instead, we use system socket to send/recv the file handle
    //       without mangling.
    int fd;
    volatile uint32_t abortFlag = 0;
    struct ncclIpcSocket ipcSock = {0};
    uint64_t opId = 0xdeadcafeb000 + (*comm)->ar2_firstgpu;
    ncclResult_t ret = ncclSuccess;
    NCCLCHECK(ncclIpcSocketInit(&ipcSock, (*comm)->ar2_nvrank, (uint64_t)opId, &abortFlag));
    (*comm)->_barrier((*comm)->comm_world);

    if ((*comm)->ar2_nvrank == 0) {
      NVTE_CALL_CHECK_CUDA_DRIVER(cuMulticastCreate, &(*comm)->mc_handle, &mcProp);
      NVTE_CALL_CHECK_CUDA_DRIVER(
          cuMemExportToShareableHandle, reinterpret_cast<void *>(&fd), (*comm)->mc_handle,
          static_cast<CUmemAllocationHandleType>(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
          (uint64_t)0);

      for (int p = 1; p < (*comm)->ar2_nvsize; p++) {
        (*comm)->_barrier((*comm)->comm_intra);
        NCCLCHECKGOTO(ncclIpcSocketSendFd(&ipcSock, fd, p, (uint64_t)opId), ret, error);
      }
    } else {
      for (int i = 0; i < (*comm)->ar2_nvrank; i++) (*comm)->_barrier((*comm)->comm_intra);
      NCCLCHECKGOTO(ncclIpcSocketRecvFd(&ipcSock, &fd), ret, error);
      for (int i = 0; i < (*comm)->ar2_nvsize - (*comm)->ar2_nvrank - 1; i++)
        (*comm)->_barrier((*comm)->comm_intra);
      NVTE_CALL_CHECK_CUDA_DRIVER(
          cuMemImportFromShareableHandle, &(*comm)->mc_handle, reinterpret_cast<void *>(fd),
          static_cast<CUmemAllocationHandleType>(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
    }
  error:
    NCCLCHECK(ncclIpcSocketClose(&ipcSock));
    close(fd);
    NVTE_CALL_CHECK_CUDA_DRIVER(cuMulticastAddDevice, (*comm)->mc_handle,
                                (CUdeviceptr)(*comm)->mydev);

    CUdeviceptr mc_va;
    NVTE_CALL_CHECK_CUDA_DRIVER(cuMemAddressReserve, &mc_va, mc_maxsize, (size_t)0, (CUdeviceptr)0U,
                                (uint64_t)0);
    NVTE_CALL_CHECK_CUDA_DRIVER(cuMemMap, mc_va, mc_maxsize, (size_t)0, (*comm)->mc_handle,
                                (uint64_t)0);

    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = (*comm)->mydev;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    NVTE_CALL_CHECK_CUDA_DRIVER(cuMemSetAccess, mc_va, mc_maxsize,
                                const_cast<CUmemAccessDesc *>(&accessDesc), (size_t)1);

    (*comm)->mc_baseptr = reinterpret_cast<void *>(mc_va);
    (*comm)->_barrier((*comm)->comm_world);
    if (!(*comm)->myrank) printf("MC initialized succesfully, window size = %ld\n", mc_maxsize);
  } else {
    if (!(*comm)->myrank) printf("MC NOT initialized and used\n");
    (*comm)->mc_maxsize = 0;
    (*comm)->mc_offset = 0;
    (*comm)->use_mc = 0;
  }

#define LOCALSIZE 4 * (NVTE_REG0_OFFSET(*comm) + NVTE_REG0_FLAGS + NVTE_REG0_COMMBUFFER * NBUF)
  // peer pointers + op flags + comm buffer

  CUDACHECK(cudaMalloc(&(*comm)->gpu_ptrs, LOCALSIZE));  // flags and pointers, no block data yet
  CUDACHECK(cudaMemset((*comm)->gpu_ptrs, 0, LOCALSIZE));
  CUDACHECK(cudaDeviceSynchronize());
  register_user_buffer_collective(&((*comm)->gpu_ptrs), LOCALSIZE, *comm, false);
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
  CUDACHECK(cudaMemset((*comm)->flags, 0, 2 * GPU_PAGE_SIZE));
  (*comm)->flags =
      reinterpret_cast<int *>(((CUdeviceptr)(*comm)->flags + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK);

  using namespace std;

  sched_param param;
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_getschedparam(&attr, &param);
  param.sched_priority = sched_get_priority_max(SCHED_FIFO);

  pthread_attr_setschedparam(&attr, &param);

  if (getenv("NVTE_UBDEBUG"))
    printf(
        "%d/%d:(%d x %d): DP %d x %d TP %d x %d, DPGROUP %dx%d TPGROUP "
        "%dx%d PIPE_ID %d/%d\n",
        myrank, numranks, myrank / numlocal, myrank % numlocal, (*comm)->my_node,
        (*comm)->ar_nvrank, (*comm)->my2_node, (*comm)->ar2_nvrank, (*comm)->num_nodes,
        (*comm)->ar_nvsize, (*comm)->num2_nodes, (*comm)->ar2_nvsize, (*comm)->pipe_id,
        pipegpus * pipenodes);
  fflush(NULL);

  return 0;
}

int create_communicator_grouped(
    communicator **comm, int myrank, int numranks, int mylocal, int numlocal, int mynode,
    int numnodes, std::function<void(void **, void *, size_t, ExtComm)> ext_alloc_copy_allgather,
    std::function<void(ExtComm)> ext_barrier, std::function<void(void *)> ext_free, int pipegpus,
    int pipenodes) {
  return create_communicator_grouped2(comm, myrank, numranks, mylocal, numlocal, mynode, numnodes,
                                      ext_alloc_copy_allgather, ext_barrier, ext_free, pipegpus,
                                      pipenodes, 1, 1);
}

int create_communicator(
    communicator **comm, int myrank, int numranks, int mylocal, int numlocal, int mynode,
    int numnodes, std::function<void(void **, void *, size_t, ExtComm)> ext_alloc_copy_allgather,
    std::function<void(ExtComm)> ext_barrier, std::function<void(void *)> ext_free) {
  return create_communicator_grouped2(comm, myrank, numranks, mylocal, numlocal, mynode, numnodes,
                                      ext_alloc_copy_allgather, ext_barrier, ext_free, 1, 1, 1, 1);
}

int create_communicator_grouped2_mpi(communicator **comm, int pipegpus, int pipenodes,
                                     int tensorgpus, int tensornodes) {
#ifdef UB_MPI_BOOTSTRAP
  // get global numbers
  int myrank, numranks;
  MPI_Comm_rank(EXT_COMM_WORLD, &myrank);
  MPI_Comm_size(EXT_COMM_WORLD, &numranks);

  // find intranode numbers and make internode communicator
  char host_name[MPI_MAX_PROCESSOR_NAME];
  char(*host_names)[MPI_MAX_PROCESSOR_NAME];
  int namelen, bytes, color;
  int rank = (*comm)->myrank, size = (*comm)->nranks;
  MPI_Get_processor_name(host_name, &namelen);
  bytes = size * sizeof(char[MPI_MAX_PROCESSOR_NAME]);
  host_names = (char(*)[MPI_MAX_PROCESSOR_NAME])malloc(bytes);
  strcpy(host_names[rank], host_name);  // NOLINT(*)
  for (int n = 0; n < size; n++)
    MPI_Bcast(&(host_names[n]), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, n, EXT_COMM_WORLD);
  qsort(host_names, size, sizeof(char[MPI_MAX_PROCESSOR_NAME]), stringCmp);

  color = 0;
  for (int n = 0; n < size; n++) {
    if (n > 0 && strcmp(host_names[n - 1], host_names[n])) color++;
    if (strcmp(host_name, host_names[n]) == 0) break;
  }
  free(host_names);

  int mylocal, numlocal;
  MPI_Comm_split(EXT_COMM_WORLD, color, rank, &EXT_COMM_INTRA);
  MPI_Comm_rank(EXT_COMM_INTRA, &mylocal);
  MPI_Comm_size(EXT_COMM_INTRA, &numlocal);

  // find internode numbers and make internode communicator
  CUDACHECK(cudaFree(0));
  int allnodes = numranks / numlocal;
  int datanodes = allnodes / pipenodes / tensornodes;
  // data reduction group node belongs, equals 0 for all if both pipenodes=1 and tensornodes=1
  int datanodegroup_id = myrank / numlocal / datanodes;
  // mpi communicator only needed for SHARP which is always allreduce1/data-parallel
  MPI_Comm_split(EXT_COMM_WORLD, mylocal + numlocal * datanodegroup_id, rank, &EXT_COMM_INTER);
  // different rails from same group are in different subcommunicators
  int mynode, numnodes;
  MPI_Comm_size(EXT_COMM_INTER, &numnodes);
  MPI_Comm_rank(EXT_COMM_INTER, &mynode);

  // finally call the abstracted constructor with MPI info
  return create_communicator_grouped2(comm, myrank, numranks, mylocal, numlocal, mynode, numnodes,
                                      &ub_alloc_copy_allgather, &ub_barrier, &ub_free, pipegpus,
                                      pipenodes, tensorgpus, tensornodes);
#else
  NVTE_UB_ERROR(std::string("Bootstrapping Userbuffers with MPI requires ") +
                std::string("building Transformer Engine with UB_MPI_BOOTSTRAP=1"));
#endif
}

int create_communicator_grouped_mpi(communicator **comm, int pipegpus, int pipenodes) {
  return create_communicator_grouped2_mpi(comm, pipegpus, pipenodes, 1, 1);
}

int create_communicator_mpi(communicator **comm) {
  return create_communicator_grouped2_mpi(comm, 1, 1, 1, 1);
}

void destroy_communicator(communicator *comm) {
  for (int hndl = 0; hndl < comm->free_region; hndl++) {
    if (comm->mem_dealloc[hndl]) {
      NVTE_CALL_CHECK_CUDA_DRIVER(cuMemAddressFree,
                                  reinterpret_cast<CUdeviceptr>(comm->ucbase_ptr[hndl]),
                                  comm->mem_size[hndl] * comm->nvsize);
      for (int rank = 0; rank < comm->nvsize; rank++) {
        NVTE_CALL_CHECK_CUDA_DRIVER(cuMemRelease, comm->uchandles[hndl][rank]);
      }
      free(reinterpret_cast<void *>(comm->uchandles[hndl]));
    } else {
      for (int rank = 0; rank < comm->nvsize; rank++) {
        if (rank != comm->nvrank) {
          cudaIpcCloseMemHandle(comm->peer_ptr[hndl][rank]);
        } else {
          comm->peer_ptr[hndl][rank] = nullptr;  // remove reference to external buffer
        }
      }
      free(comm->peer_ptr[hndl]);
    }
    comm->mem_ptr[hndl] = nullptr;
  }
  cudaFree(reinterpret_cast<void *>(comm->flags));
  cudaFree(reinterpret_cast<void *>(comm->recv_id));
  cudaFree(reinterpret_cast<void *>(comm->send_id));
  if (comm->use_mc) {
    NVTE_CALL_CHECK_CUDA_DRIVER(cuMemAddressFree, reinterpret_cast<CUdeviceptr>(comm->mc_baseptr),
                                comm->mc_maxsize);
    NVTE_CALL_CHECK_CUDA_DRIVER(cuMemRelease, comm->mc_handle);
  }
  if (comm->mem_dealloc[0]) {
    cudaFree(comm->gpu_ptrs);
  }
  free(comm->fifo);
  delete comm;
}

void destroy_communicator_mpi(communicator *comm) {
#ifdef UB_MPI_BOOTSTRAP
  MPI_Comm_free(comm->comm_inter);
  MPI_Comm_free(comm->comm_intra);
  destroy_communicator(comm);
#else
  NVTE_UB_ERROR(std::string("Communicator is not bootstrapped with MPI and ") +
                std::string("can only be deallocated with destroy_communicator()."));
#endif
}

int register_user_buffer_collective(void **gpubuff, size_t bytes, communicator *comm, bool alloc) {
  if (comm->free_region > NVTE_MAX_REGIONS) return -1;
  int hndl = comm->free_region;
  comm->peer_ptr[hndl] = reinterpret_cast<void **>(malloc(sizeof(void *) * (comm->nvsize)));
  size_t aligned_size = bytes;
  comm->memflags[hndl] = 0;
  comm->mem_dealloc[hndl] = alloc;

  if (alloc) {
    int nranks = comm->nvsize;  // total GPUs in NVLINK domain
    int myrank = comm->nvrank;
    void **remptrs = reinterpret_cast<void **>(malloc(nranks * sizeof(void *)));

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = comm->mydev;
    prop.requestedHandleTypes =
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;  // CU_MEM_HANDLE_TYPE_FABRIC;

    size_t granularity = 0;
    NVTE_CALL_CHECK_CUDA_DRIVER(
        cuMemGetAllocationGranularity, &granularity, &prop,
        static_cast<CUmemAllocationGranularity_flags>(CU_MULTICAST_GRANULARITY_MINIMUM));
    // MPI_Allreduce MAX of granularity check
    aligned_size = (bytes + granularity - 1) / granularity * granularity;

    if (comm->use_mc) {
      CUmulticastObjectProp mcProp = {};
      mcProp.numDevices = nranks;
      mcProp.size = aligned_size;
      mcProp.handleTypes = prop.requestedHandleTypes;
      NVTE_CALL_CHECK_CUDA_DRIVER(
          cuMulticastGetGranularity, &granularity, &mcProp,
          static_cast<CUmemAllocationGranularity_flags>(CU_MULTICAST_GRANULARITY_MINIMUM));
      aligned_size = (aligned_size + granularity - 1) / granularity * granularity;
    }

    prop.location.id = comm->mydev;
    comm->uchandles[hndl] = reinterpret_cast<CUmemGenericAllocationHandle *>(
        malloc(nranks * sizeof(CUmemGenericAllocationHandle)));
    NVTE_CALL_CHECK_CUDA_DRIVER(cuMemCreate, &(comm->uchandles[hndl][myrank]), aligned_size, &prop,
                                (uint64_t)0);

    int *peerfd = reinterpret_cast<int *>(malloc(nranks * sizeof(int)));
    NVTE_CALL_CHECK_CUDA_DRIVER(
        cuMemExportToShareableHandle, reinterpret_cast<void *>(&peerfd[myrank]),
        comm->uchandles[hndl][myrank],
        static_cast<CUmemAllocationHandleType>(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
        (uint64_t)0);

    volatile uint32_t abortFlag = 0;
    struct ncclIpcSocket ipcSock = {0};
    uint64_t opId = 0xdeadcafebeef;
    ncclResult_t ret = ncclSuccess;

    // All-gather POSIX file descriptors across local ranks.
    // NOTE: This cannot be done via MPI_Allgather or other external comm libraries. They mangle
    //       the file descriptor and prevent cuMemImportFromShareableHandle() from correctly
    //       interpreting the file. Instead, we use system socket to send/recv the file handle
    //       without mangling.
    NCCLCHECK(ncclIpcSocketInit(&ipcSock, myrank, (uint64_t)opId, &abortFlag));
    for (int p = 1; p < nranks; p++) {
      comm->_barrier(comm->comm_intra);
      NCCLCHECKGOTO(
          ncclIpcSocketSendFd(&ipcSock, peerfd[myrank], (myrank + p) % nranks, (uint64_t)opId), ret,
          error);
      NCCLCHECKGOTO(ncclIpcSocketRecvFd(&ipcSock, &peerfd[(myrank + nranks - p) % nranks]), ret,
                    error);
    }
  error:
    NCCLCHECK(ncclIpcSocketClose(&ipcSock));

    for (int p = 0; p < nranks; p++) {
      if (p != myrank)
        NVTE_CALL_CHECK_CUDA_DRIVER(
            cuMemImportFromShareableHandle, &comm->uchandles[hndl][p],
            reinterpret_cast<void *>(peerfd[p]),
            static_cast<CUmemAllocationHandleType>(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
      close(peerfd[p]);
    }
    CUdeviceptr ptr;
    NVTE_CALL_CHECK_CUDA_DRIVER(cuMemAddressReserve, &ptr, (size_t)(aligned_size * nranks),
                                (size_t)0, (CUdeviceptr)0, (uint64_t)0);
    comm->ucbase_ptr[hndl] = reinterpret_cast<void *>(ptr);
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    accessDesc.location.id = comm->mydev;

    for (int i = 0; i < nranks; i++) {
      remptrs[i] = reinterpret_cast<void *>(ptr + (aligned_size * i));
      NVTE_CALL_CHECK_CUDA_DRIVER(cuMemMap, reinterpret_cast<CUdeviceptr>(remptrs[i]), aligned_size,
                                  (size_t)0, comm->uchandles[hndl][i], (uint64_t)0);
      if (i == comm->nvrank) {
        if (hndl)
          *gpubuff = remptrs[i];
        else
          comm->gpu_ptrs = remptrs[i];
      }
      comm->peer_ptr[hndl][i] = remptrs[i];
    }
    NVTE_CALL_CHECK_CUDA_DRIVER(cuMemSetAccess, ptr, (size_t)(aligned_size * nranks),
                                const_cast<CUmemAccessDesc *>(&accessDesc), (size_t)1);

    if (hndl == 0) CUDACHECK(cudaMemset(comm->gpu_ptrs, 0, aligned_size));
    CUDACHECK(
        cudaMemcpy((reinterpret_cast<char *>(comm->gpu_ptrs)) + (hndl * nranks * sizeof(void *)),
                   remptrs, nranks * sizeof(void *), cudaMemcpyHostToDevice));
    free(remptrs);
    free(peerfd);
    comm->memflags[hndl] = UB_MEM_UC_CONTIG | UB_MEM_ALLOCATED;

    if (comm->use_mc && comm->mc_maxsize >= comm->mc_offset + aligned_size) {
      NVTE_CALL_CHECK_CUDA_DRIVER(cuMulticastBindMem, comm->mc_handle, comm->mc_offset,
                                  comm->uchandles[hndl][myrank], (size_t)0 /*memOffset*/,
                                  aligned_size, (uint64_t)0);
      comm->memflags[hndl] |= UB_MEM_MC_CREATED;
      comm->mc_ptr[hndl] = reinterpret_cast<char *>(comm->mc_baseptr) + comm->mc_offset;
      comm->mc_offset += aligned_size;
    } else if (!comm->myrank) {
      printf("UB: warning region %d size %ld MB registered without MC access\n", hndl,
             aligned_size / 1024 / 1024);
    }

  } else {
    assert(comm->nvsize <= 8);
    cudaIpcMemHandle_t memhndl;
    CUDACHECK(cudaIpcGetMemHandle(&memhndl, *gpubuff));

    cudaIpcMemHandle_t *tmp;
    comm->_alloc_copy_allgather(reinterpret_cast<void **>(&tmp), reinterpret_cast<void *>(&memhndl),
                                sizeof(cudaIpcMemHandle_t), comm->comm_intra);

    for (int i = 0; i < comm->nvsize; i++) {
      if (i != comm->nvrank) {
        CUDACHECK(cudaIpcOpenMemHandle(&(comm->peer_ptr[hndl][i]), tmp[i],  // NOLINT(*)
                                       cudaIpcMemLazyEnablePeerAccess));
      }
    }
    comm->peer_ptr[hndl][comm->nvrank] = *gpubuff;
    CUDACHECK(cudaDeviceSynchronize());

    CUDACHECK(cudaMemcpy(
        reinterpret_cast<char *>(comm->gpu_ptrs) + (hndl * comm->nvsize * sizeof(void *)),
        comm->peer_ptr[hndl], comm->nvsize * sizeof(void *), cudaMemcpyHostToDevice));

    CUDACHECK(cudaDeviceSynchronize());
    comm->_free(tmp);
  }
  comm->mem_size[hndl] = aligned_size;

  comm->mem_ptr[hndl] = *gpubuff;

  return comm->free_region++;
}
