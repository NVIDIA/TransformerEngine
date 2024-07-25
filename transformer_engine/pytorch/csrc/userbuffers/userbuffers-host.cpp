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

#include "common/util/cuda_driver.h"
#include "common/util/logging.h"
#include "ipcsocket.h"
#include "userbuffers.h"

#ifdef NVTE_UB_WITH_MPI
static MPI_Comm EXT_COMM_WORLD = MPI_COMM_WORLD;
static MPI_Comm EXT_COMM_INTRA;
static MPI_Comm EXT_COMM_INTER;

#define UB_MPI_CHECK(expr)                                                                   \
  do {                                                                                       \
    const int mpicode = (expr);                                                              \
    if (mpicode != MPI_SUCCESS) {                                                            \
      char mpimsg[MPI_MAX_ERROR_STRING];                                                     \
      int mpilen;                                                                            \
      MPI_Error_string(mpicode, mpimsg, &mpilen);                                            \
      std::vector<char> errmsg(1024);                                                        \
      snprintf(errmsg.data(), errmsg.size(), "%s:%d in function %s: %s", __FILE__, __LINE__, \
               __func__, mpimsg);                                                            \
      throw std::runtime_error(errmsg.data());                                               \
    }                                                                                        \
  } while (false)

void ub_mpi_allgather(void *globaldata, size_t globalbytes, void *localdata, size_t localbytes,
                      ExtComm group) {
  // UB_MPI_CHECK(MPI_Allgather(localdata, localbytes, MPI_BYTE,
  //                            globaldata, globalbytes, MPI_BYTE,
  //                            static_cast<MPI_Comm>(group)));
  MPI_Comm comm = static_cast<MPI_Comm>(group);
  int numranks;
  UB_MPI_CHECK(MPI_Comm_size(comm, &numranks));
  assert(globalbytes == numranks * localbytes);

  int myrank;
  UB_MPI_CHECK(MPI_Comm_rank(comm, &myrank));
  char *globaltarget = reinterpret_cast<char *>(globaldata) + (myrank * localbytes);
  memcpy(globaltarget, localdata, localbytes);

  for (int n = 0; n < numranks; n++) {
    globaltarget = reinterpret_cast<char *>(globaldata) + (n * localbytes);
    UB_MPI_CHECK(MPI_Bcast(globaltarget, localbytes, MPI_BYTE, n, comm));
  }
}

void ub_mpi_barrier(ExtComm group) { UB_MPI_CHECK(MPI_Barrier(static_cast<MPI_Comm>(group))); }
#else
static char EXT_COMM_WORLD[] = "world";
static char EXT_COMM_INTRA[] = "intra";
static char EXT_COMM_INTER[] = "inter";
#endif

#define MULTICAST_GB_TOTAL 512

int stringCmp(const void *a, const void *b) { return strcmp((const char *)a, (const char *)b); }

#define IPCCHECK(cmd)                                                                           \
  do {                                                                                          \
    ipcSocketResult_t r = cmd;                                                                  \
    if (r != ipcSocketSuccess) {                                                                \
      printf("Failed, UDS error %s:%d '%s'\n", __FILE__, __LINE__, ipcSocketGetErrorString(r)); \
      exit(EXIT_FAILURE);                                                                       \
    }                                                                                           \
  } while (0)

#define IPCCHECKGOTO(call, RES, label)                           \
  do {                                                           \
    RES = call;                                                  \
    if (RES != ipcSocketSuccess && RES != ipcSocketInProgress) { \
      goto label;                                                \
    }                                                            \
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
    int numnodes, std::function<void(void *, size_t, void *, size_t, ExtComm)> ext_allgather,
    std::function<void(ExtComm)> ext_barrier, int pipegpus, int pipenodes, int tensorgpus,
    int tensornodes) {
  *comm = new communicator();

  (*comm)->comm_world = EXT_COMM_WORLD;
  (*comm)->_allgather = ext_allgather;
  (*comm)->_barrier = ext_barrier;
  (*comm)->nranks = numranks;
  (*comm)->myrank = myrank;
  (*comm)->free_region = 0;
  (*comm)->launch_mode = NVTE_LAUNCH_GPU | NVTE_LAUNCH_CPU;

  int cur_dev, ndev;
  cudaDeviceProp device_prop;
  NVTE_CHECK_CUDA(cudaGetDevice(&cur_dev));
  NVTE_CHECK_CUDA(cudaGetDeviceCount(&ndev));
  NVTE_CHECK_CUDA(cudaGetDeviceProperties(&device_prop, cur_dev));
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
  NVTE_CHECK_CUDA(cudaDeviceGetAttribute(&device_clock, cudaDevAttrClockRate, cur_dev));
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
    NVTE_CHECK_CUDA(cudaSetDevice(mylocal));
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
    //       interpreting the file. Instead, we use Unix domain sockets for the kernel to
    //       recreate the correct file descriptor on every receiving rank.
    int fd;
    volatile uint32_t abortFlag = 0;
    IpcSocketHandle ipcSock = {0};
    uint64_t opId = 0xdeadcafeb000 + (*comm)->ar2_firstgpu;
    ipcSocketResult_t ret = ipcSocketSuccess;
    IPCCHECK(ipcSocketInit(&ipcSock, (*comm)->ar2_nvrank, (uint64_t)opId, &abortFlag));
    (*comm)->_barrier((*comm)->comm_world);

    if ((*comm)->ar2_nvrank == 0) {
      NVTE_CALL_CHECK_CUDA_DRIVER(cuMulticastCreate, &(*comm)->mc_handle, &mcProp);
      NVTE_CALL_CHECK_CUDA_DRIVER(
          cuMemExportToShareableHandle, reinterpret_cast<void *>(&fd), (*comm)->mc_handle,
          static_cast<CUmemAllocationHandleType>(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
          (uint64_t)0);

      for (int p = 1; p < (*comm)->ar2_nvsize; p++) {
        (*comm)->_barrier((*comm)->comm_intra);
        IPCCHECKGOTO(ipcSocketSendFd(&ipcSock, fd, p, (uint64_t)opId), ret, error);
      }
    } else {
      for (int p = 1; p < (*comm)->ar2_nvsize; p++) {
        (*comm)->_barrier((*comm)->comm_intra);
        if ((*comm)->ar2_nvrank == p) IPCCHECKGOTO(ipcSocketRecvFd(&ipcSock, &fd), ret, error);
      }
    }

  error:
    if ((*comm)->ar2_nvrank != 0) {
      NVTE_CALL_CHECK_CUDA_DRIVER(
          cuMemImportFromShareableHandle, &(*comm)->mc_handle, reinterpret_cast<void *>(fd),
          static_cast<CUmemAllocationHandleType>(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
    }
    IPCCHECK(ipcSocketClose(&ipcSock));
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

  NVTE_CHECK_CUDA(
      cudaMalloc(&(*comm)->gpu_ptrs, LOCALSIZE));  // flags and pointers, no block data yet
  NVTE_CHECK_CUDA(cudaMemset((*comm)->gpu_ptrs, 0, LOCALSIZE));
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());
  register_user_buffer_collective(&((*comm)->gpu_ptrs), LOCALSIZE, *comm, false);
  NVTE_CHECK_CUDA(cudaMalloc(&(*comm)->send_id, (*comm)->nranks * sizeof(int)));
  NVTE_CHECK_CUDA(cudaMalloc(&(*comm)->recv_id, NVTE_MAX_REGIONS * (*comm)->nranks * sizeof(int)));
  NVTE_CHECK_CUDA(cudaMemset((*comm)->send_id, 0, (*comm)->nranks * sizeof(int)));
  NVTE_CHECK_CUDA(
      cudaMemset((*comm)->recv_id, 0, NVTE_MAX_REGIONS * (*comm)->nranks * sizeof(int)));
  (*comm)->sms = 16;
  (*comm)->threads = 1024;

#define GPU_PAGE_SHIFT 16
#define GPU_PAGE_SIZE (1UL << GPU_PAGE_SHIFT)
#define GPU_PAGE_OFFSET (GPU_PAGE_SIZE - 1)
#define GPU_PAGE_MASK (~GPU_PAGE_OFFSET)

  NVTE_CHECK_CUDA(cudaMalloc(&(*comm)->flags, 2 * GPU_PAGE_SIZE));
  NVTE_CHECK_CUDA(cudaMemset((*comm)->flags, 0, 2 * GPU_PAGE_SIZE));
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
    int numnodes, std::function<void(void *, size_t, void *, size_t, ExtComm)> ext_allgather,
    std::function<void(ExtComm)> ext_barrier, int pipegpus, int pipenodes) {
  return create_communicator_grouped2(comm, myrank, numranks, mylocal, numlocal, mynode, numnodes,
                                      ext_allgather, ext_barrier, pipegpus, pipenodes, 1, 1);
}

int create_communicator(communicator **comm, int myrank, int numranks, int mylocal, int numlocal,
                        int mynode, int numnodes,
                        std::function<void(void *, size_t, void *, size_t, ExtComm)> ext_allgather,
                        std::function<void(ExtComm)> ext_barrier) {
  return create_communicator_grouped2(comm, myrank, numranks, mylocal, numlocal, mynode, numnodes,
                                      ext_allgather, ext_barrier, 1, 1, 1, 1);
}

int create_communicator_grouped2_mpi(communicator **comm, int pipegpus, int pipenodes,
                                     int tensorgpus, int tensornodes) {
#ifdef NVTE_UB_WITH_MPI
  // get global numbers
  int myrank, numranks;
  UB_MPI_CHECK(MPI_Comm_rank(EXT_COMM_WORLD, &myrank));
  UB_MPI_CHECK(MPI_Comm_size(EXT_COMM_WORLD, &numranks));

  // find intranode numbers and make internode communicator
  char hostname[MPI_MAX_PROCESSOR_NAME];
  int namelen;
  UB_MPI_CHECK(MPI_Get_processor_name(hostname, &namelen));

  char(*hostnames)[MPI_MAX_PROCESSOR_NAME] =
      static_cast<char(*)[MPI_MAX_PROCESSOR_NAME]>(malloc(numranks * MPI_MAX_PROCESSOR_NAME));
  strcpy(hostnames[myrank], hostname);  // NOLINT(*)
  for (int n = 0; n < numranks; n++)
    UB_MPI_CHECK(MPI_Bcast(&(hostnames[n]), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, n, EXT_COMM_WORLD));
  qsort(hostnames, numranks, MPI_MAX_PROCESSOR_NAME, stringCmp);

  int color = 0;
  for (int n = 0; n < numranks; n++) {
    if (n > 0 && strcmp(hostnames[n - 1], hostnames[n])) color++;
    if (strcmp(hostname, hostnames[n]) == 0) break;
  }
  free(hostnames);

  int mylocal, numlocal;
  UB_MPI_CHECK(MPI_Comm_split(EXT_COMM_WORLD, color, myrank, &EXT_COMM_INTRA));
  UB_MPI_CHECK(MPI_Comm_rank(EXT_COMM_INTRA, &mylocal));
  UB_MPI_CHECK(MPI_Comm_size(EXT_COMM_INTRA, &numlocal));

  // find internode numbers and make internode communicator
  NVTE_CHECK_CUDA(cudaFree(0));
  int allnodes = numranks / numlocal;
  int datanodes = allnodes / pipenodes / tensornodes;
  // data reduction group node belongs, equals 0 for all if both pipenodes=1 and tensornodes=1
  int datanodegroup_id = myrank / numlocal / datanodes;
  // mpi communicator only needed for SHARP which is always allreduce1/data-parallel
  UB_MPI_CHECK(MPI_Comm_split(EXT_COMM_WORLD, mylocal + numlocal * datanodegroup_id, myrank,
                              &EXT_COMM_INTER));
  // different rails from same group are in different subcommunicators
  int mynode, numnodes;
  UB_MPI_CHECK(MPI_Comm_size(EXT_COMM_INTER, &numnodes));
  UB_MPI_CHECK(MPI_Comm_rank(EXT_COMM_INTER, &mynode));

  // finally call the abstracted constructor with MPI info
  return create_communicator_grouped2(comm, myrank, numranks, mylocal, numlocal, mynode, numnodes,
                                      &ub_mpi_allgather, &ub_mpi_barrier, pipegpus, pipenodes,
                                      tensorgpus, tensornodes);
#else
  NVTE_ERROR(std::string("Bootstrapping Userbuffers with MPI requires building") +
             std::string("Transformer Engine with NVTE_UB_WITH_MPI=1 and MPI_HOME=/path/to/mpi"));
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
    if (hndl > 0 && comm->use_mc && comm->mem_dealloc[hndl]) {
      for (int rank = 0; rank < comm->nvsize; rank++) {
        if (rank == comm->nvrank) {
          NVTE_CALL_CHECK_CUDA_DRIVER(cuMemRelease, comm->uchandles[hndl][rank]);
        } else {
          comm->uchandles[hndl][rank] = 0;
        }
      }
      free(reinterpret_cast<void *>(comm->uchandles[hndl]));
    } else {
      for (int rank = 0; rank < comm->nvsize; rank++) {
        if (rank != comm->nvrank) {
          cudaIpcCloseMemHandle(comm->peer_ptr[hndl][rank]);
        } else if (comm->mem_dealloc[hndl]) {
          NVTE_CHECK_CUDA(cudaFree(comm->peer_ptr[hndl][rank]));
        } else {
          comm->peer_ptr[hndl][rank] = nullptr;  // remove reference to external buffer
        }
      }
    }
    free(comm->peer_ptr[hndl]);
    comm->mem_ptr[hndl] = nullptr;
  }
  cudaFree(reinterpret_cast<void *>(comm->recv_id));
  cudaFree(reinterpret_cast<void *>(comm->send_id));
  if (comm->use_mc) {
    NVTE_CALL_CHECK_CUDA_DRIVER(cuMemRelease, comm->mc_handle);
  }
  free(comm->fifo);
  delete comm;
}

void destroy_communicator_mpi(communicator *comm) {
#ifdef NVTE_UB_WITH_MPI
  MPI_Comm_free(static_cast<MPI_Comm *>(&(comm->comm_inter)));
  MPI_Comm_free(static_cast<MPI_Comm *>(&(comm->comm_intra)));
  destroy_communicator(comm);
#else
  NVTE_ERROR(std::string("Communicator is not bootstrapped with MPI and ") +
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

  if (comm->use_mc && alloc) {
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
    IpcSocketHandle ipcSock = {0};
    uint64_t opId = 0xdeadcafebeef;
    ipcSocketResult_t ret = ipcSocketSuccess;

    // All-gather POSIX file descriptors across local ranks
    IPCCHECK(ipcSocketInit(&ipcSock, myrank, (uint64_t)opId, &abortFlag));
    for (int p = 1; p < nranks; p++) {
      int send_to = (myrank + p) % nranks;
      int recv_from = (myrank + nranks - p) % nranks;
      comm->_barrier(comm->comm_intra);
      IPCCHECKGOTO(ipcSocketSendFd(&ipcSock, peerfd[myrank], send_to, (uint64_t)opId), ret, error);
      IPCCHECKGOTO(ipcSocketRecvFd(&ipcSock, &peerfd[recv_from]), ret, error);
    }

  error:
    IPCCHECK(ipcSocketClose(&ipcSock));

    for (int p = 0; p < nranks; p++) {
      if (p != myrank)
        NVTE_CALL_CHECK_CUDA_DRIVER(
            cuMemImportFromShareableHandle, &comm->uchandles[hndl][p],
            reinterpret_cast<void *>(peerfd[p]),
            static_cast<CUmemAllocationHandleType>(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
      close(peerfd[p]);
    }
    free(peerfd);

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

    if (hndl == 0) NVTE_CHECK_CUDA(cudaMemset(comm->gpu_ptrs, 0, aligned_size));
    NVTE_CHECK_CUDA(
        cudaMemcpy((reinterpret_cast<char *>(comm->gpu_ptrs)) + (hndl * nranks * sizeof(void *)),
                   remptrs, nranks * sizeof(void *), cudaMemcpyHostToDevice));
    free(remptrs);
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
    if (alloc) {
      NVTE_CHECK_CUDA(cudaMalloc(gpubuff, bytes));
      NVTE_CHECK_CUDA(cudaMemset(*gpubuff, 0, bytes));
    }

    NVTE_CHECK(comm->nvsize <= 8, "CUDA IPC supports only up to 8 GPUs in an NVLink domain.");
    cudaIpcMemHandle_t memhndl;
    NVTE_CHECK_CUDA(cudaIpcGetMemHandle(&memhndl, *gpubuff));

    cudaIpcMemHandle_t *tmp =
        reinterpret_cast<cudaIpcMemHandle_t *>(malloc(comm->nvsize * sizeof(cudaIpcMemHandle_t)));
    comm->_allgather(reinterpret_cast<void *>(tmp), comm->nvsize * sizeof(cudaIpcMemHandle_t),
                     reinterpret_cast<void *>(&memhndl), sizeof(cudaIpcMemHandle_t),
                     comm->comm_intra);

    for (int i = 0; i < comm->nvsize; i++) {
      if (i != comm->nvrank) {
        NVTE_CHECK_CUDA(cudaIpcOpenMemHandle(&(comm->peer_ptr[hndl][i]), tmp[i],  // NOLINT(*)
                                             cudaIpcMemLazyEnablePeerAccess));
      }
    }
    comm->peer_ptr[hndl][comm->nvrank] = *gpubuff;
    NVTE_CHECK_CUDA(cudaDeviceSynchronize());

    NVTE_CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<char *>(comm->gpu_ptrs) + (hndl * comm->nvsize * sizeof(void *)),
        comm->peer_ptr[hndl], comm->nvsize * sizeof(void *), cudaMemcpyHostToDevice));

    NVTE_CHECK_CUDA(cudaDeviceSynchronize());
    free(tmp);
  }
  comm->mem_size[hndl] = aligned_size;

  comm->mem_ptr[hndl] = *gpubuff;

  return comm->free_region++;
}
