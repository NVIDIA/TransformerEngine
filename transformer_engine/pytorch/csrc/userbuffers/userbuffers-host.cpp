/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "ipcsocket.cc"
#include "ipcsocket.h"
#include "userbuffers.h"
#include <assert.h>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <sched.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#define MULTICAST_GB_TOTAL 512
#include "nvml.h"

//static int oob_bcast(void *comm_context, void *buf, int size, int root) {
//  MPI_Bcast(buf, size, MPI_BYTE, root,
//            (reinterpret_cast<communicator *>(comm_context))->comm_inter);
//  return 0;
//}
//
//static int oob_barrier(void *comm_context) {
//  MPI_Barrier((reinterpret_cast<communicator *>(comm_context))->comm_inter);
//  return 0;
//}
//
//static int oob_gather(void *comm_context, int root, void *sbuf, void *rbuf, int len) {
//  MPI_Gather(sbuf, len, MPI_BYTE, rbuf, len, MPI_BYTE, root,
//             (reinterpret_cast<communicator *>(comm_context))->comm_inter);
//  return 0;
//}

int stringCmp(const void *a, const void *b) { return strcmp((const char *)a, (const char *)b); }

#define CUDACHECK(cmd)                                                                             \
  do {                                                                                             \
    cudaError_t e = cmd;                                                                           \
    if (e != cudaSuccess) {                                                                        \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));        \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0)

#define CUCHECK(cmd)                                                                               \
  do {                                                                                             \
    CUresult retval = cmd;                                                                         \
    if (retval != CUDA_SUCCESS) {                                                                  \
      const char *error_string;                                                                    \
      cuGetErrorString(retval, &error_string);                                                     \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, error_string);                 \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0);

#define NVTE_UB_ERROR(x)                                                                           \
  do {                                                                                             \
    throw std::runtime_error(std::string(__FILE__ ":") + std::to_string(__LINE__) +                \
                             " in function " + __func__ + ": " + x);                               \
  } while (false)
#define NCCLCHECK(cmd)                                                                             \
  do {                                                                                             \
    ncclResult_t r = cmd;                                                                          \
    if (r != ncclSuccess) {                                                                        \
      printf("Failed, NCCL error %s:%d ''\n", __FILE__, __LINE__ /*,ncclGetErrorString(r)*/);      \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0)

#define NCCLCHECKGOTO(call, RES, label)                                                            \
  do {                                                                                             \
    RES = call;                                                                                    \
    if (RES != ncclSuccess && RES != ncclInProgress) {                                             \
      goto label;                                                                                  \
    }                                                                                              \
  } while (0);

//int pipe_rank(communicator *comm, int step) {
//  int mynode = comm->myrank / comm->nvsize;
//  int mylocal = comm->nvrank;
//  int numlocal = comm->nvsize;
//
//  int newlocal1 = mylocal + step * comm->ar_nvsize * comm->ar2_nvsize;
//  int newlocal = (numlocal + (newlocal1 % numlocal)) % numlocal;
//  int newnode = mynode;
//  newnode += (newlocal1 - newlocal) / numlocal * comm->num_nodes * comm->num2_nodes;
//  int allnodes = comm->nranks / comm->nvsize;
//  newnode = (allnodes + (newnode % allnodes)) % allnodes;
//  return newnode * numlocal + newlocal;
//}

#define NVMLCHECK(cmd)                                                                             \
  do {                                                                                             \
    nvmlReturn_t e = cmd;                                                                          \
    if (e != NVML_SUCCESS) {                                                                       \
      printf("Failed: NVML error %s:%d '%s'\n", __FILE__, __LINE__, nvmlErrorString(e));           \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0)

static int mnnvl_init(communicator **comm) {
  int gpu_device;
  int flag = 0;
  CUdevice current_gpu;
  CUDACHECK(cudaGetDevice(&gpu_device));
  CUCHECK(cuDeviceGet(&current_gpu, gpu_device));
  CUCHECK(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, current_gpu));
  if (!flag) {
    UB_PRINT("CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED is not detected [%d]\n", flag);
    return 0;
  }

  // Check device count
  unsigned int nvml_device_count;
  NVMLCHECK(nvmlDeviceGetCount_v2(&nvml_device_count));
  if (!nvml_device_count) {
    UB_PRINT("No NVML devices found [%d]\n", nvml_device_count);
    return 1; // No nvml devices found
  }

  // Get device handle for the last device
  // TODO: Check all devices
  nvmlDevice_t nvml_device;
  NVMLCHECK(nvmlDeviceGetHandleByIndex_v2(nvml_device_count - 1, &nvml_device));

  // Get fabric info
  nvmlGpuFabricInfoV_t fabric_info = { .version = nvmlGpuFabricInfo_v2,
                                       .state   = NVML_GPU_FABRIC_STATE_NOT_SUPPORTED };
  NVMLCHECK(nvmlDeviceGetGpuFabricInfoV(nvml_device, &fabric_info));
  if (fabric_info.state == NVML_GPU_FABRIC_STATE_NOT_SUPPORTED) {
    UB_PRINT("MNNVL nvmlGpuFabricInfoV_t reported NVML_GPU_FABRIC_STATE_NOT_SUPPORTED [%d]", fabric_info.state); 
    return 1;
  }

// add allreduce for state
// if (fabric_info.state != NVML_GPU_FABRIC_STATE_COMPLETED) abort

  if (getenv("NVTE_UBDEBUG"))
    UB_PRINT("MNNVL nvmlGpuFabricInfoV_t fabric UUID %lx.%lx cliqueId 0x%x state %d healthMask 0x%x",
              ((long *)&fabric_info.clusterUuid)[0], ((long *)&fabric_info.clusterUuid)[1],
              fabric_info.cliqueId, fabric_info.state, fabric_info.healthMask);

  (*comm)->nvml_fabric_info = fabric_info;

  return 0;
}

static int mnnvl_detect_domains(communicator **comm) {
    int ret = 1;
    unsigned char *cluster_uuid = NULL;
    unsigned int *cluster_cliqueid = NULL;
    int mpi_status;
    int clique_size = 0;
    int myclique_rank = 0;


    cluster_uuid = (unsigned char*)malloc((*comm)->nranks * sizeof(char)*NVML_GPU_FABRIC_UUID_LEN);
    if (cluster_uuid == NULL) {
      UB_PRINT("Failed to allocate memory for UUID [%p]", cluster_uuid);
      goto error;
    }

    mpi_status = MPI_Allgather(&(*comm)->nvml_fabric_info.clusterUuid, NVML_GPU_FABRIC_UUID_LEN,
                               MPI_CHAR, cluster_uuid, NVML_GPU_FABRIC_UUID_LEN, MPI_CHAR, MPI_COMM_WORLD);
    if (mpi_status != MPI_SUCCESS) {
      UB_PRINT("MPI_Allgather failed [%d]", mpi_status);
      goto error;
    }

    cluster_cliqueid = (unsigned int*)malloc((*comm)->nranks * sizeof(int) * NVML_GPU_FABRIC_UUID_LEN);
    if (cluster_cliqueid == NULL) {
      UB_PRINT("Failed to allocate memory for UUID [%p]", cluster_cliqueid);
      goto error;
    }
    mpi_status = MPI_Allgather(&(*comm)->nvml_fabric_info.cliqueId, 1,
                               MPI_UNSIGNED, cluster_cliqueid, 1, MPI_UNSIGNED, MPI_COMM_WORLD);
    if (mpi_status != MPI_SUCCESS) {
      UB_PRINT("MPI_Allgather failed [%d]", mpi_status);
      goto error;
    }

    for (int n = 0; n < (*comm)->nranks; n++) {
      if (0 == strncmp((const char*)(*comm)->nvml_fabric_info.clusterUuid, (const char*)&cluster_uuid[n * NVML_GPU_FABRIC_UUID_LEN], NVML_GPU_FABRIC_UUID_LEN) &&
          (*comm)->nvml_fabric_info.cliqueId == cluster_cliqueid[n]) {
              if (n == (*comm)->myrank) {
                myclique_rank = clique_size;
              }
              clique_size++;
       }
    }

    (*comm)->nvrank = myclique_rank;
    (*comm)->nvsize = clique_size;
    
    if (getenv("NVTE_UBDEBUG"))
      UB_PRINT("MNNVL cliqueId 0x%x cliqueSize %d cliqueRank %d",
              (*comm)->nvml_fabric_info.cliqueId, clique_size, myclique_rank);

    ret = 0;
error:
    free(cluster_uuid);
    free(cluster_cliqueid);

    return ret;
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
  for (int i = 0; i < userbuffers_op_types; i++)
    (*comm)->basecounter[i] = 0;
  (*comm)->head = 0;
  (*comm)->tail = 0;
  (*comm)->active_nreqs = 0;
  for (int i = 0; i < userbuffers_op_types; i++)
    (*comm)->active_req[i].active = -1;

  int device_clock    = 0;
  // 110 sec wait time by default
  int sec_timeout = getenv("UB_TIMEOUT") ? atoi(getenv("UB_TIMEOUT")) : 110;
  CUDACHECK(cudaDeviceGetAttribute(&device_clock, cudaDevAttrClockRate, cur_dev));
  (*comm)->ub_timeout = 1000ull * device_clock * sec_timeout;
  if ((*comm)->myrank == 0) {
    printf("UB_TIMEOUT is set to %d sec, %" PRIu64 " cycles, freq: %dkhz\n",
            sec_timeout, (*comm)->ub_timeout, device_clock);
  }

  int ret = 0;
  int namelen, bytes, color, my_node, mylocal, numlocal, num_nodes;
  int rank = (*comm)->myrank, size = (*comm)->nranks;

#ifdef MNNVL
  if (mnnvl_init(comm))
    return 1;
  if (mnnvl_detect_domains(comm))
    return 1;

  mylocal  = (*comm)->nvrank;
  numlocal = (*comm)->nvsize;
#else
  // split communicator
  char host_name[MPI_MAX_PROCESSOR_NAME];
  char(*host_names)[MPI_MAX_PROCESSOR_NAME];
  MPI_Get_processor_name(host_name, &namelen);
  bytes = size * sizeof(char[MPI_MAX_PROCESSOR_NAME]);
  host_names = (char(*)[MPI_MAX_PROCESSOR_NAME])malloc(bytes);
  strcpy(host_names[rank], host_name);  // NOLINT(*)
  for (int n = 0; n < size; n++)
    MPI_Bcast(&(host_names[n]), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, n, MPI_COMM_WORLD);
  qsort(host_names, size, sizeof(char[MPI_MAX_PROCESSOR_NAME]), stringCmp);

  color = 0;
  for (int n = 0; n < size; n++) {
    if (n > 0 && strcmp(host_names[n - 1], host_names[n]))
      color++;
    if (strcmp(host_name, host_names[n]) == 0)
      break;
  }
  free(host_names);

  MPI_Comm_split(MPI_COMM_WORLD, color, rank, &(*comm)->comm_intra);
  // find intranode numbers and make internode communicator
  // figure out mylocal
  // MPI_Comm_rank((*comm)->comm_intra, &mylocal);
  // MPI_Comm_size((*comm)->comm_intra, &numlocal);
  MPI_Comm_rank(MPI_COMM_WORLD, &mylocal);
  MPI_Comm_size(MPI_COMM_WORLD, &numlocal);
  (*comm)->nvrank = mylocal;
  (*comm)->nvsize = numlocal;
#endif

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  int core;
  if (mylocal == 0)
    core = 50;
  if (mylocal == 1)
    core = 58;
  if (mylocal == 2)
    core = 18;
  if (mylocal == 3)
    core = 26;
  if (mylocal == 4)
    core = 114;
  if (mylocal == 5)
    core = 122;
  if (mylocal == 6)
    core = 82;
  if (mylocal == 7)
    core = 90;

  CPU_SET(core, &cpuset);
  if (!getenv("NVTE_NODOUBLE")) {
    if (core > 128)
      CPU_SET(core - 128, &cpuset);
    else
      CPU_SET(core + 128, &cpuset);
  }
  if (getenv("NVTE_DOPIN"))
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

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

  //(*comm)->pipe_id = pipegpus * pipenodegroup_id + mylocal / (datagpus * tensorgpus);

  CUDACHECK(cudaFree(0));
  int datanodegroup_id =
      myrank / numlocal / datanodes;  // data reduction group node belongs, equals 0 for all if both
                                      // pipenodes=1 and tensornodes=1
  // mpi communicator only needed for SHARP which is always
  // allreduce1/data-parallel
  MPI_Comm_split(MPI_COMM_WORLD, mylocal + numlocal * datanodegroup_id, rank, &(*comm)->comm_inter);
  // different rails from same group are in different subcommunicators

  // MPI_Comm_size((*comm)->comm_inter, &num_nodes);
  // MPI_Comm_size((*comm)->comm_inter, &num_nodes);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_node);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_node);
  (*comm)->first_node = mynode - my_node;
  (*comm)->num_nodes = num_nodes;
  (*comm)->my_node = my_node;

  (*comm)->num2_nodes = tensornodes;
  (*comm)->my2_node = (mynode / datanodes) % tensornodes;
  (*comm)->first2_node = mynode - (*comm)->my2_node * datanodes;
  //(*comm)->fifo = reinterpret_cast<ub_request *>(malloc(sizeof(ub_request) * NVTE_MAX_REQUESTS));
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
#ifdef MNNVL
    mcProp.handleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
#else
    mcProp.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#endif

    CUCHECK(cuMulticastGetGranularity(&gran, &mcProp, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    mc_maxsize = ((mc_maxsize + gran - 1) / gran) * gran;
    mcProp.size = mc_maxsize;
    (*comm)->mc_maxsize = mc_maxsize;

#ifdef MNNVL
    if ((*comm)->myrank == 0) {
      CUCHECK(cuMulticastCreate(&(*comm)->mc_handle, &mcProp));
    }

    printf("%d/%d:(%d x %d): DP %d x %d TP %d x %d, DPGROUP %dx%d TPGROUP "
           "%dx%d\n",
           myrank, nranks, myrank / numlocal, myrank % numlocal, (*comm)->my_node,
           (*comm)->ar_nvrank, (*comm)->my2_node, (*comm)->ar2_nvrank, (*comm)->num_nodes,
           (*comm)->ar_nvsize, (*comm)->num2_nodes, (*comm)->ar2_nvsize);

    CUmemFabricHandle *exphndl = (CUmemFabricHandle *)malloc(sizeof(CUmemFabricHandle));
    //if ((*comm)->ar2_nvrank == 0) {
    if ((*comm)->myrank == 0) {
      CUCHECK(cuMemExportToShareableHandle(static_cast<void *>(exphndl), (*comm)->mc_handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
    }
    //MPI_Bcast(exphndl, sizeof(CUmemFabricHandle), MPI_BYTE, 0, (*comm)->comm_inter);
    MPI_Bcast(exphndl, sizeof(CUmemFabricHandle), MPI_BYTE, 0, MPI_COMM_WORLD);
    //if ((*comm)->ar2_nvrank != 0) {
    if ((*comm)->myrank != 0) {
      CUCHECK(cuMemImportFromShareableHandle(&(*comm)->mc_handle, reinterpret_cast<void *>(exphndl), CU_MEM_HANDLE_TYPE_FABRIC));
    }
    free(exphndl);
#else
    int fd;
    volatile uint32_t abortFlag = 0;
    struct ncclIpcSocket ipcSock = {0};
    uint64_t opId = 0xdeadcafeb000 + (*comm)->ar2_firstgpu;
    ncclResult_t ret = ncclSuccess;
    NCCLCHECK(ncclIpcSocketInit(&ipcSock, (*comm)->ar2_nvrank, (uint64_t)opId, &abortFlag));
    MPI_Barrier(MPI_COMM_WORLD);

    if ((*comm)->ar2_nvrank == 0) {
      CUCHECK(cuMulticastCreate(&(*comm)->mc_handle, &mcProp));
      CUCHECK(cuMemExportToShareableHandle(&fd, (*comm)->mc_handle,
                                           CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0 /*flags*/));
      for (int p = 1; p < (*comm)->ar2_nvsize; p++) {
        MPI_Barrier((*comm)->comm_intra);
        NCCLCHECKGOTO(ncclIpcSocketSendFd(&ipcSock, fd, p, (uint64_t)opId), ret, error);
      }
    } else {
      for (int i = 0; i < (*comm)->ar2_nvrank; i++)
        MPI_Barrier((*comm)->comm_intra);
      NCCLCHECKGOTO(ncclIpcSocketRecvFd(&ipcSock, &fd), ret, error);
      for (int i = 0; i < (*comm)->ar2_nvsize - (*comm)->ar2_nvrank - 1; i++)
        MPI_Barrier((*comm)->comm_intra);
      CUCHECK(cuMemImportFromShareableHandle(&(*comm)->mc_handle, reinterpret_cast<void *>(fd),
                                             CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
    }
  error:
    NCCLCHECK(ncclIpcSocketClose(&ipcSock));
    close(fd);
#endif
    CUCHECK(cuMulticastAddDevice((*comm)->mc_handle, (*comm)->mydev));

    CUdeviceptr mc_va;
    CUCHECK(cuMemAddressReserve(&mc_va, mc_maxsize, 0, 0U, 0));
    CUCHECK(cuMemMap(mc_va, mc_maxsize, 0, (*comm)->mc_handle, 0));

    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = (*comm)->mydev;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUCHECK(cuMemSetAccess(mc_va, mc_maxsize, &accessDesc, 1));

    (*comm)->mc_baseptr = reinterpret_cast<void *>(mc_va);
    MPI_Barrier(MPI_COMM_WORLD);
    if (!(*comm)->myrank)
      printf("MC initialized succesfully, window size = %ld\n", mc_maxsize);
  } else {
    if (!(*comm)->myrank)
      printf("MC NOT initialized and used\n");
    (*comm)->mc_maxsize = 0;
    (*comm)->mc_offset = 0;
    (*comm)->use_mc = 0;
  }

#define LOCALSIZE 4 * (NVTE_REG0_OFFSET(*comm) + NVTE_REG0_FLAGS + NVTE_REG0_COMMBUFFER * NBUF)
  // peer pointers + op flags + comm buffer

  CUDACHECK(cudaMalloc(&(*comm)->gpu_ptrs,
                       LOCALSIZE));  // flags and pointers, no block data yet
  CUDACHECK(cudaMemset((*comm)->gpu_ptrs, 0, LOCALSIZE));
  CUDACHECK(cudaDeviceSynchronize());
  register_user_buffer_collective(&((*comm)->gpu_ptrs), LOCALSIZE, *comm, true);  // will use handler 0
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
    printf("%d/%d:(%d x %d): DP %d x %d TP %d x %d, DPGROUP %dx%d TPGROUP "
           "%dx%d\n",
           myrank, nranks, myrank / numlocal, myrank % numlocal, (*comm)->my_node,
           (*comm)->ar_nvrank, (*comm)->my2_node, (*comm)->ar2_nvrank, (*comm)->num_nodes,
           (*comm)->ar_nvsize, (*comm)->num2_nodes, (*comm)->ar2_nvsize);
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
}

int register_user_buffer_collective(void **gpubuff, size_t bytes, communicator *comm, bool alloc) {
  if (comm->free_region > NVTE_MAX_REGIONS)
    return -1;
  int hndl = comm->free_region;
  comm->peer_ptr[hndl] = reinterpret_cast<void **>(malloc(sizeof(void *) * (comm->nvsize)));
  size_t aligned_size = bytes;
  comm->memflags[hndl] = 0;

//  printf("Alloc register_user_buffer_collective %d\n", alloc);

  if (alloc) {
    int nranks = comm->nvsize;  // total GPUs in NVLINK domain
    int myrank = comm->nvrank;
    void **remptrs = reinterpret_cast<void **>(malloc(nranks * sizeof(void *)));

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = comm->mydev;
#ifdef MNNVL
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
#else
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#endif
    size_t granularity = 0;
    CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    // MPI_Allreduce MAX of granularity check
    aligned_size = (bytes + granularity - 1) / granularity * granularity;

    if (comm->use_mc) {
      CUmulticastObjectProp mcProp = {};
      mcProp.numDevices = nranks;
      mcProp.size = aligned_size;
      mcProp.handleTypes = prop.requestedHandleTypes;
      CUCHECK(cuMulticastGetGranularity(&granularity, &mcProp, CU_MULTICAST_GRANULARITY_MINIMUM));
      aligned_size = (aligned_size + granularity - 1) / granularity * granularity;
    }

    prop.location.id = comm->mydev;
    comm->uchandles[hndl] = 
        reinterpret_cast<CUmemGenericAllocationHandle *>(malloc(nranks * sizeof(CUmemGenericAllocationHandle)));
    CUCHECK(cuMemCreate(&(comm->uchandles[hndl][myrank]), aligned_size, &prop, 0));
#ifdef MNNVL
    CUmemFabricHandle *exphndl = (CUmemFabricHandle *)malloc(nranks * sizeof(CUmemFabricHandle));
    CUmemFabricHandle myhndl;
    CUCHECK(cuMemExportToShareableHandle(&myhndl, comm->uchandles[hndl][myrank], CU_MEM_HANDLE_TYPE_FABRIC, 0));
    //CUCHECK(cuMemExportToShareableHandle(&myhndl, comm->uchandles[hndl][myrank], CU_MEM_HANDLE_TYPE_FABRIC, 0));
    //CUCHECK(cuMemExportToShareableHandle(static_cast<void *>(&exphndl[myrank]), comm->uchandles[hndl][myrank], CU_MEM_HANDLE_TYPE_FABRIC, 0));
    //MPI_Allgather(&myhndl, sizeof(CUmemFabricHandle), MPI_BYTE, exphndl, sizeof(CUmemFabricHandle), MPI_BYTE, comm->comm_inter);
    MPI_Allgather(&myhndl, sizeof(CUmemFabricHandle), MPI_BYTE, exphndl, sizeof(CUmemFabricHandle), MPI_BYTE, MPI_COMM_WORLD);
    for (int p = 0; p < nranks; p++)
      if (p != myrank)
        CUCHECK(cuMemImportFromShareableHandle(&comm->uchandles[hndl][p], reinterpret_cast<void *>(&exphndl[p]), CU_MEM_HANDLE_TYPE_FABRIC));
    free(exphndl);
#else
    int *peerfd                  = reinterpret_cast<int *>(malloc(nranks * sizeof(int)));
    volatile uint32_t abortFlag  = 0;
    struct ncclIpcSocket ipcSock = {0};
    uint64_t opId                = 0xdeadcafebeef;
    ncclResult_t ret             = ncclSuccess;

    CUCHECK(cuMemExportToShareableHandle(&peerfd[myrank], comm->uchandles[hndl][myrank],
                                         CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0 /*flags*/));

    NCCLCHECK(ncclIpcSocketInit(&ipcSock, myrank, (uint64_t)opId, &abortFlag));
    for (int p = 1; p < nranks; p++) {
      MPI_Barrier(comm->comm_intra);
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
        CUCHECK(cuMemImportFromShareableHandle(&comm->uchandles[hndl][p],
                                               reinterpret_cast<void *>(peerfd[p]),
                                               CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
      close(peerfd[p]);
    }
    free(peerfd);
#endif
    CUdeviceptr ptr;
    CUCHECK(cuMemAddressReserve(&ptr, aligned_size * nranks, 0, 0, 0));
    comm->ucbase_ptr[hndl] = reinterpret_cast<void *>(ptr);
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    accessDesc.location.id = comm->mydev;

    for (int i = 0; i < nranks; i++) {
      CUCHECK(cuMemMap(ptr + (aligned_size * i), aligned_size, 0, comm->uchandles[hndl][i], 0));
      remptrs[i] = reinterpret_cast<void *>(ptr + (aligned_size * i));
      if (i == comm->nvrank) {
        if (hndl)
          *gpubuff = remptrs[i];
        else
          comm->gpu_ptrs = remptrs[i];
      }
      comm->peer_ptr[hndl][i] = remptrs[i];
    }
    CUCHECK(cuMemSetAccess(ptr, aligned_size * nranks, &accessDesc, 1));

    if (hndl == 0)
      CUDACHECK(cudaMemset(comm->gpu_ptrs, 0, aligned_size));
    CUDACHECK(
        cudaMemcpy((reinterpret_cast<char *>(comm->gpu_ptrs)) + (hndl * nranks * sizeof(void *)),
                   remptrs, nranks * sizeof(void *), cudaMemcpyHostToDevice));
    free(remptrs);
    comm->memflags[hndl] = UB_MEM_UC_CONTIG | UB_MEM_ALLOCATED;

    if (comm->use_mc && comm->mc_maxsize >= comm->mc_offset + aligned_size) {
      CUCHECK(cuMulticastBindMem(comm->mc_handle, comm->mc_offset, comm->uchandles[hndl][myrank],
                                 0 /*memOffset*/, aligned_size, 0));
      comm->memflags[hndl] |= UB_MEM_MC_CREATED;
      comm->mc_ptr[hndl] = reinterpret_cast<char *>(comm->mc_baseptr) + comm->mc_offset;
      comm->mc_offset += aligned_size;
    } else if (!comm->myrank) {
      printf("UB: warning region %d size %ld MB registered without MC access\n", hndl,
             aligned_size / 1024 / 1024);
    }
  } else {
    if (!comm->myrank)
      printf("UB: warning region %d size %ld MB allocated using cudaMalloc - deprecated(no MC available)\n", hndl, aligned_size / 1024 / 1024);
#ifdef MNNVL
    exit(2);
#endif
    assert(comm->nvsize <= 8);
    cudaIpcMemHandle_t *memhndl =
        reinterpret_cast<cudaIpcMemHandle_t *>(malloc(sizeof(cudaIpcMemHandle_t) * (comm->nvsize)));

    cudaIpcMemHandle_t myhndl;

    CUDACHECK(cudaIpcGetMemHandle(&myhndl, *gpubuff));

    MPI_Allgather(&myhndl, sizeof(cudaIpcMemHandle_t), MPI_BYTE, memhndl,
                  sizeof(cudaIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD);

    for (int i = 0; i < comm->nvsize; i++)
      if (i != comm->nvrank) {
        printf("cudaIpcOpenMemHandle nvsize %d nvrank %d i %d\n", comm->nvsize, comm->nvrank, i);
        CUDACHECK(cudaIpcOpenMemHandle((void **)&(comm->peer_ptr[hndl][i]),  // NOLINT(*)
                                       memhndl[i], cudaIpcMemLazyEnablePeerAccess));
      }
    comm->peer_ptr[hndl][comm->nvrank] = *gpubuff;
    CUDACHECK(cudaDeviceSynchronize());

    CUDACHECK(cudaMemcpy(
        reinterpret_cast<char *>(comm->gpu_ptrs) + (hndl * comm->nvsize * sizeof(void *)),
        comm->peer_ptr[hndl], comm->nvsize * sizeof(void *), cudaMemcpyHostToDevice));

    CUDACHECK(cudaDeviceSynchronize());
    free(memhndl);
  }
  comm->mem_size[hndl] = aligned_size;

  comm->mem_ptr[hndl] = *gpubuff;

  return comm->free_region++;
}
