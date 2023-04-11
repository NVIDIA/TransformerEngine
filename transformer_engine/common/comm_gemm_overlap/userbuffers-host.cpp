#include <mpi.h>
#include <transformer_engine/userbuffers.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include <iostream>

#ifdef MNNVL
#define CU_MEM_HANDLE_TYPE_FABRIC ((CUmemAllocationHandleType)0x8ULL)
#define CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED (0x20080404)
typedef struct CUmemFabricHandle_st {
    unsigned char data[64];
} CUmemFabricHandle;
#endif

#ifdef UCP
#define UCXCHECK(cmd) do {                         \
  ucs_status_t e = cmd;                              \
  if( e != UCS_OK ) {                          \
    printf("Failed: UCX error %s:%d'%d' %s\n",             \
        __FILE__,__LINE__,e,ucs_status_string(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)
#define UCXPUTCHECK(cmd) do {                         \
  ucs_status_t e = cmd;                              \
  if( e != UCS_OK && e!= UCS_INPROGRESS) {                          \
    printf("Failed: UCX error %s:%d'%d' %s\n",             \
        __FILE__,__LINE__,e,ucs_status_string(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)
#endif

#ifdef MULTINODE
#include <immintrin.h>
#include <x86intrin.h>
#include <sched.h>
#include <math.h>
void* proxythread(void* c);
static int oob_bcast(void *comm_context, void *buf, int size, int root) {
	MPI_Bcast(buf, size, MPI_BYTE, root, ((communicator*)comm_context)->comm_inter);
	return 0;
}

static int oob_barrier(void *comm_context) {
	MPI_Barrier(((communicator*)comm_context)->comm_inter);
	return 0;
}

static int oob_gather(void *comm_context, int root, void *sbuf, void *rbuf, int len) {
	MPI_Gather(sbuf, len, MPI_BYTE, rbuf, len, MPI_BYTE, root, ((communicator*)comm_context)->comm_inter);
	return 0;
}

int stringCmp( const void *a, const void *b)
{ return strcmp((const char*)a,(const char*)b);  }
#endif

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define CUCHECK(cmd) do {                                                                          \
    CUresult retval = cmd;                                                                       \
    if (retval != CUDA_SUCCESS) {                                                                \
      const char *error_string;                                                                  \
      cuGetErrorString(retval, &error_string);                                                   \
      printf("Failed: Cuda error %s:%d '%s'\n",                                                  \
        __FILE__,__LINE__,error_string);                                                        \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  } while (0);

int pipe_rank(communicator *comm,int step) { 
    int mynode = comm->myrank / comm->nvsize;
    int mylocal = comm->nvrank;
    int numlocal = comm->nvsize;

    int newlocal1 = mylocal+step*comm->ar_nvsize*comm->ar2_nvsize;
    int newlocal = (numlocal+(newlocal1 % numlocal))%numlocal;
    int newnode=mynode;
#ifdef MULTINODE
    newnode += (newlocal1-newlocal)/numlocal*comm->num_nodes*comm->num2_nodes;
    int allnodes=comm->nranks/comm->nvsize;
    newnode = (allnodes+(newnode % allnodes))%allnodes;
#endif
    return newnode*numlocal+newlocal;
}

int create_communicator_grouped2( communicator** comm, int pipegpus, int pipenodes, int tensorgpus, int tensornodes) {
  *comm = (communicator*)malloc(sizeof(communicator));

  int myrank,nranks,cur_dev,ndev;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  (*comm) -> nranks = nranks;
  (*comm) -> myrank = myrank;
  (*comm) -> free_region = 0;
  (*comm) -> launch_mode = LAUNCH_GPU | LAUNCH_CPU; 

  cudaDeviceProp device_prop;
  CUDACHECK(cudaGetDevice(&cur_dev));
  CUDACHECK(cudaGetDeviceCount(&ndev));
  CUDACHECK(cudaGetDeviceProperties(&device_prop, cur_dev));
  (*comm) -> sm_arch = device_prop.major;
  (*comm) -> use_rr_kernel = device_prop.major == 8;
  if(getenv("OVERRIDERR")) (*comm) -> use_rr_kernel = atoi(getenv("OVERRIDERR"));
  (*comm) -> push = device_prop.major != 8; 
  (*comm) -> use_ce = getenv("USECE")?1:0;
  (*comm) -> cga_size = getenv("CGASIZE") ? atoi(getenv("CGASIZE")) : (device_prop.major == 9 && !(*comm) -> use_rr_kernel) ? 4 : 1;


#ifdef MULTINODE
  for(int i=0;i<userbuffers_op_types;i++)
    (*comm) -> basecounter[i] = 0;
  (*comm) -> head = 0;
  (*comm) -> tail = 0;
  (*comm) -> activeproxy = 1;
  (*comm) -> active_nreqs = 0;
  for(int i=0;i<userbuffers_op_types;i++)
    (*comm) -> active_req[i].active = -1;
  
  int ret=0;
#ifndef NOSHARP
  struct sharp_coll_comm_init_spec comm_spec;
  struct sharp_coll_init_spec init_spec = {0};
#endif
  //split communicator
	char host_name[MPI_MAX_PROCESSOR_NAME];
	char (*host_names)[MPI_MAX_PROCESSOR_NAME];
	int namelen,bytes,color,my_node,mylocal,numlocal,num_nodes;
  int rank=(*comm)->myrank,size=(*comm)->nranks;
	MPI_Get_processor_name(host_name,&namelen);
	bytes = size * sizeof(char[MPI_MAX_PROCESSOR_NAME]);
	host_names = (char (*)[MPI_MAX_PROCESSOR_NAME]) malloc(bytes);
	strcpy(host_names[rank], host_name);
	for (int n=0; n<size; n++)
		MPI_Bcast(&(host_names[n]),MPI_MAX_PROCESSOR_NAME, MPI_CHAR, n, MPI_COMM_WORLD);
	qsort(host_names, size, sizeof(char[MPI_MAX_PROCESSOR_NAME]), stringCmp);
  
	color = 0;
	for (int n=0; n<size; n++)  {
  		if(n>0 && strcmp(host_names[n-1], host_names[n])) color++;
  		if(strcmp(host_name, host_names[n]) == 0) break;
	}
	free(host_names);

	MPI_Comm_split(MPI_COMM_WORLD, color, rank, &(*comm)->comm_intra);
	//find intranode numbers and make internode communicator
  //figure out mylocal
	MPI_Comm_rank( (*comm)->comm_intra, &mylocal );
  MPI_Comm_size( (*comm)->comm_intra, &numlocal );
  (*comm)->nvrank=mylocal;
  (*comm)->nvsize=numlocal;

  cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
  int core;
  if(mylocal==0) core=50;
  if(mylocal==1) core=58;
  if(mylocal==2) core=18;
  if(mylocal==3) core=26;
  if(mylocal==4) core=114;
  if(mylocal==5) core=122;
  if(mylocal==6) core=82;
  if(mylocal==7) core=90;

	CPU_SET(core,&cpuset);
  if(!getenv("NODOUBLE")) {if(core>128) CPU_SET(core-128,&cpuset); else CPU_SET(core+128,&cpuset);}
	if(getenv("DOPIN")) pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

  if(ndev==numlocal) { //all visible devices
    if(cur_dev!=mylocal) printf("%d: device used %d[%d] ,resetting device to %d\n",rank,cur_dev,ndev,mylocal);
	  CUDACHECK(cudaSetDevice(mylocal));
  }
  (*comm)->mydev=cur_dev;
  //FIXME need to check that numlocal is multiple of pipegpus x tensorgpus
  //ar1 is data
  int divgpus = pipegpus * tensorgpus;
  int datagpus = numlocal/ divgpus;
  (*comm) -> ar_nvsize = datagpus;
  (*comm) -> ar_firstgpu = mylocal - ((mylocal/tensorgpus)%datagpus) * tensorgpus;
  (*comm) -> ar_nvrank = (mylocal - (*comm) -> ar_firstgpu) / tensorgpus;
  //ar2 is tensor
  (*comm) -> ar2_nvsize = tensorgpus;
  (*comm) -> ar2_firstgpu =  mylocal - mylocal % tensorgpus;
  (*comm) -> ar2_nvrank = mylocal - (*comm) -> ar2_firstgpu;
  //ar2 has step equal to ar_nvsize
  int allnodes = nranks/numlocal;
  int mynode = myrank/numlocal;
  int datanodes = allnodes/pipenodes/tensornodes;
  int pipenodegroup_id = myrank/numlocal/(datanodes*tensornodes);

  (*comm)->pipe_id = pipegpus * pipenodegroup_id + mylocal/(datagpus*tensorgpus);

  CUDACHECK(cudaFree(0));
  int datanodegroup_id =  myrank/numlocal/datanodes; //data reduction group node belongs, equals 0 for all if both pipenodes=1 and tensornodes=1
  //mpi communicator only needed for SHARP which is always allreduce1/data-parallel
	MPI_Comm_split(MPI_COMM_WORLD, mylocal + numlocal * datanodegroup_id, rank, &(*comm)->comm_inter);
  //different rails from same group are in different subcommunicators

	MPI_Comm_size( (*comm)->comm_inter, &num_nodes );
	MPI_Comm_rank( (*comm)->comm_inter, &my_node );
  (*comm)->first_node = mynode - my_node;
  (*comm)->num_nodes = num_nodes;
  (*comm)->my_node = my_node;

  (*comm)->num2_nodes = tensornodes;
  (*comm)->my2_node = (mynode/datanodes) % tensornodes;
  (*comm)->first2_node = mynode - (*comm)->my2_node * datanodes;

  char* ib_dev_list;
  int ZIONROCE = getenv("ZIONROCE")?atoi(getenv("ZIONROCE")):0;
  int ROCE = getenv("ROCE")?atoi(getenv("ROCE")):0;
  if(ZIONROCE) ROCE=1;
  int PREOS = getenv("PREOS")?atoi(getenv("PREOS")):0;

	switch(mylocal) {
		case 0:ib_dev_list = "mlx5_0:1"; break;
		case 1:ib_dev_list = (char*)(PREOS?"mlx5_3:1":"mlx5_1:1"); break;
		case 2:ib_dev_list = (char*)(ZIONROCE?"mlx5_4:1":PREOS?"mlx5_4:1":"mlx5_2:1"); break;
		case 3:ib_dev_list = (char*)(PREOS?"mlx5_5:1":"mlx5_3:1"); break;
		case 4:ib_dev_list = (char*)(PREOS?"mlx5_6:1":"mlx5_6:1"); break;
		case 5:ib_dev_list = (char*)(PREOS?"mlx5_9:1":"mlx5_7:1"); break;
		case 6:ib_dev_list = (char*)(ZIONROCE?"mlx5_10:1":PREOS?"mlx5_10:1":"mlx5_8:1"); break;
		case 7:ib_dev_list = (char*)(PREOS?"mlx5_11:1":"mlx5_9:1"); break;
		default:break;	
	}

#ifndef NOSHARP
  //initialize SHARP context for given NIC (one NIC per peer)
	init_spec.progress_func  = NULL;
	init_spec.job_id = (gethostid() << 32) | mylocal;
	MPI_Bcast(&(init_spec.job_id), 1, MPI_LONG, 0, (*comm)->comm_inter);
	init_spec.world_rank = my_node;
	init_spec.world_size = num_nodes;
	init_spec.world_local_rank = 0;
	init_spec.enable_thread_support = 0;
	init_spec.oob_colls.barrier = oob_barrier;
	init_spec.oob_colls.bcast = oob_bcast;
	init_spec.oob_colls.gather = oob_gather;
	init_spec.oob_ctx = *comm;
	init_spec.config = sharp_coll_default_config;
  init_spec.config.ib_dev_list=ib_dev_list;
#endif

  (*comm)->fifo=(ub_request*)malloc(sizeof(ub_request)*MAX_REQUESTS);
  (*comm)->nblocks=getenv("NBLOCKS")?atoi(getenv("NBLOCKS")):8;
  (*comm)->alignblock=1024*(getenv("ALIGNBLOCK")?atoi(getenv("ALIGNBLOCK")):512);
  (*comm)->minblock=1024*(getenv("MINBLOCK")?atoi(getenv("MINBLOCK")):2*1024);
  (*comm)->asyncblocks=getenv("ASYNCBLOCKS")?atoi(getenv("ASYNCBLOCKS")):16;
#ifndef NOSHARP
	ret = sharp_coll_init(&init_spec, &((*comm)->sharp_coll_context));
	if (ret < 0) {
		fprintf(stderr, "sharp_coll_init failed: %s\n", sharp_coll_strerror(ret));
		return -1;
	}

	/* create sharp group */
	comm_spec.rank = my_node;
	comm_spec.size = num_nodes;
	comm_spec.oob_ctx = (void*)(*comm);
	comm_spec.group_world_ranks = NULL;
	ret = sharp_coll_comm_init((*comm)->sharp_coll_context, &comm_spec, &((*comm)->sharp_coll_comm));

	if (ret < 0) {
		fprintf(stderr, "sharp communicator creation failed: %s\n", sharp_coll_strerror(ret));
		return -1;
	}
#endif

  CUDACHECK(cudaMallocHost((void**)&(*comm)->hostflags,(MAX_SMS+100)*sizeof(int)));
  for(int i=0;i<100+MAX_SMS;i++) (*comm)->hostflags[i]=0;
  _mm_mfence();
  sleep(1);

#else
  if(ndev>1) { //all visible devices
    if(cur_dev!=myrank%ndev) printf("%d: device used %d[%d] ,resetting device to %d\n",myrank,cur_dev,ndev,myrank);
	  CUDACHECK(cudaSetDevice(myrank%ndev));
  }
  (*comm)->mydev = cur_dev;
  (*comm)-> nvrank = myrank;
  (*comm)-> nvsize = nranks;

  int divgpus = pipegpus * tensorgpus;
  int datagpus = nranks/divgpus;
  (*comm) -> ar_nvsize = datagpus;
  (*comm) -> ar_firstgpu = myrank - ((myrank/tensorgpus)%datagpus) * tensorgpus;
  (*comm) -> ar_nvrank = (myrank - (*comm) -> ar_firstgpu) / tensorgpus;
  //ar2 is tensor
  (*comm) -> ar2_nvsize = tensorgpus;
  (*comm) -> ar2_firstgpu = myrank - myrank % tensorgpus;
  (*comm) -> ar2_nvrank = myrank - (*comm) -> ar2_firstgpu;

  (*comm)-> pipe_id = myrank / (datagpus * tensorgpus);
  MPI_Comm_dup(MPI_COMM_WORLD,&(*comm)->comm_intra);
#endif

//init_p2p_transport();
  (*comm)->ibnvsize = getenv("IBNVSIZE")?atoi(getenv("IBNVSIZE")):(*comm)->nvsize;
#ifdef UCP
  if(!(*comm)->myrank && getenv("UBDEBUG")) printf("IBNVSIZE set to %d\n",(*comm)->ibnvsize);fflush(NULL);
  ucp_params_t ucp_params;
  ucp_worker_params_t worker_params;
  ucp_config_t *config;

  (*comm)->ucxep = (ucp_ep_h*) malloc((*comm)->nranks*sizeof(ucp_ep_h)); //endpoint to all peers

  memset(&ucp_params, 0, sizeof(ucp_params));
  memset(&worker_params, 0, sizeof(worker_params));

  UCXCHECK(ucp_config_read(NULL, NULL, &config));
  if(ROCE) {
    UCXCHECK(ucp_config_modify(config,"TLS","rc"));
    UCXCHECK(ucp_config_modify(config,"IB_GID_INDEX","3"));
    UCXCHECK(ucp_config_modify(config,"IB_TRAFFIC_CLASS","96"));
  } else {
    UCXCHECK(ucp_config_modify(config,"TLS","rc_x"));
    UCXCHECK(ucp_config_modify(config,"IB_SL","1"));
  }

  UCXCHECK(ucp_config_modify(config,"ZCOPY_THRESH","0"));
  if(ndev>1) UCXCHECK(ucp_config_modify(config,"NET_DEVICES",ib_dev_list));

  ucp_params.field_mask   = UCP_PARAM_FIELD_FEATURES;

  ucp_params.features     = UCP_FEATURE_TAG   |
                            UCP_FEATURE_RMA   |
                            UCP_FEATURE_AMO32 |
                            UCP_FEATURE_AMO64;


  UCXCHECK(ucp_init(&ucp_params, config, &(*comm)->ucp_context));
  MPI_Barrier(MPI_COMM_WORLD);
  if(!(*comm)->myrank && getenv("UBDEBUG")) ucp_config_print(config, stdout, NULL, UCS_CONFIG_PRINT_CONFIG);
  ucp_config_release(config);

  worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
  
  ucp_address_t* myucxaddr;
  UCXCHECK(ucp_worker_create((*comm)->ucp_context, &worker_params, &(*comm)->ucp_worker));
  UCXCHECK(ucp_worker_get_address((*comm)->ucp_worker, &myucxaddr, &(*comm)->ucx_addr_len));
  (*comm)->ucxaddr = malloc((*comm)->nranks*(*comm)->ucx_addr_len); //peer adresses
  memcpy((*comm)->ucxaddr+(*comm)->myrank*(*comm)->ucx_addr_len,myucxaddr,(*comm)->ucx_addr_len);
  MPI_Allgather((*comm)->ucxaddr+(*comm)->myrank*(*comm)->ucx_addr_len, (*comm)->ucx_addr_len, MPI_BYTE, (*comm)->ucxaddr, (*comm)->ucx_addr_len, MPI_BYTE,MPI_COMM_WORLD);

  ucp_ep_params_t ep_params;
  ep_params.field_mask      =  UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
 
  for(int r=0;r<(*comm)->nranks;r++) 
    (*comm)->ucxep[r]=NULL;

  //preconnect all peers (for alltoall)
  for(int r=0;r<(*comm)->nranks;r++) {
    if(r==(*comm)->myrank) continue;
    ep_params.address         = (ucp_address_t*) ((*comm)->ucxaddr+(*comm)->ucx_addr_len*r);
    UCXCHECK(ucp_ep_create((*comm)->ucp_worker, &ep_params, &(*comm)->ucxep[r]));
  }
  #if 0
  //preconnect pipeline neighbours (prev and next only)
  for(int r=-1;r<2;r++) {
    if(r==0) continue;
    //printf("[%d] adjust %d piperank %d\n",(*comm)->myrank,r,pipe_rank(*comm,r));fflush(NULL);
    ep_params.address         = (ucp_address_t*) ((*comm)->ucxaddr+(*comm)->ucx_addr_len*pipe_rank(*comm,r));
    UCXCHECK(ucp_ep_create((*comm)->ucp_worker, &ep_params, &(*comm)->ucxep[pipe_rank(*comm,r)]));
  }

#ifdef NOSHARP
  //preconnect all data-allreduce rail partners
  for(int r=0;r<(*comm)->num_nodes;r++) {
    if(r==(*comm)->my_node) continue;
    int dest=((*comm)->first_node+r)*(*comm)->nvsize+(*comm)->nvrank;
    //printf("[%d] adjust %d piperank %d\n",(*comm)->myrank,r,pipe_rank(*comm,r));fflush(NULL);
    ep_params.address         = (ucp_address_t*) ((*comm)->ucxaddr+(*comm)->ucx_addr_len*dest);
    UCXCHECK(ucp_ep_create((*comm)->ucp_worker, &ep_params, &(*comm)->ucxep[dest]));
  }
#endif
  //preconnect all tensor-allreduce rail partners
  for(int r=0;r<(*comm)->num2_nodes;r++) {
    if(r==(*comm)->my2_node) continue;
    int dest=((*comm)->first2_node+r*(*comm)->num_nodes)*(*comm)->nvsize+(*comm)->nvrank;
    //printf("[%d] adjust %d piperank %d\n",(*comm)->myrank,r,pipe_rank(*comm,r));fflush(NULL);
    ep_params.address         = (ucp_address_t*) ((*comm)->ucxaddr+(*comm)->ucx_addr_len*dest);
    UCXCHECK(ucp_ep_create((*comm)->ucp_worker, &ep_params, &(*comm)->ucxep[dest]));
  }
#endif

#endif

#ifdef NOSHARP
#define NBUF 2
#else
#define NBUF 1
#endif

#define LOCALSIZE 4*(REG0_OFFSET(*comm)+REG0_FLAGS+REG0_COMMBUFFER*NBUF)
//peer pointers + op flags + comm buffer

#ifndef MNNVL
  CUDACHECK(cudaMalloc(&(*comm)->gpu_ptrs,LOCALSIZE)); // flags and pointers, no block data yet
  CUDACHECK(cudaMemset((*comm)->gpu_ptrs,0,LOCALSIZE));
  CUDACHECK(cudaDeviceSynchronize());
  register_user_buffer_collective(&((*comm)->gpu_ptrs),LOCALSIZE,*comm); //will use handler 0
#else
  register_user_buffer_collective(&((*comm)->gpu_ptrs),LOCALSIZE,*comm,true); //will use handler 0
#endif

  CUDACHECK(cudaMalloc(&(*comm)->send_id,(*comm)->nranks*sizeof(int)));
  CUDACHECK(cudaMalloc(&(*comm)->recv_id,MAX_REGIONS*(*comm)->nranks*sizeof(int)));
  CUDACHECK(cudaMemset((*comm)->send_id,0,(*comm)->nranks*sizeof(int)));
  CUDACHECK(cudaMemset((*comm)->recv_id,0,MAX_REGIONS*(*comm)->nranks*sizeof(int)));
  (*comm)->sms=getenv("MAXSMS")?atoi(getenv("MAXSMS")):16;
  (*comm)->threads=getenv("MAXTHREADS")?atoi(getenv("MAXTHREADS")):1024;

#ifdef MULTINODE
#define GPU_PAGE_SHIFT   16
#define GPU_PAGE_SIZE    (1UL << GPU_PAGE_SHIFT)
#define GPU_PAGE_OFFSET  (GPU_PAGE_SIZE-1)
#define GPU_PAGE_MASK    (~GPU_PAGE_OFFSET)
 CUDACHECK(cudaMalloc(&(*comm)->flags,2*GPU_PAGE_SIZE));
 unsigned int flag = 1;
 //cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr)(*comm)->flags);
 CUDACHECK(cudaMemset((*comm)->flags,0,2*GPU_PAGE_SIZE));
 (*comm)->flags = (int*) (((CUdeviceptr)(*comm)->flags + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK);

using namespace std;
  (*comm)->g = gdr_open();
  if ((*comm)->g==NULL) {
    		fprintf(stderr, "gdrcopy open failed\n");
		    return -1;
  }
  gdr_mh_t mh;
  ret = gdr_pin_buffer((*comm)->g, (CUdeviceptr)(*comm)->flags, GPU_PAGE_SIZE, 0, 0, &mh);
  if(ret) {
    		fprintf(stderr, "gdr_pin_buffer failed\n");
		    return -1;
  }
  ret = gdr_map((*comm)->g, mh, (void**)&((*comm)->map_flags), GPU_PAGE_SIZE);

  if(ret) {
    		fprintf(stderr, "gdr_map failed\n");
		    return -1;
  }
  sched_param param;
  pthread_attr_t attr;
  pthread_attr_init (&attr);
  pthread_attr_getschedparam (&attr, &param);
  param.sched_priority = sched_get_priority_max(SCHED_FIFO);;
  pthread_attr_setschedparam (&attr, &param);

  // Disable proxy thread
  //ret = pthread_create(&(*comm)->proxythread,&attr,&proxythread,*comm);

  if(getenv("UBDEBUG")) printf("%d/%d:(%d x %d): DP %d x %d TP %d x %d, DPGROUP %dx%d TPGROUP %dx%d PIPE_ID %d/%d\n",myrank,nranks,myrank/numlocal,myrank%numlocal,(*comm)->my_node,(*comm)->ar_nvrank,
          (*comm)->my2_node,(*comm)->ar2_nvrank,
          (*comm)->num_nodes,(*comm)->ar_nvsize,
          (*comm)->num2_nodes,(*comm)->ar2_nvsize,
          (*comm)->pipe_id,pipegpus*pipenodes);
  fflush(NULL);
#endif

  return 0;
}
int create_communicator_grouped( communicator** comm, int pipegpus, int pipenodes) 
{ return create_communicator_grouped2(comm,pipegpus,pipenodes,1,1); }

int create_communicator( communicator** comm ) {
  return create_communicator_grouped2(comm,1,1,1,1);
}


void destroy_communicator(communicator* comm) {
#ifdef MULTINODE
  comm->activeproxy=0;
  if(!comm->myrank && getenv("UBDEBUG")) printf("waiting for userbuffers proxy thread to exit()\n");
  // Disable proxy thread
  //pthread_join(comm->proxythread,NULL);
  gdr_close(comm->g);
#endif
}

int register_user_buffer_collective(void** gpubuff, size_t bytes, communicator* comm, bool alloc) {
  if(comm->free_region > MAX_REGIONS) return -1;
  int hndl = comm->free_region;
  //printf("%d register %d size %lld\n",comm->myrank,hndl,bytes);fflush(NULL);
  comm->peer_ptr[hndl]=(void**)malloc(sizeof(void*)*(comm->nvsize));

if(alloc) {
#ifdef MNNVL
  size_t aligned_size;
  int nranks=comm->nvsize; //total GPUs in NVLINK domain
  CUmemFabricHandle *exphndl=(CUmemFabricHandle *)malloc(nranks*sizeof(CUmemFabricHandle));
  CUmemGenericAllocationHandle *remhndls=(CUmemGenericAllocationHandle *)malloc(nranks*sizeof(CUmemGenericAllocationHandle));
  void** remptrs=(void**)malloc(nranks*sizeof(void*));

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = comm->mydev;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
  size_t granularity = 0;
  CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  //MPI_Allreduce MAX of granularity check
  aligned_size = (bytes + granularity - 1) / granularity * granularity;
  
  prop.location.id = comm->mydev;
  CUCHECK(cuMemCreate(&(comm->mhndl[hndl]), aligned_size, &prop, 0));
  CUCHECK(cuMemExportToShareableHandle(static_cast<void *>(&exphndl[comm->nvrank]),comm->mhndl[hndl], CU_MEM_HANDLE_TYPE_FABRIC, 0));
  MPI_Allgather(&exphndl[comm->nvrank], sizeof(CUmemFabricHandle), MPI_BYTE, exphndl, sizeof(CUmemFabricHandle), MPI_BYTE,comm->comm_intra);
  for(int i=0;i<nranks;i++) {
    CUdeviceptr ptr;
    CUCHECK(cuMemAddressReserve(&ptr, aligned_size, 0, 0, 0));

    if(i==comm->nvrank) {
      CUCHECK(cuMemMap(ptr, aligned_size, 0, comm->mhndl[hndl], 0));
      remptrs[i]=reinterpret_cast<void *>(ptr);
      //printf("%d:mydev %d hndl%d, ptr%d=%lx\n",comm->nvrank,comm->mydev,hndl,i,remptrs[i]);fflush(NULL);
      if(hndl) *gpubuff=remptrs[i]; else comm->gpu_ptrs=remptrs[i];
    } else {
      CUCHECK(cuMemImportFromShareableHandle(&remhndls[i], reinterpret_cast<void *>(&exphndl[i]),CU_MEM_HANDLE_TYPE_FABRIC));
      CUCHECK(cuMemMap(ptr, aligned_size, 0, remhndls[i], 0));
      remptrs[i]=reinterpret_cast<void *>(ptr);
    }
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    accessDesc.location.id = comm->mydev;
	  CUCHECK(cuMemSetAccess(ptr, aligned_size, &accessDesc, 1));
    comm->peer_ptr[hndl][i]=remptrs[i];
  }

  if(hndl==0) CUDACHECK(cudaMemset(comm->gpu_ptrs,0,aligned_size));
  CUDACHECK(cudaMemcpy(((char*)(comm->gpu_ptrs))+(hndl*nranks*sizeof(void*)),remptrs,nranks*sizeof(void*),cudaMemcpyHostToDevice));
  free(remhndls);
  free(exphndl);
  free(remptrs);
} else {
#else
CUDACHECK(cudaMalloc(gpubuff,bytes)); }
#endif
  assert(comm->nvsize<=8);
  cudaIpcMemHandle_t *memhndl=(cudaIpcMemHandle_t*)malloc(sizeof(cudaIpcMemHandle_t)*(comm->nvsize));
  
  CUDACHECK(cudaIpcGetMemHandle(&memhndl[comm->nvrank],*gpubuff));
  
  MPI_Allgather(&memhndl[comm->nvrank], sizeof(cudaIpcMemHandle_t), MPI_BYTE, memhndl, sizeof(cudaIpcMemHandle_t), MPI_BYTE,comm->comm_intra);

  for(int i=0;i<comm->nvsize;i++) 
    if(i!=comm->nvrank) CUDACHECK(cudaIpcOpenMemHandle((void**)&(comm->peer_ptr[hndl][i]),memhndl[i], cudaIpcMemLazyEnablePeerAccess));
  comm->peer_ptr[hndl][comm->nvrank]=*gpubuff;
  CUDACHECK(cudaDeviceSynchronize());

  CUDACHECK(cudaMemcpy((char*)(comm->gpu_ptrs)+(hndl*comm->nvsize*sizeof(void*)),comm->peer_ptr[hndl],comm->nvsize*sizeof(void*),cudaMemcpyHostToDevice));

  CUDACHECK(cudaDeviceSynchronize());
  free(memhndl);
#ifdef MNNVL
}
#endif
  comm->mem_ptr[hndl]=*gpubuff;
#ifdef MULTINODE
#ifndef NOSHARP
  if (sharp_coll_reg_mr(comm->sharp_coll_context,*gpubuff, bytes, &(comm->mem_mr[comm->free_region])) != SHARP_COLL_SUCCESS) return -2;
#endif

#ifdef UCP
  ucp_mem_h ucp_memh;
  ucp_mem_map_params_t ucp_memmap_params;
  ucp_memmap_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS | 
	                           UCP_MEM_MAP_PARAM_FIELD_LENGTH;
  ucp_memmap_params.address    = (void *) *gpubuff;
  ucp_memmap_params.length     = bytes;
  UCXCHECK(ucp_mem_map(comm->ucp_context, &ucp_memmap_params, &ucp_memh));
  void *myrkey;

  size_t rkeysize;
  UCXCHECK(ucp_rkey_pack(comm->ucp_context, ucp_memh, &myrkey, &rkeysize));
  MPI_Allreduce(&rkeysize,&comm->rkey_size[hndl],1,MPI_LONG,MPI_MAX,MPI_COMM_WORLD);
  comm->rkeys_packed[hndl] = malloc(comm->nranks*comm->rkey_size[hndl]); //peer memory rkeys
  memcpy(comm->rkeys_packed[hndl]+comm->myrank*comm->rkey_size[hndl],myrkey,rkeysize);
  MPI_Allgather(comm->rkeys_packed[hndl]+comm->myrank*comm->rkey_size[hndl], comm->rkey_size[hndl], MPI_BYTE, comm->rkeys_packed[hndl], comm->rkey_size[hndl], MPI_BYTE,MPI_COMM_WORLD);
  comm->rkeys[hndl]=(ucp_rkey_h*)malloc(comm->nranks*sizeof(ucp_rkey_h));
  comm->peeraddr[hndl] = (void**)malloc(comm->nranks*sizeof(void*));
  comm->peeraddr[hndl][comm->myrank]=*gpubuff;
  MPI_Allgather(&comm->peeraddr[hndl][comm->myrank], sizeof(void*), MPI_BYTE, &comm->peeraddr[hndl][0], sizeof(void*), MPI_BYTE, MPI_COMM_WORLD);
  for(int r=0;r<comm->nranks;r++) {
      //if(r==comm->myrank) continue;
      if(comm->ucxep[r]==NULL) continue;
      UCXCHECK(ucp_ep_rkey_unpack(comm->ucxep[r],comm->rkeys_packed[hndl]+r*comm->rkey_size[hndl], &comm->rkeys[hndl][r]));
  }
  #endif
  #endif

  return comm->free_region++;
}


int allreduce_userbuff_inplace_gpu(const int handler, const int offset, const int elements, const int blocksize, communicator* comm, cudaStream_t stream);
int allreduce2_userbuff_inplace_gpu(const int maxcredit,const int handler, const int offset, const int elements, const int blocksize, communicator* comm, cudaStream_t stream,int op);
#if defined(MULTINODE) && defined(NOSHARP)
int reducescatter2_userbuff_inplace_gpu(const int maxcredit,const int handler, const int offset, const int elements, const int blocksize, communicator* comm, cudaStream_t stream,int op);
int allgather2_userbuff_inplace_gpu(const int maxcredit,const int handler, const int offset, const int elements, const int blocksize, communicator* comm, cudaStream_t stream,int op);
#endif

#ifdef MULTINODE
#ifndef NOSHARP
void progress_sharp_allreduce(communicator * comm) {
  const int op = userbuffers_allreduceop_sharp;
  ub_request *req = &comm->active_req[op];
  volatile int * hf = (volatile int*) comm->hostflags;
  int expecting = req->nblock + 1 + req->basecounter;

  if(req->active==-1) return; // no active op

  if(req->active==0) { //first run, initialize
    //printf("%d: ARSHARP proxy start op size %d blocks %d basecounter %d\n",comm->myrank,req->elements*2,req->numblocks,req->basecounter);fflush(NULL);
    req->nblock=0;
    req->unconfirmed_ib_in_flight=0;
    req->numblocks=(req->elements*2+req->blocksize-1)/req->blocksize;
    req->active=1;
    return;
  }

  if( req->unconfirmed_ib_in_flight < req->nblock ) //any pending sharp in flight
  if( req->nblock == req->numblocks || (req->nblock<req->numblocks && ( req->unconfirmed_ib_in_flight + comm->asyncblocks == req->nblock || hf[op] < expecting) ) ) {
      if(sharp_coll_req_test(comm->sharphndl[req->unconfirmed_ib_in_flight%MAX_SHARP]))
      {
        comm->map_flags[GF_IBSHARPDONE]=req->basecounter+req->unconfirmed_ib_in_flight+1; 
        _mm_mfence();
        req->unconfirmed_ib_in_flight++;
        if(req->nblock == req->numblocks && req->unconfirmed_ib_in_flight == req->numblocks) {
          //printf("%d: AR SHARP proxy complete op size %d blocks %d basecounter %d\n",comm->myrank,req->elements*2,req->numblocks,req->basecounter);fflush(NULL);
          req->active=-1; //operation complete on CPU side
          comm->active_nreqs--;
        }
      }
    return;
  }

  if(req->nblock < req->numblocks && hf[op] >= expecting) {
    struct sharp_coll_reduce_spec reduce_spec;

    reduce_spec.sbuf_desc.buffer.mem_handle = comm->mem_mr[req->handler];
    reduce_spec.sbuf_desc.type = SHARP_DATA_BUFFER;
    reduce_spec.sbuf_desc.mem_type = SHARP_MEM_TYPE_CUDA;
    reduce_spec.rbuf_desc.buffer.mem_handle = comm->mem_mr[req->handler];
    reduce_spec.rbuf_desc.type = SHARP_DATA_BUFFER;
    reduce_spec.rbuf_desc.mem_type = SHARP_MEM_TYPE_CUDA;
    reduce_spec.aggr_mode = SHARP_AGGREGATION_NONE;
    reduce_spec.dtype = SHARP_DTYPE_FLOAT_SHORT;
    reduce_spec.op = SHARP_OP_SUM;

    long peerblock = req->blocksize/comm->ar_nvsize;
    long blockstart=req->nblock*req->blocksize;
    long adder = req->offset*2+blockstart+peerblock*comm->ar_nvrank;
      
    if(blockstart+req->blocksize>req->elements*2) { //last block might be shorter
      peerblock = (req->elements*2-blockstart)/comm->ar_nvsize;
      adder = req->offset*2+blockstart+peerblock*comm->ar_nvrank;
    }
    reduce_spec.sbuf_desc.buffer.ptr = (char*)(comm->mem_ptr[req->handler])+adder;
    reduce_spec.rbuf_desc.buffer.ptr = (char*)(comm->mem_ptr[req->handler])+adder;
    reduce_spec.sbuf_desc.buffer.length = peerblock;
    reduce_spec.rbuf_desc.buffer.length = peerblock;
    reduce_spec.length = peerblock/2;

#ifdef NOSHARP
      //MPI_Iallreduce(reduce_spec.sbuf_desc.buffer.ptr,reduce_spec.rbuf_desc.buffer.ptr,reduce_spec.length,MPI_SHORT,MPI_SUM,comm_inter,&comm->mpihndl[req->nblock%MAX_SHARP]);
      MPI_Ibarrier(comm->comm_inter,&comm->mpihndl[req->nblock%MAX_SHARP]);
      //comm->mpihndl[req->nblock%MAX_SHARP]=NULL;
#else
      int ret = sharp_coll_do_allreduce_nb(comm->sharp_coll_comm, &reduce_spec,&comm->sharphndl[req->nblock%MAX_SHARP]);
      if (ret != SHARP_COLL_SUCCESS && !comm->my_node) fprintf(stderr, "Allreduce failed: %s\n",sharp_coll_strerror(ret));
#endif
    req->nblock++;
    return;
  }
}
#endif
#ifdef UCP
void progress_ucp_allreduce(communicator * comm, int op) {
  const int num_nodes =  op==userbuffers_allreduceop_nonsharp? comm->num_nodes : comm->num2_nodes;
  const int my_node =  op==userbuffers_allreduceop_nonsharp? comm->my_node : comm->my2_node;
  const int first_node = op==userbuffers_allreduceop_nonsharp? comm->first_node : comm->first2_node;
  const int step_node = op==userbuffers_allreduceop_nonsharp? 1 : comm->num_nodes;
  const int ar_nvsize = op==userbuffers_allreduceop_nonsharp? comm->ar_nvsize : comm->ar2_nvsize;
  const int ar_nvrank = op==userbuffers_allreduceop_nonsharp? comm->ar_nvrank : comm->ar2_nvrank;

  ub_request *req = &comm->active_req[op];
  volatile int * hf = (volatile int*) comm->hostflags;
  int expecting = req->nblock + 1 + req->basecounter;

  if(req->active==-1) return; // no active op

  if(req->active==0) { //first run, initialize
    req->nblock=0;
    //req->unconfirmed_ib_in_flight=0;
    req->numblocks=(req->elements*2+req->blocksize-1)/req->blocksize;
    req->active=1;
    //printf("%d: AR UCP(%d) proxy start op size %d blocks %d blocksize %d  basecounter %d maxcredit %d\n",comm->myrank,op,req->elements*2,req->numblocks,req->blocksize,req->basecounter,req->maxcredit);fflush(NULL);
    return;
  }

  if(req->nblock == 2*req->numblocks) {
    //printf("%d: AR UCP(%d) proxy complete op size %d blocks %d basecounter %d\n",comm->myrank,op,req->elements*2,req->numblocks,req->basecounter);fflush(NULL);
    req->active=-1; //operation complete on CPU side
    comm->active_nreqs--;
    return;
  }

  if(req->nblock < req->numblocks && hf[HF_NVRSDONE+(op&1)] >= expecting && hf[HF_NVREDUCEDONE+(op&1)]>=expecting-req->maxcredit) {
  //launch IBRS
  //printf("%d: proxy IBRS START size %d blocks %d basecounter %d\n",comm->myrank,req->elements*2,req->numblocks,req->basecounter);fflush(NULL);

    int commblock = req->nblock % req->maxcredit;
    void* userbuf = comm->mem_ptr[req->handler];
    size_t peerblock = req->blocksize/ar_nvsize;
    size_t ibblock = peerblock/num_nodes;

    size_t blockstart=req->nblock*req->blocksize;
    size_t useroffset = req->offset*2 + blockstart;//+peerblock*comm->ar_nvrank+ibblock*comm->my_node;
    //printf("bytes %ld blocks: %dx%d(ibblock/peerblock %d/%d) nblock %d blockstart %ld useroffset %ld\n",req->elements*2,req->numblocks,req->blocksize,ibblock,peerblock,req->nblock,blockstart,useroffset);fflush(NULL);
    size_t commoffset = sizeof(int)*(REG0_OFFSET(comm)+REG0_FLAGS)+commblock*peerblock;
    if(op==userbuffers_allreduceop_nonsharp) commoffset+=sizeof(int)*REG0_COMMBUFFER;

    if(blockstart+req->blocksize>req->elements*2) { //last block might be shorter
      peerblock = (req->elements*2-blockstart)/ar_nvsize;
      ibblock=peerblock/num_nodes;
    }
    for(int n=1;n<num_nodes;n++) {
      int node = (my_node+n)%num_nodes;
      //if(node==comm->my_node) continue;
      const int dest = (first_node+node*step_node)*comm->nvsize+comm->nvrank;
      const int flagoffset=((REG0_OFFSET(comm)+REG0_OPFLAGS*op+REG0_IBRS+my_node)*sizeof(int));
      //printf("proxy reg0_offset %d flags %d total %d\n",REG0_OFFSET(comm),REG0_FLAGS,REG0_OFFSET(comm)+REG0_FLAGS);
      //printf("IBRS %d(%d,%d): node %d/%d dest %d ibblock %d elts %d addr %lx commoffset %d\n",comm->myrank,comm->my_node,comm->nvrank,node,comm->num_nodes,dest,ibblock,req->elements,(uint64_t)(comm->peeraddr[0][dest]+commoffset+ibblock*my_node),ibblock*my_node);fflush(NULL);
      UCXPUTCHECK(ucp_put_nbi(comm->ucxep[dest],userbuf+useroffset+peerblock*ar_nvrank+ibblock*node,ibblock,(uint64_t)(comm->peeraddr[0][dest]+commoffset+ibblock*my_node),comm->rkeys[0][dest]));
      UCXCHECK(ucp_atomic_add32(comm->ucxep[dest],1,(uint64_t)(comm->peeraddr[0][dest]+flagoffset),comm->rkeys[0][dest]));
    }
    req->nblock++;
    return;
  }

  if(req->nblock >= req->numblocks && hf[HF_NVREDUCEDONE+(op&1)] >= expecting-req->numblocks) {
    //launch IBAG
    //printf("%d: proxy IBAG START size %d blocks %d basecounter %d\n",comm->myrank,req->elements*2,req->numblocks,req->basecounter);fflush(NULL);

    void* userbuf = comm->mem_ptr[req->handler];
    size_t peerblock = req->blocksize/ar_nvsize;
    size_t ibblock = peerblock/num_nodes;

    size_t blockstart=(req->nblock-req->numblocks)*req->blocksize;
    size_t useroffset = req->offset*2 + blockstart;
    
          
    if(blockstart+req->blocksize>req->elements*2) { //last block might be shorter
      peerblock = (req->elements*2-blockstart)/ar_nvsize;
      ibblock=peerblock/num_nodes;
    }
    useroffset+=ibblock*my_node+peerblock*ar_nvrank;
//printf("IBAG bytes %ld blocks: %dx%d(ibblock/peerblock %d/%d) nblock %d blockstart %ld useroffset %ld\n",req->elements*2,req->numblocks,req->blocksize,ibblock,peerblock,req->nblock,blockstart,useroffset);fflush(NULL);
    for(int n=1;n<num_nodes;n++) {
      int node = (my_node+n)%num_nodes;
      //if(node==comm->my_node) continue;
      const int dest = (first_node+node*step_node)*comm->nvsize+comm->nvrank;
      const int flagoffset=((REG0_OFFSET(comm)+REG0_OPFLAGS*op+REG0_IBAG+my_node)*sizeof(int));
      UCXPUTCHECK(ucp_put_nbi(comm->ucxep[dest],userbuf+useroffset,ibblock,(uint64_t)(comm->peeraddr[req->handler][dest]+useroffset),comm->rkeys[req->handler][dest]));
      UCXCHECK(ucp_atomic_add32(comm->ucxep[dest],1,(uint64_t)(comm->peeraddr[0][dest]+flagoffset),comm->rkeys[0][dest]));
    }
    req->nblock++;
    return;
  }
}

void progress_ucp_send(communicator * comm) {
  const int op = userbuffers_sendop;
  ub_request *req = &comm->active_req[op];
  volatile int * hf = (volatile int*) comm->hostflags;
  int expecting = 1 + req->basecounter;

  if(req->active==-1) return; // no active op

  if(req->active==0) { //first run, initialize
    req->active=1;
    const int dest = req->peer;

    if(comm->ucxep[dest]==NULL) { //preconnect on first send
      ucp_ep_params_t ep_params;
      ep_params.field_mask      =  UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
      ep_params.address         = (ucp_address_t*) ((comm)->ucxaddr+(comm)->ucx_addr_len*dest);
      UCXCHECK(ucp_ep_create((comm)->ucp_worker, &ep_params, &(comm)->ucxep[dest])); 
      for(int hndl=0;hndl<comm->free_region;hndl++)
        UCXCHECK(ucp_ep_rkey_unpack(comm->ucxep[dest],comm->rkeys_packed[hndl]+dest*comm->rkey_size[hndl], &comm->rkeys[hndl][dest]));
    }
    return;
  }

  if(hf[op]>=expecting) {
    const int dest = req->peer;
    const size_t bytes = req->elements;
    const int flagoffset=((REG0_OFFSET(comm)+REG0_RECV+comm->myrank*MAX_REGIONS+req->handler2)*sizeof(int));
    const void* srcaddr = comm->mem_ptr[req->handler]+req->offset;
    const int dsthndl = req->handler2;
    void *dstaddr = comm->peeraddr[dsthndl][dest]+req->offset2;
    if (bytes) ucp_put_nbi(comm->ucxep[dest],srcaddr,bytes,(uint64_t)dstaddr,comm->rkeys[dsthndl][dest]);
    UCXCHECK(ucp_atomic_add32(comm->ucxep[dest],1,(uint64_t)(comm->peeraddr[0][dest]+flagoffset),comm->rkeys[0][dest]));
    comm->active_nreqs--;
    req->active=-1;
  } else ucp_worker_progress(comm->ucp_worker);

}

void progress_ucp_alltoall(communicator * comm) {
  const int op = userbuffers_alltoall;
  ub_request *req = &comm->active_req[op];
  volatile int * hf = (volatile int*) comm->hostflags;
  int expecting = 1 + req->basecounter;

  if(req->active==-1) return; // no active op

  if(req->active==0) { //first run, initialize
    req->active=1;
    if(0)
    for(int dest=0;dest<comm->nranks;dest++)
    if(dest!=comm->myrank && comm->ucxep[dest]==NULL) { //preconnect if needed
      ucp_ep_params_t ep_params;
      ep_params.field_mask      =  UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
      ep_params.address         = (ucp_address_t*) ((comm)->ucxaddr+(comm)->ucx_addr_len*dest);
      UCXCHECK(ucp_ep_create((comm)->ucp_worker, &ep_params, &(comm)->ucxep[dest])); 
      for(int hndl=0;hndl<comm->free_region;hndl++)
        UCXCHECK(ucp_ep_rkey_unpack(comm->ucxep[dest],comm->rkeys_packed[hndl]+dest*comm->rkey_size[hndl], &comm->rkeys[hndl][dest]));
    }
    return;
  }

  if(hf[op]>=expecting) {
    const size_t bytes = req->elements;
    const int flagoffset=((REG0_OFFSET(comm)+REG0_OPFLAGS*op)*sizeof(int));
    const void* srcaddr = comm->mem_ptr[req->handler]+req->offset;
    const int dsthndl = req->handler2;
    for(int i=1;i<comm->nranks;i++) {
      int dest = (i+comm->myrank)%comm->nranks;
      void *dstaddr = comm->peeraddr[dsthndl][dest]+req->offset2+(dest*bytes);
      if (bytes) ucp_put_nbi(comm->ucxep[dest],srcaddr+(dest*bytes),bytes,(uint64_t)dstaddr,comm->rkeys[dsthndl][dest]);
      UCXCHECK(ucp_atomic_add32(comm->ucxep[dest],1,(uint64_t)(comm->peeraddr[0][dest]+flagoffset),comm->rkeys[0][dest]));
    }
    comm->active_nreqs--;
    req->active=-1;
  } else ucp_worker_progress(comm->ucp_worker);

}

#endif


#endif

void allreduce_nonsharp_inplace(const int handler,const int offset,const int elements,communicator* comm,cudaStream_t stream,int op) {
  if(elements<64) return; //sorry not implemented
  //if(comm->myrank==0) fprintf(stderr,"AR2(%d) user call launch_mode=%d\n",op,comm->launch_mode);
  const int ar_nvsize = op==userbuffers_allreduceop_nonsharp? comm->ar_nvsize : comm->ar2_nvsize;
  int blocksize=elements*2;
  int maxcredit=0;
#ifdef MULTINODE
  const int num_nodes = op==userbuffers_allreduceop_nonsharp? comm->num_nodes : comm->num2_nodes;
  blocksize = (comm->nblocks-1+(comm->alignblock-1+elements*2)/comm->alignblock)/comm->nblocks; //FIXME TUNING
  blocksize*=comm->alignblock;
  if(blocksize<comm->minblock) blocksize=comm->minblock;

  maxcredit = (elements*2+blocksize-1)/blocksize;
  //if(maxcredit>4) maxcredit=4;
  //if(maxcredit>4 && ar_nvsize==1) maxcredit=4;
  size_t peerblock = sizeof(int)*REG0_COMMBUFFER/maxcredit; //max size we can fit
  if(blocksize>peerblock*ar_nvsize) blocksize = peerblock*ar_nvsize;
#endif
  //blocksize=elements*2;
  int sms = allreduce2_userbuff_inplace_gpu(maxcredit,handler, offset, elements, blocksize, comm, stream,op);

#ifdef MULTINODE
  if(num_nodes>1 && comm->launch_mode & LAUNCH_CPU) 
  {
  if(!sms) return;
  comm->fifo[comm->head].optype=op;
  comm->fifo[comm->head].basecounter = comm->basecounter[op];
  comm->fifo[comm->head].blocksize = blocksize;
  comm->fifo[comm->head].maxcredit = maxcredit;
  comm->fifo[comm->head].handler = handler;
  comm->fifo[comm->head].offset = offset;
  comm->fifo[comm->head].elements = elements;

  int newhead=(comm->head+1)&(MAX_REQUESTS-1);
  while(newhead==comm->tail) {}
  comm->head=newhead;
  
  comm->basecounter[op]+=(elements*2+blocksize-1)/blocksize;
  }
#endif
}
void allreduce2_userbuff_inplace(const int handler,const int offset,const int elements,communicator* comm,cudaStream_t stream) {
  allreduce_nonsharp_inplace(handler,offset,elements,comm,stream,userbuffers_allreduceop_nonsharp2);
}

void allreduce_userbuff_inplace(const int handler,const int offset,const int elements,communicator* comm,cudaStream_t stream) {
  if(elements<64) return; //sorry guys no allreduce yet, maybe call MPI_Allreduce :)
  //if(comm->myrank==0) fprintf(stderr,"AR1 user call launch_mode=%d\n",comm->launch_mode);
#ifdef NOSHARP
  allreduce_nonsharp_inplace(handler,offset,elements,comm,stream,userbuffers_allreduceop_nonsharp);
  return;
#endif
  int blocksize=elements*2;
#ifdef MULTINODE
  blocksize = (comm->nblocks-1+(comm->alignblock-1+elements*2)/comm->alignblock)/comm->nblocks; //FIXME TUNING
  blocksize*=comm->alignblock;
  if(blocksize<comm->minblock) blocksize=comm->minblock;
  if(comm->ar_nvsize==1) { blocksize=elements*2; if(blocksize>512*1024*1024) blocksize=512*1024*1024; }
#endif
  //blocksize=elements*2;
  int sms = allreduce_userbuff_inplace_gpu(handler, offset, elements, blocksize, comm, stream);

#ifdef MULTINODE
  if(comm->launch_mode & LAUNCH_CPU) 
  {
  if(!sms) return;
  comm->fifo[comm->head].optype=userbuffers_allreduceop_sharp;
  comm->fifo[comm->head].basecounter = comm->basecounter[userbuffers_allreduceop_sharp];
  comm->fifo[comm->head].blocksize = blocksize;
  //comm->fifo[comm->head].sms = sms;
  comm->fifo[comm->head].handler = handler;
  comm->fifo[comm->head].offset = offset;
  comm->fifo[comm->head].elements = elements;

  int newhead=(comm->head+1)&(MAX_REQUESTS-1);
  while(newhead==comm->tail) {}
  comm->head=newhead;
  
  comm->basecounter[userbuffers_allreduceop_sharp]+=(elements*2+blocksize-1)/blocksize;
  //if(comm->myrank==0) fprintf(stderr,"cpu launched basecounter=%d\n",comm->basecounter);
  }
#endif
}

#if defined(MULTINODE) && defined(NOSHARP)
void reducescatter_userbuff_inplace(const int handler,const int offset,const int elements,communicator* comm,cudaStream_t stream) {
  if(elements<64) return;

  int op = userbuffers_allreduceop_nonsharp;
  const int ar_nvsize = op==userbuffers_allreduceop_nonsharp? comm->ar_nvsize : comm->ar2_nvsize;
  int blocksize=elements*2;
  int maxcredit=0;

  const int num_nodes = op==userbuffers_allreduceop_nonsharp? comm->num_nodes : comm->num2_nodes;
  blocksize = (comm->nblocks-1+(comm->alignblock-1+elements*2)/comm->alignblock)/comm->nblocks; //FIXME TUNING
  blocksize*=comm->alignblock;
  if(blocksize<comm->minblock) blocksize=comm->minblock;

  maxcredit = (elements*2+blocksize-1)/blocksize;
  size_t peerblock = sizeof(int)*REG0_COMMBUFFER/maxcredit; //max size we can fit
  if(blocksize>peerblock*ar_nvsize) blocksize = peerblock*ar_nvsize;

  int sms = reducescatter2_userbuff_inplace_gpu(maxcredit,handler, offset, elements, blocksize, comm, stream,op);

  if(num_nodes>1 && comm->launch_mode & LAUNCH_CPU) 
  {
  if(!sms) return;
  comm->fifo[comm->head].optype=op;
  comm->fifo[comm->head].basecounter = comm->basecounter[op];
  comm->fifo[comm->head].blocksize = blocksize;
  comm->fifo[comm->head].maxcredit = maxcredit;
  comm->fifo[comm->head].handler = handler;
  comm->fifo[comm->head].offset = offset;
  comm->fifo[comm->head].elements = elements;

  int newhead=(comm->head+1)&(MAX_REQUESTS-1);
  while(newhead==comm->tail) {}
  comm->head=newhead;
  
  comm->basecounter[op]+=(elements*2+blocksize-1)/blocksize;
}
}

void allgather_userbuff_inplace(const int handler,const int offset,const int elements,communicator* comm,cudaStream_t stream) {
  if(elements<64) return; //sorry guys no allreduce yet, maybe call MPI_Allreduce :)
  int op = userbuffers_allreduceop_nonsharp;
  const int ar_nvsize = op==userbuffers_allreduceop_nonsharp? comm->ar_nvsize : comm->ar2_nvsize;
  int blocksize=elements*2;
  int maxcredit=0;

  const int num_nodes = op==userbuffers_allreduceop_nonsharp? comm->num_nodes : comm->num2_nodes;
  blocksize = (comm->nblocks-1+(comm->alignblock-1+elements*2)/comm->alignblock)/comm->nblocks; //FIXME TUNING
  blocksize*=comm->alignblock;
  if(blocksize<comm->minblock) blocksize=comm->minblock;

  maxcredit = (elements*2+blocksize-1)/blocksize;
  size_t peerblock = sizeof(int)*REG0_COMMBUFFER/maxcredit; //max size we can fit
  if(blocksize>peerblock*ar_nvsize) blocksize = peerblock*ar_nvsize;

  int sms = allgather2_userbuff_inplace_gpu(maxcredit,handler, offset, elements, blocksize, comm, stream,op);
}
#endif


#ifdef MULTINODE
void* proxythread(void* c) {
  communicator * comm=(communicator*) c;
  
  cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
  int core;
  if(comm->nvrank==0) core=51;
  if(comm->nvrank==1) core=59;
  if(comm->nvrank==2) core=19;
  if(comm->nvrank==3) core=27;
  if(comm->nvrank==4) core=115;
  if(comm->nvrank==5) core=123;
  if(comm->nvrank==6) core=83;
  if(comm->nvrank==7) core=91;

	CPU_SET(core,&cpuset);
  if(!getenv("NODOUBLE")) {if(core>128) CPU_SET(core-128,&cpuset); else CPU_SET(core+128,&cpuset);}
	if(getenv("DOPIN")) pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

  while(true) {
     do { 
      if(!comm->activeproxy && comm->active_nreqs==0) goto exitpoint;
  #ifdef NOSHARP
      progress_ucp_allreduce(comm,userbuffers_allreduceop_nonsharp);
  #else
      progress_sharp_allreduce(comm);
  #endif
  #ifdef UCP
      progress_ucp_allreduce(comm,userbuffers_allreduceop_nonsharp2);
      progress_ucp_send(comm);
      progress_ucp_alltoall(comm);
      ucp_worker_progress(comm->ucp_worker);
  #endif
      //sched_yield();
    } while(comm->head==comm->tail);

    int op = comm->fifo[comm->tail].optype;

    if(comm->active_req[op].active == -1) {
      comm->active_nreqs++;
      memcpy((void*)&comm->active_req[op],(void*)&comm->fifo[comm->tail],sizeof(ub_request));
      comm->active_req[op].active = 0;
      comm->tail = (comm->tail+1)&(MAX_REQUESTS-1);
    }

  }

  exitpoint:
  #ifdef UCP
    UCXCHECK(ucp_worker_flush(comm->ucp_worker));
  #endif
    //fflush(NULL);
    MPI_Barrier(MPI_COMM_WORLD);
    usleep(comm->nvrank*300);
    return NULL;
}
#endif
