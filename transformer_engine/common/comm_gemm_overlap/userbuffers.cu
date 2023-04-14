#include <cuda.h>
#include <cuda_runtime.h>
#if __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#define half nv_bfloat16
#else
#include <cuda_fp16.h>
#endif
#include <assert.h>
#include <stdio.h>
#include <transformer_engine/userbuffers.h>

#define MAX_THREADS 1024
#define TIMEOUT 200000000000ull

#define CUDACHECK(cmd)                                                                      \
  do {                                                                                      \
    cudaError_t e = cmd;                                                                    \
    if (e != cudaSuccess) {                                                                 \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(EXIT_FAILURE);                                                                   \
    }                                                                                       \
  } while (0)

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_rw(const int op, const int flagoffset, const int firstrank,
                                        const int myrank, const int gpustep, const int lineoffset,
                                        const int numlines, void **commbuff, const int handleridx) {
  __shared__ int4 *userptr[RANKS];
  int *flagptr, physgpu, targetgpu, *myptr;
  int *reduceidptr, reduce_id;
  // if(blockIdx.x==0 && threadIdx.x==0) printf("%d/%d(phys %d gpustep %d firstrank %d):RRkernel(d)
  // start, size %lld\n",myrank,RANKS,gpustep*myrank+firstrank,gpustep,firstrank,numlines*16ull);
  if (threadIdx.x < RANKS) {
    physgpu = myrank * gpustep + firstrank;
    targetgpu = threadIdx.x * gpustep + firstrank;
    const int blockflagoffset = MAX_NVLINK * 2 * blockIdx.x;
    myptr = (reinterpret_cast<int *>(commbuff[physgpu])) + flagoffset;
    reduceidptr = myptr - MAX_OPS;  //+op;
    reduce_id = (*reduceidptr) + 1;
    flagptr = (reinterpret_cast<int *>(commbuff[targetgpu])) + flagoffset + blockflagoffset;
    myptr += blockflagoffset;

    flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&(myptr[targetgpu]);
    userptr[threadIdx.x] = reinterpret_cast<int4 *>(commbuff[targetgpu + handleridx]);
    clock_t s = clock64();
    while (*flag < reduce_id) {
      if (clock64() - s > TIMEOUT) {
        printf("NVONLY RSBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, reduce_id,
               *flag);
        break;
      }
    }
    reduce_id++;
  }
  __syncthreads();

  int warp = blockIdx.x + (threadIdx.x >> 5);
  int dest[RANKS];
#pragma unroll
  for (int i = 0; i < RANKS; i++) dest[i] = (i + myrank + warp) & (RANKS - 1);

  __syncthreads();
  for (int line = threadIdx.x + blockDim.x * (myrank + RANKS * blockIdx.x); line < numlines;
       line += blockDim.x * gridDim.x * RANKS) {
    int4 val[RANKS];

#pragma unroll
    for (int i = 0; i < RANKS; i++) {
      // int dest = (i+myrank+warp)&(RANKS-1);
      val[i] = userptr[dest[i]][lineoffset + line];
    }

    int4 sum = val[0];
    half *s = reinterpret_cast<half *>(&sum);

#pragma unroll
    for (int i = 1; i < RANKS; i++) {
      half *x = reinterpret_cast<half *>(&val[i]);
#pragma unroll
      for (int j = 0; j < 8; j++) s[j] += x[j];
    }
#pragma unroll
    for (int i = 0; i < RANKS; i++) {
      // int dest = (i+myrank+warp)&(RANKS-1);
      userptr[dest[i]][lineoffset + line] = sum;
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) __threadfence_system();
  __syncthreads();

  if (threadIdx.x < RANKS) {
    flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&myptr[targetgpu];
    clock_t s = clock64();
    while (*flag < reduce_id) {
      if (clock64() - s > 2ull * TIMEOUT) {
        printf("NVONLY AGBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, reduce_id,
               *flag);
        break;
      }
    }
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) *reduceidptr = reduce_id;
}  // fp16 inplace reduce kernel (Volta,Hopper)

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_rr(const int op, const int flagoffset, const int firstrank,
                                        const int myrank, const int gpustep, const int lineoffset,
                                        const int numlines, void **commbuff, const int handleridx) {
  __shared__ int4 *userptr[RANKS];
  int *flagptr, physgpu, targetgpu, *myptr;
  int *reduceidptr, reduce_id;
  // if(blockIdx.x==0 && threadIdx.x==0) printf("%d/%d(phys %d gpustep %d firstrank %d):RRkernel(d)
  // start, size %lld\n",myrank,RANKS,gpustep*myrank+firstrank,gpustep,firstrank,numlines*16ull);
  if (threadIdx.x < RANKS) {
    physgpu = myrank * gpustep + firstrank;
    targetgpu = threadIdx.x * gpustep + firstrank;
    const int blockflagoffset = MAX_NVLINK * 2 * blockIdx.x;
    myptr = (reinterpret_cast<int *>(commbuff[physgpu])) + flagoffset;
    reduceidptr = myptr - MAX_OPS;  //+op;
    reduce_id = (*reduceidptr) + 1;
    flagptr = (reinterpret_cast<int *>(commbuff[targetgpu])) + flagoffset + blockflagoffset;
    myptr += blockflagoffset;

    flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&(myptr[targetgpu]);
    userptr[threadIdx.x] = reinterpret_cast<int4 *>(commbuff[targetgpu + handleridx]);
    clock_t s = clock64();
    while (*flag < reduce_id) {
      if (clock64() - s > TIMEOUT) {
        printf("NVONLY RSBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, reduce_id,
               *flag);
        break;
      }
    }
    reduce_id++;
  }
  __syncthreads();

  int warp = blockIdx.x + (threadIdx.x >> 5);
  int dest[RANKS];
#pragma unroll
  for (int i = 0; i < RANKS; i++) dest[i] = (i + myrank + warp) & (RANKS - 1);

  __syncthreads();
  for (int line = threadIdx.x + blockDim.x * (myrank + RANKS * blockIdx.x); line < numlines;
       line += blockDim.x * gridDim.x * RANKS) {
    int4 val[RANKS];

#pragma unroll
    for (int i = 0; i < RANKS; i++) {
      // int dest = (i+myrank+warp)&(RANKS-1);
      val[i] = userptr[dest[i]][lineoffset + line];
    }

    int4 sum = val[0];
    half *s = reinterpret_cast<half *>(&sum);

#pragma unroll
    for (int i = 1; i < RANKS; i++) {
      half *x = reinterpret_cast<half *>(&val[i]);
#pragma unroll
      for (int j = 0; j < 8; j++) s[j] += x[j];
    }

    userptr[myrank][lineoffset + line] = sum;
  }
#ifdef ALLREDUCEONLYRS
  if (threadIdx.x == 0 && blockIdx.x == 0) *reduceidptr = reduce_id;
  return;
#endif
  __syncthreads();
  if (threadIdx.x == 0) __threadfence();
  __syncthreads();

  if (threadIdx.x < RANKS) {
    flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&myptr[targetgpu];
    clock_t s = clock64();
    while (*flag < reduce_id) {
      if (clock64() - s > 2ull * TIMEOUT) {
        printf("NVONLY AGBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, reduce_id,
               *flag);
        break;
      }
    }
  }

  int skipmy = 0;
#pragma unroll
  for (int i = 0; i < RANKS; i++) {
    int dst = (i + warp + myrank) & (RANKS - 1);
    if (dst == myrank) {
      skipmy++;
      continue;
    }
    dest[i - skipmy] = dst;
  }
  __syncthreads();

  for (int line = threadIdx.x + blockDim.x * RANKS * blockIdx.x; line < numlines;
       line += blockDim.x * gridDim.x * RANKS) {
    int4 val[RANKS - 1];

#pragma unroll
    for (int i = 0; i < RANKS - 1; i++) {
      // int dest = (i+1+myrank)&(RANKS-1);
      val[i] = userptr[dest[i]][lineoffset + line + blockDim.x * dest[i]];
    }

#pragma unroll
    for (int i = 0; i < RANKS - 1; i++) {
      // int dest = (i+1+myrank)&(RANKS-1);
      userptr[myrank][lineoffset + line + blockDim.x * dest[i]] = val[i];
    }
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) *reduceidptr = reduce_id;
}  // fp16 inplace reduce kernel (Ampere)

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_rr_rs(const int op, const int flagoffset, const int firstrank,
                                           const int myrank, const int gpustep,
                                           const int mylineoffset, const int totallines,
                                           void **commbuff, const int handleridx) {
  __shared__ int4 *userptr[RANKS];
  int *flagptr, physgpu, targetgpu, *myptr;
  int *reduceidptr, reduce_id;
  // if(blockIdx.x==0 && threadIdx.x==0) printf("%d/%d(phys %d gpustep %d firstrank %d):RRkernel(d)
  // start, size %lld\n",myrank,RANKS,gpustep*myrank+firstrank,gpustep,firstrank,numlines*16ull);
  if (threadIdx.x < RANKS) {
    physgpu = myrank * gpustep + firstrank;
    targetgpu = threadIdx.x * gpustep + firstrank;
    const int blockflagoffset = MAX_NVLINK * 2 * blockIdx.x;
    myptr = (reinterpret_cast<int *>(commbuff[physgpu])) + flagoffset;
    reduceidptr = myptr - MAX_OPS;  //+op;
    reduce_id = (*reduceidptr) + 1;
    flagptr = (reinterpret_cast<int *>(commbuff[targetgpu])) + flagoffset + blockflagoffset;
    myptr += blockflagoffset;

    flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&(myptr[targetgpu]);
    userptr[threadIdx.x] = reinterpret_cast<int4 *>(commbuff[targetgpu + handleridx]);
    clock_t s = clock64();
    while (*flag < reduce_id) {
      if (clock64() - s > TIMEOUT) {
        printf("NVONLY RSBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, reduce_id,
               *flag);
        break;
      }
    }
  }
  __syncthreads();

  int warp = blockIdx.x + (threadIdx.x >> 5);
  int dest[RANKS];
#pragma unroll
  for (int i = 0; i < RANKS; i++) dest[i] = (i + myrank + warp) & (RANKS - 1);

  __syncthreads();
  for (int line = threadIdx.x + blockDim.x * blockIdx.x; line < totallines;
       line += blockDim.x * gridDim.x) {
    int4 val[RANKS];

#pragma unroll
    for (int i = 0; i < RANKS; i++) {
      // int dest = (i+myrank+warp)&(RANKS-1);
      val[i] = userptr[dest[i]][mylineoffset + line];
    }

    int4 sum = val[0];
    half *s = reinterpret_cast<half *>(&sum);

#pragma unroll
    for (int i = 1; i < RANKS; i++) {
      half *x = reinterpret_cast<half *>(&val[i]);
#pragma unroll
      for (int j = 0; j < 8; j++) s[j] += x[j];
    }

    userptr[myrank][mylineoffset + line] = sum;
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) *reduceidptr = reduce_id;
}  // fp16 inplace reduce-scatter kernel

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_rr_rs_oop(const int op, const int flagoffset,
                                               const int firstrank, const int myrank,
                                               const int gpustep, const int mylineoffset,
                                               const int totallines, const int rowlines,
                                               const int skiplines, void **commbuff,
                                               const int handleridx, void *outbuf) {
  __shared__ int4 *userptr[RANKS];
  int *flagptr, physgpu, targetgpu, *myptr;
  int *reduceidptr, reduce_id;
  // if(blockIdx.x==0 && threadIdx.x==0) printf("%d/%d(phys %d gpustep %d firstrank %d):RRkernel(d)
  // start, size %lld\n",myrank,RANKS,gpustep*myrank+firstrank,gpustep,firstrank,numlines*16ull);
  if (threadIdx.x < RANKS) {
    physgpu = myrank * gpustep + firstrank;
    targetgpu = threadIdx.x * gpustep + firstrank;
    const int blockflagoffset = MAX_NVLINK * 2 * blockIdx.x;
    myptr = (reinterpret_cast<int *>(commbuff[physgpu])) + flagoffset;
    reduceidptr = myptr - MAX_OPS;  //+op;
    reduce_id = (*reduceidptr) + 1;
    flagptr = (reinterpret_cast<int *>(commbuff[targetgpu])) + flagoffset + blockflagoffset;
    myptr += blockflagoffset;

    flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&(myptr[targetgpu]);
    userptr[threadIdx.x] = reinterpret_cast<int4 *>(commbuff[targetgpu + handleridx]);
    clock_t s = clock64();
    while (*flag < reduce_id) {
      if (clock64() - s > TIMEOUT) {
        printf("NVONLY RSBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, reduce_id,
               *flag);
        break;
      }
    }
  }
  __syncthreads();

  int warp = blockIdx.x + (threadIdx.x >> 5);
  int dest[RANKS];
#pragma unroll
  for (int i = 0; i < RANKS; i++) dest[i] = (i + myrank + warp) & (RANKS - 1);

  __syncthreads();
  for (int line = threadIdx.x + blockDim.x * blockIdx.x; line < totallines;
       line += blockDim.x * gridDim.x) {
    int4 val[RANKS];

#pragma unroll
    for (int i = 0; i < RANKS; i++) {
      // int dest = (i+myrank+warp)&(RANKS-1);
      val[i] = userptr[dest[i]][mylineoffset + line];
    }

    int4 sum = val[0];
    half *s = reinterpret_cast<half *>(&sum);

#pragma unroll
    for (int i = 1; i < RANKS; i++) {
      half *x = reinterpret_cast<half *>(&val[i]);
#pragma unroll
      for (int j = 0; j < 8; j++) s[j] += x[j];
    }

    ((int4 *)outbuf)[(line / rowlines) * skiplines + (line % rowlines)] = sum;
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) *reduceidptr = reduce_id;
}  // fp16 reduce-scatter kernel (out of place)

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_rr_ag(const int op, const int flagoffset, const int firstrank,
                                           const int myrank, const int gpustep,
                                           const int mylineoffset, const int totallines,
                                           void **commbuff, const int handleridx) {
  __shared__ int4 *userptr[RANKS];
  int *flagptr, physgpu, targetgpu, *myptr;
  int *reduceidptr, reduce_id;
  // if(blockIdx.x==0 && threadIdx.x==0) printf("%d/%d(phys %d gpustep %d firstrank %d):RRkernel(d)
  // start, size %lld\n",myrank,RANKS,gpustep*myrank+firstrank,gpustep,firstrank,numlines*16ull);
  if (threadIdx.x < RANKS) {
    physgpu = myrank * gpustep + firstrank;
    targetgpu = threadIdx.x * gpustep + firstrank;
    const int blockflagoffset = MAX_NVLINK * 2 * blockIdx.x;
    myptr = (reinterpret_cast<int *>(commbuff[physgpu])) + flagoffset;
    reduceidptr = myptr - MAX_OPS;  //+op;
    reduce_id = (*reduceidptr) + 1;
    flagptr = (reinterpret_cast<int *>(commbuff[targetgpu])) + flagoffset + blockflagoffset;
    myptr += blockflagoffset;

    flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&(myptr[targetgpu]);
    userptr[threadIdx.x] = reinterpret_cast<int4 *>(commbuff[targetgpu + handleridx]);
    clock_t s = clock64();
  }

  int warp = blockIdx.x + (threadIdx.x >> 5);
  int dest[RANKS];

  int skipmy = 0;
#pragma unroll
  for (int i = 0; i < RANKS; i++) {
    int dst = (i + warp + myrank) & (RANKS - 1);
    if (dst == myrank) {
      skipmy++;
      continue;
    }
    dest[i - skipmy] = dst;
  }
  __syncthreads();

  for (int line = threadIdx.x + blockDim.x * blockIdx.x; line < totallines;
       line += blockDim.x * gridDim.x) {
    int4 val[RANKS - 1];

#pragma unroll
    for (int i = 0; i < RANKS - 1; i++) {
      // int dest = (i+1+myrank)&(RANKS-1);
      val[i] = userptr[dest[i]][mylineoffset + line + totallines * dest[i]];
    }

#pragma unroll
    for (int i = 0; i < RANKS - 1; i++) {
      // int dest = (i+1+myrank)&(RANKS-1);
      userptr[myrank][mylineoffset + line + totallines * dest[i]] = val[i];
    }
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) *reduceidptr = reduce_id;
}  // fp16 inplace reduce kernel (Ampere)

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_rw_ag(const int op, const int flagoffset, const int firstrank,
                                           const int myrank, const int gpustep,
                                           const int mylineoffset, const int totallines,
                                           void **commbuff, const int handleridx) {
  __shared__ int4 *userptr[RANKS];
  int *flagptr, physgpu, targetgpu, *myptr;
  int *reduceidptr, reduce_id;
  int4 *localptr;
  // if(blockIdx.x==0 && threadIdx.x==0) printf("%d/%d(phys %d gpustep %d firstrank %d):RRkernel(d)
  // start, size %lld\n",myrank,RANKS,gpustep*myrank+firstrank,gpustep,firstrank,numlines*16ull);
  if (threadIdx.x < RANKS) {
    physgpu = myrank * gpustep + firstrank;
    targetgpu = threadIdx.x * gpustep + firstrank;
    const int blockflagoffset = MAX_NVLINK * 2 * blockIdx.x;
    myptr = (reinterpret_cast<int *>(commbuff[physgpu])) + flagoffset;
    reduceidptr = myptr - MAX_OPS;  //+op;
    reduce_id = (*reduceidptr) + 1;
    flagptr = (reinterpret_cast<int *>(commbuff[targetgpu])) + flagoffset + blockflagoffset;
    myptr += blockflagoffset;
    userptr[threadIdx.x] = reinterpret_cast<int4 *>(commbuff[targetgpu + handleridx]);
    reduce_id++;
  }
  __syncthreads();
  localptr = userptr[myrank];

  int warp = blockIdx.x + (threadIdx.x >> 5);
  int dest[RANKS - 1];
  int skipmy = 0;
#pragma unroll
  for (int i = 0; i < RANKS; i++) {
    int dst = (i + warp + myrank) & (RANKS - 1);
    if (dst == myrank) {
      skipmy++;
      continue;
    }
    dest[i - skipmy] = dst;
  }
#define UNROLLAG 4
  __syncthreads();
  const int loop_step0 = blockDim.x * gridDim.x;
  const int loop_step = loop_step0 * UNROLLAG;
  const int start_elem = threadIdx.x + blockDim.x * blockIdx.x;
  const int end_elem = max(start_elem, totallines);
  const int aligned_elem = ((end_elem - start_elem) / loop_step) * loop_step;
  const int end_aligned = start_elem + aligned_elem;

  for (int line = start_elem; line < end_aligned; line += loop_step) {
    int4 val[UNROLLAG];
#pragma unroll
    for (int j = 0; j < UNROLLAG; j++) val[j] = localptr[mylineoffset + line + loop_step0 * j];

#pragma unroll
    for (int j = 0; j < UNROLLAG; j++)
#pragma unroll
      for (int i = 0; i < RANKS - 1; i++) {
        // int dest = (i+myrank+warp)&(RANKS-1);
        userptr[dest[i]][mylineoffset + line + j * loop_step0] = val[j];
      }
  }

  for (int line = end_aligned; line < end_elem; line += loop_step0) {
    int4 sum = localptr[mylineoffset + line];
#pragma unroll
    for (int i = 0; i < RANKS - 1; i++) {
      // int dest = (i+myrank+warp)&(RANKS-1);
      userptr[dest[i]][mylineoffset + line] = sum;
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) __threadfence_system();
  __syncthreads();

  if (threadIdx.x < RANKS) {
    flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&myptr[targetgpu];
    clock_t s = clock64();
    while (*flag < reduce_id) {
      if (clock64() - s > 2ull * TIMEOUT) {
        printf("NVONLY AGBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, reduce_id,
               *flag);
        break;
      }
    }
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) *reduceidptr = reduce_id;
}  // fp16 inplace allgather kernel (Volta,Hopper)

#ifdef MULTINODE
template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_rr_blocked(const int op, const int flagoffset,
                                                const int firstrank, const int myrank,
                                                const int lineoffset, const int numlines,
                                                void **commbuff, const int handleridx,
                                                const int peerblocklines, int *hostflags,
                                                int *gpuflag, const int numblocks) {
  const int basecounter = gpuflag[GF_STATE + op];

#define REDUCETHREADS (blockDim.x - 32)

  if (threadIdx.x < 32) {
    int *flagptr;
    if (threadIdx.x < RANKS) {
      if (!blockIdx.x) {
        flagptr = reinterpret_cast<int *>(commbuff[threadIdx.x + firstrank]);
        flagptr[flagoffset + myrank + firstrank] = basecounter;
      }
      volatile int *flag = (volatile int *)&(
          ((int *)commbuff[myrank + firstrank])[flagoffset + threadIdx.x + firstrank]);
      while (*flag < basecounter) {
      }
    }
    __syncthreads();

    int startblock = 0, endblock = numblocks;
    // if(numblocks>1) {startblock++;endblock--;}

    for (int nblock = 0; nblock < endblock; nblock++) {
      asm volatile("bar.sync 13, %0;" ::"r"(REDUCETHREADS + 32));

      if (threadIdx.x == 0) {
        __threadfence();
        if (blockIdx.x) gpuflag[op * MAX_SMS * 2 + blockIdx.x] = nblock + basecounter + 1;
      } else if (blockIdx.x == 0) {
        int expecting = (basecounter + nblock + 1);
        if (threadIdx.x < gridDim.x)
          while (((volatile int *)gpuflag)[op * MAX_SMS * 2 + threadIdx.x] < expecting) {
          }
      }
      if (!blockIdx.x) {
        asm volatile("bar.sync 15, %0;" ::"r"(32));
        if (!threadIdx.x) hostflags[0] = nblock + basecounter + 1;
      }
    }

    int cachedflag = basecounter;

#define ALLGATHERFLAG GF_IBSHARPDONE

    if (blockIdx.x == 0 && threadIdx.x < RANKS) {
      while (cachedflag < basecounter + numblocks) {
        int newflag = ((volatile int *)gpuflag)[ALLGATHERFLAG];
        if (newflag == cachedflag) continue;
        cachedflag = newflag;
        flagptr[flagoffset + myrank + 32 + firstrank] = cachedflag;
      }
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) gpuflag[GF_STATE + op] = basecounter + numblocks;

  } else

  {
    const int warp = blockIdx.x + (threadIdx.x >> 5);
    int4 *userptr[RANKS];
    int4 *userptrmyrank;
#pragma unroll
    for (int i = 0; i < RANKS; i++)
      userptr[i] = reinterpret_cast<int4 *>(
          commbuff[((i + myrank + warp) & (RANKS - 1)) + handleridx + firstrank]);
    userptrmyrank = reinterpret_cast<int4 *>(commbuff[myrank + handleridx + firstrank]);
    __syncthreads();

    int blocklineoffset = 0;

    while (blocklineoffset < numlines) {
      const int remainder = min(numlines - blocklineoffset, peerblocklines * RANKS);
      const int blocklines = remainder / RANKS;
      const int blockstart = lineoffset + blocklineoffset + blocklines * myrank;

      for (int line = threadIdx.x - 32 + REDUCETHREADS * blockIdx.x; line < blocklines;
           line += REDUCETHREADS * gridDim.x) {
        int4 val[RANKS];

#pragma unroll
        for (int i = 0; i < RANKS; i++) {
          val[i] = userptr[i][blockstart + line];
        }

        int4 sum = val[0];
        half *s = reinterpret_cast<half *>(&sum);

#pragma unroll
        for (int i = 1; i < RANKS; i++) {
          half *x = reinterpret_cast<half *>(&val[i]);
#pragma unroll
          for (int j = 0; j < sizeof(int4) / sizeof(half); j++) s[j] += x[j];
        }

        userptrmyrank[blockstart + line] = sum;
      }  // single block loop

      asm volatile("bar.sync 13, %0;" ::"r"(REDUCETHREADS + 32));

      blocklineoffset += peerblocklines * RANKS;
    }  // block loop NVLINK-REDUCESCATTER
    const int nwarps = (REDUCETHREADS >> 5) / (RANKS - 1);
    const int myblockDim = nwarps << 5;
    const int mywarp = ((threadIdx.x - 32) >> 5) / (RANKS - 1);
    const int maxthreadIdx = myblockDim * (RANKS - 1) + 32;
    const int mydest = (myrank + 1 + ((threadIdx.x - 32) >> 5) % (RANKS - 1)) & (RANKS - 1);
    const int mythreadIdx = (mywarp << 5) + (threadIdx.x & 31);
    volatile int *flag = (volatile int *)&(
        ((int *)commbuff[myrank + firstrank])[flagoffset + mydest + 32 + firstrank]);

    int4 *userptrmydest = userptr[((RANKS << 10) + mydest - myrank - warp) & (RANKS - 1)];

    blocklineoffset = 0;
    int gathercounter = basecounter + 1;
    while (blocklineoffset < numlines) {
      const int remainder = min(numlines - blocklineoffset, peerblocklines * RANKS);
      const int blocklines = remainder / RANKS;
      const int blockstart = lineoffset + blocklineoffset;

#define UNROLL 6
      int4 *myptr = &userptrmyrank[blockstart + blocklines * mydest];
      int4 *peerptr = &userptrmydest[blockstart + blocklines * mydest];

      if (threadIdx.x < maxthreadIdx) {
        const int start_elem = mythreadIdx + myblockDim * blockIdx.x;
        const int end_elem = max(start_elem, blocklines);
        const int aligned_elem = ((end_elem - start_elem) / (myblockDim * gridDim.x * UNROLL)) *
                                 (myblockDim * gridDim.x * UNROLL);
        const int end_aligned = start_elem + aligned_elem;

        if (mythreadIdx == 0) {
          while (*flag < gathercounter) {
          }
          gathercounter++;
        }

        asm volatile("bar.sync %0, %1;" ::"r"(1 + mydest), "r"(myblockDim));

        for (int line = start_elem; line < end_aligned; line += myblockDim * gridDim.x * UNROLL) {
          int4 val[UNROLL];
#pragma unroll
          for (int i = 0; i < UNROLL; i++) val[i] = peerptr[line + i * myblockDim * gridDim.x];
#pragma unroll
          for (int i = 0; i < UNROLL; i++) myptr[line + i * myblockDim * gridDim.x] = val[i];
        }
        for (int line = end_aligned; line < end_elem; line += myblockDim * gridDim.x)
          myptr[line] = peerptr[line];
      }
      blocklineoffset += peerblocklines * RANKS;

    }  // block loop for NVLINK-ALLGATHER
  }    // worker warps else block
}  // fp16 inplace reduce kernel with SHARP / in blocks

   // threadfence and SMs sync to SM0
#define SMBAR(offset, block)                                                \
  asm volatile("bar.sync 13, %0;" ::"r"(blockDim.x));                       \
  if (threadIdx.x == 0) {                                                   \
    __threadfence_system();                                                 \
    if (blockIdx.x) gpuflag[offset + blockIdx.x] = block + basecounter + 1; \
  } else if (blockIdx.x == 0) {                                             \
    int expecting = (basecounter + block + 1);                              \
    if (threadIdx.x < gridDim.x)                                            \
      while (((volatile int *)gpuflag)[offset + threadIdx.x] < expecting) { \
      }                                                                     \
  }                                                                         \
  if (blockIdx.x == 0) asm volatile("bar.sync 15, %0;" ::"r"(32));

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS) userbuffers_fp16_sum_inplace_gpu_rr_blocked2(
    const int op, const int maxcredit, const int headstart, const int myibrank, const int ibranks,
    const int commbufoffset, const int flagoffset, const int firstrank, const int myrank,
    const int gpustep, const int lineoffset, const int numlines, void **commbuff,
    const int handleridx, const int peerblocklines, int *hostflags, int *gpuflag,
    const int numblocks) {
  const int basecounter = gpuflag[GF_STATE + op];
  // if(blockIdx.x==0 && threadIdx.x==0) printf("%d(%d)[%d/%d]:AR2kernel(%d) start, size %d
  // numblocks %d(%d) basecounter %d peerblocklines %d commbufoffset %d maxcredit %d\n",
  //     myrank,gpustep*myrank+firstrank,myibrank,ibranks,op,numlines*16,numblocks,headstart,basecounter,peerblocklines,commbufoffset,maxcredit);
  if (threadIdx.x < 32) {
    int *flagptr;
    volatile int *localflag =
        (volatile int *)&(((int *)commbuff[gpustep * myrank + firstrank])[flagoffset]);
    // initial intranode barrier - once
    if (threadIdx.x < RANKS) {
      if (!blockIdx.x) {
        flagptr = reinterpret_cast<int *>(commbuff[gpustep * threadIdx.x + firstrank]);
        flagptr[flagoffset + gpustep * myrank + firstrank] = basecounter;
      }
      volatile int *flag = &localflag[gpustep * threadIdx.x + firstrank];
      while (*flag < basecounter) {
      }
    }
    __syncthreads();

    for (int nblock = 0; nblock < numblocks + headstart; nblock++) {
      if (nblock < numblocks) {
        // RS happens here
        SMBAR(op * 2 * MAX_SMS, nblock);
        if (!blockIdx.x && !threadIdx.x)
          hostflags[HF_NVRSDONE + (op & 1)] = nblock + basecounter + 1;
      }

      if (nblock >= headstart) {
        for (int ibflag = threadIdx.x; ibflag < ibranks; ibflag += 32)
          if (ibflag != myibrank)
            while (localflag[REG0_IBRS + ibflag] < basecounter + nblock - headstart + 1) {
            }
        asm volatile("bar.sync 13, %0;" ::"r"(blockDim.x));
        // REDUCE happens here
        SMBAR(op * 2 * MAX_SMS + MAX_SMS, nblock - headstart);
        if (!blockIdx.x && !threadIdx.x)
          hostflags[HF_NVREDUCEDONE + (op & 1)] = nblock + basecounter + 1 - headstart;
      }
    }
    // final part doing NVAG based on responses from NIC-RMW:IBAG

    if (blockIdx.x == 0)
      for (int nblock = 0; nblock < numblocks; nblock++) {
        const int expected = basecounter + nblock + 1;
        for (int ibflag = threadIdx.x; ibflag < ibranks; ibflag += 32)
          if (ibflag != myibrank)
            while (localflag[REG0_IBAG + ibflag] < expected) {
            }
        asm volatile("bar.sync 15, %0;" ::"r"(32));
        if (threadIdx.x < RANKS)
          flagptr[flagoffset + gpustep * myrank + MAX_NVLINK + firstrank] = expected;
      }

    if (blockIdx.x == 0 && threadIdx.x == 0) gpuflag[GF_STATE + op] = basecounter + numblocks;

  }       // sync warp
  else {  // reducethreads
    const int warp = blockIdx.x + (threadIdx.x >> 5);
    int4 *userptr[RANKS];
    int4 *userptrmyrank;
#pragma unroll
    for (int i = 0; i < RANKS; i++)
      userptr[i] = reinterpret_cast<int4 *>(
          commbuff[((i + myrank + warp) & (RANKS - 1)) * gpustep + handleridx + firstrank]);
    userptrmyrank = reinterpret_cast<int4 *>(commbuff[gpustep * myrank + handleridx + firstrank]);
    int4 *internalbuf = reinterpret_cast<int4 *>(commbuff[myrank * gpustep + firstrank] +
                                                 commbufoffset * sizeof(int));
    __syncthreads();

    int blocklineoffset = 0, rblocklineoffset = 0;

    for (int nblock = 0; nblock < numblocks + headstart; nblock++) {
      // NVRS part(only first numblocks steps)
      if (blocklineoffset < numlines) {
        const int remainder = min(numlines - blocklineoffset, peerblocklines * RANKS);
        const int blocklines = remainder / RANKS;
        const int blockstart = lineoffset + blocklineoffset + blocklines * myrank;
        if (RANKS > 1)
          for (int line = threadIdx.x - 32 + REDUCETHREADS * blockIdx.x; line < blocklines;
               line += REDUCETHREADS * gridDim.x) {
            int4 val[RANKS];

#pragma unroll
            for (int i = 0; i < RANKS; i++) {
              val[i] = userptr[i][blockstart + line];
            }

            int4 sum = val[0];
            half *s = reinterpret_cast<half *>(&sum);

#pragma unroll
            for (int i = 1; i < RANKS; i++) {
              half *x = reinterpret_cast<half *>(&val[i]);
#pragma unroll
              for (int j = 0; j < sizeof(int4) / sizeof(half); j++) s[j] += x[j];
            }

            userptrmyrank[blockstart + line] = sum;
          }  // single block loop

        asm volatile("bar.sync 13, %0;" ::"r"(REDUCETHREADS + 32));
        blocklineoffset += peerblocklines * RANKS;
      }
      if (nblock >= headstart) {
#define UNROLLRS 2
        const int remainder = min(numlines - rblocklineoffset, peerblocklines * RANKS);
        const int blocklines = remainder / RANKS;
        rblocklineoffset += peerblocklines * RANKS;
        const int ibblocklines = blocklines / ibranks;
        int4 *tempbufptr = &internalbuf[((nblock - headstart) % maxcredit) * peerblocklines];
        const int tempstart = lineoffset + (nblock - headstart) * peerblocklines * RANKS +
                              myrank * blocklines + ibblocklines * myibrank;
        // if(threadIdx.x==32) printf("[%d] block%d thread %d offset %d line %d ibblocklines %d ptr
        // %lx commbufoffset
        // %d\n",myrank,blockIdx.x,threadIdx.x,tempstart,0,ibblocklines,(void*)&tempbufptr[(1-myibrank)*ibblocklines],(1-myibrank)*ibblocklines*16);

        asm volatile("bar.sync 13, %0;" ::"r"(REDUCETHREADS + 32));

        for (int line = threadIdx.x - 32 + REDUCETHREADS * blockIdx.x; line < ibblocklines;
             line += REDUCETHREADS * gridDim.x) {
          int4 val[UNROLLRS];

#pragma unroll
          for (int i = 0; i < UNROLLRS; i++)
            val[i] = i == myibrank ? userptrmyrank[tempstart + line]
                                   : tempbufptr[i * ibblocklines + line];

          int4 sum = val[0];
          half *s = reinterpret_cast<half *>(&sum);

          for (int i = 0; i < ibranks - UNROLLRS; i++) {
            val[i % UNROLLRS] = i == myibrank ? userptrmyrank[tempstart + line]
                                              : tempbufptr[i * ibblocklines + line];
            half *x = reinterpret_cast<half *>(&val[(i + 1) % UNROLLRS]);
#pragma unroll
            for (int j = 0; j < 16; j++) s[j] += x[j];
          }
#pragma unroll
          for (int i = 1; i < UNROLLRS; i++) {
            half *x = reinterpret_cast<half *>(&val[i]);
#pragma unroll
            for (int j = 0; j < 16; j++) s[j] += x[j];
          }
          userptrmyrank[tempstart + line] = sum;
        }

        asm volatile("bar.sync 13, %0;" ::"r"(REDUCETHREADS + 32));
      }
    }  // nblock loop NVLINK-REDUCESCATTER + IBREDUCE LOCAL COMPUTE

    // if(blockIdx.x==0 && threadIdx.x==32) printf("%d:kernel32[%d] ready to AG phase, size %d
    // numblocks %d\n",myrank,REDUCETHREADS,numlines*16,numblocks);
    if (RANKS != 1) {
      const int nwarps = (REDUCETHREADS >> 5) / (RANKS - 1);
      const int myblockDim = nwarps << 5;
      const int mywarp = ((threadIdx.x - 32) >> 5) / (RANKS - 1);
      const int maxthreadIdx = myblockDim * (RANKS - 1) + 32;
      const int mydest = (myrank + 1 + ((threadIdx.x - 32) >> 5) % (RANKS - 1)) & (RANKS - 1);
      const int mythreadIdx = (mywarp << 5) + (threadIdx.x & 31);
      volatile int *flag = (volatile int *)&(
          ((int *)commbuff[gpustep * myrank + firstrank])[flagoffset + gpustep * mydest +
                                                          MAX_NVLINK + firstrank]);

      int4 *userptrmydest = userptr[((RANKS << 10) + mydest - myrank - warp) & (RANKS - 1)];

      blocklineoffset = 0;
      int gathercounter = basecounter + 1;
      while (blocklineoffset < numlines) {
        const int remainder = min(numlines - blocklineoffset, peerblocklines * RANKS);
        const int blocklines = remainder / RANKS;
        const int blockstart = lineoffset + blocklineoffset;

#define UNROLL 6
        int4 *myptr = &userptrmyrank[blockstart + blocklines * mydest];
        int4 *peerptr = &userptrmydest[blockstart + blocklines * mydest];

        if (threadIdx.x < maxthreadIdx) {
          const int start_elem = mythreadIdx + myblockDim * blockIdx.x;
          const int end_elem = max(start_elem, blocklines);
          const int aligned_elem = ((end_elem - start_elem) / (myblockDim * gridDim.x * UNROLL)) *
                                   (myblockDim * gridDim.x * UNROLL);
          const int end_aligned = start_elem + aligned_elem;

          if (mythreadIdx == 0) {
            while (*flag < gathercounter) {
            }
            gathercounter++;
          }

          asm volatile("bar.sync %0, %1;" ::"r"(1 + mydest), "r"(myblockDim));

          for (int line = start_elem; line < end_aligned; line += myblockDim * gridDim.x * UNROLL) {
            int4 val[UNROLL];
#pragma unroll
            for (int i = 0; i < UNROLL; i++) val[i] = peerptr[line + i * myblockDim * gridDim.x];
#pragma unroll
            for (int i = 0; i < UNROLL; i++) myptr[line + i * myblockDim * gridDim.x] = val[i];
          }
          for (int line = end_aligned; line < end_elem; line += myblockDim * gridDim.x)
            myptr[line] = peerptr[line];
        }
        blocklineoffset += peerblocklines * RANKS;

      }  // block loop for NVLINK-ALLGATHER
    }    // RANKS!=1
  }      // worker warps else block

}  // fp16 inplace reduce kernel with SHARP / in blocks

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS) userbuffers_fp16_sum_inplace_gpu_rr_blocked2_rs(
    const int op, const int maxcredit, const int headstart, const int myibrank, const int ibranks,
    const int commbufoffset, const int flagoffset, const int firstrank, const int myrank,
    const int gpustep, const int lineoffset, const int numlines, void **commbuff,
    const int handleridx, const int peerblocklines, int *hostflags, int *gpuflag,
    const int numblocks) {
  const int basecounter = gpuflag[GF_STATE + op];
  // if(blockIdx.x==0 && threadIdx.x==0) printf("%d(%d)[%d/%d]:AR2kernel(%d) start, size %d
  // numblocks %d(%d) basecounter %d peerblocklines %d commbufoffset %d maxcredit %d\n",
  //     myrank,gpustep*myrank+firstrank,myibrank,ibranks,op,numlines*16,numblocks,headstart,basecounter,peerblocklines,commbufoffset,maxcredit);
  if (threadIdx.x < 32) {
    int *flagptr;
    volatile int *localflag =
        (volatile int *)&(((int *)commbuff[gpustep * myrank + firstrank])[flagoffset]);
    // initial intranode barrier - once
    if (threadIdx.x < RANKS) {
      if (!blockIdx.x) {
        flagptr = reinterpret_cast<int *>(commbuff[gpustep * threadIdx.x + firstrank]);
        flagptr[flagoffset + gpustep * myrank + firstrank] = basecounter;
      }
      volatile int *flag = &localflag[gpustep * threadIdx.x + firstrank];
      while (*flag < basecounter) {
      }
    }
    __syncthreads();

    for (int nblock = 0; nblock < numblocks + headstart; nblock++) {
      if (nblock < numblocks) {
        // RS happens here
        SMBAR(op * 2 * MAX_SMS, nblock);
        if (!blockIdx.x && !threadIdx.x)
          hostflags[HF_NVRSDONE + (op & 1)] = nblock + basecounter + 1;
      }

      if (nblock >= headstart) {
        for (int ibflag = threadIdx.x; ibflag < ibranks; ibflag += 32)
          if (ibflag != myibrank)
            while (localflag[REG0_IBRS + ibflag] < basecounter + nblock - headstart + 1) {
            }
        asm volatile("bar.sync 13, %0;" ::"r"(blockDim.x));
        // REDUCE happens here
        SMBAR(op * 2 * MAX_SMS + MAX_SMS, nblock - headstart);
      }
    }
  }       // sync warp
  else {  // reducethreads
    const int warp = blockIdx.x + (threadIdx.x >> 5);
    int4 *userptr[RANKS];
    int4 *userptrmyrank;
#pragma unroll
    for (int i = 0; i < RANKS; i++)
      userptr[i] = reinterpret_cast<int4 *>(
          commbuff[((i + myrank + warp) & (RANKS - 1)) * gpustep + handleridx + firstrank]);
    userptrmyrank = reinterpret_cast<int4 *>(commbuff[gpustep * myrank + handleridx + firstrank]);
    int4 *internalbuf = reinterpret_cast<int4 *>(commbuff[myrank * gpustep + firstrank] +
                                                 commbufoffset * sizeof(int));
    __syncthreads();

    int blocklineoffset = 0, rblocklineoffset = 0;

    for (int nblock = 0; nblock < numblocks + headstart; nblock++) {
      // NVRS part(only first numblocks steps)
      if (blocklineoffset < numlines) {
        const int remainder = min(numlines - blocklineoffset, peerblocklines * RANKS);
        const int blocklines = remainder / RANKS;
        const int blockstart = lineoffset + blocklineoffset + blocklines * myrank;
        if (RANKS > 1)
          for (int line = threadIdx.x - 32 + REDUCETHREADS * blockIdx.x; line < blocklines;
               line += REDUCETHREADS * gridDim.x) {
            int4 val[RANKS];

#pragma unroll
            for (int i = 0; i < RANKS; i++) {
              val[i] = userptr[i][blockstart + line];
            }

            int4 sum = val[0];
            half *s = reinterpret_cast<half *>(&sum);

#pragma unroll
            for (int i = 1; i < RANKS; i++) {
              half *x = reinterpret_cast<half *>(&val[i]);
#pragma unroll
              for (int j = 0; j < sizeof(int4) / sizeof(half); j++) s[j] += x[j];
            }

            userptrmyrank[blockstart + line] = sum;
          }  // single block loop

        asm volatile("bar.sync 13, %0;" ::"r"(REDUCETHREADS + 32));
        blocklineoffset += peerblocklines * RANKS;
      }
      if (nblock >= headstart) {
#define UNROLLRS 2
        const int remainder = min(numlines - rblocklineoffset, peerblocklines * RANKS);
        const int blocklines = remainder / RANKS;
        rblocklineoffset += peerblocklines * RANKS;
        const int ibblocklines = blocklines / ibranks;
        int4 *tempbufptr = &internalbuf[((nblock - headstart) % maxcredit) * peerblocklines];
        const int tempstart = lineoffset + (nblock - headstart) * peerblocklines * RANKS +
                              myrank * blocklines + ibblocklines * myibrank;
        // if(threadIdx.x==32) printf("[%d] block%d thread %d offset %d line %d ibblocklines %d ptr
        // %lx commbufoffset
        // %d\n",myrank,blockIdx.x,threadIdx.x,tempstart,0,ibblocklines,(void*)&tempbufptr[(1-myibrank)*ibblocklines],(1-myibrank)*ibblocklines*16);

        asm volatile("bar.sync 13, %0;" ::"r"(REDUCETHREADS + 32));

        for (int line = threadIdx.x - 32 + REDUCETHREADS * blockIdx.x; line < ibblocklines;
             line += REDUCETHREADS * gridDim.x) {
          int4 val[UNROLLRS];

#pragma unroll
          for (int i = 0; i < UNROLLRS; i++)
            val[i] = i == myibrank ? userptrmyrank[tempstart + line]
                                   : tempbufptr[i * ibblocklines + line];

          int4 sum = val[0];
          half *s = reinterpret_cast<half *>(&sum);

          for (int i = 0; i < ibranks - UNROLLRS; i++) {
            val[i % UNROLLRS] = i == myibrank ? userptrmyrank[tempstart + line]
                                              : tempbufptr[i * ibblocklines + line];
            half *x = reinterpret_cast<half *>(&val[(i + 1) % UNROLLRS]);
#pragma unroll
            for (int j = 0; j < 16; j++) s[j] += x[j];
          }
#pragma unroll
          for (int i = 1; i < UNROLLRS; i++) {
            half *x = reinterpret_cast<half *>(&val[i]);
#pragma unroll
            for (int j = 0; j < 16; j++) s[j] += x[j];
          }
          userptrmyrank[tempstart + line] = sum;
        }

        asm volatile("bar.sync 13, %0;" ::"r"(REDUCETHREADS + 32));
      }
    }  // nblock loop NVLINK-REDUCESCATTER + IBREDUCE LOCAL COMPUTE
  }    // worker warps else block

}  // fp16 inplace reduce kernel with SHARP / in blocks

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS) userbuffers_fp16_sum_inplace_gpu_rr_blocked2_ag(
    const int op, const int maxcredit, const int headstart, const int myibrank, const int ibranks,
    const int commbufoffset, const int flagoffset, const int firstrank, const int myrank,
    const int gpustep, const int lineoffset, const int numlines, void **commbuff,
    const int handleridx, const int peerblocklines, int *hostflags, int *gpuflag,
    const int numblocks) {
  const int basecounter = gpuflag[GF_STATE + op];
  // if(blockIdx.x==0 && threadIdx.x==0) printf("%d(%d)[%d/%d]:AR2kernel(%d) start, size %d
  // numblocks %d(%d) basecounter %d peerblocklines %d commbufoffset %d maxcredit %d\n",
  //     myrank,gpustep*myrank+firstrank,myibrank,ibranks,op,numlines*16,numblocks,headstart,basecounter,peerblocklines,commbufoffset,maxcredit);
  if (threadIdx.x < 32) {
    int *flagptr;
    volatile int *localflag =
        (volatile int *)&(((int *)commbuff[gpustep * myrank + firstrank])[flagoffset]);
    if (threadIdx.x < RANKS) {
      if (!blockIdx.x) {
        flagptr = reinterpret_cast<int *>(commbuff[gpustep * threadIdx.x + firstrank]);
      }
    }
    __syncthreads();
    if (!blockIdx.x && !threadIdx.x)
      hostflags[HF_NVREDUCEDONE + (op & 1)] = numblocks + basecounter;
    // tell CPU proxy all blocks are done and ready for NVAG

    // final part doing NVAG based on responses from NIC-RMW:IBAG

    if (blockIdx.x == 0)
      for (int nblock = 0; nblock < numblocks; nblock++) {
        const int expected = basecounter + nblock + 1;
        for (int ibflag = threadIdx.x; ibflag < ibranks; ibflag += 32)
          if (ibflag != myibrank)
            while (localflag[REG0_IBAG + ibflag] < expected) {
            }
        asm volatile("bar.sync 15, %0;" ::"r"(32));
        if (threadIdx.x < RANKS)
          flagptr[flagoffset + gpustep * myrank + MAX_NVLINK + firstrank] = expected;
      }

    if (blockIdx.x == 0 && threadIdx.x == 0) gpuflag[GF_STATE + op] = basecounter + numblocks;

  }       // sync warp
  else {  // reducethreads
    const int warp = blockIdx.x + (threadIdx.x >> 5);
    int4 *userptr[RANKS];
    int4 *userptrmyrank;
#pragma unroll
    for (int i = 0; i < RANKS; i++)
      userptr[i] = reinterpret_cast<int4 *>(
          commbuff[((i + myrank + warp) & (RANKS - 1)) * gpustep + handleridx + firstrank]);
    userptrmyrank = reinterpret_cast<int4 *>(commbuff[gpustep * myrank + handleridx + firstrank]);
    __syncthreads();

    int blocklineoffset = 0, rblocklineoffset = 0;

    // if(blockIdx.x==0 && threadIdx.x==32) printf("%d:kernel32[%d] ready to AG phase, size %d
    // numblocks %d\n",myrank,REDUCETHREADS,numlines*16,numblocks);
    if (RANKS != 1) {
      const int nwarps = (REDUCETHREADS >> 5) / (RANKS - 1);
      const int myblockDim = nwarps << 5;
      const int mywarp = ((threadIdx.x - 32) >> 5) / (RANKS - 1);
      const int maxthreadIdx = myblockDim * (RANKS - 1) + 32;
      const int mydest = (myrank + 1 + ((threadIdx.x - 32) >> 5) % (RANKS - 1)) & (RANKS - 1);
      const int mythreadIdx = (mywarp << 5) + (threadIdx.x & 31);
      volatile int *flag = (volatile int *)&(
          ((int *)commbuff[gpustep * myrank + firstrank])[flagoffset + gpustep * mydest +
                                                          MAX_NVLINK + firstrank]);

      int4 *userptrmydest = userptr[((RANKS << 10) + mydest - myrank - warp) & (RANKS - 1)];

      blocklineoffset = 0;
      int gathercounter = basecounter + 1;
      while (blocklineoffset < numlines) {
        const int remainder = min(numlines - blocklineoffset, peerblocklines * RANKS);
        const int blocklines = remainder / RANKS;
        const int blockstart = lineoffset + blocklineoffset;

#define UNROLL 6
        int4 *myptr = &userptrmyrank[blockstart + blocklines * mydest];
        int4 *peerptr = &userptrmydest[blockstart + blocklines * mydest];

        if (threadIdx.x < maxthreadIdx) {
          const int start_elem = mythreadIdx + myblockDim * blockIdx.x;
          const int end_elem = max(start_elem, blocklines);
          const int aligned_elem = ((end_elem - start_elem) / (myblockDim * gridDim.x * UNROLL)) *
                                   (myblockDim * gridDim.x * UNROLL);
          const int end_aligned = start_elem + aligned_elem;

          if (mythreadIdx == 0) {
            while (*flag < gathercounter) {
            }
            gathercounter++;
          }

          asm volatile("bar.sync %0, %1;" ::"r"(1 + mydest), "r"(myblockDim));

          for (int line = start_elem; line < end_aligned; line += myblockDim * gridDim.x * UNROLL) {
            int4 val[UNROLL];
#pragma unroll
            for (int i = 0; i < UNROLL; i++) val[i] = peerptr[line + i * myblockDim * gridDim.x];
#pragma unroll
            for (int i = 0; i < UNROLL; i++) myptr[line + i * myblockDim * gridDim.x] = val[i];
          }
          for (int line = end_aligned; line < end_elem; line += myblockDim * gridDim.x)
            myptr[line] = peerptr[line];
        }
        blocklineoffset += peerblocklines * RANKS;

      }  // block loop for NVLINK-ALLGATHER
    }    // RANKS!=1
  }      // worker warps else block

}  // fp16 inplace reduce kernel with SHARP / in blocks

__global__ void userbuffers_fp16_sum_inplace_gpu_null(const int op, int *hostflags, int *gpuflag,
                                                      int numblocks) {
  const int basecounter = gpuflag[GF_STATE + op] + numblocks;
  hostflags[0] = basecounter;
  gpuflag[GF_STATE + op] = basecounter;
  while (((volatile int *)gpuflag)[GF_IBSHARPDONE] < basecounter) {
  }
}

#define callranks_block(x)                                                                    \
  if (comm->ar_nvsize == x)                                                                   \
    userbuffers_fp16_sum_inplace_gpu_rr_blocked<x><<<sms, warps * 32, 0, stream>>>(           \
        userbuffers_allreduceop_sharp, REG0_OFFSET(comm), comm->ar_firstgpu, comm->ar_nvrank, \
        offset / 8, elements / 8, (void **)(comm->gpu_ptrs), handler * comm->nvsize,          \
        blocksize / sizeof(int4) / comm->ar_nvsize, (int *)comm->hostflags, comm->flags,      \
        (elements * 2 + blocksize - 1) / blocksize);

#define callranks2_block(x)                                                                    \
  if (ar_nvsize == x) {                                                                        \
    int numblocks = (elements * 2 + blocksize - 1) / blocksize;                                \
    int headstart = numblocks - 1; /*<3?numblocks-1:3;*/                                       \
    if (headstart > maxcredit) headstart = maxcredit;                                          \
    if (x == 1) headstart = maxcredit;                                                         \
    if (headstart > numblocks) headstart = numblocks;                                          \
    if (headstart == 0) headstart = 1;                                                         \
    userbuffers_fp16_sum_inplace_gpu_rr_blocked2<x><<<sms, warps * 32, 0, stream>>>(           \
        op, maxcredit, headstart, my_node, num_nodes,                                          \
        REG0_OFFSET(comm) + REG0_FLAGS +                                                       \
            (op == userbuffers_allreduceop_nonsharp ? REG0_COMMBUFFER : 0),                    \
        REG0_OFFSET(comm) + REG0_OPFLAGS * op, ar_firstgpu, ar_nvrank, ar_step, offset / 8,    \
        elements / 8, (void **)(comm->gpu_ptrs), handler * comm->nvsize,                       \
        blocksize / sizeof(int4) / ar_nvsize, (int *)comm->hostflags, comm->flags, numblocks); \
  }

#define callranks2_block_rs(x)                                                                 \
  if (ar_nvsize == x) {                                                                        \
    int numblocks = (elements * 2 + blocksize - 1) / blocksize;                                \
    int headstart = numblocks - 1; /*<3?numblocks-1:3;*/                                       \
    if (headstart > maxcredit) headstart = maxcredit;                                          \
    if (x == 1) headstart = maxcredit;                                                         \
    if (headstart > numblocks) headstart = numblocks;                                          \
    if (headstart == 0) headstart = 1;                                                         \
    userbuffers_fp16_sum_inplace_gpu_rr_blocked2_rs<x><<<sms, warps * 32, 0, stream>>>(        \
        op, maxcredit, headstart, my_node, num_nodes,                                          \
        REG0_OFFSET(comm) + REG0_FLAGS +                                                       \
            (op == userbuffers_allreduceop_nonsharp ? REG0_COMMBUFFER : 0),                    \
        REG0_OFFSET(comm) + REG0_OPFLAGS * op, ar_firstgpu, ar_nvrank, ar_step, offset / 8,    \
        elements / 8, (void **)(comm->gpu_ptrs), handler * comm->nvsize,                       \
        blocksize / sizeof(int4) / ar_nvsize, (int *)comm->hostflags, comm->flags, numblocks); \
  }

#define callranks2_block_ag(x)                                                                 \
  if (ar_nvsize == x) {                                                                        \
    int numblocks = (elements * 2 + blocksize - 1) / blocksize;                                \
    int headstart = numblocks - 1; /*<3?numblocks-1:3;*/                                       \
    if (headstart > maxcredit) headstart = maxcredit;                                          \
    if (x == 1) headstart = maxcredit;                                                         \
    if (headstart > numblocks) headstart = numblocks;                                          \
    if (headstart == 0) headstart = 1;                                                         \
    userbuffers_fp16_sum_inplace_gpu_rr_blocked2_ag<x><<<sms, warps * 32, 0, stream>>>(        \
        op, maxcredit, headstart, my_node, num_nodes,                                          \
        REG0_OFFSET(comm) + REG0_FLAGS +                                                       \
            (op == userbuffers_allreduceop_nonsharp ? REG0_COMMBUFFER : 0),                    \
        REG0_OFFSET(comm) + REG0_OPFLAGS * op, ar_firstgpu, ar_nvrank, ar_step, offset / 8,    \
        elements / 8, (void **)(comm->gpu_ptrs), handler * comm->nvsize,                       \
        blocksize / sizeof(int4) / ar_nvsize, (int *)comm->hostflags, comm->flags, numblocks); \
  }

#endif

#define callranks(x)                                                                         \
  if (ar_nvsize == x) {                                                                      \
    int arg1 = op - MAX_OPS,                                                                 \
        arg2 = REG0_OFFSET(comm) -                                                           \
               (op == userbuffers_allreduceop_nonsharp ? 2 : 1) * REG0_SINGLENODE + MAX_OPS, \
        arg3 = ar_firstgpu, arg4 = ar_nvrank, arg5 = ar_step, arg6 = offset / 8,             \
        arg7 = elements / 8;                                                                 \
    void **arg8 = (void **)(comm->gpu_ptrs);                                                 \
    int arg9 = handler * comm->nvsize;                                                       \
    void *kernelArgs[] = {                                                                   \
        (void *)&arg1, (void *)&arg2, (void *)&arg3, (void *)&arg4, (void *)&arg5,           \
        (void *)&arg6, (void *)&arg7, (void *)&arg8, (void *)&arg9};                         \
    CUDACHECK(cudaLaunchKernelExC(                                                           \
        &cfg,                                                                                \
        (void *)(comm->use_rr_kernel ? userbuffers_fp16_sum_inplace_gpu_rr<x>                \
                                     : userbuffers_fp16_sum_inplace_gpu_rw<x>),              \
        kernelArgs));                                                                        \
  }

#define SETUP_LAUNCH_CONFIG(sms, threads, stream)                                    \
  cudaLaunchConfig_t cfg = {sms, threads, 0, stream, NULL, 0};                       \
  cudaLaunchAttribute attribute_ub[2];                                               \
  attribute_ub[1].id = cudaLaunchAttributeClusterDimension;                          \
  attribute_ub[1].val.clusterDim.x = sms % comm->cga_size == 0 ? comm->cga_size : 1; \
  attribute_ub[1].val.clusterDim.y = 1;                                              \
  attribute_ub[1].val.clusterDim.z = 1;                                              \
  attribute_ub[0].id = cudaLaunchAttributeCooperative;                               \
  cfg.attrs = attribute_ub;                                                          \
  cfg.numAttrs = comm->sm_arch >= 9 ? 2 : 1;

int allreduce_userbuff_inplace_gpu(const int handler, const int offset, const int elements,
                                   const int blocksize, communicator *comm, cudaStream_t stream) {
  // schedule GPU kernel only
  // CPU/SHARP part is responsibility of caller
  const int ar_step = comm->ar2_nvsize;
  const int op = userbuffers_allreduceop_nonsharp;
  const int ar_nvsize = comm->nvsize;
  const int ar_firstgpu = comm->ar_firstgpu;
  const int ar_nvrank = comm->ar_nvrank;
  if (elements < 8) return 0;
  int sms = sms = comm->sms;
  int warps = comm->threads / 32;
  if (warps < comm->ar_nvsize) warps = comm->ar_nvsize;
  // if(!comm->myrank) printf("allreduce %d bytes running %d x %d\n",elements*2,sms,warps*32);

  if (comm->launch_mode & LAUNCH_GPU) {
#ifdef MULTINODE

    if (comm->ar_nvsize == 1)
      userbuffers_fp16_sum_inplace_gpu_null<<<1, 1, 0, stream>>>(
          userbuffers_allreduceop_sharp, (int *)comm->hostflags, comm->flags,
          (elements * 2 + blocksize - 1) / blocksize);
    callranks_block(2) callranks_block(4) callranks_block(8)

#else
    SETUP_LAUNCH_CONFIG(sms, warps * 32, stream);
    callranks(2) callranks(4) callranks(8)
// callranks(16)
#endif
  }
  return sms;
}

int allreduce2_userbuff_inplace_gpu(const int maxcredit, const int handler, const int offset,
                                    const int elements, const int blocksize, communicator *comm,
                                    cudaStream_t stream, int op) {
  // schedule GPU kernel only
  // CPU/SHARP part is responsibility of caller
#ifdef MULTINODE
  const int num_nodes = op == userbuffers_allreduceop_nonsharp ? comm->num_nodes : comm->num2_nodes;
  const int my_node = op == userbuffers_allreduceop_nonsharp ? comm->my_node : comm->my2_node;
#endif
  const int ar_firstgpu =
      op == userbuffers_allreduceop_nonsharp ? comm->ar_firstgpu : comm->ar2_firstgpu;
  const int ar_step = op == userbuffers_allreduceop_nonsharp2 ? 1 : comm->ar2_nvsize;
  const int ar_nvsize = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvsize : comm->ar2_nvsize;
  const int ar_nvrank = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvrank : comm->ar2_nvrank;

  if (elements < 8) return 0;
  int sms = ar_nvsize == 1 ? 2 : comm->sms;
  int warps = comm->threads / 32;
  if (warps < ar_nvsize) warps = ar_nvsize;
#ifdef MULTINODE
  if (num_nodes > 1) {
    callranks2_block(1) callranks2_block(2) callranks2_block(4) callranks2_block(8)
  } else {
#endif
    SETUP_LAUNCH_CONFIG(sms, warps * 32, stream);
    callranks(2) callranks(4) callranks(8)
#ifdef MULTINODE
  }
#endif

  return sms;
}

#define callranks_ag(x)                                                                      \
  if (ar_nvsize == x) {                                                                      \
    int arg1 = op - MAX_OPS,                                                                 \
        arg2 = REG0_OFFSET(comm) -                                                           \
               (op == userbuffers_allreduceop_nonsharp ? 2 : 1) * REG0_SINGLENODE + MAX_OPS, \
        arg3 = ar_firstgpu, arg4 = ar_nvrank, arg5 = ar_step, arg7 = elements / 8 / x,       \
        arg6 = offset / 8 + (comm->use_rr_kernel ? 0 : arg4 * arg7);                         \
    void **arg8 = (void **)(comm->gpu_ptrs);                                                 \
    int arg9 = handler * comm->nvsize;                                                       \
    void *kernelArgs[] = {                                                                   \
        (void *)&arg1, (void *)&arg2, (void *)&arg3, (void *)&arg4, (void *)&arg5,           \
        (void *)&arg6, (void *)&arg7, (void *)&arg8, (void *)&arg9};                         \
    CUDACHECK(cudaLaunchKernelExC(                                                           \
        &cfg,                                                                                \
        (void *)(comm->use_rr_kernel ? userbuffers_fp16_sum_inplace_gpu_rr_ag<x>             \
                                     : userbuffers_fp16_sum_inplace_gpu_rw_ag<x>),           \
        kernelArgs));                                                                        \
  }

#define callranks_rs(x)                                                                            \
  if (ar_nvsize == x) {                                                                            \
    int arg1 = op - MAX_OPS,                                                                       \
        arg2 = REG0_OFFSET(comm) -                                                                 \
               (op == userbuffers_allreduceop_nonsharp ? 2 : 1) * REG0_SINGLENODE + MAX_OPS,       \
        arg3 = ar_firstgpu, arg4 = ar_nvrank, arg5 = ar_step, arg7 = elements / 8 / x,             \
        arg6 = offset / 8 + arg4 * arg7;                                                           \
    void **arg8 = (void **)(comm->gpu_ptrs);                                                       \
    int arg9 = handler * comm->nvsize;                                                             \
    void *kernelArgs[] = {                                                                         \
        (void *)&arg1, (void *)&arg2, (void *)&arg3, (void *)&arg4, (void *)&arg5,                 \
        (void *)&arg6, (void *)&arg7, (void *)&arg8, (void *)&arg9};                               \
    CUDACHECK(                                                                                     \
        cudaLaunchKernelExC(&cfg, (void *)userbuffers_fp16_sum_inplace_gpu_rr_rs<x>, kernelArgs)); \
  }

#define callranks_rs_oop(x)                                                                    \
  if (ar_nvsize == x) {                                                                        \
    int arg1 = op - MAX_OPS,                                                                   \
        arg2 = REG0_OFFSET(comm) -                                                             \
               (op == userbuffers_allreduceop_nonsharp ? 2 : 1) * REG0_SINGLENODE + MAX_OPS,   \
        arg3 = ar_firstgpu, arg4 = ar_nvrank, arg5 = ar_step, arg7 = elements / 8 / x,         \
        arg6 = offset / 8 + arg4 * arg7, arg8 = rowelements / 8, arg9 = strideelements / 8;    \
    void **arg10 = (void **)(comm->gpu_ptrs);                                                  \
    int arg11 = handler * comm->nvsize;                                                        \
    void *arg12 = output;                                                                      \
    void *kernelArgs[] = {(void *)&arg1, (void *)&arg2,  (void *)&arg3,  (void *)&arg4,        \
                          (void *)&arg5, (void *)&arg6,  (void *)&arg7,  (void *)&arg8,        \
                          (void *)&arg9, (void *)&arg10, (void *)&arg11, (void *)&arg12};      \
    CUDACHECK(cudaLaunchKernelExC(&cfg, (void *)userbuffers_fp16_sum_inplace_gpu_rr_rs_oop<x>, \
                                  kernelArgs));                                                \
  }

#if defined(MULTINODE) && defined(NOSHARP)
int reducescatter2_userbuff_inplace_gpu(const int maxcredit, const int handler, const int offset,
                                        const int elements, const int blocksize, communicator *comm,
                                        cudaStream_t stream, int op) {
  // schedule GPU kernel only
  // CPU/SHARP part is responsibility of caller

  const int num_nodes = op == userbuffers_allreduceop_nonsharp ? comm->num_nodes : comm->num2_nodes;
  const int my_node = op == userbuffers_allreduceop_nonsharp ? comm->my_node : comm->my2_node;
  const int ar_firstgpu =
      op == userbuffers_allreduceop_nonsharp ? comm->ar_firstgpu : comm->ar2_firstgpu;
  const int ar_step = op == userbuffers_allreduceop_nonsharp2 ? 1 : comm->ar2_nvsize;
  const int ar_nvsize = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvsize : comm->ar2_nvsize;
  const int ar_nvrank = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvrank : comm->ar2_nvrank;

  if (elements < 8) return 0;
  int sms = ar_nvsize == 1 ? 2 : comm->sms;
  int warps = comm->threads / 32;
  if (warps < ar_nvsize) warps = ar_nvsize;

  if (num_nodes > 1) {
    callranks2_block_rs(1) callranks2_block_rs(2) callranks2_block_rs(4) callranks2_block_rs(8)
  } else {
    SETUP_LAUNCH_CONFIG(sms, warps * 32, stream);
    callranks_rs(2) callranks_rs(4) callranks_rs(8)
  }
  return sms;
}

int allgather2_userbuff_inplace_gpu(const int maxcredit, const int handler, const int offset,
                                    const int elements, const int blocksize, communicator *comm,
                                    cudaStream_t stream, int op) {
  // schedule GPU kernel only
  // CPU/SHARP part is responsibility of caller

  const int num_nodes = op == userbuffers_allreduceop_nonsharp ? comm->num_nodes : comm->num2_nodes;
  const int my_node = op == userbuffers_allreduceop_nonsharp ? comm->my_node : comm->my2_node;
  const int ar_firstgpu =
      op == userbuffers_allreduceop_nonsharp ? comm->ar_firstgpu : comm->ar2_firstgpu;
  const int ar_step = op == userbuffers_allreduceop_nonsharp2 ? 1 : comm->ar2_nvsize;
  const int ar_nvsize = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvsize : comm->ar2_nvsize;
  const int ar_nvrank = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvrank : comm->ar2_nvrank;

  if (elements < 8) return 0;
  int sms = ar_nvsize == 1 ? 2 : comm->sms;
  int warps = comm->threads / 32;
  if (warps < ar_nvsize) warps = ar_nvsize;

  if (num_nodes > 1) {
    callranks2_block_ag(1) callranks2_block_ag(2) callranks2_block_ag(4) callranks2_block_ag(8)
  } else {
    SETUP_LAUNCH_CONFIG(sms, warps * 32, stream);
    callranks_ag(2) callranks_ag(4) callranks_ag(8)
  }
  return sms;
}

#endif

void allgather2_userbuff_inplace(const int handler, const int offset, const int elements,
                                 communicator *comm, cudaStream_t stream) {
  const int op = userbuffers_allreduceop_nonsharp2;
  const int blocksize = elements * 2;
  const int ar_firstgpu =
      op == userbuffers_allreduceop_nonsharp ? comm->ar_firstgpu : comm->ar2_firstgpu;
  const int ar_step = op == userbuffers_allreduceop_nonsharp2 ? 1 : comm->ar2_nvsize;
  const int ar_nvsize = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvsize : comm->ar2_nvsize;
  const int ar_nvrank = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvrank : comm->ar2_nvrank;

  if (elements < 64) return;
  int sms = ar_nvsize == 1 ? 2 : comm->sms;
  int warps = comm->threads / 32;
  if (warps < ar_nvsize) warps = ar_nvsize;

  SETUP_LAUNCH_CONFIG(sms, warps * 32, stream);
  callranks_ag(2) callranks_ag(4) callranks_ag(8)
}

void allgather2_userbuff_inplace_sliced(const int handler, const int offset, const int elements,
                                        communicator *comm, const int slice_id, const int nslices,
                                        cudaStream_t stream) {
  const int op = userbuffers_allreduceop_nonsharp2;
  const int ar_nvrank = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvrank : comm->ar2_nvrank;
  const int ar_nvsize = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvsize : comm->ar2_nvsize;
  int peerelements = elements / ar_nvsize;
  int saverrkernel = comm->use_rr_kernel;
  comm->use_rr_kernel = 0;
  allgather2_userbuff_inplace(
      handler, offset + ar_nvrank * peerelements * (nslices - 1) + slice_id * peerelements,
      elements, comm, stream);
  comm->use_rr_kernel = saverrkernel;
}

void reducescatter2_userbuff_inplace(const int handler, const int offset, const int elements,
                                     communicator *comm, cudaStream_t stream) {
  const int op = userbuffers_allreduceop_nonsharp2;
  const int blocksize = elements * 2;
  const int ar_firstgpu =
      op == userbuffers_allreduceop_nonsharp ? comm->ar_firstgpu : comm->ar2_firstgpu;
  const int ar_step = op == userbuffers_allreduceop_nonsharp2 ? 1 : comm->ar2_nvsize;
  const int ar_nvsize = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvsize : comm->ar2_nvsize;
  const int ar_nvrank = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvrank : comm->ar2_nvrank;

  if (elements < 64) return;
  int sms = ar_nvsize == 1 ? 2 : comm->sms;
  int warps = comm->threads / 32;
  if (warps < ar_nvsize) warps = ar_nvsize;

  SETUP_LAUNCH_CONFIG(sms, warps * 32, stream);
  callranks_rs(2) callranks_rs(4) callranks_rs(8)
}
void reducescatter2_userbuff_stridedoutput(void *output, const int handler, const int offset,
                                           const int rowelements, const int colelements,
                                           const int strideelements, communicator *comm,
                                           cudaStream_t stream) {
  const int elements = rowelements * colelements;
  const int op = userbuffers_allreduceop_nonsharp2;
  const int blocksize = elements * 2;
  const int ar_firstgpu =
      op == userbuffers_allreduceop_nonsharp ? comm->ar_firstgpu : comm->ar2_firstgpu;
  const int ar_step = op == userbuffers_allreduceop_nonsharp2 ? 1 : comm->ar2_nvsize;
  const int ar_nvsize = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvsize : comm->ar2_nvsize;
  const int ar_nvrank = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvrank : comm->ar2_nvrank;

  if (elements < 64) return;
  int sms = ar_nvsize == 1 ? 2 : comm->sms;
  int warps = comm->threads / 32;
  if (warps < ar_nvsize) warps = ar_nvsize;

  SETUP_LAUNCH_CONFIG(sms, warps * 32, stream);
  callranks_rs_oop(2) callranks_rs_oop(4) callranks_rs_oop(8)
}
void reducescatter2_userbuff(void *output, const int handler, const int offset, const int elements,
                             communicator *comm, cudaStream_t stream) {
  reducescatter2_userbuff_stridedoutput(output, handler, offset, elements, 1, 0, comm, stream);
}

__global__ void kuserbuffers_pullsend(int myrank, int peer, int *send_id, int *flagptr) {
  // const int signal_id=(*send_id)+1;
  //*send_id=signal_id;
  // printf("%d to %d pullsend: sending incremented %d\n",myrank,peer,signal_id);
  //*flagptr=signal_id;
  atomicAdd(flagptr, 1);
}

__global__ void kuserbuffers_inc(int *id) {
  const int signal_id = (*id) + 1;
  *id = signal_id;
}

__global__ void kuserbuffers_proxysend(int *id, int *hostflag) {
  const int signal_id = (*id) + 1;
  *hostflag = signal_id;
  *id = signal_id;
}

__global__ void kuserbuffers_dummy(void) {}

__global__ void __launch_bounds__(MAX_THREADS)
    kuserbuffers_pullrecv(int myrank, int peer, int *recv_id, int *flagptr, int4 *srcptr,
                          int4 *dstptr, const int lines) {
#define UNROLLCOPY 8
  const int start_elem = threadIdx.x + blockDim.x * blockIdx.x;
  const int end_elem = lines;
  const int aligned_elem = (end_elem - start_elem) & (~(blockDim.x * gridDim.x * UNROLLCOPY - 1));
  // /(blockDim.x*gridDim.x*UNROLLCOPY)) * (blockDim.x*gridDim.x*UNROLLCOPY);
  const int end_aligned = start_elem + aligned_elem;

  if (threadIdx.x == 0) {
    const int signal_id = (*recv_id) + 1;
    volatile int *flag = (volatile int *)flagptr;
    // printf("[%d from %d] pullrecv: waiting %d, read  %d\n",myrank,peer,signal_id,*flag);
    clock_t s = clock64();
    while (*flag < signal_id) {
      if (clock64() - s > TIMEOUT) {
        printf("[%d from %d] pullrecv: expected %d, stuck with %d\n", myrank, peer, signal_id,
               *flag);
        break;
      }
    }
    // printf("[%d from %d] pullrecv: done %d [%d]\n",myrank,peer,signal_id,*flag);
    if (lines == 0) {
      *recv_id = signal_id;
      return;
    }  // otherwise need an extra kernel
  }
  __syncthreads();

  if (end_elem <= start_elem) return;

  for (int line = start_elem; line < end_aligned; line += blockDim.x * gridDim.x * UNROLLCOPY) {
    int4 val[UNROLLCOPY];
#pragma unroll
    for (int i = 0; i < UNROLLCOPY; i++) val[i] = srcptr[line + i * blockDim.x * gridDim.x];
#pragma unroll
    for (int i = 0; i < UNROLLCOPY; i++) dstptr[line + i * blockDim.x * gridDim.x] = val[i];
  }
  for (int line = end_aligned; line < end_elem; line += blockDim.x * gridDim.x)
    dstptr[line] = srcptr[line];
}

__global__ void __launch_bounds__(MAX_THREADS)
    kuserbuffers_pushsend(int *send_id, int *flagptr, int4 *srcptr, int4 *dstptr, const int lines) {
  if (lines) {
    const int start_elem = threadIdx.x + blockDim.x * blockIdx.x;
    const int end_elem = lines;
    const int aligned_elem =
        ((end_elem - start_elem) & (~(blockDim.x * gridDim.x * UNROLLCOPY - 1)));
    // / (blockDim.x*gridDim.x*UNROLLCOPY) )* (blockDim.x*gridDim.x*UNROLLCOPY);
    const int end_aligned = start_elem + aligned_elem;
    if (end_elem > start_elem) {
      for (int line = start_elem; line < end_aligned; line += blockDim.x * gridDim.x * UNROLLCOPY) {
        int4 val[UNROLLCOPY];
#pragma unroll
        for (int i = 0; i < UNROLLCOPY; i++) val[i] = srcptr[line + i * blockDim.x * gridDim.x];
#pragma unroll
        for (int i = 0; i < UNROLLCOPY; i++) dstptr[line + i * blockDim.x * gridDim.x] = val[i];
      }
      for (int line = end_aligned; line < end_elem; line += blockDim.x * gridDim.x)
        dstptr[line] = srcptr[line];
    }
    __syncthreads();
    if (threadIdx.x) return;
    __threadfence_system();
    atomicAdd(flagptr, 1);  // otherwise need local SM sync before sending flag
    // if(blockIdx.x==0) (*send_id)+=gridDim.x;
  } else {  // 0 bytes and 1 SM only
    // const int signal_id=(*send_id)+1;
    //*flagptr=signal_id;
    atomicAdd(flagptr, 1);
    //*send_id=signal_id;
  }
}

__global__ void kuserbuffers_pushrecv(int myrank, int peer, int *recv_id, int *flagptr, int adder) {
  const int signal_id = (*recv_id) + adder;
  *recv_id = signal_id;
  volatile int *flag = (volatile int *)flagptr;
  if (*flag >= signal_id) return;
  clock_t s = clock64();
  while (*flag < signal_id) {
    if (clock64() - s > TIMEOUT) {
      printf("%d from %d] pushrecv: expected %d, stuck with %d\n", myrank, peer, signal_id, *flag);
      return;
    }
  }
}

#define CUDACHECK(cmd)                                                                      \
  do {                                                                                      \
    cudaError_t e = cmd;                                                                    \
    if (e != cudaSuccess) {                                                                 \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(EXIT_FAILURE);                                                                   \
    }                                                                                       \
  } while (0)

#ifdef UCP
#define INTRANODE(peer) ((peer / comm->ibnvsize) == (comm->myrank / comm->ibnvsize))
#else
#define INTRANODE(peer) ((peer / comm->nvsize) == (comm->myrank / comm->nvsize))
#endif

void userbuffers_send(const int srchandler, const size_t srcoffset, const int dsthandler,
                      const size_t dstoffset, const size_t bytes, communicator *comm,
                      const int peer, cudaStream_t stream) {
  int peerlocal = peer % comm->nvsize;
  void *flagptr =
      (comm->peer_ptr[0][peerlocal]) +
      ((REG0_OFFSET(comm) + REG0_RECV + comm->myrank * MAX_REGIONS + dsthandler) * sizeof(int));
  bool signalonly = (bytes / 16 == 0) || (comm->use_ce != 0);
  bool intranode = INTRANODE(peer);
// printf("%d: send to %d, push %d ce %d bytes %d signalonly %d sms %d intranode %d srco %lu dsto
// %lu\n",comm->myrank,peer,comm->push,comm->use_ce,bytes,signalonly,comm->sms,intranode,srcoffset,dstoffset);
// fflush(NULL);
#ifdef MULTINODE
  if (!intranode && (comm->launch_mode & LAUNCH_CPU)) {
    comm->fifo[comm->head].optype = userbuffers_sendop;
    comm->fifo[comm->head].basecounter = comm->basecounter[userbuffers_sendop];
    comm->fifo[comm->head].handler = srchandler;
    comm->fifo[comm->head].offset = srcoffset;
    comm->fifo[comm->head].handler2 = dsthandler;
    comm->fifo[comm->head].offset2 = dstoffset;
    comm->fifo[comm->head].elements = bytes;
    comm->fifo[comm->head].peer = peer;

    int newhead = (comm->head + 1) & (MAX_REQUESTS - 1);
    while (newhead == comm->tail) {
    }
    comm->head = newhead;
    comm->basecounter[userbuffers_sendop] += 1;
  }
  if (!intranode && (comm->launch_mode & LAUNCH_GPU)) {
    kuserbuffers_proxysend<<<1, 1, 0, stream>>>(&(comm->flags[GF_STATE + userbuffers_sendop]),
                                                comm->hostflags + userbuffers_sendop);
    return;
  }
#endif
  if (!(comm->launch_mode & LAUNCH_GPU)) return;
  if (comm->push == 0)
    kuserbuffers_pullsend<<<1, 1, 0, stream>>>(comm->myrank, peer, &(comm->send_id[peer]),
                                               (int *)flagptr);
  else {
    void *srcptr = (comm->mem_ptr[srchandler]) + srcoffset;
    void *dstptr = (comm->peer_ptr[dsthandler][peerlocal]) + dstoffset;

    if (comm->use_ce)
      CUDACHECK(cudaMemcpyAsync(dstptr, srcptr, bytes, cudaMemcpyDeviceToDevice, stream));
    // kuserbuffers_pushsend<<<signalonly?1:comm->sms,signalonly?1:1024,0,stream>>>
    //       (&comm->send_id[peer],(int*)flagptr,(int4*)srcptr,(int4*)dstptr,signalonly?0:bytes/16);
    SETUP_LAUNCH_CONFIG(signalonly ? 1 : comm->sms, signalonly ? 1 : 1024, stream);
    int *arg1 = &comm->send_id[peer], *arg2 = (int *)flagptr;
    int4 *arg3 = (int4 *)srcptr, *arg4 = (int4 *)dstptr;
    int arg5 = signalonly ? 0 : bytes / 16;
    void *kernelArgs[] = {(void *)&arg1, (void *)&arg2, (void *)&arg3, (void *)&arg4,
                          (void *)&arg5};
    CUDACHECK(cudaLaunchKernelExC(&cfg, (void *)kuserbuffers_pushsend, kernelArgs));
  }
}

__global__ void __launch_bounds__(MAX_THREADS)
    kuserbuffers_alltoall(void **baseflagptrs, int flagoffset, int4 *basesrcptr, void **dstptrs,
                          size_t dstoffset, const int lines, const int myrank) {
  if (blockIdx.x == myrank) return;
  int4 *dstptr = (int4 *)(dstptrs[blockIdx.x] + dstoffset);
  int *flagptr = (int *)(baseflagptrs[blockIdx.x] + flagoffset);
  const size_t myblockoffset = blockIdx.x * lines;
  int4 *srcptr = basesrcptr + myblockoffset;
  dstptr += myblockoffset;

  if (lines) {
    const int start_elem = threadIdx.x;
    const int end_elem = lines;
    const int aligned_elem = ((end_elem - start_elem) & (~(blockDim.x * UNROLLCOPY - 1)));
    // / (blockDim.x*gridDim.x*UNROLLCOPY) )* (blockDim.x*gridDim.x*UNROLLCOPY);
    const int end_aligned = start_elem + aligned_elem;
    if (end_elem > start_elem) {
      for (int line = start_elem; line < end_aligned; line += blockDim.x * UNROLLCOPY) {
        int4 val[UNROLLCOPY];
#pragma unroll
        for (int i = 0; i < UNROLLCOPY; i++) val[i] = srcptr[line + i * blockDim.x];
#pragma unroll
        for (int i = 0; i < UNROLLCOPY; i++) dstptr[line + i * blockDim.x] = val[i];
      }
      for (int line = end_aligned; line < end_elem; line += blockDim.x) dstptr[line] = srcptr[line];
    }
    __syncthreads();
    if (threadIdx.x) return;
    __threadfence_system();
    atomicAdd(flagptr, 1);

  } else {
    atomicAdd(flagptr, 1);
  }
}

void userbuffers_alltoall_send(const int srchandler, const size_t srcoffset, const int dsthandler,
                               const size_t dstoffset, const size_t bytes, communicator *comm,
                               cudaStream_t stream) {
#ifdef MULTINODE
  if (comm->launch_mode & LAUNCH_CPU) {
    comm->fifo[comm->head].optype = userbuffers_alltoall;
    comm->fifo[comm->head].basecounter = comm->basecounter[userbuffers_alltoall];
    comm->fifo[comm->head].handler = srchandler;
    comm->fifo[comm->head].offset = srcoffset;
    comm->fifo[comm->head].handler2 = dsthandler;
    comm->fifo[comm->head].offset2 = dstoffset;
    comm->fifo[comm->head].elements = bytes;

    int newhead = (comm->head + 1) & (MAX_REQUESTS - 1);
    while (newhead == comm->tail) {
    }
    comm->head = newhead;
    comm->basecounter[userbuffers_alltoall] += 1;
  }
  if (comm->launch_mode & LAUNCH_GPU)
    kuserbuffers_proxysend<<<1, 1, 0, stream>>>(&(comm->flags[GF_STATE + userbuffers_alltoall]),
                                                comm->hostflags + userbuffers_alltoall);
#else
  int lines = bytes / 16;
  void *srcptr = (comm->mem_ptr[srchandler]) + srcoffset;
  size_t flagoffset = ((REG0_OFFSET(comm) + REG0_OPFLAGS * userbuffers_alltoall) * sizeof(int));

  SETUP_LAUNCH_CONFIG(comm->nranks, lines == 0 ? 1 : 1024, stream);
  // kuserbuffers_alltoall<<<comm->nranks,lines==0?1:1024,0,stream>>>
  //           ((void**)(comm->gpu_ptrs),flagoffset,(int4*)srcptr,(void**)(comm->gpu_ptrs+dsthandler*comm->nvsize*sizeof(void*)),dstoffset,lines,comm->myrank);
  void **arg1 = (void **)(comm->gpu_ptrs);
  size_t arg2 = flagoffset;
  void *arg3 = srcptr;
  void **arg4 = (void **)(comm->gpu_ptrs + dsthandler * comm->nvsize * sizeof(void *));
  size_t arg5 = dstoffset;
  int arg6 = lines;
  int arg7 = comm->myrank;

  void *kernelArgs[] = {(void *)&arg1, (void *)&arg2, (void *)&arg3, (void *)&arg4,
                        (void *)&arg5, (void *)&arg6, (void *)&arg7};
  CUDACHECK(cudaLaunchKernelExC(&cfg, (void *)kuserbuffers_alltoall, kernelArgs));
#endif
}

void userbuffers_recv(const int srchandler, const size_t srcoffset, const int dsthandler,
                      const size_t dstoffset, const size_t bytes, communicator *comm,
                      const int peer, cudaStream_t stream) {
  int peerlocal = peer % comm->nvsize;
  void *flagptr = (comm->mem_ptr[0]) +
                  ((REG0_OFFSET(comm) + REG0_RECV + peer * MAX_REGIONS + dsthandler) * sizeof(int));
  bool signalonly = (bytes / 16 == 0) || (comm->use_ce != 0);
  bool intranode = INTRANODE(peer);
  if (!(comm->launch_mode & LAUNCH_GPU)) return;
  // printf("%d: recv from %d, push %d ce %d bytes %d signalonly %d sms
  // %d\n",comm->myrank,peer,comm->push,comm->use_ce,bytes,signalonly,comm->sms); fflush(NULL);
  if (comm->push == 0 && intranode) {
    void *dstptr = (comm->mem_ptr[dsthandler]) + dstoffset;
    void *srcptr = (comm->peer_ptr[srchandler][peerlocal]) + srcoffset;

    kuserbuffers_pullrecv<<<signalonly ? 1 : comm->sms, signalonly ? 1 : 1024, 0, stream>>>(
        comm->myrank, peer, &(comm->recv_id[peer * MAX_REGIONS + dsthandler]), (int *)flagptr,
        (int4 *)srcptr, (int4 *)dstptr, signalonly ? 0 : bytes / 16);
    if (!signalonly)
      kuserbuffers_inc<<<1, 1, 0, stream>>>(&(comm->recv_id[peer * MAX_REGIONS + dsthandler]));
    if (comm->use_ce) {
      CUDACHECK(cudaMemcpyAsync(dstptr, srcptr, bytes, cudaMemcpyDeviceToDevice, stream));
      // kuserbuffers_dummy<<<1,1,0,stream>>> ();
    }
  } else {
    kuserbuffers_pushrecv<<<1, 1, 0, stream>>>(
        comm->myrank, peer, &comm->recv_id[peer * MAX_REGIONS + dsthandler], (int *)flagptr,
        signalonly || !intranode ? 1 : comm->sms);
  }
}

void userbuffers_alltoall_recv(communicator *comm, cudaStream_t stream) {
  void *flagptr = (comm->mem_ptr[0]) +
                  ((REG0_OFFSET(comm) + REG0_OPFLAGS * userbuffers_alltoall) * sizeof(int));

  if (!(comm->launch_mode & LAUNCH_GPU)) return;
  kuserbuffers_pushrecv<<<1, 1, 0, stream>>>(comm->myrank, -1, (int *)(flagptr + 4), (int *)flagptr,
                                             comm->nranks - 1);
}
