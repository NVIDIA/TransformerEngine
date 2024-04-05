/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <stdio.h>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

#if __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#define half nv_bfloat16
#else
#include <cuda_fp16.h>
#endif

#include "userbuffers.h"

#define MAX_THREADS 1024
#define TIMEOUT 200000000000ull

#define CUDACHECK(cmd)                                                                             \
  do {                                                                                             \
    cudaError_t e = cmd;                                                                           \
    if (e != cudaSuccess) {                                                                        \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));        \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0)

#define ATOMIC_CONSUMER(chunk)                                                                     \
  if (counters) {                                                                                  \
    if (threadIdx.x == 0 && blockIdx.x == 0) {                                                     \
      int old_val;                                                                                 \
      while (0 != (old_val = atomicCAS(((unsigned int *)counters) + chunk, 0, 0))) {               \
      }                                                                                            \
      ((unsigned int *)counters)[chunk] = 1;                                                       \
      asm volatile("fence.sc.gpu;\n");                                                             \
    }                                                                                              \
    if (blockIdx.x == 0)                                                                           \
      __syncthreads();                                                                             \
  }

#define ATOMIC_PRODUCER(chunk)                                                                     \
  if (counters) {                                                                                  \
    ((unsigned int *)counters)[chunk] = 0;                                                         \
  }

// Return true if producer > consumer, otherwise false while preventing integer overflow
// If we expect that producer will be 2B+ messages behind consumer
#define CHECK_IDS(producer, consumer) (((unsigned)(producer) - (unsigned)(consumer)) & (~INT_MAX))

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_rw(const int op, const int flagoffset, const int firstrank,
                                        const int myrank, const int gpustep, const int lineoffset,
                                        const int numlines, void **commbuff, const int handleridx) {
  __shared__ int4 *userptr[RANKS];
  int *flagptr, physgpu, targetgpu, *myptr;
  int *reduceidptr, reduce_id;

  if (threadIdx.x < RANKS) {
    physgpu = myrank * gpustep + firstrank;
    targetgpu = threadIdx.x * gpustep + firstrank;
    const int blockflagoffset = NVTE_MAX_NVLINK * 2 * blockIdx.x;
    myptr = (reinterpret_cast<int *>(commbuff[physgpu])) + flagoffset;
    reduceidptr = myptr - NVTE_MAX_OPS;  // +op;
    reduce_id = (*reduceidptr) + 1;
    flagptr = (reinterpret_cast<int *>(commbuff[targetgpu])) + flagoffset + blockflagoffset;
    myptr += blockflagoffset;

    flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&(myptr[targetgpu]);
    userptr[threadIdx.x] = reinterpret_cast<int4 *>(commbuff[targetgpu + handleridx]);
    clock_t s = clock64();
    while (CHECK_IDS(*flag, reduce_id)) {
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
  for (int i = 0; i < RANKS; i++)
    dest[i] = (i + myrank + warp) & (RANKS - 1);

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
      for (int j = 0; j < 8; j++)
        s[j] += x[j];
    }
#pragma unroll
    for (int i = 0; i < RANKS; i++) {
      // int dest = (i+myrank+warp)&(RANKS-1);
      userptr[dest[i]][lineoffset + line] = sum;
    }
  }

  __syncthreads();
  if (threadIdx.x == 0)
    __threadfence_system();
  __syncthreads();

  if (threadIdx.x < RANKS) {
    flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&myptr[targetgpu];
    clock_t s = clock64();
    while (CHECK_IDS(*flag, reduce_id)) {
      if (clock64() - s > 2ull * TIMEOUT) {
        printf("NVONLY AGBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, reduce_id,
               *flag);
        break;
      }
    }
  }
  if (threadIdx.x == 0 && blockIdx.x == 0)
    *reduceidptr = reduce_id;
}  // fp16 inplace reduce kernel (Volta,Hopper)

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_rr(const int op, const int flagoffset, const int firstrank,
                                        const int myrank, const int gpustep, const int lineoffset,
                                        const int numlines, void **commbuff, const int handleridx) {
  __shared__ int4 *userptr[RANKS];
  int *flagptr, physgpu, targetgpu, *myptr;
  int *reduceidptr, reduce_id;
  if (threadIdx.x < RANKS) {
    physgpu = myrank * gpustep + firstrank;
    targetgpu = threadIdx.x * gpustep + firstrank;
    const int blockflagoffset = NVTE_MAX_NVLINK * 2 * blockIdx.x;
    myptr = (reinterpret_cast<int *>(commbuff[physgpu])) + flagoffset;
    reduceidptr = myptr - NVTE_MAX_OPS;  // +op;
    reduce_id = (*reduceidptr) + 1;
    flagptr = (reinterpret_cast<int *>(commbuff[targetgpu])) + flagoffset + blockflagoffset;
    myptr += blockflagoffset;

    flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&(myptr[targetgpu]);
    userptr[threadIdx.x] = reinterpret_cast<int4 *>(commbuff[targetgpu + handleridx]);
    clock_t s = clock64();
    while (CHECK_IDS(*flag, reduce_id)) {
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
  for (int i = 0; i < RANKS; i++)
    dest[i] = (i + myrank + warp) & (RANKS - 1);

  __syncthreads();
  for (int line = threadIdx.x + blockDim.x * (myrank + RANKS * blockIdx.x); line < numlines;
       line += blockDim.x * gridDim.x * RANKS) {
    int4 val[RANKS];

#pragma unroll
    for (int i = 0; i < RANKS; i++) {
      val[i] = userptr[dest[i]][lineoffset + line];
    }

    int4 sum = val[0];
    half *s = reinterpret_cast<half *>(&sum);

#pragma unroll
    for (int i = 1; i < RANKS; i++) {
      half *x = reinterpret_cast<half *>(&val[i]);
#pragma unroll
      for (int j = 0; j < 8; j++)
        s[j] += x[j];
    }

    userptr[myrank][lineoffset + line] = sum;
  }
  __syncthreads();
  if (threadIdx.x == 0)
    __threadfence();
  __syncthreads();

  if (threadIdx.x < RANKS) {
    flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&myptr[targetgpu];
    clock_t s = clock64();
    while (CHECK_IDS(*flag, reduce_id)) {
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
      val[i] = userptr[dest[i]][lineoffset + line + blockDim.x * dest[i]];
    }

#pragma unroll
    for (int i = 0; i < RANKS - 1; i++) {
      userptr[myrank][lineoffset + line + blockDim.x * dest[i]] = val[i];
    }
  }
  if (threadIdx.x == 0 && blockIdx.x == 0)
    *reduceidptr = reduce_id;
}  // fp16 inplace reduce kernel (Ampere)

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_rr_rs(const int op, const int flagoffset, const int firstrank,
                                           const int myrank, const int gpustep,
                                           const int mylineoffset, const int totallines,
                                           void **commbuff, const int handleridx) {
  __shared__ int4 *userptr[RANKS];
  volatile int *flagptr;
  int physgpu, targetgpu, *myptr;
  int *reduceidptr, reduce_id;
  int lastSM = 0;
  if (threadIdx.x < RANKS) {
    physgpu = myrank * gpustep + firstrank;
    targetgpu = threadIdx.x * gpustep + firstrank;
    myptr = (reinterpret_cast<int *>(commbuff[physgpu])) + flagoffset;
    reduceidptr = myptr - NVTE_MAX_OPS;  // +op;
    reduce_id = (*reduceidptr) + 1;
    flagptr = (reinterpret_cast<int *>(commbuff[targetgpu])) + flagoffset;
    if (blockIdx.x == 0)
      flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&(myptr[targetgpu]);
    userptr[threadIdx.x] = reinterpret_cast<int4 *>(commbuff[targetgpu + handleridx]);
    clock_t s = clock64();
    while (CHECK_IDS(*flag, reduce_id)) {
      if (clock64() - s > TIMEOUT) {
        printf("NVONLY RSBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, reduce_id,
               *flag);
        break;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const int adder = blockIdx.x == 0 ? NVTE_MAX_SMS - gridDim.x + 1 : 1;
    int old_val = atomicAdd(myptr + (NVTE_MAX_NVLINK * 2), adder);
    if (old_val + adder == NVTE_MAX_SMS * reduce_id)
      lastSM = 1;
  }

  int warp = blockIdx.x + (threadIdx.x >> 5);
  int dest[RANKS];
#pragma unroll
  for (int i = 0; i < RANKS; i++)
    dest[i] = (i + myrank + warp) & (RANKS - 1);

  __syncthreads();
  for (int line = threadIdx.x + blockDim.x * blockIdx.x; line < totallines;
       line += blockDim.x * gridDim.x) {
    int4 val[RANKS];

#pragma unroll
    for (int i = 0; i < RANKS; i++) {
      val[i] = userptr[dest[i]][mylineoffset + line];
    }

    int4 sum = val[0];
    half *s = reinterpret_cast<half *>(&sum);

#pragma unroll
    for (int i = 1; i < RANKS; i++) {
      half *x = reinterpret_cast<half *>(&val[i]);
#pragma unroll
      for (int j = 0; j < 8; j++)
        s[j] += x[j];
    }

    userptr[myrank][mylineoffset + line] = sum;
  }

  if (threadIdx.x == 0 && lastSM)
    *reduceidptr = reduce_id;
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
  volatile int *flagptr;
  int physgpu, targetgpu, *myptr;
  int *reduceidptr, reduce_id;
  int lastSM = 0;
  if (threadIdx.x < RANKS) {
    physgpu = myrank * gpustep + firstrank;
    targetgpu = threadIdx.x * gpustep + firstrank;
    myptr = (reinterpret_cast<int *>(commbuff[physgpu])) + flagoffset;
    reduceidptr = myptr - NVTE_MAX_OPS;  // +op;
    reduce_id = (*reduceidptr) + 1;
    flagptr = (reinterpret_cast<int *>(commbuff[targetgpu])) + flagoffset;
    if (blockIdx.x == 0)
      flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&(myptr[targetgpu]);
    userptr[threadIdx.x] = reinterpret_cast<int4 *>(commbuff[targetgpu + handleridx]);
    clock_t s = clock64();
    while (CHECK_IDS(*flag, reduce_id)) {
      if (clock64() - s > TIMEOUT) {
        printf("NVONLY RSBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, reduce_id,
               *flag);
        break;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const int adder = blockIdx.x == 0 ? NVTE_MAX_SMS - gridDim.x + 1 : 1;
    int old_val = atomicAdd(myptr + (NVTE_MAX_NVLINK * 2), adder);
    if (old_val + adder == NVTE_MAX_SMS * reduce_id)
      lastSM = 1;
  }

  int warp = blockIdx.x + (threadIdx.x >> 5);
  int dest[RANKS];
#pragma unroll
  for (int i = 0; i < RANKS; i++)
    dest[i] = (i + myrank + warp) & (RANKS - 1);

  __syncthreads();
  for (int line = threadIdx.x + blockDim.x * blockIdx.x; line < totallines;
       line += blockDim.x * gridDim.x) {
    int4 val[RANKS];

#pragma unroll
    for (int i = 0; i < RANKS; i++) {
      val[i] = userptr[dest[i]][mylineoffset + line];
    }

    int4 sum = val[0];
    half *s = reinterpret_cast<half *>(&sum);

#pragma unroll
    for (int i = 1; i < RANKS; i++) {
      half *x = reinterpret_cast<half *>(&val[i]);
#pragma unroll
      for (int j = 0; j < 8; j++)
        s[j] += x[j];
    }

    (reinterpret_cast<int4 *>(outbuf))[(line / rowlines) * skiplines + (line % rowlines)] = sum;
  }

  if (threadIdx.x == 0 && lastSM)
    *reduceidptr = reduce_id;
}  // fp16 reduce-scatter kernel (out of place)

#if __CUDA_ARCH__ >= 900
// All MC kernels here
template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_mc(const int op, const int flagoffset, const int firstrank,
                                        const int myrank, const int gpustep, const int lineoffset,
                                        const int numlines, void **commbuff, const int handleridx,
                                        float4 *mc_ptr) {
  int *flagptr, physgpu, targetgpu, *myptr;
  int *reduceidptr, reduce_id;

  if (threadIdx.x < RANKS) {
    physgpu = myrank * gpustep + firstrank;
    targetgpu = threadIdx.x * gpustep + firstrank;
    const int blockflagoffset = NVTE_MAX_NVLINK * 2 * blockIdx.x;
    myptr = (reinterpret_cast<int *>(commbuff[physgpu])) + flagoffset;
    reduceidptr = myptr - NVTE_MAX_OPS;  // +op;
    reduce_id = (*reduceidptr) + 1;
    flagptr = (reinterpret_cast<int *>(commbuff[targetgpu])) + flagoffset + blockflagoffset;
    myptr += blockflagoffset;

    flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&(myptr[targetgpu]);
    clock_t s = clock64();
    while (CHECK_IDS(*flag, reduce_id)) {
      if (clock64() - s > TIMEOUT) {
        printf("NVONLY RSBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, reduce_id,
               *flag);
        break;
      }
    }
    reduce_id++;
  }
  __syncthreads();
#define UNROLL_MC 8
  const int loop_step0 = blockDim.x * gridDim.x * RANKS;
  const int loop_step = loop_step0 * UNROLL_MC;
  const int start_elem = threadIdx.x + blockDim.x * (myrank + RANKS * blockIdx.x);
  const int end_elem = max(start_elem, numlines);
  const int aligned_elem = ((end_elem - start_elem) / loop_step) * loop_step;
  const int end_aligned = start_elem + aligned_elem;

  for (int line = start_elem; line < end_aligned; line += loop_step) {
    uint4 val[UNROLL_MC];
#pragma unroll
    for (int i = 0; i < UNROLL_MC; i++)
#if defined(NVTE_UB_FP16)
      asm("multimem.ld_reduce.global.add.v4.f16x2 {%0,%1,%2,%3}, [%4];"
          : "=r"(val[i].x), "=r"(val[i].y), "=r"(val[i].z), "=r"(val[i].w)
          : "l"(mc_ptr + (lineoffset + line + i * loop_step0))
          : "memory");
#else
      asm("multimem.ld_reduce.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
          : "=r"(val[i].x), "=r"(val[i].y), "=r"(val[i].z), "=r"(val[i].w)
          : "l"(mc_ptr + (lineoffset + line + i * loop_step0))
          : "memory");
#endif
#pragma unroll
    for (int i = 0; i < UNROLL_MC; i++)
      asm volatile("multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(
                       mc_ptr + (lineoffset + line + i * loop_step0)),
                   "r"(val[i].x), "r"(val[i].y), "r"(val[i].z), "r"(val[i].w)
                   : "memory");
  }
  for (int line = end_aligned; line < end_elem; line += loop_step0) {
    uint4 val;
#if defined(NVTE_UB_FP16)
    asm("multimem.ld_reduce.global.add.v4.f16x2 {%0,%1,%2,%3}, [%4];"
        : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
        : "l"(mc_ptr + (lineoffset + line))
        : "memory");
#else
    asm("multimem.ld_reduce.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
        : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
        : "l"(mc_ptr + (lineoffset + line))
        : "memory");
#endif
    asm volatile(
        "multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(mc_ptr + (lineoffset + line)),
        "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
        : "memory");
  }

  __syncthreads();
  if (threadIdx.x == 0)
    __threadfence_system();
  __syncthreads();

  if (threadIdx.x < RANKS) {
    flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&myptr[targetgpu];
    clock_t s = clock64();
    while (CHECK_IDS(*flag, reduce_id)) {
      if (clock64() - s > 2ull * TIMEOUT) {
        printf("NVONLY AGBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, reduce_id,
               *flag);
        break;
      }
    }
  }
  if (threadIdx.x == 0 && blockIdx.x == 0)
    *reduceidptr = reduce_id;
}  // fp16 inplace reduce kernel (Hopper) MC

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_mc_rs(const int op, const int flagoffset, const int firstrank,
                                           const int myrank, const int gpustep,
                                           const int mylineoffset, const int totallines,
                                           void **commbuff, const int handleridx, float4 *mc_ptr) {
  volatile int *flagptr;
  int physgpu, targetgpu, *myptr;
  int *reduceidptr, reduce_id;
  uint4 *localptr = reinterpret_cast<uint4 *>(commbuff[myrank * gpustep + firstrank + handleridx]);
  int lastSM = 0;

  if (threadIdx.x < RANKS) {
    physgpu = myrank * gpustep + firstrank;
    targetgpu = threadIdx.x * gpustep + firstrank;
    myptr = (reinterpret_cast<int *>(commbuff[physgpu])) + flagoffset;
    reduceidptr = myptr - NVTE_MAX_OPS;  // +op;
    reduce_id = (*reduceidptr) + 1;
    flagptr = (reinterpret_cast<int *>(commbuff[targetgpu])) + flagoffset;
    if (blockIdx.x == 0)
      flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&(myptr[targetgpu]);
    clock_t s = clock64();
    while (CHECK_IDS(*flag, reduce_id)) {
      if (clock64() - s > TIMEOUT) {
        printf("NVONLY RSBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, reduce_id,
               *flag);
        break;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const int adder = blockIdx.x == 0 ? NVTE_MAX_SMS - gridDim.x + 1 : 1;
    int old_val = atomicAdd(myptr + (NVTE_MAX_NVLINK * 2), adder);
    if (old_val + adder == NVTE_MAX_SMS * reduce_id)
      lastSM = 1;
  }
  const int loop_step0 = blockDim.x * gridDim.x;
  const int loop_step = loop_step0 * UNROLL_MC;
  const int start_elem = threadIdx.x + blockDim.x * blockIdx.x;
  const int end_elem = max(start_elem, totallines);
  const int aligned_elem = ((end_elem - start_elem) / loop_step) * loop_step;
  const int end_aligned = start_elem + aligned_elem;

  for (int line = start_elem; line < end_aligned; line += loop_step) {
    uint4 val[UNROLL_MC];
#pragma unroll
    for (int i = 0; i < UNROLL_MC; i++)
#if defined(NVTE_UB_FP16)
      asm("multimem.ld_reduce.global.add.v4.f16x2 {%0,%1,%2,%3}, [%4];"
          : "=r"(val[i].x), "=r"(val[i].y), "=r"(val[i].z), "=r"(val[i].w)
          : "l"(mc_ptr + (mylineoffset + line + i * loop_step0))
          : "memory");
#else
      asm("multimem.ld_reduce.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
          : "=r"(val[i].x), "=r"(val[i].y), "=r"(val[i].z), "=r"(val[i].w)
          : "l"(mc_ptr + (mylineoffset + line + i * loop_step0))
          : "memory");
#endif
#pragma unroll
    for (int i = 0; i < UNROLL_MC; i++)
      localptr[mylineoffset + line + i * loop_step0] = val[i];
  }
  for (int line = end_aligned; line < end_elem; line += loop_step0) {
    uint4 val;
#if defined(NVTE_UB_FP16)
    asm("multimem.ld_reduce.global.add.v4.f16x2 {%0,%1,%2,%3}, [%4];"
        : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
        : "l"(mc_ptr + (mylineoffset + line))
        : "memory");
#else
    asm("multimem.ld_reduce.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
        : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
        : "l"(mc_ptr + (mylineoffset + line))
        : "memory");
#endif
    localptr[mylineoffset + line] = val;
  }

  if (threadIdx.x == 0 && lastSM)
    *reduceidptr = reduce_id;
}  // fp16 inplace reduce-scatter kernel MC

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_mc_rs_oop(const int op, const int flagoffset,
                                               const int firstrank, const int myrank,
                                               const int gpustep, const int mylineoffset,
                                               const int totallines, const int rowlines,
                                               const int skiplines, void **commbuff,
                                               const int handleridx, void *outbuf, float4 *mc_ptr) {
  volatile int *flagptr;
  int physgpu, targetgpu, *myptr;
  int *reduceidptr, reduce_id;
  int lastSM = 0;

  if (threadIdx.x < RANKS) {
    physgpu = myrank * gpustep + firstrank;
    targetgpu = threadIdx.x * gpustep + firstrank;
    myptr = (reinterpret_cast<int *>(commbuff[physgpu])) + flagoffset;
    reduceidptr = myptr - NVTE_MAX_OPS;  // +op;
    reduce_id = (*reduceidptr) + 1;
    flagptr = (reinterpret_cast<int *>(commbuff[targetgpu])) + flagoffset;
    if (blockIdx.x == 0)
      flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&(myptr[targetgpu]);
    clock_t s = clock64();
    while (CHECK_IDS(*flag, reduce_id)) {
      if (clock64() - s > TIMEOUT) {
        printf("[%d] NVONLY RSBAR:SM %d [%d]:expecting %d got %d\n", myrank, blockIdx.x,
               threadIdx.x, reduce_id, *flag);
        break;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const int adder = blockIdx.x == 0 ? NVTE_MAX_SMS - gridDim.x + 1 : 1;
    int old_val = atomicAdd(myptr + (NVTE_MAX_NVLINK * 2), adder);
    if (old_val + adder == NVTE_MAX_SMS * reduce_id)
      lastSM = 1;
  }

  const int loop_step0 = blockDim.x * gridDim.x;
  const int loop_step = loop_step0 * UNROLL_MC;
  const int start_elem = threadIdx.x + blockDim.x * blockIdx.x;
  const int end_elem = max(start_elem, totallines);
  const int aligned_elem = ((end_elem - start_elem) / loop_step) * loop_step;
  const int end_aligned = start_elem + aligned_elem;
  for (int line = start_elem; line < end_aligned; line += loop_step) {
    uint4 val[UNROLL_MC];
#pragma unroll
    for (int i = 0; i < UNROLL_MC; i++)
#if defined(NVTE_UB_FP16)
      asm("multimem.ld_reduce.global.add.v4.f16x2 {%0,%1,%2,%3}, [%4];"
          : "=r"(val[i].x), "=r"(val[i].y), "=r"(val[i].z), "=r"(val[i].w)
          : "l"(mc_ptr + (mylineoffset + line + i * loop_step0))
          : "memory");
#else
      asm("multimem.ld_reduce.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
          : "=r"(val[i].x), "=r"(val[i].y), "=r"(val[i].z), "=r"(val[i].w)
          : "l"(mc_ptr + (mylineoffset + line + i * loop_step0))
          : "memory");
#endif
#pragma unroll
    for (int i = 0; i < UNROLL_MC; i++)
      (reinterpret_cast<uint4 *>(outbuf))[((line + i * loop_step0) / rowlines) * skiplines +
                                          ((line + i * loop_step0) % rowlines)] = val[i];
  }
  for (int line = end_aligned; line < end_elem; line += loop_step0) {
    uint4 val;
#if defined(NVTE_UB_FP16)
    asm("multimem.ld_reduce.global.add.v4.f16x2 {%0,%1,%2,%3}, [%4];"
        : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
        : "l"(mc_ptr + (mylineoffset + line))
        : "memory");
#else
    asm("multimem.ld_reduce.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
        : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
        : "l"(mc_ptr + (mylineoffset + line))
        : "memory");
#endif
    reinterpret_cast<uint4 *> (outbuf)[(line / rowlines) * skiplines + (line % rowlines)] = val;
  }

  if (threadIdx.x == 0 && lastSM)
    *reduceidptr = reduce_id;
}  // fp16 reduce-scatter kernel (out of place) fp16 MC

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_mc_ag(const int op, const int flagoffset, const int firstrank,
                                           const int myrank, const int gpustep,
                                           const int mylineoffset, const int totallines,
                                           void **commbuff, const int handleridx, uint4 *mc_ptr) {
  volatile int *flagptr;
  int physgpu, targetgpu, *myptr;
  int *reduceidptr, reduce_id;
  uint4 *localptr = reinterpret_cast<uint4 *>(commbuff[myrank * gpustep + firstrank + handleridx]);

  if (threadIdx.x < RANKS) {
    physgpu = myrank * gpustep + firstrank;
    targetgpu = threadIdx.x * gpustep + firstrank;
    myptr = (reinterpret_cast<int *>(commbuff[physgpu])) + flagoffset;
    reduceidptr = myptr - NVTE_MAX_OPS;  // +op;
    reduce_id = (*reduceidptr) + 1;
    flagptr = (reinterpret_cast<int *>(commbuff[targetgpu])) + flagoffset;
  }
  __syncthreads();

  const int loop_step0 = blockDim.x * gridDim.x;
  const int loop_step = loop_step0 * UNROLL_MC;
  const int start_elem = threadIdx.x + blockDim.x * blockIdx.x;
  const int end_elem = max(start_elem, totallines);
  const int aligned_elem = ((end_elem - start_elem) / loop_step) * loop_step;
  const int end_aligned = start_elem + aligned_elem;
  for (int line = start_elem; line < end_aligned; line += loop_step) {
    uint4 val[UNROLL_MC];
#pragma unroll
    for (int i = 0; i < UNROLL_MC; i++)
      val[i] = localptr[mylineoffset + line + i * loop_step0];
#pragma unroll
    for (int i = 0; i < UNROLL_MC; i++)
      asm volatile("multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(
                       mc_ptr + (mylineoffset + line + i * loop_step0)),
                   "r"(val[i].x), "r"(val[i].y), "r"(val[i].z), "r"(val[i].w)
                   : "memory");
  }
  for (int line = end_aligned; line < end_elem; line += loop_step0) {
    uint4 val = localptr[mylineoffset + line];
    asm volatile(
        "multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(mc_ptr + (mylineoffset + line)),
        "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
        : "memory");
  }

  __syncthreads();
  if (threadIdx.x == 0)
    __threadfence_system();
  __syncthreads();

  __shared__ int lastSM;
  if (threadIdx.x == 0) {
    const int adder = blockIdx.x == 0 ? NVTE_MAX_SMS - gridDim.x + 1 : 1;
    int old_val = atomicAdd(myptr + (NVTE_MAX_NVLINK * 2), adder);
    if (old_val + adder == NVTE_MAX_SMS * reduce_id)
      lastSM = 1;
    else
      lastSM = 0;
  }
  __syncthreads();
  if (lastSM && threadIdx.x < RANKS) {
    if (threadIdx.x == 0)
      *reduceidptr = reduce_id;
    flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&myptr[targetgpu];
    clock_t s = clock64();
    while (CHECK_IDS(*flag, reduce_id)) {
      if (clock64() - s > 2ull * TIMEOUT) {
        printf("NVONLY AGBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, reduce_id,
               *flag);
        break;
      }
    }
  }
}  // fp16 inplace allgather kernel (Hopper) MC

#else
template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_mc(const int op, const int flagoffset, const int firstrank,
                                        const int myrank, const int gpustep, const int lineoffset,
                                        const int numlines, void **commbuff, const int handleridx,
                                        float4 *mc_ptr) {}
template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS) userbuffers_fp16_sum_inplace_gpu_mc_rs_oop(
    const int op, const int flagoffset, const int firstrank, const int myrank, const int gpustep,
    const int mylineoffset, const int totallines, const int rowlines, const int skiplines,
    void **commbuff, const int handleridx, void *outbuf, float4 *mc_ptr) {}
template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_mc_ag(const int op, const int flagoffset, const int firstrank,
                                           const int myrank, const int gpustep,
                                           const int mylineoffset, const int totallines,
                                           void **commbuff, const int handleridx, uint4 *mc_ptr) {}
template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_mc_rs(const int op, const int flagoffset, const int firstrank,
                                           const int myrank, const int gpustep,
                                           const int mylineoffset, const int totallines,
                                           void **commbuff, const int handleridx, float4 *mc_ptr) {}
#endif

template <int RANKS, typename fp8type>
__global__ void __launch_bounds__(MAX_THREADS) userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_fp8(
    const int op, const int flagoffset, const int firstrank, const int myrank, const int gpustep,
    const int mylineoffset, const int totallines, const int rowlines, const int skiplines,
    void **commbuff, const int handleridx, void *outbuf, float *scale) {
  __shared__ int4 *userptr[RANKS];
  volatile int *flagptr;
  int physgpu, targetgpu, *myptr;
  int *reduceidptr, reduce_id;
  int lastSM = 0;
  half hscale = (half)*scale;

  if (threadIdx.x < RANKS) {
    physgpu = myrank * gpustep + firstrank;
    targetgpu = threadIdx.x * gpustep + firstrank;
    myptr = (reinterpret_cast<int *>(commbuff[physgpu])) + flagoffset;
    reduceidptr = myptr - NVTE_MAX_OPS;  // +op;
    reduce_id = (*reduceidptr) + 1;
    flagptr = (reinterpret_cast<int *>(commbuff[targetgpu])) + flagoffset;
    if (blockIdx.x == 0)
      flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&(myptr[targetgpu]);
    userptr[threadIdx.x] = reinterpret_cast<int4 *>(commbuff[targetgpu + handleridx]);
    clock_t s = clock64();
    while (CHECK_IDS(*flag, reduce_id)) {
      if (clock64() - s > TIMEOUT) {
        printf("[%d] NVONLY RSBAR:SM %d [%d]:expecting %d got %d\n", myrank, blockIdx.x,
               threadIdx.x, reduce_id, *flag);
        break;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const int adder = blockIdx.x == 0 ? NVTE_MAX_SMS - gridDim.x + 1 : 1;
    int old_val = atomicAdd(myptr + (NVTE_MAX_NVLINK * 2), adder);
    if (old_val + adder == NVTE_MAX_SMS * reduce_id)
      lastSM = 1;
  }
  int warp = blockIdx.x + (threadIdx.x >> 5);
  int dest[RANKS];
#pragma unroll
  for (int i = 0; i < RANKS; i++)
    dest[i] = (i + myrank + warp) & (RANKS - 1);

  __syncthreads();
  for (int line = threadIdx.x + blockDim.x * blockIdx.x; line < totallines;
       line += blockDim.x * gridDim.x) {
    int4 val[RANKS];

#pragma unroll
    for (int i = 0; i < RANKS; i++) {
      val[i] = userptr[dest[i]][mylineoffset + line];
    }

    int4 sum[2] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
    half *s = reinterpret_cast<half *>(&sum);

#pragma unroll
    for (int i = 0; i < RANKS; i++) {
      fp8type *x = reinterpret_cast<fp8type *>(&val[i]);
#pragma unroll
      for (int j = 0; j < sizeof(int4) / sizeof(fp8type); j++)
        s[j] += hscale * (half)(x[j]);
    }
    int hline = 2 * line;
    (reinterpret_cast<int4 *>(outbuf))[(hline / rowlines) * skiplines + (hline % rowlines)] =
        sum[0];
    hline++;
    (reinterpret_cast<int4 *>(outbuf))[(hline / rowlines) * skiplines + (hline % rowlines)] =
        sum[1];
  }

  if (threadIdx.x == 0 && lastSM)
    *reduceidptr = reduce_id;
}  // fp16 reduce-scatter kernel (out of place) (fp8->fp16)

template <int RANKS, typename fp8type>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_atomic_fp8(
        const int op, const int flagoffset, const int firstrank, const int myrank,
        const int gpustep, const int mylineoffset, const int totallines, const int rowlines,
        const int skiplines_out, const int skiplines_in, void **commbuff, const int handleridx,
        void *outbuf, float *scale, void *counters, const int numchunks, const int atomicindex) {
  __shared__ int4 *userptr[RANKS];
  volatile int *flagptr;
  int physgpu, targetgpu, *myptr;
  int *reduceidptr, reduce_id;
  int lastSM = 0;
  half hscale = (half)*scale;

  if (threadIdx.x < RANKS) {
    physgpu = myrank * gpustep + firstrank;
    targetgpu = threadIdx.x * gpustep + firstrank;
    // const int blockflagoffset = MAX_NVLINK * 2 * blockIdx.x;
    myptr = (reinterpret_cast<int *>(commbuff[physgpu])) + flagoffset;
    reduceidptr = myptr - NVTE_MAX_OPS;  // +op;
    reduce_id = (*reduceidptr);
    flagptr = (reinterpret_cast<int *>(commbuff[targetgpu])) + flagoffset;  // + blockflagoffset;
  }

  for (int chunk_i = 0; chunk_i < numchunks; chunk_i++) {
    ATOMIC_CONSUMER(chunk_i);

    lastSM = 0;
    if (threadIdx.x < RANKS) {
      reduce_id++;
      if (blockIdx.x == 0)
        flagptr[physgpu] = reduce_id;
      volatile int *flag = (volatile int *)&(myptr[targetgpu]);
      userptr[threadIdx.x] = reinterpret_cast<int4 *>(commbuff[targetgpu + handleridx]);
      clock_t s = clock64();
      while (CHECK_IDS(*flag, reduce_id)) {
        if (clock64() - s > TIMEOUT) {
          printf("[%d] NVONLY RSBAR:SM %d [%d]:expecting %d got %d\n", myrank, blockIdx.x,
                 threadIdx.x, reduce_id, *flag);
          break;
        }
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      const int adder = blockIdx.x == 0 ? NVTE_MAX_SMS - gridDim.x + 1 : 1;
      int old_val = atomicAdd(myptr + (NVTE_MAX_NVLINK * 2), /*numchunks * */ adder);
      if (old_val + adder == NVTE_MAX_SMS * (reduce_id /* + numchunks*/))
        lastSM = 1;
    }

    int warp = blockIdx.x + (threadIdx.x >> 5);
    int dest[RANKS];
#pragma unroll
    for (int i = 0; i < RANKS; i++)
      dest[i] = (i + myrank + warp) & (RANKS - 1);

    __syncthreads();
    for (int line = threadIdx.x + blockDim.x * blockIdx.x; line < totallines;
         line += blockDim.x * gridDim.x) {
      int4 val[RANKS];
      const int rowlines_in = rowlines / 2;
      const int index_in = skiplines_in == 0
                               ? mylineoffset + myrank * totallines + line
                               : (numchunks <= 1 ? 1 : chunk_i) * mylineoffset +
                                     myrank * (totallines * skiplines_in / rowlines_in) +
                                     (line / rowlines_in) * skiplines_in + (line % rowlines_in);
      const int index1_out = chunk_i * mylineoffset * 2 + ((2 * line) / rowlines) * skiplines_out +
                             ((2 * line) % rowlines);
      const int index2_out = chunk_i * mylineoffset * 2 +
                             ((2 * line + 1) / rowlines) * skiplines_out +
                             ((2 * line + 1) % rowlines);

#pragma unroll
      for (int i = 0; i < RANKS; i++) {
        val[i] = userptr[dest[i]][index_in];
      }

      int4 sum[2] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
      half *s = reinterpret_cast<half *>(&sum);

#pragma unroll
      for (int i = 0; i < RANKS; i++) {
        fp8type *x = reinterpret_cast<fp8type *>(&val[i]);
#pragma unroll
        for (int j = 0; j < sizeof(int4) / sizeof(fp8type); j++)
          s[j] += hscale * (half)(x[j]);
      }
      (reinterpret_cast<int4 *>(outbuf))[index1_out] = sum[0];
      (reinterpret_cast<int4 *>(outbuf))[index2_out] = sum[1];
    }
  }
  if (threadIdx.x == 0 && lastSM)
    *reduceidptr = reduce_id;
}  // fp16 reduce-scatter kernel (out of place) (fp8->fp16)

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_stride(const int op, const int flagoffset,
                                                      const int firstrank, const int myrank,
                                                      const int gpustep, const int mylineoffset,
                                                      const int totallines, const int rowlines,
                                                      const int skiplines, void **commbuff,
                                                      const int handleridx, void *outbuf) {
  __shared__ int4 *userptr[RANKS];
  volatile int *flagptr;
  int physgpu, targetgpu, *myptr;
  int *reduceidptr, reduce_id;
  int lastSM = 0;

  if (threadIdx.x < RANKS) {
    physgpu = myrank * gpustep + firstrank;
    targetgpu = threadIdx.x * gpustep + firstrank;
    myptr = (reinterpret_cast<int *>(commbuff[physgpu])) + flagoffset;
    reduceidptr = myptr - NVTE_MAX_OPS;  // +op;
    reduce_id = (*reduceidptr) + 1;
    flagptr = (reinterpret_cast<int *>(commbuff[targetgpu])) + flagoffset;
    if (blockIdx.x == 0)
      flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&(myptr[targetgpu]);
    userptr[threadIdx.x] = reinterpret_cast<int4 *>(commbuff[targetgpu + handleridx]);
    clock_t s = clock64();
    while (CHECK_IDS(*flag, reduce_id)) {
      if (clock64() - s > TIMEOUT) {
        printf("[%d] NVONLY RSBAR:SM %d [%d]:expecting %d got %d\n", myrank, blockIdx.x,
               threadIdx.x, reduce_id, *flag);
        break;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const int adder = blockIdx.x == 0 ? NVTE_MAX_SMS - gridDim.x + 1 : 1;
    int old_val = atomicAdd(myptr + (NVTE_MAX_NVLINK * 2), adder);
    if (old_val + adder == NVTE_MAX_SMS * reduce_id)
      lastSM = 1;
  }

  int warp = blockIdx.x + (threadIdx.x >> 5);
  int dest[RANKS];
#pragma unroll
  for (int i = 0; i < RANKS; i++)
    dest[i] = (i + myrank + warp) & (RANKS - 1);

  for (int line = threadIdx.x + blockDim.x * blockIdx.x; line < totallines;
       line += blockDim.x * gridDim.x) {
    int4 val[RANKS];
    int index_in = mylineoffset + myrank * (totallines * skiplines / rowlines) +
                   (line / rowlines) * skiplines + (line % rowlines);

#pragma unroll
    for (int i = 0; i < RANKS; i++) {
      val[i] = userptr[dest[i]][index_in];
    }

    int4 sum = val[0];
    half *s = reinterpret_cast<half *>(&sum);

#pragma unroll
    for (int i = 1; i < RANKS; i++) {
      half *x = reinterpret_cast<half *>(&val[i]);
#pragma unroll
      for (int j = 0; j < 8; j++)
        s[j] += x[j];
    }

    int index_out = (line / rowlines) * skiplines + (line % rowlines);
    (reinterpret_cast<int4 *>(outbuf))[index_out] = sum;
  }

  if (threadIdx.x == 0 && lastSM)
    *reduceidptr = reduce_id;
}  // fp16 reduce-scatter kernel (out of place) fp16

#if 0
template<int RANKS, typename fp8type>
__global__ void
__launch_bounds__(MAX_THREADS)
userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_stride_atomic_fp8(
  const int op, const int flagoffset, const int firstrank, const int myrank, const int gpustep,
  const int mylineoffset, const int totallines, const int rowlines, const int skiplines,
  const int numchunks, void **commbuff, const int handleridx, void* outbuf, void *counters,
  float* scale) {
  if (counters) {
      if ( threadIdx.x == 0 ) {
          // spin-lock on counter from producer
          int old_val;
          while (0 != (old_val = atomicCAS(((unsigned int*)counters), 0, 0) )) {}

          // make sure all threadblocks have read/waited on counters.
          int old_val2;
          atomicInc(((unsigned int *)counters)+numchunks, gridDim.x-1);
          while (0 != (old_val2 = atomicCAS(((unsigned int*)counters)+numchunks, 0, 0) )) {}

          // reset counter for next producer.
          ((unsigned int*)counters)[0] = 1;
          asm volatile ("fence.sc.gpu;\n");
      }
  }
  __syncthreads();

  __shared__ int4* userptr[RANKS];
  volatile int *flagptr;
  int physgpu, targetgpu, *myptr;
  int *reduceidptr, reduce_id;
  int lastSM = 0;
  half hscale = (half) *scale;

  if (threadIdx.x < RANKS) {
    physgpu = myrank*gpustep+firstrank;
    targetgpu = threadIdx.x*gpustep+firstrank;
    myptr = (reinterpret_cast<int*>(commbuff[physgpu])) + flagoffset;
    reduceidptr = myptr-NVTE_MAX_OPS;  // +op;
    reduce_id  =(*reduceidptr)+1;
    flagptr = (reinterpret_cast<int *>(commbuff[targetgpu])) + flagoffset;
    if (blockIdx.x == 0) flagptr[physgpu] = reduce_id;
    volatile int* flag = (volatile int*)&(myptr[targetgpu]);
    userptr[threadIdx.x] = reinterpret_cast<int4*>(commbuff[targetgpu+handleridx]);
    clock_t s = clock64();
    while (CHECK_IDS(*flag, reduce_id)) {
      if (clock64()-s > TIMEOUT) {
        printf("[%d] NVONLY RSBAR:SM %d [%d]:expecting %d got %d\n",
                myrank, blockIdx.x, threadIdx.x, reduce_id, *flag);
        break;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const int adder = blockIdx.x == 0 ? NVTE_MAX_SMS-gridDim.x+1 : 1;
    int old_val = atomicAdd(myptr+(NVTE_MAX_NVLINK*2), adder);
    if (old_val+adder == NVTE_MAX_SMS*reduce_id) lastSM = 1;
  }


  int warp = blockIdx.x+(threadIdx.x>>5);
  int dest[RANKS];
#pragma unroll
  for (int i = 0; i < RANKS; i++)
    dest[i] = (i+myrank+warp)&(RANKS-1);

       for (int line = threadIdx.x+blockDim.x*blockIdx.x;
            line < totallines; line+=blockDim.x*gridDim.x) {
        int4 val[RANKS];
        int index_in = mylineoffset + myrank*(totallines*skiplines/rowlines/2) +
                       (line/rowlines)*skiplines/2+(line%rowlines);

#pragma unroll
        for (int i = 0; i < RANKS; i++) {
           val[i] = userptr[dest[i]][index_in];
        }

        int4 sum[2] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
        half *s = reinterpret_cast<half*>(&sum);

#pragma unroll
        for (int i = 0; i < RANKS; i++) {
          fp8type *x = reinterpret_cast<fp8type*>(&val[i]);
#pragma unroll
          for (int j=0; j < sizeof(int4)/sizeof(fp8type); j++) s[j] += hscale * (half)(x[j]);
        }
        int hline = 2*line;
        int index_out1 = (hline/rowlines)*skiplines+(hline%rowlines);
        (reinterpret_cast<int4*>(outbuf))[index_out1] = sum[0];
        hline++;
        int index_out2 = (hline/rowlines)*skiplines+(hline%rowlines);
        (reinterpret_cast<int4*>(outbuf))[index_out2] = sum[1];
      }

  if (threadIdx.x == 0 && lastSM) *reduceidptr = reduce_id;
}  // fp16 reduce-scatter kernel (out of place) fp16
#endif

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_stride_atomic(
        const int op, const int flagoffset, const int firstrank, const int myrank,
        const int gpustep, const int mylineoffset, const int totallines, const int rowlines,
        const int skiplines, const int numchunks, void **commbuff, const int handleridx,
        void *outbuf, void *counters) {
  if (counters) {
    if (threadIdx.x == 0) {
      // spin-lock on counter from producer
      int old_val;
      while (0 != (old_val = atomicCAS(((unsigned int *)counters), 0, 0))) {
      }

      // make sure all threadblocks have read/waited on counters.
      int old_val2;
      atomicInc(((unsigned int *)counters) + numchunks, gridDim.x - 1);
      while (0 != (old_val2 = atomicCAS(((unsigned int *)counters) + numchunks, 0, 0))) {
      }

      // reset counter for next producer.
      ((unsigned int *)counters)[0] = 1;
      asm volatile("fence.sc.gpu;\n");
    }
  }
  __syncthreads();

  __shared__ int4 *userptr[RANKS];
  volatile int *flagptr;
  int physgpu, targetgpu, *myptr;
  int *reduceidptr, reduce_id;
  int lastSM = 0;

  if (threadIdx.x < RANKS) {
    physgpu = myrank * gpustep + firstrank;
    targetgpu = threadIdx.x * gpustep + firstrank;
    myptr = (reinterpret_cast<int *>(commbuff[physgpu])) + flagoffset;
    reduceidptr = myptr - NVTE_MAX_OPS;  // +op;
    reduce_id = (*reduceidptr) + 1;
    flagptr = (reinterpret_cast<int *>(commbuff[targetgpu])) + flagoffset;
    if (blockIdx.x == 0)
      flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&(myptr[targetgpu]);
    userptr[threadIdx.x] = reinterpret_cast<int4 *>(commbuff[targetgpu + handleridx]);
    clock_t s = clock64();
    while (CHECK_IDS(*flag, reduce_id)) {
      if (clock64() - s > TIMEOUT) {
        printf("[%d] NVONLY RSBAR:SM %d [%d]:expecting %d got %d\n", myrank, blockIdx.x,
               threadIdx.x, reduce_id, *flag);
        break;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const int adder = blockIdx.x == 0 ? NVTE_MAX_SMS - gridDim.x + 1 : 1;
    int old_val = atomicAdd(myptr + (NVTE_MAX_NVLINK * 2), adder);
    if (old_val + adder == NVTE_MAX_SMS * reduce_id)
      lastSM = 1;
  }

  int warp = blockIdx.x + (threadIdx.x >> 5);
  int dest[RANKS];
#pragma unroll
  for (int i = 0; i < RANKS; i++)
    dest[i] = (i + myrank + warp) & (RANKS - 1);

  for (int line = threadIdx.x + blockDim.x * blockIdx.x; line < totallines;
       line += blockDim.x * gridDim.x) {
    int4 val[RANKS];
    int index_in = mylineoffset + myrank * (totallines * skiplines / rowlines) +
                   (line / rowlines) * skiplines + (line % rowlines);

#pragma unroll
    for (int i = 0; i < RANKS; i++) {
      val[i] = userptr[dest[i]][index_in];
    }

    int4 sum = val[0];
    half *s = reinterpret_cast<half *>(&sum);

#pragma unroll
    for (int i = 1; i < RANKS; i++) {
      half *x = reinterpret_cast<half *>(&val[i]);
#pragma unroll
      for (int j = 0; j < 8; j++)
        s[j] += x[j];
    }

    int index_out = (line / rowlines) * skiplines + (line % rowlines);
    (reinterpret_cast<int4 *>(outbuf))[index_out] = sum;
  }

  if (threadIdx.x == 0 && lastSM)
    *reduceidptr = reduce_id;
}  // fp16 reduce-scatter kernel (out of place) fp16

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_stride_multiatomic(
        const int op, const int flagoffset, const int firstrank, const int myrank,
        const int gpustep, const int mylineoffset, const int totallines, const int rowlines,
        const int skiplines, const int numchunks, void **commbuff, const int handleridx,
        void *outbuf, void *counters) {
  for (int chunk_i = 0; chunk_i < numchunks; chunk_i++) {
    if (counters) {
      if (threadIdx.x == 0) {
        // spin-lock on counter from producer
        int old_val;
        while (0 != (old_val = atomicCAS(((unsigned int *)counters) + chunk_i, 0, 0))) {
        }

        // make sure all threadblocks have read/waited on counters.
        int old_val2;
        atomicInc(((unsigned int *)counters) + numchunks + chunk_i, gridDim.x - 1);
        while (0 !=
               (old_val2 = atomicCAS(((unsigned int *)counters) + numchunks + chunk_i, 0, 0))) {
        }

        // reset counter for next producer.
        ((unsigned int *)counters)[chunk_i] = 1;
        asm volatile("fence.sc.gpu;\n");
      }
    }
    __syncthreads();

    __shared__ int4 *userptr[RANKS];
    volatile int *flagptr;
    int physgpu, targetgpu, *myptr;
    int *reduceidptr, reduce_id;
    int lastSM = 0;

    if (threadIdx.x < RANKS) {
      physgpu = myrank * gpustep + firstrank;
      targetgpu = threadIdx.x * gpustep + firstrank;
      myptr = (reinterpret_cast<int *>(commbuff[physgpu])) + flagoffset;
      reduceidptr = myptr - NVTE_MAX_OPS;  // +op;
      reduce_id = (*reduceidptr) + 1;
      flagptr = (reinterpret_cast<int *>(commbuff[targetgpu])) + flagoffset;
      if (blockIdx.x == 0)
        flagptr[physgpu] = reduce_id;
      volatile int *flag = (volatile int *)&(myptr[targetgpu]);
      userptr[threadIdx.x] = reinterpret_cast<int4 *>(commbuff[targetgpu + handleridx]);
      clock_t s = clock64();
      while (CHECK_IDS(*flag, reduce_id)) {
        if (clock64() - s > TIMEOUT) {
          printf("[%d] NVONLY RSBAR:SM %d [%d]:expecting %d got %d\n", myrank, blockIdx.x,
                 threadIdx.x, reduce_id, *flag);
          break;
        }
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      const int adder = blockIdx.x == 0 ? NVTE_MAX_SMS - gridDim.x + 1 : 1;
      int old_val = atomicAdd(myptr + (NVTE_MAX_NVLINK * 2), adder);
      if (old_val + adder == NVTE_MAX_SMS * reduce_id)
        lastSM = 1;
    }

    int warp = blockIdx.x + (threadIdx.x >> 5);
    int dest[RANKS];
#pragma unroll
    for (int i = 0; i < RANKS; i++)
      dest[i] = (i + myrank + warp) & (RANKS - 1);

    for (int line = threadIdx.x + blockDim.x * blockIdx.x; line < totallines;
         line += blockDim.x * gridDim.x) {
      int4 val[RANKS];
      int index_in = chunk_i * mylineoffset + myrank * (totallines * skiplines / rowlines) +
                     (line / rowlines) * skiplines + (line % rowlines);

#pragma unroll
      for (int i = 0; i < RANKS; i++) {
        val[i] = userptr[dest[i]][index_in];
      }

      int4 sum = val[0];
      half *s = reinterpret_cast<half *>(&sum);

#pragma unroll
      for (int i = 1; i < RANKS; i++) {
        half *x = reinterpret_cast<half *>(&val[i]);
#pragma unroll
        for (int j = 0; j < 8; j++)
          s[j] += x[j];
      }

      int index_out = chunk_i * mylineoffset + (line / rowlines) * skiplines + (line % rowlines);
      (reinterpret_cast<int4 *>(outbuf))[index_out] = sum;
    }
    if (threadIdx.x == 0 && lastSM)
      *reduceidptr = reduce_id;
  }
}  // fp16 reduce-scatter kernel (out of place) fp16

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_rr_ag(const int op, const int flagoffset, const int firstrank,
                                           const int myrank, const int gpustep,
                                           const int mylineoffset, const int totallines,
                                           void **commbuff, const int handleridx) {
  __shared__ int4 *userptr[RANKS];
  volatile int *flagptr;
  int physgpu, targetgpu, *myptr;
  int *reduceidptr, reduce_id;
  if (threadIdx.x < RANKS) {
    physgpu = myrank * gpustep + firstrank;
    targetgpu = threadIdx.x * gpustep + firstrank;
    myptr = (reinterpret_cast<int *>(commbuff[physgpu])) + flagoffset;
    reduceidptr = myptr - NVTE_MAX_OPS;  // +op;
    reduce_id = (*reduceidptr) + 1;
    flagptr = (reinterpret_cast<int *>(commbuff[targetgpu])) + flagoffset;
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
      val[i] = userptr[dest[i]][mylineoffset + line + totallines * dest[i]];
    }

#pragma unroll
    for (int i = 0; i < RANKS - 1; i++) {
      userptr[myrank][mylineoffset + line + totallines * dest[i]] = val[i];
    }
  }
  __shared__ int lastSM;
  if (threadIdx.x == 0) {
    const int adder = blockIdx.x == 0 ? NVTE_MAX_SMS - gridDim.x + 1 : 1;
    int old_val = atomicAdd(myptr + (NVTE_MAX_NVLINK * 2), adder);
    if (old_val + adder == NVTE_MAX_SMS * reduce_id)
      lastSM = 1;
    else
      lastSM = 0;
  }
  __syncthreads();
  if (lastSM && threadIdx.x < RANKS) {
    if (threadIdx.x == 0)
      *reduceidptr = reduce_id;
    flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&myptr[targetgpu];
    clock_t s = clock64();
    while (CHECK_IDS(*flag, reduce_id)) {
      if (clock64() - s > 2ull * TIMEOUT) {
        printf("NVONLY AGBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, reduce_id,
               *flag);
        break;
      }
    }
  }
}  // fp16 inplace reduce kernel (Ampere)

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_rw_ag(const int op, const int flagoffset, const int firstrank,
                                           const int myrank, const int gpustep,
                                           const int mylineoffset, const int totallines,
                                           void **commbuff, const int handleridx) {
  __shared__ int4 *userptr[RANKS];
  volatile int *flagptr;
  int physgpu, targetgpu, *myptr;
  int *reduceidptr, reduce_id;
  int4 *localptr;
  if (threadIdx.x < RANKS) {
    physgpu = myrank * gpustep + firstrank;
    targetgpu = threadIdx.x * gpustep + firstrank;
    myptr = (reinterpret_cast<int *>(commbuff[physgpu])) + flagoffset;
    reduceidptr = myptr - NVTE_MAX_OPS;  // +op;
    reduce_id = (*reduceidptr) + 1;
    flagptr = (reinterpret_cast<int *>(commbuff[targetgpu])) + flagoffset;
    userptr[threadIdx.x] = reinterpret_cast<int4 *>(commbuff[targetgpu + handleridx]);
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
    for (int j = 0; j < UNROLLAG; j++)
      val[j] = localptr[mylineoffset + line + loop_step0 * j];

#pragma unroll
    for (int j = 0; j < UNROLLAG; j++)
#pragma unroll
      for (int i = 0; i < RANKS - 1; i++) {
        userptr[dest[i]][mylineoffset + line + j * loop_step0] = val[j];
      }
  }

  for (int line = end_aligned; line < end_elem; line += loop_step0) {
    int4 sum = localptr[mylineoffset + line];
#pragma unroll
    for (int i = 0; i < RANKS - 1; i++) {
      userptr[dest[i]][mylineoffset + line] = sum;
    }
  }

  __syncthreads();
  if (threadIdx.x == 0)
    __threadfence_system();
  __syncthreads();

  __shared__ int lastSM;
  if (threadIdx.x == 0) {
    const int adder = blockIdx.x == 0 ? NVTE_MAX_SMS - gridDim.x + 1 : 1;
    int old_val = atomicAdd(myptr + (NVTE_MAX_NVLINK * 2), adder);
    if (old_val + adder == NVTE_MAX_SMS * reduce_id)
      lastSM = 1;
    else
      lastSM = 0;
  }
  __syncthreads();
  if (lastSM && threadIdx.x < RANKS) {
    if (threadIdx.x == 0)
      *reduceidptr = reduce_id;
    flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&myptr[targetgpu];
    clock_t s = clock64();
    while (CHECK_IDS(*flag, reduce_id)) {
      if (clock64() - s > 2ull * TIMEOUT) {
        printf("NVONLY AGBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, reduce_id,
               *flag);
        break;
      }
    }
  }
}  // fp16 inplace allgather kernel (Volta,Hopper)

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_rr_blocked(const int op, const int flagoffset,
                                                const int firstrank, const int myrank,
                                                const int lineoffset, const int numlines,
                                                void **commbuff, const int handleridx,
                                                const int peerblocklines, int *hostflags,
                                                int *gpuflag, const int numblocks) {
  const int basecounter = gpuflag[NVTE_GF_STATE + op];

#define REDUCETHREADS (blockDim.x - 32)

  if (threadIdx.x < 32) {
    int *flagptr;
    if (threadIdx.x < RANKS) {
      if (!blockIdx.x) {
        flagptr = reinterpret_cast<int *>(commbuff[threadIdx.x + firstrank]);
        flagptr[flagoffset + myrank + firstrank] = basecounter;
      }
      volatile int *flag = (volatile int *)&((reinterpret_cast<int *>(
          commbuff[myrank + firstrank]))[flagoffset + threadIdx.x + firstrank]);
      while (CHECK_IDS(*flag, basecounter)) {
      }
    }
    __syncthreads();

    int startblock = 0, endblock = numblocks;

    for (int nblock = 0; nblock < endblock; nblock++) {
      asm volatile("bar.sync 13, %0;" ::"r"(REDUCETHREADS + 32));

      if (threadIdx.x == 0) {
        __threadfence();
        if (blockIdx.x)
          gpuflag[op * NVTE_MAX_SMS * 2 + blockIdx.x] = nblock + basecounter + 1;
      } else if (blockIdx.x == 0) {
        int expecting = (basecounter + nblock + 1);
        if (threadIdx.x < gridDim.x)
          while (((volatile int *)gpuflag)[op * NVTE_MAX_SMS * 2 + threadIdx.x] < expecting) {
          }
      }
      if (!blockIdx.x) {
        asm volatile("bar.sync 15, %0;" ::"r"(32));
        if (!threadIdx.x)
          hostflags[0] = nblock + basecounter + 1;
      }
    }

    int cachedflag = basecounter;

#define ALLGATHERFLAG NVTE_GF_IBSHARPDONE

    if (blockIdx.x == 0 && threadIdx.x < RANKS) {
      while (cachedflag < basecounter + numblocks) {
        int newflag = ((volatile int *)gpuflag)[ALLGATHERFLAG];
        if (newflag == cachedflag)
          continue;
        cachedflag = newflag;
        flagptr[flagoffset + myrank + 32 + firstrank] = cachedflag;
      }
    }

    if (blockIdx.x == 0 && threadIdx.x == 0)
      gpuflag[NVTE_GF_STATE + op] = basecounter + numblocks;
  } else {
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
          for (int j = 0; j < sizeof(int4) / sizeof(half); j++)
            s[j] += x[j];
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
    volatile int *flag = (volatile int *)&((reinterpret_cast<int *>(
        commbuff[myrank + firstrank]))[flagoffset + mydest + 32 + firstrank]);

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
          while (CHECK_IDS(*flag, gathercounter)) {
          }
          gathercounter++;
        }

        asm volatile("bar.sync %0, %1;" ::"r"(1 + mydest), "r"(myblockDim));

        for (int line = start_elem; line < end_aligned; line += myblockDim * gridDim.x * UNROLL) {
          int4 val[UNROLL];
#pragma unroll
          for (int i = 0; i < UNROLL; i++)
            val[i] = peerptr[line + i * myblockDim * gridDim.x];
#pragma unroll
          for (int i = 0; i < UNROLL; i++)
            myptr[line + i * myblockDim * gridDim.x] = val[i];
        }
        for (int line = end_aligned; line < end_elem; line += myblockDim * gridDim.x)
          myptr[line] = peerptr[line];
      }
      blocklineoffset += peerblocklines * RANKS;
    }  // block loop for NVLINK-ALLGATHER
  }    // worker warps else block
}  // fp16 inplace reduce kernel with SHARP / in blocks

// threadfence and SMs sync to SM0
#define SMBAR(offset, block)                                                                       \
  asm volatile("bar.sync 13, %0;" ::"r"(blockDim.x));                                              \
  if (threadIdx.x == 0) {                                                                          \
    __threadfence_system();                                                                        \
    if (blockIdx.x)                                                                                \
      gpuflag[offset + blockIdx.x] = block + basecounter + 1;                                      \
  } else if (blockIdx.x == 0) {                                                                    \
    int expecting = (basecounter + block + 1);                                                     \
    if (threadIdx.x < gridDim.x)                                                                   \
      while (((volatile int *)gpuflag)[offset + threadIdx.x] < expecting) {                        \
      }                                                                                            \
  }                                                                                                \
  if (blockIdx.x == 0)                                                                             \
    asm volatile("bar.sync 15, %0;" ::"r"(32));

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS) userbuffers_fp16_sum_inplace_gpu_rr_blocked2(
    const int op, const int maxcredit, const int headstart, const int myibrank, const int ibranks,
    const int commbufoffset, const int flagoffset, const int firstrank, const int myrank,
    const int gpustep, const int lineoffset, const int numlines, void **commbuff,
    const int handleridx, const int peerblocklines, int *hostflags, int *gpuflag,
    const int numblocks) {
  const int basecounter = gpuflag[NVTE_GF_STATE + op];
  if (threadIdx.x < 32) {
    int *flagptr;
    volatile int *localflag = (volatile int *)&(
        ((int *)commbuff[gpustep * myrank + firstrank])[flagoffset]);  // NOLINT(*)
    // initial intranode barrier - once
    if (threadIdx.x < RANKS) {
      if (!blockIdx.x) {
        flagptr = reinterpret_cast<int *>(commbuff[gpustep * threadIdx.x + firstrank]);
        flagptr[flagoffset + gpustep * myrank + firstrank] = basecounter;
      }
      volatile int *flag = &localflag[gpustep * threadIdx.x + firstrank];
      while (CHECK_IDS(*flag, basecounter)) {
      }
    }
    __syncthreads();

    for (int nblock = 0; nblock < numblocks + headstart; nblock++) {
      if (nblock < numblocks) {
        // RS happens here
        SMBAR(op * 2 * NVTE_MAX_SMS, nblock);
        if (!blockIdx.x && !threadIdx.x)
          hostflags[NVTE_HF_NVRSDONE + (op & 1)] = nblock + basecounter + 1;
      }

      if (nblock >= headstart) {
        for (int ibflag = threadIdx.x; ibflag < ibranks; ibflag += 32)
          if (ibflag != myibrank)
            while (localflag[NVTE_REG0_IBRS + ibflag] < basecounter + nblock - headstart + 1) {
            }
        asm volatile("bar.sync 13, %0;" ::"r"(blockDim.x));
        // REDUCE happens here
        SMBAR(op * 2 * NVTE_MAX_SMS + NVTE_MAX_SMS, nblock - headstart);
        if (!blockIdx.x && !threadIdx.x)
          hostflags[NVTE_HF_NVREDUCEDONE + (op & 1)] = nblock + basecounter + 1 - headstart;
      }
    }
    // final part doing NVAG based on responses from NIC-RMW:IBAG

    if (blockIdx.x == 0) {
      for (int nblock = 0; nblock < numblocks; nblock++) {
        const int expected = basecounter + nblock + 1;
        for (int ibflag = threadIdx.x; ibflag < ibranks; ibflag += 32)
          if (ibflag != myibrank)
            while (localflag[NVTE_REG0_IBAG + ibflag] < expected) {
            }
        asm volatile("bar.sync 15, %0;" ::"r"(32));
        if (threadIdx.x < RANKS)
          flagptr[flagoffset + gpustep * myrank + NVTE_MAX_NVLINK + firstrank] = expected;
      }
    }

    if (blockIdx.x == 0 && threadIdx.x == 0)
      gpuflag[NVTE_GF_STATE + op] = basecounter + numblocks;
  } else {  // sync warp
    // reducethreads
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
        if (RANKS > 1) {
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
              for (int j = 0; j < sizeof(int4) / sizeof(half); j++)
                s[j] += x[j];
            }

            userptrmyrank[blockstart + line] = sum;
          }  // single block loop
        }

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
            for (int j = 0; j < 16; j++)
              s[j] += x[j];
          }
#pragma unroll
          for (int i = 1; i < UNROLLRS; i++) {
            half *x = reinterpret_cast<half *>(&val[i]);
#pragma unroll
            for (int j = 0; j < 16; j++)
              s[j] += x[j];
          }
          userptrmyrank[tempstart + line] = sum;
        }

        asm volatile("bar.sync 13, %0;" ::"r"(REDUCETHREADS + 32));
      }
    }  // nblock loop NVLINK-REDUCESCATTER + IBREDUCE LOCAL COMPUTE

    if (RANKS != 1) {
      const int nwarps = (REDUCETHREADS >> 5) / (RANKS - 1);
      const int myblockDim = nwarps << 5;
      const int mywarp = ((threadIdx.x - 32) >> 5) / (RANKS - 1);
      const int maxthreadIdx = myblockDim * (RANKS - 1) + 32;
      const int mydest = (myrank + 1 + ((threadIdx.x - 32) >> 5) % (RANKS - 1)) & (RANKS - 1);
      const int mythreadIdx = (mywarp << 5) + (threadIdx.x & 31);
      volatile int *flag = (volatile int *)&((reinterpret_cast<int *>(
          commbuff[gpustep * myrank + firstrank]))[flagoffset + gpustep * mydest + NVTE_MAX_NVLINK +
                                                   firstrank]);

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
            while (CHECK_IDS(*flag, gathercounter)) {
            }
            gathercounter++;
          }

          asm volatile("bar.sync %0, %1;" ::"r"(1 + mydest), "r"(myblockDim));

          for (int line = start_elem; line < end_aligned; line += myblockDim * gridDim.x * UNROLL) {
            int4 val[UNROLL];
#pragma unroll
            for (int i = 0; i < UNROLL; i++)
              val[i] = peerptr[line + i * myblockDim * gridDim.x];
#pragma unroll
            for (int i = 0; i < UNROLL; i++)
              myptr[line + i * myblockDim * gridDim.x] = val[i];
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
  const int basecounter = gpuflag[NVTE_GF_STATE + op];
  if (threadIdx.x < 32) {
    int *flagptr;
    volatile int *localflag = (volatile int *)&(
        ((int *)commbuff[gpustep * myrank + firstrank])[flagoffset]);  // NOLINT(*)
    // initial intranode barrier - once
    if (threadIdx.x < RANKS) {
      if (!blockIdx.x) {
        flagptr = reinterpret_cast<int *>(commbuff[gpustep * threadIdx.x + firstrank]);
        flagptr[flagoffset + gpustep * myrank + firstrank] = basecounter;
      }
      volatile int *flag = &localflag[gpustep * threadIdx.x + firstrank];
      while (CHECK_IDS(*flag, basecounter)) {
      }
    }
    __syncthreads();

    for (int nblock = 0; nblock < numblocks + headstart; nblock++) {
      if (nblock < numblocks) {
        // RS happens here
        SMBAR(op * 2 * NVTE_MAX_SMS, nblock);
        if (!blockIdx.x && !threadIdx.x)
          hostflags[NVTE_HF_NVRSDONE + (op & 1)] = nblock + basecounter + 1;
      }

      if (nblock >= headstart) {
        for (int ibflag = threadIdx.x; ibflag < ibranks; ibflag += 32)
          if (ibflag != myibrank)
            while (localflag[NVTE_REG0_IBRS + ibflag] < basecounter + nblock - headstart + 1) {
            }
        asm volatile("bar.sync 13, %0;" ::"r"(blockDim.x));
        // REDUCE happens here
        SMBAR(op * 2 * NVTE_MAX_SMS + NVTE_MAX_SMS, nblock - headstart);
      }
    }
  } else {  // sync warp
    // reducethreads
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
        if (RANKS > 1) {
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
              for (int j = 0; j < sizeof(int4) / sizeof(half); j++)
                s[j] += x[j];
            }

            userptrmyrank[blockstart + line] = sum;
          }  // single block loop
        }

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
            for (int j = 0; j < 16; j++)
              s[j] += x[j];
          }
#pragma unroll
          for (int i = 1; i < UNROLLRS; i++) {
            half *x = reinterpret_cast<half *>(&val[i]);
#pragma unroll
            for (int j = 0; j < 16; j++)
              s[j] += x[j];
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
  const int basecounter = gpuflag[NVTE_GF_STATE + op];
  if (threadIdx.x < 32) {
    int *flagptr;
    volatile int *localflag = (volatile int *)&(
        ((int *)commbuff[gpustep * myrank + firstrank])[flagoffset]);  // NOLINT(*)
    if (threadIdx.x < RANKS) {
      if (!blockIdx.x) {
        flagptr = reinterpret_cast<int *>(commbuff[gpustep * threadIdx.x + firstrank]);
      }
    }
    __syncthreads();
    if (!blockIdx.x && !threadIdx.x)
      hostflags[NVTE_HF_NVREDUCEDONE + (op & 1)] = numblocks + basecounter;
    // tell CPU proxy all blocks are done and ready for NVAG

    // final part doing NVAG based on responses from NIC-RMW:IBAG

    if (blockIdx.x == 0) {
      for (int nblock = 0; nblock < numblocks; nblock++) {
        const int expected = basecounter + nblock + 1;
        for (int ibflag = threadIdx.x; ibflag < ibranks; ibflag += 32)
          if (ibflag != myibrank)
            while (localflag[NVTE_REG0_IBAG + ibflag] < expected) {
            }
        asm volatile("bar.sync 15, %0;" ::"r"(32));
        if (threadIdx.x < RANKS)
          flagptr[flagoffset + gpustep * myrank + NVTE_MAX_NVLINK + firstrank] = expected;
      }
    }

    if (blockIdx.x == 0 && threadIdx.x == 0)
      gpuflag[NVTE_GF_STATE + op] = basecounter + numblocks;
  } else {  // sync warp
    // reducethreads
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

    if (RANKS != 1) {
      const int nwarps = (REDUCETHREADS >> 5) / (RANKS - 1);
      const int myblockDim = nwarps << 5;
      const int mywarp = ((threadIdx.x - 32) >> 5) / (RANKS - 1);
      const int maxthreadIdx = myblockDim * (RANKS - 1) + 32;
      const int mydest = (myrank + 1 + ((threadIdx.x - 32) >> 5) % (RANKS - 1)) & (RANKS - 1);
      const int mythreadIdx = (mywarp << 5) + (threadIdx.x & 31);
      volatile int *flag = (volatile int *)&((reinterpret_cast<int *>(
          commbuff[gpustep * myrank + firstrank]))[flagoffset + gpustep * mydest + NVTE_MAX_NVLINK +
                                                   firstrank]);

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
            while (CHECK_IDS(*flag, gathercounter)) {
            }
            gathercounter++;
          }

          asm volatile("bar.sync %0, %1;" ::"r"(1 + mydest), "r"(myblockDim));

          for (int line = start_elem; line < end_aligned; line += myblockDim * gridDim.x * UNROLL) {
            int4 val[UNROLL];
#pragma unroll
            for (int i = 0; i < UNROLL; i++)
              val[i] = peerptr[line + i * myblockDim * gridDim.x];
#pragma unroll
            for (int i = 0; i < UNROLL; i++)
              myptr[line + i * myblockDim * gridDim.x] = val[i];
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
  const int basecounter = gpuflag[NVTE_GF_STATE + op] + numblocks;
  hostflags[0] = basecounter;
  gpuflag[NVTE_GF_STATE + op] = basecounter;
  while (((volatile int *)gpuflag)[NVTE_GF_IBSHARPDONE] < basecounter) {
  }
}

#define callranks_block(x)                                                                         \
  if (comm->ar_nvsize == x)                                                                        \
    userbuffers_fp16_sum_inplace_gpu_rr_blocked<x><<<sms, warps * 32, 0, stream>>>(                \
        userbuffers_allreduceop_sharp, NVTE_REG0_OFFSET(comm), comm->ar_firstgpu, comm->ar_nvrank, \
        offset / 8, elements / 8, reinterpret_cast<void **>(comm->gpu_ptrs),                       \
        handler * comm->nvsize, blocksize / sizeof(int4) / comm->ar_nvsize,                        \
        reinterpret_cast<int *>(comm->hostflags), comm->flags,                                     \
        (elements * 2 + blocksize - 1) / blocksize);

#define callranks2_block(x)                                                                        \
  if (ar_nvsize == x) {                                                                            \
    int numblocks = (elements * 2 + blocksize - 1) / blocksize;                                    \
    int headstart = numblocks - 1; /*<3?numblocks-1:3;*/                                           \
    if (headstart > maxcredit)                                                                     \
      headstart = maxcredit;                                                                       \
    if (x == 1)                                                                                    \
      headstart = maxcredit;                                                                       \
    if (headstart > numblocks)                                                                     \
      headstart = numblocks;                                                                       \
    if (headstart == 0)                                                                            \
      headstart = 1;                                                                               \
    userbuffers_fp16_sum_inplace_gpu_rr_blocked2<x><<<sms, warps * 32, 0, stream>>>(               \
        op, maxcredit, headstart, my_node, num_nodes,                                              \
        NVTE_REG0_OFFSET(comm) + NVTE_REG0_FLAGS +                                                 \
            (op == userbuffers_allreduceop_nonsharp ? NVTE_REG0_COMMBUFFER : 0),                   \
        NVTE_REG0_OFFSET(comm) + NVTE_REG0_OPFLAGS * op, ar_firstgpu, ar_nvrank, ar_step,          \
        offset / 8, elements / 8, reinterpret_cast<void **>(comm->gpu_ptrs),                       \
        handler * comm->nvsize, blocksize / sizeof(int4) / ar_nvsize,                              \
        reinterpret_cast<int *>(comm->hostflags), comm->flags, numblocks);                         \
  }

#define callranks2_block_rs(x)                                                                     \
  if (ar_nvsize == x) {                                                                            \
    int numblocks = (elements * 2 + blocksize - 1) / blocksize;                                    \
    int headstart = numblocks - 1; /*<3?numblocks-1:3;*/                                           \
    if (headstart > maxcredit)                                                                     \
      headstart = maxcredit;                                                                       \
    if (x == 1)                                                                                    \
      headstart = maxcredit;                                                                       \
    if (headstart > numblocks)                                                                     \
      headstart = numblocks;                                                                       \
    if (headstart == 0)                                                                            \
      headstart = 1;                                                                               \
    userbuffers_fp16_sum_inplace_gpu_rr_blocked2_rs<x><<<sms, warps * 32, 0, stream>>>(            \
        op, maxcredit, headstart, my_node, num_nodes,                                              \
        NVTE_REG0_OFFSET(comm) + NVTE_REG0_FLAGS +                                                 \
            (op == userbuffers_allreduceop_nonsharp ? NVTE_REG0_COMMBUFFER : 0),                   \
        NVTE_REG0_OFFSET(comm) + NVTE_REG0_OPFLAGS * op, ar_firstgpu, ar_nvrank, ar_step,          \
        offset / 8, elements / 8, reinterpret_cast<void **>(comm->gpu_ptrs),                       \
        handler * comm->nvsize, blocksize / sizeof(int4) / ar_nvsize,                              \
        reinterpret_cast<int *>(comm->hostflags), comm->flags, numblocks);                         \
  }

#define callranks2_block_ag(x)                                                                     \
  if (ar_nvsize == x) {                                                                            \
    int numblocks = (elements * 2 + blocksize - 1) / blocksize;                                    \
    int headstart = numblocks - 1; /*<3?numblocks-1:3;*/                                           \
    if (headstart > maxcredit)                                                                     \
      headstart = maxcredit;                                                                       \
    if (x == 1)                                                                                    \
      headstart = maxcredit;                                                                       \
    if (headstart > numblocks)                                                                     \
      headstart = numblocks;                                                                       \
    if (headstart == 0)                                                                            \
      headstart = 1;                                                                               \
    userbuffers_fp16_sum_inplace_gpu_rr_blocked2_ag<x><<<sms, warps * 32, 0, stream>>>(            \
        op, maxcredit, headstart, my_node, num_nodes,                                              \
        NVTE_REG0_OFFSET(comm) + NVTE_REG0_FLAGS +                                                 \
            (op == userbuffers_allreduceop_nonsharp ? NVTE_REG0_COMMBUFFER : 0),                   \
        NVTE_REG0_OFFSET(comm) + NVTE_REG0_OPFLAGS * op, ar_firstgpu, ar_nvrank, ar_step,          \
        offset / 8, elements / 8, reinterpret_cast<void **>(comm->gpu_ptrs),                       \
        handler * comm->nvsize, blocksize / sizeof(int4) / ar_nvsize,                              \
        reinterpret_cast<int *>(comm->hostflags), comm->flags, numblocks);                         \
  }

#define callranks(x)                                                                               \
  if (ar_nvsize == x) {                                                                            \
    int arg1 = op - NVTE_MAX_OPS,                                                                  \
        arg2 = NVTE_REG0_OFFSET(comm) -                                                            \
               (op == userbuffers_allreduceop_nonsharp ? 2 : 1) * NVTE_REG0_SINGLENODE +           \
               NVTE_MAX_OPS,                                                                       \
        arg3 = ar_firstgpu, arg4 = ar_nvrank, arg5 = ar_step, arg6 = offset / 8,                   \
        arg7 = elements / 8;                                                                       \
    void **arg8 = reinterpret_cast<void **>(comm->gpu_ptrs);                                       \
    int arg9 = handler * comm->nvsize;                                                             \
    void *kernelArgs[] = {reinterpret_cast<void *>(&arg1), reinterpret_cast<void *>(&arg2),        \
                          reinterpret_cast<void *>(&arg3), reinterpret_cast<void *>(&arg4),        \
                          reinterpret_cast<void *>(&arg5), reinterpret_cast<void *>(&arg6),        \
                          reinterpret_cast<void *>(&arg7), reinterpret_cast<void *>(&arg8),        \
                          reinterpret_cast<void *>(&arg9)};                                        \
    CUDACHECK(cudaLaunchKernelExC(                                                                 \
        &cfg,                                                                                      \
        reinterpret_cast<void *>(comm->use_rr_kernel ? userbuffers_fp16_sum_inplace_gpu_rr<x>      \
                                                     : userbuffers_fp16_sum_inplace_gpu_rw<x>),    \
        kernelArgs));                                                                              \
  }

#define callranksMC(x)                                                                             \
  if (ar_nvsize == x) {                                                                            \
    int arg1 = op - NVTE_MAX_OPS,                                                                  \
        arg2 = NVTE_REG0_OFFSET(comm) -                                                            \
               (op == userbuffers_allreduceop_nonsharp ? 2 : 1) * NVTE_REG0_SINGLENODE +           \
               NVTE_MAX_OPS,                                                                       \
        arg3 = ar_firstgpu, arg4 = ar_nvrank, arg5 = ar_step, arg6 = offset / 8,                   \
        arg7 = elements / 8;                                                                       \
    void **arg8 = reinterpret_cast<void **>(comm->gpu_ptrs);                                       \
    int arg9 = handler * comm->nvsize;                                                             \
    void *arg10 = comm->mc_ptr[handler];                                                           \
    void *kernelArgs[] = {reinterpret_cast<void *>(&arg1), reinterpret_cast<void *>(&arg2),        \
                          reinterpret_cast<void *>(&arg3), reinterpret_cast<void *>(&arg4),        \
                          reinterpret_cast<void *>(&arg5), reinterpret_cast<void *>(&arg6),        \
                          reinterpret_cast<void *>(&arg7), reinterpret_cast<void *>(&arg8),        \
                          reinterpret_cast<void *>(&arg9), reinterpret_cast<void *>(&arg10)};      \
    CUDACHECK(cudaLaunchKernelExC(                                                                 \
        &cfg, reinterpret_cast<void *>(userbuffers_fp16_sum_inplace_gpu_mc<x>), kernelArgs));      \
  }

#define SETUP_LAUNCH_CONFIG(sms, threads, stream)                                                  \
  cudaLaunchConfig_t cfg = {sms, threads, 0, stream, NULL, 0};                                     \
  cudaLaunchAttribute attribute_ub[2];                                                             \
  attribute_ub[1].id = cudaLaunchAttributeClusterDimension;                                        \
  attribute_ub[1].val.clusterDim.x = sms % comm->cga_size == 0 ? comm->cga_size : 1;               \
  attribute_ub[1].val.clusterDim.y = 1;                                                            \
  attribute_ub[1].val.clusterDim.z = 1;                                                            \
  attribute_ub[0].id = cudaLaunchAttributeCooperative;                                             \
  cfg.attrs = attribute_ub;                                                                        \
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
  if (elements < 8)
    return 0;
  int sms = sms = comm->sms;
  int warps = comm->threads / 32;
  if (warps < comm->ar_nvsize)
    warps = comm->ar_nvsize;

  if (comm->launch_mode & NVTE_LAUNCH_GPU) {
    if (comm->ar_nvsize == 1)
      userbuffers_fp16_sum_inplace_gpu_null<<<1, 1, 0, stream>>>(
          userbuffers_allreduceop_sharp, reinterpret_cast<int *>(comm->hostflags), comm->flags,
          (elements * 2 + blocksize - 1) / blocksize);
    callranks_block(2) callranks_block(4) callranks_block(8)
  }
  return sms;
}

int allreduce2_userbuff_inplace_gpu(const int maxcredit, const int handler, const int offset,
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

  if (elements < 8)
    return 0;
  int sms = ar_nvsize == 1 ? 2 : comm->sms;
  int warps = comm->threads / 32;
  if (warps < ar_nvsize)
    warps = ar_nvsize;
  if (num_nodes > 1) {
    callranks2_block(1) callranks2_block(2) callranks2_block(4) callranks2_block(8)
  } else {
    SETUP_LAUNCH_CONFIG(sms, warps * 32, stream);
      callranks(2) callranks(4) callranks(8)
  }
  return sms;
}

#define callranks_ag(x)                                                                            \
  if (ar_nvsize == x) {                                                                            \
    int arg1 = op - NVTE_MAX_OPS,                                                                  \
        arg2 = NVTE_REG0_OFFSET(comm) -                                                            \
               (op == userbuffers_allreduceop_nonsharp ? 2 : 1) * NVTE_REG0_SINGLENODE +           \
               NVTE_MAX_OPS,                                                                       \
        arg3 = ar_firstgpu, arg4 = ar_nvrank, arg5 = ar_step, arg7 = elements / 8 / x,             \
        arg6 = offset / 8 + (comm->use_rr_kernel ? 0 : arg4 * arg7);                               \
    void **arg8 = reinterpret_cast<void **>(comm->gpu_ptrs);                                       \
    int arg9 = handler * comm->nvsize;                                                             \
    void *kernelArgs[] = {reinterpret_cast<void *>(&arg1), reinterpret_cast<void *>(&arg2),        \
                          reinterpret_cast<void *>(&arg3), reinterpret_cast<void *>(&arg4),        \
                          reinterpret_cast<void *>(&arg5), reinterpret_cast<void *>(&arg6),        \
                          reinterpret_cast<void *>(&arg7), reinterpret_cast<void *>(&arg8),        \
                          reinterpret_cast<void *>(&arg9)};                                        \
    CUDACHECK(cudaLaunchKernelExC(                                                                 \
        &cfg,                                                                                      \
        reinterpret_cast<void *>(comm->use_rr_kernel ? userbuffers_fp16_sum_inplace_gpu_rr_ag<x>   \
                                                     : userbuffers_fp16_sum_inplace_gpu_rw_ag<x>), \
        kernelArgs));                                                                              \
  }

#define callranks_agMC(x)                                                                          \
  if (ar_nvsize == x) {                                                                            \
    int arg1 = op - NVTE_MAX_OPS,                                                                  \
        arg2 = NVTE_REG0_OFFSET(comm) -                                                            \
               (op == userbuffers_allreduceop_nonsharp ? 2 : 1) * NVTE_REG0_SINGLENODE +           \
               NVTE_MAX_OPS,                                                                       \
        arg3 = ar_firstgpu, arg4 = ar_nvrank, arg5 = ar_step, arg7 = elements / 8 / x,             \
        arg6 = offset / 8 + arg4 * arg7;                                                           \
    void **arg8 = reinterpret_cast<void **>(comm->gpu_ptrs);                                       \
    int arg9 = handler * comm->nvsize;                                                             \
    uint4 *arg10 = reinterpret_cast<uint4 *>(comm->mc_ptr[handler]);                               \
    void *kernelArgs[] = {reinterpret_cast<void *>(&arg1), reinterpret_cast<void *>(&arg2),        \
                          reinterpret_cast<void *>(&arg3), reinterpret_cast<void *>(&arg4),        \
                          reinterpret_cast<void *>(&arg5), reinterpret_cast<void *>(&arg6),        \
                          reinterpret_cast<void *>(&arg7), reinterpret_cast<void *>(&arg8),        \
                          reinterpret_cast<void *>(&arg9), reinterpret_cast<void *>(&arg10)};      \
    CUDACHECK(cudaLaunchKernelExC(                                                                 \
        &cfg, reinterpret_cast<void *>(userbuffers_fp16_sum_inplace_gpu_mc_ag<x>), kernelArgs));   \
  }

#define callranks_rs(x)                                                                            \
  if (ar_nvsize == x) {                                                                            \
    int arg1 = op - NVTE_MAX_OPS,                                                                  \
        arg2 = NVTE_REG0_OFFSET(comm) -                                                            \
               (op == userbuffers_allreduceop_nonsharp ? 2 : 1) * NVTE_REG0_SINGLENODE +           \
               NVTE_MAX_OPS,                                                                       \
        arg3 = ar_firstgpu, arg4 = ar_nvrank, arg5 = ar_step, arg7 = elements / 8 / x,             \
        arg6 = offset / 8 + arg4 * arg7;                                                           \
    void **arg8 = reinterpret_cast<void **>(comm->gpu_ptrs);                                       \
    int arg9 = handler * comm->nvsize;                                                             \
    void *kernelArgs[] = {reinterpret_cast<void *>(&arg1), reinterpret_cast<void *>(&arg2),        \
                          reinterpret_cast<void *>(&arg3), reinterpret_cast<void *>(&arg4),        \
                          reinterpret_cast<void *>(&arg5), reinterpret_cast<void *>(&arg6),        \
                          reinterpret_cast<void *>(&arg7), reinterpret_cast<void *>(&arg8),        \
                          reinterpret_cast<void *>(&arg9)};                                        \
    CUDACHECK(cudaLaunchKernelExC(                                                                 \
        &cfg, reinterpret_cast<void *>(userbuffers_fp16_sum_inplace_gpu_rr_rs<x>), kernelArgs));   \
  }

#define callranks_rsMC(x)                                                                          \
  if (ar_nvsize == x) {                                                                            \
    int arg1 = op - NVTE_MAX_OPS,                                                                  \
        arg2 = NVTE_REG0_OFFSET(comm) -                                                            \
               (op == userbuffers_allreduceop_nonsharp ? 2 : 1) * NVTE_REG0_SINGLENODE +           \
               NVTE_MAX_OPS,                                                                       \
        arg3 = ar_firstgpu, arg4 = ar_nvrank, arg5 = ar_step, arg7 = elements / 8 / x,             \
        arg6 = offset / 8 + arg4 * arg7;                                                           \
    void **arg8 = reinterpret_cast<void **>(comm->gpu_ptrs);                                       \
    int arg9 = handler * comm->nvsize;                                                             \
    void *arg10 = comm->mc_ptr[handler];                                                           \
    void *kernelArgs[] = {reinterpret_cast<void *>(&arg1), reinterpret_cast<void *>(&arg2),        \
                          reinterpret_cast<void *>(&arg3), reinterpret_cast<void *>(&arg4),        \
                          reinterpret_cast<void *>(&arg5), reinterpret_cast<void *>(&arg6),        \
                          reinterpret_cast<void *>(&arg7), reinterpret_cast<void *>(&arg8),        \
                          reinterpret_cast<void *>(&arg9), reinterpret_cast<void *>(&arg10)};      \
    CUDACHECK(cudaLaunchKernelExC(                                                                 \
        &cfg, reinterpret_cast<void *>(userbuffers_fp16_sum_inplace_gpu_mc_rs<x>), kernelArgs));   \
  }

#define callranks_rs_oop(x)                                                                        \
  if (ar_nvsize == x) {                                                                            \
    int arg1 = op - NVTE_MAX_OPS,                                                                  \
        arg2 = NVTE_REG0_OFFSET(comm) -                                                            \
               (op == userbuffers_allreduceop_nonsharp ? 2 : 1) * NVTE_REG0_SINGLENODE +           \
               NVTE_MAX_OPS,                                                                       \
        arg3 = ar_firstgpu, arg4 = ar_nvrank, arg5 = ar_step, arg7 = elements / 8 / x,             \
        arg6 = offset / 8 + arg4 * arg7, arg8 = rowelements / 8, arg9 = strideelements / 8;        \
    void **arg10 = reinterpret_cast<void **>(comm->gpu_ptrs);                                      \
    int arg11 = handler * comm->nvsize;                                                            \
    void *arg12 = output;                                                                          \
    void *kernelArgs[] = {reinterpret_cast<void *>(&arg1),  reinterpret_cast<void *>(&arg2),       \
                          reinterpret_cast<void *>(&arg3),  reinterpret_cast<void *>(&arg4),       \
                          reinterpret_cast<void *>(&arg5),  reinterpret_cast<void *>(&arg6),       \
                          reinterpret_cast<void *>(&arg7),  reinterpret_cast<void *>(&arg8),       \
                          reinterpret_cast<void *>(&arg9),  reinterpret_cast<void *>(&arg10),      \
                          reinterpret_cast<void *>(&arg11), reinterpret_cast<void *>(&arg12)};     \
    CUDACHECK(cudaLaunchKernelExC(                                                                 \
        &cfg, reinterpret_cast<void *>(userbuffers_fp16_sum_inplace_gpu_rr_rs_oop<x>),             \
        kernelArgs));                                                                              \
  }

#define callranks_rs_oop_fp8(x)                                                                    \
  if (ar_nvsize == x) {                                                                            \
    int arg1 = op - NVTE_MAX_OPS,                                                                  \
        arg2 = NVTE_REG0_OFFSET(comm) -                                                            \
               (op == userbuffers_allreduceop_nonsharp ? 2 : 1) * NVTE_REG0_SINGLENODE +           \
               NVTE_MAX_OPS,                                                                       \
        arg3 = ar_firstgpu, arg4 = ar_nvrank, arg5 = ar_step, arg7 = elements / 16 / x,            \
        arg6 = offset / 16 + arg4 * arg7, arg8 = rowelements / 8, arg9 = strideelements / 8;       \
    void **arg10 = reinterpret_cast<void **>(comm->gpu_ptrs);                                      \
    int arg11 = handler * comm->nvsize;                                                            \
    void *arg12 = output;                                                                          \
    float *arg13 = scale;                                                                          \
    void *kernelArgs[] = {reinterpret_cast<void *>(&arg1),  reinterpret_cast<void *>(&arg2),       \
                          reinterpret_cast<void *>(&arg3),  reinterpret_cast<void *>(&arg4),       \
                          reinterpret_cast<void *>(&arg5),  reinterpret_cast<void *>(&arg6),       \
                          reinterpret_cast<void *>(&arg7),  reinterpret_cast<void *>(&arg8),       \
                          reinterpret_cast<void *>(&arg9),  reinterpret_cast<void *>(&arg10),      \
                          reinterpret_cast<void *>(&arg11), reinterpret_cast<void *>(&arg12),      \
                          reinterpret_cast<void *>(&arg13)};                                       \
    CUDACHECK(cudaLaunchKernelExC(                                                                 \
        &cfg,                                                                                      \
        reinterpret_cast<void *>(userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_fp8<x, fp8type>),      \
        kernelArgs));                                                                              \
  }

#define callranks_rs_oopMC(x)                                                                      \
  if (ar_nvsize == x) {                                                                            \
    int arg1 = op - NVTE_MAX_OPS,                                                                  \
        arg2 = NVTE_REG0_OFFSET(comm) -                                                            \
               (op == userbuffers_allreduceop_nonsharp ? 2 : 1) * NVTE_REG0_SINGLENODE +           \
               NVTE_MAX_OPS,                                                                       \
        arg3 = ar_firstgpu, arg4 = ar_nvrank, arg5 = ar_step, arg7 = elements / 8 / x,             \
        arg6 = offset / 8 + arg4 * arg7, arg8 = rowelements / 8, arg9 = strideelements / 8;        \
    void **arg10 = reinterpret_cast<void **>(comm->gpu_ptrs);                                      \
    int arg11 = handler * comm->nvsize;                                                            \
    void *arg12 = output;                                                                          \
    void *arg13 = comm->mc_ptr[handler];                                                           \
    void *kernelArgs[] = {reinterpret_cast<void *>(&arg1),  reinterpret_cast<void *>(&arg2),       \
                          reinterpret_cast<void *>(&arg3),  reinterpret_cast<void *>(&arg4),       \
                          reinterpret_cast<void *>(&arg5),  reinterpret_cast<void *>(&arg6),       \
                          reinterpret_cast<void *>(&arg7),  reinterpret_cast<void *>(&arg8),       \
                          reinterpret_cast<void *>(&arg9),  reinterpret_cast<void *>(&arg10),      \
                          reinterpret_cast<void *>(&arg11), reinterpret_cast<void *>(&arg12),      \
                          reinterpret_cast<void *>(&arg13)};                                       \
    CUDACHECK(cudaLaunchKernelExC(                                                                 \
        &cfg, reinterpret_cast<void *>(userbuffers_fp16_sum_inplace_gpu_mc_rs_oop<x>),             \
        kernelArgs));                                                                              \
  }

#define callranks_rs_oop_atomic_fp8(x)                                                             \
  if (ar_nvsize == x) {                                                                            \
    int arg1 = op - NVTE_MAX_OPS,                                                                  \
        arg2 = NVTE_REG0_OFFSET(comm) -                                                            \
               (op == userbuffers_allreduceop_nonsharp ? 2 : 1) * NVTE_REG0_SINGLENODE +           \
               NVTE_MAX_OPS,                                                                       \
        arg3 = ar_firstgpu, arg4 = ar_nvrank, arg5 = ar_step, arg7 = elements / 16 / x,            \
        arg6 = offset / 16, arg8 = rowelements / 8, arg9 = strideelements_out / 8,                 \
        arg10 = strideelements_in / 16;                                                            \
    void **arg11 = reinterpret_cast<void **>(comm->gpu_ptrs);                                      \
    int arg12 = handler * comm->nvsize;                                                            \
    void *arg13 = output;                                                                          \
    float *arg14 = scale;                                                                          \
    void *arg15 = counters;                                                                        \
    int arg16 = numchunks, arg17 = atomicindex;                                                    \
    void *kernelArgs[] = {reinterpret_cast<void *>(&arg1),  reinterpret_cast<void *>(&arg2),       \
                          reinterpret_cast<void *>(&arg3),  reinterpret_cast<void *>(&arg4),       \
                          reinterpret_cast<void *>(&arg5),  reinterpret_cast<void *>(&arg6),       \
                          reinterpret_cast<void *>(&arg7),  reinterpret_cast<void *>(&arg8),       \
                          reinterpret_cast<void *>(&arg9),  reinterpret_cast<void *>(&arg10),      \
                          reinterpret_cast<void *>(&arg11), reinterpret_cast<void *>(&arg12),      \
                          reinterpret_cast<void *>(&arg13), reinterpret_cast<void *>(&arg14),      \
                          reinterpret_cast<void *>(&arg15), reinterpret_cast<void *>(&arg16),      \
                          reinterpret_cast<void *>(&arg17)};                                       \
    CUDACHECK(cudaLaunchKernelExC(                                                                 \
        &cfg,                                                                                      \
        reinterpret_cast<void *>(                                                                  \
            userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_atomic_fp8<x, fp8type>),                    \
        kernelArgs));                                                                              \
  }

#define callranks_rs_oop_stride(x)                                                                 \
  if (ar_nvsize == x) {                                                                            \
    int arg1 = op - NVTE_MAX_OPS,                                                                  \
        arg2 = NVTE_REG0_OFFSET(comm) -                                                            \
               (op == userbuffers_allreduceop_nonsharp ? 2 : 1) * NVTE_REG0_SINGLENODE +           \
               NVTE_MAX_OPS,                                                                       \
        arg3 = ar_firstgpu, arg4 = ar_nvrank, arg5 = ar_step, arg7 = elements / 8 / x,             \
        arg6 = offset / 8, arg8 = rowelements / 8, arg9 = strideelements / 8;                      \
    void **arg10 = reinterpret_cast<void **>(comm->gpu_ptrs);                                      \
    int arg11 = handler * comm->nvsize;                                                            \
    void *arg12 = output;                                                                          \
    void *kernelArgs[] = {reinterpret_cast<void *>(&arg1),  reinterpret_cast<void *>(&arg2),       \
                          reinterpret_cast<void *>(&arg3),  reinterpret_cast<void *>(&arg4),       \
                          reinterpret_cast<void *>(&arg5),  reinterpret_cast<void *>(&arg6),       \
                          reinterpret_cast<void *>(&arg7),  reinterpret_cast<void *>(&arg8),       \
                          reinterpret_cast<void *>(&arg9),  reinterpret_cast<void *>(&arg10),      \
                          reinterpret_cast<void *>(&arg11), reinterpret_cast<void *>(&arg12)};     \
    CUDACHECK(cudaLaunchKernelExC(                                                                 \
        &cfg, reinterpret_cast<void *>(userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_stride<x>),      \
        kernelArgs));                                                                              \
  }

#if 0
#define callranks_rs_oop_stride_atomic_fp8(x)                                                      \
  if (ar_nvsize == x) {                                                                            \
    int arg1 = op - NVTE_MAX_OPS,                                                                  \
        arg2 = NVTE_REG0_OFFSET(comm) -                                                            \
               (op == userbuffers_allreduceop_nonsharp ? 2 : 1) * NVTE_REG0_SINGLENODE +           \
               NVTE_MAX_OPS,                                                                       \
        arg3 = ar_firstgpu, arg4 = ar_nvrank, arg5 = ar_step, arg7 = elements / 16 / x,            \
        arg6 = offset / 16, arg8 = rowelements / 8, arg9 = strideelements / 8, arg10 = numchunks;  \
    void **arg11 = reinterpret_cast<void **>(comm->gpu_ptrs);                                      \
    int arg12 = handler * comm->nvsize;                                                            \
    void *arg13 = output;                                                                          \
    void *arg14 = counters;                                                                        \
    float *arg15 = scale;                                                                          \
    void *kernelArgs[] = {reinterpret_cast<void *>(&arg1),  reinterpret_cast<void *>(&arg2),       \
                          reinterpret_cast<void *>(&arg3),  reinterpret_cast<void *>(&arg4),       \
                          reinterpret_cast<void *>(&arg5),  reinterpret_cast<void *>(&arg6),       \
                          reinterpret_cast<void *>(&arg7),  reinterpret_cast<void *>(&arg8),       \
                          reinterpret_cast<void *>(&arg9),  reinterpret_cast<void *>(&arg10),      \
                          reinterpret_cast<void *>(&arg11), reinterpret_cast<void *>(&arg12),      \
                          reinterpret_cast<void *>(&arg13), reinterpret_cast<void *>(&arg14),      \
                          reinterpret_cast<void *>(&arg15)};                                       \
    CUDACHECK(cudaLaunchKernelExC(                                                                 \
        &cfg,                                                                                      \
        reinterpret_cast<void *>(                                                                  \
            userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_stride_atomic_fp8<x, fp8type>),             \
        kernelArgs));                                                                              \
  }
#endif

#define callranks_rs_oop_stride_atomic(x)                                                          \
  if (ar_nvsize == x) {                                                                            \
    int arg1 = op - NVTE_MAX_OPS,                                                                  \
        arg2 = NVTE_REG0_OFFSET(comm) -                                                            \
               (op == userbuffers_allreduceop_nonsharp ? 2 : 1) * NVTE_REG0_SINGLENODE +           \
               NVTE_MAX_OPS,                                                                       \
        arg3 = ar_firstgpu, arg4 = ar_nvrank, arg5 = ar_step, arg7 = elements / 8 / x,             \
        arg6 = offset / 8, arg8 = rowelements / 8, arg9 = strideelements / 8, arg10 = numchunks;   \
    void **arg11 = reinterpret_cast<void **>(comm->gpu_ptrs);                                      \
    int arg12 = handler * comm->nvsize;                                                            \
    void *arg13 = output;                                                                          \
    void *arg14 = counters;                                                                        \
    void *kernelArgs[] = {reinterpret_cast<void *>(&arg1),  reinterpret_cast<void *>(&arg2),       \
                          reinterpret_cast<void *>(&arg3),  reinterpret_cast<void *>(&arg4),       \
                          reinterpret_cast<void *>(&arg5),  reinterpret_cast<void *>(&arg6),       \
                          reinterpret_cast<void *>(&arg7),  reinterpret_cast<void *>(&arg8),       \
                          reinterpret_cast<void *>(&arg9),  reinterpret_cast<void *>(&arg10),      \
                          reinterpret_cast<void *>(&arg11), reinterpret_cast<void *>(&arg12),      \
                          reinterpret_cast<void *>(&arg13), reinterpret_cast<void *>(&arg14)};     \
    CUDACHECK(cudaLaunchKernelExC(                                                                 \
        &cfg,                                                                                      \
        reinterpret_cast<void *>(userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_stride_atomic<x>),     \
        kernelArgs));                                                                              \
  }

#define callranks_rs_oop_stride_multiatomic(x)                                                     \
  if (ar_nvsize == x) {                                                                            \
    int arg1 = op - NVTE_MAX_OPS,                                                                  \
        arg2 = NVTE_REG0_OFFSET(comm) -                                                            \
               (op == userbuffers_allreduceop_nonsharp ? 2 : 1) * NVTE_REG0_SINGLENODE +           \
               NVTE_MAX_OPS,                                                                       \
        arg3 = ar_firstgpu, arg4 = ar_nvrank, arg5 = ar_step, arg7 = elements / 8 / x,             \
        arg6 = offset / 8, arg8 = rowelements / 8, arg9 = strideelements / 8, arg10 = numchunks;   \
    void **arg11 = reinterpret_cast<void **>(comm->gpu_ptrs);                                      \
    int arg12 = handler * comm->nvsize;                                                            \
    void *arg13 = output;                                                                          \
    void *arg14 = counters;                                                                        \
    void *kernelArgs[] = {reinterpret_cast<void *>(&arg1),  reinterpret_cast<void *>(&arg2),       \
                          reinterpret_cast<void *>(&arg3),  reinterpret_cast<void *>(&arg4),       \
                          reinterpret_cast<void *>(&arg5),  reinterpret_cast<void *>(&arg6),       \
                          reinterpret_cast<void *>(&arg7),  reinterpret_cast<void *>(&arg8),       \
                          reinterpret_cast<void *>(&arg9),  reinterpret_cast<void *>(&arg10),      \
                          reinterpret_cast<void *>(&arg11), reinterpret_cast<void *>(&arg12),      \
                          reinterpret_cast<void *>(&arg13), reinterpret_cast<void *>(&arg14)};     \
    CUDACHECK(                                                                                     \
        cudaLaunchKernelExC(&cfg,                                                                  \
                            reinterpret_cast<void *>(                                              \
                                userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_stride_multiatomic<x>), \
                            kernelArgs));                                                          \
  }

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

  if (elements < 8)
    return 0;
  int sms = ar_nvsize == 1 ? 2 : comm->sms;
  int warps = comm->threads / 32;
  if (warps < ar_nvsize)
    warps = ar_nvsize;

  if (num_nodes > 1) {
    callranks2_block_rs(1) callranks2_block_rs(2) callranks2_block_rs(4) callranks2_block_rs(8)
  } else {
    SETUP_LAUNCH_CONFIG(sms, warps * 32, stream);
    if (comm->use_mc && (comm->memflags[handler] & UB_MEM_MC_CREATED)) {
      callranks_rsMC(2) callranks_rsMC(4) callranks_rsMC(8)
    } else {
      callranks_rs(2) callranks_rs(4) callranks_rs(8)
    }
  }
  return sms;
}

void reducescatter2_userbuff_strided(void *output, const int handler, const int offset,
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

  if (elements < 64)
    return;
  int sms = ar_nvsize == 1 ? 2 : comm->sms;
  int warps = comm->threads / 32;
  if (warps < ar_nvsize)
    warps = ar_nvsize;

  SETUP_LAUNCH_CONFIG(sms, warps * 32, stream);
  callranks_rs_oop_stride(2) callranks_rs_oop_stride(4) callranks_rs_oop_stride(8)
}
void reducescatter2_userbuff_strided_atomic(void *output, const int handler, const int offset,
                                            const int rowelements, const int colelements,
                                            const int strideelements, const int numchunks,
                                            void *counters, communicator *comm,
                                            cudaStream_t stream) {
  const int elements = rowelements * colelements;
  const int op = userbuffers_allreduceop_nonsharp2;
  const int blocksize = elements * 2;
  const int ar_firstgpu =
      op == userbuffers_allreduceop_nonsharp ? comm->ar_firstgpu : comm->ar2_firstgpu;
  const int ar_step = op == userbuffers_allreduceop_nonsharp2 ? 1 : comm->ar2_nvsize;
  const int ar_nvsize = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvsize : comm->ar2_nvsize;
  const int ar_nvrank = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvrank : comm->ar2_nvrank;

  if (elements < 64)
    return;
  int sms = ar_nvsize == 1 ? 2 : comm->sms;
  int warps = comm->threads / 32;
  if (warps < ar_nvsize)
    warps = ar_nvsize;

  SETUP_LAUNCH_CONFIG(sms, warps * 32, stream);
  callranks_rs_oop_stride_atomic(2) callranks_rs_oop_stride_atomic(4)
      callranks_rs_oop_stride_atomic(8)
}

#if 0
  template<typename fp8type>
  void reducescatter2_userbuff_strided_atomic_fp8(
    void* output, float *scale, const int handler, const int offset, const int rowelements,
    const int colelements, const int strideelements, const int numchunks, void *counters,
    communicator* comm, cudaStream_t stream) {
      const int elements = rowelements*colelements;
      const int op = userbuffers_allreduceop_nonsharp2;
      const int blocksize = elements;
      const int ar_firstgpu = op == userbuffers_allreduceop_nonsharp ?
                              comm->ar_firstgpu : comm->ar2_firstgpu;
      const int ar_step = op == userbuffers_allreduceop_nonsharp2 ?
                          1 : comm->ar2_nvsize;
      const int ar_nvsize = op == userbuffers_allreduceop_nonsharp ?
                            comm->ar_nvsize : comm->ar2_nvsize;
      const int ar_nvrank = op == userbuffers_allreduceop_nonsharp ?
                            comm->ar_nvrank : comm->ar2_nvrank;

      assert(comm->sm_arch >= 9);
      if (elements < 128) return;
      int sms = ar_nvsize == 1 ? 2 : comm->sms;
      int warps = comm->threads/32;
      if (warps < ar_nvsize) warps = ar_nvsize;

      SETUP_LAUNCH_CONFIG(sms, warps*32, stream);
      callranks_rs_oop_stride_atomic_fp8(2)
      callranks_rs_oop_stride_atomic_fp8(4)
      callranks_rs_oop_stride_atomic_fp8(8)
  }
#endif
template <typename fp8type>
void reducescatter2_userbuff_strided_universal_fp8(void *output, float *scale, const int handler,
                                                   const int offset, const int rowelements,
                                                   const int colelements,
                                                   const int strideelements_out,
                                                   const int strideelements_in, const int numchunks,
                                                   const int atomicindex, void *counters,
                                                   communicator *comm, cudaStream_t stream) {
  const int elements = rowelements * colelements;
  const int op = userbuffers_allreduceop_nonsharp2;
  const int blocksize = elements;
  const int ar_firstgpu =
      op == userbuffers_allreduceop_nonsharp ? comm->ar_firstgpu : comm->ar2_firstgpu;
  const int ar_step = op == userbuffers_allreduceop_nonsharp2 ? 1 : comm->ar2_nvsize;
  const int ar_nvsize = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvsize : comm->ar2_nvsize;
  const int ar_nvrank = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvrank : comm->ar2_nvrank;
  assert(comm->sm_arch >= 9);
  if (elements < 128)
    return;
  int sms = ar_nvsize == 1 ? 2 : comm->sms;
  int warps = comm->threads / 32;
  if (warps < ar_nvsize)
    warps = ar_nvsize;

  SETUP_LAUNCH_CONFIG(sms, warps * 32, stream);
  callranks_rs_oop_atomic_fp8(2) callranks_rs_oop_atomic_fp8(4) callranks_rs_oop_atomic_fp8(8)
}

template <typename fp8type>
void reducescatter2_userbuff_strided_atomic_fp8(void *output, float *scale, const int handler,
                                                const int offset, const int rowelements,
                                                const int colelements, const int strideelements_out,
                                                const int strideelements_in, const int numchunks,
                                                void *counters, communicator *comm,
                                                cudaStream_t stream) {
  reducescatter2_userbuff_strided_universal_fp8<fp8type>(
      output, scale, handler, offset, rowelements, colelements, strideelements_out,
      strideelements_in, 1, numchunks, counters /*nullptr*/, comm, stream);
}
template <typename fp8type>
void reducescatter2_userbuff_strided_multiatomic_fp8(
    void *output, float *scale, const int handler, const int offset, const int rowelements,
    const int colelements, const int strideelements_out, const int strideelements_in,
    const int numchunks, void *counters, communicator *comm, cudaStream_t stream) {
  reducescatter2_userbuff_strided_universal_fp8<fp8type>(
      output, scale, handler, offset, rowelements, colelements, strideelements_out,
      strideelements_in, numchunks, 0, counters /*nullptr*/, comm, stream);
}

void reducescatter2_userbuff_strided_multiatomic(void *output, const int handler, const int offset,
                                                 const int rowelements, const int colelements,
                                                 const int strideelements, const int numchunks,
                                                 void *counters, communicator *comm,
                                                 cudaStream_t stream) {
  const int elements = rowelements * colelements;
  const int op = userbuffers_allreduceop_nonsharp2;
  const int blocksize = elements * 2;
  const int ar_firstgpu =
      op == userbuffers_allreduceop_nonsharp ? comm->ar_firstgpu : comm->ar2_firstgpu;
  const int ar_step = op == userbuffers_allreduceop_nonsharp2 ? 1 : comm->ar2_nvsize;
  const int ar_nvsize = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvsize : comm->ar2_nvsize;
  const int ar_nvrank = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvrank : comm->ar2_nvrank;

  if (elements < 64)
    return;
  int sms = ar_nvsize == 1 ? 2 : comm->sms;
  int warps = comm->threads / 32;
  if (warps < ar_nvsize)
    warps = ar_nvsize;

  SETUP_LAUNCH_CONFIG(sms, warps * 32, stream);
  // if(comm->use_mc && (comm->memflags[handler] & NVTE_UB_MEM_MC_CREATED)) {
  //   //callranks_rs_oopMC(2)
  //   //callranks_rs_oopMC(4)
  //   //callranks_rs_oopMC(8)
  // } else {
  //   if(comm->memflags[handler] & NVTE_UB_MEM_UC_CONTIG) {
  //     //callranks_rs_oopUCPTR(2)
  //     //callranks_rs_oopUCPTR(4)
  //     //callranks_rs_oopUCPTR(8)
  //   } else {
  callranks_rs_oop_stride_multiatomic(2) callranks_rs_oop_stride_multiatomic(4)
      callranks_rs_oop_stride_multiatomic(8)
  //  }
  //}
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

  if (elements < 8)
    return 0;
  int sms = ar_nvsize == 1 ? 2 : comm->sms;
  int warps = comm->threads / 32;
  if (warps < ar_nvsize)
    warps = ar_nvsize;

  if (num_nodes > 1) {
    callranks2_block_ag(1) callranks2_block_ag(2) callranks2_block_ag(4) callranks2_block_ag(8)
  } else {
    SETUP_LAUNCH_CONFIG(sms, warps * 32, stream);
    callranks_ag(2) callranks_ag(4) callranks_ag(8)
  }
  return sms;
}

void allgather2_userbuff_inplace(const int handler, const int offset, const int elements,
                                 communicator *comm, cudaStream_t stream) {
  const int op = userbuffers_allreduceop_nonsharp2;
  const int blocksize = elements * 2;
  const int ar_firstgpu =
      op == userbuffers_allreduceop_nonsharp ? comm->ar_firstgpu : comm->ar2_firstgpu;
  const int ar_step = op == userbuffers_allreduceop_nonsharp2 ? 1 : comm->ar2_nvsize;
  const int ar_nvsize = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvsize : comm->ar2_nvsize;
  const int ar_nvrank = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvrank : comm->ar2_nvrank;

  if (elements < 64)
    return;
  int sms = ar_nvsize == 1 ? 2 : comm->sms;
  int warps = comm->threads / 32;
  if (warps < ar_nvsize)
    warps = ar_nvsize;

  SETUP_LAUNCH_CONFIG(sms, warps * 32, stream);
  if (comm->use_mc && (comm->memflags[handler] & UB_MEM_MC_CREATED)) {
    callranks_agMC(2) callranks_agMC(4) callranks_agMC(8)
  } else {
    callranks_ag(2) callranks_ag(4) callranks_ag(8)
  }
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

  if (elements < 64)
    return;
  int sms = ar_nvsize == 1 ? 2 : comm->sms;
  int warps = comm->threads / 32;
  if (warps < ar_nvsize)
    warps = ar_nvsize;

  SETUP_LAUNCH_CONFIG(sms, warps * 32, stream);
  if (comm->use_mc && (comm->memflags[handler] & UB_MEM_MC_CREATED)) {
    callranks_rsMC(2) callranks_rsMC(4) callranks_rsMC(8)
  } else {
    callranks_rs(2) callranks_rs(4) callranks_rs(8)
  }
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

  if (elements < 64)
    return;
  int sms = ar_nvsize == 1 ? 2 : comm->sms;
  int warps = comm->threads / 32;
  if (warps < ar_nvsize)
    warps = ar_nvsize;

  SETUP_LAUNCH_CONFIG(sms, warps * 32, stream);
  if (comm->use_mc && (comm->memflags[handler] & UB_MEM_MC_CREATED)) {
    callranks_rs_oopMC(2) callranks_rs_oopMC(4) callranks_rs_oopMC(8)
  } else {
    callranks_rs_oop(2) callranks_rs_oop(4) callranks_rs_oop(8)
  }
}
void reducescatter2_userbuff(void *output, const int handler, const int offset, const int elements,
                             communicator *comm, cudaStream_t stream) {
  reducescatter2_userbuff_stridedoutput(output, handler, offset, elements, 1, 0, comm, stream);
}

template <typename fp8type>
void reducescatter2_userbuff_stridedoutput_fp8(void *output, float *scale, const int handler,
                                               const int offset, const int rowelements,
                                               const int colelements, const int strideelements,
                                               communicator *comm, cudaStream_t stream) {
  const int elements = rowelements * colelements;
  const int op = userbuffers_allreduceop_nonsharp2;
  const int blocksize = elements;
  const int ar_firstgpu =
      op == userbuffers_allreduceop_nonsharp ? comm->ar_firstgpu : comm->ar2_firstgpu;
  const int ar_step = op == userbuffers_allreduceop_nonsharp2 ? 1 : comm->ar2_nvsize;
  const int ar_nvsize = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvsize : comm->ar2_nvsize;
  const int ar_nvrank = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvrank : comm->ar2_nvrank;
  assert(comm->sm_arch >= 9);
  if (elements < 128)
    return;
  int sms = ar_nvsize == 1 ? 2 : comm->sms;
  int warps = comm->threads / 32;
  if (warps < ar_nvsize)
    warps = ar_nvsize;

  SETUP_LAUNCH_CONFIG(sms, warps * 32, stream);
  callranks_rs_oop_fp8(2) callranks_rs_oop_fp8(4) callranks_rs_oop_fp8(8)
}

template <typename fp8type>
void reducescatter2_userbuff_fp8(void *output, float *scale, const int handler, const int offset,
                                 const int elements, communicator *comm, cudaStream_t stream) {
  reducescatter2_userbuff_stridedoutput_fp8<fp8type>(output, scale, handler, offset, elements, 1, 0,
                                                     comm, stream);
}

template void reducescatter2_userbuff_fp8<__nv_fp8_e5m2>(void *output, float *scale,
                                                         const int handler, const int offset,
                                                         const int elements, communicator *comm,
                                                         cudaStream_t stream = 0);
template void reducescatter2_userbuff_fp8<__nv_fp8_e4m3>(void *output, float *scale,
                                                         const int handler, const int offset,
                                                         const int elements, communicator *comm,
                                                         cudaStream_t stream = 0);
#if 0
template void reducescatter2_userbuff_strided_atomic_fp8<__nv_fp8_e4m3>(
    void* output, float *scale, const int handler, const int offset,
    const int rowelements, const int colelements, const int strideelements,
    const int numchunks, void *counters, communicator* comm, cudaStream_t stream = 0);
#endif
template void reducescatter2_userbuff_strided_atomic_fp8<__nv_fp8_e4m3>(
    void *output, float *scale, const int handler, const int offset, const int rowelements,
    const int colelements, const int strideelements_out, const int strideelements_in,
    const int numchunks, void *counters, communicator *comm, cudaStream_t stream = 0);
template void reducescatter2_userbuff_strided_multiatomic_fp8<__nv_fp8_e4m3>(
    void *output, float *scale, const int handler, const int offset, const int rowelements,
    const int colelements, const int strideelements_out, const int strideelements_in,
    const int numchunks, void *counters, communicator *comm, cudaStream_t stream = 0);
__global__ void __launch_bounds__(MAX_THREADS)
    kuserbuffers_pullsendrecv(int myrank, int peer, int *recv_id, int *send_flagptr,
                              int *recv_flagptr, int4 *srcptr, int4 *dstptr, const int lines) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    atomicAdd_system(send_flagptr, 1);
  }

#define UNROLLCOPY 8
  const int start_elem = threadIdx.x + blockDim.x * blockIdx.x;
  const int end_elem = lines;
  const int aligned_elem = (end_elem - start_elem) & (~(blockDim.x * gridDim.x * UNROLLCOPY - 1));
  const int end_aligned = start_elem + aligned_elem;

  if (threadIdx.x == 0) {
    const int signal_id = (*recv_id) + 1;
    volatile int *flag = (volatile int *)recv_flagptr;
    clock_t s = clock64();
    while (CHECK_IDS(*flag, signal_id)) {
      if (clock64() - s > TIMEOUT) {
        printf("[%d from %d] pullrecv: expected %d, stuck with %d\n", myrank, peer, signal_id,
               *flag);
        break;
      }
    }
    if (lines == 0) {
      *recv_id = signal_id;
      return;
    }  // otherwise need an extra kernel
  }
  __syncthreads();

  if (end_elem <= start_elem)
    return;

  for (int line = start_elem; line < end_aligned; line += blockDim.x * gridDim.x * UNROLLCOPY) {
    int4 val[UNROLLCOPY];
#pragma unroll
    for (int i = 0; i < UNROLLCOPY; i++)
      val[i] = srcptr[line + i * blockDim.x * gridDim.x];
#pragma unroll
    for (int i = 0; i < UNROLLCOPY; i++)
      dstptr[line + i * blockDim.x * gridDim.x] = val[i];
  }
  for (int line = end_aligned; line < end_elem; line += blockDim.x * gridDim.x)
    dstptr[line] = srcptr[line];
}

__global__ void kuserbuffers_pullsend(int myrank, int peer, int *send_id, int *flagptr) {
  atomicAdd_system(flagptr, 1);
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
  const int end_aligned = start_elem + aligned_elem;

  if (threadIdx.x == 0) {
    const int signal_id = (*recv_id) + 1;
    volatile int *flag = (volatile int *)flagptr;
    clock_t s = clock64();
    while (CHECK_IDS(*flag, signal_id)) {
      if (clock64() - s > TIMEOUT) {
        printf("[%d from %d] pullrecv: expected %d, stuck with %d\n", myrank, peer, signal_id,
               *flag);
        break;
      }
    }
    if (lines == 0) {
      *recv_id = signal_id;
      return;
    }  // otherwise need an extra kernel
  }
  __syncthreads();

  if (end_elem <= start_elem)
    return;

  for (int line = start_elem; line < end_aligned; line += blockDim.x * gridDim.x * UNROLLCOPY) {
    int4 val[UNROLLCOPY];
#pragma unroll
    for (int i = 0; i < UNROLLCOPY; i++)
      val[i] = srcptr[line + i * blockDim.x * gridDim.x];
#pragma unroll
    for (int i = 0; i < UNROLLCOPY; i++)
      dstptr[line + i * blockDim.x * gridDim.x] = val[i];
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
    const int end_aligned = start_elem + aligned_elem;
    if (end_elem > start_elem) {
      for (int line = start_elem; line < end_aligned; line += blockDim.x * gridDim.x * UNROLLCOPY) {
        int4 val[UNROLLCOPY];
#pragma unroll
        for (int i = 0; i < UNROLLCOPY; i++)
          val[i] = srcptr[line + i * blockDim.x * gridDim.x];
#pragma unroll
        for (int i = 0; i < UNROLLCOPY; i++)
          dstptr[line + i * blockDim.x * gridDim.x] = val[i];
      }
      for (int line = end_aligned; line < end_elem; line += blockDim.x * gridDim.x)
        dstptr[line] = srcptr[line];
    }
    __syncthreads();
    if (threadIdx.x)
      return;
    __threadfence_system();
    atomicAdd_system(flagptr,
                     1);  // otherwise need local SM sync before sending flag
  } else {                // 0 bytes and 1 SM only
    atomicAdd_system(flagptr, 1);
  }
}

__global__ void kuserbuffers_pushrecv(int myrank, int peer, int *recv_id, int *flagptr, int adder) {
  const int signal_id = (*recv_id) + adder;
  *recv_id = signal_id;
  volatile int *flag = (volatile int *)flagptr;
  if (*flag >= signal_id)
    return;
  clock_t s = clock64();
  while (CHECK_IDS(*flag, signal_id)) {
    if (clock64() - s > TIMEOUT) {
      printf("%d from %d] pushrecv: expected %d, stuck with %d\n", myrank, peer, signal_id, *flag);
      return;
    }
  }
}

__global__ void __launch_bounds__(MAX_THREADS)
    kuserbuffers_pushsendrecv(int *send_id, int *send_flagptr, int4 *srcptr, int4 *dstptr,
                              const int lines, int myrank, int peer, int *recv_id,
                              int *recv_flagptr, int adder) {
  if (lines) {
    const int start_elem = threadIdx.x + blockDim.x * blockIdx.x;
    const int end_elem = lines;
    const int aligned_elem =
        ((end_elem - start_elem) & (~(blockDim.x * gridDim.x * UNROLLCOPY - 1)));
    const int end_aligned = start_elem + aligned_elem;
    if (end_elem > start_elem) {
      for (int line = start_elem; line < end_aligned; line += blockDim.x * gridDim.x * UNROLLCOPY) {
        int4 val[UNROLLCOPY];
#pragma unroll
        for (int i = 0; i < UNROLLCOPY; i++) {
          val[i] = srcptr[line + i * blockDim.x * gridDim.x];
        }
#pragma unroll
        for (int i = 0; i < UNROLLCOPY; i++) {
          dstptr[line + i * blockDim.x * gridDim.x] = val[i];
        }
      }
      for (int line = end_aligned; line < end_elem; line += blockDim.x * gridDim.x) {
        dstptr[line] = srcptr[line];
      }
    }
    __syncthreads();
    if (threadIdx.x)
      return;
    __threadfence_system();
    atomicAdd_system(send_flagptr,
                     1);  // otherwise need local SM sync before sending flag
  } else {                // 0 bytes and 1 SM only
    atomicAdd_system(send_flagptr, 1);
  }

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const int signal_id = (*recv_id) + adder;
    *recv_id = signal_id;
    volatile int *flag = (volatile int *)recv_flagptr;
    if (*flag >= signal_id)
      return;
    clock_t s = clock64();
    while (CHECK_IDS(*flag, signal_id)) {
      if (clock64() - s > TIMEOUT) {
        printf("%d from %d] pushrecv: expected %d, stuck with %d\n", myrank, peer, signal_id,
               *flag);
        return;
      }
    }
  }
}

__global__ void __launch_bounds__(MAX_THREADS)
    kuserbuffers_pushsendrecv_atomic(int *send_id, int *send_flagptr, int4 *srcptr, int4 *dstptr,
                                     const int lines, int myrank, int peer, int *recv_id,
                                     int *recv_flagptr, int adder, void *counters) {
  if (lines) {
    const int start_elem = threadIdx.x + blockDim.x * blockIdx.x;
    const int end_elem = lines;
    const int aligned_elem =
        ((end_elem - start_elem) & (~(blockDim.x * gridDim.x * UNROLLCOPY - 1)));
    const int end_aligned = start_elem + aligned_elem;
    if (end_elem > start_elem) {
      for (int line = start_elem; line < end_aligned; line += blockDim.x * gridDim.x * UNROLLCOPY) {
        int4 val[UNROLLCOPY];
#pragma unroll
        for (int i = 0; i < UNROLLCOPY; i++) {
          val[i] = srcptr[line + i * blockDim.x * gridDim.x];
        }
#pragma unroll
        for (int i = 0; i < UNROLLCOPY; i++) {
          dstptr[line + i * blockDim.x * gridDim.x] = val[i];
        }
      }
      for (int line = end_aligned; line < end_elem; line += blockDim.x * gridDim.x) {
        dstptr[line] = srcptr[line];
      }
    }
    __syncthreads();
    if (threadIdx.x)
      return;
    __threadfence_system();
    atomicAdd_system(send_flagptr,
                     1);  // otherwise need local SM sync before sending flag
  } else {                // 0 bytes and 1 SM only
    atomicAdd_system(send_flagptr, 1);
  }

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const int signal_id = (*recv_id) + adder;
    *recv_id = signal_id;
    volatile int *flag = (volatile int *)recv_flagptr;
    // if(*flag>=signal_id) return;
    clock_t s = clock64();
    while (CHECK_IDS(*flag, signal_id)) {
      if (clock64() - s > TIMEOUT) {
        printf("%d from %d] pushrecv: expected %d, stuck with %d\n", myrank, peer, signal_id,
               *flag); /*return;*/
      }
    }

    // Decrement atomic val to signal current output tile finish
    if (counters) {
      ((unsigned int *)counters)[0] = 0;
      asm volatile("fence.sc.gpu;\n");
    }
  }
}

__global__ void __launch_bounds__(MAX_THREADS)
    kuserbuffers_pushsendrecv_multiatomic(int *send_id, int *send_flagptr, int4 *srcptr,
                                          int4 *dstptr, const int lines, int myrank, int peer,
                                          int *recv_id, int *recv_flagptr, int adder,
                                          void *counters, int nchunks, int send_stride,
                                          int recv_stride, bool shuffle) {
  for (int chunk_i = 0; chunk_i < nchunks - 1; chunk_i++) {
    int send_chunk_id = shuffle ? chunk_i : (nchunks + myrank - chunk_i) % nchunks;
    int recv_chunk_id = shuffle ? chunk_i + 1 : (nchunks + myrank - chunk_i - 1) % nchunks;
    int send_offset = (send_chunk_id * send_stride) / 16;
    int recv_offset = ((shuffle ? recv_chunk_id : send_chunk_id) * recv_stride) / 16;

    if (lines) {
      const int start_elem = threadIdx.x + blockDim.x * blockIdx.x;
      const int end_elem = lines;
      const int aligned_elem =
          ((end_elem - start_elem) & (~(blockDim.x * gridDim.x * UNROLLCOPY - 1)));
      const int end_aligned = start_elem + aligned_elem;
      if (end_elem > start_elem) {
        for (int line = start_elem; line < end_aligned;
             line += blockDim.x * gridDim.x * UNROLLCOPY) {
          int4 val[UNROLLCOPY];
#pragma unroll
          for (int i = 0; i < UNROLLCOPY; i++) {
            val[i] = srcptr[send_offset + line + i * blockDim.x * gridDim.x];
          }
#pragma unroll
          for (int i = 0; i < UNROLLCOPY; i++) {
            dstptr[recv_offset + line + i * blockDim.x * gridDim.x] = val[i];
          }
        }
        for (int line = end_aligned; line < end_elem; line += blockDim.x * gridDim.x) {
          dstptr[recv_offset + line] = srcptr[send_offset + line];
        }
      }
      __syncthreads();
      if (!threadIdx.x) {
        __threadfence_system();
        atomicAdd_system(send_flagptr,
                         1);  // otherwise need local SM sync before sending flag
      }
    } else {  // 0 bytes and 1 SM only
      atomicAdd_system(send_flagptr, 1);
    }

    // wait for message to arrive.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      const int signal_id = (*recv_id) + adder;
      *recv_id = signal_id;
      volatile int *flag = (volatile int *)recv_flagptr;
      // if(*flag>=signal_id) return;
      clock_t s = clock64();
      while (CHECK_IDS(*flag, signal_id)) {
        if (clock64() - s > TIMEOUT) {
          printf("%d from %d] pushrecv: expected %d, stuck with %d\n", myrank, peer, signal_id,
                 *flag); /*return;*/
        }
      }
    }

    // Producer must update counters.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      // Decrement atomic val to signal current output tile finish
      if (counters) {
        ((unsigned int *)counters)[recv_chunk_id /*chunk_i+1*/] = 0;
        asm volatile("fence.sc.gpu;\n");
      }
    }

    // sync all CTAs before moving to next chunk.
    if (threadIdx.x == 0) {
      int old_val2;
      atomicInc(((unsigned int *)counters) + nchunks + chunk_i, gridDim.x - 1);
      while (0 != (old_val2 = atomicCAS(((unsigned int *)counters) + nchunks + chunk_i, 0, 0))) {
      }
    }
    __syncthreads();
  }
}

#define CUDACHECK(cmd)                                                                             \
  do {                                                                                             \
    cudaError_t e = cmd;                                                                           \
    if (e != cudaSuccess) {                                                                        \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));        \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0)

#define INTRANODE(peer) ((peer / comm->nvsize) == (comm->myrank / comm->nvsize))

void userbuffers_send(const int srchandler, const size_t srcoffset, const int dsthandler,
                      const size_t dstoffset, const size_t bytes, communicator *comm,
                      const int peer, cudaStream_t stream) {
  int peerlocal = peer % comm->nvsize;
  void *flagptr =
      (comm->peer_ptr[0][peerlocal]) +
      ((NVTE_REG0_OFFSET(comm) + NVTE_REG0_RECV + comm->myrank * NVTE_MAX_REGIONS + dsthandler) *
       sizeof(int));
  bool signalonly = (bytes / 16 == 0) || (comm->use_ce != 0);
  bool intranode = INTRANODE(peer);
  if (!intranode && (comm->launch_mode & NVTE_LAUNCH_CPU)) {
    comm->fifo[comm->head].optype = userbuffers_sendop;
    comm->fifo[comm->head].basecounter = comm->basecounter[userbuffers_sendop];
    comm->fifo[comm->head].handler = srchandler;
    comm->fifo[comm->head].offset = srcoffset;
    comm->fifo[comm->head].handler2 = dsthandler;
    comm->fifo[comm->head].offset2 = dstoffset;
    comm->fifo[comm->head].elements = bytes;
    comm->fifo[comm->head].peer = peer;

    int newhead = (comm->head + 1) & (NVTE_MAX_REQUESTS - 1);
    while (newhead == comm->tail) {
    }
    comm->head = newhead;
    comm->basecounter[userbuffers_sendop] += 1;
  }
  if (!intranode && (comm->launch_mode & NVTE_LAUNCH_GPU)) {
    kuserbuffers_proxysend<<<1, 1, 0, stream>>>(&(comm->flags[NVTE_GF_STATE + userbuffers_sendop]),
                                                comm->hostflags + userbuffers_sendop);
    return;
  }
  if (!(comm->launch_mode & NVTE_LAUNCH_GPU))
    return;
  if (comm->push == 0) {
    kuserbuffers_pullsend<<<1, 1, 0, stream>>>(comm->myrank, peer, &(comm->send_id[peer]),
                                               reinterpret_cast<int *>(flagptr));
  } else {
    void *srcptr = (comm->mem_ptr[srchandler]) + srcoffset;
    void *dstptr = (comm->peer_ptr[dsthandler][peerlocal]) + dstoffset;

    if (comm->use_ce)
      CUDACHECK(cudaMemcpyAsync(dstptr, srcptr, bytes, cudaMemcpyDeviceToDevice, stream));
    SETUP_LAUNCH_CONFIG(signalonly ? 1 : comm->sms, signalonly ? 1 : 1024, stream);
    int *arg1 = &comm->send_id[peer], *arg2 = reinterpret_cast<int *>(flagptr);
    int4 *arg3 = reinterpret_cast<int4 *>(srcptr), *arg4 = reinterpret_cast<int4 *>(dstptr);
    int arg5 = signalonly ? 0 : bytes / 16;
    void *kernelArgs[] = {reinterpret_cast<void *>(&arg1), reinterpret_cast<void *>(&arg2),
                          reinterpret_cast<void *>(&arg3), reinterpret_cast<void *>(&arg4),
                          reinterpret_cast<void *>(&arg5)};
    CUDACHECK(
        cudaLaunchKernelExC(&cfg, reinterpret_cast<void *>(kuserbuffers_pushsend), kernelArgs));
  }
}

void userbuffers_sendrecv(const int srchandler, const int dsthandler, const size_t send_offset,
                          const size_t recv_offset, const size_t bytes, communicator *comm,
                          const int send_peer, const int recv_peer, cudaStream_t stream) {
  bool signalonly = (bytes / 16 == 0) || (comm->use_ce != 0);
  int send_peerlocal = send_peer % comm->nvsize;
  int recv_peerlocal = recv_peer % comm->nvsize;
  void *flagptr_send =
      (comm->peer_ptr[0][send_peerlocal]) +
      ((NVTE_REG0_OFFSET(comm) + NVTE_REG0_RECV + comm->myrank * NVTE_MAX_REGIONS + dsthandler) *
       sizeof(int));
  void *flagptr_recv =
      (comm->mem_ptr[0]) +
      ((NVTE_REG0_OFFSET(comm) + NVTE_REG0_RECV + recv_peer * NVTE_MAX_REGIONS + dsthandler) *
       sizeof(int));

  void *send_srcptr = (comm->mem_ptr[srchandler]) + send_offset;
  void *send_dstptr = (comm->peer_ptr[dsthandler][send_peerlocal]) + send_offset;
  if (comm->use_ce)
    CUDACHECK(cudaMemcpyAsync(send_dstptr, send_srcptr, bytes, cudaMemcpyDeviceToDevice, stream));
  SETUP_LAUNCH_CONFIG(signalonly ? 1 : comm->sms, signalonly ? 1 : 1024, stream);

  int *arg1 = &comm->send_id[send_peer];
  int *arg2 = reinterpret_cast<int *>(flagptr_send);
  int4 *arg3 = reinterpret_cast<int4 *>(send_srcptr);
  int4 *arg4 = reinterpret_cast<int4 *>(send_dstptr);
  int arg5 = signalonly ? 0 : bytes / 16;
  int arg6 = comm->myrank;
  int arg7 = recv_peer;
  int *arg8 = &comm->recv_id[recv_peer * NVTE_MAX_REGIONS + dsthandler];
  int *arg9 = reinterpret_cast<int *>(flagptr_recv);
  int arg10 = signalonly ? 1 : comm->sms;
  void *kernelArgs[] = {reinterpret_cast<void *>(&arg1), reinterpret_cast<void *>(&arg2),
                        reinterpret_cast<void *>(&arg3), reinterpret_cast<void *>(&arg4),
                        reinterpret_cast<void *>(&arg5), reinterpret_cast<void *>(&arg6),
                        reinterpret_cast<void *>(&arg7), reinterpret_cast<void *>(&arg8),
                        reinterpret_cast<void *>(&arg9), reinterpret_cast<void *>(&arg10)};
  CUDACHECK(
      cudaLaunchKernelExC(&cfg, reinterpret_cast<void *>(kuserbuffers_pushsendrecv), kernelArgs));
  //}
}

void userbuffers_sendrecv_atomic(const int srchandler, const int dsthandler,
                                 const size_t send_offset, const size_t recv_offset,
                                 const size_t bytes, communicator *comm, const int send_peer,
                                 const int recv_peer, void *counters, cudaStream_t stream) {
  assert(comm->push && comm->use_ce == 0);
  bool signalonly = (bytes / 16 == 0) || (comm->use_ce != 0);

  int send_peerlocal = send_peer % comm->nvsize;
  int recv_peerlocal = recv_peer % comm->nvsize;
  void *flagptr_send =
      (comm->peer_ptr[0][send_peerlocal]) +
      ((NVTE_REG0_OFFSET(comm) + NVTE_REG0_RECV + comm->myrank * NVTE_MAX_REGIONS + dsthandler) *
       sizeof(int));
  void *flagptr_recv =
      (comm->mem_ptr[0]) +
      ((NVTE_REG0_OFFSET(comm) + NVTE_REG0_RECV + recv_peer * NVTE_MAX_REGIONS + dsthandler) *
       sizeof(int));

  void *send_srcptr = (comm->mem_ptr[srchandler]) + send_offset;
  void *send_dstptr = (comm->peer_ptr[dsthandler][send_peerlocal]) + send_offset;
  if (comm->use_ce) {
    CUDACHECK(cudaMemcpyAsync(send_dstptr, send_srcptr, bytes, cudaMemcpyDeviceToDevice, stream));
  }
  SETUP_LAUNCH_CONFIG(signalonly ? 1 : comm->sms, signalonly ? 1 : 1024, stream);

  int *arg1 = &comm->send_id[send_peer];
  int *arg2 = reinterpret_cast<int *>(flagptr_send);
  int4 *arg3 = reinterpret_cast<int4 *>(send_srcptr);
  int4 *arg4 = reinterpret_cast<int4 *>(send_dstptr);
  int arg5 = signalonly ? 0 : bytes / 16;
  int arg6 = comm->myrank;
  int arg7 = recv_peer;
  int *arg8 = &comm->recv_id[recv_peer * NVTE_MAX_REGIONS + dsthandler];
  int *arg9 = reinterpret_cast<int *>(flagptr_recv);
  int arg10 = signalonly ? 1 : comm->sms;
  void *arg11 = counters;
  void *kernelArgs[] = {reinterpret_cast<void *>(&arg1), reinterpret_cast<void *>(&arg2),
                        reinterpret_cast<void *>(&arg3), reinterpret_cast<void *>(&arg4),
                        reinterpret_cast<void *>(&arg5), reinterpret_cast<void *>(&arg6),
                        reinterpret_cast<void *>(&arg7), reinterpret_cast<void *>(&arg8),
                        reinterpret_cast<void *>(&arg9), reinterpret_cast<void *>(&arg10),
                        reinterpret_cast<void *>(&arg11)};
  CUDACHECK(cudaLaunchKernelExC(&cfg, reinterpret_cast<void *>(kuserbuffers_pushsendrecv_atomic),
                                kernelArgs));
}

void userbuffers_sendrecv_multiatomic(const int srchandler, const int dsthandler,
                                      const size_t send_stride, const size_t recv_stride,
                                      const size_t bytes, communicator *comm, const int send_peer,
                                      const int recv_peer, const int nchunks, void *counters,
                                      bool shuffle, cudaStream_t stream) {
  assert(comm->push && comm->use_ce == 0);

  int send_peerlocal = send_peer % comm->nvsize;
  int recv_peerlocal = recv_peer % comm->nvsize;
  void *flagptr_send =
      (comm->peer_ptr[0][send_peerlocal]) +
      ((NVTE_REG0_OFFSET(comm) + NVTE_REG0_RECV + comm->myrank * NVTE_MAX_REGIONS + dsthandler) *
       sizeof(int));
  void *flagptr_recv =
      (comm->mem_ptr[0]) +
      ((NVTE_REG0_OFFSET(comm) + NVTE_REG0_RECV + recv_peer * NVTE_MAX_REGIONS + dsthandler) *
       sizeof(int));

  SETUP_LAUNCH_CONFIG(comm->sms, 1024, stream);

  int *arg1 = &comm->send_id[send_peer];
  int *arg2 = reinterpret_cast<int *>(flagptr_send);
  int4 *arg3 = reinterpret_cast<int4 *>((comm->mem_ptr[srchandler]));
  int4 *arg4 = reinterpret_cast<int4 *>((comm->peer_ptr[dsthandler][send_peerlocal]));
  int arg5 = bytes / 16;
  int arg6 = comm->myrank;
  int arg7 = recv_peer;
  int *arg8 = &comm->recv_id[recv_peer * NVTE_MAX_REGIONS + dsthandler];
  int *arg9 = reinterpret_cast<int *>(flagptr_recv);
  int arg10 = comm->sms;
  void *arg11 = counters;
  int arg12 = nchunks;
  int arg13 = send_stride;
  int arg14 = recv_stride;
  bool arg15 = shuffle;
  void *kernelArgs[] = {reinterpret_cast<void *>(&arg1),  reinterpret_cast<void *>(&arg2),
                        reinterpret_cast<void *>(&arg3),  reinterpret_cast<void *>(&arg4),
                        reinterpret_cast<void *>(&arg5),  reinterpret_cast<void *>(&arg6),
                        reinterpret_cast<void *>(&arg7),  reinterpret_cast<void *>(&arg8),
                        reinterpret_cast<void *>(&arg9),  reinterpret_cast<void *>(&arg10),
                        reinterpret_cast<void *>(&arg11), reinterpret_cast<void *>(&arg12),
                        reinterpret_cast<void *>(&arg13), reinterpret_cast<void *>(&arg14),
                        reinterpret_cast<void *>(&arg15)};
  CUDACHECK(cudaLaunchKernelExC(
      &cfg, reinterpret_cast<void *>(kuserbuffers_pushsendrecv_multiatomic), kernelArgs));
}

__global__ void __launch_bounds__(MAX_THREADS)
    kuserbuffers_alltoall(void **baseflagptrs, int flagoffset, int4 *basesrcptr, void **dstptrs,
                          size_t dstoffset, const int lines, const int myrank) {
  if (blockIdx.x == myrank)
    return;
  int4 *dstptr = reinterpret_cast<int4 *>(dstptrs[blockIdx.x] + dstoffset);
  int *flagptr = reinterpret_cast<int *>(baseflagptrs[blockIdx.x] + flagoffset);
  const size_t myblockoffset = blockIdx.x * lines;
  int4 *srcptr = basesrcptr + myblockoffset;
  dstptr += myblockoffset;

  if (lines) {
    const int start_elem = threadIdx.x;
    const int end_elem = lines;
    const int aligned_elem = ((end_elem - start_elem) & (~(blockDim.x * UNROLLCOPY - 1)));
    const int end_aligned = start_elem + aligned_elem;
    if (end_elem > start_elem) {
      for (int line = start_elem; line < end_aligned; line += blockDim.x * UNROLLCOPY) {
        int4 val[UNROLLCOPY];
#pragma unroll
        for (int i = 0; i < UNROLLCOPY; i++)
          val[i] = srcptr[line + i * blockDim.x];
#pragma unroll
        for (int i = 0; i < UNROLLCOPY; i++)
          dstptr[line + i * blockDim.x] = val[i];
      }
      for (int line = end_aligned; line < end_elem; line += blockDim.x)
        dstptr[line] = srcptr[line];
    }
    __syncthreads();
    if (threadIdx.x)
      return;
    __threadfence_system();
    atomicAdd(flagptr, 1);

  } else {
    atomicAdd(flagptr, 1);
  }
}

void userbuffers_alltoall_send(const int srchandler, const size_t srcoffset, const int dsthandler,
                               const size_t dstoffset, const size_t bytes, communicator *comm,
                               cudaStream_t stream) {
  if (comm->launch_mode & NVTE_LAUNCH_CPU) {
    comm->fifo[comm->head].optype = userbuffers_alltoall;
    comm->fifo[comm->head].basecounter = comm->basecounter[userbuffers_alltoall];
    comm->fifo[comm->head].handler = srchandler;
    comm->fifo[comm->head].offset = srcoffset;
    comm->fifo[comm->head].handler2 = dsthandler;
    comm->fifo[comm->head].offset2 = dstoffset;
    comm->fifo[comm->head].elements = bytes;

    int newhead = (comm->head + 1) & (NVTE_MAX_REQUESTS - 1);
    while (newhead == comm->tail) {
    }
    comm->head = newhead;
    comm->basecounter[userbuffers_alltoall] += 1;
  }
  if (comm->launch_mode & NVTE_LAUNCH_GPU)
    kuserbuffers_proxysend<<<1, 1, 0, stream>>>(
        &(comm->flags[NVTE_GF_STATE + userbuffers_alltoall]),
        comm->hostflags + userbuffers_alltoall);
}

void userbuffers_recv(const int srchandler, const size_t srcoffset, const int dsthandler,
                      const size_t dstoffset, const size_t bytes, communicator *comm,
                      const int peer, cudaStream_t stream) {
  int peerlocal = peer % comm->nvsize;
  void *flagptr =
      (comm->mem_ptr[0]) +
      ((NVTE_REG0_OFFSET(comm) + NVTE_REG0_RECV + peer * NVTE_MAX_REGIONS + dsthandler) *
       sizeof(int));
  bool signalonly = (bytes / 16 == 0) || (comm->use_ce != 0);
  bool intranode = INTRANODE(peer);
  if (!(comm->launch_mode & NVTE_LAUNCH_GPU))
    return;
  if (comm->push == 0 && intranode) {
    void *dstptr = (comm->mem_ptr[dsthandler]) + dstoffset;
    void *srcptr = (comm->peer_ptr[srchandler][peerlocal]) + srcoffset;

    kuserbuffers_pullrecv<<<signalonly ? 1 : comm->sms, signalonly ? 1 : 1024, 0, stream>>>(
        comm->myrank, peer, &(comm->recv_id[peer * NVTE_MAX_REGIONS + dsthandler]),
        reinterpret_cast<int *>(flagptr), reinterpret_cast<int4 *>(srcptr),
        reinterpret_cast<int4 *>(dstptr), signalonly ? 0 : bytes / 16);
    if (!signalonly)
      kuserbuffers_inc<<<1, 1, 0, stream>>>(&(comm->recv_id[peer * NVTE_MAX_REGIONS + dsthandler]));
    if (comm->use_ce) {
      CUDACHECK(cudaMemcpyAsync(dstptr, srcptr, bytes, cudaMemcpyDeviceToDevice, stream));
    }
  } else {
    kuserbuffers_pushrecv<<<1, 1, 0, stream>>>(
        comm->myrank, peer, &comm->recv_id[peer * NVTE_MAX_REGIONS + dsthandler],
        reinterpret_cast<int *>(flagptr), signalonly || !intranode ? 1 : comm->sms);
  }
}

void userbuffers_alltoall_recv(communicator *comm, cudaStream_t stream) {
  void *flagptr =
      (comm->mem_ptr[0]) +
      ((NVTE_REG0_OFFSET(comm) + NVTE_REG0_OPFLAGS * userbuffers_alltoall) * sizeof(int));

  if (!(comm->launch_mode & NVTE_LAUNCH_GPU))
    return;
  kuserbuffers_pushrecv<<<1, 1, 0, stream>>>(comm->myrank, -1, reinterpret_cast<int *>(flagptr + 4),
                                             reinterpret_cast<int *>(flagptr), comm->nranks - 1);
}

// producer
static __global__ void producer_kernel(void *atomic_ptr, int chunk_i) {
  // Decrement atomic val to signal current output tile finish
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    ((unsigned int *)atomic_ptr)[chunk_i] = 0;
  }

  // COMM kernel need to explicitely flash gmem.
  // GEMM kernel already executed, and can not see gmem
  // change without COMM kernel explicitely make change
  asm volatile("fence.sc.gpu;\n");
}

// consumer
static __global__ void consumer_kernel(void *atomic_ptr, int chunk_i) {
  // Wait for producer to change the val to 0, which signal producer ready
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int old_val;
    while (0 != (old_val = atomicCAS((unsigned int *)atomic_ptr + chunk_i, 0, 0))) {
    }
    ((unsigned int *)atomic_ptr)[chunk_i] = 1;
    asm volatile("fence.sc.gpu;\n");
  }
}

// consumer_batch
static __global__ void consumer_batch_kernel(void *atomic_ptr, int first_chunk_i, int num_chunks) {
  // Wait for producer to change the val to 0, which signal producer ready
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int old_val;
    for (int i = first_chunk_i; i < num_chunks; i++) {
      while (0 != (old_val = atomicCAS((unsigned int *)atomic_ptr + i, 0, 0))) {
      }
      ((unsigned int *)atomic_ptr)[i] = 1;
      asm volatile("fence.sc.gpu;\n");
    }
  }
}

void producer(void *atomic_ptr, int chunk_i, cudaStream_t stream) {
  dim3 block(1);
  dim3 grid(1);
  producer_kernel<<<grid, block, 0, stream>>>(atomic_ptr, chunk_i);
}

void consumer(void *atomic_ptr, int chunk_i, cudaStream_t stream) {
  dim3 block(1);
  dim3 grid(1);
  consumer_kernel<<<grid, block, 0, stream>>>(atomic_ptr, chunk_i);
}

void consumer_batch(void *atomic_ptr, int first_chunk_i, int num_chunks, cudaStream_t stream) {
  dim3 block(1);
  dim3 grid(1);
  consumer_batch_kernel<<<grid, block, 0, stream>>>(atomic_ptr, first_chunk_i, num_chunks);
}

template <typename fp8type>
__global__ void __launch_bounds__(MAX_THREADS / 4)
reduce_fp8_in_bf16_out_cuda(void *inputs, void *output, const float *scale,
                            const int num_inputs, const int input_size) {
  const size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  fp8type *inputs_fp8 = reinterpret_cast<fp8type *>(inputs);
  float accum_buf = static_cast<float>(inputs_fp8[tid]) * (*scale);
  #pragma unroll
  for (int i = 1; i < num_inputs; i++) {
    accum_buf += static_cast<float>(inputs_fp8[tid + input_size * i]) * (*scale);
  }
  half *output_half = reinterpret_cast<half *>(output);
  output_half[tid] = (half) accum_buf;
}

template <typename fp8type>
void reduce_fp8_in_bf16_out(void *inputs, void *output, float *scale, int num_inputs,
                            int input_size, cudaStream_t stream) {
  size_t num_threads = MAX_THREADS / 4;
  size_t num_blocks = (input_size +num_threads - 1) / num_threads;
  dim3 block(num_threads);
  dim3 grid(num_blocks);
  reduce_fp8_in_bf16_out_cuda<fp8type><<<grid, block, 0, stream>>>(
    inputs, output, scale, num_inputs, input_size);
}

template void reduce_fp8_in_bf16_out<__nv_fp8_e4m3>(
  void *inputs, void *output, float *scale, int num_inputs, int input_size, cudaStream_t stream);
template void reduce_fp8_in_bf16_out<__nv_fp8_e5m2>(
  void *inputs, void *output, float *scale, int num_inputs, int input_size, cudaStream_t stream);
