/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda.h>
#include <cuda_runtime.h>

#if __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#define half nv_bfloat16
#else
#include <cuda_fp16.h>
#endif

#include <assert.h>
#include <cuda_fp8.h>
#include <stdio.h>
#include <unistd.h>

#include "userbuffers.h"

#define MAX_THREADS 1024

#define CUDACHECK(cmd)                                                                      \
  do {                                                                                      \
    cudaError_t e = cmd;                                                                    \
    if (e != cudaSuccess) {                                                                 \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(EXIT_FAILURE);                                                                   \
    }                                                                                       \
  } while (0)

#define ATOMIC_CONSUMER(chunk)                                             \
  if (counters) {                                                          \
    if (threadIdx.x == 0 && blockIdx.x == 0) {                             \
      while (0 != (atomicCAS(((unsigned int *)counters) + chunk, 0, 0))) { \
      }                                                                    \
      ((unsigned int *)counters)[chunk] = 1;                               \
      asm volatile("fence.sc.gpu;\n");                                     \
    }                                                                      \
    if (blockIdx.x == 0) __syncthreads();                                  \
  }

#define ATOMIC_PRODUCER(chunk)             \
  if (counters) {                          \
    ((unsigned int *)counters)[chunk] = 0; \
  }

// Return true if producer > consumer, otherwise false while preventing integer overflow
// If we expect that producer will be 2B+ messages behind consumer
#define CHECK_IDS(producer, consumer) (((unsigned)(producer) - (unsigned)(consumer)) & (~INT_MAX))

// Strip the path from a full filename
#define FILENAME(file)                                      \
  ({                                                        \
    const char *filename = file;                            \
    const char *basename = filename;                        \
    for (const char *ptr = filename; *ptr != '\0'; ptr++) { \
      if (*ptr == '/' || *ptr == '\\') {                    \
        basename = ptr + 1;                                 \
      }                                                     \
    }                                                       \
    basename;                                               \
  })

// Printf to provide enough information so it is easier to attribute failures
#define UB_PRINT(message, ...) \
  printf("[%s:%s:%d] " message "\n", FILENAME(__FILE__), __FUNCTION__, __LINE__, __VA_ARGS__)

// Report and error on timeout
#define CHECK_TIMEOUT(t, timeout) ((clock64() - (t)) > timeout)

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_rw(const int op, const int flagoffset, const int firstrank,
                                        const int myrank, const int gpustep, const int lineoffset,
                                        const int numlines, void **commbuff, const int handleridx,
                                        const uint64_t ub_timeout) {
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
      if (CHECK_TIMEOUT(s, ub_timeout)) {
        UB_PRINT("[%d] Allreduce reduce-scatter: SM %d [%d]: expecting %d got %d", myrank,
                 blockIdx.x, threadIdx.x, reduce_id, *flag);
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
    while (CHECK_IDS(*flag, reduce_id)) {
      if (CHECK_TIMEOUT(s, ub_timeout)) {
        UB_PRINT("[%d] Allreduce Gather: SM %d [%d]: expecting %d got %d", myrank, blockIdx.x,
                 threadIdx.x, reduce_id, *flag);
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
                                        const int numlines, void **commbuff, const int handleridx,
                                        const uint64_t ub_timeout) {
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
      if (CHECK_TIMEOUT(s, ub_timeout)) {
        UB_PRINT("[%d ]Allreduce reduce-scatter:SM %d [%d]: expecting %d got %d", myrank,
                 blockIdx.x, threadIdx.x, reduce_id, *flag);
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
  __syncthreads();
  if (threadIdx.x == 0) __threadfence();
  __syncthreads();

  if (threadIdx.x < RANKS) {
    flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&myptr[targetgpu];
    clock_t s = clock64();
    while (CHECK_IDS(*flag, reduce_id)) {
      if (CHECK_TIMEOUT(s, ub_timeout)) {
        UB_PRINT("[%d] Allreduce gather: SM %d [%d]: expecting %d got %d", myrank, blockIdx.x,
                 threadIdx.x, reduce_id, *flag);
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
  if (threadIdx.x == 0 && blockIdx.x == 0) *reduceidptr = reduce_id;
}  // fp16 inplace reduce kernel (Ampere)

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_rr_rs(const int op, const int flagoffset, const int firstrank,
                                           const int myrank, const int gpustep,
                                           const int mylineoffset, const int totallines,
                                           void **commbuff, const int handleridx,
                                           const uint64_t ub_timeout) {
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
    if (blockIdx.x == 0) flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&(myptr[targetgpu]);
    userptr[threadIdx.x] = reinterpret_cast<int4 *>(commbuff[targetgpu + handleridx]);
    clock_t s = clock64();
    while (CHECK_IDS(*flag, reduce_id)) {
      if (CHECK_TIMEOUT(s, ub_timeout)) {
        UB_PRINT("[%d] Reduce-scatter: SM %d [%d]: expecting %d got %d", myrank, blockIdx.x,
                 threadIdx.x, reduce_id, *flag);
        break;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const int adder = blockIdx.x == 0 ? NVTE_MAX_SMS - gridDim.x + 1 : 1;
    int old_val = atomicAdd(myptr + (NVTE_MAX_NVLINK * 2), adder);
    if (old_val + adder == NVTE_MAX_SMS * reduce_id) lastSM = 1;
  }

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

  if (threadIdx.x == 0 && lastSM) *reduceidptr = reduce_id;
}  // fp16 inplace reduce-scatter kernel

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS) userbuffers_fp16_sum_inplace_gpu_rr_rs_oop(
    const int op, const int flagoffset, const int firstrank, const int myrank, const int gpustep,
    const int mylineoffset, const int totallines, const int rowlines, const int skiplines,
    void **commbuff, const int handleridx, void *outbuf, const uint64_t ub_timeout) {
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
    if (blockIdx.x == 0) flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&(myptr[targetgpu]);
    userptr[threadIdx.x] = reinterpret_cast<int4 *>(commbuff[targetgpu + handleridx]);
    clock_t s = clock64();
    while (CHECK_IDS(*flag, reduce_id)) {
      if (CHECK_TIMEOUT(s, ub_timeout)) {
        UB_PRINT("[%d] Reduce-scatter: SM %d [%d]: expecting %d got %d", myrank, blockIdx.x,
                 threadIdx.x, reduce_id, *flag);
        break;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const int adder = blockIdx.x == 0 ? NVTE_MAX_SMS - gridDim.x + 1 : 1;
    int old_val = atomicAdd(myptr + (NVTE_MAX_NVLINK * 2), adder);
    if (old_val + adder == NVTE_MAX_SMS * reduce_id) lastSM = 1;
  }

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

    (reinterpret_cast<int4 *>(outbuf))[(line / rowlines) * skiplines + (line % rowlines)] = sum;
  }

  if (threadIdx.x == 0 && lastSM) *reduceidptr = reduce_id;
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
        UB_PRINT("Reduce-scatter: SM %d [%d]: expecting %d got %d", blockIdx.x, threadIdx.x,
                 reduce_id, *flag);
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
  if (threadIdx.x == 0) __threadfence_system();
  __syncthreads();

  if (threadIdx.x < RANKS) {
    flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&myptr[targetgpu];
    clock_t s = clock64();
    while (CHECK_IDS(*flag, reduce_id)) {
      if (clock64() - s > 2ull * TIMEOUT) {
        UB_PRINT("Allgather: SM %d [%d]: expecting %d got %d", blockIdx.x, threadIdx.x, reduce_id,
                 *flag);
        break;
      }
    }
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) *reduceidptr = reduce_id;
}  // fp16 inplace reduce kernel (Hopper) MC

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_mc_rs(const int op, const int flagoffset, const int firstrank,
                                           const int myrank, const int gpustep,
                                           const int mylineoffset, const int totallines,
                                           void **commbuff, const int handleridx, float4 *mc_ptr,
                                           const uint64_t ub_timeout) {
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
    if (blockIdx.x == 0) flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&(myptr[targetgpu]);
    clock_t s = clock64();
    while (CHECK_IDS(*flag, reduce_id)) {
      if (CHECK_TIMEOUT(s, ub_timeout)) {
        UB_PRINT("[%d] Reduce-scatter: SM %d [%d]: expecting %d got %d", myrank, blockIdx.x,
                 threadIdx.x, reduce_id, *flag);
        break;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const int adder = blockIdx.x == 0 ? NVTE_MAX_SMS - gridDim.x + 1 : 1;
    int old_val = atomicAdd(myptr + (NVTE_MAX_NVLINK * 2), adder);
    if (old_val + adder == NVTE_MAX_SMS * reduce_id) lastSM = 1;
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
    for (int i = 0; i < UNROLL_MC; i++) localptr[mylineoffset + line + i * loop_step0] = val[i];
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

  if (threadIdx.x == 0 && lastSM) *reduceidptr = reduce_id;
}  // fp16 inplace reduce-scatter kernel MC

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_mc_rs_oop(const int op, const int flagoffset,
                                               const int firstrank, const int myrank,
                                               const int gpustep, const int mylineoffset,
                                               const int totallines, const int rowlines,
                                               const int skiplines, void **commbuff,
                                               const int handleridx, void *outbuf, float4 *mc_ptr,
                                               const uint64_t ub_timeout) {
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
    if (blockIdx.x == 0) flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&(myptr[targetgpu]);
    clock_t s = clock64();
    while (CHECK_IDS(*flag, reduce_id)) {
      if (CHECK_TIMEOUT(s, ub_timeout)) {
        UB_PRINT("[%d] Reduce-scatter: SM %d [%d]: expecting %d got %d", myrank, blockIdx.x,
                 threadIdx.x, reduce_id, *flag);
        break;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const int adder = blockIdx.x == 0 ? NVTE_MAX_SMS - gridDim.x + 1 : 1;
    int old_val = atomicAdd(myptr + (NVTE_MAX_NVLINK * 2), adder);
    if (old_val + adder == NVTE_MAX_SMS * reduce_id) lastSM = 1;
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
    reinterpret_cast<uint4 *>(outbuf)[(line / rowlines) * skiplines + (line % rowlines)] = val;
  }

  if (threadIdx.x == 0 && lastSM) *reduceidptr = reduce_id;
}  // fp16 reduce-scatter kernel (out of place) fp16 MC

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_mc_ag(const int op, const int flagoffset, const int firstrank,
                                           const int myrank, const int gpustep,
                                           const int mylineoffset, const int totallines,
                                           void **commbuff, const int handleridx, uint4 *mc_ptr,
                                           const uint64_t ub_timeout) {
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
    for (int i = 0; i < UNROLL_MC; i++) val[i] = localptr[mylineoffset + line + i * loop_step0];
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
  if (threadIdx.x == 0) __threadfence_system();
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
    if (threadIdx.x == 0) *reduceidptr = reduce_id;
    flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&myptr[targetgpu];
    clock_t s = clock64();
    while (CHECK_IDS(*flag, reduce_id)) {
      if (CHECK_TIMEOUT(s, ub_timeout)) {
        UB_PRINT("[%d] Allgather: SM %d [%d]: expecting %d got %d", myrank, blockIdx.x, threadIdx.x,
                 reduce_id, *flag);
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
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_mc_rs_oop(const int op, const int flagoffset,
                                               const int firstrank, const int myrank,
                                               const int gpustep, const int mylineoffset,
                                               const int totallines, const int rowlines,
                                               const int skiplines, void **commbuff,
                                               const int handleridx, void *outbuf, float4 *mc_ptr,
                                               const uint64_t ub_timeout) {}

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_mc_ag(const int op, const int flagoffset, const int firstrank,
                                           const int myrank, const int gpustep,
                                           const int mylineoffset, const int totallines,
                                           void **commbuff, const int handleridx, uint4 *mc_ptr,
                                           const uint64_t ub_timeout) {}

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_mc_rs(const int op, const int flagoffset, const int firstrank,
                                           const int myrank, const int gpustep,
                                           const int mylineoffset, const int totallines,
                                           void **commbuff, const int handleridx, float4 *mc_ptr,
                                           const uint64_t ub_timeout) {}
#endif

template <int RANKS, typename fp8type>
__global__ void __launch_bounds__(MAX_THREADS) userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_fp8(
    const int op, const int flagoffset, const int firstrank, const int myrank, const int gpustep,
    const int mylineoffset, const int totallines, const int rowlines, const int skiplines,
    void **commbuff, const int handleridx, void *outbuf, float *scale, const uint64_t ub_timeout) {
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
    if (blockIdx.x == 0) flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&(myptr[targetgpu]);
    userptr[threadIdx.x] = reinterpret_cast<int4 *>(commbuff[targetgpu + handleridx]);
    clock_t s = clock64();
    while (CHECK_IDS(*flag, reduce_id)) {
      if (CHECK_TIMEOUT(s, ub_timeout)) {
        UB_PRINT("[%d] Reduce-scatter: SM %d [%d]: expecting %d got %d", myrank, blockIdx.x,
                 threadIdx.x, reduce_id, *flag);
        break;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const int adder = blockIdx.x == 0 ? NVTE_MAX_SMS - gridDim.x + 1 : 1;
    int old_val = atomicAdd(myptr + (NVTE_MAX_NVLINK * 2), adder);
    if (old_val + adder == NVTE_MAX_SMS * reduce_id) lastSM = 1;
  }
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
      val[i] = userptr[dest[i]][mylineoffset + line];
    }

    int4 sum[2] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
    half *s = reinterpret_cast<half *>(&sum);

#pragma unroll
    for (int i = 0; i < RANKS; i++) {
      fp8type *x = reinterpret_cast<fp8type *>(&val[i]);
#pragma unroll
      for (int j = 0; j < sizeof(int4) / sizeof(fp8type); j++) s[j] += hscale * (half)(x[j]);
    }
    int hline = 2 * line;
    (reinterpret_cast<int4 *>(outbuf))[(hline / rowlines) * skiplines + (hline % rowlines)] =
        sum[0];
    hline++;
    (reinterpret_cast<int4 *>(outbuf))[(hline / rowlines) * skiplines + (hline % rowlines)] =
        sum[1];
  }

  if (threadIdx.x == 0 && lastSM) *reduceidptr = reduce_id;
}  // fp16 reduce-scatter kernel (out of place) (fp8->fp16)

template <int RANKS, typename fp8type>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_atomic_fp8(
        const int op, const int flagoffset, const int firstrank, const int myrank,
        const int gpustep, const int mylineoffset, const int totallines, const int rowlines,
        const int skiplines_out, const int skiplines_in, void **commbuff, const int handleridx,
        void *outbuf, float *scale, void *counters, const int numchunks, const int atomicindex,
        const uint64_t ub_timeout) {
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
      if (blockIdx.x == 0) flagptr[physgpu] = reduce_id;
      volatile int *flag = (volatile int *)&(myptr[targetgpu]);
      userptr[threadIdx.x] = reinterpret_cast<int4 *>(commbuff[targetgpu + handleridx]);
      clock_t s = clock64();
      while (CHECK_IDS(*flag, reduce_id)) {
        if (CHECK_TIMEOUT(s, ub_timeout)) {
          UB_PRINT("[%d] Reduce-scatter: SM %d [%d]: expecting %d got %d", myrank, blockIdx.x,
                   threadIdx.x, reduce_id, *flag);
          break;
        }
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      const int adder = blockIdx.x == 0 ? NVTE_MAX_SMS - gridDim.x + 1 : 1;
      int old_val = atomicAdd(myptr + (NVTE_MAX_NVLINK * 2), /*numchunks * */ adder);
      if (old_val + adder == NVTE_MAX_SMS * (reduce_id /* + numchunks*/)) lastSM = 1;
    }

    int warp = blockIdx.x + (threadIdx.x >> 5);
    int dest[RANKS];
#pragma unroll
    for (int i = 0; i < RANKS; i++) dest[i] = (i + myrank + warp) & (RANKS - 1);

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
        for (int j = 0; j < sizeof(int4) / sizeof(fp8type); j++) s[j] += hscale * (half)(x[j]);
      }
      (reinterpret_cast<int4 *>(outbuf))[index1_out] = sum[0];
      (reinterpret_cast<int4 *>(outbuf))[index2_out] = sum[1];
    }
  }
  if (threadIdx.x == 0 && lastSM) *reduceidptr = reduce_id;
}  // fp16 reduce-scatter kernel (out of place) (fp8->fp16)

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS) userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_stride(
    const int op, const int flagoffset, const int firstrank, const int myrank, const int gpustep,
    const int mylineoffset, const int totallines, const int rowlines, const int skiplines,
    void **commbuff, const int handleridx, void *outbuf, const uint64_t ub_timeout) {
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
    if (blockIdx.x == 0) flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&(myptr[targetgpu]);
    userptr[threadIdx.x] = reinterpret_cast<int4 *>(commbuff[targetgpu + handleridx]);
    clock_t s = clock64();
    while (CHECK_IDS(*flag, reduce_id)) {
      if (CHECK_TIMEOUT(s, ub_timeout)) {
        UB_PRINT("[%d] Reduce-scatter: SM %d [%d]: expecting %d got %d", myrank, blockIdx.x,
                 threadIdx.x, reduce_id, *flag);
        break;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const int adder = blockIdx.x == 0 ? NVTE_MAX_SMS - gridDim.x + 1 : 1;
    int old_val = atomicAdd(myptr + (NVTE_MAX_NVLINK * 2), adder);
    if (old_val + adder == NVTE_MAX_SMS * reduce_id) lastSM = 1;
  }

  int warp = blockIdx.x + (threadIdx.x >> 5);
  int dest[RANKS];
#pragma unroll
  for (int i = 0; i < RANKS; i++) dest[i] = (i + myrank + warp) & (RANKS - 1);

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
      for (int j = 0; j < 8; j++) s[j] += x[j];
    }

    int index_out = (line / rowlines) * skiplines + (line % rowlines);
    (reinterpret_cast<int4 *>(outbuf))[index_out] = sum;
  }

  if (threadIdx.x == 0 && lastSM) *reduceidptr = reduce_id;
}  // fp16 reduce-scatter kernel (out of place) fp16

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_stride_atomic(
        const int op, const int flagoffset, const int firstrank, const int myrank,
        const int gpustep, const int mylineoffset, const int totallines, const int rowlines,
        const int skiplines, const int numchunks, void **commbuff, const int handleridx,
        void *outbuf, void *counters, const uint64_t ub_timeout) {
  if (counters) {
    if (threadIdx.x == 0) {
      // spin-lock on counter from producer
      while (0 != (atomicCAS(((unsigned int *)counters), 0, 0))) {
      }

      // make sure all threadblocks have read/waited on counters.
      atomicInc(((unsigned int *)counters) + numchunks, gridDim.x - 1);
      while (0 != (atomicCAS(((unsigned int *)counters) + numchunks, 0, 0))) {
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
    if (blockIdx.x == 0) flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&(myptr[targetgpu]);
    userptr[threadIdx.x] = reinterpret_cast<int4 *>(commbuff[targetgpu + handleridx]);
    clock_t s = clock64();
    while (CHECK_IDS(*flag, reduce_id)) {
      if (CHECK_TIMEOUT(s, ub_timeout)) {
        UB_PRINT("[%d] Reduce-scatter: SM %d [%d]: expecting %d got %d", myrank, blockIdx.x,
                 threadIdx.x, reduce_id, *flag);
        break;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const int adder = blockIdx.x == 0 ? NVTE_MAX_SMS - gridDim.x + 1 : 1;
    int old_val = atomicAdd(myptr + (NVTE_MAX_NVLINK * 2), adder);
    if (old_val + adder == NVTE_MAX_SMS * reduce_id) lastSM = 1;
  }

  int warp = blockIdx.x + (threadIdx.x >> 5);
  int dest[RANKS];
#pragma unroll
  for (int i = 0; i < RANKS; i++) dest[i] = (i + myrank + warp) & (RANKS - 1);

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
      for (int j = 0; j < 8; j++) s[j] += x[j];
    }

    int index_out = (line / rowlines) * skiplines + (line % rowlines);
    (reinterpret_cast<int4 *>(outbuf))[index_out] = sum;
  }

  if (threadIdx.x == 0 && lastSM) *reduceidptr = reduce_id;
}  // fp16 reduce-scatter kernel (out of place) fp16

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_stride_multiatomic(
        const int op, const int flagoffset, const int firstrank, const int myrank,
        const int gpustep, const int mylineoffset, const int totallines, const int rowlines,
        const int skiplines, const int numchunks, void **commbuff, const int handleridx,
        void *outbuf, void *counters, const uint64_t ub_timeout) {
  for (int chunk_i = 0; chunk_i < numchunks; chunk_i++) {
    if (counters) {
      if (threadIdx.x == 0) {
        // spin-lock on counter from producer
        while (0 != (atomicCAS(((unsigned int *)counters) + chunk_i, 0, 0))) {
        }

        // make sure all threadblocks have read/waited on counters.
        atomicInc(((unsigned int *)counters) + numchunks + chunk_i, gridDim.x - 1);
        while (0 != (atomicCAS(((unsigned int *)counters) + numchunks + chunk_i, 0, 0))) {
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
      if (blockIdx.x == 0) flagptr[physgpu] = reduce_id;
      volatile int *flag = (volatile int *)&(myptr[targetgpu]);
      userptr[threadIdx.x] = reinterpret_cast<int4 *>(commbuff[targetgpu + handleridx]);
      clock_t s = clock64();
      while (CHECK_IDS(*flag, reduce_id)) {
        if (CHECK_TIMEOUT(s, ub_timeout)) {
          UB_PRINT("[%d] Reduce-scatter: SM %d [%d]: expecting %d got %d", myrank, blockIdx.x,
                   threadIdx.x, reduce_id, *flag);
          break;
        }
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      const int adder = blockIdx.x == 0 ? NVTE_MAX_SMS - gridDim.x + 1 : 1;
      int old_val = atomicAdd(myptr + (NVTE_MAX_NVLINK * 2), adder);
      if (old_val + adder == NVTE_MAX_SMS * reduce_id) lastSM = 1;
    }

    int warp = blockIdx.x + (threadIdx.x >> 5);
    int dest[RANKS];
#pragma unroll
    for (int i = 0; i < RANKS; i++) dest[i] = (i + myrank + warp) & (RANKS - 1);

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
        for (int j = 0; j < 8; j++) s[j] += x[j];
      }

      int index_out = chunk_i * mylineoffset + (line / rowlines) * skiplines + (line % rowlines);
      (reinterpret_cast<int4 *>(outbuf))[index_out] = sum;
    }
    if (threadIdx.x == 0 && lastSM) *reduceidptr = reduce_id;
  }
}  // fp16 reduce-scatter kernel (out of place) fp16

template <int RANKS>
__global__ void __launch_bounds__(MAX_THREADS)
    userbuffers_fp16_sum_inplace_gpu_rr_ag(const int op, const int flagoffset, const int firstrank,
                                           const int myrank, const int gpustep,
                                           const int mylineoffset, const int totallines,
                                           void **commbuff, const int handleridx,
                                           const uint64_t ub_timeout) {
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
    if (threadIdx.x == 0) *reduceidptr = reduce_id;
    flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&myptr[targetgpu];
    clock_t s = clock64();
    while (CHECK_IDS(*flag, reduce_id)) {
      if (CHECK_TIMEOUT(s, ub_timeout)) {
        UB_PRINT("[%d] Allgather: SM %d [%d]: expecting %d got %d", myrank, blockIdx.x, threadIdx.x,
                 reduce_id, *flag);
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
                                           void **commbuff, const int handleridx,
                                           const uint64_t ub_timeout) {
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
    for (int j = 0; j < UNROLLAG; j++) val[j] = localptr[mylineoffset + line + loop_step0 * j];

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
  if (threadIdx.x == 0) __threadfence_system();
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
    if (threadIdx.x == 0) *reduceidptr = reduce_id;
    flagptr[physgpu] = reduce_id;
    volatile int *flag = (volatile int *)&myptr[targetgpu];
    clock_t s = clock64();
    while (CHECK_IDS(*flag, reduce_id)) {
      if (CHECK_TIMEOUT(s, ub_timeout)) {
        UB_PRINT("[%d] Allgather: SM %d [%d]: expecting %d got %d", myrank, blockIdx.x, threadIdx.x,
                 reduce_id, *flag);
        break;
      }
    }
  }
}  // fp16 inplace allgather kernel (Volta,Hopper)

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
    uint64_t arg10 = comm->ub_timeout;                                                             \
    void *kernelArgs[] = {reinterpret_cast<void *>(&arg1), reinterpret_cast<void *>(&arg2),        \
                          reinterpret_cast<void *>(&arg3), reinterpret_cast<void *>(&arg4),        \
                          reinterpret_cast<void *>(&arg5), reinterpret_cast<void *>(&arg6),        \
                          reinterpret_cast<void *>(&arg7), reinterpret_cast<void *>(&arg8),        \
                          reinterpret_cast<void *>(&arg9), reinterpret_cast<void *>(&arg10)};      \
    CUDACHECK(cudaLaunchKernelExC(                                                                 \
        &cfg,                                                                                      \
        reinterpret_cast<void *>(comm->use_rr_kernel ? userbuffers_fp16_sum_inplace_gpu_rr_ag<x>   \
                                                     : userbuffers_fp16_sum_inplace_gpu_rw_ag<x>), \
        kernelArgs));                                                                              \
  }

#define callranks_agMC(x)                                                                        \
  if (ar_nvsize == x) {                                                                          \
    int arg1 = op - NVTE_MAX_OPS,                                                                \
        arg2 = NVTE_REG0_OFFSET(comm) -                                                          \
               (op == userbuffers_allreduceop_nonsharp ? 2 : 1) * NVTE_REG0_SINGLENODE +         \
               NVTE_MAX_OPS,                                                                     \
        arg3 = ar_firstgpu, arg4 = ar_nvrank, arg5 = ar_step, arg7 = elements / 8 / x,           \
        arg6 = offset / 8 + arg4 * arg7;                                                         \
    void **arg8 = reinterpret_cast<void **>(comm->gpu_ptrs);                                     \
    int arg9 = handler * comm->nvsize;                                                           \
    uint4 *arg10 = reinterpret_cast<uint4 *>(comm->mc_ptr[handler]);                             \
    uint64_t arg11 = comm->ub_timeout;                                                           \
    void *kernelArgs[] = {reinterpret_cast<void *>(&arg1), reinterpret_cast<void *>(&arg2),      \
                          reinterpret_cast<void *>(&arg3), reinterpret_cast<void *>(&arg4),      \
                          reinterpret_cast<void *>(&arg5), reinterpret_cast<void *>(&arg6),      \
                          reinterpret_cast<void *>(&arg7), reinterpret_cast<void *>(&arg8),      \
                          reinterpret_cast<void *>(&arg9), reinterpret_cast<void *>(&arg10),     \
                          reinterpret_cast<void *>(&arg11)};                                     \
    CUDACHECK(cudaLaunchKernelExC(                                                               \
        &cfg, reinterpret_cast<void *>(userbuffers_fp16_sum_inplace_gpu_mc_ag<x>), kernelArgs)); \
  }

#define callranks_rs(x)                                                                          \
  if (ar_nvsize == x) {                                                                          \
    int arg1 = op - NVTE_MAX_OPS,                                                                \
        arg2 = NVTE_REG0_OFFSET(comm) -                                                          \
               (op == userbuffers_allreduceop_nonsharp ? 2 : 1) * NVTE_REG0_SINGLENODE +         \
               NVTE_MAX_OPS,                                                                     \
        arg3 = ar_firstgpu, arg4 = ar_nvrank, arg5 = ar_step, arg7 = elements / 8 / x,           \
        arg6 = offset / 8 + arg4 * arg7;                                                         \
    void **arg8 = reinterpret_cast<void **>(comm->gpu_ptrs);                                     \
    int arg9 = handler * comm->nvsize;                                                           \
    uint64_t arg10 = comm->ub_timeout;                                                           \
    void *kernelArgs[] = {reinterpret_cast<void *>(&arg1), reinterpret_cast<void *>(&arg2),      \
                          reinterpret_cast<void *>(&arg3), reinterpret_cast<void *>(&arg4),      \
                          reinterpret_cast<void *>(&arg5), reinterpret_cast<void *>(&arg6),      \
                          reinterpret_cast<void *>(&arg7), reinterpret_cast<void *>(&arg8),      \
                          reinterpret_cast<void *>(&arg9), reinterpret_cast<void *>(&arg10)};    \
    CUDACHECK(cudaLaunchKernelExC(                                                               \
        &cfg, reinterpret_cast<void *>(userbuffers_fp16_sum_inplace_gpu_rr_rs<x>), kernelArgs)); \
  }

#define callranks_rsMC(x)                                                                        \
  if (ar_nvsize == x) {                                                                          \
    int arg1 = op - NVTE_MAX_OPS,                                                                \
        arg2 = NVTE_REG0_OFFSET(comm) -                                                          \
               (op == userbuffers_allreduceop_nonsharp ? 2 : 1) * NVTE_REG0_SINGLENODE +         \
               NVTE_MAX_OPS,                                                                     \
        arg3 = ar_firstgpu, arg4 = ar_nvrank, arg5 = ar_step, arg7 = elements / 8 / x,           \
        arg6 = offset / 8 + arg4 * arg7;                                                         \
    void **arg8 = reinterpret_cast<void **>(comm->gpu_ptrs);                                     \
    int arg9 = handler * comm->nvsize;                                                           \
    void *arg10 = comm->mc_ptr[handler];                                                         \
    uint64_t arg11 = comm->ub_timeout;                                                           \
    void *kernelArgs[] = {reinterpret_cast<void *>(&arg1), reinterpret_cast<void *>(&arg2),      \
                          reinterpret_cast<void *>(&arg3), reinterpret_cast<void *>(&arg4),      \
                          reinterpret_cast<void *>(&arg5), reinterpret_cast<void *>(&arg6),      \
                          reinterpret_cast<void *>(&arg7), reinterpret_cast<void *>(&arg8),      \
                          reinterpret_cast<void *>(&arg9), reinterpret_cast<void *>(&arg10),     \
                          reinterpret_cast<void *>(&arg11)};                                     \
    CUDACHECK(cudaLaunchKernelExC(                                                               \
        &cfg, reinterpret_cast<void *>(userbuffers_fp16_sum_inplace_gpu_mc_rs<x>), kernelArgs)); \
  }

#define callranks_rs_oop(x)                                                                   \
  if (ar_nvsize == x) {                                                                       \
    int arg1 = op - NVTE_MAX_OPS,                                                             \
        arg2 = NVTE_REG0_OFFSET(comm) -                                                       \
               (op == userbuffers_allreduceop_nonsharp ? 2 : 1) * NVTE_REG0_SINGLENODE +      \
               NVTE_MAX_OPS,                                                                  \
        arg3 = ar_firstgpu, arg4 = ar_nvrank, arg5 = ar_step, arg7 = elements / 8 / x,        \
        arg6 = offset / 8 + arg4 * arg7, arg8 = rowelements / 8, arg9 = strideelements / 8;   \
    void **arg10 = reinterpret_cast<void **>(comm->gpu_ptrs);                                 \
    int arg11 = handler * comm->nvsize;                                                       \
    void *arg12 = output;                                                                     \
    uint64_t arg13 = comm->ub_timeout;                                                        \
    void *kernelArgs[] = {reinterpret_cast<void *>(&arg1),  reinterpret_cast<void *>(&arg2),  \
                          reinterpret_cast<void *>(&arg3),  reinterpret_cast<void *>(&arg4),  \
                          reinterpret_cast<void *>(&arg5),  reinterpret_cast<void *>(&arg6),  \
                          reinterpret_cast<void *>(&arg7),  reinterpret_cast<void *>(&arg8),  \
                          reinterpret_cast<void *>(&arg9),  reinterpret_cast<void *>(&arg10), \
                          reinterpret_cast<void *>(&arg11), reinterpret_cast<void *>(&arg12), \
                          reinterpret_cast<void *>(&arg13)};                                  \
    CUDACHECK(cudaLaunchKernelExC(                                                            \
        &cfg, reinterpret_cast<void *>(userbuffers_fp16_sum_inplace_gpu_rr_rs_oop<x>),        \
        kernelArgs));                                                                         \
  }

#define callranks_rs_oop_fp8(x)                                                                \
  if (ar_nvsize == x) {                                                                        \
    int arg1 = op - NVTE_MAX_OPS,                                                              \
        arg2 = NVTE_REG0_OFFSET(comm) -                                                        \
               (op == userbuffers_allreduceop_nonsharp ? 2 : 1) * NVTE_REG0_SINGLENODE +       \
               NVTE_MAX_OPS,                                                                   \
        arg3 = ar_firstgpu, arg4 = ar_nvrank, arg5 = ar_step, arg7 = elements / 16 / x,        \
        arg6 = offset / 16 + arg4 * arg7, arg8 = rowelements / 8, arg9 = strideelements / 8;   \
    void **arg10 = reinterpret_cast<void **>(comm->gpu_ptrs);                                  \
    int arg11 = handler * comm->nvsize;                                                        \
    void *arg12 = output;                                                                      \
    float *arg13 = scale;                                                                      \
    uint64_t arg14 = comm->ub_timeout;                                                         \
    void *kernelArgs[] = {reinterpret_cast<void *>(&arg1),  reinterpret_cast<void *>(&arg2),   \
                          reinterpret_cast<void *>(&arg3),  reinterpret_cast<void *>(&arg4),   \
                          reinterpret_cast<void *>(&arg5),  reinterpret_cast<void *>(&arg6),   \
                          reinterpret_cast<void *>(&arg7),  reinterpret_cast<void *>(&arg8),   \
                          reinterpret_cast<void *>(&arg9),  reinterpret_cast<void *>(&arg10),  \
                          reinterpret_cast<void *>(&arg11), reinterpret_cast<void *>(&arg12),  \
                          reinterpret_cast<void *>(&arg13), reinterpret_cast<void *>(&arg14)}; \
    CUDACHECK(cudaLaunchKernelExC(                                                             \
        &cfg,                                                                                  \
        reinterpret_cast<void *>(userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_fp8<x, fp8type>),  \
        kernelArgs));                                                                          \
  }

#define callranks_rs_oopMC(x)                                                                  \
  if (ar_nvsize == x) {                                                                        \
    int arg1 = op - NVTE_MAX_OPS,                                                              \
        arg2 = NVTE_REG0_OFFSET(comm) -                                                        \
               (op == userbuffers_allreduceop_nonsharp ? 2 : 1) * NVTE_REG0_SINGLENODE +       \
               NVTE_MAX_OPS,                                                                   \
        arg3 = ar_firstgpu, arg4 = ar_nvrank, arg5 = ar_step, arg7 = elements / 8 / x,         \
        arg6 = offset / 8 + arg4 * arg7, arg8 = rowelements / 8, arg9 = strideelements / 8;    \
    void **arg10 = reinterpret_cast<void **>(comm->gpu_ptrs);                                  \
    int arg11 = handler * comm->nvsize;                                                        \
    void *arg12 = output;                                                                      \
    void *arg13 = comm->mc_ptr[handler];                                                       \
    uint64_t arg14 = comm->ub_timeout;                                                         \
    void *kernelArgs[] = {reinterpret_cast<void *>(&arg1),  reinterpret_cast<void *>(&arg2),   \
                          reinterpret_cast<void *>(&arg3),  reinterpret_cast<void *>(&arg4),   \
                          reinterpret_cast<void *>(&arg5),  reinterpret_cast<void *>(&arg6),   \
                          reinterpret_cast<void *>(&arg7),  reinterpret_cast<void *>(&arg8),   \
                          reinterpret_cast<void *>(&arg9),  reinterpret_cast<void *>(&arg10),  \
                          reinterpret_cast<void *>(&arg11), reinterpret_cast<void *>(&arg12),  \
                          reinterpret_cast<void *>(&arg13), reinterpret_cast<void *>(&arg14)}; \
    CUDACHECK(cudaLaunchKernelExC(                                                             \
        &cfg, reinterpret_cast<void *>(userbuffers_fp16_sum_inplace_gpu_mc_rs_oop<x>),         \
        kernelArgs));                                                                          \
  }

#define callranks_rs_oop_atomic_fp8(x)                                                         \
  if (ar_nvsize == x) {                                                                        \
    int arg1 = op - NVTE_MAX_OPS,                                                              \
        arg2 = NVTE_REG0_OFFSET(comm) -                                                        \
               (op == userbuffers_allreduceop_nonsharp ? 2 : 1) * NVTE_REG0_SINGLENODE +       \
               NVTE_MAX_OPS,                                                                   \
        arg3 = ar_firstgpu, arg4 = ar_nvrank, arg5 = ar_step, arg7 = elements / 16 / x,        \
        arg6 = offset / 16, arg8 = rowelements / 8, arg9 = strideelements_out / 8,             \
        arg10 = strideelements_in / 16;                                                        \
    void **arg11 = reinterpret_cast<void **>(comm->gpu_ptrs);                                  \
    int arg12 = handler * comm->nvsize;                                                        \
    void *arg13 = output;                                                                      \
    float *arg14 = scale;                                                                      \
    void *arg15 = counters;                                                                    \
    int arg16 = numchunks, arg17 = atomicindex;                                                \
    uint64_t arg18 = comm->ub_timeout;                                                         \
    void *kernelArgs[] = {reinterpret_cast<void *>(&arg1),  reinterpret_cast<void *>(&arg2),   \
                          reinterpret_cast<void *>(&arg3),  reinterpret_cast<void *>(&arg4),   \
                          reinterpret_cast<void *>(&arg5),  reinterpret_cast<void *>(&arg6),   \
                          reinterpret_cast<void *>(&arg7),  reinterpret_cast<void *>(&arg8),   \
                          reinterpret_cast<void *>(&arg9),  reinterpret_cast<void *>(&arg10),  \
                          reinterpret_cast<void *>(&arg11), reinterpret_cast<void *>(&arg12),  \
                          reinterpret_cast<void *>(&arg13), reinterpret_cast<void *>(&arg14),  \
                          reinterpret_cast<void *>(&arg15), reinterpret_cast<void *>(&arg16),  \
                          reinterpret_cast<void *>(&arg17), reinterpret_cast<void *>(&arg18)}; \
    CUDACHECK(cudaLaunchKernelExC(                                                             \
        &cfg,                                                                                  \
        reinterpret_cast<void *>(                                                              \
            userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_atomic_fp8<x, fp8type>),                \
        kernelArgs));                                                                          \
  }

#define callranks_rs_oop_stride(x)                                                            \
  if (ar_nvsize == x) {                                                                       \
    int arg1 = op - NVTE_MAX_OPS,                                                             \
        arg2 = NVTE_REG0_OFFSET(comm) -                                                       \
               (op == userbuffers_allreduceop_nonsharp ? 2 : 1) * NVTE_REG0_SINGLENODE +      \
               NVTE_MAX_OPS,                                                                  \
        arg3 = ar_firstgpu, arg4 = ar_nvrank, arg5 = ar_step, arg7 = elements / 8 / x,        \
        arg6 = offset / 8, arg8 = rowelements / 8, arg9 = strideelements / 8;                 \
    void **arg10 = reinterpret_cast<void **>(comm->gpu_ptrs);                                 \
    int arg11 = handler * comm->nvsize;                                                       \
    void *arg12 = output;                                                                     \
    uint64_t arg13 = comm->ub_timeout;                                                        \
    void *kernelArgs[] = {reinterpret_cast<void *>(&arg1),  reinterpret_cast<void *>(&arg2),  \
                          reinterpret_cast<void *>(&arg3),  reinterpret_cast<void *>(&arg4),  \
                          reinterpret_cast<void *>(&arg5),  reinterpret_cast<void *>(&arg6),  \
                          reinterpret_cast<void *>(&arg7),  reinterpret_cast<void *>(&arg8),  \
                          reinterpret_cast<void *>(&arg9),  reinterpret_cast<void *>(&arg10), \
                          reinterpret_cast<void *>(&arg11), reinterpret_cast<void *>(&arg12), \
                          reinterpret_cast<void *>(&arg13)};                                  \
    CUDACHECK(cudaLaunchKernelExC(                                                            \
        &cfg, reinterpret_cast<void *>(userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_stride<x>), \
        kernelArgs));                                                                         \
  }

#define callranks_rs_oop_stride_atomic(x)                                                        \
  if (ar_nvsize == x) {                                                                          \
    int arg1 = op - NVTE_MAX_OPS,                                                                \
        arg2 = NVTE_REG0_OFFSET(comm) -                                                          \
               (op == userbuffers_allreduceop_nonsharp ? 2 : 1) * NVTE_REG0_SINGLENODE +         \
               NVTE_MAX_OPS,                                                                     \
        arg3 = ar_firstgpu, arg4 = ar_nvrank, arg5 = ar_step, arg7 = elements / 8 / x,           \
        arg6 = offset / 8, arg8 = rowelements / 8, arg9 = strideelements / 8, arg10 = numchunks; \
    void **arg11 = reinterpret_cast<void **>(comm->gpu_ptrs);                                    \
    int arg12 = handler * comm->nvsize;                                                          \
    void *arg13 = output;                                                                        \
    void *arg14 = counters;                                                                      \
    uint64_t arg15 = comm->ub_timeout;                                                           \
    void *kernelArgs[] = {reinterpret_cast<void *>(&arg1),  reinterpret_cast<void *>(&arg2),     \
                          reinterpret_cast<void *>(&arg3),  reinterpret_cast<void *>(&arg4),     \
                          reinterpret_cast<void *>(&arg5),  reinterpret_cast<void *>(&arg6),     \
                          reinterpret_cast<void *>(&arg7),  reinterpret_cast<void *>(&arg8),     \
                          reinterpret_cast<void *>(&arg9),  reinterpret_cast<void *>(&arg10),    \
                          reinterpret_cast<void *>(&arg11), reinterpret_cast<void *>(&arg12),    \
                          reinterpret_cast<void *>(&arg13), reinterpret_cast<void *>(&arg14),    \
                          reinterpret_cast<void *>(&arg15)};                                     \
    CUDACHECK(cudaLaunchKernelExC(                                                               \
        &cfg,                                                                                    \
        reinterpret_cast<void *>(userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_stride_atomic<x>),   \
        kernelArgs));                                                                            \
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
    uint64_t arg15 = comm->ub_timeout;                                                             \
    void *kernelArgs[] = {reinterpret_cast<void *>(&arg1),  reinterpret_cast<void *>(&arg2),       \
                          reinterpret_cast<void *>(&arg3),  reinterpret_cast<void *>(&arg4),       \
                          reinterpret_cast<void *>(&arg5),  reinterpret_cast<void *>(&arg6),       \
                          reinterpret_cast<void *>(&arg7),  reinterpret_cast<void *>(&arg8),       \
                          reinterpret_cast<void *>(&arg9),  reinterpret_cast<void *>(&arg10),      \
                          reinterpret_cast<void *>(&arg11), reinterpret_cast<void *>(&arg12),      \
                          reinterpret_cast<void *>(&arg13), reinterpret_cast<void *>(&arg14),      \
                          reinterpret_cast<void *>(&arg15)};                                       \
    CUDACHECK(                                                                                     \
        cudaLaunchKernelExC(&cfg,                                                                  \
                            reinterpret_cast<void *>(                                              \
                                userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_stride_multiatomic<x>), \
                            kernelArgs));                                                          \
  }

void reducescatter2_userbuff_strided(void *output, const int handler, const int offset,
                                     const int rowelements, const int colelements,
                                     const int strideelements, communicator *comm,
                                     cudaStream_t stream) {
  const int elements = rowelements * colelements;
  const int op = userbuffers_allreduceop_nonsharp2;
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
  callranks_rs_oop_stride(2) callranks_rs_oop_stride(4) callranks_rs_oop_stride(8)
}
void reducescatter2_userbuff_strided_atomic(void *output, const int handler, const int offset,
                                            const int rowelements, const int colelements,
                                            const int strideelements, const int numchunks,
                                            void *counters, communicator *comm,
                                            cudaStream_t stream) {
  const int elements = rowelements * colelements;
  const int op = userbuffers_allreduceop_nonsharp2;
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
  callranks_rs_oop_stride_atomic(2) callranks_rs_oop_stride_atomic(4)
      callranks_rs_oop_stride_atomic(8)
}

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
  const int ar_firstgpu =
      op == userbuffers_allreduceop_nonsharp ? comm->ar_firstgpu : comm->ar2_firstgpu;
  const int ar_step = op == userbuffers_allreduceop_nonsharp2 ? 1 : comm->ar2_nvsize;
  const int ar_nvsize = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvsize : comm->ar2_nvsize;
  const int ar_nvrank = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvrank : comm->ar2_nvrank;
  assert(comm->sm_arch >= 9);
  if (elements < 128) return;
  int sms = ar_nvsize == 1 ? 2 : comm->sms;
  int warps = comm->threads / 32;
  if (warps < ar_nvsize) warps = ar_nvsize;

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
  callranks_rs_oop_stride_multiatomic(2) callranks_rs_oop_stride_multiatomic(4)
      callranks_rs_oop_stride_multiatomic(8)
}

void allgather2_userbuff_inplace(const int handler, const int offset, const int elements,
                                 communicator *comm, cudaStream_t stream) {
  const int op = userbuffers_allreduceop_nonsharp2;
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
  const int ar_firstgpu =
      op == userbuffers_allreduceop_nonsharp ? comm->ar_firstgpu : comm->ar2_firstgpu;
  const int ar_step = op == userbuffers_allreduceop_nonsharp2 ? 1 : comm->ar2_nvsize;
  const int ar_nvsize = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvsize : comm->ar2_nvsize;
  const int ar_nvrank = op == userbuffers_allreduceop_nonsharp ? comm->ar_nvrank : comm->ar2_nvrank;
  assert(comm->sm_arch >= 9);
  if (elements < 128) return;
  int sms = ar_nvsize == 1 ? 2 : comm->sms;
  int warps = comm->threads / 32;
  if (warps < ar_nvsize) warps = ar_nvsize;

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
                                                         cudaStream_t stream);
template void reducescatter2_userbuff_fp8<__nv_fp8_e4m3>(void *output, float *scale,
                                                         const int handler, const int offset,
                                                         const int elements, communicator *comm,
                                                         cudaStream_t stream);

template void reducescatter2_userbuff_strided_atomic_fp8<__nv_fp8_e4m3>(
    void *output, float *scale, const int handler, const int offset, const int rowelements,
    const int colelements, const int strideelements_out, const int strideelements_in,
    const int numchunks, void *counters, communicator *comm, cudaStream_t stream);

template void reducescatter2_userbuff_strided_multiatomic_fp8<__nv_fp8_e4m3>(
    void *output, float *scale, const int handler, const int offset, const int rowelements,
    const int colelements, const int strideelements_out, const int strideelements_in,
    const int numchunks, void *counters, communicator *comm, cudaStream_t stream);

__global__ void kuserbuffers_pullsend(int myrank, int peer, int *send_id, int *flagptr) {
  atomicAdd_system(flagptr, 1);
}

__global__ void kuserbuffers_inc(int *id) { atomicAdd(id, 1); }

__global__ void kuserbuffers_dummy(void) {}

__global__ void __launch_bounds__(MAX_THREADS)
    kuserbuffers_pullrecv(int myrank, int peer, int nvrank, int nvpeer, int *recv_id, int *flagptr,
                          int4 *srcptr, int4 *dstptr, const int lines, uint64_t ub_timeout) {
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
      if (CHECK_TIMEOUT(s, ub_timeout)) {
        UB_PRINT(
            "pullrecv [grank dst:%d global src:%d][nvrank(GPU) dst: %d src: %d]: expecting %d,"
            " observed %d",
            myrank, peer, nvrank, nvpeer, signal_id, *flag);
        break;
      }
    }
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
    atomicAdd_system(flagptr,
                     1);  // otherwise need local SM sync before sending flag
  } else {                // 0 bytes and 1 SM only
    atomicAdd_system(flagptr, 1);
  }
}

#define CHECK_CE(ce_start, ce_end) \
  ((ce_start) != nullptr && (ce_end) != nullptr && *(ce_start) != *(ce_end))

__global__ void kuserbuffers_pushrecv(int myrank, int peer, int nvrank, int nvpeer, int *recv_id,
                                      int *flagptr, int adder, uint64_t ub_timeout,
                                      int *ce_start_ptr, int *ce_end_ptr) {
  const int signal_id = (*recv_id) + adder;
  *recv_id = signal_id;
  volatile int *flag = (volatile int *)flagptr;
  if (*flag >= signal_id) return;
  clock_t s = clock64();
  while (CHECK_IDS(*flag, signal_id)) {
    if (CHECK_TIMEOUT(s, ub_timeout)) {
      UB_PRINT(
          "pushrecv [grank dst:%d global src:%d][nvrank(GPU) dst: %d src: %d]: "
          "expecting %d, observed %d",
          myrank, peer, nvrank, nvpeer, signal_id, *flag);
      if (CHECK_CE(ce_start_ptr, ce_end_ptr))
        UB_PRINT("pushrecv: CE deadlock DETECTED: %d (ce_start) != %d (ce_end)\n", *ce_start_ptr,
                 *ce_end_ptr);
      return;
    }
  }
}

__global__ void __launch_bounds__(MAX_THREADS)
    kuserbuffers_pushsendrecv(int *send_id, int *send_flagptr, int4 *srcptr, int4 *dstptr,
                              const int lines, int send_peer, int recv_peer, int *recv_id,
                              int *recv_flagptr, int adder, uint64_t ub_timeout, int nv_send,
                              int nv_recv, int *ce_start_ptr, int *ce_end_ptr) {
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
    if (threadIdx.x) return;
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
    if (*flag >= signal_id) return;
    clock_t s = clock64();
    while (CHECK_IDS(*flag, signal_id)) {
      if (CHECK_TIMEOUT(s, ub_timeout)) {
        UB_PRINT(
            "pushsendrecv [sending peer:%d receiving peer:%d][nvrank(GPU) sending peer: %d"
            " receiving peer: %d]: expecting %d, observed %d",
            send_peer, recv_peer, nv_send, nv_recv, signal_id, *flag);
        if (CHECK_CE(ce_start_ptr, ce_end_ptr))
          UB_PRINT("pushrecv: CE deadlock DETECTED: %d (ce_start) != %d (ce_end)\n", *ce_start_ptr,
                   *ce_end_ptr);
        return;
      }
    }
  }
}

__global__ void __launch_bounds__(MAX_THREADS)
    kuserbuffers_pushsendrecv_atomic(int *send_id, int *send_flagptr, int4 *srcptr, int4 *dstptr,
                                     const int lines, int send_peer, int recv_peer, int *recv_id,
                                     int *recv_flagptr, int adder, void *counters,
                                     uint64_t ub_timeout, int nv_send, int nv_recv,
                                     int *ce_start_ptr, int *ce_end_ptr) {
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
    if (threadIdx.x) return;
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
    clock_t s = clock64();
    while (CHECK_IDS(*flag, signal_id)) {
      if (CHECK_TIMEOUT(s, ub_timeout)) {
        UB_PRINT(
            "pushsendrecv atomic [sending peer:%d receiving peer:%d][nvrank(GPU) sending peer:"
            " %d receiving peer: %d]: expecting %d, observed %d",
            send_peer, recv_peer, nv_send, nv_recv, signal_id, *flag); /*return;*/
        if (CHECK_CE(ce_start_ptr, ce_end_ptr))
          UB_PRINT("pushsendrecv atomic: CE deadlock DETECTED: %d (ce_start) != %d (ce_end)\n",
                   *ce_start_ptr, *ce_end_ptr);
      }
    }

    // Decrement atomic val to signal current output tile finish
    if (counters) {
      ((unsigned int *)counters)[0] = 0;
      asm volatile("fence.sc.gpu;\n");
    }
  }
}

__global__ void __launch_bounds__(MAX_THREADS) kuserbuffers_pushsendrecv_multiatomic(
    int *send_id, int *send_flagptr, int4 *srcptr, int4 *dstptr, const int lines, int send_peer,
    int recv_peer, int *recv_id, int *recv_flagptr, int adder, void *counters, int nchunks,
    int send_stride, int recv_stride, bool shuffle, uint64_t ub_timeout, int nv_send, int nv_recv) {
  for (int chunk_i = 0; chunk_i < nchunks - 1; chunk_i++) {
    int send_chunk_id = shuffle ? chunk_i : (nchunks + send_peer - chunk_i) % nchunks;
    int recv_chunk_id = shuffle ? chunk_i + 1 : (nchunks + send_peer - chunk_i - 1) % nchunks;
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
      clock_t s = clock64();
      while (CHECK_IDS(*flag, signal_id)) {
        if (CHECK_TIMEOUT(s, ub_timeout)) {
          UB_PRINT(
              "pushsendrecv multiatomic [sending peer:%d receiving peer:%d][nvrank(GPU)"
              " sending peer: %d receiving peer: %d]: expecting %d, observed %d",
              send_peer, recv_peer, nv_send, nv_recv, signal_id, *flag); /*return;*/
          // CE mode is not supported for multi-atomic, so there is no need to check for a deadlock
          return;
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
      atomicInc(((unsigned int *)counters) + nchunks + chunk_i, gridDim.x - 1);
      while (0 != (atomicCAS(((unsigned int *)counters) + nchunks + chunk_i, 0, 0))) {
      }
    }
    __syncthreads();
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

// Return TRUE if two ranks share the same NV domain
#define INTRANODE(peer) ((peer / comm->nvsize) == (comm->myrank / comm->nvsize))

// Index corresponds to the type of flag:
// 0 - Send index counter
// 1 - CE start index counter
// 2 - CE end index counter
#define GET_SEND_PTR_BY_INDEX(peerlocal, comm, dsth, index)                                 \
  ((reinterpret_cast<char *>((comm)->peer_ptr[0][(peerlocal)])) +                           \
   ((NVTE_REG0_OFFSET(comm) + NVTE_REG0_RECV + (comm)->myrank * NVTE_MAX_REGIONS + (dsth) + \
     (index) * NVTE_MAX_NVLINK * NVTE_MAX_REGIONS) *                                        \
    sizeof(int)))

// Index corresponds to the type of flag:
// 0 - Receive index counter
// 1 - CE start index counter
// 2 - CE end index counter
#define GET_RECV_PTR_BY_INDEX(recv_peer, comm, dsth, index)                              \
  ((reinterpret_cast<char *>((comm)->mem_ptr[0])) +                                      \
   ((NVTE_REG0_OFFSET(comm) + NVTE_REG0_RECV + (recv_peer) * NVTE_MAX_REGIONS + (dsth) + \
     (index) * NVTE_MAX_NVLINK * NVTE_MAX_REGIONS) *                                     \
    sizeof(int)))

void userbuffers_send(const int srchandler, const size_t srcoffset, const int dsthandler,
                      const size_t dstoffset, const size_t bytes, communicator *comm,
                      const int peer, cudaStream_t stream) {
  int peerlocal = peer % comm->nvsize;
  void *flagptr = GET_SEND_PTR_BY_INDEX(peerlocal, comm, dsthandler, 0);
  // void *ce_send_start_ptr = GET_SEND_PTR_BY_INDEX(peerlocal, comm, dsthandler, 1);
  // void *ce_send_end_ptr   = GET_SEND_PTR_BY_INDEX(peerlocal, comm, dsthandler, 2);
  bool signalonly = (bytes / 16 == 0) || (comm->use_ce != 0);

  assert(INTRANODE(peer));

  if (!(comm->launch_mode & NVTE_LAUNCH_GPU)) return;
  if (comm->push == 0) {
    kuserbuffers_pullsend<<<1, 1, 0, stream>>>(comm->myrank, peer, &(comm->send_id[peer]),
                                               reinterpret_cast<int *>(flagptr));
  } else {
    void *srcptr = reinterpret_cast<char *>(comm->mem_ptr[srchandler]) + srcoffset;
    void *dstptr = reinterpret_cast<char *>(comm->peer_ptr[dsthandler][peerlocal]) + dstoffset;

    if (comm->use_ce) {
      // kuserbuffers_inc<<<1, 1, 0, stream>>>(reinterpret_cast<int *>(ce_send_start_ptr));
      CUDACHECK(cudaMemcpyAsync(dstptr, srcptr, bytes, cudaMemcpyDeviceToDevice, stream));
      // kuserbuffers_inc<<<1, 1, 0, stream>>>(reinterpret_cast<int *>(ce_send_end_ptr));
    }
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
  void *flagptr_send = GET_SEND_PTR_BY_INDEX(send_peerlocal, comm, dsthandler, 0);
  // void *ce_send_start_ptr = GET_SEND_PTR_BY_INDEX(send_peerlocal, comm, dsthandler, 1);
  // void *ce_send_end_ptr   = GET_SEND_PTR_BY_INDEX(send_peerlocal, comm, dsthandler, 2);
  void *flagptr_recv = GET_RECV_PTR_BY_INDEX(recv_peer, comm, dsthandler, 0);

  void *send_srcptr = reinterpret_cast<char *>(comm->mem_ptr[srchandler]) + send_offset;
  void *send_dstptr =
      reinterpret_cast<char *>(comm->peer_ptr[dsthandler][send_peerlocal]) + send_offset;

  if (comm->use_ce) {
    // kuserbuffers_inc<<<1, 1, 0, stream>>>(reinterpret_cast<int *>(ce_send_start_ptr));
    CUDACHECK(cudaMemcpyAsync(send_dstptr, send_srcptr, bytes, cudaMemcpyDeviceToDevice, stream));
    // kuserbuffers_inc<<<1, 1, 0, stream>>>(reinterpret_cast<int *>(ce_send_end_ptr));
  }
  SETUP_LAUNCH_CONFIG(signalonly ? 1 : comm->sms, signalonly ? 1 : 1024, stream);

  int *arg1 = &comm->send_id[send_peer];
  int *arg2 = reinterpret_cast<int *>(flagptr_send);
  int4 *arg3 = reinterpret_cast<int4 *>(send_srcptr);
  int4 *arg4 = reinterpret_cast<int4 *>(send_dstptr);
  int arg5 = signalonly ? 0 : bytes / 16;
  int arg6 = send_peer;
  int arg7 = recv_peer;
  int *arg8 = &comm->recv_id[recv_peer * NVTE_MAX_REGIONS + dsthandler];
  int *arg9 = reinterpret_cast<int *>(flagptr_recv);
  int arg10 = signalonly ? 1 : comm->sms;
  uint64_t arg11 = comm->ub_timeout;
  int arg12 = send_peerlocal;
  int arg13 = recv_peerlocal;
  int *arg14 = reinterpret_cast<int *>(0 ?  // temporary disable
                                           GET_RECV_PTR_BY_INDEX(recv_peer, comm, dsthandler, 1)
                                         : nullptr);
  int *arg15 = reinterpret_cast<int *>(0 ?  // temporary disable
                                           GET_RECV_PTR_BY_INDEX(recv_peer, comm, dsthandler, 2)
                                         : nullptr);
  void *kernelArgs[] = {reinterpret_cast<void *>(&arg1),  reinterpret_cast<void *>(&arg2),
                        reinterpret_cast<void *>(&arg3),  reinterpret_cast<void *>(&arg4),
                        reinterpret_cast<void *>(&arg5),  reinterpret_cast<void *>(&arg6),
                        reinterpret_cast<void *>(&arg7),  reinterpret_cast<void *>(&arg8),
                        reinterpret_cast<void *>(&arg9),  reinterpret_cast<void *>(&arg10),
                        reinterpret_cast<void *>(&arg11), reinterpret_cast<void *>(&arg12),
                        reinterpret_cast<void *>(&arg13), reinterpret_cast<void *>(&arg14),
                        reinterpret_cast<void *>(&arg15)};
  CUDACHECK(
      cudaLaunchKernelExC(&cfg, reinterpret_cast<void *>(kuserbuffers_pushsendrecv), kernelArgs));
}

void userbuffers_sendrecv_atomic(const int srchandler, const int dsthandler,
                                 const size_t send_offset, const size_t recv_offset,
                                 const size_t bytes, communicator *comm, const int send_peer,
                                 const int recv_peer, void *counters, cudaStream_t stream) {
  assert(comm->push && comm->use_ce == 0);
  bool signalonly = (bytes / 16 == 0) || (comm->use_ce != 0);

  int send_peerlocal = send_peer % comm->nvsize;
  int recv_peerlocal = recv_peer % comm->nvsize;
  void *flagptr_send = GET_SEND_PTR_BY_INDEX(send_peerlocal, comm, dsthandler, 0);
  // void *ce_send_start_ptr = GET_SEND_PTR_BY_INDEX(send_peerlocal, comm, dsthandler, 1);
  // void *ce_send_end_ptr   = GET_SEND_PTR_BY_INDEX(send_peerlocal, comm, dsthandler, 2);
  void *flagptr_recv = GET_RECV_PTR_BY_INDEX(recv_peer, comm, dsthandler, 0);

  void *send_srcptr = reinterpret_cast<char *>(comm->mem_ptr[srchandler]) + send_offset;
  void *send_dstptr =
      reinterpret_cast<char *>(comm->peer_ptr[dsthandler][send_peerlocal]) + send_offset;
  if (comm->use_ce) {
    // kuserbuffers_inc<<<1, 1, 0, stream>>>(reinterpret_cast<int *>(ce_send_start_ptr));
    CUDACHECK(cudaMemcpyAsync(send_dstptr, send_srcptr, bytes, cudaMemcpyDeviceToDevice, stream));
    // kuserbuffers_inc<<<1, 1, 0, stream>>>(reinterpret_cast<int *>(ce_send_end_ptr));
  }
  SETUP_LAUNCH_CONFIG(signalonly ? 1 : comm->sms, signalonly ? 1 : 1024, stream);

  int *arg1 = &comm->send_id[send_peer];
  int *arg2 = reinterpret_cast<int *>(flagptr_send);
  int4 *arg3 = reinterpret_cast<int4 *>(send_srcptr);
  int4 *arg4 = reinterpret_cast<int4 *>(send_dstptr);
  int arg5 = signalonly ? 0 : bytes / 16;
  int arg6 = send_peer;
  int arg7 = recv_peer;
  int *arg8 = &comm->recv_id[recv_peer * NVTE_MAX_REGIONS + dsthandler];
  int *arg9 = reinterpret_cast<int *>(flagptr_recv);
  int arg10 = signalonly ? 1 : comm->sms;
  void *arg11 = counters;
  int arg12 = comm->ub_timeout;
  int arg13 = send_peerlocal;
  int arg14 = recv_peerlocal;
  int *arg15 = reinterpret_cast<int *>(0 ?  // temporary disable
                                           GET_RECV_PTR_BY_INDEX(recv_peer, comm, dsthandler, 1)
                                         : nullptr);
  int *arg16 = reinterpret_cast<int *>(0 ?  // temporary disable
                                           GET_RECV_PTR_BY_INDEX(recv_peer, comm, dsthandler, 2)
                                         : nullptr);
  void *kernelArgs[] = {reinterpret_cast<void *>(&arg1),  reinterpret_cast<void *>(&arg2),
                        reinterpret_cast<void *>(&arg3),  reinterpret_cast<void *>(&arg4),
                        reinterpret_cast<void *>(&arg5),  reinterpret_cast<void *>(&arg6),
                        reinterpret_cast<void *>(&arg7),  reinterpret_cast<void *>(&arg8),
                        reinterpret_cast<void *>(&arg9),  reinterpret_cast<void *>(&arg10),
                        reinterpret_cast<void *>(&arg11), reinterpret_cast<void *>(&arg12),
                        reinterpret_cast<void *>(&arg13), reinterpret_cast<void *>(&arg14),
                        reinterpret_cast<void *>(&arg15), reinterpret_cast<void *>(&arg16)};
  CUDACHECK(cudaLaunchKernelExC(&cfg, reinterpret_cast<void *>(kuserbuffers_pushsendrecv_atomic),
                                kernelArgs));
}

void userbuffers_sendrecv_multiatomic(const int srchandler, const int dsthandler,
                                      const size_t send_stride, const size_t recv_stride,
                                      const size_t bytes, communicator *comm, const int send_peer,
                                      const int recv_peer, const int nchunks, void *counters,
                                      bool shuffle, cudaStream_t stream) {
  assert(comm->push && comm->use_ce == 0);
  // CE is not supported

  int send_peerlocal = send_peer % comm->nvsize;
  int recv_peerlocal = recv_peer % comm->nvsize;
  void *flagptr_send = GET_SEND_PTR_BY_INDEX(send_peerlocal, comm, dsthandler, 0);
  void *flagptr_recv = GET_RECV_PTR_BY_INDEX(recv_peer, comm, dsthandler, 0);

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
  uint64_t arg16 = comm->ub_timeout;
  int arg17 = send_peerlocal;
  int arg18 = recv_peerlocal;
  void *kernelArgs[] = {reinterpret_cast<void *>(&arg1),  reinterpret_cast<void *>(&arg2),
                        reinterpret_cast<void *>(&arg3),  reinterpret_cast<void *>(&arg4),
                        reinterpret_cast<void *>(&arg5),  reinterpret_cast<void *>(&arg6),
                        reinterpret_cast<void *>(&arg7),  reinterpret_cast<void *>(&arg8),
                        reinterpret_cast<void *>(&arg9),  reinterpret_cast<void *>(&arg10),
                        reinterpret_cast<void *>(&arg11), reinterpret_cast<void *>(&arg12),
                        reinterpret_cast<void *>(&arg13), reinterpret_cast<void *>(&arg14),
                        reinterpret_cast<void *>(&arg15), reinterpret_cast<void *>(&arg16),
                        reinterpret_cast<void *>(&arg17), reinterpret_cast<void *>(&arg18)};
  CUDACHECK(cudaLaunchKernelExC(
      &cfg, reinterpret_cast<void *>(kuserbuffers_pushsendrecv_multiatomic), kernelArgs));
}

void userbuffers_recv(const int srchandler, const size_t srcoffset, const int dsthandler,
                      const size_t dstoffset, const size_t bytes, communicator *comm,
                      const int peer, cudaStream_t stream) {
  int peerlocal = peer % comm->nvsize;
  void *flagptr = GET_RECV_PTR_BY_INDEX(peer, comm, dsthandler, 0);
  bool signalonly = (bytes / 16 == 0) || (comm->use_ce != 0);

  assert(INTRANODE(peer));

  if (!(comm->launch_mode & NVTE_LAUNCH_GPU)) return;
  if (comm->push == 0) {
    void *dstptr = reinterpret_cast<char *>(comm->mem_ptr[dsthandler]) + dstoffset;
    void *srcptr = reinterpret_cast<char *>(comm->peer_ptr[srchandler][peerlocal]) + srcoffset;

    kuserbuffers_pullrecv<<<signalonly ? 1 : comm->sms, signalonly ? 1 : 1024, 0, stream>>>(
        comm->myrank, peer, comm->nvrank, peerlocal,
        &(comm->recv_id[peer * NVTE_MAX_REGIONS + dsthandler]), reinterpret_cast<int *>(flagptr),
        reinterpret_cast<int4 *>(srcptr), reinterpret_cast<int4 *>(dstptr),
        signalonly ? 0 : bytes / 16, comm->ub_timeout);
    if (!signalonly)
      kuserbuffers_inc<<<1, 1, 0, stream>>>(&(comm->recv_id[peer * NVTE_MAX_REGIONS + dsthandler]));
    if (comm->use_ce) {
      CUDACHECK(cudaMemcpyAsync(dstptr, srcptr, bytes, cudaMemcpyDeviceToDevice, stream));
    }
  } else {
    kuserbuffers_pushrecv<<<1, 1, 0, stream>>>(
        comm->myrank, peer, comm->nvrank, peerlocal,
        &comm->recv_id[peer * NVTE_MAX_REGIONS + dsthandler], reinterpret_cast<int *>(flagptr),
        signalonly || comm->sms, comm->ub_timeout,
        reinterpret_cast<int *>(0 ?  // temporary disable
                                    GET_RECV_PTR_BY_INDEX(peer, comm, dsthandler, 1)
                                  : nullptr),
        reinterpret_cast<int *>(0 ?  // temporary disable
                                    GET_RECV_PTR_BY_INDEX(peer, comm, dsthandler, 2)
                                  : nullptr));
  }
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
    while (0 != (atomicCAS((unsigned int *)atomic_ptr + chunk_i, 0, 0))) {
    }
    ((unsigned int *)atomic_ptr)[chunk_i] = 1;
    asm volatile("fence.sc.gpu;\n");
  }
}

// consumer_batch
static __global__ void consumer_batch_kernel(void *atomic_ptr, int first_chunk_i, int num_chunks) {
  // Wait for producer to change the val to 0, which signal producer ready
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (int i = first_chunk_i; i < num_chunks; i++) {
      while (0 != (atomicCAS((unsigned int *)atomic_ptr + i, 0, 0))) {
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
  output_half[tid] = (half)accum_buf;
}

template <typename fp8type>
void reduce_fp8_in_bf16_out(void *inputs, void *output, float *scale, int num_inputs,
                            int input_size, cudaStream_t stream) {
  size_t num_threads = MAX_THREADS / 4;
  size_t num_blocks = (input_size + num_threads - 1) / num_threads;
  dim3 block(num_threads);
  dim3 grid(num_blocks);
  reduce_fp8_in_bf16_out_cuda<fp8type>
      <<<grid, block, 0, stream>>>(inputs, output, scale, num_inputs, input_size);
}

template void reduce_fp8_in_bf16_out<__nv_fp8_e4m3>(void *inputs, void *output, float *scale,
                                                    int num_inputs, int input_size,
                                                    cudaStream_t stream);
template void reduce_fp8_in_bf16_out<__nv_fp8_e5m2>(void *inputs, void *output, float *scale,
                                                    int num_inputs, int input_size,
                                                    cudaStream_t stream);
