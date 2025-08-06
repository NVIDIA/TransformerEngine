#include <cuda.h>
#include <stdio.h>
#include <transformer_engine/transformer_engine.h>

#include "./common.h"

#define TIMEOUT 2000000000ull
//#define UB_TIMEOUT_ENABLED 1

#define NVTE_UB_MAXTHREADS 1024
#define NVTE_UB_MAX_SMS 128
#define NVTE_UB_LAMPORT_INT 0xFFFAFFFA

//REG0 flags in use
#define NVTE_UB_FLAG_NVLS2_LAMPORT_ID 0
#define NVTE_UB_FLAG_NVLS2_LAMPORT_SM_SYNC 1
#define NVTE_UB_FLAG_NVLS2_LAMPORT_RS_BAR 2
#define NVTE_UB_FLAG_NVLS2_ID 3
#define NVTE_UB_FLAG_NVLS2_SM_SYNC 4
#define NVTE_UB_FLAG_NVLS2_RS_BAR 5
#define NVTE_UB_FLAG_NVLS2_AG_BAR 6

#define xhalf __nv_bfloat16

#define ATOMIC_MCINC(ptr)                                          \
  asm volatile("multimem.red.add.u32 [%0], %1;" ::"l"(ptr), "r"(1) \
               : "memor"                                           \
                 "y");
#define ATOMIC_UCINC(ptr)                                        \
  asm volatile("red.global.add.u32 [%0], %1;" ::"l"(ptr), "r"(1) \
               : "memor"                                         \
                 "y");
#define MULTIMEM_ST(val, ptr)                                                           \
  asm volatile("multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(val.x), \
               "r"(val.y), "r"(val.z), "r"(val.w)                                       \
               : "memory");

#define MULTIMEM_LD(val, ptr)                                                 \
  asm("multimem.ld_reduce.global.add.v4.bf16x2.acc::f32 {%0,%1,%2,%3}, [%4];" \
      : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)                    \
      : "l"(ptr)                                                              \
      : "memory");

#define CUDACHECK(cmd)                                                                      \
  do {                                                                                      \
    cudaError_t e = cmd;                                                                    \
    if (e != cudaSuccess) {                                                                 \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(EXIT_FAILURE);                                                                   \
    }                                                                                       \
  } while (0)

// Return true if producer > consumer, otherwise false while preventing integer overflow
// If we expect that producer will be 2B+ messages behind consumer
#define CHECK_IDS(producer, consumer) (((unsigned)(producer) - (unsigned)(consumer)) & (~INT_MAX))

__global__ void __launch_bounds__(NVTE_UB_MAXTHREADS)
    userbuffers_fp16_sum_inplace_gpu_mc(const int RANKS, const int myrank, const int numlines,
                                        int *uc_flagptr, int *mc_flagptr, float4 *mc_ptr_in,
                                        float4 *mc_ptr_out) {
  // flags[3,4,5,6]: reduce_id, sm_sync-local, flag-barrier-1,flag-barrier-2
  int reduce_id;

  if (threadIdx.x == 0) {
    cudaGridDependencySynchronize();
    if (blockIdx.x == 0) ATOMIC_MCINC(mc_flagptr + NVTE_UB_FLAG_NVLS2_RS_BAR);

    reduce_id = uc_flagptr[NVTE_UB_FLAG_NVLS2_ID] + 1;

    volatile int *flag = (volatile int *)&(uc_flagptr[NVTE_UB_FLAG_NVLS2_RS_BAR]);

    const int expected = reduce_id * RANKS;

#ifdef UB_TIMEOUT_ENABLED
    clock_t s = clock64();
#endif
    while (CHECK_IDS(*flag, expected)) {
#ifdef UB_TIMEOUT_ENABLED
      if (clock64() - s > TIMEOUT) {
        printf("NVONLY RSBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, expected,
               *flag);
        break;
      }
#endif
    }
  }

  __syncthreads();
#define UNROLL_MC 4
  const int loop_step0 = blockDim.x * gridDim.x * RANKS;
  const int loop_step = loop_step0 * UNROLL_MC;
  const int start_elem = threadIdx.x + blockDim.x * (myrank + RANKS * blockIdx.x);
  const int end_elem = max(start_elem, numlines);
  const int aligned_elem = ((end_elem - start_elem) / loop_step) * loop_step;
  const int end_aligned = start_elem + aligned_elem;

  for (int line = start_elem; line < end_aligned; line += loop_step) {
    uint4 val[UNROLL_MC];
#pragma unroll
    for (int i = 0; i < UNROLL_MC; i++) MULTIMEM_LD(val[i], mc_ptr_in + (line + i * loop_step0))
#pragma unroll
    for (int i = 0; i < UNROLL_MC; i++) MULTIMEM_ST(val[i], mc_ptr_out + (line + i * loop_step0))
  }
  for (int line = end_aligned; line < end_elem; line += loop_step0) {
    uint4 val;
    MULTIMEM_LD(val, mc_ptr_in + (line))
    MULTIMEM_ST(val, mc_ptr_out + (line))
  }

  __syncthreads();
  if (threadIdx.x != 0) return;

  __threadfence();
  const int value_to_add = blockIdx.x == 0 ? NVTE_UB_MAX_SMS - gridDim.x + 1 : 1;
  const int old_val_sm_sync = atomicAdd(uc_flagptr + NVTE_UB_FLAG_NVLS2_SM_SYNC, value_to_add);

  const int lastSM =
      (gridDim.x == 1 || old_val_sm_sync + value_to_add == reduce_id * NVTE_UB_MAX_SMS);
  if (!lastSM) return;
  __threadfence_system();
  ATOMIC_MCINC(mc_flagptr + NVTE_UB_FLAG_NVLS2_AG_BAR);
  uc_flagptr[NVTE_UB_FLAG_NVLS2_ID] = reduce_id;
  cudaTriggerProgrammaticLaunchCompletion();
  volatile int *flag = (volatile int *)&(uc_flagptr[NVTE_UB_FLAG_NVLS2_AG_BAR]);
  const int expected = reduce_id * RANKS;

#ifdef UB_TIMEOUT_ENABLED
  clock_t s = clock64();
#endif
  while (CHECK_IDS(*flag, expected)) {
#ifdef UB_TIMEOUT_ENABLED
    if (clock64() - s > TIMEOUT) {
      printf("NVONLY AGBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, expected,
             *flag);
      break;
    }
#endif
  }
}  // fp16 inplace reduce kernel (Hopper) MC

template <int RANKS>
__global__ void __launch_bounds__(NVTE_UB_MAXTHREADS)
    userbuffers_fp16_sum_inplace_gpu_uc(const int myrank, const int numlines,
                                        const int lineoffset_in, const int lineoffset_out,
                                        int *uc_flagptr, void **commbuff) {
  // flags[3,4,5,6]: reduce_id, sm_sync-local, flag-barrier-1,flag-barrier-2
  //NB! uc_flagptr is shifted by ranks*8 for easier flag offsets
  //    while lineoffset is relative to start of reg0
  __shared__ int4 *userptr[RANKS];
  __shared__ int lastSM;
  int reduce_id;

  if (threadIdx.x < RANKS) {
    int *rem_flagptr = (reinterpret_cast<int *>(commbuff[threadIdx.x]));
    cudaGridDependencySynchronize();
    if (blockIdx.x == 0) ATOMIC_UCINC(rem_flagptr + NVTE_UB_FLAG_NVLS2_RS_BAR + RANKS * 2);

    reduce_id = uc_flagptr[NVTE_UB_FLAG_NVLS2_ID] + 1;

    userptr[threadIdx.x] = (int4 *)rem_flagptr;
  }

  if (threadIdx.x == 0) {
    volatile int *flag = uc_flagptr + NVTE_UB_FLAG_NVLS2_RS_BAR;
    lastSM = 0;
    const int expected = reduce_id * RANKS;
#ifdef UB_TIMEOUT_ENABLED
    clock_t s = clock64();
#endif
    while (CHECK_IDS(*flag, expected)) {
#ifdef UB_TIMEOUT_ENABLED
      if (clock64() - s > TIMEOUT) {
        printf("NVONLY RSBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, expected,
               *flag);
        break;
      }
#endif
    }
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
      val[i] = userptr[dest[i]][lineoffset_in + line];
    }

    int4 sum = val[0];
    xhalf *s = reinterpret_cast<xhalf *>(&sum);

#pragma unroll
    for (int i = 1; i < RANKS; i++) {
      xhalf *x = reinterpret_cast<xhalf *>(&val[i]);
#pragma unroll
      for (int j = 0; j < 8; j++) s[j] += x[j];
    }
#pragma unroll
    for (int i = 0; i < RANKS; i++) {
      // int dest = (i+myrank+warp)&(RANKS-1);
      userptr[dest[i]][lineoffset_out + line] = sum;
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    __threadfence();
    const int value_to_add = blockIdx.x == 0 ? NVTE_UB_MAX_SMS - gridDim.x + 1 : 1;
    const int old_val_sm_sync = atomicAdd(uc_flagptr + NVTE_UB_FLAG_NVLS2_SM_SYNC, value_to_add);
    lastSM = (gridDim.x == 1 || old_val_sm_sync + value_to_add == reduce_id * NVTE_UB_MAX_SMS);
    if (lastSM) uc_flagptr[NVTE_UB_FLAG_NVLS2_ID] = reduce_id;
    cudaTriggerProgrammaticLaunchCompletion();
  }
  if (threadIdx.x >= RANKS) return;
  __syncthreads();
  if (!lastSM) return;
  if (threadIdx.x == 0) __threadfence_system();
  __syncthreads();
  ATOMIC_UCINC((int *)(userptr[threadIdx.x]) + NVTE_UB_FLAG_NVLS2_AG_BAR + RANKS * 2);
  if (threadIdx.x != 0) return;
  volatile int *flag = uc_flagptr + NVTE_UB_FLAG_NVLS2_AG_BAR;
  const int expected = reduce_id * RANKS;
#ifdef UB_TIMEOUT_ENABLED
  clock_t s = clock64();
#endif
  while (CHECK_IDS(*flag, expected)) {
#ifdef UB_TIMEOUT_ENABLED
    if (clock64() - s > TIMEOUT) {
      printf("NVONLY AGBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, expected,
             *flag);
      break;
    }
#endif
  }
}  // UC 2shot kernel (non-lamport)

__global__ void memset_int(uint32_t *data, int n, uint32_t val) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] = val;
  }
}

__global__ void __launch_bounds__(NVTE_UB_MAXTHREADS) userbuffers_fp16_sum_inplace_gpu_mc_lamport(
    const int RANKS, const int myrank, const int numlines, int *uc_flagptr, int *mc_flagptr,
    float4 *mc_ptr_in, float4 *mc_ptr_out, uint4 *uc_ptr_out, uint4 *clear_ptr) {
  // flags[0,1,2]: reduce_id, sm_sync-local, flag-barrier
  // those go right after rank UC pointers, but its the CPU caller who should account for it
  int reduce_id;

  if (threadIdx.x == 0) {
    cudaGridDependencySynchronize();
    if (blockIdx.x == 0) ATOMIC_MCINC(mc_flagptr + NVTE_UB_FLAG_NVLS2_LAMPORT_RS_BAR);
    reduce_id = uc_flagptr[NVTE_UB_FLAG_NVLS2_LAMPORT_ID];
    const int value_to_add = blockIdx.x == 0 ? NVTE_UB_MAX_SMS - gridDim.x + 1 : 1;
    const int old_val_sm_sync =
        atomicAdd(uc_flagptr + NVTE_UB_FLAG_NVLS2_LAMPORT_SM_SYNC, value_to_add);
    volatile int *flag = (volatile int *)&(uc_flagptr[NVTE_UB_FLAG_NVLS2_LAMPORT_RS_BAR]);
    reduce_id++;
    const int lastSM =
        (gridDim.x == 1 || old_val_sm_sync + value_to_add == reduce_id * NVTE_UB_MAX_SMS);

    if (lastSM) uc_flagptr[NVTE_UB_FLAG_NVLS2_LAMPORT_ID] = reduce_id;
    cudaTriggerProgrammaticLaunchCompletion();

    const int expected = reduce_id * RANKS;

#ifdef UB_TIMEOUT_ENABLED
    clock_t s = clock64();
#endif
    while (CHECK_IDS(*flag, expected)) {
#ifdef UB_TIMEOUT_ENABLED
      if (clock64() - s > TIMEOUT) {
        printf("NVONLY RSBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, expected,
               *flag);
        break;
      }
#endif
    }
  }
  __syncthreads();

  const int loop_step0 = blockDim.x * gridDim.x * RANKS;
  const int start_elem = threadIdx.x + blockDim.x * (myrank + RANKS * blockIdx.x);

  for (int line = start_elem; line < numlines; line += loop_step0) {
    uint4 val;
    MULTIMEM_LD(val, mc_ptr_in + (line))
    MULTIMEM_ST(val, mc_ptr_out + (line))
  }

  for (int line = threadIdx.x + blockDim.x * blockIdx.x; line < numlines;
       line += blockDim.x * gridDim.x) {
#ifdef UB_TIMEOUT_ENABLED
    clock_t s = clock64();
#endif
    while (true) {
      uint4 result;

      asm volatile("ld.volatile.v4.u32 {%0, %1, %2, %3}, [%4];"
                   : "=r"(result.x), "=r"(result.y), "=r"(result.z), "=r"(result.w)
                   : "l"(&uc_ptr_out[line])
                   : "memory");
      if (result.w != NVTE_UB_LAMPORT_INT) {
        if (clear_ptr) clear_ptr[line].w = NVTE_UB_LAMPORT_INT;
        break;
      }
#ifdef UB_TIMEOUT_ENABLED
      if (clock64() - s > TIMEOUT) {
        printf("Lamport POLL:SM %d [%d]:expecting %d got (%d,%d,%d) %d\n", blockIdx.x, threadIdx.x,
               NVTE_UB_LAMPORT_INT, result.x, result.y, result.z, result.w);
        break;
      }
#endif
    }
  }

}  // two-shot NVLS + lamport sync instead of last membar

#define SETUP_LAUNCH_CONFIG(sms, threads, stream, cga_size, pdl_launch)    \
  cudaLaunchConfig_t cfg = {sms, threads, 0, stream, NULL, 0};             \
  cudaLaunchAttribute attribute_ub[3];                                     \
  attribute_ub[2].id = cudaLaunchAttributeClusterDimension;                \
  attribute_ub[2].val.clusterDim.x = sms % cga_size == 0 ? cga_size : 1;   \
  attribute_ub[2].val.clusterDim.y = 1;                                    \
  attribute_ub[2].val.clusterDim.z = 1;                                    \
  attribute_ub[1].id = cudaLaunchAttributeCooperative;                     \
  attribute_ub[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; \
  attribute_ub[0].val.programmaticStreamSerializationAllowed = pdl_launch; \
  cfg.attrs = attribute_ub;                                                \
  cfg.numAttrs = 3;

namespace transformer_engine {

extern "C" void allreduce_2shot_mc(int ranks, int myrank, void *uc0ptr, void *mc0ptr,
                                   void *mcptr_in, void *mcptr_out, size_t bytes,
                                   cudaStream_t stream) {
  SETUP_LAUNCH_CONFIG(32, 1024, stream, 4, 1);

  int arg1 = ranks, arg2 = myrank, arg3 = bytes / 16;
  void *arg4 = uc0ptr + (ranks * 8), *arg5 = mc0ptr + (ranks * 8), *arg6 = mcptr_in,
       *arg7 = mcptr_out;
  void *kernelArgs[] = {(void *)&arg1, (void *)&arg2, (void *)&arg3, (void *)&arg4,
                        (void *)&arg5, (void *)&arg6, (void *)&arg7};
  CUDACHECK(cudaLaunchKernelExC(&cfg, (void *)(userbuffers_fp16_sum_inplace_gpu_mc), kernelArgs));
}

extern "C" void allreduce_2shot_uc(int ranks, int myrank, void *uc0ptr, void *ucptr_in,
                                   void *ucptr_out, size_t bytes, cudaStream_t stream) {
  SETUP_LAUNCH_CONFIG(64, 1024, stream, 4, 1);

  int arg1 = myrank, arg2 = bytes / 16, arg3 = (int4 *)ucptr_in - (int4 *)uc0ptr,
      arg4 = (int4 *)ucptr_out - (int4 *)uc0ptr;
  void *arg5 = uc0ptr + (ranks * 8), **arg6 = (void **)uc0ptr;
  void *kernelArgs[] = {(void *)&arg1, (void *)&arg2, (void *)&arg3,
                        (void *)&arg4, (void *)&arg5, (void *)&arg6};
#define call_uc_kernel(x) \
  if (x == ranks)         \
    CUDACHECK(            \
        cudaLaunchKernelExC(&cfg, (void *)(userbuffers_fp16_sum_inplace_gpu_uc<x>), kernelArgs));
  call_uc_kernel(2);
  call_uc_kernel(4);
  call_uc_kernel(8);
}

extern "C" void allreduce_2shot_mc_lamport(int ranks, int myrank, void *uc0ptr, void *mc0ptr,
                                           void *ucptr_out, void *mcptr_in, void *mcptr_out,
                                           void *clear_ptr, size_t bytes, bool poisoned,
                                           cudaStream_t stream) {
  if (!poisoned) {
    //user tells us destination was not pre-poisoned, so we need to do it before calling allreduce
    int threadsPerBlock = 512;
    int blocks = (bytes / 4 + threadsPerBlock - 1) / threadsPerBlock;
    memset_int<<<blocks, threadsPerBlock, 0, stream>>>((uint32_t *)ucptr_out, bytes / 4,
                                                       NVTE_UB_LAMPORT_INT);
  }
  SETUP_LAUNCH_CONFIG(64, 1024, stream, 4, 1);

  int arg1 = ranks, arg2 = myrank, arg3 = bytes / 16;
  void *arg4 = uc0ptr + (ranks * 8), *arg5 = mc0ptr + (ranks * 8), *arg6 = mcptr_in,
       *arg7 = mcptr_out, *arg8 = ucptr_out, *arg9 = clear_ptr;
  void *kernelArgs[] = {(void *)&arg1, (void *)&arg2, (void *)&arg3, (void *)&arg4, (void *)&arg5,
                        (void *)&arg6, (void *)&arg7, (void *)&arg8, (void *)&arg9};
  CUDACHECK(
      cudaLaunchKernelExC(&cfg, (void *)(userbuffers_fp16_sum_inplace_gpu_mc_lamport), kernelArgs));
}
}  // namespace transformer_engine
