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

#define FINAL_MASK 0xffffffff
template <typename T, int NUM>
__inline__ __device__ T warpReduceSumV2(T *val) {
#pragma unroll
  for (int i = 0; i < NUM; i++) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
      val[i] += __shfl_xor_sync(FINAL_MASK, val[i], mask, 32);
  }
  return (T)(0.0f);
}

template <typename T, int NUM>
__inline__ __device__ T blockReduceSumV2(T *val) {
  static __shared__ T shared[NUM][33];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduceSumV2<T, NUM>(val);

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < NUM; i++) {
      shared[i][wid] = val[i];
    }
  }

  __syncthreads();

  bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
  for (int i = 0; i < NUM; i++) {
    val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
  }
  warpReduceSumV2<T, NUM>(val);
  return (T)0.0f;
}

template <int UNROLL>
__global__ void __launch_bounds__(NVTE_UB_MAXTHREADS)
    userbuffers_fp16_sum_inplace_gpu_mc(const int RANKS, const int myrank, const int mylines,
                                        int *uc_flagptr, int *mc_flagptr, uint4 *mc_ptr_in,
                                        uint4 *mc_ptr_out, uint4 *residual_in, uint4 *residual_out,
                                        xhalf *gamma, float eps, const int hidden_size,
                                        bool fuse_layernorm) {
  // flags[3,4,5,6]: reduce_id, sm_sync-local, flag-barrier-1,flag-barrier-2
  int reduce_id;
  __shared__ float s_variance;

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

  const int loop_step0 = blockDim.x;
  const int loop_step = loop_step0 * UNROLL * gridDim.x;
  const int start_elem = threadIdx.x + blockDim.x * blockIdx.x * UNROLL;
  const int end_elem = max(start_elem, mylines);
  //const int aligned_elem = ((end_elem - start_elem) / loop_step) * loop_step;
  //const int end_aligned = start_elem + aligned_elem;

  for (int line = start_elem; line < end_elem; line += loop_step) {
    uint4 val[UNROLL];
    xhalf *x = reinterpret_cast<xhalf *>(&val[0]);
#pragma unroll
    for (int i = 0; i < UNROLL; i++) MULTIMEM_LD(val[i], mc_ptr_in + (line + i * loop_step0))

    if (residual_in != nullptr) {
      for (int i = 0; i < UNROLL; i++) {
        uint4 resval = residual_in[line + i * loop_step0];
        xhalf *y = reinterpret_cast<xhalf *>(&resval);
#pragma unroll
        for (int j = 0; j < 8; j++) x[i * 8 + j] += y[j];
        if (residual_out != nullptr) residual_out[line + i * loop_step0] = val[i];
      }
    }
    if (fuse_layernorm) {
      float local_var_sum = 0.0f;
      for (int j = 0; j < UNROLL * sizeof(int4) / sizeof(xhalf); j++)
        local_var_sum += (float)(x[j]) * (float)(x[j]);

      float packed[1] = {local_var_sum};
      blockReduceSumV2<float, 1>(packed);
      float variance = packed[0];

      if (threadIdx.x == 0) {
        variance = (variance / hidden_size);  // Var[x] = E[x²]
        s_variance = rsqrtf(variance + eps);
      }
      __syncthreads();
    }

    int i = 0;
#pragma unroll
    for (int g = 0; g < UNROLL; g++) {
      if (fuse_layernorm) {
#pragma unroll
        for (int j = 0; j < sizeof(int4) / sizeof(xhalf); j++) {
          x[i] =
              (xhalf)((float)(x[i]) * s_variance *
                      (float)
                          gamma[(threadIdx.x + g * loop_step0) * sizeof(int4) / sizeof(xhalf) + j]);
          i++;
        }
      }
      MULTIMEM_ST(val[g], mc_ptr_out + (line + g * loop_step0))
    }
  }
  /*
  for (int line = end_aligned; line < end_elem; line += loop_step0) {
    uint4 val;
    xhalf *x = reinterpret_cast<xhalf *>(&val);
    MULTIMEM_LD(val, mc_ptr_in + (line))

    if(residual_in!=nullptr) {
      uint4 resval = residual_in[line];
      xhalf *y = reinterpret_cast<xhalf *>(&resval);
      #pragma unroll
      for (int j = 0; j < 8; j++)
          x[j] += y[j];
        if(residual_out!=nullptr)
          residual_out[line]=val;
    }

    MULTIMEM_ST(val, mc_ptr_out + (line))
  }
  */
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
                                        int *uc_flagptr, void **commbuff, uint4 *residual_in,
                                        uint4 *residual_out, xhalf *gamma, float eps,
                                        const int hidden_size, bool fuse_layernorm) {
  // flags[3,4,5,6]: reduce_id, sm_sync-local, flag-barrier-1,flag-barrier-2
  //NB! uc_flagptr is shifted by ranks*8 for easier flag offsets
  //    while lineoffset is relative to start of reg0
  __shared__ uint4 *userptr[RANKS];
  __shared__ int lastSM;
  int reduce_id;

  if (threadIdx.x < RANKS) {
    int *rem_flagptr = (reinterpret_cast<int *>(commbuff[threadIdx.x]));
    cudaGridDependencySynchronize();
    if (blockIdx.x == 0) ATOMIC_UCINC(rem_flagptr + NVTE_UB_FLAG_NVLS2_RS_BAR + RANKS * 2);

    reduce_id = uc_flagptr[NVTE_UB_FLAG_NVLS2_ID] + 1;

    userptr[threadIdx.x] = (uint4 *)rem_flagptr;
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
    uint4 val[RANKS];

#pragma unroll
    for (int i = 0; i < RANKS; i++) {
      // int dest = (i+myrank+warp)&(RANKS-1);
      val[i] = userptr[dest[i]][lineoffset_in + line];
    }

    uint4 sum = val[0];
    xhalf *s = reinterpret_cast<xhalf *>(&sum);

#pragma unroll
    for (int i = 1; i < RANKS; i++) {
      xhalf *x = reinterpret_cast<xhalf *>(&val[i]);
#pragma unroll
      for (int j = 0; j < 8; j++) s[j] += x[j];
    }

    if (residual_in != nullptr) {
      uint4 resval = residual_in[lineoffset_in + line];
      xhalf *y = reinterpret_cast<xhalf *>(&resval);
#pragma unroll
      for (int j = 0; j < 8; j++) s[j] += y[j];
      if (residual_out != nullptr) residual_out[lineoffset_in + line] = sum;
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

template <int UNROLL>
__global__ void __launch_bounds__(NVTE_UB_MAXTHREADS)
    userbuffers_fp16_sum_inplace_gpu_mc_lamport(const int RANKS, const int myrank,
                                                const int mylines, const int numlines,
                                                int *uc_flagptr, int *mc_flagptr, uint4 *mc_ptr_in,
                                                uint4 *mc_ptr_out, uint4 *uc_ptr_out,
                                                uint4 *clear_ptr, uint4 *residual_in,
                                                uint4 *residual_out, xhalf *gamma, float eps,
                                                const int hidden_size, bool fuse_layernorm) {
  // flags[0,1,2]: reduce_id, sm_sync-local, flag-barrier
  // those go right after rank UC pointers, but its the CPU caller who should account for it
  int reduce_id;
  __shared__ float s_variance;

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

  const int loop_step0 = blockDim.x;
  const int loop_step = loop_step0 * UNROLL * gridDim.x;
  const int start_elem = threadIdx.x + blockDim.x * blockIdx.x * UNROLL;
  const int end_elem = max(start_elem, mylines);

  for (int line = start_elem; line < end_elem; line += loop_step) {
    uint4 val[UNROLL];
    xhalf *x = reinterpret_cast<xhalf *>(&val[0]);
#pragma unroll
    for (int i = 0; i < UNROLL; i++) MULTIMEM_LD(val[i], mc_ptr_in + (line + i * loop_step0))

    if (residual_in != nullptr) {
      for (int i = 0; i < UNROLL; i++) {
        uint4 resval = residual_in[line + i * loop_step0];
        xhalf *y = reinterpret_cast<xhalf *>(&resval);
#pragma unroll
        for (int j = 0; j < 8; j++) x[i * 8 + j] += y[j];
        if (residual_out != nullptr) residual_out[line + i * loop_step0] = val[i];
      }
    }
    if (fuse_layernorm) {
      float local_var_sum = 0.0f;
      for (int j = 0; j < UNROLL * sizeof(int4) / sizeof(xhalf); j++)
        local_var_sum += (float)(x[j]) * (float)(x[j]);

      float packed[1] = {local_var_sum};
      blockReduceSumV2<float, 1>(packed);
      float variance = packed[0];

      if (threadIdx.x == 0) {
        variance = (variance / hidden_size);  // Var[x] = E[x²]
        s_variance = rsqrtf(variance + eps);
      }
      __syncthreads();
    }

    int i = 0;
#pragma unroll
    for (int g = 0; g < UNROLL; g++) {
      if (fuse_layernorm) {
#pragma unroll
        for (int j = 0; j < sizeof(int4) / sizeof(xhalf); j++) {
          x[i] =
              (xhalf)((float)(x[i]) * s_variance *
                      (float)
                          gamma[(threadIdx.x + g * loop_step0) * sizeof(int4) / sizeof(xhalf) + j]);
          i++;
        }
      }
      MULTIMEM_ST(val[g], mc_ptr_out + (line + g * loop_step0))
    }
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

#define split_tokens(x)                                                              \
  const int elements = bytes / sizeof(half);                                         \
  const int elements_per_thread = sizeof(uint4) / sizeof(half);                      \
  int nthreads = 1024, nlines = 4;                                                   \
  size_t total_bytes = bytes / ranks, start_bytes = myrank * total_bytes;            \
  int sms = x;                                                                       \
  if (hidden_size) {                                                                 \
    assert(hidden_size <= 32768);                                                    \
    assert(elements % hidden_size == 0);                                             \
    assert(hidden_size % elements_per_thread == 0);                                  \
    int ntokens = elements / hidden_size;                                            \
    int my_tokens = ntokens / ranks;                                                 \
    int extra_tokens = ntokens % ranks;                                              \
    int first_token = myrank * my_tokens;                                            \
    first_token += myrank < extra_tokens ? myrank : extra_tokens;                    \
    if (myrank < extra_tokens) my_tokens++;                                          \
    start_bytes = first_token * hidden_size * sizeof(half);                          \
    total_bytes = my_tokens * hidden_size * sizeof(half);                            \
    nthreads = hidden_size / elements_per_thread;                                    \
    nlines = 1;                                                                      \
    while (nthreads > 1024) {                                                        \
      nlines++;                                                                      \
      assert(nlines <= 4);                                                           \
      if ((hidden_size / elements_per_thread) % nlines == 0)                         \
        nthreads = ((hidden_size / elements_per_thread)) / nlines;                   \
    }                                                                                \
    if (sms > my_tokens) sms = my_tokens;                                            \
    if (sms == 0) sms = 1;                                                           \
  }                                                                                  \
  bool residual_in_global = residual_in != nullptr && residual_in != residual_out && \
                            residual_out != nullptr;  // out residual is always local

extern "C" void allreduce_2shot_mc(int ranks, int myrank, void *uc0ptr, void *mc0ptr,
                                   void *mcptr_in, void *mcptr_out, size_t bytes, void *residual_in,
                                   void *residual_out, bool fuse_layernorm, void *gamma, float eps,
                                   const int hidden_size, cudaStream_t stream) {
  split_tokens(32);

  SETUP_LAUNCH_CONFIG(sms, nthreads, stream, 4, 1);

  int arg1 = ranks, arg2 = myrank, arg3 = total_bytes / sizeof(uint4);
  void *arg4 = uc0ptr + (ranks * 8), *arg5 = mc0ptr + (ranks * 8), *arg6 = mcptr_in + start_bytes,
       *arg7 = mcptr_out + start_bytes,
       *arg8 = residual_in_global ? residual_in + start_bytes : residual_in, *arg9 = residual_out,
       *arg10 = gamma;
  float arg11 = eps;
  int arg12 = hidden_size;
  bool arg13 = fuse_layernorm;
  void *kernelArgs[] = {(void *)&arg1, (void *)&arg2,  (void *)&arg3,  (void *)&arg4,
                        (void *)&arg5, (void *)&arg6,  (void *)&arg7,  (void *)&arg8,
                        (void *)&arg9, (void *)&arg10, (void *)&arg11, (void *)&arg12,
                        (void *)&arg13};
#define call_mc_kernel(x, cond)                                                                   \
  if (x == nlines || cond) {                                                                      \
    CUDACHECK(                                                                                    \
        cudaLaunchKernelExC(&cfg, (void *)(userbuffers_fp16_sum_inplace_gpu_mc<x>), kernelArgs)); \
    return;                                                                                       \
  }
  call_mc_kernel(1, false);
  call_mc_kernel(2, false);
  call_mc_kernel(3, false);
  call_mc_kernel(4, true);
}

extern "C" void allreduce_2shot_uc(int ranks, int myrank, void *uc0ptr, void *ucptr_in,
                                   void *ucptr_out, size_t bytes, void *residual_in,
                                   void *residual_out, bool fuse_layernorm, void *gamma, float eps,
                                   const int hidden_size, cudaStream_t stream) {
  SETUP_LAUNCH_CONFIG(64, 1024, stream, 4, 1);

  int arg1 = myrank, arg2 = bytes / 16, arg3 = (int4 *)ucptr_in - (int4 *)uc0ptr,
      arg4 = (int4 *)ucptr_out - (int4 *)uc0ptr;
  void *arg5 = uc0ptr + (ranks * 8), **arg6 = (void **)uc0ptr, *arg7 = residual_in,
       *arg8 = residual_out, *arg9 = gamma;
  float arg10 = eps;
  int arg11 = hidden_size;
  bool arg12 = fuse_layernorm;
  void *kernelArgs[] = {(void *)&arg1, (void *)&arg2,  (void *)&arg3,  (void *)&arg4,
                        (void *)&arg5, (void *)&arg6,  (void *)&arg7,  (void *)&arg8,
                        (void *)&arg9, (void *)&arg10, (void *)&arg11, (void *)&arg12};
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
                                           void *residual_in, void *residual_out,
                                           bool fuse_layernorm, void *gamma, float eps,
                                           const int hidden_size, cudaStream_t stream) {
  if (!poisoned) {
    //user tells us destination was not pre-poisoned, so we need to do it before calling allreduce
    int threadsPerBlock = 512;
    int blocks = (bytes / 4 + threadsPerBlock - 1) / threadsPerBlock;
    memset_int<<<blocks, threadsPerBlock, 0, stream>>>((uint32_t *)ucptr_out, bytes / 4,
                                                       NVTE_UB_LAMPORT_INT);
  }
  split_tokens(64);

  SETUP_LAUNCH_CONFIG(64, nthreads, stream, 4, 1);

  int arg1 = ranks, arg2 = myrank, arg3 = total_bytes / sizeof(uint4),
      arg3a = bytes / sizeof(uint4);
  void *arg4 = uc0ptr + (ranks * 8), *arg5 = mc0ptr + (ranks * 8), *arg6 = mcptr_in + start_bytes,
       *arg7 = mcptr_out + start_bytes, *arg8 = ucptr_out, *arg9 = clear_ptr,
       *arg10 = residual_in_global ? residual_in + start_bytes : residual_in, *arg11 = residual_out,
       *arg12 = gamma;
  float arg13 = eps;
  int arg14 = hidden_size;
  bool arg15 = fuse_layernorm;
  void *kernelArgs[] = {(void *)&arg1,  (void *)&arg2,  (void *)&arg3,  (void *)&arg3a,
                        (void *)&arg4,  (void *)&arg5,  (void *)&arg6,  (void *)&arg7,
                        (void *)&arg8,  (void *)&arg9,  (void *)&arg10, (void *)&arg11,
                        (void *)&arg12, (void *)&arg13, (void *)&arg14, (void *)&arg15};

#define call_mc_lamport_kernel(x, cond)                                                           \
  if (x == nlines || cond) {                                                                      \
    CUDACHECK(cudaLaunchKernelExC(&cfg, (void *)(userbuffers_fp16_sum_inplace_gpu_mc_lamport<x>), \
                                  kernelArgs));                                                   \
    return;                                                                                       \
  }

  call_mc_lamport_kernel(1, false);
  call_mc_lamport_kernel(2, false);
  call_mc_lamport_kernel(3, false);
  call_mc_lamport_kernel(4, true);
}

}  // namespace transformer_engine
