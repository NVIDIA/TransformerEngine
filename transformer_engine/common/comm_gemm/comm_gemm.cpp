/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/comm_gemm.h"

#include <cublasmp.h>
#include <cuda_runtime.h>
#include <nvshmem.h>

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../common.h"
#include "../util/logging.h"

using namespace transformer_engine;

namespace {

// TODO: log warnings on failures of the *Destroy calls below, once TE has such ability.
// For now, just silently ignoring the errors, since the only diag available in TE is throwing
// exceptions, but these calls will typically be made from destructors, so cannot throw.

template <typename HandlePtr, typename CreateFn, typename DestroyFn, typename... Args>
auto CreateWithCudaCheck(CreateFn create_fn, DestroyFn destroy_fn, Args&&... args) {
  using Handle = std::remove_pointer_t<HandlePtr>;
  HandlePtr raw{};
  NVTE_CHECK_CUDA(create_fn(&raw, std::forward<Args>(args)...));
  return std::unique_ptr<Handle, DestroyFn>(raw, destroy_fn);
}

using CudaStream =
    std::unique_ptr<std::remove_pointer_t<cudaStream_t>, decltype(&cudaStreamDestroy)>;

CudaStream CudaStreamCreate() {
  return CreateWithCudaCheck<cudaStream_t>(cudaStreamCreate, cudaStreamDestroy);
}

using CudaEvent = std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, decltype(&cudaEventDestroy)>;

CudaEvent CudaEventCreate(unsigned flags) {
  return CreateWithCudaCheck<cudaEvent_t>(cudaEventCreateWithFlags, cudaEventDestroy, flags);
}

template <bool raw_last, typename HandlePtr, typename CreateFn, typename DestroyFn,
          typename... Args>
auto CreateWithCublasMpCheck(CreateFn create_fn, DestroyFn destroy_fn, Args&&... args) {
  using Handle = std::remove_pointer_t<HandlePtr>;
  HandlePtr raw{};
  if constexpr (raw_last) {
    NVTE_CHECK_CUBLASMP(create_fn(std::forward<Args>(args)..., &raw));
  } else {
    NVTE_CHECK_CUBLASMP(create_fn(&raw, std::forward<Args>(args)...));
  }
  return std::unique_ptr<Handle, DestroyFn>(raw, destroy_fn);
}

using CublasMp =
    std::unique_ptr<std::remove_pointer_t<cublasMpHandle_t>, decltype(&cublasMpDestroy)>;

CublasMp CublasMpCreate(cudaStream_t stream) {
  return CreateWithCublasMpCheck<false, cublasMpHandle_t>(cublasMpCreate, cublasMpDestroy, stream);
}

using CublasMpGrid =
    std::unique_ptr<std::remove_pointer_t<cublasMpGrid_t>, decltype(&cublasMpGridDestroy)>;

CublasMpGrid CublasMpGridCreate(int64_t nprow, int64_t npcol, cublasMpGridLayout_t layout,
                                ncclComm_t comm) {
  return CreateWithCublasMpCheck<true, cublasMpGrid_t>(cublasMpGridCreate, cublasMpGridDestroy,
                                                       nprow, npcol, layout, comm);
}

using CublasMpMatrixDesc = std::unique_ptr<std::remove_pointer_t<cublasMpMatrixDescriptor_t>,
                                           decltype(&cublasMpMatrixDescriptorDestroy)>;

CublasMpMatrixDesc CublasMpMatrixDescCreate(int64_t m, int64_t n, int64_t mb, int64_t nb,
                                            int64_t rsrc, int64_t csrc, int64_t lld,
                                            cudaDataType_t type, cublasMpGrid_t grid) {
  return CreateWithCublasMpCheck<true, cublasMpMatrixDescriptor_t>(
      cublasMpMatrixDescriptorCreate, cublasMpMatrixDescriptorDestroy, m, n, mb, nb, rsrc, csrc,
      lld, type, grid);
}

using CublasMpMatmulDesc = std::unique_ptr<std::remove_pointer_t<cublasMpMatmulDescriptor_t>,
                                           decltype(&cublasMpMatmulDescriptorDestroy)>;

CublasMpMatmulDesc CublasMpMatmulDescCreate(cublasComputeType_t compute_type) {
  return CreateWithCublasMpCheck<false, cublasMpMatmulDescriptor_t>(
      cublasMpMatmulDescriptorCreate, cublasMpMatmulDescriptorDestroy, compute_type);
}

}  // namespace

struct NVTECommGemmCtx {
  int64_t nranks;
  int64_t rank;
  ncclComm_t comm;
  CudaStream stream;
  CudaEvent event;
  CublasMp cublas_mp;
  CublasMpGrid grid_col_major;
  CublasMpGrid grid_row_major;
  CublasMpMatrixDesc a_desc;
  CublasMpMatrixDesc b_desc;
  CublasMpMatrixDesc d_desc;
  CublasMpMatmulDesc matmul_desc;
  void* workspace;
  size_t workspace_size;
};

namespace {

int64_t block_size(NVTECommGemmCtx* ctx, int64_t global_size) {
  // Use non-cyclic layout to maximize opportunity for comm overlap.
  return (global_size + ctx->nranks - 1) / ctx->nranks;
}

void AgGemmInitMatrices(NVTECommGemmCtx* ctx, int64_t* ldd, int64_t m, int64_t n, int64_t k,
                        const Tensor* a, const Tensor* b, const Tensor* d, bool transa,
                        bool transb) {
  const auto a0 = a->flat_first_dim();
  const auto a1 = a->flat_last_dim();
  const auto b0 = b->flat_first_dim();
  const auto b1 = b->flat_last_dim();
  const auto d0 = d->flat_first_dim();
  const auto d1 = d->flat_last_dim();

  if (transa) {
    NVTE_CHECK(a1 == k, "Unsupported tensor dimension in A: expected ", k, ", got ", a1);
    NVTE_CHECK_CUBLASMP(cublasMpMatrixDescriptorInit(k, m, k, block_size(ctx, m), 0, 0, k,
                                                     get_cuda_dtype(a->dtype()),
                                                     ctx->grid_row_major.get(), ctx->a_desc.get()));
  } else {
    NVTE_CHECK(a0 == k, "Unsupported tensor dimension in A: expected ", k, ", got ", a0);
    NVTE_CHECK_CUBLASMP(cublasMpMatrixDescriptorInit(m, k, block_size(ctx, m), k, 0, 0,
                                                     block_size(ctx, m), get_cuda_dtype(a->dtype()),
                                                     ctx->grid_col_major.get(), ctx->a_desc.get()));
  }
  if (transb) {
    NVTE_CHECK(b0 == k, "Unsupported tensor dimensionin B: expected ", k, ", got ", b0);
    NVTE_CHECK_CUBLASMP(cublasMpMatrixDescriptorInit(n, k, block_size(ctx, n), k, 0, 0,
                                                     block_size(ctx, n), get_cuda_dtype(b->dtype()),
                                                     ctx->grid_col_major.get(), ctx->b_desc.get()));
  } else {
    NVTE_CHECK(b1 == k, "Unsupported tensor dimension in B: expected ", k, ", got ", b1);
    NVTE_CHECK_CUBLASMP(cublasMpMatrixDescriptorInit(k, n, k, block_size(ctx, n), 0, 0, k,
                                                     get_cuda_dtype(b->dtype()),
                                                     ctx->grid_row_major.get(), ctx->b_desc.get()));
  }
  NVTE_CHECK(d0 == n, "Unsupported tensor dimension in D: expected ", n, ", got ", d0);
  *ldd = block_size(ctx, m);
  NVTE_CHECK_CUBLASMP(cublasMpMatrixDescriptorInit(m, n, block_size(ctx, m), block_size(ctx, n), 0,
                                                   0, *ldd, get_cuda_dtype(d->dtype()),
                                                   ctx->grid_col_major.get(), ctx->d_desc.get()));
}

void GemmRsInitMatrices(NVTECommGemmCtx* ctx, int64_t* ldd, int64_t m, int64_t n, int64_t k,
                        const Tensor* a, const Tensor* b, const Tensor* d, bool transa,
                        bool transb) {
  const auto a0 = a->flat_first_dim();
  const auto a1 = a->flat_last_dim();
  const auto b0 = b->flat_first_dim();
  const auto b1 = b->flat_last_dim();
  const auto d0 = d->flat_first_dim();
  const auto d1 = d->flat_last_dim();

  if (transa) {
    NVTE_CHECK(a0 == m, "Unsupported tensor dimension in A: expected ", m, ", got ", a0);
    NVTE_CHECK_CUBLASMP(cublasMpMatrixDescriptorInit(k, m, block_size(ctx, k), m, 0, 0,
                                                     block_size(ctx, k), get_cuda_dtype(a->dtype()),
                                                     ctx->grid_col_major.get(), ctx->a_desc.get()));
  } else {
    NVTE_CHECK(a1 == m, "Unsupported tensor dimension in A: expected ", m, ", got ", a1);
    NVTE_CHECK_CUBLASMP(cublasMpMatrixDescriptorInit(m, k, m, block_size(ctx, k), 0, 0, m,
                                                     get_cuda_dtype(a->dtype()),
                                                     ctx->grid_row_major.get(), ctx->a_desc.get()));
  }
  if (transb) {
    NVTE_CHECK(b1 == n, "Unsupported tensor dimension in B: expected ", n, ", got ", b1);
    NVTE_CHECK_CUBLASMP(cublasMpMatrixDescriptorInit(
        n, k, block_size(ctx, n), block_size(ctx, k), 0, 0, block_size(ctx, n),
        get_cuda_dtype(b->dtype()), ctx->grid_row_major.get(), ctx->b_desc.get()));
  } else {
    NVTE_CHECK(b0 == n, "Unsupported tensor dimension in B: expected ", n, ", got ", b0);
    NVTE_CHECK_CUBLASMP(cublasMpMatrixDescriptorInit(
        k, n, block_size(ctx, k), block_size(ctx, n), 0, 0, block_size(ctx, k),
        get_cuda_dtype(b->dtype()), ctx->grid_col_major.get(), ctx->b_desc.get()));
  }
  NVTE_CHECK(d1 == m, "Unsupported tensor dimension in D: expected ", m, ", got ", d1);
  *ldd = m;
  NVTE_CHECK_CUBLASMP(cublasMpMatrixDescriptorInit(m, n, m, block_size(ctx, n), 0, 0, *ldd,
                                                   get_cuda_dtype(d->dtype()),
                                                   ctx->grid_row_major.get(), ctx->d_desc.get()));
}

void GemmArInitMatrices(NVTECommGemmCtx* ctx, int64_t* ldd, int64_t m, int64_t n, int64_t k,
                        const Tensor* a, const Tensor* b, const Tensor* d, bool transa,
                        bool transb) {
  const auto a0 = a->flat_first_dim();
  const auto a1 = a->flat_last_dim();
  const auto b0 = b->flat_first_dim();
  const auto b1 = b->flat_last_dim();
  const auto d0 = d->flat_first_dim();
  const auto d1 = d->flat_last_dim();

  if (transa) {
    NVTE_CHECK(a0 == m, "Unsupported tensor dimension in A: expected ", m, ", got ", a0);
    NVTE_CHECK_CUBLASMP(cublasMpMatrixDescriptorInit(k, m, block_size(ctx, k), m, 0, 0,
                                                     block_size(ctx, k), get_cuda_dtype(a->dtype()),
                                                     ctx->grid_col_major.get(), ctx->a_desc.get()));
  } else {
    NVTE_ERROR("N transpose flag is not supported for input A");
  }
  if (transb) {
    NVTE_ERROR("T transpose flag is not supported for input B");
  } else {
    NVTE_CHECK(b0 == n, "Unsupported tensor dimension in B: expected ", n, ", got ", b0);
    NVTE_CHECK_CUBLASMP(cublasMpMatrixDescriptorInit(k, n, block_size(ctx, k), n, 0, 0,
                                                     block_size(ctx, k), get_cuda_dtype(b->dtype()),
                                                     ctx->grid_col_major.get(), ctx->b_desc.get()));
  }
  NVTE_CHECK(d1 == m, "Unsupported tensor dimension in D: expected ", m, ", got ", d1);
  *ldd = m;
  NVTE_CHECK_CUBLASMP(cublasMpMatrixDescriptorInit(m, n * ctx->nranks, m, n, 0, 0, *ldd,
                                                   get_cuda_dtype(d->dtype()),
                                                   ctx->grid_row_major.get(), ctx->d_desc.get()));

  const cublasMpMatmulEpilogue_t epilogue = CUBLASMP_MATMUL_EPILOGUE_ALLREDUCE;
  NVTE_CHECK_CUBLASMP(cublasMpMatmulDescriptorAttributeSet(
      ctx->matmul_desc.get(), CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE, &epilogue,
      sizeof epilogue));
}

using InitMatricesFn = void (*)(NVTECommGemmCtx*, int64_t*, int64_t, int64_t, int64_t,
                                const Tensor*, const Tensor*, const Tensor*, bool, bool);

cublasMpMatmulAlgoType_t cublasmp_algo(NVTECommGemmAlgoType algo) {
  static const std::unordered_map<NVTECommGemmAlgoType, cublasMpMatmulAlgoType_t> s_map{
      {kNVTECommGemmAlgoDefault, CUBLASMP_MATMUL_ALGO_TYPE_DEFAULT},
      {kNVTECommGemmAlgoSplitP2P, CUBLASMP_MATMUL_ALGO_TYPE_SPLIT_P2P},
      {kNVTECommGemmAlgoSplitMulticast, CUBLASMP_MATMUL_ALGO_TYPE_SPLIT_MULTICAST},
      {kNVTECommGemmAlgoAtomicP2P, CUBLASMP_MATMUL_ALGO_TYPE_ATOMIC_P2P},
      {kNVTECommGemmAlgoAtomicMulticast, CUBLASMP_MATMUL_ALGO_TYPE_ATOMIC_MULTICAST},
  };
  auto it = s_map.find(algo);
  return it != s_map.end() ? it->second : static_cast<cublasMpMatmulAlgoType_t>(algo);
}

void cublasmp_gemm(InitMatricesFn init_matrices_fn, NVTECommGemmCtx* ctx, NVTECommGemmAlgoType algo,
                   int64_t m, int64_t n, int64_t k, const Tensor* a, const Tensor* b,
                   const Tensor* d, const Tensor* bias, const Tensor* pre_act_out, bool transa,
                   bool transb, bool grad, bool accumulate, int comm_sm_count,
                   cudaStream_t main_stream) {
  for (auto t : {a, b, d}) {
    NVTE_CHECK(is_tensor_scaling(t->scaling_mode),
               "Unsupported scaling mode: " + std::to_string(t->scaling_mode));
  }

  NVTE_CHECK_CUBLASMP(cublasMpMatmulDescriptorInit(ctx->matmul_desc.get(), CUBLAS_COMPUTE_32F));

  int64_t ldd{};
  init_matrices_fn(ctx, &ldd, m, n, k, a, b, d, transa, transb);

  const cublasOperation_t trans_a = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t trans_b = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
  NVTE_CHECK_CUBLASMP(cublasMpMatmulDescriptorAttributeSet(
      ctx->matmul_desc.get(), CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSA, &trans_a,
      sizeof trans_a));
  NVTE_CHECK_CUBLASMP(cublasMpMatmulDescriptorAttributeSet(
      ctx->matmul_desc.get(), CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSB, &trans_b,
      sizeof trans_b));
  cublasMpMatmulAlgoType_t algo_attr = cublasmp_algo(algo);
  NVTE_CHECK_CUBLASMP(cublasMpMatmulDescriptorAttributeSet(
      ctx->matmul_desc.get(), CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_ALGO_TYPE, &algo_attr,
      sizeof algo_attr));

  const cublasMpMatmulMatrixScale_t scale_mode = CUBLASMP_MATMUL_MATRIX_SCALE_SCALAR_FP32;
  if (is_fp8_dtype(a->dtype())) {
    NVTE_CHECK(a->scale_inv.dptr, "Scaling must be set for FP8 dtype");
    NVTE_CHECK_CUBLASMP(cublasMpMatmulDescriptorAttributeSet(
        ctx->matmul_desc.get(), CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_A_SCALE_MODE, &scale_mode,
        sizeof scale_mode));
    NVTE_CHECK_CUBLASMP(cublasMpMatmulDescriptorAttributeSet(
        ctx->matmul_desc.get(), CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_A_SCALE_POINTER,
        &a->scale_inv.dptr, sizeof(void*)));
  }
  if (is_fp8_dtype(b->dtype())) {
    NVTE_CHECK(b->scale_inv.dptr, "Scaling must be set for FP8 dtype");
    NVTE_CHECK_CUBLASMP(cublasMpMatmulDescriptorAttributeSet(
        ctx->matmul_desc.get(), CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_B_SCALE_MODE, &scale_mode,
        sizeof scale_mode));
    NVTE_CHECK_CUBLASMP(cublasMpMatmulDescriptorAttributeSet(
        ctx->matmul_desc.get(), CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_B_SCALE_POINTER,
        &b->scale_inv.dptr, sizeof(void*)));
  }
  if (is_fp8_dtype(d->dtype())) {
    NVTE_CHECK(d->scale.dptr, "Scaling must be set for FP8 dtype");
    NVTE_CHECK_CUBLASMP(cublasMpMatmulDescriptorAttributeSet(
        ctx->matmul_desc.get(), CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_D_SCALE_MODE, &scale_mode,
        sizeof scale_mode));
    NVTE_CHECK_CUBLASMP(cublasMpMatmulDescriptorAttributeSet(
        ctx->matmul_desc.get(), CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_D_SCALE_POINTER,
        &d->scale.dptr, sizeof(void*)));
    if (d->amax.dptr) {
      NVTE_CHECK_CUBLASMP(cublasMpMatmulDescriptorAttributeSet(
          ctx->matmul_desc.get(), CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_AMAX_D_POINTER,
          &d->amax.dptr, sizeof(void*)));
    }
  }

  // Might be set to ALLREDUCE before, need to OR with the new flags to set.
  cublasMpMatmulEpilogue_t epilogue{};
  size_t size_read{};
  NVTE_CHECK_CUBLASMP(cublasMpMatmulDescriptorAttributeGet(
      ctx->matmul_desc.get(), CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE, &epilogue,
      sizeof epilogue, &size_read));
  NVTE_CHECK(size_read == sizeof epilogue);
  // (bias, gelu, grad) -> epilogue
  const std::map<std::tuple<bool, bool, bool>, cublasMpMatmulEpilogue_t> flags_to_epilogue{
      {{true, true, false}, CUBLASMP_MATMUL_EPILOGUE_GELU_AUX_BIAS},
      {{true, true, true}, CUBLASMP_MATMUL_EPILOGUE_DGELU_BGRAD},
      {{true, false, false}, CUBLASMP_MATMUL_EPILOGUE_BIAS},
      {{true, false, true}, CUBLASMP_MATMUL_EPILOGUE_BGRADB},
      {{false, true, false}, CUBLASMP_MATMUL_EPILOGUE_GELU_AUX},
      {{false, true, true}, CUBLASMP_MATMUL_EPILOGUE_DGELU},
  };
  if (auto it =
          flags_to_epilogue.find({bias ? bias->data.dptr != nullptr : false,
                                  pre_act_out ? pre_act_out->data.dptr != nullptr : false, grad});
      it != flags_to_epilogue.end()) {
    epilogue = static_cast<cublasMpMatmulEpilogue_t>(epilogue | it->second);
    NVTE_CHECK_CUBLASMP(cublasMpMatmulDescriptorAttributeSet(
        ctx->matmul_desc.get(), CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE, &epilogue,
        sizeof epilogue));
  }

  if (bias && bias->data.dptr) {
    cudaDataType_t bias_type = get_cuda_dtype(bias->data.dtype);
    NVTE_CHECK_CUBLASMP(cublasMpMatmulDescriptorAttributeSet(
        ctx->matmul_desc.get(), CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_BIAS_DATA_TYPE, &bias_type,
        sizeof bias_type));
    NVTE_CHECK_CUBLASMP(cublasMpMatmulDescriptorAttributeSet(
        ctx->matmul_desc.get(), CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_BIAS_POINTER, &bias->data.dptr,
        sizeof bias->data.dptr));
  }

  if (pre_act_out && pre_act_out->data.dptr) {
    cudaDataType_t aux_type = get_cuda_dtype(pre_act_out->data.dtype);
    NVTE_CHECK_CUBLASMP(cublasMpMatmulDescriptorAttributeSet(
        ctx->matmul_desc.get(), CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_DATA_TYPE,
        &aux_type, sizeof aux_type));
    NVTE_CHECK_CUBLASMP(cublasMpMatmulDescriptorAttributeSet(
        ctx->matmul_desc.get(), CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_POINTER,
        &pre_act_out->data.dptr, sizeof pre_act_out->data.dptr));
    NVTE_CHECK_CUBLASMP(cublasMpMatmulDescriptorAttributeSet(
        ctx->matmul_desc.get(), CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_LD, &ldd,
        sizeof ldd));
    if (is_fp8_dtype(pre_act_out->dtype())) {
      NVTE_CHECK(pre_act_out->scale.dptr, "Scaling must be set for FP8 dtype");
      NVTE_CHECK_CUBLASMP(cublasMpMatmulDescriptorAttributeSet(
          ctx->matmul_desc.get(), CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_SCALE_MODE,
          &scale_mode, sizeof scale_mode));
      NVTE_CHECK_CUBLASMP(cublasMpMatmulDescriptorAttributeSet(
          ctx->matmul_desc.get(), CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_SCALE_POINTER,
          &pre_act_out->scale.dptr, sizeof(void*)));
      if (pre_act_out->amax.dptr) {
        NVTE_CHECK_CUBLASMP(cublasMpMatmulDescriptorAttributeSet(
            ctx->matmul_desc.get(), CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE_AUX_AMAX_POINTER,
            &pre_act_out->amax.dptr, sizeof(void*)));
      }
    }
  }

  if (comm_sm_count) {
    NVTE_CHECK_CUBLASMP(cublasMpMatmulDescriptorAttributeSet(
        ctx->matmul_desc.get(), CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_COMMUNICATION_SM_COUNT,
        &comm_sm_count, sizeof comm_sm_count));
  }

  NVTE_CHECK_CUBLASMP(cublasMpStreamSet(ctx->cublas_mp.get(), main_stream));

  size_t wrksp_size_device{};
  size_t wrksp_size_host{};

  float alpha = 1.0;
  float beta = accumulate ? 1.0 : 0.0;
  std::tuple args{ctx->cublas_mp.get(),
                  ctx->matmul_desc.get(),
                  m,
                  n,
                  k,
                  &alpha,
                  a->data.dptr,
                  1,
                  1,
                  ctx->a_desc.get(),
                  b->data.dptr,
                  1,
                  1,
                  ctx->b_desc.get(),
                  &beta,
                  accumulate ? d->data.dptr : nullptr,
                  1,
                  1,
                  accumulate ? ctx->d_desc.get() : nullptr,
                  d->data.dptr,
                  1,
                  1,
                  ctx->d_desc.get()};
  NVTE_CHECK_CUBLASMP(
      std::apply(cublasMpMatmul_bufferSize,
                 std::tuple_cat(args, std::tuple{&wrksp_size_device, &wrksp_size_host})));

  std::vector<uint8_t> workspace_host(wrksp_size_host);
  if (ctx->workspace_size < wrksp_size_device) {
    nvshmem_free(ctx->workspace);
    ctx->workspace = nvshmem_malloc(wrksp_size_device);
    ctx->workspace_size = wrksp_size_device;
  }

  NVTE_CHECK_CUBLASMP(
      std::apply(cublasMpMatmul,
                 std::tuple_cat(args, std::tuple{ctx->workspace, ctx->workspace_size,
                                                 workspace_host.data(), workspace_host.size()})));

  NVTE_CHECK_CUDA(cudaEventRecord(ctx->event.get(), main_stream));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(ctx->stream.get(), ctx->event.get(), 0));
}

}  // namespace

NVTECommGemmCtx* nvte_comm_gemm_ctx_create(ncclComm_t comm, int nranks, int rank) {
  NVTE_API_CALL(nvte_comm_gemm_ctx_create);
  auto stream = CudaStreamCreate();
  auto event = CudaEventCreate(cudaEventDisableTiming);
  auto cublas_mp = CublasMpCreate(stream.get());

  auto col_major = CublasMpGridCreate(nranks, 1, CUBLASMP_GRID_LAYOUT_COL_MAJOR, comm);
  auto row_major = CublasMpGridCreate(1, nranks, CUBLASMP_GRID_LAYOUT_ROW_MAJOR, comm);

  // Pre-creating matrix descriptors here, will be initialized with the actual params later.
  auto a_desc = CublasMpMatrixDescCreate(1, 1, 1, 1, 0, 0, 1, CUDA_R_16F, row_major.get());
  auto b_desc = CublasMpMatrixDescCreate(1, 1, 1, 1, 0, 0, 1, CUDA_R_16F, row_major.get());
  auto d_desc = CublasMpMatrixDescCreate(1, 1, 1, 1, 0, 0, 1, CUDA_R_16F, row_major.get());

  auto matmul_desc = CublasMpMatmulDescCreate(CUBLAS_COMPUTE_32F);

  return new NVTECommGemmCtx{
      .nranks = nranks,
      .rank = rank,
      .comm = comm,
      .stream = std::move(stream),
      .event = std::move(event),
      .cublas_mp = std::move(cublas_mp),
      .grid_col_major = std::move(col_major),
      .grid_row_major = std::move(row_major),
      .a_desc = std::move(a_desc),
      .b_desc = std::move(b_desc),
      .d_desc = std::move(d_desc),
      .matmul_desc = std::move(matmul_desc),
  };
}

void nvte_comm_gemm_ctx_destroy(NVTECommGemmCtx* ctx) {
  NVTE_API_CALL(nvte_comm_gemm_ctx_destroy);
  nvshmemx_sync_all_on_stream(ctx->stream.get());
  delete ctx;
}

void nvte_all_gather_gemm(NVTECommGemmCtx* ctx, int64_t m, int64_t n, int64_t k, const NVTETensor a,
                          const NVTETensor b, const NVTETensor d, const NVTETensor bias,
                          const NVTETensor pre_act_out, bool transa, bool transb, bool grad,
                          bool accumulate, int comm_sm_count, cudaStream_t main_stream,
                          NVTECommGemmAlgoType algo) {
  NVTE_API_CALL(nvte_all_gather_gemm);
  cublasmp_gemm(AgGemmInitMatrices, ctx, algo, m, n, k, convertNVTETensorCheck(a),
                convertNVTETensorCheck(b), convertNVTETensorCheck(d), convertNVTETensorCheck(bias),
                convertNVTETensorCheck(pre_act_out), transa, transb, grad, accumulate,
                comm_sm_count, main_stream);
}

void nvte_gemm_reduce_scatter(NVTECommGemmCtx* ctx, int64_t m, int64_t n, int64_t k,
                              const NVTETensor a, const NVTETensor b, const NVTETensor d,
                              const NVTETensor bias, const NVTETensor pre_act_out, bool transa,
                              bool transb, bool grad, bool accumulate, int comm_sm_count,
                              cudaStream_t main_stream, NVTECommGemmAlgoType algo) {
  NVTE_API_CALL(nvte_gemm_reduce_scatter);
  cublasmp_gemm(GemmRsInitMatrices, ctx, algo, m, n, k, convertNVTETensorCheck(a),
                convertNVTETensorCheck(b), convertNVTETensorCheck(d), convertNVTETensorCheck(bias),
                convertNVTETensorCheck(pre_act_out), transa, transb, grad, accumulate,
                comm_sm_count, main_stream);
}

void nvte_gemm_all_reduce(NVTECommGemmCtx* ctx, int64_t m, int64_t n, int64_t k, const NVTETensor a,
                          const NVTETensor b, const NVTETensor d, const NVTETensor bias,
                          const NVTETensor pre_act_out, bool transa, bool transb, bool grad,
                          bool accumulate, int comm_sm_count, cudaStream_t main_stream,
                          NVTECommGemmAlgoType algo) {
  NVTE_API_CALL(nvte_gemm_all_reduce);
  cublasmp_gemm(GemmArInitMatrices, ctx, algo, m, n, k, convertNVTETensorCheck(a),
                convertNVTETensorCheck(b), convertNVTETensorCheck(d), convertNVTETensorCheck(bias),
                convertNVTETensorCheck(pre_act_out), transa, transb, grad, accumulate,
                comm_sm_count, main_stream);
}

int64_t nvte_comm_gemm_numroc(NVTECommGemmCtx* ctx, int64_t global_size) {
  NVTE_API_CALL(nvte_comm_gemm_numroc);
  return cublasMpNumroc(global_size, block_size(ctx, global_size), ctx->rank, 0, ctx->nranks);
}
