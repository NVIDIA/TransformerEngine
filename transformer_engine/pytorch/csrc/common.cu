/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common.h"
#include "transformer_engine/transformer_engine.h"


transformer_engine::DType getTransformerEngineFP8Type(bool e4m3_if_hybrid,
                                                      const std::string &fp8_recipe) {
    // if e4m3 or hybrid + forward
    if ( (fp8_recipe == "E4M3") || ( (fp8_recipe == "HYBRID") && e4m3_if_hybrid ) ) {
        return transformer_engine::DType::kFloat8E4M3;
    }
    return transformer_engine::DType::kFloat8E5M2;
}

transformer_engine::TensorWrapper makeTransformerEngineTensor(
    void* data_ptr,
    const NVTEShape& shape,
    const transformer_engine::DType type) {
  return transformer_engine::TensorWrapper(data_ptr, shape, type);
}


transformer_engine::TensorWrapper makeTransformerEngineTensor(
    void* data_ptr,
    const std::vector<size_t>& shape,
    const transformer_engine::DType type) {
  return transformer_engine::TensorWrapper(data_ptr, shape, type);
}


transformer_engine::TensorWrapper makeTransformerEngineTensor(at::Tensor tensor) {
    transformer_engine::DType dtype = GetTransformerEngineDType(tensor.scalar_type());
    std::vector<size_t> shape;

    for (auto s : tensor.sizes()) {
        shape.push_back(s);
    }
    return makeTransformerEngineTensor(tensor.data_ptr(), shape, dtype);
}


transformer_engine::TensorWrapper makeTransformerEngineTensor(
    void* data_ptr,
    const std::vector<size_t>& shape,
    const transformer_engine::DType type,
    void* amax_ptr,
    void* scale_ptr,
    void* scale_inv_ptr) {
  return transformer_engine::TensorWrapper(data_ptr, shape, type,
                                           reinterpret_cast<float*>(amax_ptr),
                                           reinterpret_cast<float*>(scale_ptr),
                                           reinterpret_cast<float*>(scale_inv_ptr));
}


transformer_engine::TensorWrapper makeTransformerEngineTensor(at::Tensor tensor,
                                                              at::Tensor amax,
                                                              const at::Tensor scale,
                                                              at::Tensor scale_inv) {
    transformer_engine::DType dtype = GetTransformerEngineDType(tensor.scalar_type());
    std::vector<size_t> shape;

    for (auto s : tensor.sizes()) {
        shape.push_back(s);
    }
    NVTE_CHECK(amax.scalar_type() == at::kFloat);
    NVTE_CHECK(scale.scalar_type() == at::kFloat);
    NVTE_CHECK(scale_inv.scalar_type() == at::kFloat);

    return makeTransformerEngineTensor(tensor.data_ptr(), shape, dtype,
                                       amax.data_ptr(),
                                       scale.data_ptr(),
                                       scale_inv.data_ptr());
}


size_t product(const std::vector<size_t> &shape) {
    size_t ret = 1;
    for (auto s : shape) {
        ret *= s;
    }
    return ret;
}


at::Tensor allocateSpace(const NVTEShape &shape,
                         const transformer_engine::DType type,
                         bool init_to_zeros) {
    auto size = shape.ndim;
    if (size == 2 && init_to_zeros) {
        return at::zeros({static_cast<int64_t>(shape.data[0]),
                          static_cast<int64_t>(shape.data[1])},
                          at::CUDA(GetATenDType(type)));
    } else if (size == 2) {
        return at::empty({static_cast<int64_t>(shape.data[0]),
                          static_cast<int64_t>(shape.data[1])},
                          at::CUDA(GetATenDType(type)));
    } else if (size == 1 && init_to_zeros) {
        return at::zeros({static_cast<int64_t>(shape.data[0])}, at::CUDA(GetATenDType(type)));
    } else if (size == 1) {
        return at::empty({static_cast<int64_t>(shape.data[0])}, at::CUDA(GetATenDType(type)));
    }
    NVTE_CHECK(false, "Should never reach here! func: allocateSpace");
}


at::Tensor allocateTorchTensor(int M,
                               int N,
                               transformer_engine::DType dtype
) {
    return at::empty({static_cast<int64_t>(M), static_cast<int64_t>(N)},
                     at::CUDA(GetATenDType(dtype)));
}


at::Tensor allocateTorchTensor(int M,
                               transformer_engine::DType dtype
) {
    return at::empty({static_cast<int64_t>(M)},
                     at::CUDA(GetATenDType(dtype)));
}


void dispatch_layernorm(void* input,                                    // i
                        const std::vector<size_t>& input_shape,
                        const transformer_engine::DType input_type,
                        void* gamma,                                    // i
                        const std::vector<size_t>& gamma_shape,
                        const transformer_engine::DType gamma_type,
                        void* beta,                                     // i
                        const std::vector<size_t>& beta_shape,
                        const transformer_engine::DType beta_type,
                        void* scale,                                    // i
                        const std::vector<size_t>& scale_shape,
                        const transformer_engine::DType scale_type,
                        const float epsilon,                            // i
                        void* z,                                        // o
                        const std::vector<size_t>& z_shape,
                        const transformer_engine::DType z_type,
                        void* mu,                                       // o
                        const std::vector<size_t>& mu_shape,
                        const transformer_engine::DType mu_type,
                        void* rsigma,                                   // o
                        const std::vector<size_t>& rsigma_shape,
                        const transformer_engine::DType rsigma_type,
                        void* amax,                                     // o
                        const std::vector<size_t>& amax_shape,
                        const transformer_engine::DType amax_type,
                        void* scale_inv,                                // o
                        const std::vector<size_t>& scale_inv_shape,
                        const transformer_engine::DType scale_inv_type,
                        const int multiProcessorCount
) {
    auto input_cu     = makeTransformerEngineTensor(input, input_shape, input_type);
    auto gamma_cu     = makeTransformerEngineTensor(gamma, gamma_shape, gamma_type);
    auto beta_cu      = makeTransformerEngineTensor(beta, beta_shape, beta_type);
    auto z_cu         = makeTransformerEngineTensor(z, z_shape, z_type, amax, scale, scale_inv);
    auto mu_cu        = makeTransformerEngineTensor(mu, mu_shape, mu_type);
    auto rsigma_cu    = makeTransformerEngineTensor(rsigma, rsigma_shape, rsigma_type);
    transformer_engine::TensorWrapper workspace, barrier;

    // This call populates workspace and barrier tensors with the required config
    nvte_layernorm_fwd(input_cu.data(), gamma_cu.data(), beta_cu.data(),
                       epsilon, z_cu.data(), mu_cu.data(), rsigma_cu.data(),
                       at::cuda::getCurrentCUDAStream(), multiProcessorCount,
                       workspace.data(), barrier.data());

    // Fill workspace and barrier
    auto workspace_data = allocateSpace(workspace.shape(),
                                        workspace.dtype());
    auto barrier_data = allocateSpace(barrier.shape(),
                                      barrier.dtype(),
                                      true);
    workspace = makeTransformerEngineTensor(workspace_data.data_ptr(),
                                            workspace.shape(),
                                            workspace.dtype());
    barrier   = makeTransformerEngineTensor(barrier_data.data_ptr(),
                                            barrier.shape(),
                                            barrier.dtype());

    // Actual call to fwd kernel
    nvte_layernorm_fwd(input_cu.data(), gamma_cu.data(), beta_cu.data(),
                       epsilon, z_cu.data(), mu_cu.data(), rsigma_cu.data(),
                       at::cuda::getCurrentCUDAStream(), multiProcessorCount,
                       workspace.data(), barrier.data());
}


void dispatch_cast_transpose_fusion(void* input,                                            // i
                                    const std::vector<size_t>& input_shape,
                                    const transformer_engine::DType input_type,
                                    void* scale,                                            // i
                                    const std::vector<size_t>& scale_shape,
                                    const transformer_engine::DType scale_type,
                                    void* output_cast,                                      // o
                                    const std::vector<size_t>& output_cast_shape,
                                    const transformer_engine::DType output_cast_type,
                                    void* output_transpose,                                 // o
                                    const std::vector<size_t>& output_transpose_shape,
                                    const transformer_engine::DType output_transpose_type,
                                    void* amax,                                             // o
                                    const std::vector<size_t>& amax_shape,
                                    const transformer_engine::DType amax_type,
                                    void* scale_inv,                                        // o
                                    const std::vector<size_t>& scale_inv_shape,
                                    const transformer_engine::DType scale_inv_type
) {
    auto input_cu            = makeTransformerEngineTensor(input, input_shape, input_type);
    auto output_cast_cu      = makeTransformerEngineTensor(output_cast, output_cast_shape,
                                                           output_cast_type, amax, scale,
                                                           scale_inv);
    auto output_transpose_cu = makeTransformerEngineTensor(output_transpose, output_transpose_shape,
                                                           output_transpose_type, amax,
                                                           scale, scale_inv);

    nvte_cast_transpose(input_cu.data(), output_cast_cu.data(), output_transpose_cu.data(),
                        at::cuda::getCurrentCUDAStream());
}


void dispatch_gelu(void* input,                                            // i
                   const std::vector<size_t>& input_shape,
                   const transformer_engine::DType input_type,
                   void* scale,                                            // i
                   const std::vector<size_t>& scale_shape,
                   const transformer_engine::DType scale_type,
                   void* output,                                           // o
                   const std::vector<size_t>& output_shape,
                   const transformer_engine::DType output_type,
                   void* amax,                                             // o
                   const std::vector<size_t>& amax_shape,
                   const transformer_engine::DType amax_type,
                   void* scale_inv,                                        // o
                   const std::vector<size_t>& scale_inv_shape,
                   const transformer_engine::DType scale_inv_type
) {
    auto input_cu =     makeTransformerEngineTensor(input, input_shape, input_type);
    auto output_cu =    makeTransformerEngineTensor(output, output_shape, output_type,
                                                    amax, scale, scale_inv);

    nvte_gelu(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());
}


void dispatch_transpose(void* input,                                            // i
                        const std::vector<size_t>& input_shape,
                        const transformer_engine::DType input_type,
                        void* output,                                           // o
                        const std::vector<size_t>& output_shape,
                        const transformer_engine::DType output_type
) {
    auto input_cu  = makeTransformerEngineTensor(input, input_shape, input_type);
    auto output_cu = makeTransformerEngineTensor(output, output_shape, output_type);

    nvte_transpose(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());
}


void dispatch_bgrad_cast_transpose_fusion(void* input,                                          // i
                                          const std::vector<size_t>& input_shape,
                                          const transformer_engine::DType input_type,
                                          void* scale,                                          // i
                                          const std::vector<size_t>& scale_shape,
                                          const transformer_engine::DType scale_type,
                                          void* cast_output,                                    // o
                                          const std::vector<size_t>& cast_output_shape,
                                          const transformer_engine::DType cast_output_type,
                                          void* transposed_output,                              // o
                                          const std::vector<size_t>& transposed_output_shape,
                                          const transformer_engine::DType transposed_output_type,
                                          void* amax,                                           // o
                                          const std::vector<size_t>& amax_shape,
                                          const transformer_engine::DType amax_type,
                                          void* dbias,                                          // o
                                          const std::vector<size_t>& dbias_shape,
                                          const transformer_engine::DType dbias_type,
                                          void* scale_inv,                                      // o
                                          const std::vector<size_t>& scale_inv_shape,
                                          const transformer_engine::DType scale_inv_type
) {
  auto input_cu             = makeTransformerEngineTensor(input, input_shape, input_type);
  auto cast_output_cu       = makeTransformerEngineTensor(cast_output, cast_output_shape,
                                                          cast_output_type, amax, scale,
                                                          scale_inv);
  auto transposed_output_cu = makeTransformerEngineTensor(transposed_output,
                                                          transposed_output_shape,
                                                          transposed_output_type,
                                                          amax, scale, scale_inv);
  auto dbias_cu             = makeTransformerEngineTensor(dbias, dbias_shape, dbias_type);
  transformer_engine::TensorWrapper workspace;

  nvte_cast_transpose_dbias(input_cu.data(), cast_output_cu.data(),
                            transposed_output_cu.data(), dbias_cu.data(),
                            workspace.data(), at::cuda::getCurrentCUDAStream());

  // Fill workspace
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace = makeTransformerEngineTensor(workspace_data.data_ptr(),
                                          workspace.shape(),
                                          workspace.dtype());

  nvte_cast_transpose_dbias(input_cu.data(), cast_output_cu.data(),
                            transposed_output_cu.data(), dbias_cu.data(),
                            workspace.data(), at::cuda::getCurrentCUDAStream());
}


void dispatch_bgrad_dgelu_cast_transpose_fusion(
    void* input,                                            // i
    const std::vector<size_t>& input_shape,
    const transformer_engine::DType input_type,
    void* gelu_input,                                       // i
    const std::vector<size_t>& gelu_input_shape,
    const transformer_engine::DType gelu_input_type,
    void* scale,                                            // i
    const std::vector<size_t>& scale_shape,
    const transformer_engine::DType scale_type,
    void* cast_output,                                      // o
    const std::vector<size_t>& cast_output_shape,
    const transformer_engine::DType cast_output_type,
    void* transposed_output,                                // o
    const std::vector<size_t>& transposed_output_shape,
    const transformer_engine::DType transposed_output_type,
    void* amax,                                             // o
    const std::vector<size_t>& amax_shape,
    const transformer_engine::DType amax_type,
    void* dbias,                                            // o
    const std::vector<size_t>& dbias_shape,
    const transformer_engine::DType dbias_type,
    void* scale_inv,                                        // o
    const std::vector<size_t>& scale_inv_shape,
    const transformer_engine::DType scale_inv_type
) {
  transformer_engine::TensorWrapper workspace;
  auto gelu_input_cu        = makeTransformerEngineTensor(gelu_input, gelu_input_shape,
                                                          gelu_input_type);
  auto input_cu             = makeTransformerEngineTensor(input, input_shape, input_type);
  auto cast_output_cu       = makeTransformerEngineTensor(cast_output, cast_output_shape,
                                                          cast_output_type, amax, scale,
                                                          scale_inv);
  auto transposed_output_cu = makeTransformerEngineTensor(transposed_output,
                                                          transposed_output_shape,
                                                          transposed_output_type,
                                                          amax, scale, scale_inv);
  auto dbias_cu             = makeTransformerEngineTensor(dbias, dbias_shape, dbias_type);

  nvte_cast_transpose_dbias_dgelu(input_cu.data(), gelu_input_cu.data(),
                                  cast_output_cu.data(), transposed_output_cu.data(),
                                  dbias_cu.data(), workspace.data(),
                                  at::cuda::getCurrentCUDAStream());

  // Fill workspace
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace = makeTransformerEngineTensor(workspace_data.data_ptr(),
                                          workspace.shape(),
                                          workspace.dtype());

  nvte_cast_transpose_dbias_dgelu(input_cu.data(), gelu_input_cu.data(),
                                  cast_output_cu.data(), transposed_output_cu.data(),
                                  dbias_cu.data(), workspace.data(),
                                  at::cuda::getCurrentCUDAStream());
}


void dispatch_multi_cast_transpose(
    std::vector<void*> input_dptr_list,                     // i
    const std::vector<std::vector<size_t>>& input_shape_list,
    const std::vector<transformer_engine::DType>& input_type_list,
    std::vector<void*> scale_dptr_list,                     // i
    const std::vector<std::vector<size_t>>& scale_shape_list,
    const std::vector<transformer_engine::DType>& scale_type_list,
    std::vector<void*> cast_output_dptr_list,               // o
    const std::vector<std::vector<size_t>>& cast_output_shape_list,
    const std::vector<transformer_engine::DType>& cast_output_type_list,
    std::vector<void*> transposed_output_dptr_list,         // o
    const std::vector<std::vector<size_t>>& transposed_output_shape_list,
    const std::vector<transformer_engine::DType>& transposed_output_type_list,
    std::vector<void*> amax_dptr_list,                      // o
    const std::vector<std::vector<size_t>>& amax_shape_list,
    const std::vector<transformer_engine::DType>& amax_type_list,
    std::vector<void*> scale_inv_dptr_list,                 // o
    const std::vector<std::vector<size_t>>& scale_inv_shape_list,
    const std::vector<transformer_engine::DType>& scale_inv_type_list
) {
  transformer_engine::TensorWrapper workspace;

  // Construct TE tensors
  std::vector<NVTETensor> input_list,
    cast_output_list, transposed_output_list;
  std::vector<transformer_engine::TensorWrapper> tensor_wrappers;
  auto make_tensor = [&tensor_wrappers](void* dptr,
                                        const std::vector<size_t>& shape,
                                        transformer_engine::DType dtype,
                                        void* amax_dptr,
                                        void* scale_dptr,
                                        void* scale_inv_dptr)
    -> NVTETensor {
    tensor_wrappers.emplace_back(makeTransformerEngineTensor(dptr, shape, dtype, amax_dptr,
                                                             scale_dptr, scale_inv_dptr));
    return tensor_wrappers.back().data();
  };
  for (size_t i = 0; i < input_dptr_list.size(); ++i) {
    input_list.emplace_back(make_tensor(input_dptr_list[i],
                                        input_shape_list[i],
                                        input_type_list[i],
                                        nullptr,
                                        nullptr,
                                        nullptr));
    cast_output_list.emplace_back(make_tensor(cast_output_dptr_list[i],
                                              cast_output_shape_list[i],
                                              cast_output_type_list[i],
                                              amax_dptr_list[i],
                                              scale_dptr_list[i],
                                              scale_inv_dptr_list[i]));
    transposed_output_list.emplace_back(make_tensor(transposed_output_dptr_list[i],
                                                    transposed_output_shape_list[i],
                                                    transposed_output_type_list[i],
                                                    amax_dptr_list[i],
                                                    scale_dptr_list[i],
                                                    scale_inv_dptr_list[i]));
  }

  // Check tensor lists
  NVTE_CHECK(cast_output_list.size() == input_list.size(),
             "Number of input and C output tensors must match");
  NVTE_CHECK(transposed_output_list.size() == input_list.size(),
             "Number of input and T output tensors must match");

  // Launch TE kernel
  nvte_multi_cast_transpose(input_list.size(),
                            input_list.data(),
                            cast_output_list.data(),
                            transposed_output_list.data(),
                            at::cuda::getCurrentCUDAStream());
}
