/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/layer_norm.h>
#include <vector>
#include "ln.h"
#include "../common.h"

/*

Supported Type combinations:

input    compute   weights   output
=======================================
fp32     fp32      fp32      fp32
fp16     fp32      fp16      fp16
bf16     fp32      bf16      bf16
fp32     fp32      fp16      fp16
fp32     fp32      bf16      bf16
bf16     fp32      bf16      fp8

Remarks:
Output type = Weight type
Compute always in FP32

*/

namespace transformer_engine {
namespace layer_norm {

using namespace transformer_engine;

// Create registries and provide runtime versions of config hash functions.

FwdTunedRegistry FWD_TUNED_FUNCS;
BwdTunedRegistry BWD_TUNED_FUNCS;
FwdGeneralRegistry FWD_GENERAL_FUNCS;
BwdGeneralRegistry BWD_GENERAL_FUNCS;

////////////////////////////////////////////////////////////////////////////////////////////////////

uint32_t get_type_id(DType dtype) {
    if ( dtype == DType::kFloat16 ) {
        return TypeId<fp16>::Value;
    } else if ( dtype == DType::kBFloat16 ) {
        return TypeId<bf16>::Value;
    } else if ( dtype == DType::kFloat32 ) {
        return TypeId<fp32>::Value;
    } else if ( dtype == DType::kFloat8E4M3 ) {
        return TypeId<fp8e4m3>::Value;
    } else {
        NVTE_ERROR("Type not supported.");
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

uint64_t get_key(DType wtype, DType itype, DType otype, DType ctype, uint64_t hidden_size) {
    using namespace layer_norm;
    uint64_t type_key = get_type_id(wtype) | (get_type_id(itype) << 2) |
                        (get_type_id(otype) << 4) | (get_type_id(ctype) << 6);
    uint64_t launcher_key = (type_key << 32) | hidden_size;
    return launcher_key;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

layer_norm::FwdFunction & get_fwd_launcher(DType wtype,
                                           DType itype,
                                           DType otype,
                                           DType ctype,
                                           uint32_t hidden_size,
                                           uint32_t batch_size) {
    // Look for tuned kernel
    auto tuned_key = layer_norm::get_key(wtype, itype, otype, ctype, hidden_size);
    if (batch_size % 4 == 0
        && layer_norm::FWD_TUNED_FUNCS.count(tuned_key) > 0) {
        return layer_norm::FWD_TUNED_FUNCS.at(tuned_key);
    }

    // Pick general kernel
    auto general_key = layer_norm::get_key(wtype, itype, otype, ctype, 0);
    if (layer_norm::FWD_GENERAL_FUNCS.count(general_key) == 0) {
        NVTE_ERROR("FWD: Unsupported types.");
    }
    auto& general_func_map = layer_norm::FWD_GENERAL_FUNCS.at(general_key);
    auto func_iter = general_func_map.lower_bound(hidden_size);
    if (func_iter == general_func_map.end()) {
        // Hidden size is too big, need to use multi-CTA
        return general_func_map.rbegin()->second;
    } else {
        return func_iter->second;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

layer_norm::BwdFunction & get_bwd_launcher(DType wtype,
                                           DType itype,
                                           DType otype,
                                           DType ctype,
                                           uint32_t hidden_size,
                                           uint32_t batch_size) {
    // Look for tuned kernel
    auto tuned_key = layer_norm::get_key(wtype, itype, otype, ctype, hidden_size);
    if (batch_size % 4 == 0
        && layer_norm::BWD_TUNED_FUNCS.count(tuned_key) > 0) {
        return layer_norm::BWD_TUNED_FUNCS.at(tuned_key);
    }

    // Pick general kernel
    auto general_key = layer_norm::get_key(wtype, itype, otype, ctype, 0);
    if (layer_norm::BWD_GENERAL_FUNCS.count(general_key) == 0) {
        NVTE_ERROR("BWD: Unsupported types.");
    }
    auto& general_func_map = layer_norm::BWD_GENERAL_FUNCS.at(general_key);
    auto func_iter = general_func_map.lower_bound(hidden_size);
    if (func_iter == general_func_map.end()) {
        // Hidden size is too big, need to use multi-CTA
        return general_func_map.rbegin()->second;
    } else {
        return func_iter->second;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////


size_t product(const std::vector<size_t> &shape) {
    size_t ret = 1;
    for (auto s : shape) {
        ret *= s;
    }
    return ret;
}

}  // namespace layer_norm

////////////////////////////////////////////////////////////////////////////////////////////////////

void layernorm_fwd(const Tensor& x,        // BxSxhidden_size
                   const Tensor& gamma,    // hidden_size
                   const Tensor& beta,     // hidden_size
                   const Tensor& scale,
                   const float epsilon,
                   Tensor* z,
                   Tensor* mu,
                   Tensor* rsigma,
                   cudaStream_t stream,
                   const int multiprocessorCount,
                   Tensor* workspace,
                   Tensor* barrier,
                   Tensor* amax,
                   Tensor *scale_inv,
                   bool fp8_out
) {
    auto itype = x.dtype;
    auto wtype = gamma.dtype;
    auto otype = z->dtype;
    auto ctype = layer_norm::DType::kFloat32;

    NVTE_CHECK(x.shape.size() == 2);

    const size_t rows = x.shape[0];
    const size_t cols = x.shape[1];
    auto hidden_size = gamma.shape[0];

    NVTE_CHECK(gamma.shape == beta.shape);
    NVTE_CHECK(hidden_size == cols);

    NVTE_CHECK(epsilon >= 0.f);

    NVTE_CHECK(z->dptr != nullptr);
    NVTE_CHECK(z->shape == x.shape);

    NVTE_CHECK(mu->shape == std::vector<size_t>{ rows });
    NVTE_CHECK(mu->dtype == ctype);

    NVTE_CHECK(rsigma->shape == std::vector<size_t>{ rows });
    NVTE_CHECK(rsigma->dtype == ctype);

    if (fp8_out) {
        NVTE_CHECK(scale.shape == std::vector<size_t>{ 1 });
        NVTE_CHECK(scale.dptr != nullptr);
        NVTE_CHECK(scale.dtype == ctype);

        NVTE_CHECK(amax->shape == std::vector<size_t>{ 1 });
        NVTE_CHECK(amax->dptr != nullptr);
        NVTE_CHECK(amax->dtype == ctype);

        NVTE_CHECK(scale_inv->shape == std::vector<size_t>{ 1 });
        NVTE_CHECK(scale_inv->dptr != nullptr);
        NVTE_CHECK(scale_inv->dtype == ctype);
    }

    layer_norm::LaunchParams<layer_norm::FwdParams> launch_params;

    launch_params.multiprocessorCount = multiprocessorCount;
    launch_params.stream = stream;

    // Request the kernel launcher.
    auto launcher = layer_norm::get_fwd_launcher(wtype, itype, otype, ctype,
                                                 hidden_size, rows);

    // Set the kernel runtime parameters.
    layer_norm::FwdParams &params = launch_params.params;
    params.rows = rows;
    params.cols = cols;
    params.x = x.dptr;
    params.mu = mu->dptr;
    params.rs = rsigma->dptr;
    params.gamma = gamma.dptr;
    params.beta = beta.dptr;
    params.z = z->dptr;
    params.epsilon = epsilon;
    params.amax = amax->dptr;
    params.scale = scale.dptr;
    params.scale_inv = scale_inv->dptr;
    params.fp8_out = fp8_out;

    // Query the kernel-specific launch parameters.
    launcher(launch_params, true);
    if (workspace->dptr == nullptr) {
        NVTE_CHECK(barrier->dptr == nullptr);

        workspace->dtype = layer_norm::DType::kByte;
        if (launch_params.workspace_bytes == 0) {
            launch_params.workspace_bytes = 1;
        }
        workspace->shape = { launch_params.workspace_bytes };

        barrier->dtype = layer_norm::DType::kInt32;
        barrier->shape = { launch_params.barrier_size };

        return;
    }
    if ( launch_params.barrier_size > 0 ) {
        params.workspace = workspace->dptr;
        params.barrier = reinterpret_cast<int*>(barrier->dptr);
    }

    // Clear buffers
    if ( params.fp8_out ) {
        cudaMemsetAsync(params.amax, 0,
                        layer_norm::product(amax->shape) *
                        typeToSize(amax->dtype), stream);
    }
    if ( launch_params.barrier_size > 0 ) {
        cudaMemsetAsync(params.barrier, 0,
                        layer_norm::product(barrier->shape) *
                        typeToSize(barrier->dtype), stream);
    }

    // Launch the kernel.
    launcher(launch_params, false);

    return;
}

void layernorm_bwd(const Tensor& dz,
                   const Tensor& x,
                   const Tensor& mu,
                   const Tensor& rsigma,
                   const Tensor& gamma,
                   Tensor* dx,
                   Tensor* dgamma,
                   Tensor* dbeta,
                   Tensor* dgamma_part,
                   Tensor* dbeta_part,
                   cudaStream_t stream,
                   const int multiprocessorCount,
                   Tensor* workspace,
                   Tensor* barrier
) {
    using namespace transformer_engine;

    auto itype = x.dtype;
    auto wtype = gamma.dtype;
    auto otype = wtype;
    auto ctype = DType::kFloat32;

    NVTE_CHECK(dz.dtype == otype);
    NVTE_CHECK(mu.dtype == ctype);
    NVTE_CHECK(rsigma.dtype == ctype);

    NVTE_CHECK(x.shape.size() == 2);
    NVTE_CHECK(dz.shape == x.shape);
    auto rows = x.shape[0];
    auto cols = x.shape[1];

    auto hidden_size = gamma.shape[0];

    NVTE_CHECK(mu.shape[0] == rows);
    NVTE_CHECK(mu.shape == rsigma.shape);

    NVTE_CHECK(gamma.shape[0] == cols);

    NVTE_CHECK(dx->shape == x.shape);
    NVTE_CHECK(dx->dtype == x.dtype);
    NVTE_CHECK(dx->dptr != nullptr);

    NVTE_CHECK(dgamma->shape == gamma.shape);
    NVTE_CHECK(dgamma->dtype == gamma.dtype);
    NVTE_CHECK(dgamma->dptr != nullptr);

    NVTE_CHECK(dbeta->shape == gamma.shape);
    NVTE_CHECK(dbeta->dtype == gamma.dtype);
    NVTE_CHECK(dbeta->dptr != nullptr);

    layer_norm::LaunchParams<layer_norm::BwdParams> launch_params;
    launch_params.stream = stream;
    launch_params.multiprocessorCount = multiprocessorCount;

    auto launcher = layer_norm::get_bwd_launcher(wtype, itype, otype, ctype,
                                                 hidden_size, rows);

    // Set the kernel runtime parameters.
    layer_norm::BwdParams &params = launch_params.params;
    params.rows = rows;
    params.cols = cols;
    params.x = x.dptr;
    params.mu = mu.dptr;
    params.rs = rsigma.dptr;
    params.gamma = gamma.dptr;
    params.dz = dz.dptr;
    params.dx = dx->dptr;
    params.dbeta = dbeta->dptr;
    params.dgamma = dgamma->dptr;
    params.dbeta_part = dbeta_part->dptr;
    params.dgamma_part = dgamma_part->dptr;

    // Query the kernel-specific launch parameters.
    launcher(launch_params, true);

    // Populate shape and dtypes for FW to allocate memory
    if (dgamma_part->dptr == nullptr) {
        NVTE_CHECK(dbeta_part->dptr == nullptr);

        dgamma_part->dtype = ctype;
        dgamma_part->shape = { static_cast<uint64_t> (launch_params.params.ctas_per_col),
                               hidden_size };

        dbeta_part->dtype = ctype;
        dbeta_part->shape = { static_cast<uint64_t> (launch_params.params.ctas_per_col),
                              hidden_size };

        workspace->dtype = layer_norm::DType::kByte;
        workspace->shape = { launch_params.workspace_bytes };

        barrier->dtype = layer_norm::DType::kInt32;
        barrier->shape = { launch_params.barrier_size };

        return;
    }

    if ( launch_params.barrier_size > 0 ) {
        params.workspace = workspace->dptr;
        params.barrier = reinterpret_cast<int*>(barrier->dptr);
        cudaMemsetAsync(params.barrier, 0,
                        layer_norm::product(barrier->shape) *
                        typeToSize(barrier->dtype), stream);
    }

    // Launch the kernel.
    launcher(launch_params, false);
}
}  // namespace transformer_engine

void nvte_layernorm_fwd(const NVTETensor x,       // BxSxhidden_size
                        const NVTETensor gamma,   // hidden_size
                        const NVTETensor beta,    // hidden_size
                        const NVTETensor scale,   // 1
                        const float epsilon,
                        NVTETensor z,
                        NVTETensor mu,
                        NVTETensor rsigma,
                        cudaStream_t stream,
                        const int multiprocessorCount,
                        NVTETensor workspace,
                        NVTETensor barrier,
                        NVTETensor amax,
                        NVTETensor scale_inv,
                        bool fp8_out) {
  using namespace transformer_engine;
  layernorm_fwd(*reinterpret_cast<const Tensor*>(x),
                *reinterpret_cast<const Tensor*>(gamma),
                *reinterpret_cast<const Tensor*>(beta),
                *reinterpret_cast<const Tensor*>(scale),
                epsilon,
                reinterpret_cast<Tensor*>(z),
                reinterpret_cast<Tensor*>(mu),
                reinterpret_cast<Tensor*>(rsigma),
                stream,
                multiprocessorCount,
                reinterpret_cast<Tensor*>(workspace),
                reinterpret_cast<Tensor*>(barrier),
                reinterpret_cast<Tensor*>(amax),
                reinterpret_cast<Tensor*>(scale_inv),
                fp8_out);
}

void nvte_layernorm_bwd(const NVTETensor dz,       // BxSxhidden_size
                        const NVTETensor x,        // BxSxhidden_size
                        const NVTETensor mu,       // BxS, FP32!
                        const NVTETensor rsigma,   // BxS, FP32!
                        const NVTETensor gamma,    // hidden_size
                        NVTETensor dx,
                        NVTETensor dgamma,
                        NVTETensor dbeta,
                        NVTETensor dgamma_part,
                        NVTETensor dbeta_part,
                        cudaStream_t stream,
                        const int multiprocessorCount,
                        NVTETensor workspace,
                        NVTETensor barrier) {
  using namespace transformer_engine;
  layernorm_bwd(*reinterpret_cast<const Tensor*>(dz),
                *reinterpret_cast<const Tensor*>(x),
                *reinterpret_cast<const Tensor*>(mu),
                *reinterpret_cast<const Tensor*>(rsigma),
                *reinterpret_cast<const Tensor*>(gamma),
                reinterpret_cast<Tensor*>(dx),
                reinterpret_cast<Tensor*>(dgamma),
                reinterpret_cast<Tensor*>(dbeta),
                reinterpret_cast<Tensor*>(dgamma_part),
                reinterpret_cast<Tensor*>(dbeta_part),
                stream,
                multiprocessorCount,
                reinterpret_cast<Tensor*>(workspace),
                reinterpret_cast<Tensor*>(barrier));
}
