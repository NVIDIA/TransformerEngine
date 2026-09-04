/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file ffi_collectives.h
 *  \brief Borrow the XLA-owned NCCL communicator inside any FFI handler.
 *
 *  Not EP-specific -- any FFI handler that wants XLA's comm can request the
 *  clique (prepare stage) and fetch the borrowed ncclComm_t (execute stage).
 *  Absent on older XLA, in which case the borrow path must not be selected.
 */

#ifndef TRANSFORMER_ENGINE_JAX_CSRC_EXTENSIONS_FFI_COLLECTIVES_H_
#define TRANSFORMER_ENGINE_JAX_CSRC_EXTENSIONS_FFI_COLLECTIVES_H_

// NVTE_FFI_COLLECTIVES_AVAILABLE is the single source of truth for "is the
// XLA collectives FFI extension available"; callers add their own build gates
// (e.g. NVTE_WITH_NCCL_EP) on top of this.
#if __has_include("xla/ffi/api/collectives_c_api.h")
#define NVTE_FFI_COLLECTIVES_AVAILABLE 1

#include <nccl.h>

#include <cstdint>
#include <vector>

#include "xla/ffi/api/collectives_c_api.h"
#include "xla/ffi/api/ffi.h"

namespace transformer_engine {
namespace jax {

// Decoded context: the FFI api table plus the found collectives extension.
struct FfiCollectivesCtx {
  const XLA_FFI_Api* api = nullptr;
  const XLA_FFI_Collectives_Extension* ext = nullptr;
};

// Trait type for ::xla::ffi::Extension<FfiCollectives>. The public FFI
// CtxDecoding looks the extension up by kExtensionType and hands us a typed
// context, so we do not depend on XLA-internal headers that jaxlib omits.
struct FfiCollectives {
  using Type = FfiCollectivesCtx;
  using CExtension = XLA_FFI_Collectives_Extension;
  static constexpr const char* kName = "CollectivesExtension";
  static constexpr int32_t kExtensionType = XLA_FFI_Extension_Collectives;
  static constexpr int32_t kMajorVersion = XLA_FFI_Extension_Collectives_MajorVersion;
  static constexpr int32_t kMinorVersion = XLA_FFI_Extension_Collectives_MinorVersion;
  // Accept any minor within the same major so a newer runtime still binds.
  static bool Support(int32_t major, int32_t /*minor*/) { return major == kMajorVersion; }
  static Type Create(const XLA_FFI_Api* api, const CExtension* ext) { return Type{api, ext}; }
};

namespace ffi_collectives {

std::vector<XLA_FFI_ReplicaGroup> ToRawGroups(const std::vector<std::vector<int64_t>>& groups);

// Prepare stage: ask XLA to acquire the clique for `groups` (flattened-id mode).
::xla::ffi::Error RequestClique(const FfiCollectivesCtx& ctx,
                                const std::vector<std::vector<int64_t>>& groups,
                                int64_t communication_id);

// Execute stage: fetch the borrowed communicator (ncclComm_t on XLA:GPU).
::xla::ffi::ErrorOr<ncclComm_t> GetComm(const FfiCollectivesCtx& ctx,
                                        const std::vector<std::vector<int64_t>>& groups,
                                        int64_t communication_id);

// Rebuild ragged replica groups from a flat buffer of equal-size groups.
std::vector<std::vector<int64_t>> ReplicaGroupsFromFlat(const int64_t* flat, size_t count,
                                                        int64_t group_size);

}  // namespace ffi_collectives

// Generic prepare-stage handler: any FFI op that borrows XLA's comm binds this
// to request the clique before execute. Reads int64 attrs "replica_groups"
// (flat, equal-size groups), "group_size", and "communication_id".
XLA_FFI_DECLARE_HANDLER_SYMBOL(FfiRequestCliqueHandler);

}  // namespace jax
}  // namespace transformer_engine

#endif  // collectives header available

#endif  // TRANSFORMER_ENGINE_JAX_CSRC_EXTENSIONS_FFI_COLLECTIVES_H_
