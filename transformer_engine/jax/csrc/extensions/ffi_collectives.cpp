/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "ffi_collectives.h"

#ifdef NVTE_FFI_COLLECTIVES_AVAILABLE

#include <string>

#include "ffi.h"

namespace transformer_engine {
namespace jax {

namespace ffi_collectives {

namespace {

::xla::ffi::Error TakeError(const XLA_FFI_Api* api, XLA_FFI_Error* err) {
  std::string msg = ::xla::ffi::internal::GetErrorMessage(api, err);
  ::xla::ffi::internal::DestroyError(api, err);
  return ::xla::ffi::Error::Internal(msg);
}

}  // namespace

std::vector<XLA_FFI_ReplicaGroup> ToRawGroups(const std::vector<std::vector<int64_t>>& groups) {
  std::vector<XLA_FFI_ReplicaGroup> raw;
  raw.reserve(groups.size());
  for (const auto& g : groups) {
    raw.push_back(XLA_FFI_ReplicaGroup{g.data(), g.size()});
  }
  return raw;
}

::xla::ffi::Error RequestClique(const FfiCollectivesCtx& ctx,
                                const std::vector<std::vector<int64_t>>& groups,
                                int64_t communication_id) {
  std::vector<XLA_FFI_ReplicaGroup> raw = ToRawGroups(groups);
  XLA_FFI_Communicator_Request_Args args;
  args.struct_size = XLA_FFI_Communicator_Request_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.group_mode = XLA_FFI_GROUP_FLATTENED_ID;
  args.groups = raw.data();
  args.num_groups = raw.size();
  args.communication_id = communication_id;
  if (XLA_FFI_Error* err = ctx.ext->request_communicator(ctx.ext, &args)) {
    return TakeError(ctx.api, err);
  }
  return ::xla::ffi::Error::Success();
}

::xla::ffi::ErrorOr<ncclComm_t> GetComm(const FfiCollectivesCtx& ctx,
                                        const std::vector<std::vector<int64_t>>& groups,
                                        int64_t communication_id) {
  std::vector<XLA_FFI_ReplicaGroup> raw = ToRawGroups(groups);
  XLA_FFI_Communicator_Get_Args args;
  args.struct_size = XLA_FFI_Communicator_Get_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.group_mode = XLA_FFI_GROUP_FLATTENED_ID;
  args.groups = raw.data();
  args.num_groups = raw.size();
  args.communication_id = communication_id;
  args.communicator = nullptr;
  if (XLA_FFI_Error* err = ctx.ext->get_communicator(ctx.ext, &args)) {
    return TakeError(ctx.api, err);
  }
  return reinterpret_cast<ncclComm_t>(args.communicator);
}

std::vector<std::vector<int64_t>> ReplicaGroupsFromFlat(const int64_t* flat, size_t count,
                                                        int64_t group_size) {
  std::vector<std::vector<int64_t>> groups;
  if (group_size <= 0) return groups;
  for (size_t off = 0; off + group_size <= count; off += group_size) {
    groups.emplace_back(flat + off, flat + off + group_size);
  }
  return groups;
}

}  // namespace ffi_collectives

Error_Type FfiRequestCliqueFFI(FfiCollectivesCtx coll, Span_Type<int64_t> replica_groups,
                               int64_t group_size, int64_t communication_id) {
  auto groups = ffi_collectives::ReplicaGroupsFromFlat(replica_groups.begin(),
                                                       replica_groups.size(), group_size);
  return ffi_collectives::RequestClique(coll, groups, communication_id);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(FfiRequestCliqueHandler, FfiRequestCliqueFFI,
                              FFI::BindPrepare()
                                  .Ctx<::xla::ffi::Extension<FfiCollectives>>()
                                  .Attr<Span_Type<int64_t>>("replica_groups")
                                  .Attr<int64_t>("group_size")
                                  .Attr<int64_t>("communication_id"));

}  // namespace jax
}  // namespace transformer_engine

#endif  // collectives header available
