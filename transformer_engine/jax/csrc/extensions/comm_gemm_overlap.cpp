/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common/util/dlpack_helper.h"
#include "extensions.h"

void _dummy_allgather(void *global, size_t globalbytes, void *local, size_t localbytes,
                      ExtComm comm) {};

void _dummy_barrier(ExtComm comm) {};

namespace transformer_engine {

namespace jax {

static std::unordered_map<std::string, CommOverlapCore *> _overlaps;

void BootstrapCommGemmOverlap(const std::string &name, const std::string &method,
                              const std::vector<size_t> &buffer_shape, DType buffer_dtype,
                              CommOverlapType comm_type, int tp_size, int num_splits,
                              int num_max_streams, int comm_cga_size, int num_comm_sm,
                              int set_sm_margin, bool use_ce, bool atomic_gemm, bool aggregate,
                              bool pipeline_rs_overlap_first_gemm) {
#ifndef NVTE_UB_WITH_MPI
  NVTE_ERROR(
      std::string("Comm+GEMM overlap in TE/JAX requires bootstrapping Userbuffers with MPI. ") +
      std::string("Please compile TE with `NVTE_UB_WITH_MPI=1`."));
#endif

  // Initialize overlap object -- this allocates the comm buffer
  NVTE_CHECK(_overlaps.find(name) == _overlaps.end(), name, " is already initialized!");
  if (method == "ring-exchange") {
    _overlaps[name] = reinterpret_cast<CommOverlapCore *>(new CommOverlapP2PBase(
        buffer_shape, buffer_dtype, -1, -1, -1, -1, -1, -1, tp_size, &_dummy_allgather,
        &_dummy_barrier, comm_type, num_max_streams, comm_cga_size, num_comm_sm, set_sm_margin,
        use_ce, atomic_gemm, aggregate));
  } else {
    _overlaps[name] = reinterpret_cast<CommOverlapCore *>(new CommOverlapBase(
        buffer_shape, buffer_dtype, -1, -1, -1, -1, -1, -1, tp_size, &_dummy_allgather,
        &_dummy_barrier, num_splits, num_max_streams, comm_cga_size, num_comm_sm, set_sm_margin,
        atomic_gemm, pipeline_rs_overlap_first_gemm));
  }
};

void DestroyCommGemmOverlap(const std::string &name) {
  auto overlap = _overlaps.find(name);
  if (overlap != _overlaps.end()) {
    delete overlap->second;
    _overlaps.erase(overlap);
  }
};

void SetOverlapBufferScaleInverse(const std::string &name, pybind11::object scale_inv, bool grad) {
  auto scale_inv_tensor = DLPackWrapper(scale_inv, grad);
  _overlaps[name]->set_ubuf_scale_inv(reinterpret_cast<float *>(scale_inv_tensor.dptr()));
}

bool OverlapBufferIsFp8(const std::string &name) { return _overlaps[name]->is_fp8_ubuf(); }

pybind11::object GetOverlapBuffer(const std::string &name, CommOverlapType comm_type) {
  DLPackWrapper output = std::move(_overlaps[name]->get_ubuf_output(comm_type));
  auto capsule = output.capsule();
  return capsule;
};

void CopyIntoOverlapBufferImpl(cudaStream_t stream, void *input_ptr,
                               const std::vector<size_t> &shape, DType dtype,
                               const std::string &name, CommOverlapType comm_type) {
  auto input = TensorWrapper(input_ptr, shape, dtype);
  _overlaps[name]->copy_into_ubuf(stream, input, comm_type);
}

void CopyIntoOverlapBuffer(cudaStream_t stream, void **buffers, const char *opaque,
                           size_t opaque_len) {
  auto input_ptr = buffers[0];

  const auto &desc = *UnpackOpaque<CustomCallBufferDescriptor>(opaque, opaque_len);

  CopyIntoOverlapBufferImpl(stream, input_ptr,
                            std::vector<size_t>(desc.shape, desc.shape + desc.ndim), desc.dtype,
                            desc.name, desc.comm_type);
}

Error_Type CopyIntoOverlapBufferFFI(cudaStream_t stream, Buffer_Type input, std::string_view name,
                                    int32_t comm_type_flag) {
  auto input_ptr = input.untyped_data();
  auto shape = std::vector<size_t>(input.dimensions().begin(), input.dimensions().end());
  auto dtype = convert_ffi_datatype_to_te_dtype(input.element_type());

  CopyIntoOverlapBufferImpl(stream, input_ptr, shape, dtype, static_cast<std::string>(name),
                            static_cast<CommOverlapType>(comm_type_flag));

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CopyIntoOverlapBufferHandler, CopyIntoOverlapBufferFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // input
                                  .Attr<std::string_view>("name")
                                  .Attr<int32_t>("comm_type_flag"),
                              FFI_CudaGraph_Traits);

void CommGemmOverlapImpl(void *lhs, const std::vector<size_t> &lhs_shape, DType lhs_dtype,
                         float *lhs_scale_inv, bool lhs_trans, void *rhs,
                         const std::vector<size_t> &rhs_shape, DType rhs_dtype,
                         float *rhs_scale_inv, bool rhs_trans, void *out,
                         const std::vector<size_t> &out_shape, DType out_dtype, float *out_amax,
                         float *out_scale, void *bias, DType bias_dtype, void *pre_gelu_out,
                         void *extra_out, const std::vector<size_t> &extra_out_shape,
                         void *workspace, size_t workspace_size, bool fuse_gelu, bool fuse_bias,
                         bool grad, bool accumulate, bool use_split_accumulator,
                         CommOverlapType comm_type, const std::string &name, cudaStream_t stream) {
  auto lhs_ = TensorWrapper(lhs, lhs_shape, lhs_dtype, nullptr, nullptr, lhs_scale_inv);
  auto rhs_ = TensorWrapper(rhs, rhs_shape, rhs_dtype, nullptr, nullptr, rhs_scale_inv);
  auto out_ = TensorWrapper(out, out_shape, out_dtype, out_amax, out_scale, nullptr);

  auto bias_ptr = (fuse_bias) ? bias : nullptr;
  auto bias_shape = (fuse_bias) ? std::vector<size_t>(out_shape.back()) : std::vector<size_t>{0};
  auto bias_ = TensorWrapper(bias_ptr, bias_shape, bias_dtype);

  auto pre_gelu_ptr = (fuse_gelu) ? pre_gelu_out : nullptr;
  auto pre_gelu_shape = (fuse_gelu) ? out_shape : std::vector<size_t>{0};
  auto pre_gelu_out_ = TensorWrapper(pre_gelu_ptr, pre_gelu_shape, bias_dtype);

  auto workspace_ = TensorWrapper(workspace, std::vector<size_t>{workspace_size}, DType::kByte);

  auto extra_out_ =
      TensorWrapper(extra_out, extra_out_shape, lhs_dtype, nullptr, nullptr, lhs_scale_inv);

  auto overlap = _overlaps[name];
  if (comm_type == CommOverlapType::AG) {
    // AG overlap is only ring-exchange
    if (overlap->is_atomic_gemm()) {
      overlap->atomic_gemm_overlap_ag(rhs_, rhs_trans, lhs_, lhs_trans, out_, bias_, pre_gelu_out_,
                                      workspace_, grad, accumulate, use_split_accumulator,
                                      extra_out_, stream);
    } else {
      overlap->split_overlap_ag(rhs_, rhs_trans, lhs_, lhs_trans, out_, bias_, pre_gelu_out_,
                                workspace_, grad, accumulate, use_split_accumulator, extra_out_,
                                stream);
    }
  } else if (comm_type == CommOverlapType::RS) {
    if (overlap->is_atomic_gemm()) {
      overlap->atomic_gemm_overlap_rs(rhs_, rhs_trans, lhs_, lhs_trans, out_, bias_, pre_gelu_out_,
                                      workspace_, grad, accumulate, use_split_accumulator,
                                      extra_out_, stream);
    } else {
      overlap->split_overlap_rs(rhs_, rhs_trans, lhs_, lhs_trans, out_, bias_, pre_gelu_out_,
                                workspace_, grad, accumulate, use_split_accumulator, extra_out_,
                                stream);
    }
  }
}

void CommGemmOverlap(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
  // Inputs
  auto lhs = buffers[0];
  auto lhs_scale_inv = reinterpret_cast<float *>(buffers[1]);
  auto rhs = buffers[2];
  auto rhs_scale_inv = reinterpret_cast<float *>(buffers[3]);
  auto bias = buffers[4];
  auto gelu_input = buffers[5];
  auto out_amax = reinterpret_cast<float *>(buffers[6]);
  auto out_scale = reinterpret_cast<float *>(buffers[7]);

  // Outputs
  auto out = buffers[8];
  auto out_amax_new = reinterpret_cast<float *>(buffers[9]);
  auto out_scale_new = reinterpret_cast<float *>(buffers[10]);
  auto pre_gelu_out = buffers[11];
  auto bias_grad = buffers[12];
  auto extra_out = buffers[13];
  auto workspace = buffers[14];

  // Check operand-output aliases
  NVTE_CHECK(bias == bias_grad, "bias not bound to bias_grad in AG+GEMM overlap.");
  NVTE_CHECK(gelu_input == pre_gelu_out,
             "gelu_input not bound to pre_gelu_out in AG+GEMM overlap.");
  NVTE_CHECK(out_amax == out_amax_new, "out_amax not bound to out_amax_new in AG+GEMM overlap.");
  NVTE_CHECK(out_scale == out_scale_new,
             "out_scale not bound to out_scale_new in AG+GEMM overlap.");

  const auto &desc = *UnpackOpaque<CustomCallOverlapDescriptor>(opaque, opaque_len);

  auto lhs_shape =
      (desc.lhs_trans) ? std::vector<size_t>{desc.k, desc.m} : std::vector<size_t>{desc.m, desc.k};
  auto rhs_shape =
      (desc.rhs_trans) ? std::vector<size_t>{desc.n, desc.k} : std::vector<size_t>{desc.k, desc.n};
  auto out_shape = std::vector<size_t>{desc.m, desc.n};

  CommGemmOverlapImpl(lhs, lhs_shape, desc.operand_dtype, lhs_scale_inv, desc.lhs_trans, rhs,
                      rhs_shape, desc.operand_dtype, rhs_scale_inv, desc.rhs_trans, out, out_shape,
                      desc.out_dtype, out_amax, out_scale, bias, desc.bias_dtype, pre_gelu_out,
                      extra_out, lhs_shape, workspace, desc.workspace_size, desc.fuse_gelu,
                      desc.fuse_bias, desc.grad, desc.accumulate, desc.use_split_accumulator,
                      desc.comm_type, desc.name, stream);
}

Error_Type CommGemmOverlapFFI(cudaStream_t stream, Buffer_Type lhs, Buffer_Type lhs_scale_inv,
                              Buffer_Type rhs, Buffer_Type rhs_scale_inv, Buffer_Type bias,
                              Buffer_Type gelu_input, Buffer_Type out_amax, Buffer_Type out_scale,
                              Result_Type out, Result_Type out_amax_new, Result_Type out_scale_new,
                              Result_Type pre_gelu_out, Result_Type bias_grad,
                              Result_Type extra_out, Result_Type workspace, bool lhs_trans,
                              bool rhs_trans, bool fuse_gelu, bool fuse_bias, bool grad,
                              bool accumulate, bool use_split_accumulator, int32_t comm_type_flag,
                              std::string_view name) {
  // Inputs
  auto lhs_ptr = lhs.untyped_data();
  auto lhs_shape = std::vector<size_t>(lhs.dimensions().begin(), lhs.dimensions().end());
  auto lhs_dtype = convert_ffi_datatype_to_te_dtype(lhs.element_type());
  auto lhs_scale_inv_ptr = reinterpret_cast<float *>(lhs_scale_inv.untyped_data());
  auto rhs_ptr = rhs.untyped_data();
  auto rhs_shape = std::vector<size_t>(rhs.dimensions().begin(), rhs.dimensions().end());
  auto rhs_dtype = convert_ffi_datatype_to_te_dtype(rhs.element_type());
  auto rhs_scale_inv_ptr = reinterpret_cast<float *>(rhs_scale_inv.untyped_data());
  auto bias_ptr = bias.untyped_data();
  auto bias_dtype = convert_ffi_datatype_to_te_dtype(bias.element_type());
  auto gelu_input_ptr = gelu_input.untyped_data();
  auto out_amax_ptr = reinterpret_cast<float *>(out_amax.untyped_data());
  auto out_scale_ptr = reinterpret_cast<float *>(out_scale.untyped_data());

  // Outputs
  auto out_ptr = out->untyped_data();
  auto out_shape = std::vector<size_t>(out->dimensions().begin(), out->dimensions().end());
  auto out_dtype = convert_ffi_datatype_to_te_dtype(out->element_type());
  auto out_amax_new_ptr = reinterpret_cast<float *>(out_amax_new->untyped_data());
  auto out_scale_new_ptr = reinterpret_cast<float *>(out_scale_new->untyped_data());
  auto pre_gelu_ptr = pre_gelu_out->untyped_data();
  auto bias_grad_ptr = bias_grad->untyped_data();
  auto extra_out_ptr = extra_out->untyped_data();
  auto extra_out_shape =
      std::vector<size_t>(extra_out->dimensions().begin(), extra_out->dimensions().end());
  auto workspace_ptr = workspace->untyped_data();
  auto workspace_size = workspace->element_count();

  // Check operand-output aliases
  NVTE_CHECK(bias_ptr == bias_grad_ptr, "bias not bound to bias_grad in AG+GEMM overlap.");
  NVTE_CHECK(gelu_input_ptr == pre_gelu_ptr,
             "gelu_input not bound to pre_gelu_out in AG+GEMM overlap.");
  NVTE_CHECK(out_amax_ptr == out_amax_new_ptr,
             "out_amax not bound to out_amax_new in AG+GEMM overlap.");
  NVTE_CHECK(out_scale_ptr == out_scale_new_ptr,
             "out_scale not bound to out_scale_new in AG+GEMM overlap.");

  CommGemmOverlapImpl(
      lhs_ptr, lhs_shape, lhs_dtype, lhs_scale_inv_ptr, lhs_trans, rhs_ptr, rhs_shape, rhs_dtype,
      rhs_scale_inv_ptr, rhs_trans, out_ptr, out_shape, out_dtype, out_amax_ptr, out_scale_ptr,
      bias_ptr, bias_dtype, pre_gelu_ptr, extra_out_ptr, extra_out_shape, workspace_ptr,
      workspace_size, fuse_gelu, fuse_bias, grad, accumulate, use_split_accumulator,
      static_cast<CommOverlapType>(comm_type_flag), static_cast<std::string>(name), stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CommGemmOverlapHandler, CommGemmOverlapFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // lhs
                                  .Arg<Buffer_Type>()      // lhs_scale_inv
                                  .Arg<Buffer_Type>()      // rhs
                                  .Arg<Buffer_Type>()      // rhs_scale_inv
                                  .Arg<Buffer_Type>()      // bias
                                  .Arg<Buffer_Type>()      // gelu_input
                                  .Arg<Buffer_Type>()      // out_amax
                                  .Arg<Buffer_Type>()      // out_scale
                                  .Ret<Buffer_Type>()      // out
                                  .Ret<Buffer_Type>()      // out_amax_new
                                  .Ret<Buffer_Type>()      // out_scale_new
                                  .Ret<Buffer_Type>()      // pre_gelu_out
                                  .Ret<Buffer_Type>()      // bias_grad
                                  .Ret<Buffer_Type>()      // extra_out
                                  .Ret<Buffer_Type>()      // workspace
                                  .Attr<bool>("lhs_trans")
                                  .Attr<bool>("rhs_trans")
                                  .Attr<bool>("fuse_gelu")
                                  .Attr<bool>("fuse_bias")
                                  .Attr<bool>("grad")
                                  .Attr<bool>("accumulate")
                                  .Attr<bool>("use_split_accumulator")
                                  .Attr<int32_t>("comm_type_flag")
                                  .Attr<std::string_view>("name"),
                              FFI_CudaGraph_Traits);

}  // namespace jax

}  // namespace transformer_engine
