/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
class GetStreamOp : public OpKernel {
 public:
  explicit GetStreamOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* ctx) override {
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("stream_id", {1}, &output));
    auto vec = output->vec<int64_t>();
    se::Stream* stream = ctx->op_device_context()->stream();
    auto gpu_stream = se::gpu::AsGpuStreamValue(stream);
    vec(0) = static_cast<int64_t>(reinterpret_cast<uintptr_t>(gpu_stream));
  }
};

REGISTER_OP("GetStream")
    .Output("stream_id: int64")
    .SetShapeFn(shape_inference::UnknownShape);
REGISTER_OP_NO_GRADIENT("GetStream");
REGISTER_KERNEL_BUILDER(
    Name("GetStream").Device(DEVICE_GPU).HostMemory("stream_id"), GetStreamOp);
}  // namespace tensorflow
