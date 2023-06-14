/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <vector>
#include "common.h"
namespace transformer_engine {
namespace paddle_ext {

std::vector<paddle::Tensor> cast_to_fp8(const paddle::Tensor &input, const paddle::Tensor &scale,
                                        paddle::Tensor &amax, paddle::Tensor &scale_inv,  // NOLINT
                                        int64_t index, int64_t otype) {
    auto shape = GetShapeArray(input);

    auto output = paddle::empty_like(input, Nvte2PaddleDType(Int2NvteDType(otype)));

    auto input_cu = MakeNvteTensor(input);
    auto output_cu = MakeNvteTensor(
        output.data(), shape, Int2NvteDType(otype), GetDataPtr<float>(amax, index),
        const_cast<void *>(GetDataPtr<float>(scale, index)), GetDataPtr<float>(scale_inv, index));

    nvte_fp8_quantize(input_cu.data(), output_cu.data(), input.stream());

    return {output};
}

std::vector<paddle::Tensor> cast_from_fp8(const paddle::Tensor &input,
                                          const paddle::Tensor &scale_inv, int64_t index,
                                          int64_t itype, int64_t otype) {
    auto shape = GetShapeArray(input);

    auto output = paddle::empty_like(input, Nvte2PaddleDType(Int2NvteDType(otype)));
    auto input_cu =
        MakeNvteTensor(const_cast<void *>(input.data()), shape, Int2NvteDType(itype), nullptr,
                       nullptr, const_cast<void *>(GetDataPtr<float>(scale_inv, index)));
    auto output_cu = MakeNvteTensor(output);

    nvte_fp8_dequantize(input_cu.data(), output_cu.data(), input.stream());

    return {output};
}

void te_gemm(const paddle::Tensor &A, const paddle::optional<paddle::Tensor> &A_scale_inverse,
             const paddle::Tensor &B, const paddle::optional<paddle::Tensor> &B_scale_inverse,
             const paddle::optional<paddle::Tensor> &bias, paddle::Tensor &D,            // NOLINT
             paddle::optional<paddle::Tensor> &D_scale,                                  // NOLINT
             paddle::optional<paddle::Tensor> &D_amax,                                   // NOLINT
             paddle::optional<paddle::Tensor> &pre_gelu_out, paddle::Tensor &workspace,  // NOLINT
             int64_t A_index, int64_t B_index, int64_t D_index, int64_t A_type, int64_t B_type,
             int64_t D_type, int64_t bias_type, bool transa, bool transb, bool grad,
             int64_t workspace_size, bool accumulate, bool use_split_accumulator,
             int64_t math_sm_count) {
    auto te_A = MakeNvteTensor(
        const_cast<void *>(A.data()), GetShapeArray(A), Int2NvteDType(A_type), nullptr, nullptr,
        const_cast<void *>(GetOptionalDataPtr<float>(A_scale_inverse, A_index)));
    auto te_B = MakeNvteTensor(
        const_cast<void *>(B.data()), GetShapeArray(B), Int2NvteDType(B_type), nullptr, nullptr,
        const_cast<void *>(GetOptionalDataPtr<float>(B_scale_inverse, B_index)));
    auto te_D = MakeNvteTensor(D.data(), GetShapeArray(D), Int2NvteDType(D_type),
                               GetOptionalDataPtr<float>(D_amax, D_index),
                               GetOptionalDataPtr<float>(D_scale, D_index), nullptr);

    auto te_bias = MakeNvteTensor(const_cast<void *>(GetOptionalDataPtr(bias)), GetShapeArray(bias),
                                  Int2NvteDType(bias_type));

    DType gelu_dtype =
        pre_gelu_out ? Paddle2NvteDType(pre_gelu_out->dtype()) : Int2NvteDType(D_type);
    auto te_pre_gelu_out =
        MakeNvteTensor(GetOptionalDataPtr(pre_gelu_out), GetShapeArray(pre_gelu_out), gelu_dtype);
    auto te_workspace =
        MakeNvteTensor(workspace.data(), {static_cast<size_t>(workspace_size)}, DType::kByte);

    nvte_cublas_gemm(te_A.data(), te_B.data(), te_D.data(), te_bias.data(), te_pre_gelu_out.data(),
                     transa, transb, grad, te_workspace.data(), accumulate, use_split_accumulator,
                     math_sm_count, A.stream());
}

}  // namespace paddle_ext
}  // namespace transformer_engine

PD_BUILD_OP(te_gemm)
    .Inputs({"A", paddle::Optional("A_scale_inverse"), "B", paddle::Optional("B_scale_inverse"),
             paddle::Optional("bias"), "_D", paddle::Optional("_D_scale"),
             paddle::Optional("_D_amax"), paddle::Optional("_pre_gelu_out"), "_workspace"})
    .Outputs({"D", paddle::Optional("D_scale"), paddle::Optional("D_amax"),
              paddle::Optional("pre_gelu_out"), "workspace"})
    .Attrs({"A_index: int64_t", "B_index: int64_t", "D_index: int64_t", "A_type: int64_t",
            "B_type: int64_t", "D_type: int64_t", "bias_type: int64_t", "transa: bool",
            "transb: bool", "grad: bool", "workspace_size: int64_t", "accumulate: bool",
            "use_split_accumulator: bool", "math_sm_count: int64_t"})
    .SetInplaceMap({{"_D", "D"},
                    {paddle::Optional("_D_scale"), paddle::Optional("D_scale")},
                    {paddle::Optional("_D_amax"), paddle::Optional("D_amax")},
                    {paddle::Optional("_pre_gelu_out"), paddle::Optional("pre_gelu_out")},
                    {"_workspace", "workspace"}})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::te_gemm));

PD_BUILD_OP(cast_to_fp8)
    .Inputs({"Input", "Scale", "_Amax", "_ScaleInv"})
    .Outputs({"Output", "Amax", "ScaleInv"})
    .Attrs({"index: int64_t", "otype: int64_t"})
    .SetInplaceMap({{"_Amax", "Amax"}, {"_ScaleInv", "ScaleInv"}})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::cast_to_fp8));

PD_BUILD_OP(cast_from_fp8)
    .Inputs({"Input", "ScaleInv"})
    .Outputs({"Output"})
    .Attrs({"index: int64_t", "itype: int64_t", "otype: int64_t"})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::cast_from_fp8));
