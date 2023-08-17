import torch
from enum import Enum
from torch import nn, autocast
import torch.backends.cuda
import torch.backends.cudnn
import transformer_engine.pytorch.sequential as seq
from transformer_engine.pytorch.sequential.nvte import DType
import transformer_engine.pytorch as te


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-5):
        super().__init__()  # type: ignore
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm: float = x.norm(2, dim=-1, keepdim=True)  # type: ignore
        rms_x: float = x_norm / sqrt(self.hidden_dim)  # type: ignore
        y: torch.Tensor = x / (rms_x + self.eps)  # type: ignore
        return y * self.weight  # type: ignore


class NormalizationType(Enum):
    NONE = 0
    LAYERNORM = 1
    RMSNORM = 2


class ActivationType(Enum):
    NONE = 0
    RELU = 1
    GELU = 2


class InputInitMethodType(Enum):
    Normal01 = 0
    Uniform01 = 1
    Normal11 = 2
    Uniform11 = 3


def cpy(dst: torch.Tensor, src: torch.Tensor):
    dst.data = torch.as_tensor(src.data.clone().detach(), dtype=dst.dtype).detach()


def normal_range(x: torch.Tensor, kinda_min: float, kinda_max: float):
    mean = (kinda_min + kinda_max) / 2
    range = kinda_max - kinda_min
    kinda_radius = range / 2
    # if the std. dev. of the result is 1/2 radius, then
    # about 95% of values should be within 2 deviations
    # let there be some outliers for diversity
    std = kinda_radius / 2
    return torch.nn.init.normal_(x, mean, std)


def init_input(shape: tuple[int, ...], init_method: InputInitMethodType):
    in_min_val = (
        0.0
        if init_method in [InputInitMethodType.Normal01, InputInitMethodType.Uniform01]
        else -1.0
    )
    in_max_val = 1.0
    distribution = (
        torch.nn.init.uniform_
        if init_method in [InputInitMethodType.Uniform01, InputInitMethodType.Uniform11]
        else normal_range
    )

    input = torch.empty(shape, device="cuda")
    input = distribution(input, in_min_val, in_max_val)
    return input


def pt_test(
    normalization: NormalizationType,
    first_linear: bool,
    activation: ActivationType,
    second_linear: bool,
    lin1_weight: torch.Tensor,
    lin1_bias: torch.Tensor,
    lin2_weight: torch.Tensor,
    lin2_bias: torch.Tensor,
    x: torch.Tensor,
):
    modules = list[nn.Module]()

    if normalization is NormalizationType.LAYERNORM:
        modules.append(nn.LayerNorm(IN_FEATURES))
    elif normalization is NormalizationType.RMSNORM:
        modules.append(RMSNorm(IN_FEATURES))

    if first_linear:
        lin1 = nn.Linear(IN_FEATURES, OUT_FEATURES)
        cpy(lin1.weight, lin1_weight)
        cpy(lin1.bias, lin1_bias)
        modules.append(lin1)

    if activation is ActivationType.RELU:
        modules.append(nn.ReLU())
    elif activation is ActivationType.GELU:
        modules.append(nn.GELU())

    if second_linear:
        if not first_linear:
            lin2 = nn.Linear(IN_FEATURES, OUT_FEATURES)
            cpy(lin2.weight, lin1_weight)
            cpy(lin2.bias, lin1_bias)
            modules.append(lin2)
        else:
            lin2 = nn.Linear(OUT_FEATURES, IN_FEATURES)
            cpy(lin2.weight, lin2_weight)
            cpy(lin2.bias, lin2_bias)
            modules.append(lin2)

    assert len(modules) >= 1

    m = nn.Sequential(*modules)
    inp = x.detach().clone().requires_grad_()
    out = m(inp)
    out.sum().backward()
    assert inp.grad is not None
    return inp.grad


def seq_test_unfused(
    normalization: NormalizationType,
    first_linear: bool,
    activation: ActivationType,
    second_linear: bool,
    lin1_weight: torch.Tensor,
    lin1_bias: torch.Tensor,
    lin2_weight: torch.Tensor,
    lin2_bias: torch.Tensor,
    x: torch.Tensor,
):
    modules = list[nn.Module]()

    if normalization is NormalizationType.LAYERNORM:
        modules.append(seq.LayerNorm(IN_FEATURES))
    elif normalization is NormalizationType.RMSNORM:
        modules.append(seq.RMSNorm(IN_FEATURES))

    if first_linear:
        lin1 = seq.Linear(IN_FEATURES, OUT_FEATURES)
        cpy(lin1.weight, lin1_weight)
        cpy(lin1.bias, lin1_bias)
        modules.append(lin1)

    if activation is ActivationType.RELU:
        modules.append(seq.ReLU())
    elif activation is ActivationType.GELU:
        modules.append(seq.GELU())

    if second_linear:
        if not first_linear:
            lin2 = seq.Linear(IN_FEATURES, OUT_FEATURES)
            cpy(lin2.weight, lin1_weight)
            cpy(lin2.bias, lin1_bias)
            modules.append(lin2)
        else:
            lin2 = seq.Linear(OUT_FEATURES, IN_FEATURES)
            cpy(lin2.weight, lin2_weight)
            cpy(lin2.bias, lin2_bias)
            modules.append(lin2)

    assert len(modules) >= 1

    m = nn.Sequential(*modules)
    inp = x.detach().clone().requires_grad_()
    out = m(inp)
    out.sum().backward()
    assert inp.grad is not None
    return inp.grad


def seq_test_fused(
    normalization: NormalizationType,
    first_linear: bool,
    activation: ActivationType,
    second_linear: bool,
    lin1_weight: torch.Tensor,
    lin1_bias: torch.Tensor,
    lin2_weight: torch.Tensor,
    lin2_bias: torch.Tensor,
    x: torch.Tensor,
):
    modules = list[nn.Module]()

    if normalization is NormalizationType.LAYERNORM:
        modules.append(seq.LayerNorm(IN_FEATURES))
    elif normalization is NormalizationType.RMSNORM:
        modules.append(seq.RMSNorm(IN_FEATURES))

    if first_linear:
        lin1 = seq.Linear(IN_FEATURES, OUT_FEATURES)
        cpy(lin1.weight, lin1_weight)
        cpy(lin1.bias, lin1_bias)
        modules.append(lin1)

    if activation is ActivationType.RELU:
        modules.append(seq.ReLU())
    elif activation is ActivationType.GELU:
        modules.append(seq.GELU())

    if second_linear:
        if not first_linear:
            lin2 = seq.Linear(IN_FEATURES, OUT_FEATURES)
            cpy(lin2.weight, lin1_weight)
            cpy(lin2.bias, lin1_bias)
            modules.append(lin2)
        else:
            lin2 = seq.Linear(OUT_FEATURES, IN_FEATURES)
            cpy(lin2.weight, lin2_weight)
            cpy(lin2.bias, lin2_bias)
            modules.append(lin2)

    assert len(modules) >= 1

    m = seq.Sequential(*modules)
    inp = x.detach().clone().requires_grad_()
    out = m(inp)
    out.sum().backward()
    assert inp.grad is not None
    return inp.grad


def test(
    normalization: NormalizationType,
    first_linear: bool,
    activation: ActivationType,
    second_linear: bool,
    lin1_weight: torch.Tensor,
    lin1_bias: torch.Tensor,
    lin2_weight: torch.Tensor,
    lin2_bias: torch.Tensor,
    x: torch.Tensor,
):
    # Pytorch reference implementation in FP32, no TF32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pt_fp32 = pt_test(
        normalization,
        first_linear,
        activation,
        second_linear,
        lin1_weight,
        lin1_bias,
        lin2_weight,
        lin2_bias,
        x,
    )
    # Pytorch reference implementation in FP32, with TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    pt_fp32 = pt_test(
        normalization,
        first_linear,
        activation,
        second_linear,
        lin1_weight,
        lin1_bias,
        lin2_weight,
        lin2_bias,
        x,
    )
    # Pytorch reference implementation with autocast to float16
    with autocast("cuda", torch.float16):
        pt_fp16 = pt_test(
            normalization,
            first_linear,
            activation,
            second_linear,
            lin1_weight,
            lin1_bias,
            lin2_weight,
            lin2_bias,
            x,
        )
    # Pytorch reference implementation with autocast to bfloat16
    with autocast("cuda", torch.bfloat16):
        pt_bf16 = pt_test(
            normalization,
            first_linear,
            activation,
            second_linear,
            lin1_weight,
            lin1_bias,
            lin2_weight,
            lin2_bias,
            x,
        )

    with seq.environment(DType.Float32):
        sequ_fp32 = seq_test_unfused(
            normalization,
            first_linear,
            activation,
            second_linear,
            lin1_weight,
            lin1_bias,
            lin2_weight,
            lin2_bias,
            x,
        )
    with seq.environment(DType.BFloat16):
        sequ_bf16 = seq_test_unfused(
            normalization,
            first_linear,
            activation,
            second_linear,
            lin1_weight,
            lin1_bias,
            lin2_weight,
            lin2_bias,
            x,
        )
    with seq.environment(DType.Float16):
        sequ_fp16 = seq_test_unfused(
            normalization,
            first_linear,
            activation,
            second_linear,
            lin1_weight,
            lin1_bias,
            lin2_weight,
            lin2_bias,
            x,
        )

    with seq.environment(DType.Float32):
        seqf_fp32 = seq_test_fused(
            normalization,
            first_linear,
            activation,
            second_linear,
            lin1_weight,
            lin1_bias,
            lin2_weight,
            lin2_bias,
            x,
        )
    with seq.environment(DType.BFloat16):
        seqf_bf16 = seq_test_fused(
            normalization,
            first_linear,
            activation,
            second_linear,
            lin1_weight,
            lin1_bias,
            lin2_weight,
            lin2_bias,
            x,
        )
    with seq.environment(DType.Float16):
        seqf_fp16 = seq_test_fused(
            normalization,
            first_linear,
            activation,
            second_linear,
            lin1_weight,
            lin1_bias,
            lin2_weight,
            lin2_bias,
            x,
        )

    for cand in [sequ_fp32, sequ_bf16, sequ_fp16, seqf_fp32, seqf_bf16, seqf_fp16]:
        for ref in [pt_fp32, pt_fp32, pt_fp16, pt_bf16]:
            try:
                torch.testing.assert_close(cand, ref, atol=1e-5, rtol=1e-3)
                ok = True
            except AssertionError:
                ok = False
            print_result(ok)
        print()


def print_result(ok: bool):
    if ok:
        print(f"a:\033[42;97mOK\033[0m", end="")
    else:
        print(f"a:\033[41;30mWA\033[0m", end="")


BATCH_SIZE = 512
IN_FEATURES = 768
OUT_FEATURES = 4 * IN_FEATURES
TESTS = 10

for input_init_method in InputInitMethodType:
    for _ in range(TESTS):
        lin1 = nn.Linear(
            IN_FEATURES, OUT_FEATURES, device="cuda"
        )  # used for initializing weights consistently
        lin2 = nn.Linear(
            OUT_FEATURES, IN_FEATURES, device="cuda"
        )  # used for initializing weights consistently
        x = init_input((BATCH_SIZE, IN_FEATURES), input_init_method)

        for normalization in NormalizationType:
            for first_linear in [True, False]:
                for activation in ActivationType:
                    for second_linear in [True, False]:
                        # Skip invalid configurations
                        if (
                            normalization is NormalizationType.NONE
                            and not first_linear
                            and activation is ActivationType.NONE
                            and not second_linear
                        ):
                            continue  # noop model
                        if (
                            not first_linear
                            and activation is ActivationType.NONE
                            and second_linear
                        ):
                            continue  # one linear layer, symmetrical to: first_linear and activation is ActivationType.NONE and not second_linear

                        test(
                            normalization,
                            first_linear,
                            activation,
                            second_linear,
                            lin1.weight,
                            lin1.bias,
                            lin2.weight,
                            lin2.bias,
                            x,
                        )

        del lin1, lin2, x  # force recreation of tensors
