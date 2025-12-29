# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from typing import Any, Optional, Tuple
import torch
import torch.nn.functional as F

__all__ = [
    "gelu_torch",
    "geglu_torch",
    "qgelu_torch",
    "qgeglu_torch",
    "relu_torch",
    "reglu_torch",
    "srelu_torch",
    "sreglu_torch",
    "silu_torch",
    "swiglu_torch",
    "clamped_swiglu_torch",
    "dgelu_torch",
    "dgeglu_torch",
    "dqgelu_torch",
    "dqgeglu_torch",
    "drelu_torch",
    "dreglu_torch",
    "dsrelu_torch",
    "dsreglu_torch",
    "dsilu_torch",
    "dswiglu_torch",
    "clamped_dswiglu_torch",
    "dbias_dgelu_torch",
    "dbias_dsilu_torch",
    "dbias_drelu_torch",
    "dbias_dqgelu_torch",
    "dbias_dsrelu_torch",
]


def gelu_torch(input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    return F.gelu(input, approximate='tanh')


def geglu_torch(input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    a, b = input.chunk(2, dim=-1)
    return F.gelu(a, approximate='tanh') * b


def qgelu_torch(input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    return input * torch.sigmoid(1.702 * input)


def qgeglu_torch(input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    a, b = input.chunk(2, dim=-1)
    return a * torch.sigmoid(1.702 * a) * b


def relu_torch(input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    return F.relu(input)


def reglu_torch(input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    a, b = input.chunk(2, dim=-1)
    return F.relu(a) * b


def srelu_torch(input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    return torch.square(F.relu(input))


def sreglu_torch(input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    a, b = input.chunk(2, dim=-1)
    return torch.square(F.relu(a)) * b


def silu_torch(input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    return F.silu(input)


def swiglu_torch(input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    a, b = input.chunk(2, dim=-1)
    return F.silu(a) * b


def clamped_swiglu_torch(
    input: torch.Tensor,
    quantizer: Any,
    limit: float = 7.0,
    alpha: float = 1.702,
) -> torch.Tensor:
    """Clamped SwiGLU matching CUDA implementation.

    CUDA implementation:
    - a (activation): clamp to upper bound only: min(a, limit)
    - b (gate): clamp to [-limit, limit], then add 1
    - output = (a_clamped * sigmoid(alpha * a_clamped)) * b_clamped
    """
    a, b = input.chunk(2, dim=-1)
    # CUDA only clamps a to upper bound
    a_clamped = torch.clamp(a, max=limit)
    # CUDA clamps b to [-limit, limit] and adds 1
    b_clamped = torch.clamp(b, -limit, limit) + 1
    return a_clamped * torch.sigmoid(alpha * a_clamped) * b_clamped


def dgelu_torch(grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    x = fwd_input.detach().requires_grad_(True)
    with torch.enable_grad():
        y = F.gelu(x, approximate='tanh')
        y.backward(grad)
    return x.grad


def dgeglu_torch(grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    a, b = fwd_input.chunk(2, dim=-1)
    a = a.detach().requires_grad_(True)
    b = b.detach().requires_grad_(True)

    with torch.enable_grad():
        y = F.gelu(a, approximate='tanh') * b
        y.backward(grad)

    return torch.cat([a.grad, b.grad], dim=-1)


def dqgelu_torch(grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    x = fwd_input.detach().requires_grad_(True)
    with torch.enable_grad():
        y = x * torch.sigmoid(1.702 * x)
        y.backward(grad)
    return x.grad


def dqgeglu_torch(grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    a, b = fwd_input.chunk(2, dim=-1)
    a = a.detach().requires_grad_(True)
    b = b.detach().requires_grad_(True)

    with torch.enable_grad():
        y = a * torch.sigmoid(1.702 * a) * b
        y.backward(grad)

    return torch.cat([a.grad, b.grad], dim=-1)


def drelu_torch(grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    return grad * (fwd_input > 0).to(grad.dtype)


def dreglu_torch(grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    a, b = fwd_input.chunk(2, dim=-1)

    grad_a = grad * b * (a > 0).to(grad.dtype)
    grad_b = grad * F.relu(a)

    return torch.cat([grad_a, grad_b], dim=-1)


def dsrelu_torch(grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    relu_x = F.relu(fwd_input)
    return 2 * grad * relu_x * (fwd_input > 0).to(grad.dtype)


def dsreglu_torch(grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    a, b = fwd_input.chunk(2, dim=-1)

    relu_a = F.relu(a)
    grad_a = grad * b * 2 * relu_a * (a > 0).to(grad.dtype)
    grad_b = grad * torch.square(relu_a)

    return torch.cat([grad_a, grad_b], dim=-1)


def dsilu_torch(grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    x = fwd_input.detach().requires_grad_(True)
    with torch.enable_grad():
        y = F.silu(x)
        y.backward(grad)
    return x.grad


def dswiglu_torch(grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> torch.Tensor:
    a, b = fwd_input.chunk(2, dim=-1)
    a = a.detach().requires_grad_(True)
    b = b.detach().requires_grad_(True)

    with torch.enable_grad():
        y = F.silu(a) * b
        y.backward(grad)

    return torch.cat([a.grad, b.grad], dim=-1)


def clamped_dswiglu_torch(
    grad: torch.Tensor,
    fwd_input: torch.Tensor,
    quantizer: Any,
    limit: float = 7.0,
    alpha: float = 1.702,
) -> torch.Tensor:
    """Backward pass for clamped SwiGLU matching CUDA implementation.

    CUDA implementation:
    - a (activation): clamp to upper bound only, derivative is 0 if a > limit
    - b (gate): clamp to [-limit, limit] and add 1, derivative is 0 outside range
    """
    a, b = fwd_input.chunk(2, dim=-1)

    # CUDA only clamps a to upper bound
    a_clamped = torch.clamp(a, max=limit)
    # CUDA clamps b to [-limit, limit] and adds 1
    b_clamped = torch.clamp(b, -limit, limit) + 1

    a_clamped = a_clamped.detach().requires_grad_(True)
    b_clamped = b_clamped.detach().requires_grad_(True)

    with torch.enable_grad():
        y = a_clamped * torch.sigmoid(alpha * a_clamped) * b_clamped
        y.backward(grad)

    # Derivative of a clamp (upper bound only): 0 if a > limit
    grad_a = a_clamped.grad * (a <= limit).to(grad.dtype)
    # Derivative of b clamp ([-limit, limit]): 0 outside range
    grad_b = b_clamped.grad * ((b >= -limit) & (b <= limit)).to(grad.dtype)

    return torch.cat([grad_a, grad_b], dim=-1)


def dbias_dgelu_torch(
    grad: torch.Tensor,
    fwd_input: torch.Tensor,
    quantizer: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grad_input = dgelu_torch(grad, fwd_input, quantizer)

    grad_bias = grad.sum(dim=tuple(range(grad.ndim - 1)))

    return grad_input, grad_bias


def dbias_dsilu_torch(
    grad: torch.Tensor,
    fwd_input: torch.Tensor,
    quantizer: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grad_input = dsilu_torch(grad, fwd_input, quantizer)

    grad_bias = grad.sum(dim=tuple(range(grad.ndim - 1)))

    return grad_input, grad_bias


def dbias_drelu_torch(
    grad: torch.Tensor,
    fwd_input: torch.Tensor,
    quantizer: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grad_input = drelu_torch(grad, fwd_input, quantizer)

    grad_bias = grad.sum(dim=tuple(range(grad.ndim - 1)))

    return grad_input, grad_bias


def dbias_dqgelu_torch(
    grad: torch.Tensor,
    fwd_input: torch.Tensor,
    quantizer: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grad_input = dqgelu_torch(grad, fwd_input, quantizer)

    grad_bias = grad.sum(dim=tuple(range(grad.ndim - 1)))

    return grad_input, grad_bias


def dbias_dsrelu_torch(
    grad: torch.Tensor,
    fwd_input: torch.Tensor,
    quantizer: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grad_input = dsrelu_torch(grad, fwd_input, quantizer)

    grad_bias = grad.sum(dim=tuple(range(grad.ndim - 1)))

    return grad_input, grad_bias
