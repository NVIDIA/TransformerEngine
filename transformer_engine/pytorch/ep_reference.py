# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Pure PyTorch semantic reference for a SwiGLU MoE with expert parallelism.

The implementation deliberately favors readable semantics over performance.  It
supports a one-rank execution path and a variable-size ``all_to_all_single`` EP
path, plus BF16, MXFP8, and NVFP4 block-scaled public outputs.

The quantized tensor layouts are logical (unswizzled) layouts.  A production
kernel may reorder scale factors internally without changing this API contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F


class MoeFormat(str, Enum):
    """Public and communication formats supported by the reference."""

    BF16 = "bf16"
    MXFP8 = "mxfp8"
    NVFP4 = "nvfp4"


def _parse_format(value: Union[MoeFormat, str]) -> MoeFormat:
    if isinstance(value, MoeFormat):
        return value
    try:
        return MoeFormat(value.lower())
    except (AttributeError, ValueError) as exc:
        choices = ", ".join(item.value for item in MoeFormat)
        raise ValueError(f"unsupported format {value!r}; expected one of: {choices}") from exc


def _require_torch_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if dtype is None:
        raise RuntimeError(f"this PyTorch build does not provide torch.{name}")
    return dtype


def _normalize_axis(axis: int, ndim: int) -> int:
    normalized = axis + ndim if axis < 0 else axis
    if normalized < 0 or normalized >= ndim:
        raise IndexError(f"axis {axis} is out of range for a {ndim}-D tensor")
    return normalized


def _shape_with_axis(shape: Sequence[int], axis: int, value: int) -> Tuple[int, ...]:
    result = list(shape)
    result[axis] = value
    return tuple(result)


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


@dataclass(frozen=True)
class BlockScaledTensor:
    """Portable data-plus-scale representation for MXFP8 or NVFP4.

    ``logical_shape`` describes the dequantized tensor.  For MXFP8, ``data``
    has that shape and uses E4M3.  For NVFP4, ``data`` is a uint8 tensor with
    two E2M1 values per byte along ``axis`` (low nibble first).  ``scale``
    replaces that axis by one scale per block.
    """

    data: torch.Tensor
    scale: torch.Tensor
    format: Union[MoeFormat, str]
    logical_shape: Tuple[int, ...]
    axis: int = -1

    def __post_init__(self) -> None:
        fmt = _parse_format(self.format)
        if fmt is MoeFormat.BF16:
            raise ValueError("BlockScaledTensor only represents mxfp8 or nvfp4")
        shape = tuple(int(dim) for dim in self.logical_shape)
        if not shape or any(dim < 0 for dim in shape):
            raise ValueError(f"logical_shape must contain non-negative dimensions, got {shape}")
        axis = _normalize_axis(self.axis, len(shape))
        object.__setattr__(self, "format", fmt)
        object.__setattr__(self, "logical_shape", shape)
        object.__setattr__(self, "axis", axis)
        self._validate_storage()

    @property
    def block_size(self) -> int:
        return 32 if self.format is MoeFormat.MXFP8 else 16

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.logical_shape

    @property
    def device(self) -> torch.device:
        return self.data.device

    def _validate_storage(self) -> None:
        if self.data.device != self.scale.device:
            raise ValueError("block-scaled data and scale must be on the same device")

        logical_extent = self.logical_shape[self.axis]
        scale_shape = _shape_with_axis(
            self.logical_shape,
            self.axis,
            _ceil_div(logical_extent, self.block_size),
        )
        if tuple(self.scale.shape) != scale_shape:
            raise ValueError(f"scale shape must be {scale_shape}, got {tuple(self.scale.shape)}")

        if self.format is MoeFormat.MXFP8:
            expected_dtype = _require_torch_dtype("float8_e4m3fn")
            expected_scale_dtype = _require_torch_dtype("float8_e8m0fnu")
            data_shape = self.logical_shape
            if self.data.dtype != expected_dtype:
                raise TypeError(
                    f"mxfp8 data must have dtype {expected_dtype}, got {self.data.dtype}"
                )
        else:
            expected_scale_dtype = _require_torch_dtype("float8_e4m3fn")
            fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
            if self.data.dtype != torch.uint8 and self.data.dtype != fp4_dtype:
                raise TypeError("nvfp4 data must be packed uint8 or torch.float4_e2m1fn_x2")
            data_shape = _shape_with_axis(
                self.logical_shape,
                self.axis,
                _ceil_div(logical_extent, 2),
            )

        if tuple(self.data.shape) != data_shape:
            raise ValueError(f"data shape must be {data_shape}, got {tuple(self.data.shape)}")
        if self.scale.dtype != expected_scale_dtype:
            raise TypeError(f"scale must have dtype {expected_scale_dtype}, got {self.scale.dtype}")

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Return the logical tensor with block scales applied."""

        logical_extent = self.logical_shape[self.axis]
        scale = self.scale.movedim(self.axis, -1).float()
        expanded_scale = scale.repeat_interleave(self.block_size, dim=-1)[..., :logical_extent]

        if self.format is MoeFormat.MXFP8:
            values = self.data.movedim(self.axis, -1).float()
        else:
            packed = self.data
            if packed.dtype != torch.uint8:
                packed = packed.view(torch.uint8)
            packed = packed.movedim(self.axis, -1)
            low = packed & 0x0F
            high = packed >> 4
            codes = torch.stack((low, high), dim=-1).flatten(-2)[..., :logical_extent]
            table = torch.tensor(
                [
                    0.0,
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    3.0,
                    4.0,
                    6.0,
                    -0.0,
                    -0.5,
                    -1.0,
                    -1.5,
                    -2.0,
                    -3.0,
                    -4.0,
                    -6.0,
                ],
                dtype=torch.float32,
                device=packed.device,
            )
            values = table[codes.long()]

        return (values * expanded_scale).movedim(-1, self.axis).to(dtype)


def _nearest_e2m1_codes(values: torch.Tensor) -> torch.Tensor:
    """Quantize to E2M1 nibble codes with round-to-nearest, ties-to-even."""

    levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=values.device,
    )
    magnitudes = values.abs().unsqueeze(-1)
    distances = (magnitudes - levels).abs()
    minimum = distances.amin(dim=-1, keepdim=True)
    candidates = distances == minimum
    codes = torch.arange(8, dtype=torch.int64, device=values.device)
    any_code = torch.where(candidates, codes, 8).amin(dim=-1)
    even_code = torch.where(candidates & ((codes & 1) == 0), codes, 8).amin(dim=-1)
    magnitude_code = torch.where(even_code < 8, even_code, any_code)
    sign_code = torch.signbit(values).to(torch.int64) << 3
    return magnitude_code | sign_code


def quantize_blockwise(
    tensor: torch.Tensor,
    format: Union[MoeFormat, str],
    *,
    axis: int = -1,
) -> BlockScaledTensor:
    """Quantize a floating tensor into logical MXFP8 or NVFP4 blocks.

    MXFP8 uses 32-value blocks, E4M3 payloads, and E8M0 scales rounded toward
    positive infinity.  NVFP4 uses 16-value blocks, packed E2M1 payloads, and
    E4M3 scales rounded to nearest.
    """

    fmt = _parse_format(format)
    if fmt is MoeFormat.BF16:
        raise ValueError("quantize_blockwise requires mxfp8 or nvfp4")
    if not tensor.is_floating_point():
        raise TypeError(f"tensor must be floating point, got {tensor.dtype}")

    axis = _normalize_axis(axis, tensor.ndim)
    logical_shape = tuple(tensor.shape)
    moved = tensor.float().movedim(axis, -1)
    logical_extent = moved.shape[-1]
    block_size = 32 if fmt is MoeFormat.MXFP8 else 16
    block_count = _ceil_div(logical_extent, block_size)
    padded_extent = block_count * block_size
    if padded_extent != logical_extent:
        moved = F.pad(moved, (0, padded_extent - logical_extent))
    blocks = moved.reshape(*moved.shape[:-1], block_count, block_size)

    value_limit = 448.0 if fmt is MoeFormat.MXFP8 else 6.0
    scale_float = blocks.abs().amax(dim=-1) / value_limit
    if fmt is MoeFormat.MXFP8:
        safe_scale = torch.where(scale_float > 0, scale_float, 1.0)
        scale_float = torch.where(
            scale_float > 0,
            torch.pow(2.0, torch.ceil(torch.log2(safe_scale))),
            torch.zeros_like(scale_float),
        )
        scale_dtype = _require_torch_dtype("float8_e8m0fnu")
    else:
        scale_dtype = _require_torch_dtype("float8_e4m3fn")

    scale = scale_float.to(scale_dtype)
    scale_for_math = scale.float()
    reciprocal = torch.where(scale_for_math > 0, scale_for_math.reciprocal(), 0.0)
    normalized = (blocks * reciprocal.unsqueeze(-1)).clamp(-value_limit, value_limit)

    if fmt is MoeFormat.MXFP8:
        data_dtype = _require_torch_dtype("float8_e4m3fn")
        data = normalized.to(data_dtype).reshape(*moved.shape)[..., :logical_extent]
    else:
        codes = _nearest_e2m1_codes(normalized).reshape(*moved.shape)
        low = codes[..., 0::2]
        high = codes[..., 1::2]
        data = (low | (high << 4)).to(torch.uint8)[..., : _ceil_div(logical_extent, 2)]

    return BlockScaledTensor(
        data=data.movedim(-1, axis).contiguous(),
        scale=scale.movedim(-1, axis).contiguous(),
        format=fmt,
        logical_shape=logical_shape,
        axis=axis,
    )


MoeTensor = Union[torch.Tensor, BlockScaledTensor]


@dataclass(frozen=True)
class _DispatchPlan:
    """Send-side routing derived from ``topk_idx``; identical in fwd and bwd."""

    send_expert: torch.Tensor  # local expert id per sent route
    send_weight: torch.Tensor  # router weight per sent route
    send_token_idx: torch.Tensor  # source token per sent route
    send_slot_idx: torch.Tensor  # source top-k slot per sent route
    send_counts: Tuple[int, ...]  # routes sent to each rank
    recv_counts: Tuple[int, ...]  # routes received from each rank


def _tensor_device(tensor: MoeTensor) -> torch.device:
    return tensor.device


def _decode_tensor(
    tensor: MoeTensor,
    *,
    name: str,
    expected_shape: Tuple[int, ...],
    quantized_axis: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(tensor, BlockScaledTensor):
        if tensor.logical_shape != expected_shape:
            raise ValueError(
                f"{name} logical shape must be {expected_shape}, got {tensor.logical_shape}"
            )
        if tensor.axis != _normalize_axis(quantized_axis, len(expected_shape)):
            raise ValueError(f"{name} must be block-scaled along axis {quantized_axis}")
        return tensor.dequantize(dtype=dtype)

    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"{name} shape must be {expected_shape}, got {tuple(tensor.shape)}")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be floating point or BlockScaledTensor, got {tensor.dtype}")
    return tensor.to(dtype=dtype)


def _format_round_trip(
    tensor: torch.Tensor,
    format: MoeFormat,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if format is MoeFormat.BF16:
        return tensor.to(torch.bfloat16).to(dtype)
    return quantize_blockwise(tensor, format, axis=-1).dequantize(dtype=dtype)


class MoeEpReference:
    """Reference implementation of routed SwiGLU experts plus EP dispatch.

    Global experts are assigned contiguously: rank ``r`` owns
    ``[r * experts_per_rank, (r + 1) * experts_per_rank)``.  Pass an explicit
    initialized process group for multi-rank execution; ``None`` means a
    one-rank reference even if the default distributed group is initialized.
    """

    def __init__(
        self,
        *,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        top_k: int,
        ep_group: Optional[dist.ProcessGroup] = None,
        max_tokens_per_rank: Optional[int] = None,
        output_format: Union[MoeFormat, str] = MoeFormat.BF16,
        combine_format: Union[MoeFormat, str] = MoeFormat.BF16,
        apply_topk_in_fc1: bool = True,
        gate_up_clamp: Optional[float] = None,
        generate_c: bool = False,
        compute_dtype: torch.dtype = torch.float32,
    ) -> None:
        for name, value in (
            ("num_experts", num_experts),
            ("hidden_size", hidden_size),
            ("intermediate_size", intermediate_size),
            ("top_k", top_k),
        ):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) cannot exceed num_experts ({num_experts})")
        if max_tokens_per_rank is not None and max_tokens_per_rank < 0:
            raise ValueError("max_tokens_per_rank must be non-negative")
        if compute_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(
                f"compute_dtype must be torch.float32 or torch.bfloat16, got {compute_dtype}"
            )

        if ep_group is None:
            ep_size, ep_rank = 1, 0
        else:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError(
                    "ep_group requires an initialized torch.distributed process group"
                )
            ep_size = dist.get_world_size(ep_group)
            ep_rank = dist.get_rank(ep_group)
        if num_experts % ep_size != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by EP size ({ep_size})"
            )

        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.ep_group = ep_group
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.experts_per_rank = num_experts // ep_size
        self.max_tokens_per_rank = max_tokens_per_rank
        self.output_format = _parse_format(output_format)
        self.combine_format = _parse_format(combine_format)
        self.apply_topk_in_fc1 = bool(apply_topk_in_fc1)
        self.gate_up_clamp = None if gate_up_clamp is None else abs(float(gate_up_clamp))
        self.generate_c = bool(generate_c)
        self.compute_dtype = compute_dtype

        for name, fmt in (
            ("output_format", self.output_format),
            ("combine_format", self.combine_format),
        ):
            required_multiple = (
                32 if fmt is MoeFormat.MXFP8 else 16 if fmt is MoeFormat.NVFP4 else 1
            )
            if hidden_size % required_multiple != 0:
                raise ValueError(
                    f"hidden_size ({hidden_size}) must be divisible by {required_multiple} for"
                    f" {name}={fmt.value}"
                )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"experts={self.num_experts}, local_experts={self.experts_per_rank}, "
            f"hidden={self.hidden_size}, intermediate={self.intermediate_size}, "
            f"top_k={self.top_k}, ep_rank={self.ep_rank}/{self.ep_size}, "
            f"output={self.output_format.value}, combine={self.combine_format.value}, "
            f"compute_dtype={self.compute_dtype})"
        )

    def _collective_device(self, device: torch.device) -> torch.device:
        """Device the process group can run ``all_to_all_single`` on.

        Gloo only implements all-to-all for CPU tensors, so CUDA tensors are
        staged through host memory; NCCL groups communicate in place.
        """
        if device.type != "cpu" and dist.get_backend(self.ep_group) == "gloo":
            return torch.device("cpu")
        return device

    def _exchange_counts(self, send_counts: torch.Tensor) -> torch.Tensor:
        if self.ep_size == 1:
            return send_counts.clone()
        comm_device = self._collective_device(send_counts.device)
        staged = send_counts.to(comm_device)
        recv_counts = torch.empty_like(staged)
        dist.all_to_all_single(recv_counts, staged, group=self.ep_group)
        return recv_counts.to(send_counts.device)

    def _all_to_all(
        self,
        send: torch.Tensor,
        send_counts: Sequence[int],
        recv_counts: Sequence[int],
    ) -> torch.Tensor:
        if self.ep_size == 1:
            return send.clone()
        comm_device = self._collective_device(send.device)
        staged = send.contiguous().to(comm_device)
        output_shape = (sum(recv_counts), *send.shape[1:])
        recv = torch.empty(output_shape, dtype=send.dtype, device=comm_device)
        dist.all_to_all_single(
            recv,
            staged,
            output_split_sizes=list(recv_counts),
            input_split_sizes=list(send_counts),
            group=self.ep_group,
        )
        return recv.to(send.device)

    def _dispatch_plan(self, topk_idx: torch.Tensor, topk_weights: torch.Tensor) -> _DispatchPlan:
        """Route valid ``topk_idx`` entries to destination ranks, stably by rank.

        Backward reuses this so gradient re-dispatch reproduces the exact
        forward route order.
        """

        device = topk_idx.device
        token_count = topk_idx.shape[0]
        flat_expert = topk_idx.reshape(-1).to(torch.int64)
        flat_weight = topk_weights.reshape(-1).float()
        valid = flat_expert != -1
        invalid_negative = flat_expert < -1
        invalid_high = flat_expert >= self.num_experts
        if bool((invalid_negative | invalid_high).any().item()):
            bad = flat_expert[invalid_negative | invalid_high][0].item()
            raise ValueError(f"topk_idx contains out-of-range expert id {bad}")

        flat_token = torch.arange(token_count, device=device).repeat_interleave(self.top_k)
        flat_slot = torch.arange(self.top_k, device=device).repeat(token_count)
        expert = flat_expert[valid]
        destination = torch.div(expert, self.experts_per_rank, rounding_mode="floor")
        order = torch.argsort(destination, stable=True)

        send_counts_tensor = torch.bincount(
            destination.index_select(0, order), minlength=self.ep_size
        ).to(torch.int64)
        recv_counts_tensor = self._exchange_counts(send_counts_tensor)
        return _DispatchPlan(
            send_expert=expert.index_select(0, order).remainder(self.experts_per_rank),
            send_weight=flat_weight[valid].index_select(0, order),
            send_token_idx=flat_token[valid].index_select(0, order),
            send_slot_idx=flat_slot[valid].index_select(0, order),
            send_counts=tuple(int(v) for v in send_counts_tensor.cpu().tolist()),
            recv_counts=tuple(int(v) for v in recv_counts_tensor.cpu().tolist()),
        )

    def _run_local_experts(
        self,
        tokens: torch.Tensor,
        local_expert_idx: torch.Tensor,
        route_weight: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc2_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        output = torch.empty(
            (tokens.shape[0], self.hidden_size),
            dtype=self.compute_dtype,
            device=tokens.device,
        )
        fc1_c_rows = [] if self.generate_c else None
        for expert in range(self.experts_per_rank):
            positions = torch.nonzero(local_expert_idx == expert, as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            expert_tokens = tokens.index_select(0, positions)
            gate_up = expert_tokens @ fc1_weight[expert]
            if fc1_c_rows is not None:
                # Raw pre-SwiGLU accumulator: before clamp, no router weight.
                fc1_c_rows.append(gate_up.to(torch.bfloat16))
            gate, up = gate_up.split(self.intermediate_size, dim=-1)
            if self.gate_up_clamp is not None:
                gate = gate.clamp(max=self.gate_up_clamp)
                up = up.clamp(min=-self.gate_up_clamp, max=self.gate_up_clamp)
            intermediate = F.silu(gate) * up
            weights = route_weight.index_select(0, positions).unsqueeze(-1)
            if self.apply_topk_in_fc1:
                intermediate = intermediate * weights
            expert_output = intermediate @ fc2_weight[expert]
            if not self.apply_topk_in_fc1:
                expert_output = expert_output * weights
            expert_output = _format_round_trip(
                expert_output,
                self.combine_format,
                dtype=self.compute_dtype,
            )
            output.index_copy_(0, positions, expert_output)
        fc1_c = None
        if fc1_c_rows is not None:
            fc1_c = (
                torch.cat(fc1_c_rows)
                if fc1_c_rows
                else torch.empty(
                    (0, 2 * self.intermediate_size), dtype=torch.bfloat16, device=tokens.device
                )
            )
        return output, fc1_c

    def __call__(
        self,
        activation: MoeTensor,
        fc1_weight: MoeTensor,
        fc2_weight: MoeTensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Union[MoeTensor, Tuple[MoeTensor, torch.Tensor, torch.Tensor]]:
        """Run dispatch, local experts, return routing, top-k reduce, and encode.

        Shapes:
            activation: ``(T, H)``
            fc1_weight: ``(E_local, H, 2 * I)``
            fc2_weight: ``(E_local, I, H)``
            topk_idx/topk_weights: ``(T, K)``

        Returns the ``(T, H)`` result, or ``(result, fc1_c, route_metadata)``
        when constructed with ``generate_c=True``.  ``fc1_c`` is the BF16
        pre-SwiGLU FC1 accumulator of every route this rank's experts
        processed, ``(local_routes, 2 * I)``, grouped by local expert and
        ordered within each expert by (source rank, source token-major route
        order); captured before the gate/up clamp, without the router weight.
        ``route_metadata`` is Int32 ``(local_routes, 4)`` with columns
        ``(local_expert, src_rank, src_token, src_slot)``; row ``i`` identifies
        the route behind ``fc1_c`` row ``i`` for the backward gradient
        re-dispatch.
        """

        if topk_idx.ndim != 2:
            raise ValueError(f"topk_idx must be 2-D, got shape {tuple(topk_idx.shape)}")
        token_count = topk_idx.shape[0]
        route_shape = (token_count, self.top_k)
        if tuple(topk_idx.shape) != route_shape:
            raise ValueError(f"topk_idx shape must be {route_shape}, got {tuple(topk_idx.shape)}")
        if tuple(topk_weights.shape) != route_shape:
            raise ValueError(
                f"topk_weights shape must be {route_shape}, got {tuple(topk_weights.shape)}"
            )
        if topk_idx.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"topk_idx must be int32 or int64, got {topk_idx.dtype}")
        if not topk_weights.is_floating_point():
            raise TypeError(f"topk_weights must be floating point, got {topk_weights.dtype}")
        if self.max_tokens_per_rank is not None and token_count > self.max_tokens_per_rank:
            raise ValueError(
                f"token count {token_count} exceeds max_tokens_per_rank={self.max_tokens_per_rank}"
            )

        device = _tensor_device(activation)
        inputs = {
            "fc1_weight": _tensor_device(fc1_weight),
            "fc2_weight": _tensor_device(fc2_weight),
            "topk_idx": topk_idx.device,
            "topk_weights": topk_weights.device,
        }
        for name, input_device in inputs.items():
            if input_device != device:
                raise ValueError(f"{name} must be on {device}, got {input_device}")

        activation_float = _decode_tensor(
            activation,
            name="activation",
            expected_shape=(token_count, self.hidden_size),
            quantized_axis=1,
            dtype=self.compute_dtype,
        )
        fc1_float = _decode_tensor(
            fc1_weight,
            name="fc1_weight",
            expected_shape=(self.experts_per_rank, self.hidden_size, 2 * self.intermediate_size),
            quantized_axis=1,
            dtype=self.compute_dtype,
        )
        fc2_float = _decode_tensor(
            fc2_weight,
            name="fc2_weight",
            expected_shape=(self.experts_per_rank, self.intermediate_size, self.hidden_size),
            quantized_axis=1,
            dtype=self.compute_dtype,
        )

        plan = self._dispatch_plan(topk_idx, topk_weights)
        send_token_idx = plan.send_token_idx
        send_slot_idx = plan.send_slot_idx
        send_counts, recv_counts = plan.send_counts, plan.recv_counts
        send_tokens = activation_float.index_select(0, send_token_idx)

        recv_tokens = self._all_to_all(send_tokens, send_counts, recv_counts)
        recv_expert = self._all_to_all(plan.send_expert, send_counts, recv_counts)
        recv_weight = self._all_to_all(plan.send_weight, send_counts, recv_counts).to(
            dtype=self.compute_dtype
        )

        route_metadata = None
        if self.generate_c:
            recv_src_rank = torch.repeat_interleave(
                torch.arange(self.ep_size, device=device),
                torch.tensor(recv_counts, device=device),
            )
            recv_token = self._all_to_all(send_token_idx, send_counts, recv_counts)
            recv_slot = self._all_to_all(send_slot_idx, send_counts, recv_counts)
            # Stable sort by local expert reproduces the fc1_c row order
            # (grouped by expert; source order preserved within each group).
            fc1_c_order = torch.argsort(recv_expert, stable=True)
            route_metadata = (
                torch.stack((recv_expert, recv_src_rank, recv_token, recv_slot), dim=1)
                .index_select(0, fc1_c_order)
                .to(torch.int32)
            )
        # recv rows are ordered by source rank, then that source's token-major
        # route order, so the per-expert position grouping below realizes the
        # documented fc1_c ordering.
        recv_output, fc1_c = self._run_local_experts(
            recv_tokens,
            recv_expert,
            recv_weight,
            fc1_float,
            fc2_float,
        )

        returned = self._all_to_all(recv_output, recv_counts, send_counts)
        combine_plane = torch.zeros(
            (token_count * self.top_k, self.hidden_size),
            dtype=self.compute_dtype,
            device=device,
        )
        send_flat_slot = send_token_idx * self.top_k + send_slot_idx
        combine_plane.index_copy_(0, send_flat_slot, returned)
        reduced = combine_plane.view(token_count, self.top_k, self.hidden_size).sum(dim=1)

        if self.output_format is MoeFormat.BF16:
            output = reduced.to(torch.bfloat16)
        else:
            output = quantize_blockwise(reduced, self.output_format, axis=-1)
        if self.generate_c:
            return output, fc1_c, route_metadata
        return output

    def backward(
        self,
        grad_output: torch.Tensor,
        activation: MoeTensor,
        fc1_weight: MoeTensor,
        fc2_weight: MoeTensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        fc1_c: torch.Tensor,
        route_metadata: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backward pass consuming the ``generate_c=True`` stash.

        ``fc1_c`` is the recompute source: gate/up, the clamp masks, SwiGLU,
        and the FC2 input are all rebuilt from it, so no post-SwiGLU forward
        intermediate needs to be saved.  ``route_metadata`` alone reconstructs
        the mapping between re-dispatched rows and ``fc1_c`` rows and drives
        the gradient return scatter.

        Quantization round-trips (input decode, ``combine_format``,
        ``output_format``) are treated as straight-through identities;
        ``grad_output`` is the ``(T, H)`` gradient of the dequantized output.

        Returns ``(grad_activation, grad_fc1_weight, grad_fc2_weight,
        grad_topk_weights)``. Activation and expert weight gradients use
        ``compute_dtype``. Router weights are cast to ``compute_dtype`` for the
        activation scaling path; the returned
        ``grad_topk_weights`` remain float32.
        """

        if not self.generate_c:
            raise RuntimeError(
                "backward requires the operator to be constructed with generate_c=True"
            )
        token_count = topk_idx.shape[0]
        if tuple(grad_output.shape) != (token_count, self.hidden_size):
            raise ValueError(
                f"grad_output shape must be {(token_count, self.hidden_size)}, got"
                f" {tuple(grad_output.shape)}"
            )
        if not grad_output.is_floating_point():
            raise TypeError(f"grad_output must be floating point, got {grad_output.dtype}")

        device = _tensor_device(activation)
        two_i = 2 * self.intermediate_size
        activation_float = _decode_tensor(
            activation,
            name="activation",
            expected_shape=(token_count, self.hidden_size),
            quantized_axis=1,
            dtype=self.compute_dtype,
        )
        fc1_float = _decode_tensor(
            fc1_weight,
            name="fc1_weight",
            expected_shape=(self.experts_per_rank, self.hidden_size, two_i),
            quantized_axis=1,
            dtype=self.compute_dtype,
        )
        fc2_float = _decode_tensor(
            fc2_weight,
            name="fc2_weight",
            expected_shape=(self.experts_per_rank, self.intermediate_size, self.hidden_size),
            quantized_axis=1,
            dtype=self.compute_dtype,
        )
        if fc1_c.shape != (int(route_metadata.shape[0]), two_i):
            raise ValueError(
                f"fc1_c shape must be {(int(route_metadata.shape[0]), two_i)}, got"
                f" {tuple(fc1_c.shape)}"
            )

        # Re-dispatch the FC1 inputs, router weights, and output gradients
        # along the identical forward routes.
        plan = self._dispatch_plan(topk_idx, topk_weights)
        send_counts, recv_counts = plan.send_counts, plan.recv_counts
        grad_output_float = grad_output.to(dtype=self.compute_dtype)
        recv_tokens = self._all_to_all(
            activation_float.index_select(0, plan.send_token_idx), send_counts, recv_counts
        )
        # Same FP32-on-wire / compute_dtype-for-activation cast as forward.
        recv_weight = self._all_to_all(plan.send_weight, send_counts, recv_counts).to(
            dtype=self.compute_dtype
        )
        recv_grad = self._all_to_all(
            grad_output_float.index_select(0, plan.send_token_idx), send_counts, recv_counts
        )

        # route_metadata rows are in fc1_c order; sorting them by
        # (src_rank, src_token, src_slot) reproduces the receive order, giving
        # the permutation between re-dispatched rows and fc1_c rows.
        metadata = route_metadata.to(device=device, dtype=torch.int64)
        local_routes = metadata.shape[0]
        if local_routes > 0:
            token_span = int(metadata[:, 2].max().item()) + 1
            recv_key = (metadata[:, 1] * token_span + metadata[:, 2]) * self.top_k + metadata[:, 3]
            perm = torch.argsort(recv_key)  # perm[j] = fc1_c row at receive position j
        else:
            perm = torch.empty((0,), dtype=torch.int64, device=device)
        x_rows = torch.empty_like(recv_tokens)
        x_rows.index_copy_(0, perm, recv_tokens)
        w_rows = torch.empty_like(recv_weight)
        w_rows.index_copy_(0, perm, recv_weight)
        dy_rows = torch.empty_like(recv_grad)
        dy_rows.index_copy_(0, perm, recv_grad)

        c_rows = fc1_c.to(dtype=self.compute_dtype)
        expert_rows = metadata[:, 0]
        d_x_rows = torch.zeros(
            (local_routes, self.hidden_size),
            dtype=self.compute_dtype,
            device=device,
        )
        d_w_rows = torch.zeros((local_routes,), dtype=torch.float32, device=device)
        grad_fc1 = torch.zeros_like(fc1_float)
        grad_fc2 = torch.zeros_like(fc2_float)
        for expert in range(self.experts_per_rank):
            positions = torch.nonzero(expert_rows == expert, as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            c = c_rows.index_select(0, positions)
            x = x_rows.index_select(0, positions)
            w = w_rows.index_select(0, positions).unsqueeze(-1)
            d_y = dy_rows.index_select(0, positions)

            gate, up = c.split(self.intermediate_size, dim=-1)
            if self.gate_up_clamp is not None:
                g = gate.clamp(max=self.gate_up_clamp)
                u = up.clamp(min=-self.gate_up_clamp, max=self.gate_up_clamp)
            else:
                g, u = gate, up
            sig = torch.sigmoid(g)
            s = g * sig
            h = s * u

            if self.apply_topk_in_fc1:
                h_fc2 = h * w
                d_y_pre = d_y
            else:
                h_fc2 = h
                d_y_pre = d_y * w
            grad_fc2[expert] = h_fc2.transpose(0, 1) @ d_y_pre
            d_h_fc2 = d_y_pre @ fc2_float[expert].transpose(0, 1)
            if self.apply_topk_in_fc1:
                d_h = d_h_fc2 * w
                d_w_rows[positions] = (d_h_fc2 * h).sum(dim=-1).to(dtype=self.compute_dtype).float()
            else:
                d_h = d_h_fc2
                d_w_rows[positions] = (
                    (d_y * (h @ fc2_float[expert])).sum(dim=-1).to(dtype=self.compute_dtype).float()
                )

            d_g = d_h * u * (sig * (1 + g * (1 - sig)))
            d_u = d_h * s
            if self.gate_up_clamp is not None:
                d_gate = d_g * (gate <= self.gate_up_clamp)
                d_up = d_u * ((up >= -self.gate_up_clamp) & (up <= self.gate_up_clamp))
            else:
                d_gate, d_up = d_g, d_u
            d_c = torch.cat((d_gate, d_up), dim=-1)
            grad_fc1[expert] = x.transpose(0, 1) @ d_c
            d_x_rows.index_copy_(0, positions, d_c @ fc1_float[expert].transpose(0, 1))

        # Return the route gradients to their source ranks and scatter-add.
        returned_dx = self._all_to_all(d_x_rows.index_select(0, perm), recv_counts, send_counts)
        returned_dw = self._all_to_all(d_w_rows.index_select(0, perm), recv_counts, send_counts)
        grad_activation = torch.zeros(
            (token_count, self.hidden_size),
            dtype=self.compute_dtype,
            device=device,
        )
        grad_activation.index_add_(0, plan.send_token_idx, returned_dx)
        grad_topk_weights = torch.zeros(
            (token_count * self.top_k,), dtype=torch.float32, device=device
        )
        grad_topk_weights.index_copy_(
            0, plan.send_token_idx * self.top_k + plan.send_slot_idx, returned_dw
        )
        return (
            grad_activation,
            grad_fc1,
            grad_fc2,
            grad_topk_weights.view(token_count, self.top_k),
        )


__all__ = [
    "BlockScaledTensor",
    "MoeEpReference",
    "MoeFormat",
    "MoeTensor",
    "quantize_blockwise",
]
