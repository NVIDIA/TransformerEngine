# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""High-precision passthrough quantizer and tensor.

``IdentityQuantizer`` stores a tensor directly without a low-precision encoding,
preserving the input dtype by default. It exists so the ``CustomRecipe`` +
``qfactory`` machinery can express *high-precision* tensors and, composed inside
a :class:`HybridQuantizer`, *high-precision directions* without scattering
``None``/``isinstance`` special-cases across the modules. Here high precision
means the held compute dtype, typically BF16, FP16, or FP32; it does not mean
FP32 specifically.
"""

from __future__ import annotations
from typing import Any, Iterable, Optional, Tuple

import torch
from torch.ops import aten

from .storage.identity_tensor_storage import IdentityTensorStorage
from ..quantized_tensor import QuantizedTensor, QuantizedTensorStorage, Quantizer


class IdentityQuantizer(Quantizer):
    """Quantizer that produces a high-precision passthrough representation.

    Returns an :class:`IdentityTensorStorage` (or :class:`IdentityTensor`)
    holding the tensor directly, without a low-precision encoding.
    ``general_gemm`` materializes it as a plain tensor, so a GEMM consumes it
    in the held dtype.

    Parameters
    ----------
    dtype : torch.dtype, optional
        If set, the held tensor is cast to this dtype on quantize. ``None``
        (default) keeps the input's dtype.
    rowwise, columnwise : bool
        Usage flags (kept for interface compatibility; the single
        high-precision buffer serves both directions).
    """

    def __init__(
        self,
        *,
        dtype: Optional[torch.dtype] = None,
        rowwise: bool = True,
        columnwise: bool = True,
    ) -> None:
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.dtype = dtype

    def copy(self) -> "IdentityQuantizer":
        """Create shallow copy."""
        quantizer = IdentityQuantizer(
            dtype=self.dtype,
            rowwise=self.rowwise_usage,
            columnwise=self.columnwise_usage,
        )
        quantizer.internal = self.internal
        quantizer.optimize_for_gemm = self.optimize_for_gemm
        return quantizer

    def _maybe_cast(self, tensor: torch.Tensor) -> torch.Tensor:
        # Detach so the held buffer is plain "data" with no autograd graph edge,
        # mirroring the real quantizers (whose quantize kernels emit fresh,
        # non-differentiable tensors). Autograd connectivity for the *quantize*
        # op is provided separately by ``_QuantizeFunc`` in ``Quantizer.quantize``;
        # the surrounding TE module Function computes dgrad/wgrad manually. Without
        # the detach the produced tensor aliases a grad-requiring input (e.g. the
        # weight workspace returned across the module Function boundary), which
        # creates a spurious empty grad edge.
        out = tensor.detach()
        if self.dtype is not None and out.dtype != self.dtype:
            return out.to(self.dtype)
        return out

    def quantize_impl(self, tensor: torch.Tensor) -> QuantizedTensorStorage:
        data = self._maybe_cast(tensor)
        if self.internal:
            return IdentityTensorStorage(
                hp_data=data,
                fake_dtype=data.dtype,
                quantizer=self,
            )
        # requires_grad=False: this is the quantized "data" tensor. Autograd
        # connectivity is provided by ``_QuantizeFunc`` in ``Quantizer.quantize``
        # (mirrors the real quantizers, which return non-differentiable data).
        return IdentityTensor(
            data.shape,
            data.dtype,
            hp_data=data,
            quantizer=self,
            requires_grad=False,
            device=data.device,
        )

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
        pin_memory: bool = False,
    ) -> "IdentityTensor":
        if device is None:
            device = torch.device("cuda")
        device = torch.device(device)
        data_dtype = self.dtype if self.dtype is not None else dtype
        data = torch.empty(tuple(shape), dtype=data_dtype, device=device, pin_memory=pin_memory)
        return IdentityTensor(
            data.shape,
            data_dtype,
            hp_data=data,
            quantizer=self,
            requires_grad=requires_grad,
            device=device,
        )

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: QuantizedTensorStorage,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> QuantizedTensorStorage:
        if not isinstance(dst, IdentityTensorStorage):
            raise ValueError(
                f"IdentityQuantizer can only update IdentityTensorStorage, got {type(dst).__name__}"
            )
        data = self._maybe_cast(src)
        if (
            dst._hp_data is not None
            and dst._hp_data.shape == data.shape
            and dst._hp_data.dtype == data.dtype
            and dst._hp_data.device == data.device
        ):
            if noop_flag is None:
                dst._hp_data.copy_(data)
            else:
                torch.where(noop_flag == 0, data, dst._hp_data, out=dst._hp_data)
        else:
            if noop_flag is not None and noop_flag.item() != 0:
                return dst
            dst._hp_data = data.detach()
        dst._dtype = data.dtype
        return dst

    def calibrate(self, tensor: torch.Tensor) -> None:
        # No state to calibrate.
        return

    def _get_compatible_recipe(self):
        # Only reachable via CustomRecipe (qfactory returns IdentityQuantizer).
        from transformer_engine.common.recipe import CustomRecipe  # avoid circular import

        return CustomRecipe


class IdentityTensor(IdentityTensorStorage, QuantizedTensor):
    """High-precision passthrough tensor produced by :class:`IdentityQuantizer`.

    Presents as a standard tensor of its nominal dtype; internally it just
    holds data directly in that dtype, without a low-precision encoding.
    """

    def __repr__(self, *, tensor_contents=None):
        return f"IdentityTensor(dtype={self.dtype}, data={self._hp_data})"

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return IdentityTensorStorage.dequantize(self, dtype=dtype)

    def view(self, *shape) -> "IdentityTensor":
        # pylint: disable=missing-function-docstring
        flat_shape = shape[0] if len(shape) == 1 and not isinstance(shape[0], int) else shape
        return self._wrap_data_view(self._hp_data.view(*flat_shape))

    def detach(self) -> "IdentityTensor":
        # pylint: disable=missing-function-docstring
        return self._wrap_data_view(self._hp_data.detach(), requires_grad=False)

    def clone(self) -> "IdentityTensor":
        # pylint: disable=missing-function-docstring
        data = self._hp_data.detach().clone() if self._hp_data is not None else None
        return IdentityTensor(
            self.shape,
            self.dtype,
            hp_data=data,
            quantizer=self._quantizer,
            requires_grad=self.requires_grad,
            device=self.device,
            stride=data.stride(),
            storage_offset=data.storage_offset(),
        )

    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> "IdentityTensor":
        """Return an IdentityTensor with contiguous high-precision storage."""
        if self._hp_data is not None and self._hp_data.is_contiguous(memory_format=memory_format):
            return self
        return self._wrap_data_view(self._hp_data.contiguous(memory_format=memory_format))

    def __reduce_ex__(self, protocol: int) -> tuple:
        """Custom pickling that preserves the high-precision payload."""
        return (
            _make_identity_tensor_in_reduce_ex,
            (self._hp_data, self._quantizer, self.dtype, self.shape),
        )

    def fsdp_pre_all_gather(  # pylint: disable=unused-argument
        self, mesh, orig_size, contiguous_orig_stride, module, mp_policy
    ):
        """Extract the high-precision buffer for FSDP2 all-gather."""
        return (self._hp_data,), (self._quantizer,)

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional["IdentityTensor"] = None,
    ):
        """Rebuild IdentityTensor from the gathered high-precision buffer."""
        (data,) = all_gather_outputs
        (quantizer,) = metadata
        logical_dtype = (
            quantizer.dtype
            if quantizer is not None and quantizer.dtype is not None
            else param_dtype
        )
        if data.dtype != logical_dtype:
            raise RuntimeError(
                "IdentityTensor FSDP payload dtype does not match its logical dtype: "
                f"payload={data.dtype}, logical={logical_dtype}."
            )
        if out is not None:
            out._hp_data = data
            out._dtype = logical_dtype
        else:
            out = IdentityTensor(
                shape=data.shape,
                dtype=logical_dtype,
                hp_data=data,
                quantizer=quantizer,
                requires_grad=False,
                device=data.device,
            )
        return out, all_gather_outputs

    def _wrap_data_view(
        self, data: torch.Tensor, *, requires_grad: Optional[bool] = None
    ) -> "IdentityTensor":
        requires_grad = self.requires_grad if requires_grad is None else requires_grad
        return IdentityTensor(
            shape=data.shape,
            dtype=self.dtype,
            hp_data=data,
            quantizer=self._quantizer,
            requires_grad=requires_grad,
            device=data.device,
            stride=data.stride(),
            storage_offset=data.storage_offset(),
        )

    @classmethod
    def _delegate_view_op(cls, func, tensor, args, kwargs):
        """Apply an alias-preserving view op to the held tensor and rewrap it."""

        result = func(tensor._hp_data, *args[1:], **kwargs)

        def _wrap(value):
            if isinstance(value, torch.Tensor):
                return tensor._wrap_data_view(value)
            return value

        if isinstance(result, tuple):
            return tuple(_wrap(value) for value in result)
        if isinstance(result, list):
            return [_wrap(value) for value in result]
        return _wrap(result)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func == aten.detach.default:
            return args[0].detach()

        if func == aten.clone.default:
            return args[0].clone()

        if func in (
            aten.view.default,
            aten.split.Tensor,
            aten.as_strided.default,
            aten.slice.Tensor,
        ):
            # Preserve optional arguments exactly; omitted as_strided offset
            # means reuse the input view's current storage offset, not zero.
            return cls._delegate_view_op(func, args[0], args, kwargs)

        if func == aten.copy_.default:
            dst, src = args[0], args[1]
            if isinstance(dst, IdentityTensor):
                src_data = src._hp_data if isinstance(src, IdentityTensor) else src
                dst._hp_data.copy_(src_data, *args[2:], **kwargs)
                return dst

        if func == aten.new_zeros.default:
            tensor = args[0]
            new_shape = args[1]
            if tensor._quantizer is not None:
                out = tensor._quantizer.make_empty(
                    new_shape,
                    dtype=kwargs.get("dtype") or tensor.dtype,
                    device=kwargs.get("device") or tensor.device,
                    pin_memory=bool(kwargs.get("pin_memory", False)),
                )
                out._hp_data.zero_()
                return out

        return super().__torch_dispatch__(func, types, args, kwargs)


def _make_identity_tensor_in_reduce_ex(
    hp_data: torch.Tensor,
    quantizer: Optional[Quantizer],
    dtype: torch.dtype,
    shape: torch.Size,
) -> IdentityTensor:
    """Reconstruct an ``IdentityTensor`` from its ``__reduce_ex__`` payload."""
    return IdentityTensor(
        shape=shape,
        dtype=dtype,
        hp_data=hp_data,
        quantizer=quantizer,
        requires_grad=False,
        device=hp_data.device if hp_data is not None else None,
    )
