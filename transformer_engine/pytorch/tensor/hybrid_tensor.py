# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tensor class with hybrid quantized data (different formats for rowwise vs columnwise)"""

from __future__ import annotations
from typing import Any, Dict, Iterable, Optional, Tuple

import torch

from .storage.hybrid_tensor_storage import HybridQuantizedTensorStorage
from ..quantized_tensor import QuantizedTensor, QuantizedTensorStorage, Quantizer

aten = torch.ops.aten


class HybridQuantizer(Quantizer):
    """Quantizer that composes two existing quantizers for different directions.

    Performs two-pass quantization: the rowwise_quantizer produces rowwise
    quantized data and the columnwise_quantizer produces columnwise quantized
    data. The results are wrapped in a HybridQuantizedTensor.

    Parameters
    ----------
    rowwise_quantizer : Quantizer
        Quantizer for the rowwise direction (e.g. MXFP8Quantizer).
    columnwise_quantizer : Quantizer
        Quantizer for the columnwise direction (e.g. NVFP4Quantizer).

    """

    rowwise_quantizer: Quantizer
    columnwise_quantizer: Quantizer

    def __init__(
        self,
        *,
        rowwise_quantizer: Quantizer,
        columnwise_quantizer: Quantizer,
    ) -> None:
        super().__init__(rowwise=True, columnwise=True)
        self.rowwise_quantizer = rowwise_quantizer
        self.columnwise_quantizer = columnwise_quantizer

        # Pin each sub-quantizer to its designated direction
        self.rowwise_quantizer.set_usage(rowwise=True, columnwise=False)
        self.columnwise_quantizer.set_usage(rowwise=False, columnwise=True)

    def quantize_impl(self, tensor: torch.Tensor) -> QuantizedTensor:
        rowwise_result = self.rowwise_quantizer.quantize(tensor)
        columnwise_result = self.columnwise_quantizer.quantize(tensor)

        if self.internal:
            return HybridQuantizedTensorStorage(
                rowwise_storage=rowwise_result,
                columnwise_storage=columnwise_result,
                rowwise_quantizer=self.rowwise_quantizer,
                columnwise_quantizer=self.columnwise_quantizer,
                quantizer=self,
                fake_dtype=tensor.dtype,
            )

        return HybridQuantizedTensor(
            shape=tensor.shape,
            dtype=tensor.dtype,
            rowwise_storage=rowwise_result,
            columnwise_storage=columnwise_result,
            rowwise_quantizer=self.rowwise_quantizer,
            columnwise_quantizer=self.columnwise_quantizer,
            quantizer=self,
        )

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
        pin_memory: bool = False,
    ) -> HybridQuantizedTensor:
        self.rowwise_quantizer.internal = True
        rowwise_empty = self.rowwise_quantizer.make_empty(
            shape,
            dtype=dtype,
            device=device,
            pin_memory=pin_memory,
        )
        self.rowwise_quantizer.internal = False

        self.columnwise_quantizer.internal = True
        columnwise_empty = self.columnwise_quantizer.make_empty(
            shape,
            dtype=dtype,
            device=device,
            pin_memory=pin_memory,
        )
        self.columnwise_quantizer.internal = False

        return HybridQuantizedTensor(
            shape=shape,
            dtype=dtype,
            requires_grad=requires_grad,
            device=device,
            rowwise_storage=rowwise_empty,
            columnwise_storage=columnwise_empty,
            rowwise_quantizer=self.rowwise_quantizer,
            columnwise_quantizer=self.columnwise_quantizer,
            quantizer=self,
        )

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: QuantizedTensorStorage,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> QuantizedTensorStorage:
        """Re-quantize both sub-storages of a hybrid tensor in-place.

        Delegates to each sub-quantizer's update_quantized, which writes
        new quantized data + scales into the existing sub-storage buffers.
        """
        if not isinstance(dst, HybridQuantizedTensorStorage):
            raise ValueError(
                "HybridQuantizer can only update HybridQuantizedTensorStorage, got"
                f" {type(dst).__name__}"
            )
        if dst._rowwise_storage is not None:
            self.rowwise_quantizer.update_quantized(src, dst._rowwise_storage, noop_flag=noop_flag)
        if dst._columnwise_storage is not None:
            self.columnwise_quantizer.update_quantized(
                src, dst._columnwise_storage, noop_flag=noop_flag
            )
        return dst

    def set_usage(
        self, *, rowwise: Optional[bool] = None, columnwise: Optional[bool] = None
    ) -> None:
        super().set_usage(rowwise=rowwise, columnwise=columnwise)

    def supports_only_rowwise_all_gather(self) -> bool:
        """Whether TP activation all-gather must preserve rowwise data.

        Used by ``_linear_forward_impl`` / ``_linear_backward`` to decide
        which direction of the saved activation to keep for the backward
        input-AG: ``True`` keeps rowwise (drops columnwise),
        ``False`` keeps columnwise (drops rowwise, default for block-
        scaled formats whose columnwise is directly consumable by wgrad).

        Why hybrid needs a custom rule
        ------------------------------
        ``gather_along_first_dim`` has no hybrid-specific dispatch, so
        hybrid falls through to the generic BF16 fallback::

            inp.dequantize() → all_gather BF16 → quantizer(out)

        The direction we preserve must therefore be one the hybrid can
        dequantize. Two cases force rowwise preservation:

        1. The rowwise sub-quantizer itself declares rowwise-only AG
           (e.g. Float8 delayed / current scaling). Propagating keeps
           hybrid consistent with its component semantics.
        2. The columnwise sub-quantizer is :class:`NVFP4Quantizer`:
           ``NVFP4TensorStorage`` has no columnwise dequantize
           (``_FromNVFP4Func.forward`` raises for ``is_colwise=True``),
           so a columnwise-only NVFP4 sub-storage cannot traverse the
           BF16 fallback. Rowwise preservation routes the fallback
           through NVFP4's working rowwise dequantize instead.

        For MXFP8 / Float8Block / Float8CurrentScaling columnwise sub-
        quantizers, columnwise dequantize works and the default
        (``False``) keeps the smaller, wgrad-ready columnwise shard
        saved — which is the more efficient memory choice.

        TODO(negvet): Add native hybrid dispatch to
        ``gather_along_first_dim`` to remove the BF16 detour.

        * **Scope.** Branch at the top of ``gather_along_first_dim`` that
          detects ``HybridQuantizedTensorStorage`` / ``HybridQuantizer``,
          extracts ``rowwise_sub_storage`` and ``columnwise_sub_storage``
          with their sub-quantizers, dispatches each to its native
          ``_all_gather_{fp8,mxfp8,nvfp4,fp8_blockwise}`` path, and wraps
          the gathered sub-storages back into a ``HybridQuantizedTensor``.
          Each per-format AG routine already supports rowwise-only or
          columnwise-only input natively (including NVFP4 columnwise —
          it gathers packed FP4 bytes without dequantize).

        * **Impact.** Replaces 2×–4× BF16 bandwidth cost with native
          quantized AG. Mirrors the FSDP2 native-AG pattern we already
          ship on ``fsdp_pre_all_gather`` / ``fsdp_post_all_gather``.
          Once it lands, the ``NVFP4Quantizer`` branch in this method
          can be removed (columnwise NVFP4 AG works natively), leaving
          only the rowwise-sub-quantizer propagation.

        * **Implementation notes.** Compose async handles across the two
          per-direction AG calls into a single handle object with a
          ``.wait()`` that waits on both. Pass ``out_shape=None`` to the
          recursive calls so each format computes its own packed shape.
          Preserve FP8 current / delayed rowwise-only semantics on
          Hopper / L40 (``_all_gather_fp8`` reads ``inp._data`` which
          may be ``None`` for a columnwise-only FP8 sub-storage on
          those architectures).
        """
        if self.rowwise_quantizer.supports_only_rowwise_all_gather():
            return True
        # Local import avoids a circular dependency chain
        # (nvfp4_tensor → quantized_tensor → hybrid_tensor at module import).
        from .nvfp4_tensor import NVFP4Quantizer  # noqa: PLC0415

        if isinstance(self.columnwise_quantizer, NVFP4Quantizer):
            return True
        return False

    def _get_compatible_recipe(self):
        # HybridQuantizer is only reachable via CustomRecipe (the qfactory
        # returns HybridQuantizer per role). Checking that the autocast recipe
        # is also CustomRecipe catches the obvious mismatch (e.g. hybrid
        # quantized_model_init + built-in MXFP8BlockScaling autocast).
        # We trust that users who write a CustomRecipe know what they're doing
        # with regard to per-operand scaling mode compatibility.
        #
        # TODO(negvet): validate per-operand scaling-mode compatibility at
        # recipe-build time instead of at cuBLAS-dispatch time. Concretely:
        #   1. Walk the qfactory outputs for a given module_type (``linear``,
        #      ``grouped_linear``, ``dpa``) — call the factory for each
        #      ``QuantizerRole.tensor_type`` the module uses.
        #   2. Extract the scaling_mode of each sub-quantizer:
        #        weight_row, weight_col   (from HybridQuantizer)
        #        input_row,  input_col    (from HybridQuantizer)
        #        grad_output_row, grad_output_col   (plain quantizer OR
        #                                            HybridQuantizer)
        #   3. Assert the three GEMM pairs share a scaling_mode each:
        #        fprop  TN: weight_row  == input_row         (FormatA)
        #        dgrad  NN: weight_col  == grad_output_row   (FormatB)
        #        wgrad  NT: input_col   == grad_output_col   (FormatC)
        #      Mismatches raise ``ValueError`` naming the offending slots, e.g.
        #      "dgrad GEMM: weight columnwise format (MXFP8) does not match
        #      grad_output rowwise format (NVFP4)".
        #   4. Blocked on `semantic_quantizer_roles` / PR #2620 for the
        #      ``QuantizerRole`` dataclass — the factory signature is role-
        #      aware only on that branch.
        from transformer_engine.common.recipe import CustomRecipe  # avoid circular import

        return CustomRecipe


class HybridQuantizedTensor(HybridQuantizedTensorStorage, QuantizedTensor):
    """Quantized tensor holding data in two different formats per direction.

    The tensor presents as having a standard, higher-precision dtype, but
    internally stores rowwise data in one quantized format and columnwise
    data in another.

    Parameters
    ----------
    shape : iterable of int
        Tensor dimensions.
    dtype : torch.dtype
        Nominal tensor datatype.
    rowwise_storage : QuantizedTensorStorage
        Sub-storage for rowwise quantized data.
    columnwise_storage : QuantizedTensorStorage
        Sub-storage for columnwise quantized data.
    rowwise_quantizer : Quantizer, optional
        Quantizer used for the rowwise sub-storage.
    columnwise_quantizer : Quantizer, optional
        Quantizer used for the columnwise sub-storage.
    quantizer : HybridQuantizer, optional
        Parent hybrid quantizer.
    requires_grad : bool, default = False
        Whether to compute gradients for this tensor.

    """

    def __new__(
        cls,
        *args,
        rowwise_storage: Optional[QuantizedTensorStorage],
        columnwise_storage: Optional[QuantizedTensorStorage],
        rowwise_quantizer: Optional[Quantizer] = None,
        columnwise_quantizer: Optional[Quantizer] = None,
        quantizer: Optional[Quantizer] = None,
        **kwargs,
    ):
        instance = super().__new__(
            cls,
            *args,
            rowwise_storage=rowwise_storage,
            columnwise_storage=columnwise_storage,
            rowwise_quantizer=rowwise_quantizer,
            columnwise_quantizer=columnwise_quantizer,
            quantizer=quantizer,
            **kwargs,
        )
        return instance

    def __repr__(self, *, tensor_contents=None):
        row_type = (
            type(self._rowwise_storage).__name__ if self._rowwise_storage is not None else "None"
        )
        col_type = (
            type(self._columnwise_storage).__name__
            if self._columnwise_storage is not None
            else "None"
        )
        return (
            f"HybridQuantizedTensor(rowwise={row_type}, columnwise={col_type}, dtype={self.dtype})"
        )

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if dtype is None:
            dtype = self.dtype
        return HybridQuantizedTensorStorage.dequantize(self, dtype=dtype)

    def detach(self) -> HybridQuantizedTensor:
        """Return a new HybridQuantizedTensor with cloned sub-storage wrappers.

        Each sub-storage is re-wrapped via its own ``make_like`` so the
        new hybrid tensor has independent sub-storage objects that share
        the *underlying* buffer tensors with ``self``. This is required for
        the cpu_offload_v2 pattern at ``cpu_offload.py:378-382``::

            tensor_copy = tensor.detach()
            saved_tensors, _ = tensor_copy.prepare_for_saving()  # nulls fields

        If ``detach()`` merely shared sub-storage objects, the
        ``prepare_for_saving`` call above would null out fields on the
        original ``tensor`` too (since both hybrids would point at the same
        sub-storage Python objects), and subsequent operations — even a
        bare ``.device`` read during ``_check_if_offload`` for a follow-up
        ``push_tensor`` on the same original — would crash with
        ``<native> has no data!``.
        """
        row = None
        if self._rowwise_storage is not None:
            row_cls = type(self._rowwise_storage)
            if hasattr(row_cls, "make_like"):
                row = row_cls.make_like(self._rowwise_storage)
            else:
                # Storage-only sub-storages (HybridQuantizer.internal=True
                # path) don't have make_like; the cpu_offload_v2 path does
                # not hit this branch, but keep the behaviour safe by
                # sharing the reference as before.
                row = self._rowwise_storage
        col = None
        if self._columnwise_storage is not None:
            col_cls = type(self._columnwise_storage)
            if hasattr(col_cls, "make_like"):
                col = col_cls.make_like(self._columnwise_storage)
            else:
                col = self._columnwise_storage
        return HybridQuantizedTensor(
            shape=self.shape,
            dtype=self.dtype,
            rowwise_storage=row,
            columnwise_storage=col,
            rowwise_quantizer=self._rowwise_quantizer,
            columnwise_quantizer=self._columnwise_quantizer,
            quantizer=self._quantizer,
        )

    def get_metadata(self) -> Dict[str, Any]:
        return HybridQuantizedTensorStorage.get_metadata(self)

    # ── FSDP2 protocol ──────────────────────────────────────────────

    def fsdp_pre_all_gather(self, mesh, orig_size, contiguous_orig_stride, module, mp_policy):
        """Extract plain tensor buffers from both sub-storages for FSDP2 all-gather.

        Always send both directions. This gives a stable buffer count/shape
        across forward and backward, at the cost of gathering the unused
        direction each pass. No requantization, no BF16 fallback.

        Buffer extraction is delegated to each sub-storage's
        :meth:`QuantizedTensorStorage.fsdp_extract_buffers`, which strips any
        format-specific padding (e.g. MXFP8 block-scale alignment) before the
        gather so concatenation along dim-0 is well-defined.

        TODO(negvet): bandwidth optimization — pack both directions into a
        single flat buffer sized ``max(row_bytes, col_bytes)`` (not
        ``row_bytes + col_bytes``) to halve comm volume for asymmetric format
        pairs. Planned implementation: a new per-sub-storage
        ``fsdp_pack_into(flat_buffer, offset, meta)`` helper that layouts
        both directions back-to-back with offsets stored in the metadata
        tuple; ``fsdp_post_all_gather`` would slice the gathered flat buffer
        using those offsets.
        """
        # Quick, targeted error for sub-storages whose FSDP2 support isn't
        # implemented yet (e.g. NVFP4). Without this, users hit
        # NotImplementedError from deep inside fsdp_extract_buffers with a
        # generic message.
        for role, sub in (
            ("rowwise", self._rowwise_storage),
            ("columnwise", self._columnwise_storage),
        ):
            if sub is None:
                continue
            try:
                sub.fsdp_buffer_fields()
            except NotImplementedError as err:
                raise NotImplementedError(
                    "Hybrid FSDP2 all-gather is not supported for a "
                    f"{type(sub).__name__} {role} sub-storage: it does not "
                    "implement fsdp_buffer_fields. "
                    "See hybrid_quantization_fsdp.md section 9 (Gap 5) — "
                    "NVFP4 sub-storages need packed-FP4 dim-0 alignment, "
                    "columnwise dequantization and RHT-cache handling before "
                    "they can be gathered. Use a supported sub-quantizer "
                    "(Float8CurrentScaling, MXFP8, Float8Block) or run without "
                    "FSDP2."
                ) from err

        row_buffers: Tuple[Optional[torch.Tensor], ...] = ()
        col_buffers: Tuple[Optional[torch.Tensor], ...] = ()
        row_meta: Optional[Dict[str, Any]] = None
        col_meta: Optional[Dict[str, Any]] = None
        if self._rowwise_storage is not None:
            row_buffers, row_meta = self._rowwise_storage.fsdp_extract_buffers()
        if self._columnwise_storage is not None:
            col_buffers, col_meta = self._columnwise_storage.fsdp_extract_buffers()

        sharded_tensors = row_buffers + col_buffers

        metadata = (
            len(row_buffers),
            row_meta,
            col_meta,
            self._rowwise_storage,  # original sharded sub-storage (for make_like on iter-1)
            self._columnwise_storage,
            self._rowwise_quantizer,
            self._columnwise_quantizer,
            self._quantizer,
        )
        return sharded_tensors, metadata

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[HybridQuantizedTensor] = None,
    ):
        """Reconstruct HybridQuantizedTensor from all-gathered buffers.

        On iteration 1 (``out=None``): clone each sub-storage via
        :meth:`make_like` from the sharded original, then delegate the
        gathered-buffer writeback (and any format-specific re-padding) to
        :meth:`QuantizedTensorStorage.fsdp_assign_gathered`.
        On iteration 2+ (``out=prev``): delegate directly to the existing
        sub-storages' ``fsdp_assign_gathered``.
        """
        (
            n_row_buffers,
            row_meta,
            col_meta,
            orig_row_sub,
            orig_col_sub,
            row_quantizer,
            col_quantizer,
            hybrid_quantizer,
        ) = metadata

        row_gathered = all_gather_outputs[:n_row_buffers]
        col_gathered = all_gather_outputs[n_row_buffers:]

        def _infer_shape(gathered_buffers):
            for buf in gathered_buffers:
                if buf is not None:
                    return buf.shape
            return None

        if out is not None:
            # Iteration 2+: in-place field update on existing sub-storages
            if out._rowwise_storage is not None and row_meta is not None:
                out._rowwise_storage.fsdp_assign_gathered(row_gathered, row_meta)
            if out._columnwise_storage is not None and col_meta is not None:
                out._columnwise_storage.fsdp_assign_gathered(col_gathered, col_meta)
        else:
            # First iteration: clone the original sharded sub-storages via make_like,
            # then write gathered (full-size) buffers via each sub-storage's own
            # fsdp_assign_gathered so padding is re-applied where applicable.
            row_sub = None
            if orig_row_sub is not None and isinstance(orig_row_sub, QuantizedTensor):
                gathered_shape = _infer_shape(row_gathered)
                row_sub = type(orig_row_sub).make_like(orig_row_sub, shape=gathered_shape)
                if row_meta is not None:
                    row_sub.fsdp_assign_gathered(row_gathered, row_meta)

            col_sub = None
            if orig_col_sub is not None and isinstance(orig_col_sub, QuantizedTensor):
                gathered_shape = _infer_shape(col_gathered)
                col_sub = type(orig_col_sub).make_like(orig_col_sub, shape=gathered_shape)
                if col_meta is not None:
                    col_sub.fsdp_assign_gathered(col_gathered, col_meta)

            ref_sub = row_sub if row_sub is not None else col_sub
            out = HybridQuantizedTensor(
                shape=(
                    ref_sub.shape
                    if ref_sub is not None
                    else _infer_shape(row_gathered + col_gathered)
                ),
                dtype=param_dtype,
                rowwise_storage=row_sub,
                columnwise_storage=col_sub,
                rowwise_quantizer=row_quantizer,
                columnwise_quantizer=col_quantizer,
                quantizer=hybrid_quantizer,
            )

        return out, all_gather_outputs

    @classmethod
    def _delegate_reshape_op(cls, func, tensor, args, kwargs):
        """Delegate a shape-altering op (slice, as_strided) to each sub-storage.

        Returns a new ``HybridQuantizedTensor`` when every non-None sub-storage
        returns a ``QuantizedTensorStorage`` of the same kind (i.e. real
        op support, as Float8Tensor provides via its own
        ``__torch_dispatch__``). Returns ``None`` when any sub-storage
        dequantized to a plain ``torch.Tensor`` (i.e. the sub-storage does not
        support this op — MXFP8Tensor / Float8BlockwiseQTensor fall through
        that way for real slicing today). On ``None`` the caller should defer
        to ``super().__torch_dispatch__`` for a consistent BF16 fallback.
        """

        def _delegate(sub):
            if sub is None:
                return None
            return func(sub, *args[1:], **kwargs)

        row_out = _delegate(tensor._rowwise_storage)
        col_out = _delegate(tensor._columnwise_storage)

        row_ok = row_out is None or isinstance(row_out, QuantizedTensorStorage)
        col_ok = col_out is None or isinstance(col_out, QuantizedTensorStorage)
        if not (row_ok and col_ok):
            return None
        if row_out is None and col_out is None:
            return None

        ref = row_out if row_out is not None else col_out
        return HybridQuantizedTensor(
            shape=ref.shape,
            dtype=tensor.dtype,
            rowwise_storage=row_out,
            columnwise_storage=col_out,
            rowwise_quantizer=tensor._rowwise_quantizer,
            columnwise_quantizer=tensor._columnwise_quantizer,
            quantizer=tensor._quantizer,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func == aten.detach.default:
            return args[0].detach()

        # ── FSDP2: view ──────────────────────────────────────────────
        if func == aten.view.default:
            tensor = args[0]
            shape = args[1]
            row_view = None
            col_view = None
            if tensor._rowwise_storage is not None:
                row_view = tensor._rowwise_storage.view(shape)
            if tensor._columnwise_storage is not None:
                col_view = tensor._columnwise_storage.view(shape)
            return HybridQuantizedTensor(
                shape=shape,
                dtype=tensor.dtype,
                rowwise_storage=row_view,
                columnwise_storage=col_view,
                rowwise_quantizer=tensor._rowwise_quantizer,
                columnwise_quantizer=tensor._columnwise_quantizer,
                quantizer=tensor._quantizer,
            )

        # ── FSDP2: split ─────────────────────────────────────────────
        if func == aten.split.Tensor:
            tensor = args[0]
            split_size = args[1]
            dim = kwargs.get("dim", args[2] if len(args) > 2 else 0)

            if dim != 0:
                return super().__torch_dispatch__(func, types, args, kwargs)

            row_pieces = (
                torch.split(tensor._rowwise_storage, split_size, dim=dim)
                if tensor._rowwise_storage is not None
                else None
            )
            col_pieces = (
                torch.split(tensor._columnwise_storage, split_size, dim=dim)
                if tensor._columnwise_storage is not None
                else None
            )

            if row_pieces is None and col_pieces is None:
                return super().__torch_dispatch__(func, types, args, kwargs)

            num_pieces = len(row_pieces) if row_pieces is not None else len(col_pieces)
            return [
                HybridQuantizedTensor(
                    shape=(row_pieces[i].shape if row_pieces is not None else col_pieces[i].shape),
                    dtype=tensor.dtype,
                    rowwise_storage=row_pieces[i] if row_pieces is not None else None,
                    columnwise_storage=col_pieces[i] if col_pieces is not None else None,
                    rowwise_quantizer=tensor._rowwise_quantizer,
                    columnwise_quantizer=tensor._columnwise_quantizer,
                    quantizer=tensor._quantizer,
                )
                for i in range(num_pieces)
            ]

        # ── FSDP2: as_strided / slice ────────────────────────────────
        # Fast path for no-op (common during FSDP2 reset_sharded_param);
        # otherwise delegate per sub-storage so we inherit each sub-storage's
        # own support level. Float8Tensor implements real slicing/as_strided
        # via `_data.__torch_dispatch__`; MXFP8Tensor and Float8BlockwiseQTensor
        # handle only the no-op case and fall through to dequantize for real
        # ops (matching their vanilla FSDP2 behaviour). If any sub-storage
        # returns a plain torch.Tensor (dequantized), we can't rewrap into a
        # hybrid so we fall through to super() for a consistent BF16 fallback.
        if func == aten.as_strided.default:
            tensor = args[0]
            shape = args[1]
            strides = args[2]
            if (
                len(shape) == len(strides) == 2
                and tuple(strides) == (shape[-1], 1)
                and tuple(shape) == tuple(tensor.size())
            ):
                return HybridQuantizedTensor.make_like(tensor)
            return cls._delegate_reshape_op(
                func, tensor, args, kwargs
            ) or super().__torch_dispatch__(func, types, args, kwargs)

        if func == aten.slice.Tensor:
            tensor = args[0]
            dim = args[1]
            start = args[2]
            length = args[3]
            if start == 0 and length == tensor.size(dim):
                return HybridQuantizedTensor.make_like(tensor)
            return cls._delegate_reshape_op(
                func, tensor, args, kwargs
            ) or super().__torch_dispatch__(func, types, args, kwargs)

        # ── FSDP2: copy_ ─────────────────────────────────────────────
        # Fast path for hybrid-to-hybrid (FSDP2 fills buffer allocated via
        # new_zeros/make_empty). Other src types (e.g. a BF16 master weight
        # during checkpoint load) fall through to QuantizedTensor's base
        # dispatch which routes to ``dst.quantize_(src)``.
        if func == aten.copy_.default:
            dst, src = args[0], args[1]
            if isinstance(dst, HybridQuantizedTensor) and isinstance(src, HybridQuantizedTensor):
                if dst._rowwise_storage is not None and src._rowwise_storage is not None:
                    aten.copy_.default(dst._rowwise_storage, src._rowwise_storage)
                if dst._columnwise_storage is not None and src._columnwise_storage is not None:
                    aten.copy_.default(dst._columnwise_storage, src._columnwise_storage)
                return dst

        # ── FSDP2: new_zeros ─────────────────────────────────────────
        if func == aten.new_zeros.default:
            tensor = args[0]
            new_shape = args[1]
            if tensor._quantizer is not None:
                # FSDP2 allocates new_zeros buffers as all-gather destinations
                # that are immediately overwritten by copy_. Use make_empty
                # (uninitialized storage with the right container shape/fields).
                return tensor._quantizer.make_empty(
                    new_shape,
                    dtype=tensor.dtype,
                    device=tensor.device,
                )

        # ── FSDP2: clone ─────────────────────────────────────────────
        if func == aten.clone.default:
            tensor = args[0]
            row_clone = (
                torch.clone(tensor._rowwise_storage)
                if tensor._rowwise_storage is not None
                else None
            )
            col_clone = (
                torch.clone(tensor._columnwise_storage)
                if tensor._columnwise_storage is not None
                else None
            )
            return HybridQuantizedTensor(
                shape=tensor.shape,
                dtype=tensor.dtype,
                rowwise_storage=row_clone,
                columnwise_storage=col_clone,
                rowwise_quantizer=tensor._rowwise_quantizer,
                columnwise_quantizer=tensor._columnwise_quantizer,
                quantizer=tensor._quantizer,
            )

        return super().__torch_dispatch__(func, types, args, kwargs)
