# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for GEMM extensions"""

from typing import Iterable, Optional, Tuple, Union, List
import os
import functools
import torch
import transformer_engine_torch as tex
from ..constants import TE_DType, DType
from ..utils import get_sm_count, _empty_tensor

from ..quantized_tensor import Quantizer
from ..tensor.storage.float8_blockwise_tensor_storage import Float8BlockwiseQTensorStorage
from ..tensor.storage.grouped_tensor_storage import GroupedTensorStorage
from ..tensor.storage.nvfp4_tensor_storage import NVFP4TensorStorage
from ..tensor.utils import is_custom
from ..custom_recipes.gemm import custom_gemm
from ...debug.pytorch.debug_quantization import DebugQuantizer


__all__ = [
    "general_gemm",
    "general_grouped_gemm",
    "general_grouped_gemm_for_grouped_tensor",
]


_NUM_MAX_UB_STREAMS = 3


def get_cublas_workspace_size_bytes() -> None:
    """Return 32 MiB if using hopper, 4 MiB for all other architectures."""
    if torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 9:
        # 32 MiB for NVFP4 GEMM, plus additional 1024 B for alignment and misc scales
        return 32 * 1024 * 1024 + 1024
    return 4_194_304


@functools.lru_cache(maxsize=None)
def get_cublas_workspace(device: int, ub: bool, grouped_gemm: bool) -> torch.Tensor:
    """Returns workspace for cublas GEMM."""
    assert not (ub and grouped_gemm), "UB is unsupported for grouped GEMM."

    if ub:
        return torch.empty(
            get_cublas_workspace_size_bytes() * _NUM_MAX_UB_STREAMS,
            dtype=torch.uint8,
            device=device,
        )
    if grouped_gemm:
        _multi_stream_cublas_workspace = []
        for _ in range(tex.get_num_cublas_streams()):
            _multi_stream_cublas_workspace.append(
                torch.empty(get_cublas_workspace_size_bytes(), dtype=torch.uint8, device=device)
            )
        return _multi_stream_cublas_workspace

    return torch.empty(get_cublas_workspace_size_bytes(), dtype=torch.uint8, device=device)


def validate_gemm_scale(scale: Optional[float], required: bool) -> float:
    """Validate whether a GEMM scaling factor is consistent with its usage"""
    if required:
        return scale if scale is not None else 1.0
    if scale not in (0.0, None):
        raise ValueError("scale must be zero")
    return 0.0


def _is_nvfp4_row_scaled_tensor(tensor: torch.Tensor) -> bool:
    """Whether tensor carries row-scaled NVFP4 global amax metadata."""
    return isinstance(tensor, NVFP4TensorStorage) and tensor._row_scaled_nvfp4


def _is_nvfp4_per_token_tensor(tensor: torch.Tensor) -> bool:
    """Whether tensor was produced by the NVFP4 per-token cast.

    Per-token tensors carry per-row + per-col vector amaxes that cuBLASLt
    cannot consume directly; ``general_gemm`` must route to the fused
    per-token CUTLASS GEMM (``tex.nvfp4_cutlass_per_token_gemm``).
    """
    return isinstance(tensor, NVFP4TensorStorage) and getattr(tensor, "_per_token", False)


def _nvfp4_per_token_select(
    tensor: NVFP4TensorStorage,
    *,
    use_columnwise: bool,
    side: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Pick rowwise vs columnwise data / scale_inv / amax for one operand.

    Returns ``(data, sf, amax, sf_swizzled)``. ``sf_swizzled`` is the
    per-direction swizzle flag the kernel needs.

    Note: the per-token quantize kernel only emits *rowwise* SFs in
    swizzled layout; columnwise SFs are always linear. So whenever this
    helper picks the columnwise side, ``sf_swizzled`` is forced ``False``
    even if the tensor's flag advertises swizzle.
    """
    if use_columnwise:
        data = tensor._columnwise_data
        sf = tensor._columnwise_scale_inv
        amax = tensor._amax_columnwise
        if data is None:
            raise RuntimeError(
                f"NVFP4 per-token GEMM ({side}=columnwise) requires columnwise "
                "data on the operand. Did the cast emit columnwise=True?"
            )
        sf_swizzled = False
    else:
        data = tensor._rowwise_data
        sf = tensor._rowwise_scale_inv
        amax = tensor._amax_rowwise
        if data is None:
            raise RuntimeError(
                f"NVFP4 per-token GEMM ({side}=rowwise) requires rowwise "
                "data on the operand. Did the cast emit rowwise=True?"
            )
        sf_swizzled = bool(getattr(tensor, "_with_gemm_swizzled_scales", False))
    if amax is None:
        raise RuntimeError(
            f"NVFP4 per-token GEMM ({side}) requires the per-token amax "
            "vector. Got None — was the tensor built with per_token=True?"
        )
    return data, sf, amax, sf_swizzled


def _nvfp4_per_token_gemm(
    A: NVFP4TensorStorage,
    B: NVFP4TensorStorage,
    *,
    transa: bool,
    transb: bool,
    out: Optional[torch.Tensor],
    out_dtype: Optional[torch.dtype],
    bias: Optional[torch.Tensor],
    grad: bool,
    accumulate: bool,
    gelu: bool,
    quantization_params: Optional[Quantizer],
    ub: Optional[Union[tex.CommOverlap, tex.CommOverlapP2P]],
    extra_output: Optional[torch.Tensor],
) -> torch.Tensor:
    """Dispatch per-token NVFP4 GEMM via fused CUTLASS EVT path.

    Replaces tex.generic_gemm for per-token NVFP4 tensors. The fused EVT
    GEMM consumes per-row outer amax for kernel-A and per-col outer amax
    for kernel-B, folding the (M,) * (N,) outer-scale vectors directly
    into the bf16 epilogue. One launch, no separate post-scale pass.

    Three TE layouts are supported, matching the three GEMM call sites
    in pytorch/module/linear.py:

    +--------+---------------+-------------------------+--------------------+
    | layout | TE call site  | math                    | quant directions   |
    +========+===============+=========================+====================+
    | TN     | fwd           | D = X @ W.T (M, N)      | A.row + B.row      |
    | NN     | dgrad         | dX = dY @ W (M, K)      | A.col + B.row      |
    | NT     | wgrad         | dW = dY.T @ X (N, K)    | A.col + B.col      |
    +--------+---------------+-------------------------+--------------------+

    The kernel itself is fixed-shape — given (a, b) it computes
    D = a @ b.T with kernel_M = a.shape[0], kernel_N = b.shape[0],
    kernel_K = a.shape[-1] * 2. Each TE layout picks the right (rowwise
    vs columnwise) view of A and B so the contraction dim is the
    contiguous (inner) dim of both kernel operands. The per-token outer
    amax used by the EVT swap accordingly: rowwise-data uses
    ``_amax_rowwise``, columnwise-data uses ``_amax_columnwise``.

    Layout details (TE convention is cuBLAS column-major, so the
    operand passed as kernel-A actually corresponds to TE B; see
    ``general_gemm`` and the bench script for verification):

      * TN (fwd): kernel_a = B.rowwise (M, K); kernel_b = A.rowwise
        (N, K). Contraction = K.
      * NN (dgrad, output dX = (M, K)): kernel_a = B.rowwise
        (M, N — dY rowwise); kernel_b = A.columnwise (K, N — W
        columnwise, raw bytes are W.T rowwise). Contraction = N.
      * NT (wgrad, output dW = (N, K)): kernel_a = B.columnwise
        (N, M — dY columnwise); kernel_b = A.columnwise (K, M — X
        columnwise). Contraction = M.

    All three call the same CUTLASS kernel; the only thing that changes
    is which (data, sf, amax) tuple of each tensor we hand in.
    """
    # Hard-fail unsupported config rather than silently degrade.
    if accumulate:
        # NOTE: te.Linear sets accumulate=True for wgrad when
        # fuse_wgrad_accumulation=True. Per-token kernel doesn't yet
        # support D += A @ B (would need an out_init epilogue node).
        raise NotImplementedError(
            "NVFP4 per-token GEMM does not yet support accumulate=True. "
            "Disable fuse_wgrad_accumulation when using NVFP4 per-token "
            "(or fall back to prod NVFP4)."
        )
    if gelu:
        raise NotImplementedError("NVFP4 per-token GEMM does not yet support fused gelu.")
    if ub is not None:
        raise NotImplementedError("NVFP4 per-token GEMM does not yet support comm-overlap.")
    if extra_output is not None:
        raise NotImplementedError("NVFP4 per-token GEMM does not yet support extra_output.")
    if quantization_params is not None:
        # The fused EVT only emits bf16; output quantization would have
        # to wrap the result in a separate post-cast (TODO).
        raise NotImplementedError(
            "NVFP4 per-token GEMM does not yet support output quantization. "
            "Set the relevant grad_input_quantizer / output_quantizer to None."
        )
    if grad and bias is not None:
        # In bwd path, `bias` is the cuBLAS-style fused-dbias accumulator,
        # not a forward-style additive offset. The per-token kernel
        # doesn't compute dbias internally, so silently broadcast-adding
        # `bias` would corrupt dW / dX. te.Linear's NVFP4 wgrad call
        # already passes bias=None when fp8=True, so this branch is
        # defensive against future regressions.
        raise NotImplementedError(
            "NVFP4 per-token GEMM does not support fused dbias in bwd. "
            "Compute grad_bias separately (e.g. dY.sum(dim=0)) and pass "
            "bias=None to general_gemm in the bwd path."
        )

    # Resolve layout -> per-operand rowwise/columnwise selection.
    #
    # TE's layout strings ("TN" / "NN" / "NT") are written in cuBLAS
    # column-major terms. Because torch row-major data is reinterpreted
    # by cuBLAS as col-major (a torch (P, Q) row-major tensor is a (Q,P)
    # col-major matrix), the rule that makes the kernel operand's
    # contraction dim contiguous is:
    #
    #   transX = True  -> use X.rowwise    (X is consumed "as-is")
    #   transX = False -> use X.columnwise (X is implicitly transposed)
    #
    # which gives this 3-row dispatch table (TT is not exercised by any
    # current TE module and we reject it cleanly):
    #
    #   layout | TE site | (transa, transb) | A view  | B view  |
    #   ------ + ------- + ---------------- + ------- + ------- +
    #   TN     | fwd     | (T, N)           | rowwise | rowwise |
    #   NN     | dgrad   | (N, N)           | colwise | rowwise |
    #   NT     | wgrad   | (N, T)           | colwise | colwise |
    #
    # On top of this, the kernel takes a fixed pair (kernel_a, kernel_b)
    # and computes D = kernel_a @ kernel_b^T. The TE-vs-kernel swap
    # (kernel_a = TE B, kernel_b = TE A) is the same swap used by the
    # row-scaled NVFP4 GEMM helper above.
    if (transa, transb) == (True, False):
        layout_label = "TN"
        a_use_col, b_use_col = False, False
    elif (transa, transb) == (False, False):
        layout_label = "NN"
        a_use_col, b_use_col = True, False
    elif (transa, transb) == (False, True):
        layout_label = "NT"
        a_use_col, b_use_col = True, True
    else:
        raise NotImplementedError(
            "NVFP4 per-token GEMM does not support TT layout "
            f"(got transa={transa}, transb={transb})."
        )

    # Pick the (data, sf, amax) tuple from each tensor and the swizzle
    # flag. _nvfp4_per_token_select hard-fails if the requested
    # direction is missing or amax is None.
    ka_data, ka_sf, ka_amax, a_sf_swizzled = _nvfp4_per_token_select(
        B, use_columnwise=b_use_col, side="kernel_a (=TE B)"
    )
    kb_data, kb_sf, kb_amax, b_sf_swizzled = _nvfp4_per_token_select(
        A, use_columnwise=a_use_col, side="kernel_b (=TE A)"
    )

    # Rowwise data can carry the activation's original N-D leading dims
    # (e.g. byte-shape [batch, seq, K/2] for a 3D transformer activation,
    # see NVFP4Tensor rowwise byte_shape = shape[:-1] + [shape[-1]//2]),
    # while the per-token amax / SF were computed over the FLATTENED row
    # count (batch*seq). Columnwise data is always already 2D
    # ([feature, tokens/2]). Flatten any rowwise operand to 2D so the
    # "shape[0] == kernel row-count" invariant below holds for N-D inputs
    # too -- this is a contiguous reshape (no copy) and a no-op for the 2D
    # case. ka = select(B, use_columnwise=b_use_col); kb = select(A,
    # use_columnwise=a_use_col), so an operand is rowwise iff its
    # *_use_col flag is False.
    #
    # We also capture kernel_a's original leading dims so the output is
    # returned with the activation's N-D shape (e.g. [batch, seq, N]). The
    # fwd GEMM output is NOT reshaped by te.Linear (it is returned as-is and
    # fed straight into residual/bias adds), so prod's generic_gemm returns
    # N-D and we must match. The kernel writes a flat (M, N) buffer, so we
    # restore the leading dims afterward. Only the rowwise operand can be
    # N-D; if kernel_a is columnwise (wgrad), the output is a 2D (N, K) grad
    # and there is nothing to restore.
    out_lead = None
    if not b_use_col:
        out_lead = tuple(ka_data.shape[:-1])
        ka_data = ka_data.reshape(-1, ka_data.shape[-1])
    if not a_use_col:
        kb_data = kb_data.reshape(-1, kb_data.shape[-1])

    # Recover (m_kernel, n_kernel, k_kernel) from data shapes. The kernel
    # computes D = ka_data @ kb_data^T, so m_kernel = ka_data.shape[0],
    # n_kernel = kb_data.shape[0], k_kernel (contraction dim) =
    # ka_data.shape[-1] * 2 (rowwise_data is byte-packed, last dim = K/2).
    M = ka_data.shape[0]
    K = ka_data.shape[-1] * 2
    N = kb_data.shape[0]
    if kb_data.shape[-1] * 2 != K:
        raise RuntimeError(
            f"NVFP4 per-token GEMM K mismatch ({layout_label}): "
            f"kernel_a K={K}, kernel_b K={kb_data.shape[-1] * 2}. "
            "Likely operand misalignment between the two NVFP4 tensors."
        )
    if ka_amax.shape[0] != M or kb_amax.shape[0] != N:
        raise RuntimeError(
            f"NVFP4 per-token GEMM amax-vector mismatch ({layout_label}): "
            f"ka_amax={tuple(ka_amax.shape)} (expected ({M},)), "
            f"kb_amax={tuple(kb_amax.shape)} (expected ({N},))."
        )

    # Allocate output if needed.
    out_dtype = out_dtype or torch.bfloat16
    if out_dtype != torch.bfloat16:
        raise NotImplementedError(f"NVFP4 per-token GEMM only emits bf16 (requested {out_dtype}).")
    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=ka_data.device)
    elif out.numel() != M * N:
        raise RuntimeError(
            f"NVFP4 per-token GEMM output shape mismatch ({layout_label}): "
            f"got out.shape={tuple(out.shape)} (numel={out.numel()}), "
            f"expected ({M}, {N})."
        )
    else:
        # A caller may hand in an N-D pre-allocated output (matching the
        # activation's leading dims). The kernel writes a flat (M, N) buffer;
        # view it 2D for the call. The view shares storage so writes land in
        # the caller's tensor (requires contiguous out, which TE guarantees).
        out = out.view(M, N)

    # The fused EVT GEMM does its own SF swizzle internally if the SFs
    # are not pre-swizzled. One launch total either way.
    tex.nvfp4_cutlass_per_token_gemm(
        ka_data,
        kb_data,
        ka_sf,
        kb_sf,
        ka_amax,
        kb_amax,
        out,
        M,
        N,
        K,
        a_sf_swizzled,
        b_sf_swizzled,
    )

    if bias is not None:
        # Forward path: bias is (N,) and we broadcast-add into each row
        # post-GEMM. Negligible cost vs the GEMM. (grad=True with bias
        # is rejected upstream; this branch only fires for fwd.)
        out.add_(bias.to(dtype=out.dtype))

    # Restore the activation's N-D leading dims (out is a flat (M, N) buffer
    # for the kernel). A no-op when kernel_a was already 2D (out_lead is a
    # 1-tuple) or columnwise (wgrad, out_lead is None). Shares storage, so
    # a caller-provided N-D `out` is filled in place either way.
    if out_lead is not None and len(out_lead) != 1:
        out = out.view(*out_lead, N)

    return out


def _nvfp4_per_token_grouped_gemm(
    A: List[NVFP4TensorStorage],
    B: List[NVFP4TensorStorage],
    out: List[torch.Tensor],
    *,
    transa: bool,
    transb: bool,
    m_splits: List[int],
    single_output: bool,
) -> List[torch.Tensor]:
    """Native grouped (MoE) per-token NVFP4 GEMM (one ptr-array CUTLASS launch).

    Per-group analogue of ``_nvfp4_per_token_gemm``: applies the same
    layout -> rowwise/columnwise operand selection and the kernel_a = TE B,
    kernel_b = TE A swap to every group, then issues a single grouped launch
    over all non-empty experts. Empty experts (``m_splits[g] == 0``) are
    zeroed and excluded from the launch (the kernel requires M, N, K > 0).

    Only the plain D = bf16(alpha_a * alpha_b * (A @ B^T)) path is handled
    here; callers must route accumulate / bias / gelu / output-quantization
    to the per-expert fallback (those are rejected by the dense kernel too).
    """
    # Same (transa, transb) -> rowwise/columnwise table as _nvfp4_per_token_gemm.
    if (transa, transb) == (True, False):
        layout_label = "TN"
        a_use_col, b_use_col = False, False
    elif (transa, transb) == (False, False):
        layout_label = "NN"
        a_use_col, b_use_col = True, False
    elif (transa, transb) == (False, True):
        layout_label = "NT"
        a_use_col, b_use_col = True, True
    else:
        raise NotImplementedError(
            "NVFP4 per-token grouped GEMM does not support TT layout "
            f"(got transa={transa}, transb={transb})."
        )

    num_gemms = len(A)

    # Resolve the per-group output views (single contiguous buffer sliced
    # along the token/M dim, or one tensor per group).
    if single_output:
        assert m_splits is not None, "single_output grouped GEMM requires m_splits."
        out_init = out[0]
        out_views = []
        start = 0
        for size in m_splits:
            out_views.append(out_init[start : start + size])
            start += size
    else:
        out_init = None
        out_views = out

    # Collect per-group kernel operands for the non-empty experts.
    a_data_l, b_data_l, a_sf_l, b_sf_l, alpha_a_l, alpha_b_l, d_l = ([] for _ in range(7))
    a_sf_swz_seen: Optional[bool] = None
    b_sf_swz_seen: Optional[bool] = None

    for g in range(num_gemms):
        # Empty expert: weight grad is the zero matrix; fwd/dgrad slice is
        # empty. The CUTLASS kernel asserts M, N, K > 0, so handle here.
        if m_splits is not None and m_splits[g] == 0:
            if out_views[g].numel() != 0:
                out_views[g].zero_()
            continue

        ka_data, ka_sf, ka_amax, a_sf_swz = _nvfp4_per_token_select(
            B[g], use_columnwise=b_use_col, side="kernel_a (=TE B)"
        )
        kb_data, kb_sf, kb_amax, b_sf_swz = _nvfp4_per_token_select(
            A[g], use_columnwise=a_use_col, side="kernel_b (=TE A)"
        )

        # Flatten any rowwise (possibly N-D) operand to 2D (no-op / no-copy
        # for the already-2D grouped operands).
        if not b_use_col:
            ka_data = ka_data.reshape(-1, ka_data.shape[-1])
        if not a_use_col:
            kb_data = kb_data.reshape(-1, kb_data.shape[-1])

        M = ka_data.shape[0]
        K = ka_data.shape[-1] * 2
        N = kb_data.shape[0]
        if kb_data.shape[-1] * 2 != K:
            raise RuntimeError(
                f"NVFP4 per-token grouped GEMM K mismatch ({layout_label}, group {g}): "
                f"kernel_a K={K}, kernel_b K={kb_data.shape[-1] * 2}."
            )
        if ka_amax.shape[0] != M or kb_amax.shape[0] != N:
            raise RuntimeError(
                f"NVFP4 per-token grouped GEMM amax mismatch ({layout_label}, group {g}): "
                f"ka_amax={tuple(ka_amax.shape)} (exp ({M},)), "
                f"kb_amax={tuple(kb_amax.shape)} (exp ({N},))."
            )

        # The SF swizzle flag must be uniform across groups (all share one
        # quantizer state); the C++ wrapper takes a single flag per side.
        if a_sf_swz_seen is None:
            a_sf_swz_seen, b_sf_swz_seen = a_sf_swz, b_sf_swz
        elif (a_sf_swz, b_sf_swz) != (a_sf_swz_seen, b_sf_swz_seen):
            raise RuntimeError(
                "NVFP4 per-token grouped GEMM requires a uniform SF-swizzle "
                "state across all groups."
            )

        a_data_l.append(ka_data)
        b_data_l.append(kb_data)
        a_sf_l.append(ka_sf)
        b_sf_l.append(kb_sf)
        alpha_a_l.append(ka_amax)
        alpha_b_l.append(kb_amax)
        d_l.append(out_views[g].view(M, N))

    if a_data_l:
        tex.nvfp4_cutlass_grouped_per_token_gemm(
            a_data_l,
            b_data_l,
            a_sf_l,
            b_sf_l,
            alpha_a_l,
            alpha_b_l,
            d_l,
            bool(a_sf_swz_seen),
            bool(b_sf_swz_seen),
        )

    if single_output:
        return out_init
    return out


def _nvfp4_row_scaled_gemm_inputs(
    A: NVFP4TensorStorage,
    B: NVFP4TensorStorage,
    *,
    transa: bool,
) -> Tuple[NVFP4TensorStorage, NVFP4TensorStorage, torch.Tensor]:
    """Return GEMM aliases and FP32 output scales for row-scaled NVFP4."""
    A_metadata = A.get_metadata()
    weight_amax = A._amax_rowwise if transa else A._amax_columnwise
    assert weight_amax is not None and weight_amax.numel() == 1
    A_metadata["amax_rowwise" if transa else "amax_columnwise"] = weight_amax.new_ones(1)
    A_metadata["row_scaled_nvfp4"] = False

    B_metadata = B.get_metadata()
    rhs_rowwise_amax = B._amax_rowwise
    assert rhs_rowwise_amax is not None
    B_metadata["amax_rowwise"] = rhs_rowwise_amax.new_ones(1)
    B_metadata["row_scaled_nvfp4"] = False

    assert rhs_rowwise_amax.dtype == torch.float32 and weight_amax.dtype == torch.float32
    return (
        NVFP4TensorStorage(**A_metadata),
        NVFP4TensorStorage(**B_metadata),
        (rhs_rowwise_amax * weight_amax).view(-1, 1),
    )


def general_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    quantization_params: Optional[Quantizer] = None,
    gelu: bool = False,
    gelu_in: torch.Tensor = None,
    alpha: float = 1.0,
    beta: Optional[float] = None,
    accumulate: bool = False,
    layout: str = "TN",
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    use_split_accumulator: bool = False,
    grad: bool = False,
    ub: Union[tex.CommOverlap, tex.CommOverlapP2P] = None,
    ub_type: tex.CommOverlapType = None,
    extra_output: Optional[torch.Tensor] = None,
    bulk_overlap: bool = False,
) -> Iterable[Optional[torch.Tensor]]:
    """GEMM supporting fp8 inputs."""

    assert layout in ("TN", "NN", "NT"), f"GEMM layout {layout} not supported."
    transa = layout[0] == "T"
    transb = layout[1] == "T"

    alpha = validate_gemm_scale(alpha, True)
    beta = validate_gemm_scale(beta, accumulate)
    workspace = get_cublas_workspace(A.device.index, ub is not None, False)

    if ub_type is not None:
        assert ub is not None, (
            f"{'AG+GEMM' if ub_type == tex.CommOverlapType.AG else 'GEMM+RS'} overlap requires"
            + "a valid `ub` communicator object."
        )

    if ub is not None:
        assert ub_type is not None, "Comm+GEMM overlap requires a valid `comm_type` argument."
        if ub_type == tex.CommOverlapType.RS:
            if not (bulk_overlap and not ub.is_fp8_ubuf()):
                assert extra_output is not None, "GEMM+RS overlap requires extra output tensor."

    if out is not None:
        if not out.is_contiguous():
            raise ValueError("Output tensor is not contiguous.")

    # If A or B are custom tensors -> dispatch to quantizers's qgemm implementation
    if is_custom(A) or is_custom(B):
        return custom_gemm(
            A,
            B,
            workspace,
            out_dtype,
            quantization_params,
            gelu,
            gelu_in,
            accumulate,
            layout,
            out,
            bias,
            use_split_accumulator,
            grad,
        )

    debug_quantizer = None
    if isinstance(quantization_params, DebugQuantizer):
        debug_quantizer = quantization_params
        quantization_params = quantization_params.parent_quantizer
        A = A.get_tensor(not transa)
        B = B.get_tensor(transb)

    # Use bfloat16 as default bias_dtype
    bias_dtype = TE_DType[torch.bfloat16 if bias is None else bias.dtype]

    if isinstance(A, Float8BlockwiseQTensorStorage) or isinstance(B, Float8BlockwiseQTensorStorage):
        # FP8 block-scaling requires split accumulator
        use_split_accumulator = True

    args = (
        A,
        transa,  # transa
        B,
        transb,  # transb
        out,
        quantization_params,
        TE_DType[out_dtype] if out_dtype is not None else None,
        bias,
        bias_dtype,
        gelu,
        gelu_in,
        grad,  # grad
        workspace,
        workspace.shape[0],
        accumulate,
        use_split_accumulator,
    )
    kwargs = {
        "comm_overlap": ub,
        "comm_type": ub_type,
        "extra_output": extra_output,
        "bulk_overlap": bulk_overlap,
        "alpha": alpha,
        "beta": beta,
    }

    # Per-token NVFP4 dispatches to fused EVT GEMM that consumes per-row
    # (M,) and per-col (N,) outer-amax vectors directly. cuBLASLt cannot,
    # so this MUST short-circuit before the row-scaled-or-generic fork.
    if _is_nvfp4_per_token_tensor(A) or _is_nvfp4_per_token_tensor(B):
        if not (_is_nvfp4_per_token_tensor(A) and _is_nvfp4_per_token_tensor(B)):
            raise NotImplementedError(
                "NVFP4 per-token GEMM requires both A and B to be per-token tensors. "
                "Mixing per-token + prod NVFP4 in one GEMM is not supported."
            )
        out = _nvfp4_per_token_gemm(
            A,
            B,
            transa=transa,
            transb=transb,
            out=out,
            out_dtype=out_dtype,
            bias=bias,
            grad=grad,
            accumulate=accumulate,
            gelu=gelu,
            quantization_params=quantization_params,
            ub=ub,
            extra_output=extra_output,
        )
        bias_grad = None
        gelu_input = None
        extra_output = None
    elif not _is_nvfp4_row_scaled_tensor(A) and not _is_nvfp4_row_scaled_tensor(B):
        out, bias_grad, gelu_input, extra_output = tex.generic_gemm(*args, **kwargs)
    else:
        if _is_nvfp4_row_scaled_tensor(A):
            raise NotImplementedError("Row-scaled NVFP4 GEMM does not support row-scaled A.")
        assert layout[1] == "N", "Row-scaled NVFP4 GEMM currently supports N-layout B only."
        if grad:
            raise RuntimeError(
                "Row-scaled NVFP4 GEMM currently supports fprop only. "
                "Backward NVFP4 gradient quantizers should use scalar global amax."
            )
        assert not gelu, "Row-scaled NVFP4 GEMM currently does not support fused GELU."
        assert not accumulate, "Row-scaled NVFP4 GEMM currently does not support accumulation."
        assert (
            quantization_params is None
        ), "Row-scaled NVFP4 GEMM currently does not support output quantization."
        assert ub is None, "Row-scaled NVFP4 GEMM currently does not support CommOverlap."
        assert (
            extra_output is None
        ), "Row-scaled NVFP4 GEMM currently does not support extra output."
        assert not bulk_overlap, "Row-scaled NVFP4 GEMM currently does not support bulk overlap."
        assert out is None or (
            isinstance(out, torch.Tensor) and not is_custom(out)
        ), "Row-scaled NVFP4 GEMM currently supports only plain torch.Tensor outputs."
        assert isinstance(
            A, NVFP4TensorStorage
        ), "Row-scaled NVFP4 GEMM currently requires NVFP4 A."
        # cuBLAS folds NVFP4 global amax values into GEMM alpha. Keep the row-scaled
        # recipe's global scales out of alpha and apply them in FP32 below.
        gemm_A, gemm_B, rowwise_global_scales = _nvfp4_row_scaled_gemm_inputs(A, B, transa=transa)

        requested_out, requested_out_dtype = out, out_dtype
        fp32_out = (
            torch.empty_like(requested_out, dtype=torch.float32)
            if requested_out is not None
            else None
        )
        gemm_args = list(args)
        gemm_args[0] = gemm_A  # A
        gemm_args[2] = gemm_B  # B
        gemm_args[4] = fp32_out  # out
        gemm_args[5] = None  # quantization_params
        gemm_args[6] = TE_DType[torch.float32]  # out_dtype
        gemm_args[7] = None  # bias
        out, bias_grad, gelu_input, extra_output = tex.generic_gemm(*gemm_args, **kwargs)
        out_2d = out.reshape(-1, out.shape[-1])

        assert rowwise_global_scales.dtype == torch.float32 and out.dtype == torch.float32
        assert rowwise_global_scales.numel() == out_2d.shape[0]

        out_2d.mul_(rowwise_global_scales)
        if bias is not None:
            out_2d.add_(bias.to(dtype=torch.float32))

        if requested_out is not None:
            requested_out.copy_(out.to(dtype=requested_out.dtype))
            out = requested_out
        elif requested_out_dtype is not None and requested_out_dtype != torch.float32:
            out = out.to(dtype=requested_out_dtype)

    if debug_quantizer is not None:
        out = debug_quantizer.process_gemm_output(out)

    return out, bias_grad, gelu_input, extra_output


def general_grouped_gemm(
    A: List[torch.Tensor],
    B: List[torch.Tensor],
    out: List[torch.Tensor],
    quantization_params: List[Optional[Quantizer]],
    out_dtype: torch.dtype,
    layout: str = "TN",
    m_splits: Optional[List[int]] = None,
    gelu: bool = False,
    grad=False,
    accumulate: bool = False,
    bias: Optional[List[torch.Tensor]] = None,
    use_bias: bool = False,
    use_split_accumulator: bool = False,
    D_dtype: Optional[DType] = None,
    single_output=False,
) -> Tuple[List[torch.Tensor], ...]:
    """
    TN layout Grouped GEMM with fp8 inputs.
    """
    num_gemms = len(A)

    transa = layout[0] == "T"
    transb = layout[1] == "T"

    empty_tensor = _empty_tensor()
    empty_tensors = [empty_tensor] * num_gemms

    # Use bfloat16 as default bias_dtype
    gelu_input = empty_tensors
    out_dtype = TE_DType[out[0].dtype] if D_dtype is None else D_dtype

    sm_count = get_sm_count()
    workspaces = get_cublas_workspace(A[0].device.index, False, True)

    if grad and use_bias:
        grad_bias = [
            torch.empty(B[i].size(1), dtype=out[0].dtype, device="cuda") for i in range(num_gemms)
        ]
    else:
        grad_bias = empty_tensors
    bias = bias if use_bias else empty_tensors
    if use_bias:
        bias_dtype = TE_DType[grad_bias[0].dtype] if grad else TE_DType[bias[0].dtype]
    else:
        bias_dtype = TE_DType[torch.bfloat16]

    # Per-token NVFP4 grouped GEMM. The plain D = bf16(alpha_a * alpha_b *
    # (A @ B^T)) path is served by a native single-launch CUTLASS grouped
    # per-token kernel (_nvfp4_per_token_grouped_gemm). Cases the dense
    # per-token kernel can't fuse (accumulate / bias / gelu / output
    # quantization) fall through to a Python loop over per-shape `general_gemm`
    # calls, which routes each shape to the same fused EVT GEMM (identical
    # numerics, at the cost of num_gemms launch overhead).
    if any(_is_nvfp4_per_token_tensor(t) for t in A) or any(
        _is_nvfp4_per_token_tensor(t) for t in B
    ):
        for tensor in A:
            if not _is_nvfp4_per_token_tensor(tensor):
                raise NotImplementedError(
                    "NVFP4 per-token grouped GEMM requires all A operands to be per-token."
                )
        for tensor in B:
            if not _is_nvfp4_per_token_tensor(tensor):
                raise NotImplementedError(
                    "NVFP4 per-token grouped GEMM requires all B operands to be per-token."
                )

        # Native single-launch CUTLASS grouped per-token kernel. Used for the
        # plain D = bf16(alpha_a * alpha_b * (A @ B^T)) path; accumulate / bias
        # / gelu / output-quantization are rejected by the dense per-token
        # kernel too, so those fall through to the per-expert loop below.
        _native_ok = (
            not accumulate
            and not gelu
            and not use_bias
            and D_dtype is None
            and all(qp is None for qp in quantization_params)
        )
        if _native_ok and not int(os.getenv("NVTE_NVFP4_PER_TOKEN_GROUPED_FALLBACK", "0")):
            transa = layout[0] == "T"
            transb = layout[1] == "T"
            out = _nvfp4_per_token_grouped_gemm(
                A,
                B,
                out,
                transa=transa,
                transb=transb,
                m_splits=m_splits,
                single_output=single_output,
            )
            return out, grad_bias, gelu_input

        if single_output:
            assert (
                m_splits is not None
            ), "NVFP4 per-token grouped GEMM requires m_splits with single output."
            out_init = out[0]
            start_idx = 0
            out_views = []
            for i in range(num_gemms):
                size = m_splits[i]
                out_views.append(out_init[start_idx : start_idx + size])
                start_idx += size
        else:
            out_views = out
        for i in range(num_gemms):
            if out_views[i].numel() == 0:
                continue
            # An expert that received 0 tokens this step has m_splits[i] == 0.
            # The token count is the GEMM contraction dim for wgrad (layout NT)
            # whose output is the weight grad [out, in] -> nonzero numel, so the
            # numel guard above misses it and the per-token cutlass kernel would
            # assert M > 0. Skip the launch; the empty-expert wgrad is the zero
            # matrix (or a no-op add when accumulating into main_grad).
            if m_splits is not None and m_splits[i] == 0:
                if not accumulate:
                    out_views[i].zero_()
                continue
            general_gemm(
                A[i],
                B[i],
                quantization_params=quantization_params[i],
                out_dtype=out_views[i].dtype,
                out=out_views[i],
                gelu=gelu,
                accumulate=accumulate,
                layout=layout,
                bias=bias[i] if use_bias else None,
                use_split_accumulator=use_split_accumulator,
                grad=grad,
            )
        if single_output:
            out = out_init
        return out, grad_bias, gelu_input

    if any(_is_nvfp4_row_scaled_tensor(tensor) for tensor in A):
        raise NotImplementedError("Row-scaled NVFP4 grouped GEMM does not support row-scaled A.")
    if any(_is_nvfp4_row_scaled_tensor(tensor) for tensor in B):
        assert D_dtype is None, "Row-scaled NVFP4 grouped GEMM currently does not support D_dtype."
        if single_output:
            assert (
                m_splits is not None
            ), "Row-scaled NVFP4 grouped GEMM requires m_splits with single output."
        out_init = out[0] if single_output else None
        if single_output:
            start_idx = 0
            out_views = []
            for i in range(num_gemms):
                size = m_splits[i]
                out_views.append(out_init[start_idx : start_idx + size])
                start_idx += size
        else:
            out_views = out
        for i in range(num_gemms):
            if out_views[i].numel() == 0:
                continue
            general_gemm(
                A[i],
                B[i],
                quantization_params=quantization_params[i],
                out_dtype=out_views[i].dtype,
                out=out_views[i],
                gelu=gelu,
                accumulate=accumulate,
                layout=layout,
                bias=bias[i] if use_bias else None,
                use_split_accumulator=use_split_accumulator,
                grad=grad,
            )
        if single_output:
            out = out_init
        return out, grad_bias, gelu_input

    if isinstance(quantization_params[0], DebugQuantizer):
        assert not gelu, "GELU not supported in debug mode"
        if single_output:
            out_init = out[0]
            start_idx = 0
            out = [None] * num_gemms
            for i in range(num_gemms):
                size = m_splits[i]
                out[i] = out_init[start_idx : start_idx + size]
                start_idx += size
        for i in range(num_gemms):
            _, bias_or_grad, _, _ = general_gemm(
                A[i],
                B[i],
                quantization_params=quantization_params[i],
                out_dtype=out[0].dtype,
                layout=layout,
                accumulate=accumulate,
                out=out[i],
                bias=bias[i] if use_bias else None,
                use_split_accumulator=use_split_accumulator,
                grad=grad,
            )
            if grad and use_bias:
                grad_bias[i] = bias_or_grad
        if single_output:
            out = out_init

        return out, grad_bias if grad else bias, None

    if gelu:
        gelu_input = [
            torch.empty_like(o, dtype=bias_dtype, memory_format=torch.contiguous_format)
            for o in out
        ]  # this should differ with respect to single output

    bias = tex.te_general_grouped_gemm(
        A,
        transa,
        B,
        transb,
        out,
        out_dtype,
        m_splits,
        grad_bias if grad else bias,
        bias_dtype,
        single_output,
        gelu_input,  # this is pre_gelu_out
        grad,  # grad
        workspaces,
        workspaces[0].shape[0],
        accumulate,
        use_split_accumulator,
        sm_count - int(os.getenv("NVTE_EXT_MARGIN_SM", str(sm_count))),
    )

    return out, bias, gelu_input


@functools.lru_cache(maxsize=None)
def get_grouped_gemm_setup_workspace_size(num_tensors: int) -> int:
    """Return workspace size for grouped GEMM pointer setup."""
    return tex.get_grouped_gemm_setup_workspace_size(num_tensors)


@functools.lru_cache(maxsize=None)
def _get_fp32_ones_tensor(num_tensors: int, device: torch.device) -> torch.Tensor:
    """Cached ones tensor."""
    return torch.ones(num_tensors, dtype=torch.float32, device=device)


@functools.lru_cache(maxsize=None)
def _get_fp32_zeros_tensor(num_tensors: int, device: torch.device) -> torch.Tensor:
    """Cached zeros tensor."""
    return torch.zeros(num_tensors, dtype=torch.float32, device=device)


def general_grouped_gemm_for_grouped_tensor(
    A,
    B,
    out,
    *,
    layout: str = "TN",
    accumulate: bool = False,
    use_split_accumulator: bool = False,
    bias=None,
    bias_scale: Optional[torch.Tensor] = None,
    grad: bool = False,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Grouped GEMM using GroupedTensor inputs.

    This uses nvte_grouped_gemm and supports different per-matrix shapes.

    The caller must ensure that GroupedTensor metadata is already compatible with the
    underlying GEMM implementation (e.g., aligned offsets and output metadata layout).
    """
    assert layout in ("TN", "NN", "NT"), f"GEMM layout {layout} not supported."
    if grad:
        raise NotImplementedError("grad is not supported for grouped_tensor GEMM yet.")
    transa = layout[0] == "T"
    transb = layout[1] == "T"
    is_discrete_out = isinstance(out, list)
    is_discrete_in = isinstance(A, list)
    if is_discrete_in and is_discrete_out:
        raise ValueError("Both A and out are discrete. This is not supported yet.")

    if isinstance(A, GroupedTensorStorage) and A.row_scaled_nvfp4:
        raise NotImplementedError("Row-scaled NVFP4 GroupedTensor GEMM is not supported yet.")
    if isinstance(B, GroupedTensorStorage) and B.row_scaled_nvfp4:
        raise NotImplementedError("Row-scaled NVFP4 GroupedTensor GEMM is not supported yet.")
    if isinstance(out, GroupedTensorStorage) and out.row_scaled_nvfp4:
        raise NotImplementedError("Row-scaled NVFP4 GroupedTensor GEMM is not supported yet.")

    if is_discrete_out:
        # wgrad case.
        grouped_gemm_impl = tex.te_general_grouped_gemm_for_discrete_out
    elif is_discrete_in:
        # Use-case: forward pass with list of weights.
        grouped_gemm_impl = tex.te_general_grouped_gemm_for_discrete_in
    else:
        # Use-case: Single Grouped Parameter for Weight/ Weight Grads.
        grouped_gemm_impl = tex.te_general_grouped_gemm_for_grouped_tensor

    if is_discrete_out and bias is not None:
        raise ValueError(
            "Bias is not supported when out is a list (discrete_out mode) yet. "
            "Apply bias manually after the GEMM."
        )

    if bias_scale is not None and bias is None:
        raise ValueError("bias_scale requires bias to be provided.")

    num_tensors = B.num_tensors
    rowwise = B.rowwise_data
    device = rowwise.device if rowwise is not None else B.columnwise_data.device

    # Hopper (SM90) uses a single shared alpha/beta scalar;
    # Blackwell+ (SM100) supports per-group alpha/beta arrays.
    per_group = torch.cuda.get_device_capability() >= (10, 0)
    num_alphabeta = num_tensors if per_group else 1

    if alpha is None:
        alpha = _get_fp32_ones_tensor(num_alphabeta, device)
    if beta is None:
        if accumulate:
            beta = _get_fp32_ones_tensor(num_alphabeta, device)
        else:
            beta = _get_fp32_zeros_tensor(num_alphabeta, device)

    if not alpha.is_cuda or not beta.is_cuda:
        raise ValueError("alpha and beta must be CUDA tensors.")

    workspace_setup = torch.empty(
        get_grouped_gemm_setup_workspace_size(num_tensors),
        dtype=torch.uint8,
        device=device,
    )
    workspace_cublas = torch.empty(
        get_cublas_workspace_size_bytes(),
        dtype=torch.uint8,
        device=device,
    )

    sm_count = get_sm_count()
    sm_count = sm_count - int(os.getenv("NVTE_EXT_MARGIN_SM", str(sm_count)))

    return grouped_gemm_impl(
        A,
        transa,
        B,
        transb,
        out,
        bias,
        bias_scale,
        alpha,
        beta,
        workspace_setup,
        workspace_cublas,
        use_split_accumulator,
        sm_count,
    )
