# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Opt-in PyTorch composition of the external MegaUP projection kernels.

This prototype deliberately keeps kernels in the optional external package. It
does not introduce a TE kernel backend or change parameter-gradient reduction.
"""

import os

import torch
import torch.distributed as dist

from transformer_engine.pytorch.quantization import FP8GlobalStateManager
from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage
from transformer_engine.pytorch.cpu_offload import is_cpu_offload_enabled
from transformer_engine.pytorch.constants import CPLoadBalancingStrategy
from transformer_engine.debug.pytorch.debug_state import TEDebugState
from .dot_product_attention.context_parallel import (
    get_seq_chunk_ids_for_reordering_before_attn,
    thd_cp_rank_order_to_sequence_order,
    thd_sequence_order_to_cp_rank_order,
)


# Packed QKV communicates 3F channels; the output projection communicates F.
# Remember both fixed-capacity arenas so incompatible calls can fall back.
_RUNTIMES = {}


class _THDReorder(torch.autograd.Function):
    """Reuse TE's THD permutation with its explicit inverse in autograd."""

    @staticmethod
    def forward(ctx, x, cu_seqlens, cp, to_sequence):
        ctx.save_for_backward(cu_seqlens)
        ctx.cp, ctx.to_sequence = cp, to_sequence
        reorder = (
            thd_cp_rank_order_to_sequence_order
            if to_sequence
            else thd_sequence_order_to_cp_rank_order
        )
        return reorder(x.contiguous(), cu_seqlens, cp)

    @staticmethod
    def backward(ctx, grad):
        (cu_seqlens,) = ctx.saved_tensors
        reorder = (
            thd_sequence_order_to_cp_rank_order
            if ctx.to_sequence
            else thd_cp_rank_order_to_sequence_order
        )
        return reorder(grad.contiguous(), cu_seqlens, ctx.cp), None, None, None


def _linear_supported(linear):
    """The same local eligibility test is used before MHA's collective decision."""
    from transformer_engine.pytorch.distributed_weight import is_distributed_weight
    from transformer_engine.pytorch.utils import (
        torch_is_autocast_gpu_enabled,
        torch_get_autocast_gpu_dtype,
    )

    weight = getattr(linear, "weight", None)
    return (
        linear is not None
        and type(weight) in (torch.Tensor, torch.nn.Parameter)
        and not is_distributed_weight(weight)
        and not torch.compiler.is_compiling()
        and not FP8GlobalStateManager.is_fp8_enabled()
        and not FP8GlobalStateManager.is_fp8_calibration()
        and not TEDebugState.debug_enabled
        and not is_cpu_offload_enabled()
        and (
            not torch_is_autocast_gpu_enabled() or torch_get_autocast_gpu_dtype() == torch.bfloat16
        )
        and not linear.use_bias
        and linear.tp_size == 1
        and not linear.sequence_parallel
        and linear.fsdp_group is None
        and not linear.is_fsdp2
        and not linear.fuse_wgrad_accumulation
        and not linear.wgrad_store.delay_wgrad_compute()
        and not any(
            getattr(linear, name)
            for name in (
                "ub_overlap_ag_fprop",
                "ub_overlap_rs_fprop",
                "ub_overlap_ag_dgrad",
                "ub_overlap_rs_dgrad",
                "ub_bulk_dgrad",
                "ub_bulk_wgrad",
            )
        )
    )


class _CPProjection:
    """Per-call GEMM/exchange policy; tensors remain owned by Linear autograd."""

    def __init__(
        self,
        runtime,
        peer,
        head_dim=128,
        interleave_heads=None,
        validate_group=True,
        validation_state=None,
    ):
        if head_dim <= 0 or (interleave_heads is not None and interleave_heads <= 0):
            raise ValueError("CP projection head dimensions must be positive")
        self.runtime, self.peer, self.head_dim = runtime, peer, head_dim
        self.interleave_heads = interleave_heads
        self.validate_group, self.validation_state = validate_group, validation_state

    def validate(self, linear, x, fp8_output, fp8_grad):
        weight = getattr(linear, "weight", None)
        if not (
            _linear_supported(linear)
            and type(x) in (torch.Tensor, torch.nn.Parameter)
            and x.is_cuda
            and not torch.cuda.is_current_stream_capturing()
            and x.dtype == weight.dtype == torch.bfloat16
            and x.device == weight.device
            and x.ndim == 2
            and not fp8_output
            and not fp8_grad
        ):
            raise ValueError(
                "CP projection hooks require eager BF16 TP1 plain Linear without auxiliary features"
            )
        n, k = weight.shape
        p, d = self.runtime.up, self.head_dim
        if k % d or n % d or (n if self.peer else k) % (p * d):
            raise ValueError("CP projection weights must contain whole heads per rank")
        if self.interleave_heads is not None and n != 3 * self.interleave_heads * d:
            raise ValueError("CP QKV packing does not match the weight rows")
        if x.shape[1] != (k if self.peer else k // self.runtime.up):
            raise ValueError("CP projection input width does not match the weight")
        if not self.peer and x.shape[0] % self.runtime.up:
            raise ValueError("CP projection input rows must be divisible by CP size")

    def forward(self, weight, x):
        from mega_up import peer_write_gemm_thd, prepare_headshard_pull_gemm_thd

        n, k = weight.shape
        d, p = self.head_dim, self.runtime.up
        if self.peer:
            if self.interleave_heads is not None:
                weight = weight.view(3, self.interleave_heads, d, k).transpose(0, 1).reshape(n, k)
            weight = weight.contiguous()
            y = peer_write_gemm_thd(
                x.detach().contiguous().view(-1, k // d, d),
                weight.detach(),
                self.runtime,
                validate_group=self.validate_group,
            )
            return y.reshape(-1, n // p).detach(), x, weight
        weight = weight.contiguous()
        pull = prepare_headshard_pull_gemm_thd(
            weight.detach(),
            self.runtime,
            tokens=x.shape[0] // p,
            heads=k // d,
            head_dim=d,
            workspace_epochs=1,
            validate_group=self.validate_group,
        )
        y = pull(
            x.detach().contiguous().view(-1, k // (p * d), d), validate_group=self.validate_group
        )
        return y.reshape(-1, n).detach(), pull.materialized_input, weight

    def dgrad(self, weight, dy, need_dx):
        from mega_up import peer_write_gemm_thd, prepare_headshard_pull_gemm_thd

        n, k = weight.shape
        d, p = self.head_dim, self.runtime.up
        if self.peer and not need_dx:
            # Wgrad-only needs the inverse exchange, not an otherwise unused GEMM.
            send = dy.detach().contiguous().view(p, -1, n // p)
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send, group=self.runtime.group)
            return None, recv.movedim(0, 1).reshape(-1, n)
        wt = weight.detach().t().contiguous()
        if self.peer:
            pull = prepare_headshard_pull_gemm_thd(
                wt,
                self.runtime,
                tokens=dy.shape[0] // p,
                heads=n // d,
                head_dim=d,
                workspace_epochs=1,
                validate_group=self.validate_group,
            )
            dx = pull(
                dy.detach().contiguous().view(-1, n // (p * d), d),
                validate_group=self.validate_group,
            )
            if self.validate_group and self.validation_state is not None:
                self.validation_state["ready"] = True
            return dx.reshape(-1, k), pull.materialized_input
        dx = peer_write_gemm_thd(
            dy.detach().contiguous().view(-1, n // d, d),
            wt,
            self.runtime,
            validate_group=self.validate_group,
        )
        return dx.reshape(-1, k // p), dy

    def restore_weight_rows(self, dw):
        if dw is not None and self.interleave_heads is not None:
            dw = dw.view(self.interleave_heads, 3, self.head_dim, -1).transpose(0, 1).reshape_as(dw)
        return dw


class _CPAttention:
    """Layout adapters around the existing MHA forward, with no model parameters."""

    def __init__(self, mha, x, runtimes, cu, batch, sequence, validate, contract):
        self.format, self.cu = mha.qkv_format, cu
        self.batch, self.sequence, self.cp = batch, sequence, runtimes[0].up
        self.heads, self.d = mha.num_attention_heads, mha.hidden_size_per_attention_head
        self.hidden = mha.proj.out_features
        self.qkv_width = mha.hidden_size_q + 2 * mha.hidden_size_kv
        self.peer = _CPProjection(
            runtimes[0],
            True,
            self.d,
            None if mha.qkv_weight_interleaved else self.heads,
            validate,
            contract,
        )
        self.pull = _CPProjection(runtimes[1], False, self.d, validate_group=validate)
        self.to_sequence = self.to_rank = None
        if self.format != "thd":
            key = (self.cp, batch, sequence, self.format, x.device)
            cached = getattr(mha, "_megaup_dense_rows", None)
            if cached is None or cached[0] != key:
                rows = torch.arange(self.cp * batch * sequence, device=x.device).view(
                    self.cp, batch, 2, sequence // 2
                )
                rows = rows.permute(1, 0, 2, 3).reshape(batch, 2 * self.cp, sequence // 2)
                before = get_seq_chunk_ids_for_reordering_before_attn(self.cp, x.device)
                forward = rows.index_select(1, before).reshape(batch, self.cp * sequence)
                if self.format == "sbhd":
                    forward = forward.transpose(0, 1)
                forward = forward.flatten()
                cached = (key, forward, torch.argsort(forward))
                object.__setattr__(mha, "_megaup_dense_rows", cached)
            self.to_sequence, self.to_rank = cached[1:]

    def local_rows(self, x):
        if self.format == "sbhd":
            x = x.transpose(0, 1)
        return x.contiguous().view(-1, self.hidden)

    def global_order(self, raw):
        width = self.qkv_width // self.cp
        if self.format == "thd":
            return _THDReorder.apply(
                raw.view(-1, width // self.d, self.d), self.cu, self.cp, True
            ).view(-1, width)
        shape = (self.batch, self.cp * self.sequence)
        if self.format == "sbhd":
            shape = shape[::-1]
        return raw.index_select(0, self.to_sequence).view(*shape, width)

    def restore_output(self, x):
        if self.format == "thd":
            return x
        out = x.view(self.batch, self.sequence, self.hidden)
        return out.transpose(0, 1).contiguous() if self.format == "sbhd" else out


def _thd_values(cu):
    """Read boundary values; keep this host synchronization explicit."""
    return tuple(cu.tolist())


def _thd_metadata(x, cp, cu_seqlens, max_seqlens):
    """Validate the no-padding-gap subset before any rank enters MegaUP barriers."""
    first, *others = cu_seqlens
    for cu in cu_seqlens:
        if not (
            isinstance(cu, torch.Tensor)
            and cu.device == x.device
            and cu.dtype == torch.int32
            and cu.ndim == 1
            and cu.is_contiguous()
        ):
            raise ValueError("THD boundaries must be contiguous device int32 vectors")
    bounds = _thd_values(first)
    if any(cu is not first and _thd_values(cu) != bounds for cu in others):
        raise ValueError("THD Q/KV logical and physical boundaries must match")
    lengths = tuple(b - a for a, b in zip(bounds, bounds[1:]))
    if not (
        bounds
        and bounds[0] == 0
        and lengths
        and bounds[-1] == cp * x.shape[0]
        and all(length > 0 and length % (2 * cp) == 0 for length in lengths)
    ):
        raise ValueError("THD requires positive, dual-chunk-aligned global lengths")
    maximum = max(lengths)
    maxima = tuple(maximum if value is None else value for value in max_seqlens)
    if any(type(value) is not int or value < maximum for value in maxima):
        raise ValueError("THD maximum sequence lengths must cover every sequence")
    return first, bounds, maxima


def _cached_thd_metadata(mha, x, cp, metadata, max_seqlens):
    """One-entry cache for trusted, normally version-tracked boundary tensors."""
    if not all(isinstance(cu, torch.Tensor) for cu in metadata) or any(
        maximum is not None and type(maximum) is not int for maximum in max_seqlens
    ):
        return _thd_metadata(x, cp, metadata, max_seqlens)
    try:
        key = (
            cp,
            tuple(x.shape),
            x.device,
            max_seqlens,
            tuple(
                (
                    cu._version,
                    tuple(cu.shape),
                    cu.stride(),
                    cu.dtype,
                    cu.device,
                    cu.data_ptr(),
                    cu.storage_offset(),
                )
                for cu in metadata
            ),
        )
    except RuntimeError:
        # Inference tensors have no version counter; reread rather than cache.
        return _thd_metadata(x, cp, metadata, max_seqlens)
    cached = getattr(mha, "_megaup_thd_metadata_cache", None)
    if (
        cached is not None
        and cached[0] == key
        and all(old is new for old, new in zip(cached[1], metadata))
    ):
        return cached[2]
    result = _thd_metadata(x, cp, metadata, max_seqlens)
    # Strong references prevent allocator address reuse from resembling a hit.
    object.__setattr__(mha, "_megaup_thd_metadata_cache", (key, metadata, result))
    return result


def prepare_megaup(
    mha,
    x,
    *,
    eligible_call,
    rotary_pos_emb,
    attn_mask_type,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    cu_seqlens_q_padded=None,
    cu_seqlens_kv_padded=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
):
    """Prepare projection hooks and external CP exchanges, or choose rank-uniform fallback."""
    trusted = os.getenv("NVTE_MHA_MEGAUP_TRUSTED", "0") == "1"
    contract = getattr(mha, "_megaup_validated_contract", None) if trusted else None
    group = mha.core_attention.cp_group
    if contract is not None and contract["group"] is not group:
        raise RuntimeError("Trusted MegaUP requires the initialized CP group")
    if not isinstance(group, dist.ProcessGroup) or dist.get_world_size(group) == 1:
        return None
    if x.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("NVTE_MHA_MEGAUP does not support CUDA graph capture")

    cp = dist.get_world_size(group)
    heads, head_dim = mha.num_attention_heads, mha.hidden_size_per_attention_head
    # Model width, query attention width and packed KV-group width can differ.
    hidden, groups = mha.proj.out_features, mha.num_gqa_groups
    width = heads * head_dim
    qkv_width = (heads + 2 * groups) * head_dim
    thd = mha.qkv_format == "thd"
    cu_seqlens, bounds, maxima = None, None, (None, None)
    metadata_valid = True
    metadata = (cu_seqlens_q, cu_seqlens_kv, cu_seqlens_q_padded, cu_seqlens_kv_padded)
    if thd:
        try:
            if trusted:
                cu_seqlens, bounds, maxima = _cached_thd_metadata(
                    mha, x, cp, metadata, (max_seqlen_q, max_seqlen_kv)
                )
            else:
                cu_seqlens, bounds, maxima = _thd_metadata(
                    x, cp, metadata, (max_seqlen_q, max_seqlen_kv)
                )
        except (ValueError, TypeError, RuntimeError, IndexError):
            metadata_valid = False
    else:
        metadata_valid = all(value is None for value in (*metadata, max_seqlen_q, max_seqlen_kv))
    qkv = getattr(mha, "qkv", None)
    weight = getattr(qkv, "weight", None)
    out_weight = getattr(mha.proj, "weight", None)
    valid = (
        eligible_call
        and metadata_valid
        and mha.attention_type == "self"
        and not mha.input_layernorm
        and mha.tp_size == 1
        and not mha.sequence_parallel
        # Interleaved GQA stores each group's queries, K and V contiguously,
        # so one rank shard contains whole groups without repeating KV.
        and (groups == heads or mha.qkv_weight_interleaved)
        and mha.q_norm is None
        and mha.k_norm is None
        and mha.softmax_type == "vanilla"
        and mha.core_attention.cp_comm_type == "a2a"
        and mha.core_attention.load_balancing_strategy is CPLoadBalancingStrategy.DUAL_CHUNK_SWAP
        and mha.core_attention.attention_dropout == 0
        and not mha.core_attention.return_max_logit
        and mha.qkv_format in ("bshd", "sbhd", "thd")
        and attn_mask_type in (("padding_causal", "padding") if thd else ("causal", "no_mask"))
        and not FP8GlobalStateManager.is_fp8_enabled()
        and not FP8GlobalStateManager.is_fp8_calibration()
        and not TEDebugState.debug_enabled
        and not is_cpu_offload_enabled()
        and not getattr(qkv, "use_bias", True)
        and not mha.proj.use_bias
        and not getattr(qkv, "fuse_wgrad_accumulation", True)
        and not mha.proj.fuse_wgrad_accumulation
        and _linear_supported(qkv)
        and _linear_supported(mha.proj)
        and not isinstance(x, QuantizedTensorStorage)
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim == (2 if thd else 3)
        and x.shape[-1] == hidden
        and cp in (2, 4, 8)
        and heads % cp == 0
        and groups % cp == 0
        and hidden % 256 == 0
        and width % 256 == 0
        and qkv_width % 256 == 0
        and (width // cp) % 128 == 0
        and (qkv_width // cp) % 128 == 0
        and torch.cuda.get_device_capability(x.device)[0] == 10
        and isinstance(weight, torch.Tensor)
        and not isinstance(weight, QuantizedTensorStorage)
        and weight.dtype == torch.bfloat16
        and tuple(weight.shape) == (qkv_width, hidden)
        and weight.device == x.device
        and isinstance(out_weight, torch.Tensor)
        and not isinstance(out_weight, QuantizedTensorStorage)
        and out_weight.dtype == torch.bfloat16
        and tuple(out_weight.shape) == (hidden, width)
        and out_weight.device == x.device
        and out_weight.is_contiguous()
    )
    shape = tuple(x.shape)
    batch, sequence = (
        (1, shape[0])
        if thd and len(shape) == 2
        else (
            (shape[:2] if mha.qkv_format == "bshd" else shape[1::-1]) if len(shape) == 3 else (0, 0)
        )
    )
    runtimes = _RUNTIMES.get(group)
    valid = valid and sequence > 0 and (thd or sequence % 2 == 0) and batch > 0
    if runtimes is not None:
        valid = valid and all(
            runtime.hidden == channels
            and batch * sequence <= runtime.max_rows
            and runtime.device == x.device
            for runtime, channels in zip(runtimes, (qkv_width, width))
        )

    # A local unsupported input must not leave peers entering symmetric barriers.
    signature = (
        bool(valid),
        shape,
        heads,
        groups,
        hidden,
        head_dim,
        mha.qkv_format,
        mha.qkv_weight_interleaved,
        attn_mask_type,
        rotary_pos_emb is not None,
        bounds,
        maxima,
        None if runtimes is None else tuple((r.hidden, r.max_rows) for r in runtimes),
    )
    if contract is not None:
        # Trusted callers keep all ranks on one fixed contract. A local change
        # cannot safely reenter initialization collectives while peers launch payloads.
        if not valid or contract["signature"] != signature:
            raise RuntimeError("Trusted MegaUP contract changed; reset on all CP ranks first")
        validate_group = not contract["ready"]
    else:
        records = [None] * cp
        dist.all_gather_object(records, (signature, trusted), group=group)
        if not all(record == (signature, trusted) for record in records) or not valid:
            return None
        validate_group = True
        import_error = None
        try:
            from mega_up import initialize_runtime
        except ImportError as error:
            import_error = str(error)
        dist.all_gather_object(records, import_error, group=group)
        if any(error is not None for error in records):
            raise ImportError(f"NVTE_MHA_MEGAUP requires the optional mega_up package: {records}")
        if runtimes is None:
            runtimes = tuple(
                initialize_runtime(
                    group, hidden=channels, max_rows=batch * sequence, device=x.device
                )
                for channels in (qkv_width, width)
            )
            _RUNTIMES[group] = runtimes
        if trusted:
            # Publish only after setup succeeds; old graphs retain their own state.
            contract = {
                "signature": signature[:-1] + (tuple((r.hidden, r.max_rows) for r in runtimes),),
                "group": group,
                "ready": False,
            }
            object.__setattr__(mha, "_megaup_validated_contract", contract)
    plan = _CPAttention(mha, x, runtimes, cu_seqlens, batch, sequence, validate_group, contract)
    plan.maxima = maxima
    return plan
