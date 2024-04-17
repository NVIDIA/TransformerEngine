# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Base modules and utilities for TransformerEngine PyTorch API"""
import io
import os
import pickle
import warnings
from abc import ABC, abstractmethod
from typing import Generator, Union, Optional, Tuple, List
from contextlib import contextmanager

import torch
import torch.nn.functional as F

import transformer_engine_extensions as tex
from ._common import _ParameterInitMeta
from ..export import is_in_onnx_export_mode
from ..fp8 import (
    get_default_fp8_recipe,
    get_fp8_te_dtype,
    FP8GlobalStateManager,
)
from ..distributed import (
    gather_along_first_dim,
    is_fp8_activation_recompute_enabled,
    in_fp8_activation_recompute_phase,
)
from ..cpp_extensions import (
    fp8_cast_transpose_fused,
    fp8_cast_transpose_bgrad_fused,
    cast_to_fp8,
)
from ..constants import dist_group_type
from ..float8_tensor import Float8Tensor

_2X_ACC_FPROP = False
_2X_ACC_DGRAD = True
_2X_ACC_WGRAD = True
_cublas_workspace = None
_ub_communicators = None
_NUM_MAX_UB_STREAMS = 3
layers_atomic_ring_exchange = []


def get_cublas_workspace_size_bytes() -> None:
    """Return 32 MiB if using hopper, 4 MiB for all other architectures."""
    if torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 9:
        return 33_554_432
    return 4_194_304


def get_workspace() -> torch.Tensor:
    """Returns workspace for cublas."""
    global _cublas_workspace
    if _cublas_workspace is None:
        _cublas_workspace = torch.empty(
            get_cublas_workspace_size_bytes(), dtype=torch.uint8, device="cuda"
        )
    return _cublas_workspace


def initialize_ub(
    shape: list,
    tp_size: int,
    use_fp8: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    ub_cfgs: Optional[dict] = None
) -> None:
    """Initialize communicators for TP comm overlap using userbuffers."""
    global _ub_communicators
    assert _ub_communicators is None, "UB communicators are already initialized."
    _ub_communicators = {}
    rank_id = torch.distributed.get_rank()

    # Increase the workspace by the number of maximum concurrent streams
    global _cublas_workspace
    _cublas_workspace = get_workspace().repeat(_NUM_MAX_UB_STREAMS)

    # Default buffer precision: AllGather buffers use fp8 when using fp8 recipe
    layers_all_gather_overlap = [
        "qkv_fprop", "qkv_dgrad", "proj_dgrad", "fc1_fprop", "fc1_dgrad", "fc2_dgrad"
    ]
    layers_reduce_scatter_overlap = ["proj_fprop", "fc2_fprop", "qkv_wgrad", "fc1_wgrad"]
    dgrad_reduce_scatter_overlap = ["qkv_dgrad", "fc1_dgrad"]
    # Default overlap methods for layers
    methods = {
        "ring_exchange":["qkv_fprop", "fc1_fprop", "proj_dgrad", "fc2_dgrad"],
        "pipeline":["proj_fprop", "fc2_fprop"],
        "bulk":["qkv_dgrad", "qkv_wgrad", "fc1_dgrad", "fc1_wgrad"],
    }

    # AG-RS overlap pairs of layers forming a tensor-parallel block
    ag_rs_pairs = {"qkv_fprop":"proj_fprop", "fc1_fprop":"fc2_fprop"}
    rs_ag_pairs = {v : k for k, v in ag_rs_pairs.items()}
    global layers_atomic_ring_exchange
    layers_atomic_ring_exchange = []

    def get_method(name):
        for method, names in methods.items():
            if name in names:
                return method
        raise KeyError(f"Given layer name {name} does not exist.")

    def add_ub(
        name: str,
        method: str,
        is_reduce_scatter: int,
        num_sm: int = 16,
        cga_size: int = 2,
        set_sm_margin: int = 0,
        num_splits: int = 0,
        aggregate: int = 0,
        atomic_gemm: int = 0,
        fp8_buf: bool = False,
    ) -> None:
        if atomic_gemm:
            warnings.warn(
                "Atomic GEMM uses a beta API from cublas and is not tested for all use cases."
            )
            assert use_fp8, "Atomic GEMM overlap supported only for FP8 GEMM."
            if method == 'bulk':
                warnings.warn(
                    f"At {name}, atoimic GEMM not is supported for a bulk overlap."
                    "Defaulting to `atomic_gemm=False`."
                )
                atomic_gemm = 0
        if not is_reduce_scatter and method == 'pipeline':
            raise ValueError(
                f"At {name}, `pipeline` overlap method is not supported for AllGather."
            )
        # Check if both AG and RS overlaps use `atomic GEMM`` + `p2p ring-exchange`.
        # Using atomic GEMM + p2p ring-exchange in only one of the pair breaks functionality.
        global layers_atomic_ring_exchange
        if atomic_gemm and method == "ring_exchange" and name in ag_rs_pairs:
            layers_atomic_ring_exchange += [name, ag_rs_pairs[name]]
        if name in rs_ag_pairs:
            assert_message = (
                f"At {name}, atomic AG-GEMM overlap with `ring_exchange` shuffles GEMM chunk "
                "outputs, and  RS-GEMM overlap un-suffle them. When one of the GEMM-AG and "
                "GEMM-RS overlaps forming a TP block (e.g., qkv_fprop and proj_fprop) uses "
                "`atomic gemm` and `ring_exhcnage`, its pair must use the same overlap config "
                "for functionality."
            )
            if name in layers_atomic_ring_exchange:
                assert atomic_gemm and method == "ring_exchange", assert_message
            else:
                if atomic_gemm and method == "ring_exchange":
                    assert rs_ag_pairs[name] in layers_atomic_ring_exchange, assert_message

        sample_buffer = torch.empty(
            shape,
            dtype=torch.uint8 if (use_fp8 and fp8_buf) else dtype,
            device='cuda')
        if method == 'ring_exchange':
            ub_obj = tex.UbufP2PCommOverlap(
                    sample_buffer,          # Sample userbuffer
                    rank_id,                # Rank id
                    tp_size,                # TP size
                    num_sm,                 # Number of communication SMs
                    cga_size,               # CGA cluster size
                    set_sm_margin,          # Set SM margin
                    aggregate,              # Aggregate 2X GEMM chunks
                    _NUM_MAX_UB_STREAMS,    # Max concurrent GEMM streams
                    is_reduce_scatter,      # overlap with reduce scatter
                    atomic_gemm,            # use a single GEMM with atomic-counters
                    torch.Tensor(),         # empty tensor to pass to counters
                )
        else:
            ub_obj = tex.UbufCommOverlap(
                    sample_buffer,          # Sample userbuffer
                    rank_id,                # Rank id
                    tp_size,                # TP size
                    num_sm,                 # Number of communication SMs
                    cga_size,               # CGA cluster size
                    num_splits,             # Number of communication splits
                    set_sm_margin,          # Set SM margin
                    _NUM_MAX_UB_STREAMS,    # Max concurrent GEMM streams
                    atomic_gemm,            # use a single GEMM with atomic-counters
                    torch.Tensor(),         # empty tensor to pass to counters
                )
        _ub_communicators[name] = ub_obj

    if ub_cfgs is not None:
        for name in dgrad_reduce_scatter_overlap:
            if name in ub_cfgs and 'method' in ub_cfgs[name] and ub_cfgs[name]['method'] != 'bulk':
                wgrad_name = name.replace('dgrad','wgrad')
                assert wgrad_name not in ub_cfgs
                layers_reduce_scatter_overlap.remove(wgrad_name)
                layers_reduce_scatter_overlap.append(name)

    for name in (methods["ring_exchange"]+methods["pipeline"]+methods["bulk"]):
        if ub_cfgs is not None and name in ub_cfgs:
            ub_cfg = ub_cfgs[name]
            method = ub_cfg.get("method", get_method(name))
            num_sm = ub_cfg.get("num_sm", 16)
            cga_size = ub_cfg.get("cga_size", 2)
            num_splits = ub_cfg.get("num_splits", 4 if method == "pipeline" else 0)
            set_sm_margin = ub_cfg.get("set_sm_margin", 0)
            aggregate = ub_cfg.get("aggregate", 0)
            atomic_gemm = ub_cfg.get("atomic_gemm", 0)
            is_reduce_scatter = 1 if name in layers_reduce_scatter_overlap else 0
            # Support FP8 userbuffer when (1) AllGather and (2) FP8-GEMM output ReduceScatter
            fp8_buf = ((name in layers_all_gather_overlap) or
                      (ub_cfg.get("fp8_buf", False) and name in methods["pipeline"]))
            add_ub(
                name,
                method,
                is_reduce_scatter,
                num_sm,
                cga_size,
                set_sm_margin,
                num_splits,
                aggregate,
                atomic_gemm,
                fp8_buf,
            )
        else:
            method = get_method(name)
            add_ub(
                name,
                method=method,
                is_reduce_scatter=1 if name in layers_reduce_scatter_overlap else 0,
                num_splits=4 if method == "pipeline" else 0,
                fp8_buf=name in layers_all_gather_overlap,
            )


def get_ub(name: str):
    """Get userbuffer communicator corresponding to give key."""
    global _ub_communicators
    assert _ub_communicators is not None, "UB manager is not initialized."
    assert name in _ub_communicators, f"UB for {name} is not registered."
    return _ub_communicators[name]


class TransformerEngineBaseModule(torch.nn.Module, ABC):
    """Base TE module."""

    def __init__(self) -> None:
        super().__init__()
        assert torch.cuda.is_available(), "TransformerEngine needs CUDA."
        self.fp8_initialized = False
        self.fp8 = False
        self.fp8_calibration = False
        self.fp8_meta = {}
        self.fp8_meta["fp8_checkpoint"] = False
        self.fp8_meta["fp8_group"] = None
        self.fp8_meta["recipe"] = get_default_fp8_recipe()
        self.fp8_meta_tensors_initialized = False
        self.tp_group = None
        self.tp_size = 1
        self.sequence_parallel = False
        self.fp8_weight_shapes = []
        self.param_init_meta = {}
        self.primary_weights_in_fp8 = FP8GlobalStateManager.with_fp8_parameters()

    def adjust_amax_history_length(self, length: int, fwd: Optional[bool] = None) -> None:
        """Increase or decrease size of amax history based on given `length`.

        .. warning::
            This changes the underlying amax memory location.
        """
        if fwd is None:
            fp8_meta_tensor_keys = ("scaling_fwd", "scaling_bwd")
        else:
            fp8_meta_tensor_keys = ("scaling_fwd" if fwd else "scaling_bwd",)

        for meta_key in fp8_meta_tensor_keys:
            if meta_key not in self.fp8_meta:
                # Handles non-parameter FP8 modules, e.g. DPA.
                continue
            curr_len = self.fp8_meta[meta_key].amax_history.shape[0]
            if length == curr_len:
                continue
            if length < curr_len:
                self.fp8_meta[meta_key].amax_history = (
                    self.fp8_meta[meta_key].amax_history[: length].clone())
            elif length > curr_len:
                extra_rows = length - curr_len
                self.fp8_meta[meta_key].amax_history = F.pad(
                    self.fp8_meta[meta_key].amax_history, pad=(0, 0, 0, extra_rows)
                )

            # Update the global buffers with new amax and history pointers.
            if FP8GlobalStateManager.get_buffer_info() in self.fp8_meta:
                fwd_pos, fwd_key, bwd_pos, bwd_key = (
                    self.fp8_meta[FP8GlobalStateManager.get_buffer_info()])
                for pos, buffer_key in zip((fwd_pos, bwd_pos), (fwd_key, bwd_key)):
                    if buffer_key in FP8GlobalStateManager.global_amax_buffer:
                        assert (
                            buffer_key in FP8GlobalStateManager.global_amax_history_buffer
                        ), "TE internal error during amax history change."
                        FP8GlobalStateManager.global_amax_buffer[buffer_key][pos] = (
                            self.fp8_meta[meta_key].amax_history[0])
                        FP8GlobalStateManager.global_amax_history_buffer[buffer_key][pos] = (
                            self.fp8_meta[meta_key].amax_history)

    def set_meta_tensor(self, fwd: bool) -> None:
        """Init scales and amaxes for fwd | bwd."""
        fp8_meta_tensor_key = "scaling_fwd" if fwd else "scaling_bwd"

        if self.fp8_meta_tensors_initialized:
            # Handle changed amax history size.
            self.adjust_amax_history_length(self.fp8_meta["recipe"].amax_history_len, fwd=fwd)
            return

        # Max. number of fp8 tensors per GEMM = 3 (input, weight, output) for fwd and
        # 2 (grad_output and grad_input) for bwd
        num_fp8_tensors = (
            self.fp8_meta["num_gemms"] * 3 if fwd else self.fp8_meta["num_gemms"] * 2
        )

        self.fp8_meta[fp8_meta_tensor_key] = tex.FP8TensorMeta()
        self.fp8_meta[fp8_meta_tensor_key].scale = torch.ones(
            num_fp8_tensors, dtype=torch.float32, device="cuda"
        )
        self.fp8_meta[fp8_meta_tensor_key].scale_inv = torch.ones(
            num_fp8_tensors, dtype=torch.float32, device="cuda"
        )
        self.fp8_meta[fp8_meta_tensor_key].amax_history = torch.zeros(
            self.fp8_meta["recipe"].amax_history_len,
            num_fp8_tensors,
            dtype=torch.float32,
            device="cuda",
        )

    def init_fp8_meta_tensors(self) -> None:
        """Init scales and amaxes."""
        self.set_meta_tensor(True)
        self.set_meta_tensor(False)
        self.fp8_meta_tensors_initialized = True

    def get_fp8_meta_tensors(self) -> None:
        """Get scales and amaxes."""
        fwd_key, bwd_key = "scaling_fwd", "scaling_bwd"
        if fwd_key not in self.fp8_meta or bwd_key not in self.fp8_meta:
            return None

        fp8_meta_tensors = {fwd_key: [], bwd_key: []}
        with torch.no_grad():
            for key in (fwd_key, bwd_key):
                fp8_meta_tensors[key].append(self.fp8_meta[key].scale.clone())
                fp8_meta_tensors[key].append(self.fp8_meta[key].scale_inv.clone())
                fp8_meta_tensors[key].append(self.fp8_meta[key].amax_history.clone())
        return fp8_meta_tensors

    def reset_fp8_meta_tensors(self, fp8_meta_tensors=None) -> None:
        """Reset scales and amaxes."""
        def reset(key):
            if key in self.fp8_meta:
                if fp8_meta_tensors is None:
                    self.fp8_meta[key].scale.copy_(torch.ones_like(self.fp8_meta[key].scale))
                    self.fp8_meta[key].scale_inv.copy_(
                        torch.ones_like(self.fp8_meta[key].scale_inv))
                    self.fp8_meta[key].amax_history.copy_(
                        torch.zeros_like(self.fp8_meta[key].amax_history))
                else:
                    assert key in fp8_meta_tensors, "Cannot reset fp8 tensors."
                    self.fp8_meta[key].scale.copy_(fp8_meta_tensors[key][0])
                    self.fp8_meta[key].scale_inv.copy_(fp8_meta_tensors[key][1])
                    self.fp8_meta[key].amax_history.copy_(fp8_meta_tensors[key][2])
        with torch.no_grad():
            reset("scaling_fwd")
            reset("scaling_bwd")

    def get_extra_state(self) -> torch.Tensor:
        """Save before checkpointing."""
        state = None

        fp8_checkpoint = self.fp8_meta["fp8_checkpoint"] or self.fp8 or self.fp8_calibration

        if fp8_checkpoint:
            state = {}
            state["scale_fwd"] = self.fp8_meta["scaling_fwd"].scale
            state["scale_inv_fwd"] = self.fp8_meta["scaling_fwd"].scale_inv
            state["amax_history_fwd"] = self.fp8_meta["scaling_fwd"].amax_history
            state["scale_bwd"] = self.fp8_meta["scaling_bwd"].scale
            state["scale_inv_bwd"] = self.fp8_meta["scaling_bwd"].scale_inv
            state["amax_history_bwd"] = self.fp8_meta["scaling_bwd"].amax_history

            # Store other pickelable values.
            extra = {}
            for k, v in self.fp8_meta.items():
                if isinstance(v, (bool, int, float, str, tuple, list)):
                    extra[k] = v
            state["extra_fp8_variables"] = extra

        if is_in_onnx_export_mode():
            state_serialized = torch.frombuffer(pickle.dumps(state), dtype=torch.uint8)
        else:
            state_serialized = io.BytesIO()
            torch.save(state, state_serialized)

        return state_serialized

    def set_extra_state(self, state: torch.Tensor) -> None:
        """Load previous state."""
        if state is None:
            return

        if isinstance(state, torch.Tensor):
            state = pickle.loads(state.detach().cpu().numpy().tobytes())
        elif isinstance(state, io.BytesIO):
            state.seek(0)
            state = torch.load(state, map_location='cuda')
        else:
            raise RuntimeError("Unsupported checkpoint format.")

        if state is None:
            return

        # Load extra items.
        self.fp8_meta.update(state["extra_fp8_variables"])
        self.fp8_meta["recipe"].amax_history_len = state["amax_history_fwd"].shape[0]
        if "global_fp8_buffer_pos_fwd_recompute" in self.fp8_meta:
            del self.fp8_meta["global_fp8_buffer_pos_fwd_recompute"]

        # Initialize before loading.
        self.init_fp8_meta_tensors()
        self.fp8_meta["scaling_fwd"].scale.copy_(state["scale_fwd"])
        self.fp8_meta["scaling_fwd"].amax_history.copy_(state["amax_history_fwd"])
        self.fp8_meta["scaling_bwd"].scale.copy_(state["scale_bwd"])
        self.fp8_meta["scaling_bwd"].amax_history.copy_(state["amax_history_bwd"])
        self.fp8_meta["scaling_fwd"].scale_inv.copy_(state["scale_inv_fwd"])
        self.fp8_meta["scaling_bwd"].scale_inv.copy_(state["scale_inv_bwd"])

    def set_activation_dtype(self, inp: torch.Tensor) -> None:
        """Get activation data type for AMP."""
        # Native AMP (`torch.autocast`) gets highest priority
        if torch.is_autocast_enabled():
            self.activation_dtype = torch.get_autocast_gpu_dtype()
            return

        # All checks after this have already been performed once, thus skip
        if hasattr(self, "activation_dtype") and self.activation_dtype == inp.dtype:
            return

        dtype = inp.dtype
        for name, param in self.named_parameters():
            if param is not None:
                assert dtype == param.dtype, (
                    "Data types for parameters must match when outside of autocasted region. "
                    f" Found input dtype: {dtype} and {name!r} dtype: {param.dtype}"
                )
        for name, buf in self.named_buffers():
            if buf is not None:
                assert dtype == buf.dtype, (
                    "Data types for buffers must match when outside of autocasted region. "
                    f" Found input dtype: {dtype} and {name!r} dtype: {buf.dtype}"
                )
        self.activation_dtype = dtype

    def set_fp8_weights(self) -> None:
        """Construct workspace buffers for FP8 weights, if needed

        These workspace buffers are used for FP8 training when the
        module parameters are not natively in FP8 and there are
        multiple microbatches per training step. The buffers, with
        names like `weight1_fp8` and `weight1_t_fp8`, cache the FP8
        values and transposed FP8 values in between microbatches. They
        are not registered as module parameters or buffers since we
        don't want them to be affected by `.to` and since they aren't
        needed for checkpointing.

        """
        if not self.fp8 or self.primary_weights_in_fp8:
            return

        for i, shape in enumerate(self.fp8_weight_shapes, start=1):
            weight_cast_attr = f"weight{i}_fp8"
            weight_transpose_attr = f"weight{i}_t_fp8"

            if (
                hasattr(self, weight_cast_attr)
                and getattr(self, weight_cast_attr).shape == shape
            ):
                return

            setattr(
                self,
                weight_cast_attr,
                Float8Tensor(
                    data=torch.empty(
                        shape,
                        device=torch.cuda.current_device(),
                        dtype=torch.uint8,
                    ),
                    fp8_dtype=tex.DType.kFloat8E4M3,
                    fp8_scale_inv=1,
                )
            )
            setattr(
                self,
                weight_transpose_attr,
                Float8Tensor(
                    data=torch.empty(
                        shape[1],
                        shape[0],
                        device=torch.cuda.current_device(),
                        dtype=torch.uint8,
                    ),
                    fp8_dtype=tex.DType.kFloat8E4M3,
                    fp8_scale_inv=1,
                )
            )

    def set_tensor_parallel_group(self, tp_group: Union[dist_group_type, None]) -> None:
        """
        Set the tensor parallel group for the given
        module before executing the forward pass.

        Parameters
        ----------
        tp_group : ProcessGroup, default = `None`
                  tensor parallel process group.
        """
        self.tp_group = tp_group
        self.tp_group_initialized = True

    def _get_fp8_params(self) -> Union[List[torch.Tensor], None]:
        """returns the FP8 weights."""
        fp8_params = []
        for param in self.parameters():
            if isinstance(param, Float8Tensor) and param.requires_grad:
                fp8_params.append(param)
        if len(fp8_params) == 0:
            return None
        return fp8_params

    # This routine is shared across FP8 and FP8_calibration paths so should not actually
    # assume FP8 execution.
    def init_fp8_metadata(self, num_gemms: int = 1) -> None:
        """Initialize fp8 related metadata and tensors during fprop."""
        self.fp8_parameters = FP8GlobalStateManager.with_fp8_parameters()
        self.fp8 = FP8GlobalStateManager.is_fp8_enabled()
        self.fp8_calibration = FP8GlobalStateManager.is_fp8_calibration()
        self.fp8_meta["fp8_checkpoint"] = self.fp8 or self.fp8_calibration

        if self.fp8_parameters and not self.fp8_initialized:
            self.fp8_meta["num_gemms"] = num_gemms
            self.init_fp8_meta_tensors()

        if self.fp8 or self.fp8_calibration:
            # FP8 init has already been run and recipe is the same, don't do anything.
            if (self.fp8_initialized
                and FP8GlobalStateManager.get_fp8_recipe() == self.fp8_meta["recipe"]):
                return

            # Set FP8, recipe, and other FP8 metadata
            self.fp8_meta["recipe"] = FP8GlobalStateManager.get_fp8_recipe()
            self.fp8_meta["num_gemms"] = num_gemms
            self.fp8_meta["fp8_group"] = FP8GlobalStateManager.get_fp8_group()

            # Set FP8_MAX per tensor according to recipe
            self.fp8_meta["fp8_max_fwd"] = self.fp8_meta["recipe"].fp8_format.value.max_fwd
            self.fp8_meta["fp8_max_bwd"] = self.fp8_meta["recipe"].fp8_format.value.max_bwd

            # Allocate scales and amaxes
            self.init_fp8_meta_tensors()
            self.fp8_initialized = True
        else:
            # If fp8 isn't enabled, turn off and return.
            self.fp8_initialized = False
            return

    @contextmanager
    def prepare_forward(
        self,
        inp: torch.Tensor,
        is_first_microbatch: Union[bool, None],
        num_gemms: int = 1,
        allow_non_contiguous: bool = False,
    ) -> Generator[torch.Tensor, None, None]:
        """Checks and prep for FWD.
        The context manager is needed because there isn't a way for a module to know
        if it's the last FP8 module in the forward autocast. It is useful
        to setup the forward aggregated amax reduction for every module
        just in case. The autocast exit will pick up the most recent one.
        """
        # Activation recomputation is used and this is the second forward phase.
        if self.fp8 and in_fp8_activation_recompute_phase():
            FP8GlobalStateManager.get_old_fp8_meta_tensors_for_recompute(self.fp8_meta)
        else:
            assert inp.is_cuda, "TransformerEngine needs CUDA."

            if self.tp_size > 1:
                assert self.tp_group_initialized, "TP group not initialized."

            self.set_activation_dtype(inp)
            self.init_fp8_metadata(num_gemms=num_gemms)

            # Create persistent tensors for fp8 weights and their transposes
            # only when fp8 weight caching is used and weights are not in fp8
            if is_first_microbatch is not None and not self.primary_weights_in_fp8:
                self.set_fp8_weights()

            if self.fp8 and self.sequence_parallel:
                assert self.fp8_meta["recipe"].reduce_amax, \
                "Amax reduction across tensor parallel group is " \
                "necessary when using sequence parallelism with FP8."

            if self.fp8 and not FP8GlobalStateManager.fp8_graph_capturing():
                FP8GlobalStateManager.add_fp8_tensors_to_global_buffer(
                    self.fp8_meta, fp8_weights=self._get_fp8_params())

            # Activation recomputation is used and this is the first forward phase.
            if (
                self.fp8
                and self.training
                and is_fp8_activation_recompute_enabled()
            ):
                FP8GlobalStateManager.copy_forward_fp8_meta_tensors_for_recompute(self.fp8_meta)

        with torch.cuda.nvtx.range(self.__class__.__name__ + " forward"):
            if not allow_non_contiguous:
                yield inp.contiguous()
            else:
                yield inp

        if self.fp8 and in_fp8_activation_recompute_phase():
            FP8GlobalStateManager.restore_fp8_meta_tensors(self.fp8_meta)
            return

    def set_nccl_overlap_warning_if_tp(self) -> None:
        """When using TP, the NCCL communication needs to be scheduled
        before the GEMM for there to be a guaranteed overlap. From the
        host side in TE, the comm calls are always launched first, but
        to ensure that the GEMM isn't scheduled first, the environment
        variable `CUDA_DEVICE_MAX_CONNECTIONS` needs to be set to 1 to
        force a single channel.
        """
        if self.tp_size == 1:
            return
        num_cuda_work_queues = int(os.getenv("CUDA_DEVICE_MAX_CONNECTIONS", "0"))
        if num_cuda_work_queues != 1:
            warnings.warn(
                "To guarantee overlapping TP and SP collectives with the backward"
                "GEMMs, set environment variable CUDA_DEVICE_MAX_CONNECTIONS = 1"
            )

    @staticmethod
    def grad_output_preprocess(
        ctx, grad_output: torch.Tensor, row_parallel_mode: bool
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        """Utility function for backward.
        Returns tuple in order (all optional/None based on training precion/recipe):
            R1: gathered `grad_output` in higher precision.
            R2: gathered `grad_output` in FP8.
            R3: R2 transposed.
            R4: bias gradient on R1.

        """
        if isinstance(grad_output, Float8Tensor):
            grad_output._data = grad_output._data.contiguous()
        else:
            grad_output = grad_output.contiguous()
        grad_output_mat = grad_output.view(-1, grad_output.shape[-1])
        gather_grad_output = row_parallel_mode and ctx.sequence_parallel

        # No-FP8 case: bgrad is fused with wgrad for this case.
        if not ctx.fp8:
            if gather_grad_output:
                if not ctx.ub_overlap_ag:
                    grad_output_mat, _ = gather_along_first_dim(
                        grad_output_mat, ctx.tp_group
                    )
                else:
                    ctx.ub_obj_gradout.copy_input_to_ubuf(grad_output, True)
                    grad_output_mat = ctx.ub_obj_gradout.get_ubuf_output(1)
            return grad_output_mat, None, None, None

        fp8_dtype_backward = get_fp8_te_dtype(
            ctx.fp8_meta["recipe"], fprop_tensor=False
        )

        # FP8 case with non-FP8 wgrad
        if (
            gather_grad_output
            and ctx.fp8_meta["recipe"].override_linear_precision.wgrad
        ):
            assert (
                not ctx.ub_overlap_ag
            ), "override_linear_precision.wgrad not supported with UB AG overlap"
            grad_output_mat, _ = gather_along_first_dim(grad_output_mat, ctx.tp_group)
        # FP8 case with gather: unfused bgrad, cast, transpose for efficient gather
        elif gather_grad_output:
            if ctx.use_bias:
                grad_bias = grad_output_mat.sum(dim=0)
            else:
                grad_bias = None
            if ctx.ub_overlap_ag:
                grad_output_c = ctx.ub_obj_gradout.get_ubuf_output(0)
            else:
                grad_output_c = torch.empty_like(grad_output_mat, dtype=torch.uint8)
            if not isinstance(grad_output_mat, Float8Tensor):
                cast_to_fp8(
                    grad_output_mat,
                    ctx.fp8_meta["scaling_bwd"],
                    tex.FP8BwdTensors.GRAD_OUTPUT1,
                    fp8_dtype_backward,
                    out=grad_output_c,
                )
            else:
                grad_output_c = grad_ouput_mat # pylint: disable=undefined-variable
            if not ctx.ub_overlap_ag:
                grad_output_c, _ = gather_along_first_dim(grad_output_c, ctx.tp_group)
                if not isinstance(grad_output_c, Float8Tensor):
                    grad_output_t = tex.fp8_transpose(grad_output_c, fp8_dtype_backward)
                else:
                    grad_output_t = grad_output_c.transpose_2d()
            else:
                grad_output_c = ctx.ub_obj_gradout.get_ubuf_output(1)
                grad_output_t = None

            return grad_output_mat, grad_output_c, grad_output_t, grad_bias

        # FP8 case without gather: cast, transpose, bgrad fused
        if ctx.use_bias:
            grad_output_mat_no_fp8 = grad_output_mat
            if isinstance(grad_output_mat, Float8Tensor):
                grad_output_mat_no_fp8 = grad_output_mat.from_float8(grad_output_mat.dtype)
            grad_bias, grad_output_c, grad_output_t = fp8_cast_transpose_bgrad_fused(
                grad_output_mat_no_fp8,
                ctx.fp8_meta["scaling_bwd"],
                tex.FP8BwdTensors.GRAD_OUTPUT1,
                fp8_dtype_backward,
            )
        else:
            if not ctx.fp8_meta["recipe"].override_linear_precision.wgrad:
                if isinstance(grad_output_mat, Float8Tensor):
                    grad_output_c = grad_output_mat
                    grad_output_t = grad_output_c.transpose_2d()
                else:
                    grad_output_c, grad_output_t = fp8_cast_transpose_fused(
                        grad_output_mat,
                        ctx.fp8_meta["scaling_bwd"],
                        tex.FP8BwdTensors.GRAD_OUTPUT1,
                        fp8_dtype_backward,
                    )
            else:
                grad_output_t = None
                if not isinstance(grad_output_mat, Float8Tensor):
                    grad_output_c = cast_to_fp8(
                        grad_output_mat,
                        ctx.fp8_meta["scaling_bwd"],
                        tex.FP8BwdTensors.GRAD_OUTPUT1,
                        fp8_dtype_backward,
                    )
                else:
                    grad_output_c = grad_output_mat
            grad_bias = None

        return grad_output_mat, grad_output_c, grad_output_t, grad_bias

    def get_fp8_weights_empty_tensors(
        self,
        is_first_microbatch: Union[bool, None],
    ) -> List[Float8Tensor]:
        """
        Returns empty tensors to be later used to store fp8 version of weights
        and their transposes (for the bwd pass) for this batch (or microbatch).
        When `is_first_microbatch` is `None`, this is especially useful since
        we then don't need to store the fp8 weights that are needed for one time
        only in the forward pass. Note that we still need to store the tensor
        for the fp8 weight transpose which is at least needed in the backward
        pass but that's taken care of by storing the transpose tensor in
        `ctx.save_for_backward`.
        """
        assert is_first_microbatch is None, "Should only be here when "\
                                            "`is_first_microbatch` is None!"
        fp8_weight_tensors = []
        for shape in self.fp8_weight_shapes:
            fp8_weight_tensors.append(
                Float8Tensor(
                    data=torch.empty(
                        shape,
                        device=torch.cuda.current_device(),
                        dtype=torch.uint8,
                    ),
                    fp8_dtype=tex.DType.kFloat8E4M3,
                    fp8_scale_inv=1,
                )
            )
            fp8_weight_tensors.append(
                Float8Tensor(
                    data=torch.empty(
                        shape[1],
                        shape[0],
                        device=torch.cuda.current_device(),
                        dtype=torch.uint8,
                    ),
                    fp8_dtype=tex.DType.kFloat8E4M3,
                    fp8_scale_inv=1,
                )
            )
        return fp8_weight_tensors

    def register_parameter(self, name, param, **kwargs):
        """
        Thin wrapper around PyTorch parameter registration to stash additional parameter
        metedata used in deferred initialization.
        """
        super().register_parameter(name, param)
        self.param_init_meta[name] = _ParameterInitMeta(**kwargs)

    def reset_parameters(self, defer_init: Optional[bool] = False) -> None:
        """
        Reset all module parameters to initial values. Unless deferred initialization
        is specified, all parameters on a 'meta' device are also materialized on a real cuda
        device before the values are reset to initial.
        """
        if defer_init:
            return

        for name, param in self.named_parameters(recurse=False):
            # Ensure parameter is on a real device
            if param.device == torch.device('meta'):
                param = torch.empty_like(param, device='cuda')

            # Initialize the parameter values on device
            init_fn = self.param_init_meta[name].init_fn
            get_rng_state_tracker = self.param_init_meta[name].get_rng_state_tracker
            if get_rng_state_tracker is None:
                init_fn(param)
            else:
                with get_rng_state_tracker().fork():
                    init_fn(param)

            # If primary weights are in fp8, wrap the parameter as Float8Tensor
            fp8_meta_index = self.param_init_meta[name].fp8_meta_index
            if self.primary_weights_in_fp8 and fp8_meta_index is not None:
                param = Float8Tensor.to_float8(
                    param,
                    fp8_meta=self.fp8_meta,
                    fp8_meta_index=fp8_meta_index,
                    amax=torch.empty(1, device="cuda"),  # Dummy amax to avoid overwriting history.
                )

            # Redo parameter wrap in case we broke it above
            # NOTE: Currently this can only be broken when primary weights are in Fp8 but
            #       re-applying the nn.Parameter() wrap is a no-op when the input is already
            #       a parameter so we always re-apply it just for extra safety.
            setattr(self, name, torch.nn.Parameter(param))

    @abstractmethod
    def forward(self):
        """Needs override."""

    @abstractmethod
    def get_fp8_weights_scratchpad(
        self,
        is_first_microbatch: Union[bool, None],
    ) -> List[torch.Tensor]:
        """Needs override."""
