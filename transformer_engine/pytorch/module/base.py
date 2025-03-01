# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Base modules and utilities for TransformerEngine PyTorch API"""
import io
import os
import pickle
import warnings
import socket
import fcntl
import struct
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union
from contextlib import contextmanager

import torch
import torch.nn.functional as F

import transformer_engine_torch as tex
from transformer_engine.common.recipe import Recipe

from ._common import _ParameterInitMeta
from ..fp8 import (
    MXFP8BlockScalingRecipeState,
    DelayedScalingRecipeState,
    FP8GlobalStateManager,
    RecipeState,
)
from ..distributed import (
    gather_along_first_dim,
    is_fp8_activation_recompute_enabled,
    in_fp8_activation_recompute_phase,
    _fsdp_gather_tensors,
)
from ..constants import dist_group_type
from ..tensor import QuantizedTensor, Quantizer
from ..tensor._internal.float8_tensor_base import Float8TensorBase
from ..tensor._internal.mxfp8_tensor_base import MXFP8TensorBase

__all__ = ["initialize_ub", "destroy_ub"]

_2X_ACC_FPROP = False
_2X_ACC_DGRAD = True
_2X_ACC_WGRAD = True
_multi_stream_cublas_workspace = []
_cublas_workspace = None
_ub_communicators = None
_NUM_MAX_UB_STREAMS = 3
_MIN_STREAM_PRIORITY, _MAX_STREAM_PRIORITY = None, None
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


def get_multi_stream_cublas_workspace() -> List[torch.Tensor]:
    """Returns workspace for multi-stream cublas."""
    global _multi_stream_cublas_workspace
    if not _multi_stream_cublas_workspace:
        for _ in range(tex._num_cublas_streams):
            _multi_stream_cublas_workspace.append(
                torch.empty(get_cublas_workspace_size_bytes(), dtype=torch.uint8, device="cuda")
            )
    return _multi_stream_cublas_workspace


def initialize_ub(
    shape: list,
    tp_size: int,
    use_fp8: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    ub_cfgs: Optional[dict] = None,
    bootstrap_backend: Union[str, torch.distributed.Backend] = None,
) -> None:
    r"""
    Initialize the Userbuffers communicator for overlapping tensor-parallel communications with
    GEMM compute in te.Linear, te.LayerNormLinear and te.LayerNormMLP modules.

    Parameters
    ----------
    shape : list
            shape of the communication buffer, typically set to be the same as the global shape of
            the input tensor to a te.TransformerLayer forward pass, with the sequence and batch
            dimensions collapsed together -- i.e.: `(sequence_length * batch_size, hidden_size)`
    tp_size : int
              number of GPUs in the tensor-parallel process group
    use_fp8 : bool = False
              allocate the communication buffer for FP8 GEMM inputs/outputs
    dtype : torch.dtype = torch.bfloat16
            non-FP8 data type of the communication buffer when `use_fp8 = False`
    ub_cfgs: dict = None
             Configuration dictionary with the structure
             ```
             {
                <gemm_name> : {
                    "method": <"ring_exchange" or "pipeline">,
                    "is_reduce_scatter": bool,
                    "num_sm": int,
                    "cga_size": int,
                    "set_sm_margin": bool,
                    "num_splits": int,
                    "aggregate": bool,
                    "atomic_gemm": bool,
                    "use_ce": bool,
                    "fp8_buf": bool,
                }
             }
             ```
             for `te.TransformerLayer` GEMM layers in `["qkv_fprop", "qkv_dgrad", "qkv_wgrad",
             "proj_fprop", "proj_dgrad", "proj_wgrad", "fc1_fprop", "fc1_dgrad", "fc2_dgrad",
             "fc2_fprop", "fc2_dgrad"]`.
    bootstrap_backend : str = None
                        `torch.distributed` communication backend for the all-gather, broadcast and
                        barrier collectives during Userbuffers initialization. Not all backends are
                        valid for every cluster configuration and distributed launch method even if
                        they are available in PyTorch. When left unset, the initialization prefers
                        to use the MPI backend, falling back first on Gloo and then NCCL if MPI is
                        not available. Setting `NVTE_UB_WITH_MPI=1` when building TE overrides this
                        option and always initializes Userbuffers with direct MPI calls in C++,
                        which also requires `MPI_HOME=/path/to/mpi/root` to be set at compile time.
    """
    if not tex.device_supports_multicast():
        assert bool(int(os.getenv("UB_SKIPMC", "0"))), (
            "CUDA device, driver and/or toolkit version does not support comm+GEMM overlap with "
            + "CUDA Multicast. Launch app with UB_SKIPMC=1 to try CUDA IPC instead."
        )

    global _ub_communicators
    assert _ub_communicators is None, "UB communicators are already initialized."
    _ub_communicators = {}

    if tex.ubuf_built_with_mpi():
        # We're bootstrapping with direct calls to MPI in Userbuffers code so we need to force
        # an MPI_Init() here by creating a new MPI process group...
        assert torch.distributed.is_mpi_available()
        _ = torch.distributed.new_group(backend="mpi")
        helper = tex.CommOverlapHelper()
    else:
        # Bootstrapping with torch.distributed API, so check backend and construct
        # intra/inter-node process groups...
        assert (
            torch.distributed.is_initialized()
        ), "torch.distributed must be initialized before Userbuffers"
        if bootstrap_backend is None:
            bootstrap_backend = "nccl"
            if torch.distributed.is_mpi_available():
                bootstrap_backend = "mpi"
            elif torch.distributed.is_gloo_available():
                bootstrap_backend = "gloo"
        else:
            assert bootstrap_backend in [
                "gloo",
                "mpi",
                "nccl",
            ], "Invalid torch.distributed backend for bootstrapping Userbuffers!"
            assert torch.distributed.is_backend_available(bootstrap_backend), (
                f"PyTorch must be compiled with '{bootstrap_backend}' support in order to "
                f"bootstrap Userbuffers with '{bootstrap_backend}' collectives."
            )

        world_group = torch.distributed.new_group(backend=bootstrap_backend)
        world_rank = torch.distributed.get_rank(world_group)
        world_size = torch.distributed.get_world_size(world_group)

        # We have single-node NVLink so we can color based on physical node hostnames.
        # NOTE: Prefer a network interface defined via the NVTE_UB_SOCKET_IFNAME variable, and
        #       otherwise fall back on NCCL_SOCKET_IFNAME or GLOO_SOCKET_IFNAME depending on
        #       the chosen bootstrap backend.
        mydomain = socket.gethostname()
        ifname = os.getenv(
            "NVTE_UB_SOCKET_IFNAME", os.getenv(f"{bootstrap_backend.upper()}_SOCKET_IFNAME")
        )
        if ifname is not None:
            # Make sure the ifname found in the environment is a valid network interface
            if ifname in [name for _, name in socket.if_nameindex()]:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    mydomain = socket.inet_ntoa(
                        fcntl.ioctl(
                            s.fileno(), 0x8915, struct.pack("256s", ifname[:15].encode("UTF-8"))
                        )[20:24]
                    )
                except OSError as err:
                    raise OSError(f"Invalid network interface: {ifname}") from err
                finally:
                    s.close()
            else:
                ifname_warning = (
                    f"'{ifname}' is not a valid network interface! `te.initialize_ub()` will"
                    + " attempt to detect ranks on the same node by matching "
                    + "'socket.gethostname()', which is known to fail on virtual clusters like "
                    + "Kubernetes. If Userbuffers initialization fails, please set the "
                    + "'NVTE_UB_SOCKET_IFNAME' variable in your environment to the correct network "
                    + "interface."
                )
                warnings.warn(ifname_warning, UserWarning)

        # Allgather the domain colors across ranks and reduce to a list of unique domains
        domain_per_rank_list = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(domain_per_rank_list, mydomain, world_group)
        unique_domains = []
        for domain in domain_per_rank_list:
            if domain not in unique_domains:
                unique_domains.append(domain)
        num_domains = len(unique_domains)

        if num_domains > 1:
            # DP/TP model replicated on multiple NVLink domains
            ranks_per_domain_list = [[] for _ in range(num_domains)]
            mydomain_idx = -1
            for i, domain in enumerate(domain_per_rank_list):
                domain_idx = unique_domains.index(domain)
                ranks_per_domain_list[domain_idx].append(i)
                if domain == mydomain:
                    mydomain_idx = domain_idx
            assert mydomain_idx >= 0, "Internal TE error!"

            intra_domain_group, _ = torch.distributed.new_subgroups_by_enumeration(
                ranks_per_domain_list, backend=bootstrap_backend
            )
            local_rank = torch.distributed.get_rank(intra_domain_group)
            intra_domain_ranks = torch.distributed.get_process_group_ranks(intra_domain_group)

            inter_domain_group, _ = torch.distributed.new_subgroups_by_enumeration(
                [list(ranks) for ranks in zip(*ranks_per_domain_list)],
                backend=bootstrap_backend,
            )

            helper = tex.CommOverlapHelper(world_group, intra_domain_group, inter_domain_group)

        else:
            # TP model on single NVLink domain, no replication, no data-parallelism
            mydomain_idx = 0
            local_rank = world_rank
            intra_domain_ranks = list(range(world_size))

            helper = tex.CommOverlapHelper(world_group)

        if world_rank == 0:
            print(f"!!! [UB] Number of NVLink domains: {num_domains}\n", end="", flush=True)
        if local_rank == 0:
            print(
                f"!!! [UB] Global ranks on domain {mydomain_idx}: {intra_domain_ranks}\n",
                end="",
                flush=True,
            )

    # Increase the workspace by the number of maximum concurrent streams
    global _cublas_workspace
    _cublas_workspace = get_workspace().repeat(_NUM_MAX_UB_STREAMS)

    # Default buffer precision: AllGather buffers use fp8 when using fp8 recipe
    layers_all_gather_overlap = [
        "qkv_fprop",
        "qkv_dgrad",
        "proj_dgrad",
        "fc1_fprop",
        "fc1_dgrad",
        "fc2_dgrad",
    ]
    layers_reduce_scatter_overlap = ["proj_fprop", "fc2_fprop", "qkv_wgrad", "fc1_wgrad"]
    dgrad_reduce_scatter_overlap = ["qkv_dgrad", "fc1_dgrad"]
    # Default overlap methods for layers
    methods = {
        "ring_exchange": ["qkv_fprop", "fc1_fprop", "proj_dgrad", "fc2_dgrad"],
        "pipeline": ["proj_fprop", "fc2_fprop"],
        "bulk": ["qkv_dgrad", "qkv_wgrad", "fc1_dgrad", "fc1_wgrad"],
    }

    # AG-RS overlap pairs of layers forming a tensor-parallel block
    ag_rs_pairs = {"qkv_fprop": "proj_fprop", "fc1_fprop": "fc2_fprop"}
    rs_ag_pairs = {v: k for k, v in ag_rs_pairs.items()}
    global layers_atomic_ring_exchange
    layers_atomic_ring_exchange = []

    def get_method(name):
        for method, names in methods.items():
            if name in names:
                return method
        raise KeyError(f"Given layer name {name} does not exist.")

    def get_default_config(name):
        global _MIN_STREAM_PRIORITY, _MAX_STREAM_PRIORITY
        method = get_method(name)
        is_reduce_scatter = name in layers_reduce_scatter_overlap
        if _MIN_STREAM_PRIORITY is None or _MAX_STREAM_PRIORITY is None:
            _MIN_STREAM_PRIORITY, _MAX_STREAM_PRIORITY = tex.get_stream_priority_range()
        default_cfg = {
            "method": method,
            "is_reduce_scatter": is_reduce_scatter,
            "num_sm": 1 if method == "ring_exchange" else 16,
            "cga_size": 1 if method == "ring_exchange" else 2,
            "set_sm_margin": not method == "ring_exchange",
            "num_splits": tp_size if method == "ring_exchange" else 4,
            "aggregate": False,
            "atomic_gemm": False,
            "use_ce": True,
            "fp8_buf": name in layers_all_gather_overlap,
            "comm_priority": _MAX_STREAM_PRIORITY,
            "gemm_priority": _MIN_STREAM_PRIORITY,
            "pipeline_rs_overlap_first_gemm": False,
        }
        return default_cfg

    def add_ub(
        name: str,
        method: str,
        is_reduce_scatter: bool,
        num_sm: int = 16,
        cga_size: int = 2,
        set_sm_margin: bool = False,
        num_splits: int = 0,
        aggregate: bool = False,
        atomic_gemm: bool = False,
        use_ce: bool = True,
        fp8_buf: bool = False,
        comm_priority: int = 0,
        gemm_priority: int = 0,
        pipeline_rs_overlap_first_gemm: bool = False,
    ) -> None:
        if atomic_gemm:
            warnings.warn(
                "Atomic GEMM uses a beta API from cublas and is not tested for all use cases."
            )
            assert use_fp8, "Atomic GEMM overlap supported only for FP8 GEMM."
            if method == "bulk":
                warnings.warn(
                    f"At {name}, atoimic GEMM not is supported for a bulk overlap."
                    "Defaulting to `atomic_gemm=False`."
                )
                atomic_gemm = 0
        if not is_reduce_scatter and method == "pipeline":
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

        buffer_dtype = torch.uint8 if (use_fp8 and fp8_buf) else dtype
        if method == "ring_exchange":
            ub_obj = tex.CommOverlapP2P(
                shape,  # Communication buffer shape
                buffer_dtype,  # Communication buffer data type
                helper,  # Helper for torch.distributed callbacks during bootstrapping
                tp_size,  # Tensor-parallel group size (may be different than local_size)
                tex.CommOverlapType.RS if is_reduce_scatter else tex.CommOverlapType.AG,
                num_max_streams=_NUM_MAX_UB_STREAMS,
                comm_cga_size=cga_size,
                num_comm_sm=num_sm,
                set_sm_margin=set_sm_margin,
                atomic_gemm=atomic_gemm,
                use_ce=use_ce,
                aggregate=aggregate,
                gemm_priority=gemm_priority,
                comm_priority=comm_priority,
            )
        else:
            ub_obj = tex.CommOverlap(
                shape,  # Communication buffer shape
                buffer_dtype,  # Communication buffer data type
                helper,  # Helper for torch.distributed callbacks during bootstrapping
                tp_size,  # Tensor-parallel group size (may be different than local_size)
                num_splits=num_splits,
                num_max_streams=_NUM_MAX_UB_STREAMS,
                comm_cga_size=cga_size,
                num_comm_sm=num_sm,
                set_sm_margin=set_sm_margin,
                atomic_gemm=atomic_gemm,
                gemm_priority=gemm_priority,
                comm_priority=comm_priority,
                rs_overlap_first_gemm=pipeline_rs_overlap_first_gemm,
            )
        _ub_communicators[name] = ub_obj

    if ub_cfgs is not None:
        for name in dgrad_reduce_scatter_overlap:
            if name in ub_cfgs and "method" in ub_cfgs[name] and ub_cfgs[name]["method"] != "bulk":
                wgrad_name = name.replace("dgrad", "wgrad")
                assert wgrad_name not in ub_cfgs
                layers_reduce_scatter_overlap.remove(wgrad_name)
                layers_all_gather_overlap.remove(name)
                layers_reduce_scatter_overlap.append(name)
                methods["bulk"].remove(name)
                new_method = ub_cfgs[name]["method"]
                methods[new_method].append(name)

    for name in methods["ring_exchange"] + methods["pipeline"] + methods["bulk"]:
        ub_cfg = get_default_config(name)
        if ub_cfgs is not None and name in ub_cfgs:
            fp8_buf = (name in layers_all_gather_overlap) or (
                ub_cfgs[name].get("fp8_buf", False) and name in methods["pipeline"]
            )
            ub_cfg.update(ub_cfgs[name])
            ub_cfg["fp8_buf"] = fp8_buf
        add_ub(name, **ub_cfg)


def get_ub(name: str):
    """Get userbuffer communicator corresponding to give key."""
    assert _ub_communicators is not None, "UB manager is not initialized."
    assert name in _ub_communicators, f"UB for {name} is not registered."
    return _ub_communicators[name]


def destroy_ub():
    """Destroy all allocated userbuffer communicators."""
    global _ub_communicators
    _ub_communicators = None
    global layers_atomic_ring_exchange
    layers_atomic_ring_exchange = []


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
        self.fp8_meta_tensors_initialized = False
        self.quantizers = {"scaling_fwd": {}, "scaling_bwd": {}}
        self.tp_group = None
        self.tp_size = 1
        self.sequence_parallel = False
        self.param_init_meta = {}
        self.primary_weights_in_fp8 = FP8GlobalStateManager.with_fp8_parameters()
        self.fsdp_wrapped = False
        self.fsdp_group = None
        self._fp8_workspaces: Dict[str, QuantizedTensor] = {}
        self.activation_dtype: Optional[torch.dtype] = None

    # Names of attributes that can be set quickly (see __setattr__
    # method)
    _fast_setattr_names: Set[str] = {
        "activation_dtype",
        "fp8",
        "fp8_initialized",
        "fp8_calibration",
        "fp8_parameters",
    }

    def __setattr__(self, name: str, value: Any) -> None:
        if name in TransformerEngineBaseModule._fast_setattr_names:
            # torch.nn.Module has a custom __setattr__ that handles
            # modules, parameters, and buffers. This is unnecessary
            # overhead when setting plain attrs.
            self.__dict__[name] = value
        else:
            # Default case
            super().__setattr__(name, value)

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
                    self.fp8_meta[meta_key].amax_history[:length].clone()
                )
            elif length > curr_len:
                extra_rows = length - curr_len
                self.fp8_meta[meta_key].amax_history = F.pad(
                    self.fp8_meta[meta_key].amax_history, pad=(0, 0, 0, extra_rows)
                )

            # Update quantizers with new amax pointers.
            self.quantizers[meta_key] = self.fp8_meta[meta_key].make_quantizers()

            # Update the global buffers with new amax and history pointers.
            if FP8GlobalStateManager.get_buffer_info() in self.fp8_meta:
                fwd_pos, fwd_key, bwd_pos, bwd_key = self.fp8_meta[
                    FP8GlobalStateManager.get_buffer_info()
                ]
                for pos, buffer_key in zip((fwd_pos, bwd_pos), (fwd_key, bwd_key)):
                    if buffer_key in FP8GlobalStateManager.global_amax_buffer:
                        assert (
                            buffer_key in FP8GlobalStateManager.global_amax_history_buffer
                        ), "TE internal error during amax history change."
                        FP8GlobalStateManager.global_amax_buffer[buffer_key][pos] = self.fp8_meta[
                            meta_key
                        ].amax_history[0]
                        FP8GlobalStateManager.global_amax_history_buffer[buffer_key][pos] = (
                            self.fp8_meta[meta_key].amax_history
                        )

    def set_meta_tensor(self, fwd: bool, recipe: Recipe) -> None:
        """Init scales and amaxes for fwd | bwd."""
        fp8_meta_tensor_key = "scaling_fwd" if fwd else "scaling_bwd"

        # Return early if recipe state matches recipe
        if self.fp8_meta_tensors_initialized:
            recipe_state = self.fp8_meta[fp8_meta_tensor_key]
            if recipe.delayed() and isinstance(recipe_state, DelayedScalingRecipeState):
                self.adjust_amax_history_length(recipe.amax_history_len, fwd=fwd)
                return
            if recipe.mxfp8() and isinstance(recipe_state, MXFP8BlockScalingRecipeState):
                return

        # Max. number of fp8 tensors per GEMM = 3 (input, weight, output) for fwd and
        # 2 (grad_output and grad_input) for bwd
        num_fp8_tensors = self.fp8_meta["num_gemms"] * 3 if fwd else self.fp8_meta["num_gemms"] * 2

        # Initialize recipe state and quantizers
        recipe_state = RecipeState.create(
            recipe,
            mode=("forward" if fwd else "backward"),
            num_quantizers=num_fp8_tensors,
        )

        self.fp8_meta[fp8_meta_tensor_key] = recipe_state
        self.quantizers[fp8_meta_tensor_key] = recipe_state.make_quantizers()

    def init_fp8_meta_tensors(self, recipe: Recipe) -> None:
        """Init scales and amaxes."""
        self.set_meta_tensor(True, recipe)
        self.set_meta_tensor(False, recipe)

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
                fp8_meta_tensors[key].append(self.fp8_meta[key].amax_history.clone())
        return fp8_meta_tensors

    def reset_fp8_meta_tensors(self, fp8_meta_tensors=None) -> None:
        """Reset scales and amaxes."""

        def reset(key):
            if key in self.fp8_meta:
                if fp8_meta_tensors is None:
                    self.fp8_meta[key].scale.copy_(torch.ones_like(self.fp8_meta[key].scale))
                    self.fp8_meta[key].amax_history.copy_(
                        torch.zeros_like(self.fp8_meta[key].amax_history)
                    )
                else:
                    assert key in fp8_meta_tensors, "Cannot reset fp8 tensors."
                    self.fp8_meta[key].scale.copy_(fp8_meta_tensors[key][0])
                    self.fp8_meta[key].amax_history.copy_(fp8_meta_tensors[key][1])

        with torch.no_grad():
            reset("scaling_fwd")
            reset("scaling_bwd")

    def get_extra_state(self) -> torch.Tensor:
        """Save before checkpointing."""

        # This implementation is working around a few issues:
        #
        # (1) PyTorch's "extra state" infrastructure might be able to
        #     support any picklable type, but they make no guarantees.
        #     We have experienced problems (e.g. in ONNX export) with
        #     non-tensor extra state.
        # (2) PyTorch's checkpointing infrastructure does not remap
        #     devices for "extra state" like it does for "state dict".
        #     Thus, we want to avoid putting extra state on the GPU
        #     since it may be loaded on the wrong device.
        # (3) The extra state consists of many small tensors. If we
        #     want to copy them all to CPU, then we need to avoid the
        #     overhead of many GPU-CPU memory transfers.
        #
        # See: https://github.com/NVIDIA/TransformerEngine/pull/351
        # See: https://github.com/NVIDIA/TransformerEngine/pull/363

        def to_cpu(src: torch.Tensor) -> torch.Tensor:
            """Helper function to make CPU copy of tensor

            Memory transfer is asynchronous w.r.t. host, so GPU should
            be synchronized before using result.

            """
            dst = torch.empty_like(src, device="cpu")
            dst.copy_(src, non_blocking=True)
            return dst

        # Store FP8 state if needed
        state = None
        fp8_checkpoint = self.fp8_meta["fp8_checkpoint"] or self.fp8 or self.fp8_calibration
        if fp8_checkpoint:

            # Copy tensors to CPU and store
            state = {}
            state["recipe"] = self.fp8_meta["recipe"]
            if state["recipe"].delayed():
                state["scale_fwd"] = to_cpu(self.fp8_meta["scaling_fwd"].scale)
                state["amax_history_fwd"] = to_cpu(self.fp8_meta["scaling_fwd"].amax_history)
                state["scale_bwd"] = to_cpu(self.fp8_meta["scaling_bwd"].scale)
                state["amax_history_bwd"] = to_cpu(self.fp8_meta["scaling_bwd"].amax_history)

            # Store other pickelable values
            extra = {}
            for k, v in self.fp8_meta.items():
                if k != "buffer_index_and_autocast_key" and isinstance(
                    v, (bool, int, float, str, tuple, list)
                ):
                    extra[k] = v
            state["extra_fp8_variables"] = extra

        # Serialize state into byte tensor
        torch.cuda.synchronize()
        state_serialized = bytearray(pickle.dumps(state))
        state_serialized = torch.frombuffer(state_serialized, dtype=torch.uint8)
        return state_serialized

    def set_extra_state(self, state: torch.Tensor) -> None:
        """Load previous state."""
        if state is None:
            return

        # Load state
        if isinstance(state, torch.Tensor):
            # Default format: byte tensor with pickled data
            state = pickle.loads(state.detach().cpu().numpy().tobytes())
        elif isinstance(state, io.BytesIO):
            # Deprecated format with io.BytesIO
            state.seek(0)
            state = torch.load(state, map_location="cuda")
        else:
            raise RuntimeError("Unsupported checkpoint format.")

        if state is None:
            return

        # Load extra items
        self.fp8_meta.update(state["extra_fp8_variables"])
        self.fp8_meta["recipe"] = state["recipe"]
        if "global_fp8_buffer_pos_fwd_recompute" in self.fp8_meta:
            del self.fp8_meta["global_fp8_buffer_pos_fwd_recompute"]

        # Initialize before loading
        self.init_fp8_meta_tensors(self.fp8_meta["recipe"])

        def copy_tensor(src: torch.Tensor, dst: torch.Tensor) -> None:
            """Helper function to copy tensor from CPU

            Memory transfer is asynchronous w.r.t. host, so GPU should
            be synchronized before using result.

            """
            dst.copy_(src, non_blocking=True)

        # Load tensors
        if self.fp8_meta["recipe"].delayed():
            copy_tensor(state["scale_fwd"], self.fp8_meta["scaling_fwd"].scale)
            copy_tensor(state["amax_history_fwd"], self.fp8_meta["scaling_fwd"].amax_history)
            copy_tensor(state["scale_bwd"], self.fp8_meta["scaling_bwd"].scale)
            copy_tensor(state["amax_history_bwd"], self.fp8_meta["scaling_bwd"].amax_history)
        torch.cuda.synchronize()

    def set_activation_dtype(self, inp: torch.Tensor) -> None:
        """Get activation data type for AMP."""
        # Native AMP (`torch.autocast`) gets highest priority
        if torch.is_autocast_enabled():
            self.activation_dtype = torch.get_autocast_gpu_dtype()
            return

        # All checks after this have already been performed once, thus skip
        if self.activation_dtype == inp.dtype:
            return

        dtype = inp.dtype
        for name, param in self.named_parameters():
            if param is not None:
                assert dtype == param.dtype, (
                    "Data types for parameters must match when outside of autocasted region. "
                    f" Found input dtype: {dtype} and {name!r} dtype: {param.dtype}"
                )
        self.activation_dtype = dtype

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
        for param in self.parameters(recurse=False):
            if isinstance(param, QuantizedTensor) and param.requires_grad:
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
        fp8_enabled = self.fp8 or self.fp8_calibration
        self.fp8_meta["fp8_checkpoint"] = self.fp8 or self.fp8_calibration

        if self.fp8_parameters or fp8_enabled:
            if (
                self.fp8_initialized
                and FP8GlobalStateManager.get_fp8_recipe() == self.fp8_meta["recipe"]
            ):
                # FP8 init has already been run and recipe is the same, don't do anything.
                return
            self.fp8_meta["recipe"] = FP8GlobalStateManager.get_fp8_recipe()
        else:
            # If fp8 isn't enabled, turn off and return.
            self.fp8_initialized = False
            return

        if self.fp8_parameters and not self.fp8_initialized:
            self.fp8_meta["num_gemms"] = num_gemms
            self.init_fp8_meta_tensors(self.fp8_meta["recipe"])

        if fp8_enabled:
            # Set FP8 and other FP8 metadata
            self.fp8_meta["num_gemms"] = num_gemms
            self.fp8_meta["fp8_group"] = FP8GlobalStateManager.get_fp8_group()

            # Set FP8_MAX per tensor according to recipe
            self.fp8_meta["fp8_max_fwd"] = self.fp8_meta["recipe"].fp8_format.value.max_fwd
            self.fp8_meta["fp8_max_bwd"] = self.fp8_meta["recipe"].fp8_format.value.max_bwd

            # Allocate scales and amaxes
            self.init_fp8_meta_tensors(self.fp8_meta["recipe"])
            self.fp8_initialized = True

            self.fp8_meta["recipe"] = FP8GlobalStateManager.get_fp8_recipe()

    @contextmanager
    def prepare_forward(
        self,
        inp: torch.Tensor,
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

            if self.fp8 and self.sequence_parallel and self.fp8_meta["recipe"].delayed():
                assert self.fp8_meta["recipe"].reduce_amax, (
                    "Amax reduction across tensor parallel group is "
                    "necessary when using sequence parallelism with FP8."
                )

            if self.fp8 and not FP8GlobalStateManager.fp8_graph_capturing():
                FP8GlobalStateManager.add_fp8_tensors_to_global_buffer(self.fp8_meta)

            # Activation recomputation is used and this is the first forward phase.
            if self.fp8 and self.training and is_fp8_activation_recompute_enabled():
                FP8GlobalStateManager.copy_forward_fp8_meta_tensors_for_recompute(self.fp8_meta)

        with torch.cuda.nvtx.range(self.__class__.__name__ + " forward"):
            if not allow_non_contiguous and not inp.is_contiguous():
                inp = inp.contiguous()
            yield inp

        if self.fp8 and in_fp8_activation_recompute_phase():
            FP8GlobalStateManager.restore_fp8_meta_tensors(self.fp8_meta)

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
        ctx,
        grad_output: torch.Tensor,
        row_parallel_mode: bool,
        quantizer: Optional[Quantizer],
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        """Utility function for backward.
        Returns tuple in order (all optional/None based on training precion/recipe):
            R1: gathered `grad_output`.
            R2: bias gradient on R1.

        """
        grad_output = grad_output.reshape((-1, grad_output.shape[-1]))
        grad_output = grad_output.contiguous()
        gather_grad_output = row_parallel_mode and ctx.sequence_parallel

        # Non-FP8 case: bgrad is fused with wgrad for this case.
        if not ctx.fp8:
            if gather_grad_output:
                if not ctx.ub_overlap_ag:
                    grad_output, _ = gather_along_first_dim(grad_output, ctx.tp_group)
                else:
                    ctx.ub_obj_gradout.copy_into_buffer(grad_output, quantizer, local_chunk=True)
                    grad_output = ctx.ub_obj_gradout.get_buffer(quantizer)
            return grad_output, None

        # FP8 with all-gather: unfused bgrad, fused cast + transpose
        if gather_grad_output:
            grad_bias = None
            if ctx.use_bias:
                grad_bias = grad_output.view(-1, grad_output.shape[-1]).sum(dim=0)
            if ctx.ub_overlap_ag:
                # Quantize the gradient if needed
                if not isinstance(
                    grad_output, (QuantizedTensor, Float8TensorBase, MXFP8TensorBase)
                ):
                    grad_output = quantizer(grad_output)

                # Copy into communication buffer, and replace original gradient with it
                ctx.ub_obj_gradout.copy_into_buffer(grad_output, quantizer, local_chunk=True)
                grad_output = ctx.ub_obj_gradout.get_buffer(quantizer)
            else:
                grad_output, _ = gather_along_first_dim(
                    grad_output,
                    ctx.tp_group,
                    quantizer=quantizer,
                )
            return grad_output, grad_bias

        # FP8 without all-gather: fused bgrad + cast + transpose
        grad_bias = None
        if ctx.use_bias:
            if isinstance(grad_output, (QuantizedTensor, Float8TensorBase, MXFP8TensorBase)):
                grad_bias = grad_output.dequantize().view(-1, grad_output.shape[-1]).sum(dim=0)
            else:
                grad_bias, grad_output = tex.bgrad_quantize(grad_output, quantizer)
        if not isinstance(grad_output, (QuantizedTensor, Float8TensorBase, MXFP8TensorBase)):
            grad_output = quantizer(grad_output)
        return grad_output, grad_bias

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
            if param.device == torch.device("meta"):
                param = torch.empty_like(param, device="cuda")

            # Initialize the parameter values on device
            init_fn = self.param_init_meta[name].init_fn
            get_rng_state_tracker = self.param_init_meta[name].get_rng_state_tracker
            if get_rng_state_tracker is None:
                init_fn(param)
            else:
                if hasattr(self, "rng_tracker_name") and self.rng_tracker_name:
                    with get_rng_state_tracker().fork(self.rng_tracker_name):
                        init_fn(param)
                else:
                    with get_rng_state_tracker().fork():
                        init_fn(param)

            # If primary weights are in fp8, wrap the parameter as FP8Tensor
            fp8_meta_index = self.param_init_meta[name].fp8_meta_index
            if self.primary_weights_in_fp8 and fp8_meta_index is not None:
                quantizer = self.quantizers["scaling_fwd"][fp8_meta_index]
                assert (
                    quantizer is not None
                )  # to use primary fp8 weight one needs to use FP8 autocast with specific recipe.
                quantizer.internal = False
                param = quantizer(param)

            # Redo parameter wrap in case we broke it above
            # NOTE: Currently this can only be broken when primary weights are in Fp8 but
            #       re-applying the nn.Parameter() wrap is a no-op when the input is already
            #       a parameter so we always re-apply it just for extra safety.
            setattr(self, name, torch.nn.Parameter(param))

    @abstractmethod
    def forward(self):
        """Needs override."""

    def get_weight_workspace(
        self,
        *,
        tensor: Optional[torch.Tensor] = None,
        quantizer: Optional[Quantizer] = None,
        cache_name: Optional[str] = None,
        update_workspace: bool = True,
        skip_update_flag: Optional[torch.Tensor] = None,
        fsdp_group: Optional[dist_group_type] = None,
    ) -> QuantizedTensor:
        """Get FP8 workspace buffer and maybe update its values

        The workspace buffer may be cached for future function calls.

        Parameters
        ----------
        tensor : torch.Tensor, optional
            Values to copy into workspace. Required if the workspace
            is being constructed or updated.
        quantizer: Quantizer, optional
            Quantizer used to cast the weights. Required if the
            workspace is being constructed or updated.
        cache_name: str, optional
            Key for caching.
        update_workspace: bool, default = `True`
            Update workspace with values from `tensor`.
        skip_update_flag: torch.Tensor, optional
            GPU flag to skip updating the workspace. Take precedence
            over `update_workspace` if provided.
        fsdp_group: bool, default = None
            FSDP process group that the weights are distributed over.
        """

        # Try getting workspace from cache
        out = None
        if cache_name is not None:
            out = self._fp8_workspaces.get(cache_name, None)

        # Gather cached Fp8 workspace if it's distributed
        # NOTE: FSDP sharding is supported only for Fp8 buffers and will not work
        #       for models initialized with Fp8 primary weights.
        if (
            out is not None
            and tensor is not None
            and fsdp_group is not None
            and out.data.shape != tensor.data.shape
        ):
            _fsdp_gather_tensors(fsdp_group, [tensor.data.shape], out)

        # Construct workspace if needed
        if out is None:
            if tensor is None or quantizer is None:
                raise ValueError(
                    "tensor and quantizer kwargs must be provided to construct FP8 workspace"
                )
            out = quantizer(tensor)

            # Update cache
            if cache_name is not None:
                self._fp8_workspaces[cache_name] = out
            return out

        # Update workspace if needed
        if skip_update_flag is not None:
            update_workspace = True
        if update_workspace:
            if tensor is None:
                raise ValueError("tensor kwarg must be provided to update FP8 workspace")
            if hasattr(out, "quantize_"):
                out.quantize_(tensor, noop_flag=skip_update_flag)
            else:
                tex.quantize(tensor, quantizer, out, skip_update_flag)

        return out

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """
        This function loads tensors and extra state including fp8 metadata.
        This metadata is essential for copying fp8 tensors, as the copy_ function
        uses the scale_inv parameter from fp8_meta to set the correct scaling factor
        for the new tensor.
        Hence, this extra state must be loaded before the tensor copying process,
        not after, as is typically done in _load_from_state_dict.
        Tensors are copied into fp8 tensors only when self.primary_weights_in_fp8=True,
        otherwise, this behavior is not required.
        """
        if self.primary_weights_in_fp8:
            extra_state_key = prefix + torch.nn.modules.module._EXTRA_STATE_KEY_SUFFIX
            if extra_state_key in state_dict:
                self.set_extra_state(state_dict[extra_state_key])
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
