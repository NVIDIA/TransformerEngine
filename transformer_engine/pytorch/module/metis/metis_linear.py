# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Linear API"""
from typing import Any, Callable, Dict, Optional, Tuple, Union, List
from functools import reduce, partial
from operator import mul as multiply_op
import warnings

import torch

from transformer_engine.pytorch.distributed import dist_group_type
from transformer_engine.pytorch.transformer import TransformerEngineBaseModule
import transformer_engine_torch as tex
import debugpy

# from ....debug.pytorch.debug_state import TEDebugState
from .metis_context import LinearLowbitContext

__all__ = ["MetisLinear"]

class MetisLinear(TransformerEngineBaseModule):
    """Applies a linear transformation to the incoming data :math:`y = xA^T + b`

    On NVIDIA GPUs it is a drop-in replacement for `torch.nn.Linear`.

    Parameters
    ----------
    in_features : int
                 size of each input sample.
    out_features : int
                  size of each output sample.
    bias : bool, default = `True`
          if set to `False`, the layer will not learn an additive bias.
    init_method : Callable, default = `None`
                 used for initializing weights in the following way: `init_method(weight)`.
                 When set to `None`, defaults to `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    get_rng_state_tracker : Callable, default = `None`
                 used to get the random number generator state tracker for initializing weights.
    rng_tracker_name : str, default = `None`
                 the param passed to get_rng_state_tracker to get the specific rng tracker.
    parameters_split : Optional[Union[Tuple[str, ...], Dict[str, int]]], default = None
                      Configuration for splitting the weight and bias tensors along dim 0 into
                      multiple PyTorch parameters. If a list or tuple of strings is provided,
                      they are used to make the names of equally-sized parameters. If a dict
                      (preferably an OrderedDict) is provided, the keys are used as names and
                      values as split sizes along dim 0. The resulting parameters will have
                      names that end in `_weight` or `_bias`, so trailing underscores are
                      stripped from any provided names.
    device : Union[torch.device, str], default = "cuda"
          The device on which the parameters of the model will be allocated. It is the user's
          responsibility to ensure all parameters are moved to the GPU before running the
          forward pass.
    name: str, default = `None`
        name of the module, currently used for debugging purposes.

    Parallelism parameters
    ----------------------
    sequence_parallel : bool, default = `False`
                       if set to `True`, uses sequence parallelism.
    tp_group : ProcessGroup, default = `None`
              tensor parallel process group.
    tp_size : int, default = 1
             used as TP (tensor parallel) world size when TP groups are not formed during
             initialization. In this case, users must call the
             `set_tensor_parallel_group(tp_group)` method on the initialized module before the
             forward pass to supply the tensor parallel group needed for tensor and sequence
             parallel collectives.
    parallel_mode : {None, 'column', 'row'}, default = `None`
                   used to decide whether this Linear layer is Column Parallel Linear or Row
                   Parallel Linear as described `here <https://arxiv.org/pdf/1909.08053.pdf>`_.
                   When set to `None`, no communication is performed.

    Optimization parameters
    -----------------------
    fuse_wgrad_accumulation : bool, default = 'False'
                             if set to `True`, enables fusing of creation and accumulation of
                             the weight gradient. When enabled, it is assumed that the weights
                             have an additional `main_grad` attribute (used instead of the
                             regular `grad`) which is a pre-allocated buffer of the correct
                             size to accumulate gradients in. This argument along with
                             weight tensor having attribute 'overwrite_main_grad' set to True
                             will overwrite `main_grad` instead of accumulating.
    return_bias : bool, default = `False`
                 when set to `True`, this module will not apply the additive bias itself, but
                 instead return the bias value during the forward pass together with the
                 output of the linear transformation :math:`y = xA^T`. This is useful when
                 the bias addition can be fused to subsequent operations.
    params_dtype : torch.dtype, default = `torch.get_default_dtype()`
                  it controls the type used to allocate the initial parameters. Useful when
                  the model is trained with lower precision and the original FP32 parameters
                  would not fit in GPU memory.
    delay_wgrad_compute : bool, default = `False`
                         Whether or not to delay weight gradient computation. If set to `True`,
                         it's the user's responsibility to call `module.backward_dw` to compute
                         weight gradients.
    symmetric_ar_type : {None, 'multimem_all_reduce', 'two_shot', 'one_shot'}, default = None
                   Type of symmetric memory all-reduce to use during the forward pass.
                   This can help in latency bound communication situations.
                   Requires PyTorch version 2.7.0 or higher. When set to None, standard all-reduce
                   is used.
    save_original_input : bool, default = `False`
                       If set to `True`, always saves the original input tensor rather than the
                       cast tensor. In some scenarios, the input tensor is used by multiple modules,
                       and saving the original input tensor may reduce the memory usage.
                       Cannot work with FP8 DelayedScaling recipe.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sequence_parallel: bool = False,
        fuse_wgrad_accumulation: bool = False,
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        rng_tracker_name: Optional[str] = None,
        init_method: Optional[Callable] = None,
        bias: bool = True,
        return_bias: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        parallel_mode: Optional[str] = None,
        parameters_split: Optional[Union[Tuple[str, ...], Dict[str, int]]] = None,
        device: Union[torch.device, str] = "cuda",
        ub_overlap_ag: bool = False,
        ub_overlap_rs: bool = False,
        ub_overlap_rs_dgrad: bool = False,
        ub_bulk_dgrad: bool = False,
        ub_bulk_wgrad: bool = False,
        ub_name: Optional[str] = None,
        delay_wgrad_compute: bool = False,
        symmetric_ar_type: Optional[str] = None,
        save_original_input: bool = False,
        name: Optional[str] = None,
    ) -> None:
        # print("current LinearLowbitContext=", LinearLowbitContext())
        from transformer_engine.pytorch.module.linear import Linear  # avoid circular import
        super().__init__()
        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.in_features = in_features
        self.out_features = out_features
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.use_bias = bias
        self.return_bias = return_bias
        self.apply_bias = bias and not return_bias
        self.get_rng_state_tracker = get_rng_state_tracker
        self.rng_tracker_name = rng_tracker_name
        self.symmetric_ar_type = symmetric_ar_type
        self.save_original_input = save_original_input
        self.name = name
        self.weight_svd_has_initialized = False
        self.commonMetisSvdFunction_args = {
            "return_bias": return_bias,
            "get_rng_state_tracker": get_rng_state_tracker,
            "rng_tracker_name": rng_tracker_name,
            "parameters_split": parameters_split,
            "device": device,
            "params_dtype": params_dtype,
            "sequence_parallel": sequence_parallel,
            "tp_group": tp_group,
            "tp_size": tp_size,
            "parallel_mode": parallel_mode,
            "fuse_wgrad_accumulation": fuse_wgrad_accumulation,
            "ub_overlap_ag": ub_overlap_ag,
            "ub_overlap_rs": ub_overlap_rs,
            "ub_overlap_rs_dgrad": ub_overlap_rs_dgrad,
            "ub_bulk_dgrad": ub_bulk_dgrad,
            "ub_bulk_wgrad": ub_bulk_wgrad,
            "ub_name": ub_name,
            "delay_wgrad_compute": delay_wgrad_compute,
            "symmetric_ar_type": symmetric_ar_type,
            "save_original_input": save_original_input,
            "name": name,
        }
        # print("Metis linear==",LinearLowbitContext())
        if LinearLowbitContext.enable_lowbit and not LinearLowbitContext.enable_weight_svd:
            # only quantize activation
            self.linear_residual = Linear(in_features,out_features,bias=bias,enable_metis=True,init_method=init_method,**self.commonMetisSvdFunction_args)
        else:
            # only quantize weight
            self.linear_residual = Linear(in_features,out_features,bias=bias,enable_metis=False,init_method=init_method,**self.commonMetisSvdFunction_args)
        # debugpy.breakpoint()
        if LinearLowbitContext.enable_weight_svd:

            self.weight_svd_decomposition()
            # print(super(self.vlinear.weight,).__repr__)
            # print(self.vlinear)

    @torch.no_grad()
    def initialize_weight_svd_decomposition(self):
        from transformer_engine.pytorch.module.linear import Linear  # avoid circular import
        device = self.linear_residual.weight.device
        weight_fp32 = self.linear_residual.weight.float()
        u, s, v = torch.linalg.svd(weight_fp32, full_matrices=False)
        u = u.to(device = self.linear_residual.weight.get_device(),dtype=self.linear_residual.weight.dtype)
        s = s.to(device = self.linear_residual.weight.get_device(),dtype=self.linear_residual.weight.dtype)
        v = v.to(device = self.linear_residual.weight.get_device(),dtype=self.linear_residual.weight.dtype)
        if self.use_bias:
            bias = self.linear_residual.bias.to(device=device)
        else:
            bias = None
        w = self.linear_residual.weight.to(device=device)
        # forward svd low rank
        if LinearLowbitContext.forward_svd_rank > 0:
            self.linear_residual = Linear(
                self.linear_residual.weight.shape[1], 
                self.linear_residual.weight.shape[0],
                bias=True if not bias is None else False, 
                **self.commonMetisSvdFunction_args
            )
                # device=device
            if not bias is None:
                self.linear_residual.bias.copy_(bias)
            self.linear_residual.weight.copy_(
                w - \
                u[:,LinearLowbitContext.forward_svd_rank:] @ \
                torch.diag(s[LinearLowbitContext.forward_svd_rank:]) @ \
                v[LinearLowbitContext.forward_svd_rank:]
                )
        self.weight_svd_has_initialized = True
        return u,s,v,bias

    @torch.no_grad()
    def update_weight_svd_decomposition(self):
        assert self.weight_svd_has_initialized
        weight_fp32 = (self.ulinear.weight @ torch.diag(self.s) @  self.vlinear.weight).float()
        u, s, v = torch.linalg.svd(
            weight_fp32, full_matrices=False)
        u = u.to(device = self.linear_residual.weight.get_device(),dtype=self.linear_residual.weight.dtype)
        s = s.to(device = self.linear_residual.weight.get_device(),dtype=self.linear_residual.weight.dtype)
        v = v.to(device = self.linear_residual.weight.get_device(),dtype=self.linear_residual.weight.dtype)
        bias = self.ulinear.bias
        return u,s,v,bias

    @staticmethod
    @torch.no_grad()
    def init_tensor_with_data(source_tensor:torch.Tensor, weight_tensor:torch.Tensor):
        weight_tensor.copy_(source_tensor.detach())

    def weight_svd_decomposition(self):
        print("start weight_svd_decomposition")
        from transformer_engine.pytorch.module.linear import Linear # avoid circular import
        if not self.weight_svd_has_initialized:
          u,s,v,bias = self.initialize_weight_svd_decomposition()
        else:
          u,s,v,bias = self.update_weight_svd_decomposition()
        if LinearLowbitContext.enable_lowbit: 
            # nv fp8
            # ******************************************************************
            # self.ss = u @ s @ u.transpose()
            # with fp8_model_init(enabled=True):
            #     self.uvlinear = te.Linear(
            #         self.linear_residual.weight.shape[1], 
            #         self.linear_residual.weight.shape[0], 
            #         init_method=partial(BitLinear._init_telinear, u @ v), 
            #         bias=False, 
            #         device=self.device
            #     )
            if LinearLowbitContext.forward_svd_rank > 0:
                self.vlinear = Linear(
                    v.shape[1], 
                    LinearLowbitContext.forward_svd_rank, # v.shape[0] // 30, 
                    init_method=partial(MetisLinear.init_tensor_with_data,v[: LinearLowbitContext.forward_svd_rank, :]),
                    bias = False,
                    enable_metis=True,
                    **self.commonMetisSvdFunction_args
                    )
                self.ulinear = Linear(
                    LinearLowbitContext.forward_svd_rank, # u.shape[1] // 30,
                    u.shape[0], 
                    bias=False,
                    init_method=partial(MetisLinear.init_tensor_with_data,u[:, : LinearLowbitContext.forward_svd_rank]), 
                    **self.commonMetisSvdFunction_args,
                )
                # self.vlinear.weight.copy_(v[: LinearLowbitContext.forward_svd_rank, :])
                # self.ulinear.weight.copy_(u[:, : LinearLowbitContext.forward_svd_rank])
            else:
                self.vlinear = Linear(
                    v.shape[1], 
                    v.shape[0], # v.shape[0] // 30, 
                    init_method=partial(MetisLinear.init_tensor_with_data,v),
                    bias=False,
                    enable_metis=True,
                    **self.commonMetisSvdFunction_args,
                    )
                self.ulinear = Linear(
                    u.shape[1], # u.shape[1] // 30, 
                    u.shape[0], 
                    init_method=partial(MetisLinear.init_tensor_with_data,u),
                    bias=False,
                    **self.commonMetisSvdFunction_args,
                )
                # self.vlinear.weight.copy_(v)
                # self.ulinear.weight.copy_(u)


            # forward svd low rank
            if LinearLowbitContext.forward_svd_rank > 0 and bias :
                self.ulinear.bias.copy_(bias)
        else:
            self.vlinear = Linear(
                v.shape[1], 
                v.shape[0], 
                init_method=partial(MetisLinear.init_tensor_with_data,v),
                bias=False,
                **self.commonMetisSvdFunction_args),
            self.ulinear = Linear(
                u.shape[1],
                u.shape[0],
                bias=False,
                init_method=partial(MetisLinear.init_tensor_with_data,u),
                **self.commonMetisSvdFunction_args)

            # self.vlinear.weight.data = v.data
            # self.ulinear.weight.data = u.data
            if bias:
                self.ulinear.bias.data = self.linear_residual.bias.data
                #     self.linear_residual.bias.clone().cuda(self.linear_residual.weight.get_device())
                # )
        
        if LinearLowbitContext.forward_svd_rank > 0:
            self.register_parameter(
                "s", 
                torch.nn.Parameter(s[:LinearLowbitContext.forward_svd_rank]),
            )
        else:
            self.register_parameter(
                "s", 
                torch.nn.Parameter(s),
            )
            # self.linear_residual = None
    def forward(self,
        inp: torch.Tensor,
        **kvargs
        ) -> torch.Tensor:
        
        if LinearLowbitContext.enable_weight_svd:
            y = self.vlinear(inp,**kvargs)
            y = torch.mul(self.s, y)
            y = self.ulinear(y,**kvargs)
            if LinearLowbitContext.forward_svd_rank > 0:
                y += self.linear_residual(inp,**kvargs)
            
        else:
            y = self.linear_residual(inp,**kvargs)
        
        return y

    def __repr__(self):
        # 基础信息
        header = (
            f"{type(self).__name__}(\n"
            f"  in_features={self.in_features},\n"
            f"  out_features={self.out_features},\n"
            f"  bias={self.use_bias},\n"
            f"  enable_weight_svd={self.weight_svd_has_initialized},\n"
            f"  name={repr(self.name)}"
        )

        # SVD 已初始化的情况
        if self.weight_svd_has_initialized:
            svd_rank = self.s.shape[0]
            header += f",\n  weight_svd_rank={svd_rank}"
            lines = [header + ",\n"]

            def indent(text, n=4):
                pad = " " * n
                return pad + text.replace("\n", "\n" + pad)

            # lines.append("  submodules:\n")

            # 主分支（线性残差）
            if hasattr(self, "linear_residual"):
                lines.append(indent(f"linear_residual: {repr(self.linear_residual)} # enable_metis={self.linear_residual.enable_metis}\n", 6))

            # SVD 分支
            if hasattr(self, "vlinear"):
                lines.append(indent(f"vlinear: {repr(self.vlinear)} # enable_metis={self.vlinear.enable_metis}\n", 6))
            if hasattr(self, "ulinear"):
                lines.append(indent(f"ulinear: {repr(self.ulinear)} # enable_metis={self.ulinear.enable_metis}\n", 6))
            lines.append(indent(f"s: Tensor(shape={tuple(self.s.shape)}, dtype={self.s.dtype})\n", 6))

            return "".join(lines) + ")"

        # 未初始化 SVD 的情况
        else:
            lines = [header + ",\n"]
            # lines.append("  submodules:\n")
            lines.append("      linear_residual: " + repr(self.linear_residual) + f"# enable_metis={self.linear_residual.enable_metis}" + "\n")
            lines.append(")")
            return "".join(lines)