# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused Adam optimizer."""
from __future__ import annotations
from collections.abc import Iterable
from copy import deepcopy
from itertools import chain
from typing import Optional
import warnings

import torch
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor, Float8Quantizer
from .multi_tensor_apply import multi_tensor_applier


def get_fp8_meta(fp8_tensor):
    """FP8 metadata getter."""
    assert isinstance(fp8_tensor, Float8Tensor), "Fused optimizer supports only Float8Tensor class"
    if fp8_tensor._quantizer is None:
        raise RuntimeError("FP8 quantizer data is not initialized.")

    quantizer = fp8_tensor._quantizer

    scale = quantizer.scale
    amax = quantizer.amax
    scale_inv = fp8_tensor._scale_inv
    return scale, amax, scale_inv


class FusedAdam(torch.optim.Optimizer):
    """Implements Adam algorithm.

    Currently GPU-only.

    This version of fused Adam implements 2 fusions.

      * Fusion of the Adam update's elementwise operations
      * A multi-tensor apply launch that batches the elementwise updates applied to
        all the model's parameters into one or a few kernel launches.

    :class:`te.optimizers.FusedAdam` may be used as a drop-in replacement for ``torch.optim.AdamW``,
    or ``torch.optim.Adam`` with ``adam_w_mode=False``::

        opt = te.optimizers.FusedAdam(model.parameters(), lr = ....)
        ...
        opt.step()

    :class:`te.optimizers.FusedAdam` may be used with or without Amp.

    Adam was been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in FusedAdam!
        bias_correction (bool, optional): apply correction factor to
            moment estimates. (default: True)
        adam_w_mode (boolean, optional): Apply L2 regularization or weight decay
            True for decoupled weight decay(also known as AdamW) (default: True)
        capturable (bool, optional): whether to use the version of the optimizer
            that can be used with CUDA Graphs. (default: False)
        master_weights (bool, optional): whether to maintain FP32 master weights
           in the optimizer with FP16/BF16 mixed precision training.
            (default: False)
        master_weight_dtype (torch.dtype, optional): The dtype of master weights.
            If master_weights is False, this will be ignored. It can be one of
            [torch.float32, torch.float16]. If it's not torch.float32, the optimizer
            will create a FP32 scalar scaling factor to ensure precision.
            (default: torch.float32)
        exp_avg_dtype (torch.dtype, optional): The dtype of exp_avg. It can be
            one of [torch.float32, torch.float16, torch.uint8], where torch.uint8
            represents FP8. If it's not torch.float32, the optimizer will create
            a FP32 scalar scaling factor to ensure precision.
            (default: torch.float32)
        exp_avg_sq_dtype (torch.dtype, optional): The dtype of exp_avg_sq. It
            can be one of [torch.float32, torch.float16, torch.uint8], where
            torch.uint8 represents FP8. If it's not torch.float32, the optimizer
            will create a FP32 scalar scaling factor to ensure precision.
            (default: torch.float32)
        use_decoupled_grad (bool, optional): Whether to use ".decoupled_grad"
            instead of ".grad" for reading gradients. It's useful when the dtypes
            of grad and param are different.
            (default: False)
        store_param_remainders (bool, optional): Whether to store entire FP32 master
            params or just store the trailing 16 remainder bits. Whole FP32 master can be
            reconstructed from BF16 params plus the trailing remainder bits. Works only
            when param type is BF16 and master weight type is FP32, no effect otherwise.
            Useful memory saving optimization.
            (default: False)


    .. _Adam - A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter | dict],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        *,
        bias_correction=True,
        adam_w_mode=True,
        capturable=False,
        master_weights=False,
        master_weight_dtype=torch.float32,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
        use_decoupled_grad=False,
        store_param_remainders=False,
        set_grad_none: Optional[bool] = None,  # deprecated
    ):

        if amsgrad:
            raise RuntimeError("FusedAdam does not support the AMSGrad variant.")

        # Add constraints to dtypes of states.
        if master_weights and master_weight_dtype not in [torch.float32, torch.float16]:
            raise RuntimeError("FusedAdam only supports fp32/fp16 master weights.")
        if exp_avg_dtype not in [torch.float32, torch.float16, torch.uint8]:
            raise RuntimeError("FusedAdam only supports fp32/fp16/fp8 exp_avg.")
        if exp_avg_sq_dtype not in [torch.float32, torch.float16, torch.uint8]:
            raise RuntimeError("FusedAdam only supports fp32/fp16/fp8 exp_avg_sq.")

        # Currently, capturable mode only supports fp32 master weights and optimizer states.
        # The reason is, if the master weights or optimizer states are not in fp32 dtype,
        # they will be copied to temporary fp32 buffers first. These fp32 buffers are then
        # used as inputs for the kernel. Consequently, the pointer for earch `.step()` differs,
        # making CUDA Graph inapplicable in this scenario.
        if capturable and master_weights and master_weight_dtype != torch.float32:
            raise RuntimeError("Capturable mode only supports fp32 master weights.")
        if capturable and exp_avg_dtype != torch.float32:
            raise RuntimeError("Capturable mode only supports fp32 exp_avg.")
        if capturable and exp_avg_sq_dtype != torch.float32:
            raise RuntimeError("Capturable mode only supports fp32 exp_avg_sq")
        if capturable and store_param_remainders:
            raise RuntimeError("Capturable mode doesn't support storing param remainders")

        # If the optimizer is capturable then LR should be a tensor (on GPU)
        lr = torch.tensor(lr, dtype=torch.float32) if capturable else lr
        defaults = {
            "lr": lr,
            "bias_correction": bias_correction,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)
        self.adam_w_mode = 1 if adam_w_mode else 0

        self.capturable = capturable
        self.master_weights = master_weights

        if capturable:
            for idx, group in enumerate(self.param_groups):
                if len(group["params"]) == 0:
                    continue
                device = group["params"][0].device
                for item in ["lr"]:
                    self.param_groups[idx][item] = group[item].to(device=device)

            self._step_supports_amp_scaling = True

        # Skip buffer
        self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device="cuda")
        self.multi_tensor_adam = tex.multi_tensor_adam
        self.multi_tensor_adam_param_remainder = tex.multi_tensor_adam_param_remainder
        self.multi_tensor_adam_fp8 = tex.multi_tensor_adam_fp8
        self.multi_tensor_adam_capturable = tex.multi_tensor_adam_capturable
        self.multi_tensor_adam_capturable_master = tex.multi_tensor_adam_capturable_master

        self.master_weight_dtype = master_weight_dtype
        self.exp_avg_dtype = exp_avg_dtype
        self.exp_avg_sq_dtype = exp_avg_sq_dtype
        self.name_to_dtype_map = {
            "exp_avg": self.exp_avg_dtype,
            "exp_avg_sq": self.exp_avg_sq_dtype,
            "master_param": self.master_weight_dtype,
        }
        self.dtype_to_range_map = {
            torch.float16: torch.full(
                [1], torch.finfo(torch.float16).max / 2.0, dtype=torch.float32
            ),
            torch.uint8: torch.full([1], 448.0, dtype=torch.float32),
        }
        self._scales = {}
        self.use_decoupled_grad = use_decoupled_grad
        # Works only when master params is in FP32
        self.store_param_remainders = (
            store_param_remainders and master_weights and master_weight_dtype == torch.float32
        )

        # Deprecated options
        self.set_grad_none = set_grad_none
        if self.set_grad_none is not None:
            warnings.warn(
                "set_grad_none kwarg in FusedAdam constructor is deprecated. "
                "Use set_to_none kwarg in zero_grad instead.",
                DeprecationWarning,
            )

    def zero_grad(self, set_to_none: Optional[bool] = None) -> None:
        """Reset parameter gradients.

        Arguments:
            set_to_none (bool, optional): whether to set grads to `None`
                instead of zeroing out buffers. (default: True)

        """

        # Handle deprecated set_grad_none option
        if self.set_grad_none is not None:
            if set_to_none is not None and set_to_none != self.set_grad_none:
                raise ValueError(
                    f"Called zero_grad with set_to_none={set_to_none}, "
                    f"but FusedAdam was initialized with set_grad_none={self.set_grad_none}"
                )
            set_to_none = self.set_grad_none
        if set_to_none is None:
            set_to_none = True

        if not self.use_decoupled_grad and not set_to_none:
            super().zero_grad(set_to_none=set_to_none)
            return

        for group in self.param_groups:
            for p in group["params"]:
                if self.use_decoupled_grad and set_to_none:
                    p.decoupled_grad = None
                elif self.use_decoupled_grad and not set_to_none:
                    p.decoupled_grad.zero_()
                elif not self.use_decoupled_grad and set_to_none:
                    p.grad = None

    def _apply_scale(self, state_name, unscaled_state, scaled_state, scale):
        """Apply scaling on `unscaled_state`. `scaled_state` and `scale` will be written inplace.

        Arguments:
            state_name (string): Name of optimizer states, can be one of 'exp_avg', 'exp_avg_sq',
                and 'master_param`.
            unscaled_state (torch.Tensor): An unscaled high-precision tensor.
            scaled_state (torch.Tensor): An scaled low-precision tensor.
            scale (torch.Tensor): A FP32 tensor representing the scaling factor.
        """
        assert unscaled_state.dtype == torch.float32
        dtype = self.name_to_dtype_map[state_name]
        if dtype == torch.uint8:
            assert isinstance(scaled_state, Float8Tensor)
            assert len(scaled_state._quantizer.scale) == 1, (
                "Only scaling with one scaling factor                per tensor is supported by the"
                " FusedAdam."
            )
        else:
            assert scaled_state.dtype == dtype

        max_range = self.dtype_to_range_map[dtype]
        if max_range.device != scaled_state.device:
            max_range = max_range.to(scaled_state.device)
            self.dtype_to_range_map[scaled_state.dtype] = max_range
        if unscaled_state.device != scaled_state.device:
            unscaled_state = unscaled_state.to(scaled_state.device)
        min_val, max_val = torch.aminmax(unscaled_state)
        absmax = torch.maximum(-min_val, max_val)
        absmax = absmax.to(dtype=torch.float32, device=unscaled_state.device)
        torch.div(absmax, max_range, out=scale)
        if isinstance(scaled_state, Float8Tensor):
            scaled_state._quantizer.scale.copy_(1 / scale)
            scaled_state.copy_(unscaled_state)
        else:
            rscale = torch.where(scale > 0, scale.reciprocal(), 0.0)
            unscaled_state.mul_(rscale)
            scaled_state.copy_(unscaled_state)

    def get_unscaled_state(self, param, state_name):
        """Return the unscaled state corresponding to the input `param` and `state_name`.

        Arguments:
            param (torch.nn.Parameter): One of parameters in this optimizer.
            state_name (string): Name of optimizer states, can be one of 'exp_avg', 'exp_avg_sq',
                and 'master_param`.
        """
        state = self.state[param]
        dtype = self.name_to_dtype_map[state_name]
        if dtype == torch.uint8:
            unscaled = state[state_name].float()
        elif dtype == torch.float16:
            assert state[state_name].dtype == torch.float16
            unscaled = state[state_name].float()
            unscaled.mul_(self._scales[param][state_name])
        elif dtype == torch.float32:
            if (
                self.store_param_remainders
                and state_name == "master_param"
                and param.dtype == torch.bfloat16
            ):
                assert state[state_name].dtype == torch.int16
            else:
                assert state[state_name].dtype == torch.float32
            unscaled = state[state_name]
        else:
            raise RuntimeError(f"Dtype of {state_name} can only be fp8/fp16/fp32.")
        return unscaled

    def set_scaled_state(self, param, state_name, unscaled_state):
        """Set the optimizer state.

        If the dtype of the corresponding optimizer state is not FP32,
        it will do scaling automatically.

        Arguments:
            param (torch.nn.Parameter): One of parameters in this optimizer.
            state_name (string): Name of optimizer states, can be one of 'exp_avg', 'exp_avg_sq',
                and 'master_param`.
            unscaled_state (torch.Tensor): The original high-precision(FP32) state.
        """
        store_param_remainders = (
            self.store_param_remainders
            and state_name == "master_param"
            and param.dtype == torch.bfloat16
        )

        if store_param_remainders:
            assert unscaled_state.dtype == torch.int16
        else:
            assert unscaled_state.dtype == torch.float32
        state = self.state[param]
        if state_name not in state:
            self._initialize_state(param, state_name, False, store_param_remainders)

        dtype = self.name_to_dtype_map[state_name]
        if dtype != torch.float32:
            scale = self._scales[param]
            self._apply_scale(state_name, unscaled_state, state[state_name], scale[state_name])
        else:
            state[state_name].copy_(unscaled_state)

    def _initialize_state(
        self, param, state_name, zero_buffer: bool, store_param_remainders: bool = False
    ):
        """Initialize one of the optimizer states according to `state_name`.

        Arguments:
            param (torch.nn.Parameter): One of parameters in this optimizer.
            state_name (string): Name of optimizer states, can be one of 'exp_avg', 'exp_avg_sq',
                and 'master_param`.
            zero_buffer (bool): Whether to initialize the optimizer state with zeros.
            store_param_remainders (bool): Store only trailing remainder bits.
        """
        dtype = self.name_to_dtype_map[state_name]
        if store_param_remainders:
            data = torch.zeros_like(param, dtype=torch.int16)
        else:
            data = torch.empty_like(param, dtype=dtype)
        if zero_buffer:
            data.zero_()

        if dtype == torch.uint8:
            quantizer = Float8Quantizer(
                scale=torch.ones([1], dtype=torch.float32, device=param.device),
                amax=torch.zeros([1], dtype=torch.float32, device=param.device),
                fp8_dtype=tex.DType.kFloat8E4M3,
            )
            self.state[param][state_name] = quantizer.make_empty(param.shape)
            self.state[param][state_name].quantize_(data.float())
        else:

            self.state[param][state_name] = data

        # Create scale if necessary.
        if dtype != torch.float32:
            if param not in self._scales:
                self._scales[param] = {}
            self._scales[param][state_name] = torch.ones(
                [1], dtype=torch.float32, device=param.device
            )

    def initialize_state(self, param, store_param_remainders):
        """Initialize optimizer states.

        Arguments:
            param (torch.nn.Parameter): One of parameters in this optimizer.
            store_param_remainders (bool): Store trailing remainder bits.
        """
        self._initialize_state(param, "exp_avg", zero_buffer=True)
        self._initialize_state(param, "exp_avg_sq", zero_buffer=True)
        if self.master_weights:
            self._initialize_state(
                param,
                "master_param",
                zero_buffer=False,
                store_param_remainders=store_param_remainders,
            )
            if not store_param_remainders:
                self.set_scaled_state(param, "master_param", param.clone().detach().float())

    def state_dict(self):
        """Override the state_dict() of pytorch. Before returning the state_dict, cast all
        non-fp32 states to fp32.
        """
        state_dict = super().state_dict()

        groups = self.param_groups
        saved_groups = deepcopy(state_dict["param_groups"])
        id_map = dict(
            zip(
                chain.from_iterable(g["params"] for g in saved_groups),
                chain.from_iterable(g["params"] for g in groups),
            )
        )
        for k, v in state_dict["state"].items():
            if k in id_map:
                param = id_map[k]
                new_v = {}
                for name in v:
                    new_v[name] = self.get_unscaled_state(param, name)
                state_dict["state"][k] = new_v

        return state_dict

    def load_state_dict(self, state_dict):
        """Override the load_state_dict() of pytorch. Since pytorch's load_state_dict forces the
        state to be the same dtype as param, We need to manully set the state again.
        """
        super().load_state_dict(state_dict)

        groups = self.param_groups
        saved_groups = deepcopy(state_dict["param_groups"])
        id_map = dict(
            zip(
                chain.from_iterable(g["params"] for g in saved_groups),
                chain.from_iterable(g["params"] for g in groups),
            )
        )
        for k, v in state_dict["state"].items():
            if k in id_map:
                param = id_map[k]
                self.state[param] = {}
                for name in v:
                    if v[name] is None:
                        continue
                    if (
                        self.store_param_remainders
                        and name == "master_param"
                        and param.dtype == torch.bfloat16
                    ):
                        self.set_scaled_state(param, name, v[name])
                        assert v[name].dtype == torch.int16
                    else:
                        self.set_scaled_state(param, name, v[name].float())

    def step(self, closure=None, grad_scaler=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            grad_scaler (torch.cuda.amp.GradScaler, optional):
                gradient scaler (default: None)
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if len(group["params"]) == 0:
                continue
            device = group["params"][0].device
            bias_correction = 1 if group["bias_correction"] else 0
            beta1, beta2 = group["betas"]

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if "step" in group:
                group["step"] += (
                    1 if not self.capturable else (self._dummy_overflow_buf != 1).to(torch.int)
                )
            else:
                group["step"] = (
                    1 if not self.capturable else torch.tensor([1], dtype=torch.int, device=device)
                )

            # create lists for multi-tensor apply
            p_main_of_fp8_model = []
            p_main_of_f16_model = []
            g_of_fp8_model = []
            g_of_f16_model = []
            g_of_f32_model = []
            m_of_fp8_model = []
            m_of_f16_model = []
            m_of_f32_model = []
            v_of_fp8_model = []
            v_of_f16_model = []
            v_of_f32_model = []
            p_fp8_model = []
            p_f16_model = []
            p_f32_model = []
            # fp8 meta
            scales = []
            amaxes = []
            scale_invs = []

            # Lists for scaling
            unscaled_lists = {"exp_avg": [], "exp_avg_sq": [], "master_param": []}
            scaled_lists = {"exp_avg": [], "exp_avg_sq": [], "master_param": []}
            state_scales = {"exp_avg": [], "exp_avg_sq": [], "master_param": []}

            # Only used when extra params include fp8 tensors. Otherwise, it doesn't matter what the out_dtype is.
            out_dtype = tex.DType.kFloat32

            has_fp16 = False
            has_bf16 = False

            for p in group["params"]:
                state = self.state[p]

                store_param_remainders = self.store_param_remainders and p.dtype == torch.bfloat16

                # State initialization
                if len(state) == 0:
                    self.initialize_state(p, store_param_remainders)

                if self.use_decoupled_grad:
                    p_grad = p.decoupled_grad if hasattr(p, "decoupled_grad") else None
                else:
                    p_grad = p.grad

                if p_grad is None:
                    continue
                if p_grad.data.is_sparse:
                    raise RuntimeError("FusedAdam does not support sparse gradients.")

                # Unscaling
                unscaled_state = {}
                for name in ["exp_avg", "exp_avg_sq", "master_param"]:
                    if name in state:
                        if name == "master_param" and store_param_remainders:
                            unscaled_state[name] = self.state[p][name]
                            assert unscaled_state[name].dtype == torch.int16
                        else:
                            unscaled = self.get_unscaled_state(p, name)
                            unscaled_state[name] = unscaled
                        if self.name_to_dtype_map[name] != torch.float32:
                            unscaled_lists[name].append(unscaled)
                            scaled_lists[name].append(state[name])
                            state_scales[name].append(self._scales[p][name])

                if isinstance(p, Float8Tensor):
                    out_dtype = p._fp8_dtype
                    p_fp8_model.append(p._data.data)
                    scale, amax, scale_inv = get_fp8_meta(p)
                    scales.append(scale)
                    amaxes.append(amax)
                    scale_invs.append(scale_inv)
                    if self.master_weights:
                        p_main_of_fp8_model.append(unscaled_state["master_param"].data)
                    g_of_fp8_model.append(p_grad.data)
                    m_of_fp8_model.append(unscaled_state["exp_avg"])
                    v_of_fp8_model.append(unscaled_state["exp_avg_sq"])
                elif p.dtype in [torch.float16, torch.bfloat16]:
                    has_fp16 = has_fp16 or p.dtype == torch.float16
                    has_bf16 = has_bf16 or p.dtype == torch.bfloat16
                    p_f16_model.append(p.data)
                    if self.master_weights:
                        p_main_of_f16_model.append(unscaled_state["master_param"].data)
                    g_of_f16_model.append(p_grad.data)
                    m_of_f16_model.append(unscaled_state["exp_avg"])
                    v_of_f16_model.append(unscaled_state["exp_avg_sq"])
                elif p.dtype == torch.float32:
                    p_f32_model.append(p.data)
                    g_of_f32_model.append(p_grad.data)
                    m_of_f32_model.append(unscaled_state["exp_avg"])
                    v_of_f32_model.append(unscaled_state["exp_avg_sq"])
                else:
                    raise RuntimeError(
                        "FusedAdam only support model weights in fp32, fp16, bf16 and fp8"
                    )

                if self.capturable and len(p_fp8_model) > 0:
                    raise RuntimeError(
                        "FusedAdam does not support FP8 model weights with capturable=True."
                    )

                if has_fp16 and has_bf16:
                    if self.store_param_remainders:
                        raise RuntimeError(
                            "FusedAdam doesn't support a mix of FP16/BF16 weights + Store param"
                            " remainder."
                        )

                    # simple to add support for this, but not needed for now
                    raise RuntimeError(
                        "FusedAdam does not support a mix of float16 and bfloat16 model weights."
                    )

            def apply_multi_tensor_adam(adam_func, tensor_lists, inv_scale=None, out_dtype=None):
                # Closures defined in a loop can have unexpected
                # behavior when called outside the loop. However, this
                # function is called in the same loop iteration as it
                # is defined.
                # pylint: disable=cell-var-from-loop
                inv_scale_arg = () if inv_scale is None else (inv_scale,)
                out_dtype_arg = () if out_dtype is None else (out_dtype,)
                multi_tensor_applier(
                    adam_func,
                    self._dummy_overflow_buf,
                    tensor_lists,
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                    group["step"],
                    self.adam_w_mode,
                    bias_correction,
                    group["weight_decay"],
                    *inv_scale_arg,
                    *out_dtype_arg,
                )

            if self.capturable:
                # If the optimizer is capturable, then if there's a grad scaler it works
                # on the GPU + a different multi_tensor_applier should be called

                # overflow check of gradients
                found_inf = (
                    grad_scaler._check_inf_per_device(self)[device]
                    if grad_scaler is not None
                    else torch.zeros((1,), device=device)
                )
                self._dummy_overflow_buf.copy_(found_inf)

                # get unscale scale factor
                scale, inv_scale = None, None
                if grad_scaler:
                    scale = grad_scaler._get_scale_async()
                    inv_scale = scale.double().reciprocal().float()
                else:
                    scale = torch.ones((1,), device=device)
                    inv_scale = torch.ones((1,), device=device)

                if self.master_weights:
                    if len(p_f16_model) > 0:
                        tensor_lists = [
                            g_of_f16_model,
                            p_f16_model,
                            m_of_f16_model,
                            v_of_f16_model,
                            p_main_of_f16_model,
                        ]
                        apply_multi_tensor_adam(
                            self.multi_tensor_adam_capturable_master, tensor_lists, inv_scale
                        )
                    if len(p_f32_model) > 0:
                        tensor_lists = [
                            g_of_f32_model,
                            p_f32_model,
                            m_of_f32_model,
                            v_of_f32_model,
                        ]
                        apply_multi_tensor_adam(
                            self.multi_tensor_adam_capturable, tensor_lists, inv_scale
                        )
                else:
                    if len(p_f16_model) > 0:
                        tensor_lists = [g_of_f16_model, p_f16_model, m_of_f16_model, v_of_f16_model]
                        apply_multi_tensor_adam(
                            self.multi_tensor_adam_capturable, tensor_lists, inv_scale
                        )
                    if len(p_f32_model) > 0:
                        tensor_lists = [g_of_f32_model, p_f32_model, m_of_f32_model, v_of_f32_model]
                        apply_multi_tensor_adam(
                            self.multi_tensor_adam_capturable, tensor_lists, inv_scale
                        )

            elif self.master_weights:  # and self.capturable=False
                if len(p_f16_model) > 0:
                    tensor_lists = [
                        g_of_f16_model,
                        p_f16_model,
                        m_of_f16_model,
                        v_of_f16_model,
                        p_main_of_f16_model,
                    ]
                    if self.store_param_remainders and has_bf16 and not has_fp16:
                        # When you have BF16 params and need FP32 master params, you can reconstruct
                        # the FP32 master params with BF16 params + int16 remainders
                        apply_multi_tensor_adam(
                            self.multi_tensor_adam_param_remainder, tensor_lists
                        )
                    else:
                        apply_multi_tensor_adam(self.multi_tensor_adam, tensor_lists)
                if len(p_fp8_model) > 0:
                    tensor_lists = [
                        g_of_fp8_model,
                        p_fp8_model,
                        m_of_fp8_model,
                        v_of_fp8_model,
                        p_main_of_fp8_model,
                        scales,
                        amaxes,
                        scale_invs,
                    ]
                    apply_multi_tensor_adam(self.multi_tensor_adam_fp8, tensor_lists, out_dtype)
                if len(p_f32_model) > 0:
                    tensor_lists = [
                        g_of_f32_model,
                        p_f32_model,
                        m_of_f32_model,
                        v_of_f32_model,
                    ]
                    apply_multi_tensor_adam(self.multi_tensor_adam, tensor_lists)
            else:  # self.master_weights=False and self.capturable=False
                if len(p_f16_model) > 0:
                    tensor_lists = [g_of_f16_model, p_f16_model, m_of_f16_model, v_of_f16_model]
                    apply_multi_tensor_adam(self.multi_tensor_adam, tensor_lists)
                if len(p_f32_model) > 0:
                    tensor_lists = [g_of_f32_model, p_f32_model, m_of_f32_model, v_of_f32_model]
                    apply_multi_tensor_adam(self.multi_tensor_adam, tensor_lists)

            # Scaling
            for name in ["exp_avg", "exp_avg_sq", "master_param"]:
                if len(unscaled_lists[name]) > 0:
                    for unscaled, scaled, scale in zip(
                        unscaled_lists[name], scaled_lists[name], state_scales[name]
                    ):
                        self._apply_scale(name, unscaled, scaled, scale)

            # Try to reclaim the temporary fp32 buffers.
            del unscaled_lists

        return loss
