# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused Adam optimizer."""
from copy import deepcopy
from itertools import chain

import torch
import transformer_engine_torch as tex
from .multi_tensor_apply import multi_tensor_applier
from ..float8_tensor import Float8Tensor


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
        bias_correction (bool, optional): apply correction factor to
            moment estimates. (default: True)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in FusedAdam!
        adam_w_mode (boolean, optional): Apply L2 regularization or weight decay
            True for decoupled weight decay(also known as AdamW) (default: True)
        set_grad_none (bool, optional): whether set grad to None when zero_grad()
            method is called. (default: True)
        capturable (bool, optional): whether to use the version of the optimizer
            that can be used with CUDA Graphs. (default: False)
        master_weights (bool, optional): whether to maintain FP32 master weights
           in the optimizer with FP16 mixed precision training, currently can
           only be used with capturable set to True. (default: False)
        master_weights_dtype (torch.dtype, optional): The dtype of master weights.
           If master_weights is False, this will be ignored. (default: torch.float32)
        m_dtype (torch.dtype, optional): The dtype of exp_avg in adam. (default: torch.float32)
        v_dtype (torch.dtype, optional): The dtype of exp_avg_sq in adam. (default: torch.float32)
        decoupled_grads (bool, optional): Whether to use ".decoupled_grad" instead
           of ".grad". It's used when the dtypes of grad and param are different. (default: False)

    .. _Adam - A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        adam_w_mode=True,
        weight_decay=0.0,
        amsgrad=False,
        set_grad_none=True,
        capturable=False,
        master_weights=False,
        master_weights_dtype=torch.float32,
        m_dtype=torch.float32,
        v_dtype=torch.float32,
        decoupled_grads=False,
    ):

        if amsgrad:
            raise RuntimeError("FusedAdam does not support the AMSGrad variant.")

        # Add constraints to dtypes of optimizer states.
        # Because torch currently doesn't have fp8 dtype, so uint8 is used to represent fp8.
        if m_dtype not in [torch.float32, torch.half, torch.bfloat16, torch.uint8]:
            raise RuntimeError("FusedAdam only supports fp32/fp16/bf16/fp8 m.")
        if v_dtype not in [torch.float32, torch.half, torch.bfloat16, torch.uint8]:
            raise RuntimeError("FusedAdam only supports fp32/fp16/bf16/fp8 v.")
        if master_weights:
            if master_weights_dtype not in [torch.float32, torch.half, torch.bfloat16]:
                raise RuntimeError("FusedAdam only supports fp32/fp16/bf16 master weights.")

        # Currently, capturable mode supports only fp32 master weights and optimizer states. If the
        # master weights or optimizer states are not fp32 tensors, they are copied to temporary fp32
        # buffers. These fp32 buffers are then used as inputs for the kernel. Consequently, the
        # pointer for earch `.step()` differs, making CUDA Graph inapplicable in this scenario.
        if capturable and m_dtype != torch.float32:
            raise RuntimeError("Capturable mode only supports fp32 m.")
        if capturable and v_dtype != torch.float32:
            raise RuntimeError("Capturable mode only supports fp32 v")
        if capturable and master_weights and master_weights_dtype != torch.float32:
            raise RuntimeError("Capturable mode only supports fp32 master weights.")

        # If the optimizer is capturable then LR should be a tensor (on GPU)
        lr = torch.tensor(lr, dtype=torch.float32) if capturable else lr
        defaults = dict(
            lr=lr,
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        self.adam_w_mode = 1 if adam_w_mode else 0
        self.set_grad_none = set_grad_none

        self.capturable = capturable
        self.master_weights = master_weights
        self.decoupled_grads = decoupled_grads

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
        self.multi_tensor_adam_master = tex.multi_tensor_adam_master
        self.multi_tensor_adam_capturable = tex.multi_tensor_adam_capturable
        self.multi_tensor_adam_capturable_master = tex.multi_tensor_adam_capturable_master

        self.m_dtype = m_dtype
        self.v_dtype = v_dtype
        self.master_weights_dtype = master_weights_dtype
        self._scales = {}

        self._key_to_dtype_map = {
            "exp_avg": self.m_dtype,
            "exp_avg_sq": self.v_dtype,
            "master_param": self.master_weights_dtype,
        }

        self._key_to_range_map = {}
        for key, dtype in self._key_to_dtype_map.items():
            if dtype == torch.uint8:
                self._key_to_range_map[key] = torch.full([1], 448.0, dtype=torch.float32)
            else:
                self._key_to_range_map[key] = \
                    torch.full([1], torch.finfo(dtype).max/2, dtype=torch.float32)

        self._state_names = ["exp_avg", "exp_avg_sq"]
        if self.master_weights:
            self._state_names.append("master_param")

    def zero_grad(self):
        if not self.decoupled_grads and not self.set_grad_none:
            super().zero_grad()
            return

        for group in self.param_groups:
            for p in group["params"]:
                if self.decoupled_grads and self.set_grad_none:
                    p.decoupled_grad = None
                elif self.decoupled_grads and not self.set_grad_none:
                    p.decoupled_grad.zero_()
                elif not self.decoupled_grads and self.set_grad_none:
                    p.grad = None

    def _apply_scale(self, key, unscaled_state, scaled_state, scale):
        """
        `scaled_state` and `scale` will be written inplace.
        """
        if self._key_to_range_map[key].device != scaled_state.device:
            self._key_to_range_map[key] = self._key_to_range_map[key].to(scaled_state.device)
        if unscaled_state.device != scaled_state.device:
            unscaled_state = unscaled_state.to(scaled_state.device)
        max_range = self._key_to_range_map[key]
        min_val, max_val = torch.aminmax(unscaled_state)
        absmax = torch.maximum(-min_val, max_val)
        absmax = absmax.to(dtype=torch.float32, device=unscaled_state.device)
        torch.div(absmax, max_range, out=scale)
        if isinstance(scaled_state, Float8Tensor):
            scaled_state._scale_inv.copy_(scale)
            scaled_state.copy_(unscaled_state)
        else:
            rscale = torch.where(scale > 0, scale.reciprocal(), 0.0)
            unscaled_state.mul_(rscale)
            scaled_state.copy_(unscaled_state)

    def _initialize_state(self, param, key, zero_buffer: bool):
        # Create optimizer state.
        buffer = torch.zeros_like(param) if zero_buffer else torch.empty_like(param)
        dtype = self._key_to_dtype_map[key]
        if dtype == torch.uint8:
            self.state[param][key] = Float8Tensor.to_float8(buffer)
        else:
            self.state[param][key] = buffer.to(dtype)
        # Create scale if necessary.
        if dtype not in [torch.float32, torch.bfloat16]:
            if param not in self._scales:
                self._scales[param] = {}
            self._scales[param][key] = torch.ones([1], dtype=torch.float32, device=param.device)

    def initialize_state(self, param):
        self._initialize_state(param, "exp_avg", zero_buffer=True)
        self._initialize_state(param, "exp_avg_sq", zero_buffer=True)
        if self.master_weights:
            self._initialize_state(param, "master_param", zero_buffer=False)
            self._set_state(param, "master_param", param.clone().detach().float())

    def _get_state(self, param, key):
        dtype = self._key_to_dtype_map[key]
        if dtype not in [torch.float32, torch.bfloat16]:
            scaled_state = self.state[param][key]
            if isinstance(scaled_state, Float8Tensor):
                unscaled_state = scaled_state.float()
            else:
                unscaled_state = torch.empty_like(scaled_state, dtype=torch.float32)
                unscaled_state.copy_(scaled_state)
                scale = self._scales[param][key]
                unscaled_state.mul_(scale)
            return unscaled_state
        else:
            return self.state[param][key].float()

    def get_state(self, param):
        ret = {}
        for name in self._state_names:
            ret[name] = self._get_state(param, name)
        return ret

    def _set_state(self, param, key, state):
        dtype = self._key_to_dtype_map[key]
        if dtype not in [torch.float32, torch.bfloat16]:
            scaled_state = self.state[param][key]
            scale = self._scales[param][key]
            self._apply_scale(key, state, scaled_state, scale)
        else:
            self.state[param][key].copy_(state)

    def set_state(self, param, state):
        for name in self._state_names:
            self._set_state(param, name, state[name])

    def state_dict(self):
        state_dict = super().state_dict()

        # Before returning the state_dict, cast all non-fp32 states to fp32.
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
                for name in v:
                    v[name] = self._get_state(param, name)

        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)

        # Since pytorch's load_state_dict forces the state to be the same dtype as param,
        # We need to manully set the state again.
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
                for name in self._state_names:
                    self._initialize_state(param, name, zero_buffer=False)
                    self._set_state(param, name, v[name].float())

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
            g_16, p_16, m_16, v_16, p_16_master = [], [], [], [], []
            g_bf, p_bf, m_bf, v_bf, p_bf_master = [], [], [], [], []
            g_32, p_32, m_32, v_32, p_32_master = [], [], [], [], []

            # Create lists for scaling
            unscaled_buffers = {"exp_avg": [], "exp_avg_sq": [], "master_param": []}
            scaled_buffers = {"exp_avg": [], "exp_avg_sq": [], "master_param": []}
            scales = {"exp_avg": [], "exp_avg_sq": [], "master_param": []}

            for p in group["params"]:
                if self.decoupled_grads:
                    if p.decoupled_grad is None:
                        continue
                    if p.decoupled_grad.data.is_sparse:
                        raise RuntimeError("FusedAdam does not support sparse gradients.")
                else:
                    if p.grad is None:
                        continue
                    if p.grad.data.is_sparse:
                        raise RuntimeError("FusedAdam does not support sparse gradients.")

                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    self.initialize_state(p)

                # Unscaling
                state_buffer = {}
                for name in self._state_names:
                    dtype = self._key_to_dtype_map[name]
                    if dtype not in [torch.float32, torch.bfloat16]:
                        scales[name].append(self._scales[p][name])
                        scaled_buffers[name].append(state[name])
                        unscaled_buffers[name].append(self._get_state(p, name))
                        state_buffer[name] = unscaled_buffers[name][-1]
                    else:
                        state_buffer[name] = state[name]

                if self.decoupled_grads:
                    g = p.decoupled_grad
                else:
                    g = p.grad

                if isinstance(p, Float8Tensor):
                    p_data = p._data
                else:
                    p_data = p.data

                if p.dtype == torch.float16:
                    if self.master_weights:
                        p_16_master.append(state_buffer["master_param"].data)
                    g_16.append(g)
                    p_16.append(p_data)
                    m_16.append(state_buffer["exp_avg"])
                    v_16.append(state_buffer["exp_avg_sq"])
                elif p.dtype == torch.bfloat16:
                    if self.master_weights:
                        p_bf_master.append(state_buffer["master_param"].data)
                    g_bf.append(g)
                    p_bf.append(p_data)
                    m_bf.append(state_buffer["exp_avg"])
                    v_bf.append(state_buffer["exp_avg_sq"])
                elif p.dtype == torch.float32:
                    if self.master_weights:
                        p_32_master.append(state_buffer["master_param"].data)
                    g_32.append(g)
                    p_32.append(p_data)
                    m_32.append(state_buffer["exp_avg"])
                    v_32.append(state_buffer["exp_avg_sq"])
                else:
                    raise RuntimeError("FusedAdam only support fp16, bf16 and fp32.")

            # If the optimizer is capturable, then if there's a grad scaler it works
            # on the GPU + a different multi_tensor_applier should be called
            if self.capturable:
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

                if len(g_16) > 0:
                    multi_tensor_applier(
                        (
                            self.multi_tensor_adam_capturable_master
                            if self.master_weights
                            else self.multi_tensor_adam_capturable
                        ),
                        self._dummy_overflow_buf,
                        (
                            [g_16, p_16, m_16, v_16, p_16_master]
                            if self.master_weights
                            else [g_16, p_16, m_16, v_16]
                        ),
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                        inv_scale,
                    )

                if len(g_bf) > 0:
                    multi_tensor_applier(
                        (
                            self.multi_tensor_adam_capturable_master
                            if self.master_weights
                            else self.multi_tensor_adam_capturable
                        ),
                        self._dummy_overflow_buf,
                        (
                            [g_bf, p_bf, m_bf, v_bf, p_bf_master]
                            if self.master_weights
                            else [g_bf, p_bf, m_bf, v_bf]
                        ),
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                        inv_scale,
                    )

                if len(g_32) > 0:
                    multi_tensor_applier(
                        (
                            self.multi_tensor_adam_capturable_master
                            if self.master_weights
                            else self.multi_tensor_adam_capturable
                        ),
                        self._dummy_overflow_buf,
                        (
                            [g_32, p_32, m_32, v_32, p_32_master]
                            if self.master_weights
                            else [g_32, p_32, m_32, v_32]
                        ),
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                        inv_scale,
                    )
            else:
                if len(g_16) > 0:
                    multi_tensor_applier(
                        (
                            self.multi_tensor_adam_master
                            if self.master_weights
                            else self.multi_tensor_adam
                        ),
                        self._dummy_overflow_buf,
                        (
                            [g_16, p_16, m_16, v_16, p_16_master]
                            if self.master_weights
                            else [g_16, p_16, m_16, v_16]
                        ),
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                    )

                if len(g_bf) > 0:
                    multi_tensor_applier(
                        (
                            self.multi_tensor_adam_master
                            if self.master_weights
                            else self.multi_tensor_adam
                        ),
                        self._dummy_overflow_buf,
                        (
                            [g_bf, p_bf, m_bf, v_bf, p_bf_master]
                            if self.master_weights
                            else [g_bf, p_bf, m_bf, v_bf]
                        ),
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                    )

                if len(g_32) > 0:
                    multi_tensor_applier(
                        (
                            self.multi_tensor_adam_master
                            if self.master_weights
                            else self.multi_tensor_adam
                        ),
                        self._dummy_overflow_buf,
                        (
                            [g_32, p_32, m_32, v_32, p_32_master]
                            if self.master_weights
                            else [g_32, p_32, m_32, v_32]
                        ),
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                    )

            # Scaling
            for name in self._state_names:
                for unscaled, scaled, scale in zip(
                    unscaled_buffers[name], scaled_buffers[name], scales[name]):
                    self._apply_scale(name, unscaled, scaled, scale)

            # Try to reclaim the temporary fp32 buffers.
            del unscaled_buffers, scaled_buffers, scales

        return loss
