# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused Adam optimizer."""
import warnings
import itertools
import torch
import transformer_engine_torch as tex
from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from .multi_tensor_apply import multi_tensor_applier


def get_fp8_meta(fp8_tensor):
    """FP8 metadata getter."""
    if fp8_tensor._fp8_meta is None:
        raise RuntimeError("FP8 meta data is not initialized.")

    fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(
        forward=fp8_tensor._fp8_meta_forward,
    )

    fp8_meta_index = fp8_tensor._fp8_meta_index
    scale = fp8_tensor._fp8_meta[fp8_meta_key].scale[fp8_meta_index]
    amax = fp8_tensor._fp8_meta[fp8_meta_key].amax_history[0][fp8_meta_index]
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
        fuse_dtype_casting (bool, optional): whether to update extra parameters.
            This is useful when the optimizer needs to update master weights and model
            weights in the same kernel. The extra_params should have the same length as
            the params and should be of type torch.float16, torch.bfloat16 or Float8Tensor.
            (default: False)


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
        fuse_dtype_casting=False,
    ):

        if amsgrad:
            raise RuntimeError("FusedAdam does not support the AMSGrad variant.")
        if master_weights and not capturable:
            raise RuntimeError(
                "Master weights is currently only supported with the capturable version."
            )
        if fuse_dtype_casting and capturable:
            raise RuntimeError(
                "Fuse dtype casting is currently only supported with the non-capturable version."
            )
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
        self.fuse_dtype_casting = fuse_dtype_casting

        # Not exposed to the user yet.
        # This is used to store the extra parameters that are passed to the optimizer.
        # When fuse_dtype_casting is True, self._extra_param_groups need be reset from outside.
        self._extra_param_groups = None

        # Create full precision master weights
        self.param_groups_master = []
        for _, pg in enumerate(self.param_groups):
            param_list = pg["params"]
            self.param_groups_master.append(
                {
                    "params": [
                        p.clone().detach().float() if self.master_weights else None
                        for p in param_list
                    ],
                }
            )

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
        self.multi_tensor_adam_fp8 = tex.multi_tensor_adam_fp8
        self.multi_tensor_adam_capturable = tex.multi_tensor_adam_capturable
        self.multi_tensor_adam_capturable_master = tex.multi_tensor_adam_capturable_master

    def zero_grad(self):
        if self.set_grad_none:
            for group in self.param_groups:
                for p in group["params"]:
                    p.grad = None
        else:
            super().zero_grad()

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

        extra_param_groups = self.param_groups_master
        if self.fuse_dtype_casting:
            assert self._extra_param_groups is not None, "extra_param_groups should not be None."
            extra_param_groups = self._extra_param_groups

        for group, extra_group in zip(self.param_groups, extra_param_groups):
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
            # For master_weights=False
            g_16, p_16, m_16, v_16 = [], [], [], []
            g_32, p_32, m_32, v_32 = [], [], [], []
            # For master_weights=True
            p_main_of_fp8_model = []
            p_main_of_f16_model = []
            p_main_of_f32_model = []
            g_main_of_fp8_model = []
            g_main_of_f16_model = []
            m_of_fp8_model = []
            m_of_f16_model = []
            v_of_fp8_model = []
            v_of_f16_model = []
            p_f16_model = []
            p_fp8_model = []
            # fp8 meta
            scales = []
            amaxes = []
            scale_invs = []

            # Only used when extra params include fp8 tensors. Otherwise, it doesn't matter what the out_dtype is.
            out_dtype = tex.DType.kFloat32

            this_extra_group = extra_group
            if not isinstance(extra_group, list):
                this_extra_group = extra_group["params"]

            for p, p_extra in zip(group["params"], this_extra_group):
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError("FusedAdam does not support sparse gradients.")

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data).float()
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data).float()

                if self.fuse_dtype_casting:
                    if isinstance(p_extra, Float8Tensor):
                        out_dtype = p_extra._fp8_dtype
                        p_fp8_model.append(p_extra._data.data)
                        scale, amax, scale_inv = get_fp8_meta(p_extra)
                        scales.append(scale)
                        amaxes.append(amax)
                        scale_invs.append(scale_inv)
                        p_main_of_fp8_model.append(p.data)
                        g_main_of_fp8_model.append(p.grad.data)
                        m_of_fp8_model.append(state["exp_avg"])
                        v_of_fp8_model.append(state["exp_avg_sq"])
                    elif p_extra.dtype in [torch.float16, torch.bfloat16]:
                        p_f16_model.append(p_extra.data)
                        p_main_of_f16_model.append(p.data)
                        g_main_of_f16_model.append(p.grad.data)
                        m_of_f16_model.append(state["exp_avg"])
                        v_of_f16_model.append(state["exp_avg_sq"])
                    else:
                        raise RuntimeError(
                            "FusedAdam only support model weights in fp16/bf16 and fp8 when"
                            " fuse_dtype_casting is True."
                        )

                else:
                    if p.dtype in [torch.float16, torch.bfloat16]:
                        if self.master_weights:
                            p_main_of_f16_model.append(p_extra.data)
                        g_16.append(p.grad.data)
                        p_16.append(p.data)
                        m_16.append(state["exp_avg"])
                        v_16.append(state["exp_avg_sq"])
                    elif p.dtype == torch.float32:
                        if self.master_weights:
                            p_main_of_f32_model.append(p_extra.data)
                        g_32.append(p.grad.data)
                        p_32.append(p.data)
                        m_32.append(state["exp_avg"])
                        v_32.append(state["exp_avg_sq"])
                    else:
                        raise RuntimeError(
                            "FusedAdam only support fp16/bf16 and fp32 when fuse_dtype_casting is"
                            " False."
                        )

            def apply_multi_tensor_adam(adam_func, tensor_lists, inv_scale=None, out_dtype=None):
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

            if self.fuse_dtype_casting:
                if len(p_f16_model) > 0:
                    tensor_lists = [
                        g_main_of_f16_model,
                        p_main_of_f16_model,
                        m_of_f16_model,
                        v_of_f16_model,
                        p_f16_model,
                    ]
                    apply_multi_tensor_adam(self.multi_tensor_adam, tensor_lists)
                if len(p_fp8_model) > 0:
                    tensor_lists = [
                        g_main_of_fp8_model,
                        p_main_of_fp8_model,
                        m_of_fp8_model,
                        v_of_fp8_model,
                        p_fp8_model,
                        scales,
                        amaxes,
                        scale_invs,
                    ]
                    apply_multi_tensor_adam(self.multi_tensor_adam_fp8, tensor_lists, out_dtype)

            elif self.capturable:
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
                    if len(g_16) > 0:
                        tensor_lists = [g_16, p_16, m_16, v_16, p_main_of_f16_model]
                        apply_multi_tensor_adam(
                            self.multi_tensor_adam_capturable_master, tensor_lists, inv_scale
                        )
                    if len(g_32) > 0:
                        tensor_lists = [g_32, p_32, m_32, v_32, p_main_of_f32_model]
                        apply_multi_tensor_adam(
                            self.multi_tensor_adam_capturable_master, tensor_lists, inv_scale
                        )
                else:
                    if len(g_16) > 0:
                        tensor_lists = [g_16, p_16, m_16, v_16]
                        apply_multi_tensor_adam(
                            self.multi_tensor_adam_capturable, tensor_lists, inv_scale
                        )
                    if len(g_32) > 0:
                        tensor_lists = [g_32, p_32, m_32, v_32]
                        apply_multi_tensor_adam(
                            self.multi_tensor_adam_capturable, tensor_lists, inv_scale
                        )
            else:  # self.master_weights=False and self.capturable=False
                if len(g_16) > 0:
                    tensor_lists = [g_16, p_16, m_16, v_16]
                    apply_multi_tensor_adam(self.multi_tensor_adam, tensor_lists)
                if len(g_32) > 0:
                    tensor_lists = [g_32, p_32, m_32, v_32]
                    apply_multi_tensor_adam(self.multi_tensor_adam, tensor_lists)

        return loss
