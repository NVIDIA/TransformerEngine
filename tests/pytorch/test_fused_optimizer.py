# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import copy
from contextlib import nullcontext

import pytest
import torch
from torch import nn
from torch.testing._internal.common_device_type import largeTensorTest
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling
from transformer_engine.pytorch.attention.multi_head_attention import MultiheadAttention
from transformer_engine.pytorch import fp8_model_init
from transformer_engine.pytorch.utils import is_bf16_compatible
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.utils import gpu_autocast_ctx

# Check if FP8 is supported
fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()


class TestFusedOptimizer:

    def setup_method(self, *, iters: int = 7) -> None:
        self.iters = iters
        torch.manual_seed(9876)

    def gen_param_optim(self, tensors, options, tst_options=None):

        # Adding this to make backward compatible with existing tests. Just in
        # case "tst_options" are not provided, it gets a copy of options
        # which contains the parameters for the reference optimizer
        if tst_options == None:
            tst_options = options

        ref_param = []
        tst_param = []
        for tensor in tensors:
            ref_param.append(torch.nn.Parameter(tensor.clone()))
            tst_param.append(torch.nn.Parameter(tensor.clone()))

        ref_optim = self.ref_optim(ref_param, **options)
        tst_optim = self.fused_optim(tst_param, **tst_options)

        return (ref_param, tst_param, ref_optim, tst_optim)

    def gen_grad(self, ref_param, tst_param):
        for p_ref, p_tst in zip(ref_param, tst_param):
            p_ref.grad = torch.rand_like(p_ref)
            p_tst.grad = p_ref.grad

    def gen_mixed_grad(self, ref_param, tst_param, scale=1.0):
        half_grads = []
        for p_ref, p_tst in zip(ref_param, tst_param):
            half_grads.append(torch.rand_like(p_ref).half())
            p_ref.grad = half_grads[-1].float() / scale
        return half_grads

    def gen_single_type_test(
        self, param_type=torch.float, device="cuda", *, skip_assert: bool = False
    ):
        nelem = 278011

        # Some ref and test optimizers may require different set of options.
        # This is a quick workaround to add that functionality while making
        # minimum changes in existing code.
        # If there is no "tst_options" field provided, safe to initialize
        # the test optimizer with the parameters of reference optimizer.
        if not hasattr(self, "tst_options"):
            self.tst_options = self.options

        tensor = torch.rand(nelem, dtype=param_type, device=device)

        ref_param, tst_param, ref_optim, tst_optim = self.gen_param_optim(
            [tensor], self.options, self.tst_options
        )

        for i in range(self.iters):
            self.gen_grad(ref_param, tst_param)
            ref_optim.step()
            tst_optim.step()
            if skip_assert:
                return
            torch.testing.assert_close(ref_param, tst_param)


class TestFusedAdam(TestFusedOptimizer):

    def setup_method(self) -> None:
        super().setup_method()
        self.options = {
            "lr": 5e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-08,
            "weight_decay": 0,
            "amsgrad": False,
        }
        self.ref_optim = torch.optim.Adam
        self.fused_optim = te.optimizers.FusedAdam

    def test_float(self):
        self.gen_single_type_test(param_type=torch.float)

    # NOTE(mkozuki): Current threshold values look too small for BFloat16.
    # TODO(mkozuki): Refactor `TestFusedOptimizer`
    def test_half(self):
        self.gen_single_type_test(param_type=torch.float16, skip_assert=True)

    def test_bfloat16(self):
        self.gen_single_type_test(param_type=torch.bfloat16, skip_assert=True)

    def test_multi_params(self):
        sizes = [[4096, 1024], [4096], [4096, 2048], [32320, 1024], [1]]

        tensors = []
        for size in sizes:
            tensors.append(torch.rand(size, dtype=torch.float, device="cuda"))
        ref_param, tst_param, ref_optim, tst_optim = self.gen_param_optim(tensors, self.options)

        for i in range(self.iters):
            self.gen_grad(ref_param, tst_param)
            ref_optim.step()
            tst_optim.step()

            torch.testing.assert_close(ref_param, tst_param)

    def test_adam_option(self):
        nelem = 1
        adam_option = {
            "lr": 0.01,
            "betas": (0.6, 0.9),
            "eps": 3e-06,
            "weight_decay": 0,
            "amsgrad": False,
        }

        tensor = torch.rand(nelem, dtype=torch.float, device="cuda")
        ref_param, tst_param, ref_optim, tst_optim = self.gen_param_optim([tensor], adam_option)

        for i in range(self.iters):
            self.gen_grad(ref_param, tst_param)
            ref_optim.step()
            tst_optim.step()

            torch.testing.assert_close(ref_param, tst_param)

    def test_frozen_model(self):
        nelem = 1
        adam_option = {
            "lr": 0.01,
            "betas": (0.6, 0.9),
            "eps": 3e-06,
            "weight_decay": 0,
            "amsgrad": False,
        }

        tensor = torch.rand(nelem, dtype=torch.float, device="cuda")
        ref_param, tst_param, ref_optim, tst_optim = self.gen_param_optim([tensor], adam_option)

        # Add an empty param group which may occur for pipeline parallel p-tuning
        tst_optim.add_param_group({"params": []})

        for i in range(self.iters):
            self.gen_grad(ref_param, tst_param)
            ref_optim.step()
            tst_optim.step()

            torch.testing.assert_close(ref_param, tst_param)

    def gen_precision_aware_test(
        self,
        use_fp8_params,
        param_dtype,
        use_master_weights,
        master_weight_dtype,
        grad_dtype,
        exp_avg_dtype,
        exp_avg_sq_dtype,
        store_param_remainders=False,
        model_rtol=None,
        model_atol=None,
        master_rtol=None,
        master_atol=None,
        skip_assert=False,
    ):
        build_model_context = nullcontext
        build_model_context_args = {}
        if use_fp8_params:
            build_model_context = fp8_model_init
            build_model_context_args["enabled"] = True

        with build_model_context(**build_model_context_args):
            model = MultiheadAttention(
                hidden_size=1024,
                num_attention_heads=16,
                layer_number=1,
                params_dtype=param_dtype,
                fuse_qkv_params=True,
            ).cuda()

        ref_params = []
        model_params = []

        for p in model.parameters():
            if p.requires_grad:
                ref_params.append(p.detach().clone().float())
                model_params.append(p)

        options = {
            "lr": 1,
            "betas": (0.1, 0.25),
            "eps": 1e-08,
            "weight_decay": 0,
            "amsgrad": False,
        }

        ref_optim = torch.optim.Adam(ref_params, **options)
        tst_optim = te.optimizers.FusedAdam(
            model_params,
            master_weights=use_master_weights,
            master_weight_dtype=master_weight_dtype,
            exp_avg_dtype=exp_avg_dtype,
            exp_avg_sq_dtype=exp_avg_sq_dtype,
            use_decoupled_grad=True,
            store_param_remainders=store_param_remainders,
            **options,
        )

        def test_one_iteration(ref_optimizer, tst_optimizer):
            for p_ref, p in zip(ref_params, model_params):
                p_ref.grad = torch.rand_like(p_ref)
                p.decoupled_grad = p_ref.grad.clone().to(grad_dtype)
            ref_optimizer.step()
            tst_optimizer.step()
            if use_master_weights and not store_param_remainders:
                master_weights_to_fp32 = [
                    tst_optim.get_unscaled_state(p, "master_param") for p in model_params
                ]
                if not skip_assert:
                    torch.testing.assert_close(
                        ref_params,
                        master_weights_to_fp32,
                        rtol=master_rtol,
                        atol=master_atol,
                        equal_nan=True,
                    )
            ref_params_to_model_dtype = [p.to(param_dtype) for p in ref_params]
            if not skip_assert:
                torch.testing.assert_close(
                    ref_params_to_model_dtype,
                    model_params,
                    rtol=model_rtol,
                    atol=model_atol,
                    equal_nan=True,
                )

        for i in range(self.iters):
            test_one_iteration(ref_optim, tst_optim)

        state_dict = tst_optim.state_dict()
        tst_optim = te.optimizers.FusedAdam(
            model_params,
            master_weights=use_master_weights,
            master_weight_dtype=master_weight_dtype,
            exp_avg_dtype=exp_avg_dtype,
            exp_avg_sq_dtype=exp_avg_sq_dtype,
            use_decoupled_grad=True,
            store_param_remainders=store_param_remainders,
            **options,
        )
        tst_optim.load_state_dict(state_dict)

        for i in range(self.iters):
            test_one_iteration(ref_optim, tst_optim)

    def test_fp32_no_master(self):
        self.gen_precision_aware_test(
            use_fp8_params=False,
            param_dtype=torch.float32,
            use_master_weights=False,
            master_weight_dtype=torch.float32,
            grad_dtype=torch.float32,
            exp_avg_dtype=torch.float32,
            exp_avg_sq_dtype=torch.float32,
        )

    @pytest.mark.skipif(not is_bf16_compatible(), reason="bf16 if not supported")
    def test_fp32_master(self):
        self.gen_precision_aware_test(
            use_fp8_params=False,
            param_dtype=torch.bfloat16,
            use_master_weights=True,
            master_weight_dtype=torch.float32,
            grad_dtype=torch.float32,
            exp_avg_dtype=torch.float32,
            exp_avg_sq_dtype=torch.float32,
        )

    @pytest.mark.skipif(not is_bf16_compatible(), reason="bf16 if not supported")
    def test_fp32_master_store_param_remainders(self):
        self.gen_precision_aware_test(
            use_fp8_params=False,
            param_dtype=torch.bfloat16,
            use_master_weights=True,
            master_weight_dtype=torch.float32,
            grad_dtype=torch.float32,
            exp_avg_dtype=torch.float32,
            exp_avg_sq_dtype=torch.float32,
            store_param_remainders=True,
        )

    @pytest.mark.skipif(not is_bf16_compatible(), reason="bf16 if not supported")
    def test_fp16_master(self):
        self.gen_precision_aware_test(
            use_fp8_params=False,
            param_dtype=torch.bfloat16,
            use_master_weights=True,
            master_weight_dtype=torch.half,
            grad_dtype=torch.float32,
            exp_avg_dtype=torch.float32,
            exp_avg_sq_dtype=torch.float32,
            master_rtol=2e-3,
            master_atol=2e-3,
        )

    @pytest.mark.skipif(not is_bf16_compatible(), reason="bf16 if not supported")
    def test_bf16_grad(self):
        self.gen_precision_aware_test(
            use_fp8_params=False,
            param_dtype=torch.bfloat16,
            use_master_weights=True,
            master_weight_dtype=torch.float32,
            grad_dtype=torch.bfloat16,
            exp_avg_dtype=torch.float32,
            exp_avg_sq_dtype=torch.float32,
            master_rtol=2e-3,
            master_atol=2e-3,
        )

    @pytest.mark.skipif(not is_bf16_compatible(), reason="bf16 if not supported")
    def test_fp16_exp_avg(self):
        self.gen_precision_aware_test(
            use_fp8_params=False,
            param_dtype=torch.bfloat16,
            use_master_weights=True,
            master_weight_dtype=torch.float32,
            grad_dtype=torch.float32,
            exp_avg_dtype=torch.half,
            exp_avg_sq_dtype=torch.float32,
            master_rtol=2e-3,
            master_atol=2e-3,
        )

    @pytest.mark.skipif(not is_bf16_compatible(), reason="bf16 if not supported")
    def test_bf16_exp_avg(self):
        self.gen_precision_aware_test(
            use_fp8_params=False,
            param_dtype=torch.bfloat16,
            use_master_weights=True,
            master_weight_dtype=torch.float32,
            grad_dtype=torch.float32,
            exp_avg_dtype=torch.bfloat16,
            exp_avg_sq_dtype=torch.float32,
            master_rtol=2e-3,
            master_atol=2e-3,
        )

    @pytest.mark.skipif(not is_bf16_compatible(), reason="bf16 if not supported")
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    def test_fp8_exp_avg(self):
        self.gen_precision_aware_test(
            use_fp8_params=False,
            param_dtype=torch.bfloat16,
            use_master_weights=True,
            master_weight_dtype=torch.float32,
            grad_dtype=torch.float32,
            exp_avg_dtype=torch.uint8,
            exp_avg_sq_dtype=torch.float32,
            master_rtol=1e-2,
            master_atol=1e-2,
        )

    @pytest.mark.skipif(not is_bf16_compatible(), reason="bf16 if not supported")
    def test_fp16_exp_avg_sq(self):
        self.gen_precision_aware_test(
            use_fp8_params=False,
            param_dtype=torch.bfloat16,
            use_master_weights=True,
            master_weight_dtype=torch.float32,
            grad_dtype=torch.float32,
            exp_avg_dtype=torch.float32,
            exp_avg_sq_dtype=torch.half,
            master_rtol=2e-3,
            master_atol=2e-3,
        )

    @pytest.mark.skipif(not is_bf16_compatible(), reason="bf16 if not supported")
    def test_bf16_exp_avg_sq(self):
        self.gen_precision_aware_test(
            use_fp8_params=False,
            param_dtype=torch.bfloat16,
            use_master_weights=True,
            master_weight_dtype=torch.float32,
            grad_dtype=torch.float32,
            exp_avg_dtype=torch.float32,
            exp_avg_sq_dtype=torch.bfloat16,
            master_rtol=2e-3,
            master_atol=2e-3,
        )

    @pytest.mark.skipif(not is_bf16_compatible(), reason="bf16 if not supported")
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    def test_fp8_exp_avg_sq(self):
        self.gen_precision_aware_test(
            use_fp8_params=False,
            param_dtype=torch.bfloat16,
            use_master_weights=True,
            master_weight_dtype=torch.float32,
            grad_dtype=torch.float32,
            exp_avg_dtype=torch.float32,
            exp_avg_sq_dtype=torch.uint8,
            skip_assert=True,
        )

    @pytest.mark.skipif(not is_bf16_compatible(), reason="bf16 if not supported")
    def test_bf16_model_weight_cast(self):
        dtype = torch.bfloat16
        model = MultiheadAttention(
            hidden_size=1024,
            num_attention_heads=16,
            layer_number=1,
            params_dtype=dtype,
            fuse_qkv_params=True,
        ).cuda()
        ref_params = []
        model_params = []
        for p in model.parameters():
            if p.requires_grad:
                ref_params.append(p.detach().clone().float())
                model_params.append(p)
        options = {
            "lr": 5e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-08,
            "weight_decay": 0,
            "amsgrad": False,
        }
        ref_optim = torch.optim.Adam(ref_params, **options)
        tst_optim = te.optimizers.FusedAdam(
            model_params, master_weights=True, use_decoupled_grad=True, **options
        )

        for i in range(self.iters):
            for p_ref, p in zip(ref_params, model_params):
                p_ref.grad = torch.rand_like(p_ref)
                p.decoupled_grad = p_ref.grad.clone()
            ref_optim.step()
            tst_optim.step()
            master_params = [tst_optim.get_unscaled_state(p, "master_param") for p in model_params]
            torch.testing.assert_close(ref_params, master_params)
            model_params_to_fp32 = [p.float() for p in model_params]
            torch.testing.assert_close(
                ref_params, model_params_to_fp32, rtol=1e-3, atol=1e-3, equal_nan=True
            )

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    def test_fp8_model_weight_cast(self):
        dtype = torch.bfloat16
        with fp8_model_init(enabled=True, recipe=DelayedScaling()):
            model = MultiheadAttention(
                hidden_size=1024,
                num_attention_heads=16,
                layer_number=1,
                params_dtype=dtype,
                fuse_qkv_params=True,
            ).cuda()
        ref_params = []
        model_params = []
        for p in model.parameters():
            if p.requires_grad:
                ref_params.append(p.detach().clone().float())
                model_params.append(p)
        options = {
            "lr": 5e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-08,
            "weight_decay": 0,
            "amsgrad": False,
        }
        ref_optim = torch.optim.Adam(ref_params, **options)
        tst_optim = te.optimizers.FusedAdam(
            model_params, master_weights=True, use_decoupled_grad=True, **options
        )

        for i in range(self.iters):
            for p_ref, p in zip(ref_params, model_params):
                p_ref.grad = torch.rand_like(p_ref)
                p.decoupled_grad = p_ref.grad.clone()
            ref_optim.step()
            tst_optim.step()
            master_params = [tst_optim.get_unscaled_state(p, "master_param") for p in model_params]
            torch.testing.assert_close(ref_params, master_params)
            model_params_to_fp32 = [p.float() for p in model_params]
            torch.testing.assert_close(
                ref_params, model_params_to_fp32, rtol=1e-2, atol=1e-2, equal_nan=True
            )


class TestFusedSGD(TestFusedOptimizer):

    def setup_method(self) -> None:
        super().setup_method()
        self.options = {"lr": 0.25, "momentum": 0.125}
        self.ref_optim = torch.optim.SGD
        self.fused_optim = te.optimizers.FusedSGD

    def test_float(self):
        self.gen_single_type_test(param_type=torch.float)

    def test_half(self):
        self.gen_single_type_test(param_type=torch.float16)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.reshape(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y


class AdamTest:

    def setup_method(self, *, seed: int = 0) -> None:
        torch.manual_seed(seed)

        self.model = Model().cuda()
        self.model_ = Model().cuda()
        self.model_.load_state_dict(copy.deepcopy(self.model.state_dict()))

        self.lr = 0.00001
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

    def test_grad_scaler(self):
        params_ = [p for p in self.model_.parameters() if p.requires_grad]
        optimizer_ = te.optimizers.FusedAdam(params_, lr=self.lr, capturable=False)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        scaler_ = torch.cuda.amp.GradScaler(enabled=True)

        for i in range(100):
            x = torch.rand([32, 1, 28, 28]).cuda().to(memory_format=torch.channels_last)
            x_ = x.clone()
            gt = torch.rand([32, 10]).cuda()
            gt_ = gt.clone()

            # Reference
            with gpu_autocast_ctx(enabled=True):
                y = self.model(x)
                loss = ((gt - y) ** 2).mean()

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            # DUT
            with gpu_autocast_ctx(enabled=True):
                y = self.model_(x)
                loss_ = ((gt_ - y) ** 2).mean()

            scaler_.scale(loss_).backward()
            scaler_.step(optimizer_)
            scaler_.update()

            for module in zip(self.model.modules(), self.model_.modules()):
                m = module[0]
                m_ = module[1]
                if isinstance(m, nn.Conv2d) or isinstance(m_, nn.Linear):
                    torch.testing.assert_close(
                        m.weight, m_.weight, atol=1e-3, rtol=1e-3, equal_nan=True
                    )
                    torch.testing.assert_close(
                        m.weight.grad,
                        m_.weight.grad,
                        atol=1e-3,
                        rtol=1e-3,
                        equal_nan=True,
                    )

            # Init for next iteration
            self.optimizer.zero_grad()
            optimizer_.zero_grad()

            self.model_.load_state_dict(copy.deepcopy(self.model.state_dict()))

    def test_grad_scaler_capturable(self):
        params_ = [p for p in self.model_.parameters() if p.requires_grad]
        optimizer_ = te.optimizers.FusedAdam(params_, lr=self.lr, capturable=True)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        scaler_ = torch.cuda.amp.GradScaler(enabled=True)

        for i in range(100):
            x = torch.rand([32, 1, 28, 28]).cuda().to(memory_format=torch.channels_last)
            x_ = x.clone()
            gt = torch.rand([32, 10]).cuda()
            gt_ = gt.clone()

            # Reference
            with gpu_autocast_ctx(enabled=True):
                y = self.model(x)
                loss = ((gt - y) ** 2).mean()

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            # DUT
            with gpu_autocast_ctx(enabled=True):
                y = self.model_(x)
                loss_ = ((gt_ - y) ** 2).mean()

            scaler_.scale(loss_).backward()
            scaler_.step(optimizer_)
            scaler_.update()

            for module in zip(self.model.modules(), self.model_.modules()):
                m = module[0]
                m_ = module[1]
                if isinstance(m, nn.Conv2d) or isinstance(m_, nn.Linear):
                    torch.testing.assert_close(
                        m.weight, m_.weight, atol=1e-3, rtol=1e-3, equal_nan=True
                    )
                    torch.testing.assert_close(
                        m.weight.grad,
                        m_.weight.grad,
                        atol=1e-3,
                        rtol=1e-3,
                        equal_nan=True,
                    )

            # Init for next iteration
            self.optimizer.zero_grad()
            optimizer_.zero_grad()

            self.model_.load_state_dict(copy.deepcopy(self.model.state_dict()))

    def test_grad_scaler_capturable_master(self):
        # Cast conv layers to FP16
        for m in self.model_.modules():
            if m.__class__ in [torch.nn.Conv2d]:
                m.half()
        params_ = [p for p in self.model_.parameters() if p.requires_grad]
        master_weights = [p.float() for p in self.model_.parameters() if p.requires_grad]
        optimizer_ = te.optimizers.FusedAdam(
            params_, lr=self.lr, capturable=True, master_weights=master_weights
        )
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        scaler_ = torch.cuda.amp.GradScaler(enabled=True)

        for i in range(100):
            x = torch.rand([32, 1, 28, 28]).cuda().to(memory_format=torch.channels_last)
            x_ = x.clone()
            gt = torch.rand([32, 10]).cuda()
            gt_ = gt.clone()

            # Reference
            with gpu_autocast_ctx(enabled=True):
                y = self.model(x)
                loss = ((gt - y) ** 2).mean()

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            # DUT
            with gpu_autocast_ctx(enabled=True):
                y = self.model_(x)
                loss_ = ((gt_ - y) ** 2).mean()

            scaler_.scale(loss_).backward()
            scaler_.step(optimizer_)
            scaler_.update()

            for module in zip(self.model.modules(), self.model_.modules()):
                m = module[0]
                m_ = module[1]
                if isinstance(m, nn.Conv2d) or isinstance(m_, nn.Linear):
                    torch.testing.assert_close(
                        m.weight,
                        m_.weight.float(),
                        atol=1e-3,
                        rtol=1e-3,
                        equal_nan=True,
                    )
                    torch.testing.assert_close(
                        m.weight.grad,
                        m_.weight.grad.float(),
                        atol=1e-3,
                        rtol=1e-3,
                        equal_nan=True,
                    )

            # Init for next iteration
            self.optimizer.zero_grad()
            optimizer_.zero_grad()

            self.model_.load_state_dict(copy.deepcopy(self.model.state_dict()))

    def test_native(self):
        params_ = [p for p in self.model_.parameters() if p.requires_grad]
        optimizer_ = te.optimizers.FusedAdam(params_, lr=self.lr, capturable=False)

        for i in range(100):
            x = torch.rand([32, 1, 28, 28]).cuda().to(memory_format=torch.channels_last)
            x_ = x.clone()
            gt = torch.rand([32, 10]).cuda()
            gt_ = gt.clone()

            # Reference
            y = self.model(x)
            loss = ((gt - y) ** 2).mean()

            loss.backward()
            self.optimizer.step()

            # DUT
            y = self.model_(x)
            loss_ = ((gt_ - y) ** 2).mean()

            loss_.backward()
            optimizer_.step()

            for module in zip(self.model.modules(), self.model_.modules()):
                m = module[0]
                m_ = module[1]
                if isinstance(m, nn.Conv2d) or isinstance(m_, nn.Linear):
                    torch.testing.assert_close(
                        m.weight, m_.weight, atol=1e-3, rtol=1e-3, equal_nan=True
                    )
                    torch.testing.assert_close(
                        m.weight.grad,
                        m_.weight.grad,
                        atol=1e-3,
                        rtol=1e-3,
                        equal_nan=True,
                    )

            # Init for next iteration
            self.optimizer.zero_grad()
            optimizer_.zero_grad()

            self.model_.load_state_dict(copy.deepcopy(self.model.state_dict()))

    @largeTensorTest("60GB", "cuda")
    def test_large_tensor(self):
        t = torch.zeros(2359332864, dtype=torch.half, device="cuda")
        t2 = torch.zeros(2359332864, dtype=torch.half, device="cuda")
        grad = torch.randn_like(t)
        t.grad = grad
        t2.grad = grad
        params = [t]
        params2 = [t2]
        optimizer = te.optimizers.FusedAdam(params, lr=self.lr)
        optimizer.step()
        optimizer2 = torch.optim.Adam(params2, lr=self.lr)
        torch.testing.assert_close(t, t2)
        torch.cuda.synchronize()
