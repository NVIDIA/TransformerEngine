# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import functools
import itertools
import os
import random
import tempfile
from string import Template

import pytest
import torch

import nvdlfw_inspect.api as debug_api
import transformer_engine.debug
import transformer_engine.pytorch as tepytorch
import transformer_engine_torch as tex
from transformer_engine.common.recipe import DelayedScaling, Format
from transformer_engine.pytorch.fp8 import _default_sf_compute
from transformer_engine.pytorch.tensor.float8_tensor import (
    Float8Quantizer,
    Float8CurrentScalingQuantizer,
)
from transformer_engine.pytorch.module.base import (
    _2X_ACC_DGRAD,
    _2X_ACC_FPROP,
    _2X_ACC_WGRAD,
)

all_boolean = [True, False]
FP8_FORMAT = Format.HYBRID
AMAX_HISTORY_LEN = 16
FP8_RECIPE = DelayedScaling(
    fp8_format=FP8_FORMAT, amax_history_len=AMAX_HISTORY_LEN, amax_compute_algo="max"
)
SEED = 1234
IN_SIZE = 128
OUT_SIZE = 64
BATCH_SIZE = 16
SEQ_LEN = 128
LOSS_FN = torch.nn.functional.cross_entropy


def _cast_to_fp8(tensor, scale, dtype):
    tensor = tensor.contiguous()
    if type(scale) == torch.Tensor:
        amax = scale.abs().max().float()
        quantizer = Float8Quantizer(scale, amax, dtype)
    else:
        quantizer = Float8CurrentScalingQuantizer(scale, device=tensor.device)

    return quantizer(tensor)


def _get_current_scale(tensor, fp8_dtype):
    if fp8_dtype == tex.DType.kFloat8E4M3:
        fp8_max = Format.E4M3.value.max_fwd
    else:
        fp8_max = Format.E5M2.value.max_fwd

    amax = tensor.abs().max().float()
    one = torch.ones(1, device=tensor.device)

    return _default_sf_compute(amax, one, fp8_max, 0).detach()


def _fake_cast(tensor, fp8_dtype, scale):
    scale = scale or _get_current_scale(tensor, fp8_dtype)
    fp8_tensor = _cast_to_fp8(tensor, scale, fp8_dtype)

    return fp8_tensor.dequantize()


def _fp8_gemm_kernel(tensor1, scale1, dtype1, tensor2, scale2, dtype2, use_split_accumulator):
    fp8_tensor1 = _cast_to_fp8(tensor1, scale1, dtype1)
    fp8_tensor2 = _cast_to_fp8(tensor2, scale2, dtype2)

    out, *_ = tepytorch.cpp_extensions.general_gemm(
        fp8_tensor1,
        fp8_tensor2,
        tepytorch.module.base.get_workspace(),
        torch.float32,
        use_split_accumulator=use_split_accumulator,
    )
    out.requires_grad = True
    return out.T


def _emulate_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    fprop_fp8: bool = False,
    fprop_input_fake_quant: tex.DType = None,
    fprop_input_scale: torch.Tensor = None,
    fprop_weight_fake_quant: tex.DType = None,
    fprop_weight_scale: torch.Tensor = None,
    dgrad_fp8: bool = False,
    dgrad_gradient_fake_quant: tex.DType = None,
    dgrad_gradient_scale: torch.Tensor = None,
    dgrad_weight_fake_quant: tex.DType = None,
    dgrad_weight_scale: torch.Tensor = None,
    wgrad_fp8: bool = False,
    wgrad_gradient_fake_quant: tex.DType = None,
    wgrad_gradient_scale: torch.Tensor = None,
    wgrad_input_fake_quant: tex.DType = None,
    wgrad_input_scale: torch.Tensor = None,
    loss_multiplier: float = 1.0,
    activation_sync=None,
    gradient_sync=None,
):
    _scalar = lambda x: torch.Tensor([x]).cuda() if type(x) in [float, torch.Tensor] else x
    if fprop_fp8:
        activation = _fp8_gemm_kernel(
            input,
            _scalar(fprop_input_scale or 1.0),
            tex.DType.kFloat8E4M3,
            weight,
            _scalar(fprop_weight_scale or 1.0),
            tex.DType.kFloat8E4M3,
            _2X_ACC_FPROP,
        )
        activation = activation.clone().detach().contiguous().requires_grad_(True)
    else:
        fprop_input = (
            _fake_cast(input, fprop_input_fake_quant, _scalar(fprop_input_scale))
            if fprop_input_fake_quant is not None
            else input
        )
        fprop_weight = (
            _fake_cast(weight, fprop_weight_fake_quant, _scalar(fprop_weight_scale))
            if fprop_weight_fake_quant is not None
            else weight
        )

        activation = (fprop_input @ fprop_weight.T).contiguous()

    if activation_sync:
        activation = activation_sync(activation)

    activation.retain_grad()

    (loss_multiplier * activation.sum()).backward(retain_graph=True)
    gradient = activation.grad.clone()

    if gradient_sync:
        gradient = gradient_sync(gradient)

    if dgrad_fp8:
        dgrad = _fp8_gemm_kernel(
            weight.T,
            _scalar(dgrad_weight_scale or 1.0),
            tex.DType.kFloat8E4M3,
            gradient,
            _scalar(dgrad_gradient_scale or 1.0),
            tex.DType.kFloat8E5M2,
            _2X_ACC_DGRAD,
        ).T
    else:
        dgrad_gradient = (
            _fake_cast(gradient, dgrad_gradient_fake_quant, _scalar(dgrad_gradient_scale))
            if dgrad_gradient_fake_quant is not None
            else gradient
        )

        dgrad_weight = (
            _fake_cast(weight, dgrad_weight_fake_quant, _scalar(dgrad_weight_scale))
            if dgrad_weight_fake_quant is not None
            else weight
        )
        dgrad = dgrad_gradient @ dgrad_weight

    if wgrad_fp8:
        wgrad = _fp8_gemm_kernel(
            input.T,
            _scalar(wgrad_input_scale or 1.0),
            tex.DType.kFloat8E4M3,
            gradient.T,
            _scalar(wgrad_gradient_scale or 1.0),
            tex.DType.kFloat8E5M2,
            _2X_ACC_WGRAD,
        ).T
    else:
        wgrad_gradient = (
            _fake_cast(gradient, wgrad_gradient_fake_quant, _scalar(wgrad_gradient_scale))
            if wgrad_gradient_fake_quant is not None
            else gradient
        )
        wgrad_input = (
            _fake_cast(input, wgrad_input_fake_quant, _scalar(wgrad_input_scale))
            if wgrad_input_fake_quant is not None
            else input
        )
        wgrad_input = wgrad_input.contiguous()
        wgrad_gradient = wgrad_gradient.contiguous()
        wgrad, *_ = tepytorch.cpp_extensions.general_gemm(
            wgrad_input,
            wgrad_gradient,
            tepytorch.module.base.get_workspace(),
            torch.float32,
            layout="NT",
            grad=True,
            use_split_accumulator=_2X_ACC_WGRAD,
        )

    return {"activation": activation, "wgrad": wgrad, "dgrad": dgrad}


def _init_debug(config_name, log_dir, feature_dirs):
    debug_api.initialize(
        config_file=config_name,
        feature_dirs=feature_dirs,
        log_dir=log_dir,
        default_logging_enabled=True,
    )


def create_config_file(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    kwargs["config_file"] = temp_file
                    kwargs["log_dir"] = temp_dir
                    result = func(*args, **kwargs)
                finally:
                    temp_file_name = temp_file.name
                    debug_api.end_debug()
            os.unlink(temp_file_name)
        return result

    return wrapper


def _cmp(ground_truth, output):
    torch.testing.assert_close(ground_truth["activation"], output["activation"])
    torch.testing.assert_close(ground_truth["wgrad"], output["wgrad"])
    torch.testing.assert_close(ground_truth["dgrad"], output["dgrad"])


def _init_model(weight):
    model = transformer_engine.pytorch.Linear(IN_SIZE, OUT_SIZE, name="linear")
    with torch.no_grad():
        model.weight.copy_(weight.contiguous())
    return model


def _run_forward_backward(x, model, loss_scale=1.0, is_first_microbatch=None):
    with tepytorch.fp8_autocast(enabled=True, fp8_recipe=FP8_RECIPE):
        y = model(x, is_first_microbatch=is_first_microbatch)
    (y.sum() * loss_scale).backward()
    debug_api.step()
    return y


def _get_tensors():
    torch.manual_seed(SEED)
    x = torch.randn((SEQ_LEN * BATCH_SIZE, IN_SIZE), requires_grad=True).cuda()
    x.retain_grad()
    weight = torch.randn((OUT_SIZE, IN_SIZE)).cuda()
    return x, weight


DISABLE_FP8_CONFIG = Template(
    """disable_fp8_config:
  enabled: True
  layers:
    layer_types: [linear]
  transformer_engine:
    DisableFP8GEMM:
      enabled: True
      gemms: [$gemms]
"""
)


@pytest.mark.parametrize("fprop_fp8", all_boolean)
@pytest.mark.parametrize("dgrad_fp8", all_boolean)
@pytest.mark.parametrize("wgrad_fp8", all_boolean)
def test_disable_fp8_gemms(feature_dirs, fprop_fp8, dgrad_fp8, wgrad_fp8):
    run_disable_fp8_gemms(feature_dirs, fprop_fp8, dgrad_fp8, wgrad_fp8)


def disable_fp8_gemms_create_config(fprop_fp8, dgrad_fp8, wgrad_fp8, config_file):
    gemms = ""
    if not fprop_fp8:
        gemms += "fprop,"
    if not dgrad_fp8:
        gemms += "dgrad,"
    if not wgrad_fp8:
        gemms += "wgrad,"
    if len(gemms) > 0:
        gemms = gemms[:-1]  # remove last ','
    config_file.write(DISABLE_FP8_CONFIG.safe_substitute(gemms=gemms))
    config_file.flush()


@create_config_file
def run_disable_fp8_gemms(feature_dirs, fprop_fp8, dgrad_fp8, wgrad_fp8, **kwargs):
    disable_fp8_gemms_create_config(fprop_fp8, dgrad_fp8, wgrad_fp8, kwargs["config_file"])
    fp8_kwargs = {
        "fprop_fp8": fprop_fp8,
        "dgrad_fp8": dgrad_fp8,
        "wgrad_fp8": wgrad_fp8,
    }

    _init_debug(kwargs["config_file"].name, kwargs["log_dir"], feature_dirs)
    x, weight = _get_tensors()
    model = _init_model(weight)
    y = _run_forward_backward(x, model)

    output = {"activation": y.clone(), "wgrad": model.weight.grad.clone(), "dgrad": x.grad.clone()}

    x.grad.zero_()
    ground_truth = _emulate_linear(x, weight, **fp8_kwargs)
    _cmp(ground_truth, output)


def test_disable_fp8_layer(feature_dirs):
    run_disable_fp8_layer(feature_dirs)


DISABLE_FP8_LAYER_CONFIG = """disable_fp8_config:
  enabled: True
  layers:
    layer_types: [linear]
  transformer_engine:
    DisableFP8Layer:
      enabled: True
"""


@create_config_file
def run_disable_fp8_layer(feature_dirs, **kwargs):
    kwargs["config_file"].write(DISABLE_FP8_LAYER_CONFIG)
    kwargs["config_file"].flush()

    x, weight = _get_tensors()

    ground_truth = _emulate_linear(x, weight)
    x.grad.zero_()

    _init_debug(kwargs["config_file"].name, kwargs["log_dir"], feature_dirs)

    model = _init_model(weight)
    y = _run_forward_backward(x, model)

    output = {"activation": y.clone(), "wgrad": model.weight.grad.clone(), "dgrad": x.grad.clone()}
    _cmp(ground_truth, output)


random.seed(1234)

all_combinations = list(itertools.product(all_boolean, repeat=6))
subset_combinations = random.sample(all_combinations, 20)


@pytest.mark.parametrize(
    "fprop_inp, fprop_weight, dgrad_weight, dgrad_grad, wgrad_input, wgrad_grad",
    subset_combinations,
)
def test_per_tensor_scaling(
    feature_dirs, fprop_inp, fprop_weight, dgrad_weight, dgrad_grad, wgrad_input, wgrad_grad
):
    if not any([fprop_inp, fprop_weight, dgrad_weight, dgrad_grad, wgrad_input, wgrad_grad]):
        pytest.skip("Skipping test because all parameters are False")
    run_per_tensor_scaling(
        feature_dirs, fprop_inp, fprop_weight, dgrad_weight, dgrad_grad, wgrad_input, wgrad_grad
    )


PER_TENSOR_SCALING_CONFIG = Template(
    """per_tensor_scaling_config:
  enabled: True
  layers:
    layer_types: [linear]
  transformer_engine:
    PerTensorScaling:
      enabled: True
      gemms_struct:
$gemms
"""
)


def _prepare_per_tensor_scaling_config(
    fprop_inp, fprop_weight, dgrad_weight, dgrad_grad, wgrad_input, wgrad_grad, config_file
):
    gemms = ""
    title = lambda x: f"      - gemm: {x}\n        tensors: ["

    def add_tensor(if_add, gemm_name):
        nonlocal gemms
        if if_add:
            gemms += gemm_name + ","

    if fprop_inp or fprop_weight:
        gemms += title("fprop")
        add_tensor(fprop_inp, "activation")
        add_tensor(fprop_weight, "weight")
        gemms = gemms[:-1] + "]\n"
    if dgrad_weight or dgrad_grad:
        gemms += title("dgrad")
        add_tensor(dgrad_weight, "weight")
        add_tensor(dgrad_grad, "gradient")
        gemms = gemms[:-1] + "]\n"
    if wgrad_input or wgrad_grad:
        gemms += title("wgrad")
        add_tensor(wgrad_input, "activation")
        add_tensor(wgrad_grad, "gradient")
        gemms = gemms[:-1] + "]\n"
    config_file.write(PER_TENSOR_SCALING_CONFIG.safe_substitute(gemms=gemms))
    config_file.flush()


def set_scaling_factors(model, input_kwargs, fp8_kwargs):
    # Copy fp8 scaling factors into fp8_kwargs dict if respective flag in input_kwargs is set.
    if not input_kwargs["fprop_inp"]:
        fp8_kwargs["fprop_input_scale"] = model.fp8_meta["scaling_fwd"].scale[0].clone()
    if not input_kwargs["fprop_weight"]:
        fp8_kwargs["fprop_weight_scale"] = model.fp8_meta["scaling_fwd"].scale[1].clone()
    if not input_kwargs["dgrad_grad"]:
        fp8_kwargs["dgrad_gradient_scale"] = model.fp8_meta["scaling_bwd"].scale[0].clone()
    if not input_kwargs["dgrad_weight"]:
        fp8_kwargs["dgrad_weight_scale"] = model.fp8_meta["scaling_fwd"].scale[1].clone()
    if not input_kwargs["wgrad_grad"]:
        fp8_kwargs["wgrad_gradient_scale"] = model.fp8_meta["scaling_bwd"].scale[0].clone()
    if not input_kwargs["wgrad_input"]:
        fp8_kwargs["wgrad_input_scale"] = model.fp8_meta["scaling_fwd"].scale[0].clone()


def set_current_scaling_factors(x, weight, y, input_kwargs, fp8_kwargs):
    # Compute per tensor scaling factor if respective flag in input_kwargs is set.
    if input_kwargs["fprop_inp"]:
        fp8_kwargs["fprop_input_scale"] = tex.DType.kFloat8E4M3
    if input_kwargs["fprop_weight"]:
        fp8_kwargs["fprop_weight_scale"] = tex.DType.kFloat8E4M3
    if input_kwargs["dgrad_grad"]:
        fp8_kwargs["dgrad_gradient_scale"] = tex.DType.kFloat8E5M2
    if input_kwargs["dgrad_weight"]:
        fp8_kwargs["dgrad_weight_scale"] = tex.DType.kFloat8E4M3
    if input_kwargs["wgrad_grad"]:
        fp8_kwargs["wgrad_gradient_scale"] = tex.DType.kFloat8E5M2
    if input_kwargs["wgrad_input"]:
        fp8_kwargs["wgrad_input_scale"] = tex.DType.kFloat8E4M3


@create_config_file
def run_per_tensor_scaling(
    feature_dirs,
    fprop_inp,
    fprop_weight,
    dgrad_weight,
    dgrad_grad,
    wgrad_input,
    wgrad_grad,
    **kwargs,
):
    input_kwargs = {
        "fprop_inp": fprop_inp,
        "fprop_weight": fprop_weight,
        "dgrad_weight": dgrad_weight,
        "dgrad_grad": dgrad_grad,
        "wgrad_input": wgrad_input,
        "wgrad_grad": wgrad_grad,
    }
    fp8_kwargs = {
        "fprop_fp8": True,
        "dgrad_fp8": True,
        "wgrad_fp8": True,
    }
    """
        Runs a test to validate per-tensor (current) scaling in FP8 computations.
        The function performs warm-up iterations to populate the amax buffer of the model and compute scaling factors based on delayed scaling.
        Subsequently, weights and inputs are switched to ensure their current scaling factors differ from those based on delayed scaling;
        similarly, the loss is multiplied by a large factor to alter the gradient's magnitude,
        creating a discrepancy between the original (delayed) and per-tensor (current) scaling factors.
        Finally, a linear pass is emulated, and the results are compared.”
    """
    _prepare_per_tensor_scaling_config(
        fprop_inp,
        fprop_weight,
        dgrad_weight,
        dgrad_grad,
        wgrad_input,
        wgrad_grad,
        kwargs["config_file"],
    )
    _init_debug(kwargs["config_file"].name, kwargs["log_dir"], feature_dirs)

    warmup_input, warmup_weight = _get_tensors()
    model = _init_model(warmup_weight)

    # Warmup run to setup amax and scaling factors.
    for _ in range(AMAX_HISTORY_LEN):
        _run_forward_backward(warmup_input, model)

    x = torch.randn_like(warmup_input, requires_grad=True).cuda()
    weight = torch.randn_like(warmup_weight, requires_grad=True).cuda()
    model.weight.data = weight.data
    x.retain_grad()

    # delayed scaling factor
    # need to be collected before forward pass with test data,
    # because this forward pass changes scaling factors
    set_scaling_factors(model, input_kwargs, fp8_kwargs)

    LOSS_MULTIPLIER = 100

    with tepytorch.fp8_autocast(enabled=True, fp8_recipe=FP8_RECIPE):
        y = model(x, is_first_microbatch=True)
        model.zero_grad()
        y.retain_grad()
        (
            LOSS_MULTIPLIER * y.sum()
        ).backward()  # Loss multiplication to change gradient's order of magintude

    output = {"activation": y.clone(), "wgrad": model.weight.grad.clone(), "dgrad": x.grad.clone()}

    # per tensor - current - scaling factors
    # need to be collected after forward pass with test data,
    # because gradient(y.grad) cannot be accessed before forward,
    # but it needs to be collected.
    set_current_scaling_factors(x, weight, y, input_kwargs, fp8_kwargs)

    ground_truth = _emulate_linear(x, weight, loss_multiplier=LOSS_MULTIPLIER, **fp8_kwargs)
    _cmp(ground_truth, output)


@pytest.mark.parametrize(
    "fprop_inp, fprop_weight, dgrad_weight, dgrad_grad, wgrad_input, wgrad_grad",
    subset_combinations,
)
def test_microbatching_per_tensor_scaling(
    feature_dirs, fprop_inp, fprop_weight, dgrad_weight, dgrad_grad, wgrad_input, wgrad_grad
):
    if not any([fprop_inp, fprop_weight, dgrad_weight, dgrad_grad, wgrad_input, wgrad_grad]):
        pytest.skip("Skipping test because all parameters are False")

    @create_config_file
    def run_microbatching_test(
        feature_dirs,
        fprop_inp,
        fprop_weight,
        dgrad_weight,
        dgrad_grad,
        wgrad_input,
        wgrad_grad,
        **kwargs,
    ):
        # Prepare the configuration file
        _prepare_per_tensor_scaling_config(
            fprop_inp,
            fprop_weight,
            dgrad_weight,
            dgrad_grad,
            wgrad_input,
            wgrad_grad,
            kwargs["config_file"],
        )

        # Initialize debug
        _init_debug(kwargs["config_file"].name, kwargs["log_dir"], feature_dirs)

        # Get data
        x_full, weight = _get_tensors()
        microbatch_size = x_full.size(0) // 2
        x_mb1 = x_full[:microbatch_size, ...].clone().detach().requires_grad_(True)
        x_mb2 = x_full[microbatch_size:, ...].clone().detach().requires_grad_(True)

        def init_and_warmup():
            model = _init_model(weight)
            _run_forward_backward(x_mb1, model, loss_scale=0.5)
            _run_forward_backward(x_mb2, model, loss_scale=0.5)
            return model

        # Run without is_first_microbatch

        model = init_and_warmup()  # running next 2 iters does not change amaxes and scaling factors
        y_mb1 = _run_forward_backward(x_mb1, model, loss_scale=0.5)
        y_mb2 = _run_forward_backward(x_mb2, model, loss_scale=0.5)

        # Collect outputs
        output1 = {
            "activation": torch.cat([y_mb1.clone(), y_mb2.clone()], dim=0),
            "wgrad": model.weight.grad.clone(),
            "dgrad": torch.cat([x_mb1.grad.clone(), x_mb2.grad.clone()], dim=0),
        }

        # Run with is_first_microbatch
        model = init_and_warmup()  # running next 2 iters does not change amaxes and scaling factors
        y_mb1 = _run_forward_backward(x_mb1, model, loss_scale=0.5, is_first_microbatch=True)
        y_mb2 = _run_forward_backward(x_mb2, model, loss_scale=0.5, is_first_microbatch=False)

        # Collect outputs
        output2 = {
            "activation": torch.cat([y_mb1.clone(), y_mb2.clone()], dim=0),
            "wgrad": model.weight.grad.clone(),
            "dgrad": torch.cat([x_mb1.grad.clone(), x_mb2.grad.clone()], dim=0),
        }

        # Compare outputs
        torch.testing.assert_close(output1["activation"], output2["activation"], atol=1.0, rtol=0.5)
        torch.testing.assert_close(output1["dgrad"], output2["dgrad"], atol=1.0, rtol=0.5)
        torch.testing.assert_close(output1["wgrad"], output2["wgrad"], atol=1.0, rtol=0.5)

    # Run the test
    run_microbatching_test(
        feature_dirs, fprop_inp, fprop_weight, dgrad_weight, dgrad_grad, wgrad_input, wgrad_grad
    )


all_combinations = list(
    itertools.product([tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2, None], repeat=6)
)
subset_combinations = random.sample(all_combinations, 10)


@pytest.mark.parametrize(
    "fprop_inp, fprop_weight, dgrad_weight, dgrad_grad, wgrad_input, wgrad_grad",
    subset_combinations,
)
def test_fake_quant_fp8(
    feature_dirs, fprop_inp, fprop_weight, dgrad_weight, dgrad_grad, wgrad_input, wgrad_grad
):
    run_fake_quant_fp8(
        feature_dirs, fprop_inp, fprop_weight, dgrad_weight, dgrad_grad, wgrad_input, wgrad_grad
    )


FAKE_QUANT_CONFIG = Template(
    """fake_quant_config:
  enabled: True
  layers:
    layer_types: [linear]
  transformer_engine:
    FakeQuant:
      enabled: True
      gemms_struct:
$gemms
"""
)


def fake_quant_fp8_create_config(
    fprop_inp, fprop_weight, dgrad_weight, dgrad_grad, wgrad_input, wgrad_grad, config_file
):
    format_to_str = {tex.DType.kFloat8E4M3: "FP8E4M3", tex.DType.kFloat8E5M2: "FP8E5M2"}
    gemms = ""

    def _add_tensor(quant_format, tensor):
        nonlocal gemms
        if quant_format:
            gemms += " " * 8 + "- tensor: " + tensor + "\n"
            gemms += " " * 8 + "  quant_format: " + format_to_str[quant_format] + "\n"

    title = lambda x: f"      - gemm: {x}\n        tensors_struct:\n"
    if fprop_inp or fprop_weight:
        gemms += title("fprop")
        _add_tensor(fprop_inp, "activation")
        _add_tensor(fprop_weight, "weight")
        gemms = gemms[:-1] + "\n"
    if dgrad_weight or dgrad_grad:
        gemms += title("dgrad")
        _add_tensor(dgrad_weight, "weight")
        _add_tensor(dgrad_grad, "gradient")
        gemms = gemms[:-1] + "\n"
    if wgrad_input or wgrad_grad:
        gemms += title("wgrad")
        _add_tensor(wgrad_input, "activation")
        _add_tensor(wgrad_grad, "gradient")
        gemms = gemms[:-1] + "\n"
    config = FAKE_QUANT_CONFIG.safe_substitute(gemms=gemms)
    config_file.write(config)
    config_file.flush()


@create_config_file
def run_fake_quant_fp8(
    feature_dirs,
    fprop_inp,
    fprop_weight,
    dgrad_weight,
    dgrad_grad,
    wgrad_input,
    wgrad_grad,
    **kwargs,
):
    fp8_kwargs = {
        "fprop_input_fake_quant": fprop_inp,
        "fprop_weight_fake_quant": fprop_weight,
        "dgrad_gradient_fake_quant": dgrad_grad,
        "dgrad_weight_fake_quant": dgrad_weight,
        "wgrad_gradient_fake_quant": wgrad_grad,
        "wgrad_input_fake_quant": wgrad_input,
        "fprop_fp8": not (fprop_inp or fprop_weight),
        "dgrad_fp8": not (dgrad_weight or dgrad_grad),
        "wgrad_fp8": not (wgrad_grad or wgrad_input),
    }
    fake_quant_fp8_create_config(
        fprop_inp,
        fprop_weight,
        dgrad_weight,
        dgrad_grad,
        wgrad_input,
        wgrad_grad,
        kwargs["config_file"],
    )
    _init_debug(kwargs["config_file"].name, kwargs["log_dir"], feature_dirs)

    x, weight = _get_tensors()
    model = _init_model(weight)
    y = _run_forward_backward(x, model)

    output = {"activation": y.clone(), "wgrad": model.weight.grad.clone(), "dgrad": x.grad.clone()}
    ground_truth = _emulate_linear(x, weight, **fp8_kwargs)
    _cmp(ground_truth, output)
