# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
This file contains tests for exporting TransformerEngine models to ONNX.

The purpose of these tests is validation that TE models are converted to their correct ONNX
representation. Toward this end, each test captures the output of a TE module forward pass,
converts the TE module to ONNX, and uses ONNX Runtime (ORT) to execute the ONNX graph and
validate the output against TE's output.

Until FP8 is introduced to the ONNX standard, FP8 QuantizeLinear/DequantizeLinear is implemented
using custom ORT operations.

To run many repetitive tests use pytest-loop:
    $ python3 -m pip install pytest-loop
    $ pytest --loop 1000 tests/pytorch/test_onnx_export.py::test_export_layernorm

For reproducibility use: torch.manual_seed(0)
"""

import os
import tempfile
import pytest
import warnings
import numpy as np
import onnxruntime as ort
import torch
from torch import nn as nn
from typing import Optional, Union, Tuple, List
from onnxruntime_extensions import PyCustomOpDef, get_library_path, onnx_op
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
import transformer_engine_torch as tex
from transformer_engine.pytorch.export import is_in_onnx_export_mode, te_translation_table
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.utils import get_default_init_method

# Global test configuration knobs.

# Enable this to serialize test inputs and outputs to file (as a Polygraphy RunResults instance).
SAVE_TEST_IO = bool(int(os.getenv("NVTE_ONNX_EXPORT_SAVE_TEST_IO", "0")))

if SAVE_TEST_IO:
    from polygraphy.json import save_json
    from polygraphy.comparator import RunResults

# The directory where generated ONNX test models are stored.
NVTE_TEST_ARTIFACTS_DIR = os.environ.get("NVTE_TEST_ARTIFACTS_DIR")
NVTE_TEST_ARTIFACTS_DIR = NVTE_TEST_ARTIFACTS_DIR or os.path.join(
    tempfile.gettempdir(), "./gen_onnx_models"
)


# The directory where this file is stored.
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()
mxfp8_available, reason_for_no_mxfp8 = FP8GlobalStateManager.is_mxfp8_available()

fp8_recipes = []
if mxfp8_available:
    fp8_recipes.append(recipe.MXFP8BlockScaling())
if fp8_available:
    fp8_recipes.append(recipe.DelayedScaling())
fp8_recipes.append(None)

supported_activations = ["gelu", "relu", "reglu", "geglu", "swiglu"]

all_normalizations = ["LayerNorm", "RMSNorm"]


@onnx_op(
    op_type="trt::TRT_FP8QuantizeLinear",
    domain="trt",
    inputs=[
        PyCustomOpDef.dt_float,
        PyCustomOpDef.dt_float,
    ],
    outputs=[PyCustomOpDef.dt_uint8],
)
def trt_fp8_quantize(t, scale):
    """FP8 quantization extension for ONNX Runtime."""
    x = torch.from_numpy(t).cuda()
    q = te.tensor.float8_tensor.Float8Quantizer(
        scale=1 / torch.from_numpy(scale).cuda(),
        amax=torch.zeros([1]).cuda(),
        fp8_dtype=tex.DType.kFloat8E4M3,
    )
    return q(x)._data.cpu().numpy()


@onnx_op(
    op_type="trt::TRT_FP8DequantizeLinear",
    domain="trt",
    inputs=[
        PyCustomOpDef.dt_uint8,
        PyCustomOpDef.dt_float,
    ],
    outputs=[PyCustomOpDef.dt_float],
)
def trt_fp8_dequantize(t, scale):
    """FP8 dequantization extension for ONNX Runtime."""
    x = torch.from_numpy(t).cuda()
    q = te.tensor.float8_tensor.Float8Quantizer(
        scale=1 / torch.from_numpy(scale).cuda(),
        amax=torch.zeros([1]).cuda(),
        fp8_dtype=tex.DType.kFloat8E4M3,
    )
    quantizer_tensor = q.create_tensor_from_data(x, fake_dtype=torch.float32)
    return quantizer_tensor.dequantize().cpu().numpy()


@onnx_op(
    op_type="trt::TRT_MXFP8QuantizeLinear",
    domain="trt",
    inputs=[
        PyCustomOpDef.dt_float,
    ],
    outputs=[PyCustomOpDef.dt_uint8, PyCustomOpDef.dt_uint8],
)
def trt_mxfp8_quantize(t):
    """MXFP8 quantization extension for ONNX Runtime."""
    x = torch.from_numpy(t).cuda()
    q = te.tensor.mxfp8_tensor.MXFP8Quantizer(tex.DType.kFloat8E4M3)
    return q(x)._rowwise_data.cpu().numpy(), q(x)._rowwise_scale_inv.cpu().numpy()


@onnx_op(
    op_type="trt::TRT_MXFP8DequantizeLinear",
    domain="trt",
    inputs=[
        PyCustomOpDef.dt_uint8,
        PyCustomOpDef.dt_uint8,
    ],
    outputs=[PyCustomOpDef.dt_float],
)
def trt_mxfp8_dequantize(t, scale_inv):
    """MXFP8 dequantization extension for ONNX Runtime."""
    x = torch.from_numpy(t).cuda()
    scale_inv_tensor = torch.from_numpy(scale_inv).cuda()
    q = te.tensor.mxfp8_tensor.MXFP8Quantizer(tex.DType.kFloat8E4M3)
    quantizer_tensor = q.create_tensor_from_data(x, scale_inv_tensor, fake_dtype=torch.float32)
    return quantizer_tensor.dequantize().cpu().numpy()


@pytest.fixture()
def seed_default_rng():
    """Reseed the PRNG for test reproducibility"""
    torch.manual_seed(1234)


@pytest.fixture()
def set_max_seq_len(max_seq_len=128):
    """Set the maximum sequence length that can be used for attention masking"""
    os.environ["NVTE_ONNX_KVCACHE_MAX_SEQ_LEN"] = f"{max_seq_len}"


@pytest.fixture(autouse=True)
def reset_global_fp8_state():
    yield
    FP8GlobalStateManager.reset()


def do_export(
    model: torch.nn.Module,
    inp: torch.Tensor,
    fname: str,
    fp8_recipe: recipe.Recipe,
    input_names: List[str] = None,
    output_names: List[str] = None,
    dynamic_shapes: List[str] = None,
):
    """Export to ONNX"""
    input_names = input_names or ["input"]
    output_names = output_names or ["output"]

    with torch.inference_mode(), te.fp8_autocast(
        enabled=fp8_recipe is not None, fp8_recipe=fp8_recipe
    ), warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=torch.jit.TracerWarning, module=r".*")

        model.cuda().eval()
        os.makedirs(NVTE_TEST_ARTIFACTS_DIR, exist_ok=True)
        fname = os.path.join(NVTE_TEST_ARTIFACTS_DIR, fname)

        inps = inp if isinstance(inp, list) or isinstance(inp, tuple) else (inp,)
        assert len(inps) == len(input_names)
        inds_to_del = [i for i in range(len(inps)) if inps[i] is None]
        input_names = [input_names[i] for i in range(len(inps)) if i not in inds_to_del]

        model(*inps)  # warm-up run
        with te.export.onnx_export(True):
            model(*inps)
        with te.export.onnx_export(True):
            torch.onnx.export(
                model,
                inps,
                fname,
                dynamo=True,
                custom_translation_table=te_translation_table,
                verbose=True,
                dynamic_shapes=dynamic_shapes,
                input_names=input_names,
                output_names=output_names,
                optimize=inps[0].dtype
                != torch.bfloat16,  # optimizer does not work with bfloat16 yet - will need to change that after onnxscript supports bfloat16
            )


def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.type(torch.float32)
        tensor = tensor.detach().cpu().numpy()
    return tensor


def set_layer_scale(module: torch.nn.Module, scale: float, num_gemms: int):
    """Initialize the FP8 quantization scales in module"""
    module.init_fp8_metadata(num_gemms)
    for quantizer in module.quantizers["scaling_fwd"]:
        quantizer.scale = torch.ones(1, dtype=torch.float32, device="cuda") * scale


def te_infer(
    model: torch.nn.Module,
    inps: Union[Tuple[torch.Tensor], torch.Tensor],
    is_fp8: bool,
    fp8_recipe: recipe.Recipe,
):
    """Transformer Engine forward propagation."""
    with torch.inference_mode(), te.fp8_autocast(
        enabled=is_fp8, fp8_recipe=fp8_recipe
    ), warnings.catch_warnings():
        te_outputs = model(*inps if isinstance(inps, tuple) else (inps,))
        if not isinstance(te_outputs, tuple):
            te_outputs = (te_outputs,)
        return te_outputs


def compare_outputs(
    onnx_outputs, te_outputs, atol, rtol, max_errors_printed, allow_cnt_errors, fname
):
    """Compare ORT and TE outputs."""
    assert len(onnx_outputs) == len(te_outputs)
    # Compare ORT and PyTorch outputs.
    for onnx_output, te_output in zip(onnx_outputs, te_outputs):
        # np.isclose: abs(a - b) <= (atol + rtol * abs(b))
        te_output = to_numpy(te_output)
        onnx_output = to_numpy(onnx_output)
        ac = ~np.isclose(onnx_output, te_output, atol=atol, rtol=rtol)
        mismatches = ac.nonzero()
        mismatched_ids = [loc for loc in zip(*mismatches)]
        if mismatched_ids:
            # Log some information in case of error.
            print("*" * 100)
            nb_errors = len(mismatched_ids)
            nb_vals = min(nb_errors, max_errors_printed)
            print(f"Detected {nb_errors} diverging values (output shape={onnx_output.shape})")
            print(f"Showing first {nb_vals} errors (ONNX -- TE):")
            abs_err = np.abs(onnx_output - te_output)
            errors = abs_err[mismatches]
            for loc in mismatched_ids[:nb_vals]:
                ref = te_output[loc]
                print(
                    f"{onnx_output[loc]} -- {te_output[loc]} err={abs_err[loc]} >"
                    f" {atol + rtol * abs(ref)}"
                )
            print(f"Max error: {np.max(errors)}")
            if nb_errors > allow_cnt_errors:
                raise ValueError(f"Output validation of {fname} failed with {nb_errors} errors")


def serialize_inputs_outputs(
    fname: str,
    inputs: Union[Tuple[torch.Tensor], torch.Tensor],
    te_outputs: List[torch.Tensor],
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
):
    if not SAVE_TEST_IO:
        return

    fname = os.path.join(NVTE_TEST_ARTIFACTS_DIR, fname)

    input_names = input_names or ["input"]
    output_names = output_names or ["output"]
    inputs = inputs if isinstance(inputs, list) or isinstance(inputs, tuple) else (inputs,)
    named_inputs = zip(input_names, inputs)
    input_data = [{k: v.cpu() for k, v in named_inputs if v is not None}]
    json_fname = fname[: -len(".onnx")] + "_inputs.json"
    save_json(input_data, json_fname, description="custom input data")

    json_fname = fname[: -len(".onnx")] + "_output.json"
    named_outputs = zip(output_names, te_outputs)
    output_data = {k: v.detach().cpu() for k, v in named_outputs if v is not None}
    custom_outputs = RunResults()
    custom_outputs.add([output_data], runner_name="custom_runner")
    custom_outputs.save(json_fname)


def validate_result(
    fname: str,
    inps: Union[Tuple[torch.Tensor], torch.Tensor],
    model: torch.nn.Module,
    atol: float = 1.0e-8,  # np.isclose default atol
    rtol: float = 1.0e-5,  # np.isclose default rtol
    max_errors_printed: int = 10,
    is_fp8: bool = False,
    allow_cnt_errors: int = 0,
    input_names: List[str] = None,
    output_names: List[str] = None,
    te_outputs: List[torch.Tensor] = None,
):
    """Compare the outputs of a Transformer Engine (TE) module vs the outputs of its ONNX
    representation using ONNX Runtime (ORT) and ensure they are close.

    The purpose of the output comparison is to validate that TE models are converted to
    their correct ONNX representation by testing that TE and ORT outputs match within some
    small threshold (allowing for finite precision errors).

    Argument `allow_cnt_errors` reduces test failure noise due to spurious errors by ignoring,
    a very small number (0-3) of outliers. This is fine to do because these outliers are due to
    small kernel implementation differences between TE and ORT and do not imply an incorrect ONNX
    representation (the tests assume both ORT or TE kernels are correct).

    Argument `te_outputs` can be used to provide pre-computed TE outputs.
    """

    def create_ort_session(fname: str, is_fp8: bool):
        def load_custom_ops(session_opts: ort.SessionOptions):
            """For FP8 validation with ORT we need to load our custom FP8 Q/DQ extension."""
            session_opts.register_custom_ops_library(get_library_path())
            print("registered custom FP8 Q/DQ ops!")

        """Create an ONNX Runtime session for validation."""
        kwargs = {"providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]}
        if is_fp8:
            sess_options = ort.SessionOptions()
            load_custom_ops(sess_options)
            kwargs["sess_options"] = sess_options

        s = ort.InferenceSession(fname, **kwargs)
        return s

    def create_ort_input_dict(session, inputs):
        inputs = inputs if isinstance(inputs, list) or isinstance(inputs, tuple) else (inputs,)
        input_names = [x.name for x in session.get_inputs()]
        inps = [to_numpy(x) for x in inputs if x is not None]
        inp_dict = dict(zip(input_names, inps))
        return inp_dict

    input_names = input_names or ["input"]
    output_names = output_names or ["output"]

    # Run ORT session and TE model.
    fname = os.path.join(NVTE_TEST_ARTIFACTS_DIR, fname)
    if not te_outputs:
        te_outputs = te_infer(model, inps, is_fp8)
    ort_s = create_ort_session(fname, is_fp8)
    input_feed = create_ort_input_dict(ort_s, inps)
    onnx_outputs = ort_s.run(None, input_feed=input_feed)
    compare_outputs(
        onnx_outputs, te_outputs, atol, rtol, max_errors_printed, allow_cnt_errors, fname
    )


def dtype2str(dtype: torch.dtype, fake_bf16_io=False):
    if fake_bf16_io:
        assert dtype == torch.bfloat16
        return "_fake_bf16"
    return {
        torch.float32: "_fp32",
        torch.float16: "_fp16",
        torch.bfloat16: "_bf16",
    }[dtype]


def as_te_type(dtype: torch.dtype):
    return {
        torch.float32: tex.DType.kFloat32,
        torch.float16: tex.DType.kFloat16,
        torch.bfloat16: tex.DType.kBFloat16,
    }[dtype]


def get_attn_mask_str(use_mask, attn_mask_type):
    # See FusedScaleMaskSoftmax::forward_fused_softmax for logic behind names.
    if attn_mask_type is None:
        return "_mask" if use_mask else "_no-mask"
    attn_mask_str = "_arbitrary-no-mask"
    attn_mask_str = "_causal-mask" if attn_mask_type == "causal" else attn_mask_str
    attn_mask_str = (
        "_arbitrary-mask" if use_mask and attn_mask_type == "arbitrary" else attn_mask_str
    )
    return attn_mask_str


"""
Test cases begin here.
"""


def _test_export_linear(
    fp8_recipe: recipe.Recipe = fp8_recipes[0],
    use_bias: bool = True,
    return_bias: bool = False,
    precision: torch.dtype = torch.float32,
):
    if return_bias and not use_bias:
        pytest.skip("Cannot return bias when bias is disabled")

    # Set dimensions (these are arbitrary).
    batch_size = 4
    in_features = 64
    out_features = 64
    hidden_size = 64

    class Test_Linear(nn.Module):
        def __init__(self, in_features, out_features, use_bias, return_bias, precision):
            super().__init__()
            self.linear = te.Linear(
                in_features,
                out_features,
                bias=use_bias,
                return_bias=return_bias,
                params_dtype=precision,
            )

        def forward(self, inp):
            ret = self.linear(inp)
            return ret

    inp = torch.randn(batch_size, hidden_size, in_features, device="cuda", dtype=precision)
    fp8_str = "_fp8" if fp8_recipe is not None else ""
    bias_str = "_bias" if use_bias else ""
    high_prec_str = dtype2str(precision)
    fname = f"te.linear{fp8_str}{bias_str}{high_prec_str}.onnx"
    with te.fp8_autocast(enabled=fp8_recipe is not None, fp8_recipe=fp8_recipe):
        model = Test_Linear(in_features, out_features, use_bias, return_bias, precision).to(
            device="cuda"
        )
        # dynamic shape
        bs = torch.export.Dim("bs", min=2, max=1256)
        do_export(
            model,
            inp,
            fname,
            fp8_recipe,
            dynamic_shapes={"inp": {0: bs}},
        )
        te_outputs = te_infer(model, inp, is_fp8=fp8_recipe is not None, fp8_recipe=fp8_recipe)
        serialize_inputs_outputs(fname, inp, te_outputs)

        if precision in (torch.bfloat16,):
            return
        if fp8_recipe is None:
            validate_result(fname, inp, model, atol=1e-3, te_outputs=te_outputs)
        else:
            validate_result(
                fname, inp, model, atol=1e-2, is_fp8=fp8_recipe is not None, te_outputs=te_outputs
            )


@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("precision", [torch.float32, torch.float16, torch.bfloat16])
def test_export_linear_recipe(seed_default_rng, fp8_recipe, precision):
    _test_export_linear(fp8_recipe=fp8_recipe, precision=precision)


@pytest.mark.parametrize("use_bias", [True, False])
def test_export_linear_use_bias(seed_default_rng, use_bias):
    _test_export_linear(use_bias=use_bias)


@pytest.mark.parametrize("return_bias", [True, False])
def test_export_linear_return_bias(seed_default_rng, return_bias):
    _test_export_linear(return_bias=return_bias)


def _test_export_layernorm(
    fp8_recipe: recipe.Recipe = fp8_recipes[0],
    precision: torch.dtype = torch.float32,
    zero_centered_gamma: bool = False,
    normalization: str = all_normalizations[0],
):
    # Set dimensions (these are arbitrary).
    batch_size = 4
    in_features = 64
    out_features = 256
    hidden_size = 256

    inp = torch.ones(batch_size, in_features, out_features, device="cuda", dtype=precision)
    fp8_str = "_fp8" if fp8_recipe is not None else ""
    high_prec_str = dtype2str(precision)
    fname = f"te.layernorm_linear{fp8_str}{high_prec_str}.onnx"

    with torch.no_grad():
        with te.fp8_autocast(enabled=fp8_recipe is not None, fp8_recipe=fp8_recipe):
            layernorm_cls = te.LayerNorm if normalization == "LayerNorm" else te.RMSNorm
            model = layernorm_cls(
                hidden_size,
                params_dtype=precision,
                zero_centered_gamma=zero_centered_gamma,
            ).to(device="cuda")

            # dynamic shape
            bs = torch.export.Dim("bs", min=2, max=1256)
            do_export(model, inp, fname, fp8_recipe, dynamic_shapes={"input": {0: bs}})
            te_outputs = te_infer(model, inp, is_fp8=fp8_recipe is not None, fp8_recipe=fp8_recipe)
            serialize_inputs_outputs(fname, inp, te_outputs)
            if precision in (torch.bfloat16,):
                return
            if fp8_recipe is None:
                validate_result(fname, inp, model, atol=1e-3, te_outputs=te_outputs)
            elif precision != torch.bfloat16:
                validate_result(
                    fname,
                    inp,
                    model,
                    atol=1e-3,
                    is_fp8=fp8_recipe is not None,
                    te_outputs=te_outputs,
                )


@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("precision", [torch.float32, torch.float16, torch.bfloat16])
def test_export_layernorm_recipe(seed_default_rng, fp8_recipe, precision):
    _test_export_layernorm(fp8_recipe=fp8_recipe, precision=precision)


def test_export_layernorm_zero_centered_gamma(seed_default_rng):
    _test_export_layernorm(zero_centered_gamma=True)


@pytest.mark.parametrize("normalization", all_normalizations)
def test_export_layernorm_normalization(seed_default_rng, normalization):
    _test_export_layernorm(normalization=normalization)


def _test_export_layernorm_linear(
    scale_factor: float = 112,
    fp8_recipe: recipe.Recipe = fp8_recipes[0],
    use_bias: bool = True,
    return_bias: bool = False,
    return_layernorm_output: bool = False,
    precision: torch.dtype = torch.float32,
    zero_centered_gamma: bool = False,
    normalization: str = all_normalizations[0],
):
    if return_bias and not use_bias:
        pytest.skip("Cannot return bias when bias is disabled")

    # Set dimensions (these are arbitrary).
    in_features = 64
    out_features = 256
    hidden_size = 256

    inp = torch.randn(in_features, out_features, device="cuda", dtype=precision)
    fp8_str = "_fp8" if fp8_recipe is not None else ""
    bias_str = "_bias" if use_bias else ""
    high_prec_str = dtype2str(precision)
    fname = f"te.layernorm_linear{fp8_str}{bias_str}{high_prec_str}.onnx"

    with torch.no_grad():
        with te.fp8_autocast(enabled=fp8_recipe is not None, fp8_recipe=fp8_recipe):
            model = te.LayerNormLinear(
                hidden_size,
                3 * hidden_size,
                bias=use_bias,
                return_bias=return_bias,
                return_layernorm_output=return_layernorm_output,
                params_dtype=precision,
                zero_centered_gamma=zero_centered_gamma,
                normalization=normalization,
            ).to(device="cuda")
            if fp8_recipe is not None:
                set_layer_scale(model, scale_factor, num_gemms=2)
            do_export(model, inp, fname, fp8_recipe)

            te_outputs = te_infer(model, inp, is_fp8=fp8_recipe is not None, fp8_recipe=fp8_recipe)
            serialize_inputs_outputs(fname, inp, te_outputs)
            if precision in (torch.bfloat16,):
                return
            if fp8_recipe is None:
                validate_result(fname, inp, model, atol=1e-3, te_outputs=te_outputs)
            elif precision != torch.bfloat16:
                validate_result(
                    fname,
                    inp,
                    model,
                    atol=1e-3,
                    is_fp8=fp8_recipe is not None,
                    te_outputs=te_outputs,
                )


@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("precision", [torch.float32, torch.float16, torch.bfloat16])
def test_export_layernorm_linear_recipe(seed_default_rng, fp8_recipe, precision):
    _test_export_layernorm_linear(fp8_recipe=fp8_recipe, precision=precision)


def test_export_layernorm_linear_return_ln_out(seed_default_rng):
    _test_export_layernorm_linear(return_layernorm_output=True)


def test_export_layernorm_linear_zero_centered_gamma(seed_default_rng):
    _test_export_layernorm_linear(zero_centered_gamma=True)


@pytest.mark.parametrize("normalization", all_normalizations[1:])
def test_export_layernorm_linear_normalization(seed_default_rng, normalization):
    _test_export_layernorm_linear(normalization=normalization)


def test_export_layernorm_linear_no_bias(seed_default_rng):
    _test_export_layernorm_linear(use_bias=False)


def test_export_layernorm_linear_return_bias(seed_default_rng):
    _test_export_layernorm_linear(return_bias=True)


def _test_export_layernorm_mlp(
    scale_factor: float = 112,
    fp8_recipe: recipe.Recipe = fp8_recipes[0],
    use_bias: bool = True,
    return_bias: bool = False,
    return_layernorm_output: bool = False,
    precision: torch.dtype = torch.float32,
    zero_centered_gamma: bool = False,
    activation: str = supported_activations[0],
    normalization: str = all_normalizations[0],
):
    if return_bias and not use_bias:
        pytest.skip("Cannot return bias when bias is disabled")

    # Set dimensions (these are arbitrary).
    in_features = 64
    out_features = 256
    hidden_size = 256
    ffn_hidden_size = 256

    inp = torch.randn(in_features, out_features, device="cuda", dtype=precision)
    fp8_str = "_fp8" if fp8_recipe is not None else ""
    bias_str = "_bias" if use_bias else ""
    high_prec_str = dtype2str(precision)
    fname = f"te.layernorm_mlp{fp8_str}{bias_str}{high_prec_str}_{activation}.onnx"
    with te.fp8_autocast(enabled=fp8_recipe is not None, fp8_recipe=fp8_recipe):
        model = te.LayerNormMLP(
            hidden_size,
            ffn_hidden_size,
            bias=use_bias,
            return_bias=return_bias,
            return_layernorm_output=return_layernorm_output,
            params_dtype=precision,
            zero_centered_gamma=zero_centered_gamma,
            activation=activation,
            normalization=normalization,
        ).to(device="cuda")
        if fp8_recipe is not None:
            set_layer_scale(model, scale_factor, num_gemms=2)
        do_export(model, inp, fname, fp8_recipe)
        te_outputs = te_infer(model, inp, is_fp8=fp8_recipe is not None, fp8_recipe=fp8_recipe)
        serialize_inputs_outputs(fname, inp, te_outputs)
        if precision in (torch.bfloat16,):
            return
        atol = (
            2e-2 if fp8_recipe is not None else (5e-1 if activation == "swiglu" else 1e-3)
        )  # TODO(pgadzinski) - check 2e-2
        validate_result(
            fname, inp, model, atol=atol, is_fp8=fp8_recipe is not None, te_outputs=te_outputs
        )


@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("precision", [torch.float32, torch.float16, torch.bfloat16])
def test_export_layernorm_mlp(seed_default_rng, fp8_recipe, precision):
    _test_export_layernorm_mlp(fp8_recipe=fp8_recipe, precision=precision)


def test_export_layernorm_mlp_return_layernorm_output(seed_default_rng):
    _test_export_layernorm_mlp(return_layernorm_output=True)


def test_export_layernorm_mlp_return_bias(seed_default_rng):
    _test_export_layernorm_mlp(return_bias=True)


def test_export_layernorm_mlp_no_bias(seed_default_rng):
    _test_export_layernorm_mlp(use_bias=False)


def test_export_layernorm_mlp_zero_centered_gamma(seed_default_rng):
    _test_export_layernorm_mlp(zero_centered_gamma=True)


@pytest.mark.parametrize("normalization", all_normalizations[1:])
def test_export_layernorm_mlp_normalization(seed_default_rng, normalization):
    _test_export_layernorm_mlp(normalization=normalization)


@pytest.mark.parametrize("activation", supported_activations[1:])
def test_export_layernorm_mlp_activation(seed_default_rng, activation):
    _test_export_layernorm_mlp(activation=activation)


@pytest.mark.parametrize(
    "precision,      use_mask, attn_mask_type",
    [
        (torch.float32, True, "arbitrary"),  # calls forward_torch_softmax (apply user mask)
        (torch.float32, False, "no_mask"),  # calls forward_torch_softmax (apply no mask)
        (torch.float16, False, "causal"),  # calls forward_torch_softmax (apply dynamic onnx mask)
        (torch.float16, True, "arbitrary"),  # calls forward_torch_softmax (apply user mask)
        (torch.float16, False, "no_mask"),  # calls forward_torch_softmax (apply no mask)
        (torch.bfloat16, False, "causal"),  # calls forward_torch_softmax (apply dynamic onnx mask)
        (torch.bfloat16, True, "arbitrary"),  # calls forward_torch_softmax (apply user mask)
        (torch.bfloat16, False, "no_mask"),  # calls forward_torch_softmax (apply no mask)
    ],
)
def test_export_core_attention(
    precision: torch.dtype,
    use_mask: bool,
    attn_mask_type: str,
):
    # Set dimensions (these are arbitrary).
    seq_len, batch_size, num_attention_heads, kv_channels = (64, 4, 1, 64)
    qkv_size = (seq_len, batch_size, num_attention_heads, kv_channels)
    qkv_format = "sbhd"

    query_layer = torch.randn(qkv_size, dtype=precision, device="cuda")
    key_layer = torch.randn(qkv_size, dtype=precision, device="cuda")
    value_layer = torch.randn(qkv_size, dtype=precision, device="cuda")
    input_names = ["query", "key", "value", "attention_mask"]
    attention_mask = None
    if use_mask:
        # Generate a random mask with 50% probability for 0 or 1.
        probs = 0.5 * torch.ones(batch_size, 1, 1, seq_len, device="cuda", dtype=precision)
        attention_mask = torch.bernoulli(probs).to("cuda", dtype=torch.bool)
    inp = (query_layer, key_layer, value_layer, attention_mask)

    mask_str = get_attn_mask_str(use_mask, attn_mask_type)
    high_prec_str = dtype2str(precision)
    fname = f"te.core_attention{mask_str}{high_prec_str}.onnx"

    model = te.attention.DotProductAttention(
        num_attention_heads=num_attention_heads,
        kv_channels=kv_channels,
        attention_dropout=0.5,
        qkv_format=qkv_format,
        attn_mask_type=attn_mask_type,
    ).to(device="cuda")
    do_export(model, inp, fname, input_names=input_names, fp8_recipe=None)
    te_outputs = te_infer(model, inp, is_fp8=False, fp8_recipe=None)
    serialize_inputs_outputs(fname, inp, te_outputs, input_names=input_names)
    if precision in (torch.bfloat16,):
        return
    validate_result(
        fname, inp, model, is_fp8=True, atol=1e-2, input_names=input_names, te_outputs=te_outputs
    )


test_configs_attention_type = [
    # "input_layernorm, attention_type, fuse_qkv_params"
    (True, "self", True),
    (False, "self", True),
    (True, "self", False),
    (False, "self", False),
    (True, "cross", True),
    (False, "cross", True),
    (True, "cross", False),
    (False, "cross", False),
]


def _test_export_multihead_attention(
    fp8_recipe: recipe.Recipe = fp8_recipes[0],
    use_mask: bool = True,
    precision: torch.dtype = torch.float32,
    input_layernorm: bool = True,
    attention_type: str = "self",
    fuse_qkv_params: bool = True,
):
    hidden_size = 256
    sequence_length = 128
    batch_size = 4
    num_attention_heads = 32
    kv_channels = 8
    attention_dropout = 0.1
    layernorm_epsilon = 1e-5
    init_method = output_layer_init_method = get_default_init_method()
    attention_args = (
        hidden_size,
        num_attention_heads,
        kv_channels,
        attention_dropout,
        layernorm_epsilon,
        init_method,
        output_layer_init_method,
    )
    attn_mask_type = "arbitrary" if use_mask else "no_mask"

    hidden_states_context = torch.randn(
        sequence_length, batch_size, hidden_size, dtype=precision, device="cuda"
    )
    attention_mask = None
    if use_mask and attn_mask_type != "causal":
        # Generate a random mask with 50% probability for 0 or 1.
        probs = 0.5 * torch.ones(
            batch_size, 1, sequence_length, sequence_length, device="cuda", dtype=precision
        )
        attention_mask = torch.bernoulli(probs).to("cuda", dtype=torch.bool)

    encoder_output = None

    if attention_type == "cross":
        encoder_output = torch.randn(
            sequence_length, batch_size, hidden_size, dtype=precision, device="cuda"
        )

    fp8_str = "_fp8" if fp8_recipe is not None else ""
    dtype_str = dtype2str(precision)
    attn_type_str = "_self-attention" if attention_type == "self" else "_cross-attention"
    fuse_qkv_str = "_fused-qkv" if fuse_qkv_params else ""
    attn_mask_str = get_attn_mask_str(use_mask, attn_mask_type)
    input_ln_str = "_input-ln" if input_layernorm else ""
    fname = f"te.multihead_attention{fp8_str}{attn_mask_str}{attn_type_str}{input_ln_str}{fuse_qkv_str}{dtype_str}.onnx"

    model = te.MultiheadAttention(
        *attention_args,
        attn_mask_type=attn_mask_type,
        params_dtype=precision,
        return_layernorm_output=False,
        input_layernorm=input_layernorm,
        attention_type=attention_type,
        fuse_qkv_params=fuse_qkv_params,
        return_bias=True,
    ).to(device="cuda")

    inp_context = (hidden_states_context, attention_mask, encoder_output)
    input_names = ["hidden_states", "attention_mask", "encoder_output"]
    output_names = ["attention_output", "attention_bias"]
    seq = torch.export.Dim("seq", min=2, max=1256)
    bs = torch.export.Dim("bs", min=2, max=1256)
    do_export(
        model,
        inp_context,
        fname,
        fp8_recipe,
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes={
            "hidden_states": {0: seq, 1: bs},
            "attention_mask": {2: seq, 0: bs} if use_mask else None,
            "encoder_output": {0: seq, 1: bs} if attention_type == "cross" else None,
        },
    )
    te_outputs = te_infer(model, inp_context, is_fp8=fp8_recipe is not None, fp8_recipe=fp8_recipe)
    serialize_inputs_outputs(
        fname, inp_context, te_outputs, input_names=input_names, output_names=output_names
    )
    if precision in (torch.bfloat16,):
        return

    if fp8_recipe is None:
        validate_result(
            fname,
            inp_context,
            model,
            atol=1e-3,
            input_names=input_names,
            output_names=output_names,
            te_outputs=te_outputs,
        )
    else:
        validate_result(
            fname,
            inp_context,
            model,
            atol=1e-2,
            is_fp8=fp8_recipe is not None,
            input_names=input_names,
            output_names=output_names,
            allow_cnt_errors=3,
            te_outputs=te_outputs,
        )

    # In GPT generative phase (inference) the input sequence is smaller than the maximum
    # allowed sequence length and we want to test this condition.
    # Pretend that we're in generative phase when it makes sense (causal mask and self-attention).
    is_generative_phase = attn_mask_type == "causal" and attention_type == "self"
    if is_generative_phase:
        seq_len_offset = 8
        hidden_states_generative = torch.randn(
            sequence_length - seq_len_offset,
            batch_size,
            hidden_size,
            dtype=precision,
            device="cuda",
        )
        inp_generative = (hidden_states_generative, attention_mask, encoder_output)
        if fp8_recipe is None:
            validate_result(
                fname,
                inp_generative,
                model,
                atol=1e-3,
                input_names=input_names,
                output_names=output_names,
            )
        else:
            validate_result(
                fname,
                inp_generative,
                model,
                atol=1e-2,
                is_fp8=fp8_recipe is not None,
                input_names=input_names,
                output_names=output_names,
                allow_cnt_errors=3,
            )


@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("precision", [torch.float32, torch.float16, torch.bfloat16])
def test_export_multihead_attention_recipe(fp8_recipe, precision):
    _test_export_multihead_attention(fp8_recipe=fp8_recipe, precision=precision)


def test_export_multihead_attention_no_mask():
    _test_export_multihead_attention(use_mask=False)


def test_export_multihead_attention_no_input_layernorm():
    _test_export_multihead_attention(input_layernorm=False)


def test_export_multihead_attention_cross_attn():
    _test_export_multihead_attention(attention_type="cross")


def test_export_multihead_attention_unfused_qkv_params():
    _test_export_multihead_attention(fuse_qkv_params=False)


def _test_export_transformer_layer(
    fp8_recipe: recipe.Recipe = fp8_recipes[0],
    use_mask: bool = True,
    attn_mask_type: str = "arbitrary",
    output_layernorm: bool = False,
    precision: torch.dtype = torch.float32,
    fuse_qkv_params: bool = True,
    zero_centered_gamma: bool = False,
    activation: str = supported_activations[0],
):
    # Layer configuration
    hidden_size = 64
    sequence_length = 128
    batch_size = 1
    ffn_hidden_size = 256
    num_attention_heads = 4

    input_tensor = torch.rand(
        sequence_length, batch_size, hidden_size, dtype=precision, device="cuda"
    )
    input_names = ["input", "attention_mask"]
    attention_mask = None
    if use_mask and attn_mask_type != "causal":
        # Generate a random mask with 50% probability for 0 or 1.
        probs = 0.5 * torch.ones(
            batch_size, 1, sequence_length, sequence_length, device="cuda", dtype=precision
        )
        attention_mask = torch.bernoulli(probs).to("cuda", dtype=torch.bool)
    inp = (input_tensor, attention_mask)

    fp8_str = "_fp8" if fp8_recipe is not None else ""
    fuse_qkv_params_str = "_fused-qkv" if fuse_qkv_params else ""
    high_prec_str = dtype2str(precision)
    attn_mask_str = get_attn_mask_str(use_mask, attn_mask_type)
    fname = f"te.transformer_layer{fp8_str}{attn_mask_str}{fuse_qkv_params_str}{high_prec_str}_{activation}.onnx"

    model = te.TransformerLayer(
        hidden_size,
        ffn_hidden_size,
        num_attention_heads,
        self_attn_mask_type=attn_mask_type,
        output_layernorm=output_layernorm,
        params_dtype=precision,
        fuse_qkv_params=fuse_qkv_params,
        zero_centered_gamma=zero_centered_gamma,
        activation=activation,
    ).to(device="cuda")
    do_export(model, inp, fname, fp8_recipe, input_names=input_names)
    te_outputs = te_infer(model, inp, is_fp8=fp8_recipe is not None, fp8_recipe=fp8_recipe)
    serialize_inputs_outputs(
        fname,
        inp,
        te_outputs,
        input_names=input_names,
    )
    if precision in (torch.bfloat16,):
        return
    atol = 5e-1 if fp8_recipe is not None else (5e-1 if activation == "swiglu" else 5e-3)
    validate_result(
        fname,
        inp,
        model,
        atol=atol,
        is_fp8=fp8_recipe is not None,
        input_names=input_names,
        te_outputs=te_outputs,
    )


@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("precision", [torch.float32, torch.float16, torch.bfloat16])
def test_export_transformer_layer_recipe(fp8_recipe, precision):
    _test_export_transformer_layer(fp8_recipe=fp8_recipe, precision=precision)


def test_export_transformer_layer_no_mask():
    _test_export_transformer_layer(use_mask=False)


def test_export_transformer_layer_output_layernorm():
    _test_export_transformer_layer(output_layernorm=True)


def test_export_transformer_layer_unfused_qkv_params():
    _test_export_transformer_layer(fuse_qkv_params=False)


def test_export_transformer_layer_zero_centered_gamma():
    _test_export_transformer_layer(zero_centered_gamma=True)


@pytest.mark.parametrize("activation", supported_activations[1:])
def test_export_transformer_layer_activation(activation):
    _test_export_transformer_layer(activation=activation)


@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@pytest.mark.parametrize("precision", [torch.float16, torch.bfloat16])
def test_export_gpt_generation(
    fp8_recipe: recipe.Recipe,
    precision: torch.dtype,
):
    """Test that the ONNX model can correctly handle inputs with different shapes and that
    the attention mask is adjusted on-the-fly to different sequence lengths.
    """

    # Layer configuration
    hidden_size = 64
    sequence_length = 128
    batch_size = 4
    ffn_hidden_size = 256
    num_attention_heads = 4
    attention_mask = None
    use_mask = True
    attn_mask_type = "causal"
    fuse_qkv_params = True
    output_layernorm = False

    fp8_str = "_fp8" if fp8_recipe is not None else ""
    fuse_qkv_params_str = "_fused-qkv" if fuse_qkv_params else ""
    high_prec_str = dtype2str(precision)
    attn_mask_str = get_attn_mask_str(use_mask, attn_mask_type)
    fname = f"te.transformer_layer_generative{fp8_str}{attn_mask_str}{fuse_qkv_params_str}{high_prec_str}.onnx"

    model = te.TransformerLayer(
        hidden_size,
        ffn_hidden_size,
        num_attention_heads,
        self_attn_mask_type=attn_mask_type,
        output_layernorm=output_layernorm,
        params_dtype=precision,
        fuse_qkv_params=fuse_qkv_params,
    ).to(device="cuda")

    # "Context phase": use full input sequence length
    input_names = ["input"]
    output_names = ["output"]
    input_tensor = torch.rand(
        sequence_length, batch_size, hidden_size, dtype=precision, device="cuda"
    )
    inp = (input_tensor,)
    # dynamic shape
    seq = torch.export.Dim("seq", min=2, max=1256)
    bs = torch.export.Dim("bs", min=2, max=1256)
    do_export(
        model,
        inp,
        fname,
        fp8_recipe,
        dynamic_shapes={"hidden_states": {0: seq, 1: bs}},
    )
    te_outputs = te_infer(model, inp, is_fp8=fp8_recipe is not None, fp8_recipe=fp8_recipe)
    serialize_inputs_outputs(
        fname, inp, te_outputs, input_names=input_names, output_names=output_names
    )
    if precision not in (torch.bfloat16,):
        validate_result(
            fname,
            inp,
            model,
            atol=1e-2,
            is_fp8=fp8_recipe is not None,
            input_names=input_names,
            te_outputs=te_outputs,
        )

    # "Generative phase": use a single input (sequence len=1). For FP8 we need to pad the sequence to mult of 8 and for MXFP8 we need to pad to mult of 32.
    sequence_length = 1 if fp8_recipe is None else 32
    input_tensor = torch.rand(
        sequence_length, batch_size, hidden_size, dtype=precision, device="cuda"
    )
    inp = (input_tensor, attention_mask)
    te_outputs = te_infer(model, inp, is_fp8=fp8_recipe is not None, fp8_recipe=fp8_recipe)
    serialize_inputs_outputs(fname, inp, te_outputs, input_names=input_names)
    if precision not in (torch.bfloat16,):
        validate_result(
            fname,
            inp,
            model,
            atol=1e-2,
            is_fp8=fp8_recipe is not None,
            input_names=input_names,
            te_outputs=te_outputs,
        )


@pytest.mark.parametrize("enabled", [True, False])
def test_export_ctx_manager(enabled):
    assert is_in_onnx_export_mode() == False
    with te.onnx_export(enabled):
        assert is_in_onnx_export_mode() == enabled
    assert is_in_onnx_export_mode() == False
