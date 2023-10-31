# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
This file contains tests for saving and loading TransformerEngine torch checkpoints.

The purpose of this test is to validate the TransformerEngine hooks for saving FP8 metadata
in torch checkpoints, which are called as part of torch.save() and torch.load().
The test verifies the values of FP8 metadata object after saving and loading a checkpoint
are identical to the original values.
"""

import tempfile
import pytest
import torch
import transformer_engine.pytorch as te
import transformer_engine_extensions as tex
from transformer_engine.pytorch.cpp_extensions import fp8_gemm, cast_to_fp8
from transformer_engine.pytorch.module.base import get_workspace
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule


def init_meta(size: int=1):
    meta = tex.FP8TensorMeta()
    meta.scale = torch.ones(size, dtype=torch.float32, device="cuda")
    meta.scale_inv = torch.ones(size, dtype=torch.float32, device="cuda")
    meta.amax_history = torch.zeros(1, size, dtype=torch.float32, device="cuda")
    return meta


@pytest.mark.parametrize("scale_fwd", [224, 112, 66])
@pytest.mark.parametrize("scale_bwd", [448, 33])
@pytest.mark.parametrize("history_fwd", [1.23, 4.56])
@pytest.mark.parametrize("history_bwd", [2.34, 5.67])
def test_export_loaded_checkpoint(scale_fwd, scale_bwd, history_fwd, history_bwd):

    # Skip FP8 tests on non-hopper devices
    if torch.cuda.get_device_properties(torch.cuda.current_device()).major < 9:
        pytest.skip("Device compute capability 9.x required for FP8 execution.")

    tmp_filename = tempfile.NamedTemporaryFile().name

    precision = torch.float32

    class Test_TE_Export(TransformerEngineBaseModule):
        def __init__(self, precision, use_bias):
            super().__init__()
            self.use_bias = use_bias
            self.precision = precision

            self.fp8_tensor_inp = tex.FP8FwdTensors.GEMM1_INPUT
            self.fp8_tensor_weight = tex.FP8FwdTensors.GEMM1_WEIGHT
            nb_inp_scales = nb_weight_scales = 1
            self.meta_inp = init_meta(nb_inp_scales)
            self.meta_weight = init_meta(nb_weight_scales)

            bias_size = nb_weight_scales
            self.bias = torch.randn(bias_size, dtype=precision, device="cuda")

            self.inp_type = tex.DType.kFloat8E4M3
            self.weights_type = tex.DType.kFloat8E4M3
            self.outp_type = precision

        def forward(self, inp, weight):
            inp_fp8 = cast_to_fp8(
                inp,
                self.meta_inp,
                self.fp8_tensor_inp,
                self.inp_type)

            weight_fp8 = cast_to_fp8(
                weight,
                self.meta_weight,
                self.fp8_tensor_weight,
                self.weights_type)

            ret = fp8_gemm(
                weight_fp8,
                self.meta_weight.scale_inv,
                self.fp8_tensor_weight,
                self.inp_type,
                inp_fp8,
                self.meta_inp.scale_inv,
                self.fp8_tensor_inp,
                self.weights_type,
                self.outp_type,
                get_workspace(),
                bias=self.bias,
                use_bias=self.use_bias,
                use_split_accumulator=False)
            return ret

    model_in = Test_TE_Export(precision, True)
    with te.fp8_autocast(enabled=True):
        model_in.init_fp8_metadata()
        # scaling fwd
        model_in.fp8_meta["scaling_fwd"].scale = torch.ones(3, dtype=torch.float32, device="cuda") * scale_fwd
        model_in.fp8_meta["scaling_fwd"].scale_inv = torch.ones(3, dtype=torch.float32, device="cuda") / scale_fwd
        model_in.fp8_meta["scaling_fwd"].amax_history = torch.ones(3, dtype=torch.float32, device="cuda") * history_fwd
        # scaling bwd
        model_in.fp8_meta["scaling_bwd"].scale = torch.ones(2, dtype=torch.float32, device="cuda") * scale_bwd
        model_in.fp8_meta["scaling_bwd"].scale_inv = torch.ones(2, dtype=torch.float32, device="cuda") / scale_bwd
        model_in.fp8_meta["scaling_bwd"].amax_history = torch.ones(2, dtype=torch.float32, device="cuda") * history_bwd

    torch.save(model_in.state_dict(), tmp_filename)

    model_out = Test_TE_Export(precision, True)
    model_out.load_state_dict(torch.load(tmp_filename))
    model_out.eval()

    # scaling fwd
    assert torch.allclose(model_in.fp8_meta["scaling_fwd"].scale, model_out.fp8_meta["scaling_fwd"].scale)
    assert torch.allclose(model_in.fp8_meta["scaling_fwd"].scale_inv, model_out.fp8_meta["scaling_fwd"].scale_inv)
    assert torch.allclose(model_in.fp8_meta["scaling_fwd"].amax_history, model_out.fp8_meta["scaling_fwd"].amax_history)
    # scaling bwd
    assert torch.allclose(model_in.fp8_meta["scaling_bwd"].scale, model_out.fp8_meta["scaling_bwd"].scale)
    assert torch.allclose(model_in.fp8_meta["scaling_bwd"].scale_inv, model_out.fp8_meta["scaling_bwd"].scale_inv)
    assert torch.allclose(model_in.fp8_meta["scaling_bwd"].amax_history, model_out.fp8_meta["scaling_bwd"].amax_history)

