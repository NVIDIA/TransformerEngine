# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch
import torch.nn.functional as F

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.common import recipe
from transformer_engine.pytorch.constants import FP8FwdTensorIdx
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
from transformer_engine.pytorch.tensor.float8_tensor import Float8CurrentScalingQuantizer


class ToyLinear(TransformerEngineBaseModule):
    """Minimal TE module: full FP8 quantizer setup, plain F.linear compute."""

    def __init__(self, in_features, out_features, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype, device=device)
        )
        torch.nn.init.normal_(self.weight)

    def _get_weight_tensors(self):
        return [self.weight]

    def _get_weight_quantizers(self):
        if not self.fp8 and not self.fp8_calibration:
            return [None]
        wq = self.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM1_WEIGHT]
        wq.internal = True
        return [wq]

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        inp = self.prepare_forward(inp, num_gemms=1)
        try:
            (w,) = self._get_weight_tensors()
            (_wq,) = self._get_weight_quantizers()
            return F.linear(inp, w)
        finally:
            self.end_forward()


_fp8_available = te.is_fp8_available()
_mxfp8_available = te.is_mxfp8_available()
_fp8_block_scaling_available = te.is_fp8_block_scaling_available()
_nvfp4_available = te.is_nvfp4_available()

_recipes = [
    pytest.param(recipe.DelayedScaling(), False, id="no_fp8"),
    pytest.param(
        recipe.Float8CurrentScaling(),
        True,
        id="float8_current_scaling",
        marks=pytest.mark.skipif(not _fp8_available, reason="FP8 not supported"),
    ),
    pytest.param(
        recipe.MXFP8BlockScaling(),
        True,
        id="mxfp8_block_scaling",
        marks=pytest.mark.skipif(not _mxfp8_available, reason="MXFP8 not supported"),
    ),
    pytest.param(
        recipe.Float8BlockScaling(),
        True,
        id="float8_block_scaling",
        marks=pytest.mark.skipif(
            not _fp8_block_scaling_available, reason="FP8 block scaling not supported"
        ),
    ),
    pytest.param(
        recipe.NVFP4BlockScaling(),
        True,
        id="nvfp4_block_scaling",
        marks=pytest.mark.skipif(not _nvfp4_available, reason="NVFP4 not supported"),
    ),
]


@pytest.mark.parametrize("fp8_recipe,fp8_enabled", _recipes)
def test_autocast_sanity(fp8_recipe, fp8_enabled):
    """ToyLinear inside te.autocast compiles with fullgraph=True."""
    torch._dynamo.reset()

    model = ToyLinear(32, 64, device="cuda", dtype=torch.bfloat16)
    inp = torch.randn(8, 32, dtype=torch.bfloat16, device="cuda", requires_grad=True)

    def fn(inp):
        with te.autocast(recipe=fp8_recipe, enabled=fp8_enabled):
            return model(inp)

    compiled = torch.compile(fn, fullgraph=True)
    out = compiled(inp)
    out.sum().backward()


def _make_tagged_qfactory(tag: str):
    def qfactory(role: str):
        q = Float8CurrentScalingQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            device=torch.device("cuda"),
        )
        q._tag = f"{tag}:{role}"
        return q
    return qfactory


@pytest.mark.skipif(not _fp8_available, reason="FP8 not supported")
def test_autocast_nested_sanity():
    """Nested/sequential te.autocast with different CustomRecipes compiles with fullgraph=True."""
    torch._dynamo.reset()

    recipe0 = recipe.CustomRecipe(qfactory=_make_tagged_qfactory("R0"))
    recipe1 = recipe.CustomRecipe(qfactory=_make_tagged_qfactory("R1"))
    recipe2 = recipe.CustomRecipe(qfactory=_make_tagged_qfactory("R2"))

    model = ToyLinear(32, 32, device="cuda", dtype=torch.bfloat16)
    inp = torch.randn(8, 32, dtype=torch.bfloat16, device="cuda", requires_grad=True)

    def fn(inp):
        with te.autocast(recipe=recipe0):
            out = model(inp)
            with te.autocast(recipe=recipe1):
                out = model(out)
        with te.autocast(recipe=recipe2):
            out = model(out)
        return out

    compiled = torch.compile(fn, fullgraph=True)
    out = compiled(inp)
    out.sum().backward()
