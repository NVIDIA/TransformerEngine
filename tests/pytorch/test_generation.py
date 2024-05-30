# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

import transformer_engine.pytorch as te

# First - tests on InferenceParams methods

# Second - tests illustrating how generation works - both thd and bshd
# how to do this - compare with case without cache

class TestInferenceParams:
    def test

class TestGeneration:
    def test_bshd_generation(self):
        # normal generation with torch Transformer Layer without cache.

        # generation using TE - copy weight, setup inference params, run 3 iterations

        # compare outputs

    def test_thd_generation(self):
        # similarly -