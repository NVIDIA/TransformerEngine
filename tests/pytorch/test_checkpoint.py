# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

import argparse
import functools
import pathlib
import sys

import pytest
import torch

import transformer_engine.pytorch as te

# Import utility functions
_current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(_current_dir))
from utils import make_recipe

# Check supported quantization schemes
fp8_available, reason_for_no_fp8 = te.fp8.FP8GlobalStateManager.is_fp8_available()
mxfp8_available, reason_for_no_mxfp8 = te.fp8.FP8GlobalStateManager.is_mxfp8_available()


# Test cases for loading checkpoint files
_TestLoadCheckpoint_name_list: tuple[str, ...] = (
    "linear",
    "layernorm_linear",
    "layernorm_mlp",
    "layernorm",
    "rmsnorm",
    "transformer_layer",
    "ops_linear",
    "linear.fp8",
    "ops_linear.fp8",
    "linear.mxfp8",
    "ops_linear.mxfp8",
)


class TestLoadCheckpoint:
    """Tests for loading checkpoint files

    Tests assume that checkpoint files have already been created. In
    order to regenerate checkpoint files, e.g. after a breaking change
    in the checkpoint format, run this file directly as a Python
    script: `python3 test_checkpoint.py --save-checkpoint all`.

    """

    @staticmethod
    def _make_module(name: str) -> torch.nn.Module:
        """ "Construct a module"""
        if name == "linear":
            return te.Linear(1, 1)
        if name == "layernorm_linear":
            return te.LayerNormLinear(1, 1)
        if name == "layernorm_mlp":
            return te.LayerNormMLP(1, 1)
        if name == "layernorm":
            return te.LayerNorm(1)
        if name == "rmsnorm":
            return te.RMSNorm(1)
        if name == "transformer_layer":
            return te.TransformerLayer(1, 1, 1)
        if name == "ops_linear":
            return te.ops.Linear(1, 1)
        if name == "linear.fp8":
            with te.fp8_model_init(recipe=make_recipe("fp8")):
                return te.Linear(16, 16)
        if name == "ops_linear.fp8":
            with te.fp8_model_init(recipe=make_recipe("fp8")):
                return te.ops.Linear(16, 16)
        if name == "linear.mxfp8":
            with te.fp8_model_init(recipe=make_recipe("mxfp8")):
                return te.Linear(32, 32)
        if name == "ops_linear.mxfp8":
            with te.fp8_model_init(recipe=make_recipe("mxfp8")):
                return te.ops.Linear(32, 32)
        raise ValueError(f"Unrecognized module name ({name})")

    @staticmethod
    def _checkpoint_file(name: str) -> pathlib.Path:
        """Path to a module's checkpoint file"""
        return _current_dir / "checkpoints" / f"{name}.pt"

    @staticmethod
    def _save_checkpoint(name: str) -> None:
        """Save a module's checkpoint file"""
        module = TestLoadCheckpoint._make_module(name)
        checkpoint_file = TestLoadCheckpoint._checkpoint_file(name)
        torch.save(module.state_dict(), checkpoint_file)
        print(f"Saved checkpoint for {name} at {checkpoint_file}")

    @pytest.mark.parametrize("name", _TestLoadCheckpoint_name_list)
    def test_module(self, name: str) -> None:
        """Test for loading a module's checkpoint file"""

        # Skip if quantization is not supported
        quantization = None
        if "." in name:
            quantization = name.split(".")[1]
        if quantization == "fp8" and not fp8_available:
            pytest.skip(reason_for_no_fp8)
        if quantization == "mxfp8" and not mxfp8_available:
            pytest.skip(reason_for_no_mxfp8)

        # Construct module
        module = self._make_module(name)

        # Load checkpoint from file
        checkpoint_file = self._checkpoint_file(name)
        if not checkpoint_file.is_file():
            raise FileNotFoundError(f"Could not find checkpoint file at {checkpoint_file}")
        state_dict = torch.load(checkpoint_file, weights_only=False)

        # Update module from checkpoint
        module.load_state_dict(state_dict, strict=True)


def main() -> None:
    """Main function

    Typically used to generate checkpoint files.

    """

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-checkpoint",
        type=str,
        default=None,
        help="Save a checkpoint file for a module",
    )
    args = parser.parse_args()

    # Save checkpoint files if needed
    if args.save_checkpoint is not None:
        if args.save_checkpoint == "all":
            for name in _TestLoadCheckpoint_name_list:
                TestLoadCheckpoint._save_checkpoint(name)
        else:
            TestLoadCheckpoint._save_checkpoint(args.save_checkpoint)


if __name__ == "__main__":
    main()
