# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""End-to-end numerical tests for contiguous GDN context parallelism."""

import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile

import pytest
import torch

import transformer_engine
from test_gdn_attention import _GDN_AVAILABLE


_SEED = 2026


def _generate_random_lengths(count, lower_bound, upper_bound, *, total=None):
    """Sample reproducible lengths in [lower_bound, upper_bound], optionally fixing the sum."""
    generator = random.Random(_SEED)
    if total is None:
        return [generator.randint(lower_bound, upper_bound) for _ in range(count)]
    assert count * lower_bound <= total <= count * upper_bound
    lengths = []
    for remaining in range(count - 1, -1, -1):
        # Reserve enough tokens to keep the remaining documents within bounds.
        lower = max(lower_bound, total - remaining * upper_bound)
        upper = min(upper_bound, total - remaining * lower_bound)
        length = generator.randint(lower, upper)
        lengths.append(length)
        total -= length
    generator.shuffle(lengths)
    return lengths


_GEOMETRY_LENGTHS = _generate_random_lengths(3, 65, 192)

GDN_CP_CASES = {
    # One long shared document exercises many internal chunks on every rank.
    "cp_0": dict(lengths=_generate_random_lengths(1, 7168, 8192), norm=True),
    # Dense FP16 training uses a multiple of 8 for equal CP=2/4/8 slices, with checkpointing.
    "cp_1": dict(
        # Pick a deterministic multiple of 8 in [768, 1024].
        lengths=[8 * random.Random(_SEED).randint(96, 128)] * 4,
        layout="bshd",
        dtype="fp16",
        heads=4,
        value_dim=128,
        initial=False,
        checkpoint=True,
    ),
    # An odd dense length needs explicit global offsets and uneven rank-local slices.
    "cp_2": dict(
        # Pick a deterministic odd number in [769, 1025].
        lengths=[2 * random.Random(_SEED).randint(384, 512) + 1] * 3,
        layout="sbhd",
        heads=4,
        key_dim=128,
        norm=True,
        dense_global=True,
        final=False,
    ),
    # Keep 4096 tokens so the 2049-token document spans three CP=4 ranks; vary the short tail.
    "cp_3": dict(
        lengths=[2049, 1023] + _generate_random_lengths(14, 32, 128, total=1024),
        dtype="fp16",
        key_dim=128,
        value_dim=128,
        norm=True,
        fallback=True,
    ),
    # Whole documents align with ranks for CP=2/4/8 and need no summary exchange.
    "cp_4": dict(lengths=[64] * 32, heads=8, initial=False, final=False),
    # Empty documents, partial features and a late-token loss also exercise CPU metadata.
    "cp_5": dict(
        lengths=[0, _GEOMETRY_LENGTHS[0], 0, *_GEOMETRY_LENGTHS[1:]],
        key_dim=32,
        value_dim=96,
        norm=True,
        checkpoint=True,
        last_loss=True,
        cpu_metadata=True,
    ),
    # A single token leaves empty ranks; only final-state losses drive backward.
    "cp_6": dict(lengths=[0, 1, 0], state_only=True),
    # Keep 1024 tokens and a shared first document; chain states after inference-cache warmup.
    "cp_7": dict(
        kind="state_chaining",
        lengths=[513] + _generate_random_lengths(3, 64, 256, total=511),
        heads=2,
        key_dim=128,
        value_dim=64,
        norm=True,
        checkpoint=True,
    ),
}


@pytest.mark.parametrize("name", GDN_CP_CASES)
def test_gdn_with_cp(name):
    """Compare one distributed scenario's outputs and gradients to native FP64."""
    available = torch.cuda.device_count()
    # Test only the largest supported CP size available to keep the CI matrix focused.
    size = next((size for size in (8, 4, 2) if available >= size), None)
    if size is None:
        pytest.skip(f"GDN CP test requires at least 2 GPUs, but found {available}")
    if not _GDN_AVAILABLE:
        if os.getenv("NVTE_GDN_CP_TEST_REQUIRED", "0") == "1":
            pytest.fail("Required GDN CP test needs the cuDNN frontend GDN runtime")
        pytest.skip("GDN CP test requires the cuDNN frontend GDN runtime")
    case = GDN_CP_CASES[name]
    # TODO(KshitijLakhani): Create a common worker pool for all attention variants.
    # Locate the distributed runner beside this test file.
    runner = Path(__file__).with_name("run_gdn_with_cp.py")
    env = os.environ.copy()
    # Preserve the TE package imported by pytest when workers start from the tests directory.
    package_root = str(Path(transformer_engine.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join((package_root, env.get("PYTHONPATH", "")))
    # Capture all workers' output and errors in a per-case log for pytest.
    with tempfile.TemporaryDirectory(prefix=f"te-gdn-cp-{name}-") as log_dir:
        log_path = Path(log_dir) / "worker.log"
        with log_path.open("w", encoding="utf-8") as log:
            # Launch one worker per GPU with pytest's Python and this case's configuration.
            with subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc_per_node={size}",
                    str(runner),
                    "--config",
                    json.dumps(case),
                ],
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                # Isolate the launcher and workers so timeout cleanup can stop them all.
                start_new_session=True,
                env=env,
            ) as process:
                try:
                    # Wait for workers, allowing time for FP64 reference and cuDNN compilation.
                    returncode = process.wait(timeout=300)
                except subprocess.TimeoutExpired:
                    # Stop the launcher and all GPU workers if a case hangs.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                    pytest.fail(f"{name} timed out after 300 seconds\n{log_path.read_text()}")
        output = log_path.read_text()
        # Report a failed worker run with its captured output.
        assert returncode == 0, output
