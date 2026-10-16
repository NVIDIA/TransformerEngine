# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Regression for the opt-in compact D256 backward initialization path.

Each arm uses a fresh process because the C++ experiment flags are cached.
This checks local attention correctness and the allocation path, not CP speed.
"""

import os
from pathlib import Path
import subprocess
import sys

import pytest


GEOMETRIES = {"causal": (256, 256, True), "upper": (256, 128, False), "lower": (128, 256, False)}


def _worker(geometry, mode, output):
    import torch
    import transformer_engine
    from transformer_engine.pytorch.cpp_extensions.fused_attn import (
        FusedAttnBackend,
        fused_attn_bwd,
    )

    torch.cuda.set_device(0)
    assert torch.cuda.get_device_capability() == (10, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    batch, heads, kv_heads, dim = 2, 8, 2, 256
    sq, sk, causal = GEOMETRIES[geometry]
    generator = torch.Generator(device="cpu").manual_seed(20261015)

    def random_tensor(shape):
        # CPU generation keeps the test operands identical across processes/SM counts.
        return torch.randn(shape, generator=generator).to(device="cuda", dtype=torch.bfloat16)

    q = random_tensor((batch, sq, heads, dim))
    k = random_tensor((batch, sk, kv_heads, dim))
    v = random_tensor(k.shape)
    do = random_tensor(q.shape)
    qr = q.transpose(1, 2).float()
    kr = k.transpose(1, 2).float().repeat_interleave(heads // kv_heads, 1)
    vr = v.transpose(1, 2).float().repeat_interleave(heads // kv_heads, 1)
    dr = do.transpose(1, 2).float()
    scores = qr @ kr.transpose(-1, -2) * 0.0625
    if causal:
        scores.masked_fill_(
            torch.arange(sk, device="cuda")[None, :] > torch.arange(sq, device="cuda")[:, None],
            -float("inf"),
        )
    probability = scores.softmax(-1)
    reference_o = probability @ vr
    o = reference_o.transpose(1, 2).contiguous().to(torch.bfloat16)
    stats = scores.logsumexp(-1).transpose(1, 2).contiguous().view(batch * sq, heads, 1)
    ds = probability * (dr @ vr.transpose(-1, -2) - (dr * reference_o).sum(-1, keepdim=True))
    references = {
        "dq": (ds @ kr * 0.0625).transpose(1, 2),
        "dk": (ds.transpose(-1, -2) @ qr * 0.0625)
        .view(batch, kv_heads, heads // kv_heads, sk, dim)
        .sum(2)
        .transpose(1, 2),
        "dv": (probability.transpose(-1, -2) @ dr)
        .view(batch, kv_heads, heads // kv_heads, sk, dim)
        .sum(2)
        .transpose(1, 2),
    }
    cq = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * sq
    ck = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * sk
    rng = torch.zeros(2, device="cuda", dtype=torch.int64)
    inputs = dict(q=q, k=k, v=v, do=do, o=o, stats=stats, cq=cq, ck=ck)
    saved = {name: value.clone() for name, value in inputs.items()}

    def call():
        values = fused_attn_bwd(
            sq,
            sk,
            cq,
            ck,
            q.view(batch * sq, heads, dim),
            k.view(batch * sk, kv_heads, dim),
            v.view(batch * sk, kv_heads, dim),
            o.view(batch * sq, heads, dim),
            do.view(batch * sq, heads, dim),
            q.dtype,
            [stats, rng],
            FusedAttnBackend.F16_arbitrary_seqlen,
            cu_seqlens_q_padded=cq,
            cu_seqlens_kv_padded=ck,
            attn_scale=0.0625,
            qkv_layout="thd_thd_thd",
            dqkv_layout="thd_thd_thd",
            o_format="thd",
            do_format="thd",
            attn_mask_type="padding_causal" if causal else "padding",
            window_size=(-1, 0) if causal else (-1, -1),
            bottom_right_diagonal=False,
            deterministic=False,
        )
        return {
            name: value.view_as(references[name]) for name, value in zip(references, values[:3])
        }

    for _ in range(3):
        actual = call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = call()
    comparisons = 0
    for factor in (1.0, 0.5, 1.0):
        do.copy_(saved["do"] * factor)
        for value in actual.values():
            value.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        for name, value in actual.items():
            expected = references[name] * factor
            torch.testing.assert_close(value.float(), expected, atol=0.025, rtol=0.025)
            error = value.float() - expected
            rms = error.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-8)
            assert rms.item() < 0.02, (geometry, mode, name, rms.item())
            comparisons += 1
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        graph.replay()
        torch.cuda.synchronize()
    kernels = [
        event.name
        for event in profiler.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    fills = sum("FillFunctor<c10::BFloat16>" in name for name in kernels)
    assert fills == (0 if mode == "on" else 3), kernels
    assert any("cp_uniform_lse_to_bhs" in name for name in kernels), kernels
    assert any("cudnn_generated_fort" in name and "bprop" in name for name in kernels), kernels
    for name, value in actual.items():
        torch.testing.assert_close(value.float(), references[name], atol=0.025, rtol=0.025)
        comparisons += 1
    for name, value in inputs.items():
        assert torch.equal(value, saved[name]), name
    torch.save(
        dict(
            geometry=geometry,
            mode=mode,
            shape=[batch, heads, kv_heads, sq, sk, dim],
            gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
            gradients={name: value.cpu() for name, value in actual.items()},
            kernels=kernels,
            fills=fills,
            comparisons=comparisons,
            te_source=transformer_engine.__file__,
        ),
        output,
    )


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_compact_d256_backward_initialization(geometry, tmp_path):
    import torch
    from transformer_engine.pytorch import get_cudnn_version

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("The compact D256 initialization experiment is restricted to SM100.")
    if get_cudnn_version() < (9, 29, 0):
        pytest.skip("This bounded native D256 regression requires cuDNN 9.29 or newer.")
    results = {}
    for mode in ("off", "on", "poison"):
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("NVTE_CP_EXPERIMENTAL_", "CUDNN_TE_CP_"))
            and key not in ("CUDNN_FORT_DEBUG_KERNEL_PATH", "CUDNN_FORT_CODE_GEN_ONLY_DBG")
        }
        env.update(
            NVTE_CP_EXPERIMENTAL_UNIFORM_DENSE_BWD="1",
            NVTE_CP_EXPERIMENTAL_D256_BWD_NO_ZERO=str(int(mode != "off")),
            NVTE_CP_EXPERIMENTAL_D256_BWD_POISON_GRAD=str(int(mode == "poison")),
            NVTE_FLASH_ATTN="0",
            NVTE_FUSED_ATTN="1",
            NVTE_ALLOW_NONDETERMINISTIC_ALGO="1",
        )
        output = tmp_path / (mode + ".pt")
        child = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), geometry, mode, str(output)],
            env=env,
            text=True,
            capture_output=True,
            timeout=180,
            check=False,
        )
        output.with_suffix(".stdout.log").write_text(child.stdout)
        output.with_suffix(".stderr.log").write_text(child.stderr)
        assert child.returncode == 0, child.stdout + child.stderr
        log = child.stdout + child.stderr
        assert ("TE_CP_NATIVE_D256_NO_ZERO_BWD" in log) == (mode != "off")
        assert ("TE_CP_NATIVE_D256_POISON_BWD" in log) == (mode == "poison")
        results[mode] = torch.load(output, weights_only=True)
        assert results[mode]["comparisons"] == 12
    for mode in ("on", "poison"):
        for name, original in results["off"]["gradients"].items():
            torch.testing.assert_close(results[mode]["gradients"][name], original, atol=0, rtol=0)
        assert results[mode]["te_source"] == results["off"]["te_source"]
        attention = [name for name in results[mode]["kernels"] if "cudnn_generated_fort" in name]
        baseline = [name for name in results["off"]["kernels"] if "cudnn_generated_fort" in name]
        assert attention == baseline


if __name__ == "__main__":
    assert len(sys.argv) == 4 and sys.argv[1] in GEOMETRIES
    assert sys.argv[2] in ("off", "on", "poison")
    _worker(*sys.argv[1:])
