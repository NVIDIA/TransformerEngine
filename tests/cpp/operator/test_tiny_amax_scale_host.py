#!/usr/bin/env python3
"""Host checks for flooring a tiny positive FP8 amax at max_fp8 * FLT_MIN."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HOST_SRC = Path(__file__).resolve().parent / "test_tiny_amax_scale_host.cpp"
RECIPE_CUH = ROOT / "transformer_engine" / "common" / "recipe" / "recipe_common.cuh"
REFERENCE_PY = ROOT / "tests" / "pytorch" / "references" / "quantize_scale_calc.py"
CUSTOM_REFERENCE_PY = (
    ROOT / "transformer_engine" / "pytorch" / "custom_recipes" / "reference_current_scaling.py"
)
JAX_QUANTIZER_PY = ROOT / "transformer_engine" / "jax" / "quantize" / "quantizer.py"

E4M3_MAX = np.float32(448.0)
E5M2_MAX = np.float32(57344.0)
FLT_MIN = np.float32(np.finfo(np.float32).tiny)
TINY = np.float32(7.7e-37)


def bits(value) -> int:
    return int(np.asarray(value, dtype=np.float32).view(np.uint32))


def from_bits(value: int) -> np.float32:
    return np.asarray(np.uint32(value), dtype=np.uint32).view(np.float32)


def is_subnormal(value) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float32).view(np.uint32)
    exponent = raw & np.uint32(0x7F800000)
    mantissa = raw & np.uint32(0x007FFFFF)
    return (exponent == 0) & (mantissa != 0)


def scales_from_amax(amax, max_fp8, pow2: bool, guarded: bool, epsilon: float = 0.0):
    """Host copy of compute_scale_from_amax, vectorized over amax."""
    amax = np.asarray(amax, dtype=np.float32).copy()
    epsilon_f = np.float32(epsilon)
    amax = np.where(amax < epsilon_f, epsilon_f, amax)
    max_fp8 = np.float32(max_fp8)
    guard = np.float32(max_fp8 * FLT_MIN)
    early = (~np.isfinite(amax)) | (amax == np.float32(0))
    usable = amax
    if guarded:
        usable = np.where(amax < guard, guard, amax)
    usable = np.where(early, np.float32(1), usable)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scale = np.divide(max_fp8, usable)
    scale = np.where(np.isinf(scale), np.float32(np.finfo(np.float32).max), scale).astype(
        np.float32
    )
    if pow2:
        masked = scale.view(np.uint32) & np.uint32(0xFF800000)
        scale = masked.view(np.float32)
    scale = np.where(early, np.float32(1), scale).astype(np.float32)
    scale_inv = np.divide(np.float32(1), scale).astype(np.float32)
    return scale, scale_inv


def fp8_grid(fmt: str):
    if fmt == "e4m3":
        bias, man_bits, max_exp, skip = 7, 3, 15, {(15, 7)}
    elif fmt == "e5m2":
        bias, man_bits, max_exp, skip = 15, 2, 30, set()
    else:
        raise ValueError(fmt)
    values = [np.float32(0)]
    for exp in range(0, max_exp + 1):
        for man in range(1 << man_bits):
            if exp == 0 and man == 0:
                continue
            if (exp, man) in skip:
                continue
            if exp == 0:
                value = np.float32((man / (1 << man_bits)) * 2.0 ** (1 - bias))
            else:
                value = np.float32((1.0 + man / (1 << man_bits)) * 2.0 ** (exp - bias))
            values.append(value)
    return np.asarray(values, dtype=np.float32)


GRIDS = {"e4m3": fp8_grid("e4m3"), "e5m2": fp8_grid("e5m2")}


def quantize_fp8(scaled, fmt: str) -> np.ndarray:
    """Round scaled binary32 values to the nearest finite FP8 code and return them."""
    scaled = np.asarray(scaled, dtype=np.float32)
    grid = GRIDS[fmt]
    sign = np.sign(scaled).astype(np.float32)
    sign = np.where(sign == 0, np.float32(1), sign)
    magnitude = np.abs(scaled)
    # Values below half the smallest subnormal quantize to zero.
    index = np.argmin(np.abs(magnitude[..., None] - grid), axis=-1)
    return (sign * grid[index]).astype(np.float32)


def quant_dequant(x, amax, scale, scale_inv, fmt: str):
    del amax
    quantized = quantize_fp8(np.asarray(x, dtype=np.float32) * scale, fmt)
    return quantized, (quantized * scale_inv).astype(np.float32)


def block_views_1d(x, block: int):
    matrix = np.asarray(x, dtype=np.float32)
    rows, cols = matrix.shape
    if cols % block != 0:
        raise ValueError(f"K={cols} is not divisible by block {block}")
    view = matrix.reshape(rows, cols // block, block)
    amax = np.max(np.abs(view), axis=-1)
    return view, amax


def apply_1d(x, block, max_fp8, pow2, guarded, fmt):
    view, amax = block_views_1d(x, block)
    scale, scale_inv = scales_from_amax(amax, max_fp8, pow2, guarded)
    quantized = quantize_fp8(view * scale[..., None], fmt)
    dequant = (quantized * scale_inv[..., None]).astype(np.float32)
    return (
        dequant.reshape(x.shape),
        quantized.reshape(x.shape),
        scale,
        scale_inv,
    )


def apply_2d(weight, block, max_fp8, pow2, guarded, fmt):
    matrix = np.asarray(weight, dtype=np.float32)
    rows, cols = matrix.shape
    if rows % block != 0 or cols % block != 0:
        raise ValueError(f"shape {matrix.shape} is not divisible by {block}")
    view = matrix.reshape(rows // block, block, cols // block, block).transpose(0, 2, 1, 3)
    amax = np.max(np.abs(view), axis=(-1, -2))
    scale, scale_inv = scales_from_amax(amax, max_fp8, pow2, guarded)
    quantized = quantize_fp8(view * scale[..., None, None], fmt)
    dequant = (quantized * scale_inv[..., None, None]).astype(np.float32)
    restored = dequant.transpose(0, 2, 1, 3).reshape(matrix.shape)
    return restored, scale, scale_inv


def run_host() -> list[dict]:
    binary = Path("/tmp/fp8_scale_guard_host")
    compile_cmd = ["g++", "-O2", "-std=c++17", "-Wall", "-Wextra", str(HOST_SRC), "-o", str(binary)]
    compiled = subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
    if compiled.stderr.strip():
        print(compiled.stderr, file=sys.stderr)
    completed = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    rows = []
    for line in completed.stdout.splitlines():
        parts = line.split()
        if parts[0] in {"e4m3", "e5m2"}:
            rows.append(
                {
                    "format": parts[0],
                    "pow2": bool(int(parts[1])),
                    "label": parts[2],
                    "amax": parts[3],
                    "scale": parts[4],
                    "scale_inv": parts[5],
                    "legacy_scale": parts[6],
                    "legacy_scale_inv": parts[7],
                }
            )
        elif parts[0] == "HOST_OK":
            continue
        else:
            print(line, file=sys.stderr)
    if "HOST_OK" not in completed.stdout:
        raise AssertionError("host checker did not report success")
    return rows


def cross_check_numpy(rows: list[dict]) -> None:
    maxima = {"e4m3": E4M3_MAX, "e5m2": E5M2_MAX}
    for row in rows:
        amax = from_bits(int(row["amax"], 16))
        for guarded, scale_key, inv_key in (
            (True, "scale", "scale_inv"),
            (False, "legacy_scale", "legacy_scale_inv"),
        ):
            scale, scale_inv = scales_from_amax(amax, maxima[row["format"]], row["pow2"], guarded)
            if bits(scale) != int(row[scale_key], 16) or bits(scale_inv) != int(row[inv_key], 16):
                raise AssertionError(f"numpy/host mismatch for {row} guarded={guarded}")


def check_sources() -> None:
    recipe = RECIPE_CUH.read_text()
    if (
        "amax < amax_floor" not in recipe
        or "amax == 0.f" not in recipe
        or "#include <cfloat>" not in recipe
    ):
        raise AssertionError("recipe_common.cuh does not floor a positive tiny amax")
    reference = REFERENCE_PY.read_text()
    if (
        "torch.finfo(torch.float32).tiny" not in reference
        or "amax < guard" not in reference
        or "amax == 0" not in reference
    ):
        raise AssertionError("quantize_scale_calc.py does not mirror the floor")
    custom = CUSTOM_REFERENCE_PY.read_text()
    if "amax < guard" not in custom or "amax == 0" not in custom:
        raise AssertionError("reference_current_scaling.py does not mirror the floor")
    jax = JAX_QUANTIZER_PY.read_text()
    if (
        "jnp.finfo(jnp.float32).tiny" not in jax
        or "amax < guard" not in jax
        or "amax > 0" not in jax
    ):
        raise AssertionError("jax quantizer.py does not mirror the floor")


def assert_random_bitwise(fmt: str, max_fp8, pow2: bool) -> dict:
    rng = np.random.default_rng(20261028)
    tensor = rng.standard_normal((64, 256)).astype(np.float32)
    new_dq, new_q, new_scale, new_inv = apply_1d(tensor, 128, max_fp8, pow2, True, fmt)
    old_dq, old_q, old_scale, old_inv = apply_1d(tensor, 128, max_fp8, pow2, False, fmt)
    if not np.array_equal(new_q.view(np.uint32), old_q.view(np.uint32)):
        raise AssertionError(f"random quantized bits changed for {fmt} pow2={pow2}")
    if not np.array_equal(new_scale.view(np.uint32), old_scale.view(np.uint32)):
        raise AssertionError(f"random scale bits changed for {fmt} pow2={pow2}")
    if not np.array_equal(new_inv.view(np.uint32), old_inv.view(np.uint32)):
        raise AssertionError(f"random scale_inv bits changed for {fmt} pow2={pow2}")
    if not np.array_equal(new_dq.view(np.uint32), old_dq.view(np.uint32)):
        raise AssertionError(f"random dequant bits changed for {fmt} pow2={pow2}")
    return {
        "format": fmt,
        "pow2": pow2,
        "elements": int(tensor.size),
        "blocks": int(new_scale.size),
        "bitwise_identical": True,
    }


def _finite_normal_invs(invs) -> None:
    for inv in invs:
        array = np.asarray(inv, dtype=np.float32)
        if not np.isfinite(array).all() or np.any(is_subnormal(array)) or np.any(array < FLT_MIN):
            raise AssertionError("scale_inv is non-finite or smaller than FLT_MIN")


def linear_case(recipe: str) -> dict:
    """Column-shaped GEMM with one tiny block, for blockwise and tensorwise scales."""
    rng = np.random.default_rng(7)
    rows, in_features, out_features = 64, 256, 128
    activation = rng.standard_normal((rows, in_features)).astype(np.float32) * np.float32(0.2)
    weight = rng.standard_normal((out_features, in_features)).astype(np.float32) * np.float32(0.05)
    grad_out = rng.standard_normal((rows, out_features)).astype(np.float32) * np.float32(0.2)
    activation[0, 0:128] = TINY
    grad_out[0, 0:128] = TINY
    pow2 = recipe == "blockwise"
    if recipe == "blockwise":
        weight[0:128, 0:128] = TINY

    def quantize_weight(guarded: bool):
        if recipe == "blockwise":
            return apply_2d(weight, 128, E4M3_MAX, True, guarded, "e4m3")
        scale, scale_inv = scales_from_amax(np.max(np.abs(weight)), E4M3_MAX, False, guarded)
        quantized = quantize_fp8(weight * scale, "e4m3")
        return (quantized * scale_inv).astype(np.float32), scale, scale_inv

    def quantize_matrix(matrix, guarded: bool):
        if recipe == "blockwise":
            return apply_1d(matrix, 128, E4M3_MAX, True, guarded, "e4m3")
        scale, scale_inv = scales_from_amax(np.max(np.abs(matrix)), E4M3_MAX, False, guarded)
        quantized = quantize_fp8(matrix * scale, "e4m3")
        dequant = (quantized * scale_inv).astype(np.float32)
        return dequant, quantized, scale, scale_inv

    def forward_backward(guarded: bool, x_value, dy_value):
        x_dq, x_q, x_scale, x_inv = quantize_matrix(x_value, guarded)
        w_dq, w_scale, w_inv = quantize_weight(guarded)
        dy_dq, dy_q, dy_scale, dy_inv = quantize_matrix(dy_value, guarded)
        output = (x_dq @ w_dq.T).astype(np.float32)
        dgrad = (dy_dq @ w_dq).astype(np.float32)
        wgrad = (dy_dq.T @ x_dq).astype(np.float32)
        return output, dgrad, wgrad, (x_inv, w_inv, dy_inv, x_q, dy_q)

    output, dgrad, wgrad, meta = forward_backward(True, activation, grad_out)
    for name, tensor in (("y", output), ("dgrad", dgrad), ("wgrad", wgrad)):
        if not np.isfinite(tensor).all():
            raise AssertionError(f"{recipe} {name} contains a non-finite value")
    _finite_normal_invs(meta[:3])
    legacy = forward_backward(False, activation, grad_out)
    legacy_subnormal = any(bool(np.any(is_subnormal(inv))) for inv in legacy[3][:3])
    result = {
        "recipe": recipe,
        "pow2_scales": pow2,
        "y_finite": True,
        "dgrad_finite": True,
        "wgrad_finite": True,
        "scale_inv_ge_flt_min": True,
        "y_max_abs": float(np.max(np.abs(output))),
        "dgrad_max_abs": float(np.max(np.abs(dgrad))),
        "wgrad_max_abs": float(np.max(np.abs(wgrad))),
        "legacy_subnormal_on_partial_blocks": legacy_subnormal,
    }
    if recipe == "blockwise":
        if not legacy_subnormal:
            raise AssertionError("blockwise legacy path did not reproduce a subnormal scale_inv")
        injected_codes = meta[3][0, 0:128]
        if np.all(injected_codes == 0):
            raise AssertionError("injected 1x128 activation was flushed to zero")
        if bits(meta[0][0, 0]) != 0x00800000:
            raise AssertionError("injected block scale_inv is not FLT_MIN")
        result["injected_block_kept"] = True
        result["injected_block_scale_inv_bits"] = f"{bits(meta[0][0, 0]):08x}"
        result["injected_block_max_abs_code"] = float(np.max(np.abs(injected_codes)))
        return result
    if legacy_subnormal:
        raise AssertionError("partial injection unexpectedly changed the tensorwise amax")
    if not np.array_equal(output.view(np.uint32), legacy[0].view(np.uint32)):
        raise AssertionError("tensorwise partial injection changed the guarded result")
    tiny_activation = np.full_like(activation, TINY)
    tiny_out, tiny_dgrad, tiny_wgrad, tiny_meta = forward_backward(True, tiny_activation, grad_out)
    legacy_tiny = forward_backward(False, tiny_activation, grad_out)
    if not (
        np.isfinite(tiny_out).all()
        and np.isfinite(tiny_dgrad).all()
        and np.isfinite(tiny_wgrad).all()
    ):
        raise AssertionError("tiny tensorwise activation produced a non-finite gradient")
    _finite_normal_invs(tiny_meta[:3])
    if not np.any(is_subnormal(legacy_tiny[3][0])):
        raise AssertionError("legacy tensorwise path did not leave a subnormal scale_inv")
    if np.all(tiny_meta[3] == 0):
        raise AssertionError("fully tiny tensorwise activation was flushed to zero")
    if bits(tiny_meta[0]) != 0x00800000:
        raise AssertionError("tiny tensorwise scale_inv is not FLT_MIN")
    result["partial_block_bitwise_identical"] = True
    result["full_tensor_tiny_kept"] = True
    result["full_tensor_tiny_y_max_abs"] = float(np.max(np.abs(tiny_out)))
    result["full_tensor_tiny_scale_inv_bits"] = f"{bits(tiny_meta[0]):08x}"
    result["legacy_subnormal_on_full_tensor"] = True
    return result


def grouped_linear_case() -> dict:
    rng = np.random.default_rng(11)
    experts, tokens, in_features, out_features = 2, 32, 128, 128
    activation = rng.standard_normal((experts, tokens, in_features)).astype(np.float32)
    weight = rng.standard_normal((experts, out_features, in_features)).astype(
        np.float32
    ) * np.float32(0.05)
    grad_out = rng.standard_normal((experts, tokens, out_features)).astype(np.float32)
    activation[1].fill(TINY)
    grad_out[1].fill(TINY)
    outputs = []
    for expert in range(experts):
        x_dq, _, _, x_inv = apply_1d(activation[expert], 128, E4M3_MAX, True, True, "e4m3")
        w_dq, _, w_inv = apply_2d(weight[expert], 128, E4M3_MAX, True, True, "e4m3")
        dy_dq, _, _, dy_inv = apply_1d(grad_out[expert], 128, E4M3_MAX, True, True, "e4m3")
        output = x_dq @ w_dq.T
        dgrad = dy_dq @ w_dq
        wgrad = dy_dq.T @ x_dq
        for tensor in (output, dgrad, wgrad):
            if not np.isfinite(tensor).all():
                raise AssertionError("grouped linear produced a non-finite value")
        for tensor in (x_inv, w_inv, dy_inv):
            if np.any(is_subnormal(tensor)) or np.any(
                np.asarray(tensor, dtype=np.float32) < FLT_MIN
            ):
                raise AssertionError("grouped linear stored a subnormal scale_inv")
        outputs.append(output)
        if expert == 1:
            if np.all(x_dq == 0) or np.all(output == 0):
                raise AssertionError("tiny expert activation was flushed to zero")
            if not np.all(x_inv == FLT_MIN):
                raise AssertionError("tiny expert scale_inv was not floored at FLT_MIN")
    return {
        "experts": experts,
        "tiny_expert_output_kept": True,
        "tiny_expert_output_max_abs": float(np.max(np.abs(outputs[1]))),
        "normal_expert_output_max_abs": float(np.max(np.abs(outputs[0]))),
    }


def train_proxy(steps: int = 15) -> dict:
    rng = np.random.default_rng(19)
    tokens, in_features, hidden, out_features = 32, 128, 128, 128
    data_x = rng.standard_normal((steps, tokens, in_features)).astype(np.float32) * np.float32(0.3)
    data_y = rng.standard_normal((steps, tokens, out_features)).astype(np.float32)

    start_w1 = rng.standard_normal((hidden, in_features)).astype(np.float32) * np.float32(0.05)
    start_w2 = rng.standard_normal((out_features, hidden)).astype(np.float32) * np.float32(0.05)

    def execute(guarded: bool, inject: bool):
        w1 = start_w1.copy()
        w2 = start_w2.copy()
        losses = []
        grad_norms = []
        subnormal_inv = False
        started = time.perf_counter()
        for step in range(steps):
            activation = data_x[step].copy()
            if inject:
                activation[0, :] = TINY
            x_dq, _, _, x_inv = apply_1d(activation, 128, E4M3_MAX, True, guarded, "e4m3")
            if inject and guarded and np.all(x_dq[0] == 0):
                raise AssertionError("injected activation row was flushed to zero")
            subnormal_inv = subnormal_inv or bool(np.any(is_subnormal(x_inv)))
            hidden_pre = (x_dq @ w1.T).astype(np.float32)
            h_dq, _, _, h_inv = apply_1d(hidden_pre, 128, E4M3_MAX, True, guarded, "e4m3")
            subnormal_inv = subnormal_inv or bool(np.any(is_subnormal(h_inv)))
            pred = (h_dq @ w2.T).astype(np.float32)
            residual = (pred - data_y[step]).astype(np.float32)
            loss = np.float32(np.mean(np.square(residual)))
            dy = (np.float32(2.0) * residual / np.float32(residual.size)).astype(np.float32)
            if inject:
                dy[0, :] = TINY
            dy_dq, _, _, dy_inv = apply_1d(dy, 128, E4M3_MAX, True, guarded, "e4m3")
            subnormal_inv = subnormal_inv or bool(np.any(is_subnormal(dy_inv)))
            dw2 = (dy_dq.T @ h_dq).astype(np.float32)
            dh = (dy_dq @ w2).astype(np.float32)
            dh_dq, _, _, dh_inv = apply_1d(dh, 128, E4M3_MAX, True, guarded, "e4m3")
            subnormal_inv = subnormal_inv or bool(np.any(is_subnormal(dh_inv)))
            dw1 = (dh_dq.T @ x_dq).astype(np.float32)
            if not np.isfinite(loss) or not np.isfinite(dw1).all() or not np.isfinite(dw2).all():
                raise AssertionError(f"non-finite training state guarded={guarded} inject={inject}")
            grad_norm = np.float32(
                math.sqrt(float(np.sum(np.square(dw1)) + np.sum(np.square(dw2))))
            )
            w1 -= np.float32(0.05) * dw1
            w2 -= np.float32(0.05) * dw2
            losses.append(float(loss))
            grad_norms.append(float(grad_norm))
        return {
            "losses": losses,
            "grad_norms": grad_norms,
            "elapsed_sec": time.perf_counter() - started,
            "subnormal_scale_inv": subnormal_inv,
        }

    normal_new = execute(True, False)
    normal_old = execute(False, False)
    injected_new = execute(True, True)
    injected_old = execute(False, True)
    if (
        normal_new["losses"] != normal_old["losses"]
        or normal_new["grad_norms"] != normal_old["grad_norms"]
    ):
        raise AssertionError("normal-data loss or grad norm changed")
    if injected_new["subnormal_scale_inv"] or not injected_old["subnormal_scale_inv"]:
        raise AssertionError("injected smoke did not isolate the legacy subnormal")
    return {
        "steps": steps,
        "normal_losses": normal_new["losses"],
        "normal_grad_norms": normal_new["grad_norms"],
        "normal_elapsed_new_sec": normal_new["elapsed_sec"],
        "normal_elapsed_old_sec": normal_old["elapsed_sec"],
        "bitwise_loss_and_grad_norm": True,
        "injected_losses_new": injected_new["losses"],
        "injected_grad_norms_new": injected_new["grad_norms"],
        "injected_losses_old": injected_old["losses"],
        "injected_elapsed_new_sec": injected_new["elapsed_sec"],
        "injected_elapsed_old_sec": injected_old["elapsed_sec"],
        "injected_new_subnormal": injected_new["subnormal_scale_inv"],
        "injected_old_subnormal": injected_old["subnormal_scale_inv"],
    }


def probe_gpu_stack() -> dict:
    torch_status = "not installed"
    try:
        import torch  # type: ignore

        torch_status = f"{torch.__version__} cuda={torch.cuda.is_available()}"
    except Exception as exc:  # noqa: BLE001 - the absence itself is the result.
        torch_status = f"unavailable ({type(exc).__name__}: {exc})"
    te_status = "not installed"
    try:
        import transformer_engine  # type: ignore

        te_status = getattr(transformer_engine, "__file__", "imported")
    except Exception as exc:  # noqa: BLE001
        te_status = f"unavailable ({type(exc).__name__}: {exc})"
    return {
        "nvidia_smi": shutil.which("nvidia-smi") or "not found",
        "nvcc": shutil.which("nvcc") or "not found",
        "torch": torch_status,
        "transformer_engine_import": te_status,
        "cuda_operator_tests": "not executed; test_operator requires nvcc, CUDA, and a GPU",
        "pytorch_quantization_tests": (
            "not executed; no PyTorch or CUDA runtime in this environment"
        ),
    }


def main() -> None:
    check_sources()
    rows = run_host()
    cross_check_numpy(rows)
    random_rows = [
        assert_random_bitwise("e4m3", E4M3_MAX, False),
        assert_random_bitwise("e4m3", E4M3_MAX, True),
        assert_random_bitwise("e5m2", E5M2_MAX, False),
        assert_random_bitwise("e5m2", E5M2_MAX, True),
    ]
    # 2D 128x128 random tiles must also be unchanged.
    rng = np.random.default_rng(3)
    weight = rng.standard_normal((256, 256)).astype(np.float32)
    for pow2 in (False, True):
        new = apply_2d(weight, 128, E4M3_MAX, pow2, True, "e4m3")
        old = apply_2d(weight, 128, E4M3_MAX, pow2, False, "e4m3")
        if any(not np.array_equal(a.view(np.uint32), b.view(np.uint32)) for a, b in zip(new, old)):
            raise AssertionError("2D random block quantization changed")
    linear = [linear_case("blockwise"), linear_case("tensorwise")]
    grouped = grouped_linear_case()
    training = train_proxy()
    environment = probe_gpu_stack()
    # Confirm the FP8 grid contains the format maxima used by the guard.
    if E4M3_MAX not in GRIDS["e4m3"] or E5M2_MAX not in GRIDS["e5m2"]:
        raise AssertionError("FP8 grid is missing its maximum finite value")
    summary = {
        "host_rows": rows,
        "random": random_rows,
        "linear": linear,
        "grouped": grouped,
        "training_proxy": training,
        "environment": environment,
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
