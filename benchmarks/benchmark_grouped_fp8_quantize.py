# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark grouped FP8 tensor-scaling quantization."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import torch
import transformer_engine.common as te_common
from transformer_engine.pytorch import Float8CurrentScalingQuantizer, Float8Quantizer
from transformer_engine.pytorch.utils import is_non_tn_fp8_gemm_supported
import transformer_engine_torch as tex


MODES = ("rowwise", "columnwise", "both")
SHAPE_CASES = ("same-shape", "varying-first", "varying-last")
EXTENSION_ROOT_ENV = "NVTE_BENCHMARK_EXPECT_EXTENSION_ROOT"


def _default_output_path() -> str:
    output_dir = os.environ.get("ORCHESTRA_BENCHMARK_OUTPUT_DIR")
    if output_dir:
        return str(Path(output_dir) / "grouped_fp8_quantize_report.json")
    return "grouped_fp8_quantize_report.json"


def _check_output_path(output_path: str) -> None:
    wrapper_raw_report = os.environ.get("ORCHESTRA_BENCHMARK_RAW_REPORT")
    if wrapper_raw_report is None:
        return
    if Path(output_path).expanduser().resolve() == Path(wrapper_raw_report).expanduser().resolve():
        raise ValueError(
            "--output must not point at ORCHESTRA_BENCHMARK_RAW_REPORT because the "
            "benchmark wrapper writes its own command report there. Use a separate "
            "script report path under ORCHESTRA_BENCHMARK_OUTPUT_DIR instead; this "
            "script mirrors that completed report for the wrapper fetch path."
        )


def _write_report(report: Dict[str, object], output_path: str) -> None:
    payload = json.dumps(report, indent=2)
    output = Path(output_path).expanduser()
    if str(output.parent) not in ("", "."):
        output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    _mirror_report_to_orchestra_raw_path(output, payload)
    print(payload)


def _mirror_report_to_orchestra_raw_path(output_path: Path, payload: str) -> None:
    wrapper_raw_report = os.environ.get("ORCHESTRA_BENCHMARK_RAW_REPORT")
    if not wrapper_raw_report:
        return

    raw_report = Path(wrapper_raw_report).expanduser()
    if output_path.resolve() == raw_report.resolve():
        return
    if str(raw_report.parent) not in ("", "."):
        raw_report.parent.mkdir(parents=True, exist_ok=True)
    raw_report.write_text(payload, encoding="utf-8")


def _usage_for_mode(mode: str) -> Dict[str, bool]:
    return {
        "rowwise": mode in ("rowwise", "both"),
        "columnwise": mode in ("columnwise", "both"),
    }


def _make_current_quantizer(mode: str) -> Float8CurrentScalingQuantizer:
    usage = _usage_for_mode(mode)
    quantizer = Float8CurrentScalingQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        device="cuda",
        force_pow_2_scales=False,
        amax_epsilon=0.0,
    )
    quantizer.set_usage(**usage)
    return quantizer


def _make_delayed_quantizer(mode: str, scale: torch.Tensor, amax: torch.Tensor) -> Float8Quantizer:
    usage = _usage_for_mode(mode)
    return Float8Quantizer(
        scale=scale,
        amax=amax,
        fp8_dtype=tex.DType.kFloat8E4M3,
        **usage,
    )


def _group_quantize(
    tensor: torch.Tensor,
    quantizer,
    num_groups: int,
    first_dims: Optional[torch.Tensor],
    *,
    output=None,
    last_dims: Optional[torch.Tensor] = None,
):
    kwargs = {}
    if output is not None:
        kwargs["output"] = output
    if last_dims is not None:
        kwargs["last_dims"] = last_dims
    return tex.group_quantize(tensor, quantizer, num_groups, first_dims, **kwargs)


def _varying_dims(base: int, num_groups: int) -> List[int]:
    step = 128
    center = (num_groups - 1) / 2.0
    dims = []
    for idx in range(num_groups):
        value = base + int(idx - center) * step
        dims.append(max(step, (value // step) * step))
    return dims


def _varying_last_dims(base: int, num_groups: int) -> List[int]:
    aligned_dims = _varying_dims(base, num_groups)
    offsets = (13, 29, 47, 61, 83, 97, 109, 127)
    return [dim + offsets[idx % len(offsets)] for idx, dim in enumerate(aligned_dims)]


def _make_inputs(
    *,
    shape_case: str,
    num_buffers: int,
    num_groups: int,
    rows_per_group: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> Tuple[List[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any]]:
    if shape_case == "same-shape":
        rows = num_groups * rows_per_group
        inputs = [
            torch.randn(rows, hidden_size, dtype=dtype, device="cuda")
            for _ in range(num_buffers)
        ]
        element_counts = {
            "actual_elements": rows * hidden_size,
            "allocated_elements": rows * hidden_size,
            "unused_tail_elements": 0,
            "actual_rows": rows,
            "allocated_rows": rows,
            "first_dims": [rows_per_group] * num_groups,
            "last_dims": [hidden_size] * num_groups,
        }
        return inputs, None, None, element_counts

    if shape_case == "varying-first":
        first_dims_list = _varying_dims(rows_per_group, num_groups)
        actual_rows = sum(first_dims_list)
        allocated_rows = actual_rows * 2
        first_dims = torch.tensor(first_dims_list, dtype=torch.int64, device="cuda")
        inputs = []
        for _ in range(num_buffers):
            tensor = torch.randn(allocated_rows, hidden_size, dtype=dtype, device="cuda")
            tensor[actual_rows:].fill_(10000.0)
            inputs.append(tensor)
        element_counts = {
            "actual_elements": actual_rows * hidden_size,
            "allocated_elements": allocated_rows * hidden_size,
            "unused_tail_elements": (allocated_rows - actual_rows) * hidden_size,
            "actual_rows": actual_rows,
            "allocated_rows": allocated_rows,
            "first_dims": first_dims_list,
            "last_dims": [hidden_size] * num_groups,
        }
        return inputs, first_dims, None, element_counts

    if shape_case == "varying-last":
        last_dims_list = _varying_last_dims(hidden_size, num_groups)
        total_cols = sum(last_dims_list)
        actual_elements = rows_per_group * total_cols
        last_dims = torch.tensor(last_dims_list, dtype=torch.int64, device="cuda")
        inputs = []
        for _ in range(num_buffers):
            flat = torch.empty(actual_elements, dtype=dtype, device="cuda")
            offset = 0
            for cols in last_dims_list:
                group = torch.randn(rows_per_group, cols, dtype=dtype, device="cuda")
                flat[offset : offset + group.numel()].copy_(group.reshape(-1))
                offset += group.numel()
            inputs.append(flat.view(rows_per_group, total_cols))
        element_counts = {
            "actual_elements": actual_elements,
            "allocated_elements": actual_elements,
            "unused_tail_elements": 0,
            "actual_rows": rows_per_group,
            "allocated_rows": rows_per_group,
            "first_dims": [rows_per_group] * num_groups,
            "last_dims": last_dims_list,
        }
        return inputs, None, last_dims, element_counts

    raise ValueError(f"Unknown shape case: {shape_case}")


def _prepare_mode(
    mode: str,
    inputs: List[torch.Tensor],
    first_dims: Optional[torch.Tensor],
    last_dims: Optional[torch.Tensor],
    num_groups: int,
) -> Tuple[List[Float8Quantizer], list]:
    delayed_quantizers = []
    outputs = []
    current_quantizer = _make_current_quantizer(mode)
    for tensor in inputs:
        prepared = _group_quantize(
            tensor, current_quantizer, num_groups, first_dims, last_dims=last_dims
        )
        delayed_quantizer = _make_delayed_quantizer(mode, prepared.scale, prepared.amax)
        output = _group_quantize(
            tensor, delayed_quantizer, num_groups, first_dims, last_dims=last_dims
        )
        delayed_quantizers.append(delayed_quantizer)
        outputs.append(output)
    return delayed_quantizers, outputs


def _prepare_benchmark_state(
    args: argparse.Namespace, shape_case: str, mode: str
) -> Dict[str, Any]:
    inputs, first_dims, last_dims, element_counts = _make_inputs(
        shape_case=shape_case,
        num_buffers=args.num_buffers,
        num_groups=args.num_groups,
        rows_per_group=args.rows_per_group,
        hidden_size=args.hidden_size,
        dtype=torch.bfloat16,
    )
    quantizers, outputs = _prepare_mode(mode, inputs, first_dims, last_dims, args.num_groups)
    first_output = outputs[0]
    return {
        "shape_case": shape_case,
        "mode": mode,
        "inputs": inputs,
        "first_dims": first_dims,
        "last_dims": last_dims,
        "element_counts": element_counts,
        "quantizers": quantizers,
        "outputs": outputs,
        "materialized_rowwise": first_output.rowwise_data is not None,
        "materialized_columnwise": first_output.columnwise_data is not None,
    }


def _warmup_benchmark_state(args: argparse.Namespace, state: Dict[str, Any]) -> None:
    for iteration in range(args.warmup_iters):
        idx = iteration % len(state["inputs"])
        _group_quantize(
            state["inputs"][idx],
            state["quantizers"][idx],
            args.num_groups,
            state["first_dims"],
            output=state["outputs"][idx],
            last_dims=state["last_dims"],
        )


def _bytes_per_call(
    actual_elements: int,
    input_element_size: int,
    materialized_rowwise: bool,
    materialized_columnwise: bool,
) -> int:
    output_copies = int(materialized_rowwise) + int(materialized_columnwise)
    return actual_elements * (input_element_size + output_copies)


def _run_timed_loop(
    *,
    shape_case: str,
    mode: str,
    inputs: List[torch.Tensor],
    quantizers: List[Float8Quantizer],
    outputs: list,
    first_dims: Optional[torch.Tensor],
    last_dims: Optional[torch.Tensor],
    num_groups: int,
    iterations: int,
    profile: bool,
) -> float:
    torch.cuda.synchronize()
    if profile:
        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.nvtx.range_push(f"grouped_fp8_quantize:{shape_case}:{mode}")
    start.record()
    for iteration in range(iterations):
        idx = iteration % len(inputs)
        _group_quantize(
            inputs[idx],
            quantizers[idx],
            num_groups,
            first_dims,
            output=outputs[idx],
            last_dims=last_dims,
        )
    end.record()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    if profile:
        torch.cuda.cudart().cudaProfilerStop()
    return start.elapsed_time(end) / 1000.0


def _target_tbps(shape_case: str) -> float:
    return 4.0 if shape_case == "varying-last" else 6.0


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _extension_path(path: Optional[str]) -> str:
    return str(Path(path).resolve()) if path else ""


def _loaded_extension_metadata() -> Dict[str, object]:
    common_path = te_common._get_shared_object_file("core")  # pylint: disable=protected-access
    torch_path = te_common._get_shared_object_file("torch")  # pylint: disable=protected-access
    metadata: Dict[str, object] = {
        "python_package_path": _extension_path(tex.__file__),
        "core_shared_object_path": str(common_path.resolve()),
        "torch_shared_object_path": str(torch_path.resolve()),
    }
    expected_root = os.environ.get(EXTENSION_ROOT_ENV)
    if expected_root:
        root = Path(expected_root).resolve()
        metadata["expected_extension_root"] = str(root)
        metadata["loaded_from_expected_root"] = (
            _path_is_relative_to(common_path.resolve(), root)
            and _path_is_relative_to(torch_path.resolve(), root)
        )
        if not metadata["loaded_from_expected_root"]:
            raise RuntimeError(
                "Transformer Engine shared objects were not loaded from the expected "
                f"baseline build root {root}: core={common_path.resolve()}, "
                f"torch={torch_path.resolve()}"
            )
    return metadata


def _make_result(
    args: argparse.Namespace,
    state: Dict[str, Any],
    elapsed_sec: float,
    iterations: int,
) -> Dict[str, object]:
    element_counts = state["element_counts"]
    input_element_size = torch.tensor([], dtype=torch.bfloat16).element_size()
    relevant_bytes = _bytes_per_call(
        element_counts["actual_elements"],
        input_element_size,
        state["materialized_rowwise"],
        state["materialized_columnwise"],
    )
    excluded_tail_bytes = _bytes_per_call(
        element_counts["unused_tail_elements"],
        input_element_size,
        state["materialized_rowwise"],
        state["materialized_columnwise"],
    )
    bandwidth_tbps = relevant_bytes * iterations / elapsed_sec / 1.0e12
    threshold = _target_tbps(state["shape_case"])
    case_id = f"{state['shape_case']}/{state['mode']}"
    return {
        "case_id": case_id,
        "shape_case": state["shape_case"],
        "mode": state["mode"],
        "num_groups": args.num_groups,
        "first_dims": element_counts["first_dims"],
        "last_dims": element_counts["last_dims"],
        "hidden_size_argument": args.hidden_size,
        "input_dtype": "bfloat16",
        "output_dtype": "float8_e4m3",
        "non_tn_fp8_gemm_supported": is_non_tn_fp8_gemm_supported(),
        "materialized_rowwise": state["materialized_rowwise"],
        "materialized_columnwise": state["materialized_columnwise"],
        "num_distinct_input_buffers": args.num_buffers,
        "warmup_iterations": args.warmup_iters,
        "timed_iterations": iterations,
        "elapsed_sec": elapsed_sec,
        "relevant_bytes_per_call": relevant_bytes,
        "allocated_elements": element_counts["allocated_elements"],
        "actual_elements": element_counts["actual_elements"],
        "unused_tail_elements": element_counts["unused_tail_elements"],
        "tail_fraction_of_allocated": (
            element_counts["unused_tail_elements"] / element_counts["allocated_elements"]
        ),
        "allocated_rows": element_counts["allocated_rows"],
        "actual_rows": element_counts["actual_rows"],
        "allocated_bytes_excluded_from_bandwidth": excluded_tail_bytes,
        "bandwidth_TBps_actual_bytes": bandwidth_tbps,
        "target_TBps_actual_bytes": threshold,
        "passes_target": bandwidth_tbps >= threshold,
        "gb200_peak_bandwidth_TBps_expectation": 8.0,
        "fraction_of_8TBps_peak": bandwidth_tbps / 8.0,
        "timed_call": (
            "tex.group_quantize(input, Float8Quantizer(precomputed_scale), ..., "
            "output=preallocated)"
        ),
    }


def _result_case_id(result: Dict[str, object]) -> str:
    return str(result.get("case_id", f"{result['shape_case']}/{result['mode']}"))


def _annotate_results(
    results: List[Dict[str, object]], run_role: str, git_ref: str
) -> List[Dict[str, object]]:
    annotated = []
    for result in results:
        result = dict(result)
        case_id = _result_case_id(result)
        result["case_id"] = case_id
        result["run_role"] = run_role
        result["git_ref"] = git_ref
        result["result_id"] = f"{run_role}/{case_id}"
        annotated.append(result)
    return annotated


def _make_measurements(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    measurements = []
    for idx, result in enumerate(results):
        case_id = _result_case_id(result)
        run_role = str(result.get("run_role", "candidate"))
        measurements.append(
            {
                "measurement_id": f"{run_role}/{case_id}/bandwidth_TBps_actual_bytes",
                "case_id": case_id,
                "result_id": str(result.get("result_id", f"{run_role}/{case_id}")),
                "shape_case": str(result["shape_case"]),
                "mode": str(result["mode"]),
                "run_role": run_role,
                "git_ref": str(result.get("git_ref", "")),
                "metric": "bandwidth_TBps_actual_bytes",
                "value": float(result["bandwidth_TBps_actual_bytes"]),
                "unit": "TB/s",
                "iteration": idx,
                "higher_is_better": True,
            }
        )
    return measurements


def _selected_modes(args: argparse.Namespace) -> Tuple[str, ...]:
    return MODES if args.mode == "all" else (args.mode,)


def _selected_shape_cases(args: argparse.Namespace) -> Tuple[str, ...]:
    return SHAPE_CASES if args.shape_case == "all" else (args.shape_case,)


def _expected_case_ids(
    shape_cases: Tuple[str, ...], modes: Tuple[str, ...]
) -> Tuple[str, ...]:
    return tuple(f"{shape_case}/{mode}" for shape_case in shape_cases for mode in modes)


def _require_case_ids(
    results: List[Dict[str, object]], expected_case_ids: Tuple[str, ...], label: str
) -> None:
    result_case_ids = {_result_case_id(result) for result in results}
    missing_case_ids = sorted(set(expected_case_ids) - result_case_ids)
    if missing_case_ids:
        raise RuntimeError(f"{label} missing required case IDs: {missing_case_ids}")


def benchmark_case_mode(args: argparse.Namespace, shape_case: str, mode: str) -> Dict[str, object]:
    state = _prepare_benchmark_state(args, shape_case, mode)
    _warmup_benchmark_state(args, state)
    iterations = args.profile_iters if args.profile else args.iters
    elapsed_sec = _run_timed_loop(
        shape_case=shape_case,
        mode=mode,
        inputs=state["inputs"],
        quantizers=state["quantizers"],
        outputs=state["outputs"],
        first_dims=state["first_dims"],
        last_dims=state["last_dims"],
        num_groups=args.num_groups,
        iterations=iterations,
        profile=args.profile,
    )
    return _make_result(args, state, elapsed_sec, iterations)


def _git_output(repo_root: Path, *args: str) -> str:
    try:
        output = subprocess.check_output(
            ["git", *args], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""
    return output.strip()


def _benchmark_subprocess_command(
    args: argparse.Namespace,
    script_path: Path,
    output_path: Path,
    shape_case: str,
) -> List[str]:
    command = [
        sys.executable,
        str(script_path),
        "--shape-case",
        shape_case,
        "--mode",
        args.mode,
        "--num-groups",
        str(args.num_groups),
        "--rows-per-group",
        str(args.rows_per_group),
        "--hidden-size",
        str(args.hidden_size),
        "--num-buffers",
        str(args.num_buffers),
        "--warmup-iters",
        str(args.warmup_iters),
        "--iters",
        str(args.iters),
        "--profile-iters",
        str(args.profile_iters),
        "--output",
        str(output_path),
    ]
    if args.profile:
        command.append("--profile")
    return command


def _install_baseline_package(baseline_worktree: Path, install_root: Path) -> Dict[str, object]:
    install_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["NVTE_FRAMEWORK"] = "pytorch"
    submodule_command = ["git", "submodule", "update", "--init", "--recursive"]
    submodule_completed = subprocess.run(
        submodule_command,
        cwd=baseline_worktree,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if submodule_completed.returncode != 0:
        print(submodule_completed.stdout, file=sys.stderr, end="")
        submodule_completed.check_returncode()

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        "--no-deps",
        "--target",
        str(install_root),
        ".",
    ]
    completed = subprocess.run(
        command,
        cwd=baseline_worktree,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        print(completed.stdout, file=sys.stderr, end="")
        completed.check_returncode()

    core_objects = sorted(
        str(path.resolve()) for path in install_root.glob("**/libtransformer_engine*.so")
    )
    torch_objects = sorted(
        str(path.resolve()) for path in install_root.glob("**/transformer_engine_torch*.so")
    )
    if not core_objects or not torch_objects:
        raise RuntimeError(
            "Baseline package build completed without the required Transformer Engine "
            f"shared objects under {install_root}: core={core_objects}, torch={torch_objects}"
        )
    return {
        "submodule_command": " ".join(submodule_command),
        "build_command": " ".join(command),
        "build_cwd": str(baseline_worktree),
        "git_commit": _git_output(baseline_worktree, "rev-parse", "HEAD"),
        "install_root": str(install_root),
        "nvte_framework": env["NVTE_FRAMEWORK"],
        "core_shared_objects": core_objects,
        "torch_shared_objects": torch_objects,
    }


def _run_benchmark_subprocess(
    command: List[str],
    cwd: Path,
    output_path: Path,
    pythonpath_entries: List[Path],
    expected_extension_root: Optional[Path] = None,
) -> Dict[str, object]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    pythonpath_parts = [str(path) for path in pythonpath_entries]
    if pythonpath:
        pythonpath_parts.append(pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    if expected_extension_root is not None:
        env[EXTENSION_ROOT_ENV] = str(expected_extension_root)
    else:
        env.pop(EXTENSION_ROOT_ENV, None)
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        print(completed.stdout, file=sys.stderr, end="")
        completed.check_returncode()
    with open(output_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _same_shape_comparisons(
    baseline_results: List[Dict[str, object]], candidate_results: List[Dict[str, object]]
) -> List[Dict[str, object]]:
    baseline_by_case = {
        _result_case_id(result): result
        for result in baseline_results
        if result["shape_case"] == "same-shape"
    }
    candidate_by_case = {
        _result_case_id(result): result
        for result in candidate_results
        if result["shape_case"] == "same-shape"
    }
    comparisons = []
    for case_id in sorted(set(baseline_by_case) & set(candidate_by_case)):
        baseline = baseline_by_case[case_id]
        candidate = candidate_by_case[case_id]
        baseline_bandwidth = float(baseline["bandwidth_TBps_actual_bytes"])
        candidate_bandwidth = float(candidate["bandwidth_TBps_actual_bytes"])
        candidate_to_baseline = (
            candidate_bandwidth / baseline_bandwidth if baseline_bandwidth > 0.0 else 0.0
        )
        target_tbps = _target_tbps(str(candidate["shape_case"]))
        comparisons.append(
            {
                "case_id": case_id,
                "shape_case": str(candidate["shape_case"]),
                "mode": str(candidate["mode"]),
                "baseline_result_id": str(baseline["result_id"]),
                "candidate_result_id": str(candidate["result_id"]),
                "baseline_ref": str(baseline["git_ref"]),
                "candidate_ref": str(candidate["git_ref"]),
                "baseline_bandwidth_TBps_actual_bytes": baseline_bandwidth,
                "candidate_bandwidth_TBps_actual_bytes": candidate_bandwidth,
                "candidate_to_baseline_ratio": candidate_to_baseline,
                "minimum_candidate_to_baseline_ratio": 0.95,
                "target_TBps_actual_bytes": target_tbps,
                "passes_baseline_ratio": candidate_to_baseline >= 0.95,
                "passes_absolute_target": candidate_bandwidth >= target_tbps,
            }
        )
    return comparisons


def benchmark_same_session(args: argparse.Namespace) -> Dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    candidate_ref = _git_output(repo_root, "rev-parse", "HEAD") or "candidate"
    baseline_ref = args.baseline_ref
    if args.shape_case not in ("all", "same-shape"):
        raise ValueError("--baseline-ref requires --shape-case all or same-shape")

    with tempfile.TemporaryDirectory(prefix="grouped_fp8_baseline_") as temp_dir:
        temp_path = Path(temp_dir)
        baseline_worktree = temp_path / "baseline"
        baseline_install_root = temp_path / "baseline_install"
        baseline_output = temp_path / "baseline.json"
        candidate_output = temp_path / "candidate.json"
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(baseline_worktree), baseline_ref],
            cwd=repo_root,
            check=True,
        )
        try:
            baseline_build = _install_baseline_package(
                baseline_worktree, baseline_install_root
            )
            baseline_report = _run_benchmark_subprocess(
                _benchmark_subprocess_command(
                    args,
                    repo_root / "benchmarks/benchmark_grouped_fp8_quantize.py",
                    baseline_output,
                    "same-shape",
                ),
                baseline_worktree,
                baseline_output,
                [baseline_install_root],
                baseline_install_root,
            )
            candidate_report = _run_benchmark_subprocess(
                _benchmark_subprocess_command(
                    args,
                    repo_root / "benchmarks/benchmark_grouped_fp8_quantize.py",
                    candidate_output,
                    args.shape_case,
                ),
                repo_root,
                candidate_output,
                [repo_root],
            )
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(baseline_worktree)],
                cwd=repo_root,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    baseline_results = _annotate_results(
        baseline_report["results"], "baseline", baseline_ref
    )
    candidate_results = _annotate_results(
        candidate_report["results"], "candidate", candidate_ref
    )
    modes = _selected_modes(args)
    baseline_case_ids = _expected_case_ids(("same-shape",), modes)
    candidate_case_ids = _expected_case_ids(_selected_shape_cases(args), modes)
    _require_case_ids(baseline_results, baseline_case_ids, "baseline_results")
    _require_case_ids(candidate_results, candidate_case_ids, "candidate_results")
    same_shape_comparisons = _same_shape_comparisons(baseline_results, candidate_results)
    _require_case_ids(
        same_shape_comparisons,
        baseline_case_ids,
        "same_shape_baseline_comparisons",
    )
    return {
        "schema_version": "benchmark_raw_report/v1",
        "benchmark": "grouped_fp8_tensor_scaling_quantize",
        "baseline_mode": "same_session",
        "baseline_ref": baseline_ref,
        "candidate_ref": candidate_ref,
        "profile_enabled": args.profile,
        "profile_after_warmup": True,
        "regular_iterations": args.iters,
        "profile_iterations": args.profile_iters,
        "baseline_build": baseline_build,
        "baseline_extension_metadata": baseline_report.get("extension_metadata", {}),
        "candidate_extension_metadata": candidate_report.get("extension_metadata", {}),
        "baseline_results": baseline_results,
        "candidate_results": candidate_results,
        "results": baseline_results + candidate_results,
        "measurements": _make_measurements(baseline_results + candidate_results),
        "same_shape_baseline_comparisons": same_shape_comparisons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all",) + MODES, default="all")
    parser.add_argument("--shape-case", choices=("all",) + SHAPE_CASES, default="all")
    parser.add_argument("--num-groups", type=int, default=8)
    parser.add_argument("--rows-per-group", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument("--num-buffers", type=int, default=8)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--profile-iters", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--baseline-ref",
        default="",
        help=(
            "Run a same-session baseline comparison by checking out this ref in a "
            "temporary worktree, benchmarking same-shape modes there, benchmarking the "
            "requested candidate cases in this worktree, and writing one combined JSON report."
        ),
    )
    parser.add_argument(
        "--output",
        default=_default_output_path(),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _check_output_path(args.output)
    if args.rows_per_group % 128 != 0:
        raise ValueError("--rows-per-group must be divisible by 128 for the grouped FP8 kernel")
    if args.hidden_size % 128 != 0:
        raise ValueError("--hidden-size must be divisible by 128 for the grouped FP8 kernel")
    if args.baseline_ref:
        report = benchmark_same_session(args)
        _write_report(report, args.output)
        return

    extension_metadata = _loaded_extension_metadata()
    modes = _selected_modes(args)
    shape_cases = _selected_shape_cases(args)
    results = [
        benchmark_case_mode(args, shape_case, mode)
        for shape_case in shape_cases
        for mode in modes
    ]
    results = _annotate_results(
        results, "candidate", _git_output(Path(__file__).resolve().parents[1], "rev-parse", "HEAD")
    )
    report = {
        "schema_version": "benchmark_raw_report/v1",
        "benchmark": "grouped_fp8_tensor_scaling_quantize",
        "profile_enabled": args.profile,
        "profile_after_warmup": True,
        "regular_iterations": args.iters,
        "profile_iterations": args.profile_iters,
        "extension_metadata": extension_metadata,
        "candidate_results": results,
        "results": results,
        "measurements": _make_measurements(results),
    }
    _write_report(report, args.output)


if __name__ == "__main__":
    main()
