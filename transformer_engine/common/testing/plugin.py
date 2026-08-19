# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Pytest plugin driving benchmarkable tests."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
import sys
from typing import Any
import warnings

import pytest

from .artifacts import write_run_artifacts
from .case import Case, CaseSkip
from .declaration import declared_axes, normalize_argnames, set_plugin_active
from .decorator import (
    BENCHMARK_MARKER,
    CASE_MARKER,
    MARKERS,
    MODE_BENCHMARK,
    MODE_CORRECTNESS,
    SUPPRESS_MARKER,
)
from .device import build_architectures, cuda_available, device_architecture
from .runner import _run_benchmark_point, _run_correctness
from .timing import TIMING_METHOD


def pytest_addoption(parser):
    """Register the benchmarkable command-line options, once per parser."""
    # Keyed to the parser, not to a module-level flag: pytest builds a fresh parser per
    # pytest.main() call, and a sticky flag would hide --nvte-benchmark from a second run.
    if getattr(parser, "_te_benchmarkable_registered", False):
        return
    parser._te_benchmarkable_registered = True  # pylint: disable=protected-access

    group = parser.getgroup("benchmarkable")
    group.addoption(
        "--nvte-benchmark",
        action="store_true",
        default=False,
        help=(
            "Run the benchmark matrix only, gated by a one-time correctness check at "
            "each benchmark point."
        ),
    )
    group.addoption(
        "--nvte-benchmark-report-dir",
        default=None,
        help="Directory for benchmark_report/v1 artifacts.",
    )
    group.addoption("--nvte-benchmark-warmup", type=int, default=5)
    group.addoption("--nvte-benchmark-iterations", type=int, default=20)
    group.addoption("--nvte-benchmark-inner-iterations", type=int, default=1)
    group.addoption("--nvte-benchmark-min-run-time", type=float, default=0.0)
    group.addoption(
        "--nvte-benchmark-no-reference",
        action="store_true",
        default=False,
        help="Skip timing the reference variant.",
    )


def _resolve_mode(config) -> str:
    """Return ``MODE_BENCHMARK`` if ``--nvte-benchmark`` was passed, else ``MODE_CORRECTNESS``."""
    return MODE_BENCHMARK if config.getoption("--nvte-benchmark") else MODE_CORRECTNESS


def pytest_configure(config):
    """Resolve the mode before test modules are imported, and register markers."""
    # Ahead of the double-fire guard, so a second registered copy of this hook still
    # sets the flag; Case.__post_init__ reads it to detect a plugin-less session.
    set_plugin_active(True)

    if getattr(config, "_benchmarkable_configured", False):
        return
    config._benchmarkable_configured = True  # pylint: disable=protected-access

    for name, description in MARKERS:
        config.addinivalue_line("markers", f"{name}: {description}")

    if _resolve_mode(config) == MODE_CORRECTNESS:
        return

    if config.getoption("--nvte-benchmark-inner-iterations") < 1:
        raise pytest.UsageError("--nvte-benchmark-inner-iterations must be at least 1.")
    if config.getoption("--nvte-benchmark-iterations") < 1:
        raise pytest.UsageError("--nvte-benchmark-iterations must be at least 1.")


def _is_case_bearing(node) -> bool:
    """Whether the test returns a Case, so this plugin runs it instead of pytest."""
    # get_closest_marker, not node.keywords: keywords also carry every ancestor node's
    # bare name, so an unmarked test under a path named nvte_case would match here.
    return node.get_closest_marker(CASE_MARKER) is not None


def _is_benchmark_eligible(node) -> bool:
    """Whether the test declared benchmark axes and nothing suppressed them."""
    # Suppression is a union over the test, its class and its module, and nothing
    # re-enables: skip and skipif only ever subtract benchmarking.
    if node.get_closest_marker(BENCHMARK_MARKER) is None:
        return False
    return next(node.iter_markers(name=SUPPRESS_MARKER), None) is None


def _substitutions(config) -> dict:
    """Per-session record of which test definitions had a benchmark axis substituted."""
    registry = getattr(config, "_benchmarkable_substitutions", None)
    if registry is None:
        registry = {}
        config._benchmarkable_substitutions = registry  # pylint: disable=protected-access
    return registry


def _definition_key(node) -> tuple:
    """Key one test definition by the collector and name that ``pytest_generate_tests``
    and every item it generates share."""
    # Not the function object: a method inherited by two classes is one function but two
    # definitions, and one class substituting an axis says nothing about the other.
    return (node.parent, getattr(node, "originalname", None) or node.name)


# trylast is required: pluggy's LIFO ordering would otherwise run this hook before
# pytest's own -k/-m deselection, and the guard must see the final item list.
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    """Narrow the session to benchmark points, then run the post-collection guard."""
    if _resolve_mode(config) == MODE_CORRECTNESS:
        return

    substituted = _substitutions(config)
    kept, dropped, suppressed, unmatched = [], [], 0, 0
    for item in items:
        eligible = _is_benchmark_eligible(item)
        if eligible and substituted.get(_definition_key(item), False):
            kept.append(item)
            continue
        dropped.append(item)
        if eligible:
            unmatched += 1
        elif _is_case_bearing(item):
            suppressed += 1
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = kept

    _guard_benchmark_collection(config, kept, suppressed, unmatched)


def _guard_benchmark_collection(config, kept, suppressed, unmatched) -> None:
    """Fail loudly when a requested benchmark mode has nothing valid to time."""
    if not kept:
        raise pytest.UsageError(_zero_points_message(suppressed, unmatched))

    if config.option.collectonly:
        return

    # Without cuda-python, device.synchronize() is a silent no-op: every measurement
    # would be host launch time only and the report would still claim a device ran.
    if not cuda_available():
        raise pytest.UsageError(
            "--nvte-benchmark requires a CUDA device reachable through cuda-python; none is "
            "available. Install cuda-python (the 'test' extra) or drop --nvte-benchmark."
        )

    _guard_build_architecture()


def _zero_points_message(suppressed, unmatched) -> str:
    """Explain a zero-point benchmark session without blaming the wrong thing."""
    if suppressed:
        detail = (
            f"all {suppressed} Case-bearing test(s) here carry @benchmark.skip or a true"
            " @benchmark.skipif."
        )
    elif unmatched:
        detail = (
            f"{unmatched} benchmark-eligible test(s) here parametrize none of the axes"
            " declared on their class."
        )
    else:
        detail = (
            "check the -k/-m expression, and that a test under the selected paths declares"
            " axes with @benchmark(argnames, values)."
        )
    return f"--nvte-benchmark collected zero benchmark points: {detail}"


def _guard_build_architecture() -> None:
    """Fail loudly when the loaded library's embedded SASS cannot run on this device."""
    # Only a proven mismatch may raise: an unavailable build architecture (no
    # cuobjdump, or the library was not located) or an unknown device architecture
    # leaves the comparison undecidable and must not block the run.
    build = build_architectures()
    if not build.get("available"):
        return
    device_arch = device_architecture()
    if device_arch is None:
        return

    embedded = build["cuda_architectures"]
    if device_arch.rstrip("af") in {arch.rstrip("af") for arch in embedded}:
        return

    raise pytest.UsageError(
        f"{build['library']} was built for "
        f"{', '.join(embedded) or '(no architectures embedded)'}, but this device is "
        f"{device_arch}, so no kernel can launch. Rebuild with an NVTE_CUDA_ARCHS "
        "matching this device (use 100 for sm_100a/sm_103a)."
    )


def benchmark_settings(config) -> dict[str, Any]:
    """Return the resolved benchmark settings for this session."""
    report_dir = config.getoption("--nvte-benchmark-report-dir")
    return {
        "mode": _resolve_mode(config),
        "warmup": config.getoption("--nvte-benchmark-warmup"),
        "iterations": config.getoption("--nvte-benchmark-iterations"),
        "inner_iterations": config.getoption("--nvte-benchmark-inner-iterations"),
        "min_run_time": config.getoption("--nvte-benchmark-min-run-time"),
        "timing_method": TIMING_METHOD,
        "report_dir": Path(report_dir) if report_dir else None,
        "no_reference": config.getoption("--nvte-benchmark-no-reference"),
    }


@pytest.hookimpl(hookwrapper=True)
def pytest_generate_tests(metafunc):
    """Swap benchmark values into a test's existing parametrize marks.

    Substituting in place preserves the parametrization's positional structure, so ``pytest.param``
    ids and coupled argnames keep working. Class-level declarations are picked up via
    ``metafunc.cls``; function-level entries win. Mutates ``own_markers``, not public pytest API
    (pytest is pinned to 8.2.1).
    """
    saved = []
    # Eligibility is checked before anything else: a suppressed or undeclared sibling
    # method must not inherit its class's declarations, nor be failed by the check below.
    eligible = _is_benchmark_eligible(metafunc.definition)
    try:
        if eligible and _resolve_mode(metafunc.config) == MODE_BENCHMARK:
            own = declared_axes(metafunc.function)
            declarations = dict(own)
            if metafunc.cls is not None:
                for key, values in declared_axes(metafunc.cls).items():
                    declarations.setdefault(key, values)
            node = metafunc.definition
            while node is not None and declarations:
                markers = getattr(node, "own_markers", None)
                if markers:
                    for index, mark in enumerate(markers):
                        if mark.name != "parametrize":
                            continue
                        key = normalize_argnames(mark.args[0])
                        if key in declarations:
                            saved.append((markers, index, mark))
                            markers[index] = pytest.mark.parametrize(
                                mark.args[0], declarations[key]
                            ).mark
                node = getattr(node, "parent", None)
            _check_own_declarations_matched(metafunc, own, saved)
            _substitutions(metafunc.config)[_definition_key(metafunc.definition)] = bool(saved)
        yield
    finally:
        # Class- and module-level marks are shared, so a substituted mark left in place
        # corrupts every sibling test for the rest of the session. The finally must
        # therefore span the validation calls and the yield, not just the swap loop.
        for markers, index, original in saved:
            markers[index] = original


def _check_own_declarations_matched(metafunc, own, saved) -> None:
    """Fail loudly when an axis declared on this test itself matches no parametrize mark:
    a typo would otherwise silently benchmark the correctness values. An axis inherited from
    a class is best-effort, since a sibling method need not parametrize it."""
    matched = {normalize_argnames(mark.args[0]) for _, _, mark in saved}
    missing = sorted(set(own) - matched)
    if not missing:
        return
    raise pytest.UsageError(
        f"{metafunc.definition.nodeid}: @benchmark declared axes {missing}, which match no "
        "pytest.mark.parametrize on this test, its class, or its module. Check the "
        'spelling, and declare a coupled group exactly as parametrized ("m,n,k").'
    )


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    """Run a Case-bearing test's Case according to the active mode."""
    if not _is_case_bearing(pyfuncitem):
        return None
    # An async test cannot return a Case, and leaving it to pytest keeps pytest's own
    # warn-and-skip for it byte-identical.
    if inspect.iscoroutinefunction(pyfuncitem.obj) or inspect.isasyncgenfunction(pyfuncitem.obj):
        return None

    argnames = pyfuncitem._fixtureinfo.argnames  # pylint: disable=protected-access
    kwargs = {name: pyfuncitem.funcargs[name] for name in argnames}
    case = pyfuncitem.obj(**kwargs)
    if not isinstance(case, Case):
        return _dispose_of_non_case(pyfuncitem, case)

    # CaseSkip from setup() is a coverage skip (unavailable backend or arch), not a
    # failure.
    if _resolve_mode(pyfuncitem.config) == MODE_CORRECTNESS:
        try:
            _run_correctness(case)
        except CaseSkip as exc:
            pytest.skip(str(exc))
        return True

    settings = benchmark_settings(pyfuncitem.config)
    try:
        records = _run_benchmark_point(case, settings, pyfuncitem)
    except CaseSkip as exc:
        pytest.skip(str(exc))
    store = getattr(pyfuncitem.config, "_benchmarkable_records", None)
    if store is None:
        store = []
        pyfuncitem.config._benchmarkable_records = store  # pylint: disable=protected-access
    store.extend(records)
    return True


def _dispose_of_non_case(pyfuncitem, result):
    """Handle a Case-bearing test that returned no Case, exactly as pytest would."""
    if declared_axes(pyfuncitem.function):
        raise TypeError(
            f"{pyfuncitem.nodeid}: @benchmark declares axes on this test, so it must return "
            f"a Case; got {type(result).__name__}. A test that is Case-bearing but never "
            "benchmarked is written @benchmark.skip."
        )
    if _resolve_mode(pyfuncitem.config) == MODE_BENCHMARK:
        pytest.skip("returns no Case, so it is not a benchmark point.")
    # Mirrors _pytest/python.py::pytest_pyfunc_call, which owns this test in every other
    # respect: an inherited declaration is a blanket statement, not a per-method claim.
    if result is not None:
        warnings.warn(
            pytest.PytestReturnNotNoneWarning(
                f"Expected None, but {pyfuncitem.nodeid} returned {result!r}, which will be "
                "an error in a future version of pytest.  Did you mean to use `assert` "
                "instead of `return`?"
            )
        )
    return True


def pytest_sessionfinish(session, exitstatus):  # pylint: disable=unused-argument
    """Write benchmark artifacts once the session completes. Guarded against a second
    registered copy of the plugin, since this hook is not ``firstresult``."""
    config = session.config
    if getattr(config, "_benchmarkable_report_written", False):
        return
    config._benchmarkable_report_written = True  # pylint: disable=protected-access
    records = getattr(config, "_benchmarkable_records", None)
    if not records:
        return
    settings = benchmark_settings(config)
    if settings["report_dir"] is None:
        print(
            f"\nWarning: {len(records)} benchmark record(s) were collected but "
            "--nvte-benchmark-report-dir was not set, so they were discarded. Pass "
            "--nvte-benchmark-report-dir to write a benchmark_report/v1 artifact.",
            file=sys.stderr,
        )
        return

    selection = {
        "mode": settings["mode"],
        "warmup": settings["warmup"],
        "iterations": settings["iterations"],
        "inner_iterations": settings["inner_iterations"],
        "min_run_time": settings["min_run_time"],
        "timing_method": settings["timing_method"],
        "include_reference": not settings["no_reference"],
        "args": list(config.invocation_params.args),
    }
    # Only argv[0]'s basename is kept: the full launcher path leaks the invoking user's
    # home directory into every persisted report, and selection["args"] has the rest.
    command = [os.path.basename(sys.argv[0])] + list(sys.argv[1:])
    paths = write_run_artifacts(
        settings["report_dir"],
        records,
        command,
        selection,
    )
    print(f"\nWrote benchmark report: {paths['report']}")
