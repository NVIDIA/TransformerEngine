..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Benchmarkable Tests
===================

A benchmarkable test is an ordinary pytest test that returns a ``Case`` instead of asserting, so
one definition of setup, evaluation, reference and verification serves both correctness testing
and benchmarking. Running pytest normally checks correctness; adding ``--nvte-benchmark`` times
the same code instead.

Writing the test
----------------

Build a ``Case`` from four callables and return it:

.. code-block:: python

   from transformer_engine.common.testing import Case, benchmark

   def test_something(shape, dtype):
       def setup():
           return make_inputs(shape, dtype)          # deterministic

       def evaluate(state):
           return te_implementation(state)           # the Transformer Engine path

       def reference(state):
           return naive_implementation(state)        # what it should agree with

       def verify(actual, expected):
           torch.testing.assert_close(actual, expected, **dtype_tols(dtype))

       return Case(setup=setup, evaluate=evaluate, reference=reference, verify=verify)

``setup`` must be deterministic, because benchmark mode calls it again for each timed variant.
``verify`` is required whenever ``reference`` is set; there is no default comparator, so build one
on ``tests/pytorch/utils.py::dtype_tols`` or ``tests/jax/utils.py::assert_allclose``. Raise
``CaseSkip`` from ``setup`` when a backend or architecture is unavailable and the test is skipped.

Optional fields: ``reset(state)`` runs between timed samples for cases that mutate their state,
``time_reference=False`` records only the Transformer Engine path, and ``bytes_moved`` / ``flops``
add ``bandwidth_GBps`` and ``tflops`` to the recorded numbers.

Marking it for benchmarking
---------------------------

``@benchmark(argnames, values)`` gives an axis the values it should take when benchmarking. It
does not create an axis: the values replace those of an existing ``pytest.mark.parametrize`` with
the same argnames, so correctness parametrization is untouched. Coupled argnames are written
exactly as parametrized (``"m,n,k"``).

Abridged from ``tests/pytorch/test_fused_rope.py``:

.. code-block:: python

   @benchmark("dtype", [torch.bfloat16])
   @benchmark("seq_length", [8192])
   @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
   @pytest.mark.parametrize("seq_length", [2048, 4096])
   def test_fused_rope(dtype, seq_length):
       ...
       return Case(setup=setup, evaluate=evaluate, reference=reference, verify=verify)

An axis you do not declare keeps its full correctness values, so declare enough of them to keep
the benchmark matrix small -- a single benchmark shape usually means pinning most axes to one
value each.

``@benchmark`` also applies to a class, where it covers every test method:

.. code-block:: python

   @benchmark("b,s_q,s_kv,h", [(8, 2048, 2048, 16)])
   @pytest.mark.parametrize("b, s_q, s_kv, h", [...])
   class TestSoftmaxPrimitives:
       @staticmethod
       def test_forward(b, s_q, s_kv, h, dtype):
           ...
           return Case(setup=setup, evaluate=evaluate, reference=reference, verify=verify)

       @staticmethod
       @benchmark.skip(reason="returns no Case")
       def test_backward(b, s_q, s_kv, h, dtype):
           ...

Use ``@benchmark.skip`` or ``@benchmark.skipif(condition)`` for a test that returns a ``Case`` but
should not be benchmarked -- a correctness-only test written in this style, or one whose benchmark
you are temporarily disabling.

Running benchmarks
------------------

.. code-block:: shell

   python3 -m pytest tests/pytorch/test_fused_rope.py --nvte-benchmark \
       --nvte-benchmark-report-dir /tmp/te-bench

``--nvte-benchmark`` selects benchmark mode and deselects everything else. Each point is checked
for correctness once before it is timed, so a benchmark run also verifies the shapes it measures.

Options, with defaults:

* ``--nvte-benchmark-iterations`` (20) -- minimum timed samples per variant.
* ``--nvte-benchmark-warmup`` (5) -- untimed calls before sampling.
* ``--nvte-benchmark-inner-iterations`` (1) -- calls per timed sample. Raise it for kernels short
  enough that host launch latency dominates.
* ``--nvte-benchmark-min-run-time`` (0.0) -- keep sampling until this many seconds have elapsed.
* ``--nvte-benchmark-no-reference`` (off) -- skip timing the reference variant.
* ``--nvte-benchmark-report-dir`` (unset) -- where to write the JSON, JSONL and CSV reports.
  Without it, the collected numbers are discarded with a warning.
