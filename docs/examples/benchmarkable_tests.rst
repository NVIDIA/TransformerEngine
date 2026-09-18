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
       def setup(state):
           return make_inputs(shape, dtype)          # deterministic

       def evaluate(state):
           return te_implementation(state)           # the Transformer Engine path

       def reference(state):
           return naive_implementation(state)        # what it should agree with

       def verify(actual, expected):
           torch.testing.assert_close(actual, expected, **dtype_tols(dtype))

       return Case(setup=setup, evaluate=evaluate, reference=reference, verify=verify)

``setup`` receives the test's state and returns it; it is ``None`` for an ordinary
single-GPU test. ``setup`` must be deterministic, because benchmark mode calls it again for each
timed variant.
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

Running across multiple GPUs
----------------------------

A ``Case`` that sets ``num_gpus`` runs on that many ranks, and the harness launches them.
The test says only how many ranks it wants and what each one does, so what gets recorded is
the compute rather than the process startup and rendezvous in front of it.

Multi-rank Cases add three callables. ``dist_init`` receives this rank, the world size and
the coordinator address and port, builds the process group, and returns it as the state;
every other callable receives that state, and ``setup`` must keep the distributed part of it
intact when it adds the test data:

.. code-block:: python

   def test_row_parallel(hidden_size, dtype):
       def dist_init(rank, world, coordinator_addr, coordinator_port):
           torch.cuda.set_device(rank)
           dist.init_process_group(
               backend="nccl",
               init_method=f"tcp://{coordinator_addr}:{coordinator_port}",
               rank=rank,
               world_size=world,
               timeout=datetime.timedelta(seconds=120),
           )
           return {"pg": dist.group.WORLD, "rank": rank, "world": world}

       def dist_clean(state):
           dist.destroy_process_group()

       def barrier(state):
           dist.barrier(group=state["pg"])

       def setup(state):
           state["x"] = make_input(hidden_size, state["rank"], dtype)
           return state                      # keep what dist_init put there

       ...
       return Case(
           setup=setup, evaluate=evaluate, reference=reference, verify=verify,
           dist_init=dist_init, dist_clean=dist_clean, barrier=barrier,
           num_gpus=min(4, torch.cuda.device_count()),
       )

The coordinator is a TCP endpoint the harness picks per launch, on a port the OS assigns
and has just confirmed free, so nothing collides with the previous config or with a
concurrent pytest session. ``tcp://`` is what JAX's
``jax.distributed.initialize`` needs, since it cannot rendezvous through a file. A test that
prefers ``env://`` can ignore the arguments entirely: ``MASTER_ADDR``, ``MASTER_PORT``,
``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK`` and ``LOCAL_WORLD_SIZE`` describe the same launch,
so the rendezvous matches what the test would see under ``torchrun``. A test that prefers
``file://`` is free to choose its own path.

``dist_init`` and ``dist_clean`` run once per point, so the process group survives every
timed sample; ``setup`` and ``reset`` run per variant and touch only the test data.
``dist_clean`` runs whether the body succeeded or failed.

``barrier`` is required whenever ``num_gpus`` is greater than one, and must block the host
rather than only ordering the stream. The harness calls it before each timed sample and
outside the measured interval: without it, a rank arriving late has its wait recorded as
this operation's cost on every rank that arrived on time.

``num_gpus`` comes from the test, which is what lets the harness stay framework-agnostic.
Guard the test so it skips when the box has too few GPUs. ``timeout`` (default 1800s)
budgets the whole launch, including interpreter startup and rendezvous on every rank.

Every rank records its own timings, and the report keeps them separate: ``world_size`` is
part of a record's identity, so a four-rank run never compares against an eight-rank one,
while ``rank`` distinguishes records within a launch so load imbalance stays visible.
``--nvte-benchmark-min-run-time`` cannot be combined with a multi-rank Case, because ranks
would leave the sampling loop after different numbers of iterations.

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
