---
name: transformer-engine-debug-attention
description: Explain Transformer Engine attention backend selection, diagnose wrong outputs or gradients and distributed stalls, or verify an attention workaround or fix on a target build.
---

# Debug Transformer Engine Attention module

## 1. Find the first failing call

Start with the question the user needs answered. For an explanation, use the supplied logs or code and identify any missing fact that limits the answer. For a diagnosis, use a permitted run when a comparison would distinguish plausible causes. For verification, identify the target build, the required backend or acceptable fallback, and the behavior that must pass.

Locate the earliest failing phase in the available evidence. If running a comparison, keep the original input or test available. A training failure may require the real backward pass to reproduce.

If build or device identity matters, use the recorded values. When they are missing and execution is in scope, run this PyTorch probe in the failing worker's environment with the same device selection. It identifies the imported TE package, PyTorch build, and GPU capability. On releases exposing the cuDNN helper, it also reports the version TE sees.

```python
import torch
import transformer_engine as te
import transformer_engine.pytorch as te_torch

print(te.__version__, te.__file__)
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())
if hasattr(te_torch, "get_cudnn_version"):
    print("cuDNN reported by TE:", te_torch.get_cudnn_version())
```

Read the selector output for the affected call. If it is missing and a run is needed, enable attention debug logging on a PyTorch release that supports it. Substitute the user's reproducer for `repro.py`, retaining its arguments and launcher. Level 2 shows the configuration and selection reasoning.

```sh
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 python repro.py
```

Record the selected backend or the reason TE excluded the expected one. Package versions identify possible implementations. Selection logs tell you which one TE chose for these inputs.

## 2. Choose one comparison

Use the row matching the earliest failure. If a comparison is needed, keep the failing case as one arm, change one factor that gives the competing explanations different predictions, and check which backend each arm selected.

### Route by the first failing stage

| Stage | What you might see | Next check |
| --- | --- | --- |
| Package or extension load | Import error, missing shared library, undefined symbol | Run the imports in order in the failing worker. Record the first failing import and its underlying loader error. If the package resolves but the PyTorch extension fails to load, check the extension and its dependencies. |
| Framework wrapper or input preparation | Shape, dtype, device, mask, or sequence-offset error | Check whether the call reaches `DotProductAttention` with the intended arguments. Compare the effective mask, logical and padded offsets, maximum lengths, and per-rank shapes with the caller's values. |
| Backend selector | `NoBackend` or an unexpected fallback | Identify the rejected condition on the affected rank and phase. Check whether it excluded only the preferred backend or every candidate. On fallback, record which backend TE actually selected. |
| Backend plan or kernel launch | cuDNN support error after selection, invalid launch, or missing device code | After selection, determine whether the backend failed to build a plan or failed to execute one. For a plan error, compare the loaded runtime at fixed geometry. For missing device code, check the build's GPU targets. |
| Forward result | NaNs, all-zero outputs, or incorrect padding or masking | Check that Q/K/V are finite at entry, then compare valid output rows with an independent reference. If those match, inspect pad rows consumed by later operations and find the first operation that propagates the discrepancy. |
| Backward result | Forward passes, but gradients are absent, non-finite, or wrong | With the same upstream gradient, distinguish an absent gradient from a returned tensor that is zero, non-finite, or numerically different. Check the specific input and the forward state its backward requires. |
| Compilation, capture, or replay | Eager passes, compiled execution fails, or later replays diverge | Check whether compilation or capture completes. If it does, compare the first and later replays separately with eager results for the corresponding inputs. |
| Distributed execution | Watchdog, stalled peer, or a failure only with context parallelism | Follow a waiting rank to its peer. Identify the earliest local error or missing operation, then compare the failing rank's sequence metadata and phase with a rank that completed it. |

For an out-of-memory error, identify the allocation phase and rank first. Verify dispatch in that run. Changing backends can introduce a larger allocation and confound the comparison.

If a flag or eligibility rule is unclear, consult the attention guide or environment-variable reference for the installed TE release. If the logs still leave the selection unexplained, follow `DotProductAttention` into its selector in the revision that ran. For a failure specific to context parallelism, inspect the per-rank arguments at the selected adapter. From the root of a TE source checkout, this search locates the selector and its diagnostic.

```sh
rg -n 'get_attention_backend|NoBackend' transformer_engine/pytorch/attention
```

### Run the comparison for that stage

**Unexpected selection or failed plan.** Start with the selector's stated reason. Check the effective option at the call, the required forward or backward phase, and the particular constraint it names. For context parallelism, calculate the sequence shape on the affected rank. If TE selected the expected backend and plan construction failed, compare the dependency actually loaded in that worker with a working build on the same input.

For PyTorch releases that support this switch, the following control removes cuDNN FusedAttention from consideration. Use it when that family was selected in the failing run. Inspect dispatch again: the remaining candidates may also be ineligible, and test harnesses can reset backend flags.

```sh
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FUSED_ATTN=0 python repro.py
```

If the failing run selected FlashAttention 4, use `NVTE_FLASH_ATTN_V4=0` where supported to exclude that generation and check which candidate handles the same call.

**Wrong outputs or gradients.** Compare attention outputs with an independent reference using the same causal or window mask, valid tokens, and sequence boundaries. For a Boolean attention mask, TE treats `True` as excluded and PyTorch functional scaled dot product attention (SDPA) treats `True` as visible. Check the visible keys for one query at the mask boundary. For a small SDPA reference, run the reference call inside `with sdpa_kernel(SDPBackend.MATH):` from `torch.nn.attention` where available to select the math backend explicitly.

For packed batches, retain the logical and padded offsets if they differ. Check padding rows that later operations read. If forward agrees, pass the same incoming output gradient to both computations and compare query, key, and value gradients individually. When random inputs or single-sequence replay pass, restore the smallest failing real batch. Restore graph mode or preceding operations if replaying the captured tensors alone still passes.

**Stall or watchdog.** Find the first rank-local error or last completed phase and identify what its peers were waiting for. Follow an earlier selector or computation failure first. Otherwise, compare a different eligible backend on the original topology, or test whether the same backend stalls locally on a smaller topology. If reduction removes the symptom, return to the failing rank layout.

## 3. Resolve an inconclusive comparison

### Does a direct attention call reproduce the wrapper failure?

Capture inputs at the wrapper's `DotProductAttention` call and replay that call with the same mode, mask, and backend selection. Compare its output before subsequent model operations. If the direct call passes, inspect Q, K, and V at both call sites before copying or reshaping them. In PyTorch, print the shape, strides, and offset of each tensor.

```python
for x in (q, k, v):
    print(x.shape, x.stride(), x.storage_offset())
```

Equal shapes can hide different memory layouts. Preserve views into shared storage, the caller's mask, and its packed sequence offsets in the replay. If the mismatch now reproduces at the attention boundary, use that smaller call for the next comparison.

### Did backward produce a gradient for each input?

For an eager PyTorch probe, run a disposable forward with gradient tracking enabled because the gradient call can free its backward graph. Use the exact gradient-enabled tensors passed into that forward and the same incoming `dout` as the failing case.

```python
dq, dk, dv = torch.autograd.grad(
    out, (q, k, v), grad_outputs=dout,
    allow_unused=True, materialize_grads=False,
)
```

Inspect the returned gradients. A `None` identifies an input for which autograd returned no gradient. Trace that input's connection to the output and the backward function that should supply it. For a returned zero or non-finite tensor, compare the same input gradient with the independent reference.

### Does the result change across replays?

For a PyTorch CUDA Graph, choose two same-shaped inputs with distinguishable eager outputs. Copy each input into the capture's existing input tensors, replay, and save a copy of the output before the next replay overwrites it. Compare each saved output with its corresponding eager result. If the second output still matches the first eager result, check that the caller updated the captured buffers. Keep the failing replay sequence when testing a correction.

### Other targeted checks

**If the result changes between the first and later FusedAttention calls,** compare the same call in a fresh process and on repetition. On TE releases that support it, `NVTE_FUSED_ATTN_CACHE_DEBUG=2` traces cuDNN graph builds, cache hits or misses, and executions. Append `:N` to trace rank N in a distributed run. Match the first differing event to the affected call. A cache hit establishes reuse. Check the result of that execution separately.

**If dispatch is known but the kernel is still in question,** profile just the affected forward or backward call. Check which GPU kernel actually launched before attributing a difference to a particular implementation or precision.

**If a CUDA error surfaces at a later call,** try a separate reduced eager run with synchronous launches. This can move the error report closer to the failing operation.

```sh
CUDA_LAUNCH_BLOCKING=1 python repro.py
```

The setting changes scheduling and timing. Use the result to localize the error, then rerun with the original execution settings for verification.

## 4. Verify the requested outcome

For an **explanation or diagnosis**, point to the selector reason, first mismatch, rank-local error, or paired result that supports the conclusion. If two causes still fit, name the observation that would distinguish them. Stop when the evidence answers the user's question.

For **verification**, report the case, build, observed backend, affected phase, and result. A passing fallback verifies a workaround on that route. To claim recovery of the original path, confirm the formerly failing case executed on the intended backend and met its numerical or completion criterion. If the target case skipped or only forward ran, state which requested check remains open.
