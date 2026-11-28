---
name: review
description: >-
  Review a Transformer Engine change (local diff, branch, or pull request) for reuse of existing
  infrastructure, layering, CPU overhead and host-device syncs, input validation, test economy, and
  documentation. Use when asked to review TE code or prepare a PR for review.
license: Apache-2.0
---

# Review a Transformer Engine change

Review the diff against `AGENTS.md` and the `develop` skill. Read only the `develop` references for
what the diff touches. Get the diff with `git diff <base>...HEAD` or `gh pr diff <number>`. Report
findings only: do not edit code or post PR comments unless asked.

## Process

1. **Triage.** Read the diff once, classify it (common kernel, binding, PyTorch, JAX, tests, docs),
   and apply only the checks that fit. List each skipped area in the review header with a one-line
   reason; a bare label is not a reason. If a later check hits a skipped area, review it.
2. **Review.** Read the changed code with its callers, consumers, and closest sibling. List what
   the sibling or replaced path did that the new code drops (a check, a cleanup, a default). Check
   both sides of each contract: C API vs binding, forward vs backward, PyTorch vs JAX, docs vs code.
3. **Refute.** Try to disprove each finding from the code (a guard above the line, a caller that
   never passes that input, a sibling that already handles it) and drop it if disproved. Ask the
   author only about intent or a measurement the repository cannot answer.

## Checks

Flag every violation of the rules in the `develop` skill and the references for the touched area
(`pytorch.md`, `jax.md`, `tests.md`), and of the architecture rules in `AGENTS.md`. Review-specific
points:

- **Reuse.** Search the repository before accepting "nothing fits"; a parallel mechanism or copied
  block is a finding even when it works.
- **Tests.** For each new test finish: "This fails if someone later changes ____, and that hurts
  users because ____." A blank of "the code I just wrote" is a run-once test. Name the surviving
  test before suggesting a delete, and check the test can fail (see `tests.md`).
- **Docs and API.** Public API backward compatible; new `NVTE_*` variables in `docs/envvars.rst`.

## Evidence

- Cite `file:line` for the changed line and for the other side, plus a concrete trigger (shape,
  dtype, layout, ranks) that fails. Verify API, version, and architecture claims by grep or
  reading, never from memory.
- A cost claim needs evidence: a count of ops, copies, or syncs on the path, a benchmark, or a
  profile or `ptxas -v` for a kernel. Say when you could not measure.

## Stay quiet about

Lines outside the diff unless the diff makes them wrong; pre-existing issues the change does not
touch or worsen; what pre-commit or the L0 lint scripts enforce; inputs no caller can produce;
"sibling code does the same" as an excuse for a missing safety check. Do not hunt for findings:
"no findings" is a valid result. Exception: duplicated logic or a hand-rolled helper may point at
related existing code the change should reuse.

## Report

One entry per finding, most severe first, then questions and one line on what was checked and
found clean:

    [severity] file:line - rule
    Claim: one testable sentence.
    Evidence: the changed line and the other side, with file:line.
    Trigger: the input that fails, and how (optional for naming and style).
    Suggest: the smallest fix, naming the code to reuse.

Severity: `blocker` (crash, hang, wrong result, ABI or API break, a new host sync with no recorded
code-owner confirmation), `major` (regression, contract mismatch, parallel mechanism, silent
fallback), `minor` (naming, comments, docs, test economy).

On a follow-up review, read earlier findings and replies first: report only new issues or "fix
verified" / "fix incomplete", and verify an author's reason once before withdrawing a finding.
