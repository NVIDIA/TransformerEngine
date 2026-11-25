# Evaluate the HF-to-TE conversion workflow

The [cases](evals.json) exercise different stopping points in the [workflow](../SKILL.md).
Grade the written answers and submitted code by inspection. GPU execution is outside this suite.

| Case | Requested work | Evidence to inspect |
|---|---|---|
| 001 · Qwen3.5 | Steps 1–3, written plan | Target feasibility, chosen boundary and checkpoint map for one expert, grounded in the pinned public metadata. |
| 002 · vLLM | Outside conversion scope | An answer to the serving question. |
| 003 · ESM-2 | Workflow referral | The BioNeMo Recipes acceleration skill and why it applies. |
| 004 · DeepSeek-V3 | Steps 1–4, forward test source | Two-head projection map, callable and test code for the BF16 projected contribution. |
| 006 · Mamba | Partial conversion and training test source | Trainable adapter and BF16 forward/backward test code covering recurrence, causality and mapped gradients. |

## Paired evaluation

From the repository root, run `skillevaluator evaluate skills/convert-hf-to-te` on a supported runner.
Give both arms the same prompts and access to the pinned public sources. Verify that the agent,
model and effective time budget match. Grade each answer and its complete submitted files against
the same case assertions. No case uses bundled fixtures.
