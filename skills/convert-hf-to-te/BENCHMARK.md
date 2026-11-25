# Skill Benchmark: convert-hf-to-te

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `convert-hf-to-te`
- Validation status: `passed`
- Evaluation date: 2026-11-25
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 5 evaluation tasks (4 positive, 1 negative)
- Dataset digest: `sha256:ccb38af4f6811591262f8f59e921b0fd34c9376acc585696029b28aa0bbcceb0` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| convert-hf-to-te | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| convert-hf-to-te | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| convert-hf-to-te | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| convert-hf-to-te | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| convert-hf-to-te | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `convert-hf-to-te`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 81.4% — baseline ran, but no comparable score was available; uplift unavailable | 84.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 80.0% → 80.0% (±0.0 points) | 60.0% → 70.0% (+10.0 points) |
| Correctness | 100.0% → 88.0% (-12.0 points) | 100.0% → 100.0% (±0.0 points) |
| Discoverability | 81.5% — baseline ran, but no comparable score was available; uplift unavailable | 87.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 96.0% → 75.3% (-20.7 points) | 84.5% → 100.0% (+15.5 points) |
| Efficiency | 82.1% — baseline ran, but no comparable score was available; uplift unavailable | 65.8% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 88.9% → 88.8% (-0.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 75.8% → 80.9% (+5.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,044,896 | 1,751,656 | +293,240 | +16.74% | skill 5/5; base 5/5 |
| claude-code | convert-hf-to-te-001 | 458,405 | 565,868 | -107,463 | -18.99% | skill 1/1; base 1/1 |
| claude-code | convert-hf-to-te-002 | 36,548 | 35,911 | +637 | +1.77% | skill 1/1; base 1/1 |
| claude-code | convert-hf-to-te-003 | 71,422 | 106,498 | -35,076 | -32.94% | skill 1/1; base 1/1 |
| claude-code | convert-hf-to-te-004 | 1,185,787 | 833,768 | +352,019 | +42.22% | skill 1/1; base 1/1 |
| claude-code | convert-hf-to-te-006 | 292,734 | 209,611 | +83,123 | +39.66% | skill 1/1; base 1/1 |
| codex | All cases | 1,460,608 | 1,719,756 | -259,148 | -15.07% | skill 5/5; base 5/5 |
| codex | convert-hf-to-te-001 | 333,040 | 399,003 | -65,963 | -16.53% | skill 1/1; base 1/1 |
| codex | convert-hf-to-te-002 | 26,656 | 26,899 | -243 | -0.90% | skill 1/1; base 1/1 |
| codex | convert-hf-to-te-003 | 46,603 | 43,434 | +3,169 | +7.30% | skill 1/1; base 1/1 |
| codex | convert-hf-to-te-004 | 555,542 | 736,110 | -180,568 | -24.53% | skill 1/1; base 1/1 |
| codex | convert-hf-to-te-006 | 498,767 | 514,310 | -15,543 | -3.02% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 3,505,504 | 3,471,412 | +34,092 | +0.98% | skill 10/10; base 10/10 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 10 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 5 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/convert-hf-to-te/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/convert-hf-to-te/SKILL.md`)
- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/bionemo-recipes-acceleration/SKILL.md (`SKILL.md:204`)
- **LOW** QUALITY/quality_correctness: No examples provided (`skills/convert-hf-to-te/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Description very long (245 chars, recommend 50-150) (`skills/convert-hf-to-te/SKILL.md`)
- 5 additional finding(s) are available in the full evaluation artifacts.

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
