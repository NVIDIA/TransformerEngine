# Skill Benchmark: convert-hf-to-te

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `convert-hf-to-te`
- Validation status: `passed`
- Evaluation date: 2026-11-26
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 6 evaluation tasks (5 positive, 1 negative)
- Dataset digest: `sha256:b956789e60d2f41f23acc718ddcf99f42a96405ca2a6a86f54ac6ab0efae1aa3` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 84.0% — baseline ran, but no comparable score was available; uplift unavailable | 77.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 75.0% → 75.0% (±0.0 points) | 66.7% → 58.3% (-8.4 points) |
| Correctness | 100.0% → 86.7% (-13.3 points) | 100.0% → 96.7% (-3.3 points) |
| Discoverability | 83.0% — baseline ran, but no comparable score was available; uplift unavailable | 84.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 98.3% → 98.3% (±0.0 points) | 93.5% → 88.3% (-5.2 points) |
| Efficiency | 76.8% — baseline ran, but no comparable score was available; uplift unavailable | 60.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 87.2% → 85.9% (-1.3 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 75.5% → 75.7% (+0.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 3,130,236 | 2,705,716 | +424,520 | +15.69% | skill 6/6; base 6/6 |
| claude-code | convert-hf-to-te-001 | 678,576 | 510,920 | +167,656 | +32.81% | skill 1/1; base 1/1 |
| claude-code | convert-hf-to-te-002 | 35,886 | 36,154 | -268 | -0.74% | skill 1/1; base 1/1 |
| claude-code | convert-hf-to-te-003 | 77,132 | 142,222 | -65,090 | -45.77% | skill 1/1; base 1/1 |
| claude-code | convert-hf-to-te-004 | 382,962 | 681,200 | -298,238 | -43.78% | skill 1/1; base 1/1 |
| claude-code | convert-hf-to-te-006 | 435,647 | 315,074 | +120,573 | +38.27% | skill 1/1; base 1/1 |
| claude-code | convert-hf-to-te-007 | 1,520,033 | 1,020,146 | +499,887 | +49.00% | skill 1/1; base 1/1 |
| codex | All cases | 2,073,128 | 2,555,807 | -482,679 | -18.89% | skill 6/6; base 6/6 |
| codex | convert-hf-to-te-001 | 296,247 | 192,921 | +103,326 | +53.56% | skill 1/1; base 1/1 |
| codex | convert-hf-to-te-002 | 30,840 | 26,499 | +4,341 | +16.38% | skill 1/1; base 1/1 |
| codex | convert-hf-to-te-003 | 54,150 | 38,479 | +15,671 | +40.73% | skill 1/1; base 1/1 |
| codex | convert-hf-to-te-004 | 649,640 | 756,717 | -107,077 | -14.15% | skill 1/1; base 1/1 |
| codex | convert-hf-to-te-006 | 308,970 | 989,618 | -680,648 | -68.78% | skill 1/1; base 1/1 |
| codex | convert-hf-to-te-007 | 733,281 | 551,573 | +181,708 | +32.94% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 5,203,364 | 5,261,523 | -58,159 | -1.11% | skill 12/12; base 12/12 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 10 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 6 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/convert-hf-to-te/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/convert-hf-to-te/SKILL.md`)
- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/bionemo-recipes-acceleration/SKILL.md (`SKILL.md:226`)
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
