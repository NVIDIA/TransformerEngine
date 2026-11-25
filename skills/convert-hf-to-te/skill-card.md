## Description: <br>
Adapt a Hugging Face PyTorch model for training with compatible Transformer Engine modules, replacing a decoder layer, sublayer or individual operation, mapping its checkpoint weights, and choosing fusions that preserve the requested workload's behavior. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to convert Hugging Face PyTorch models (dense or mixture-of-experts) to use NVIDIA Transformer Engine modules for training, including choosing replacement granularity and fusions, mapping checkpoint weights, and validating numerical parity and performance against the original model. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Not Specified] <br>
**Credential Type(s):** [None identified] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Transformer Engine documentation](https://docs.nvidia.com/deeplearning/transformer-engine/index.html) <br>
- [Transformer Engine PyTorch API](https://docs.nvidia.com/deeplearning/transformer-engine/api/pytorch.html) <br>
- [Transformer Engine operation fuser guide](https://docs.nvidia.com/deeplearning/transformer-engine/examples/op_fuser/op_fuser.html) <br>
- [te_llama.py example](https://github.com/NVIDIA/TransformerEngine/blob/db1aff638b070cb9d08a0dc91629ff1457ead413/docs/examples/te_llama/te_llama.py) <br>
- [te_gemma.py example](https://github.com/NVIDIA/TransformerEngine/blob/db1aff638b070cb9d08a0dc91629ff1457ead413/docs/examples/te_gemma/te_gemma.py) <br>
- [te_mixtral.py example](https://github.com/NVIDIA/TransformerEngine/blob/db1aff638b070cb9d08a0dc91629ff1457ead413/docs/examples/te_mixtral/te_mixtral.py) <br>
- [Mixtral HF-to-TE weight mapping](https://github.com/NVIDIA/TransformerEngine/blob/db1aff638b070cb9d08a0dc91629ff1457ead413/docs/examples/te_mixtral/hf_to_te_weights.py) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Files, Shell commands, Analysis] <br>
**Output Format:** [Python source files and Markdown reports with inline code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
5 evaluation tasks (4 positive, 1 negative), 1 attempt per task, each run with and without the skill in its own isolated sandbox pod (`k8s-sandbox` environment); evaluator version `1.6.10`, evaluated 2026-11-25. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from `security` (100%). <br>
- Correctness: Is the answer correct? Scored from `accuracy` (100%). <br>
- Discoverability: Was the right skill loaded when needed? Scored from `skill_execution` (100%). <br>
- Effectiveness: Did the skill help complete the task? Scored from `goal_accuracy` (50%) + `behavior_check` (50%). <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Scored from `skill_efficiency` (50%) + `token_efficiency` (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Security: unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Skill Execution: whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool Productivity: tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Accuracy: final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Goal Accuracy: whether the user's goal was achieved. <br>
- `behavior_check`: Behavior Check: whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Token Efficiency: actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
> ✅ **Overall verdict: PASS — Recommended for publication**

### Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

### Results at a Glance

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

#### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 88.9% → 88.8% (-0.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 75.8% → 80.9% (+5.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 10 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 5 task(s) |

## Skill Version(s): <br>
0.4.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
