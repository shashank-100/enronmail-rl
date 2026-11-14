# Results

## Task and checker verification
- 200 eval + 1000 train multi-hop tasks (reply_date, next_subject, count, unanswerable; 25% each), disjoint.
- Oracle answers score R = 1.0 on 200/200 eval tasks; wrong answers score 0. Count tasks collapse duplicate copies
  of the same email (Enron stores one per folder with different Message-IDs).

## Claude agent baseline (Claude Opus 5.5, medium effort, headless Claude Code with ./search and ./read only)
100 eval tasks, 25 per kind. A = answer correct, E = evidence recall, C = citation precision, R = reward.

| Kind | A | E | C | R | Tool turns |
|---|---|---|---|---|---|
| reply_date | 0.96 | 0.96 | 0.95 | 0.94 | 3.0 |
| next_subject | 0.96 | 0.98 | 0.98 | 0.95 | 3.0 |
| count | **0.80** | 0.95 | 1.00 | 0.86 | 3.4 |
| unanswerable | 1.00 | 1.00 | **0.84** | 0.94 | 3.8 |
| **all** | **0.93** | 0.97 | 0.94 | **0.92** | 3.3 |

- Counting is the hard part: emails stored once per folder look like separate messages, and the agent miscounts 20%.
- It always abstains on unanswerable questions, but cites (irrelevant) emails 16% of the time when it does.
- Cost: $6.72 equivalent for 100 tasks. Per-task logs: `docs/runs/claude_agent_runs.jsonl`.

## RL: Qwen2.5-14B-Instruct + GRPO (ART), 1x H100
Pending the GPU run (`vm/setup.sh`, then `vm/train_art.py`). The job runs 2 preflight rollouts, a base-model eval,
then GRPO (8 groups x 6 rollouts per step, lr 1.2e-5) with held-out evals every 20 steps.
