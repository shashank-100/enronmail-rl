# Results

## Task and checker verification
- 200 eval + 1000 train multi-hop tasks (reply_date, next_subject, count, unanswerable; 25% each), disjoint.
- Oracle answers score R = 1.0 on 200/200 eval tasks; wrong answers score 0. Count tasks collapse duplicate copies
  of the same email (Enron stores one per folder with different Message-IDs).

## Claude agent baseline
(filled in by `enronmail_rl.agent`, see below)

## RL: Qwen2.5-14B-Instruct + GRPO (ART), 1x H100
Pending the GPU run (`vm/setup.sh`, then `vm/train_art.py`). The job runs 2 preflight rollouts, a base-model eval,
then GRPO (8 groups x 6 rollouts per step, lr 1.2e-5) with held-out evals every 20 steps.
