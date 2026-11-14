# EnronMail-RL

Multi-hop questions over the Enron email archive, with **exactly verifiable answers and evidence**, a reward that scores
answer, evidence recall and citation precision, a tool-using agent baseline, and RL on a small model on a laptop.

Built on [OpenPipe ART·E](https://github.com/OpenPipe/email-deep-research) (Enron + RL). ART·E's questions are
single-hop and LLM-written, judged by an LLM. This project adds what that setup can't measure:

- **Multi-hop tasks generated from email metadata**, so every answer and its evidence set are exact (no LLM judge):
  - `reply_date`: X emailed Y about S; when did Y first reply? (find the original, then the reply)
  - `next_subject`: after X's email about S, what did Y send X next?
  - `count`: how many emails did X send Y in a month? (aggregate; duplicate copies across folders count once)
  - `unanswerable`: a plausible question about an email that never happened; the right answer is "unknown"
- **A citation-aware reward**: `R = 0.5·answer + 0.3·evidence recall + 0.2·citation precision − 0.01·tool calls`.
  Padding citations is penalised; answering "unknown" to an answerable question scores 0.
- **Verified tasks**: an oracle scores R = 1.0 on all 200 eval tasks; wrong answers score 0.

## Setup
```bash
uv sync
mkdir -p data   # download corbt/enron-emails (3 parquet shards, ~490 MB) into data/emails_*.parquet
uv run python -m enronmail_rl.db      # 517,401 emails -> data/enron.db (SQLite + FTS5, ~2 GB)
uv run python -m enronmail_rl.tasks   # data/tasks_eval.jsonl (200) + data/tasks_train.jsonl (1000), disjoint
```

## Run
```bash
uv run python -m enronmail_rl.cli search --from a@enron.com --to b@enron.com --after 2001-01-01   # the agent's tools
uv run python -m enronmail_rl.agent --n 100              # Claude Code agent with ./search and ./read only
uv run --extra rl python -m enronmail_rl.rl --out runs/rl  # SFT + GRPO on Qwen2.5-0.5B (Apple silicon, MLX)
```

## Results
See `docs/RESULTS.md`.
