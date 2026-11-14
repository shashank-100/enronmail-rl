#!/usr/bin/env bash
# One-shot VM setup for the 14B RL run. Run from the repo root on the GPU VM:  bash vm/setup.sh
# Model download (~30 GB) and Enron DB build (~5-8 min CPU) run in parallel to keep idle GPU time short.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== GPU"; nvidia-smi --query-gpu=name,memory.total --format=csv
python3 - <<'EOF'
import shutil; free = shutil.disk_usage(".").free / 1e9
assert free > 80, f"need ~80 GB free disk, have {free:.0f} GB"
print(f"disk ok: {free:.0f} GB free")
EOF

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync
uv pip install "openpipe-art[backend]==0.5.24" "huggingface_hub[cli]"

mkdir -p data logs
(  # model weights
  uv run huggingface-cli download Qwen/Qwen2.5-14B-Instruct > logs/model_download.log 2>&1 && echo "model downloaded"
) &
(  # Enron DB
  for i in 0 1 2; do curl -sSL -o "data/emails_$i.parquet" "https://huggingface.co/datasets/corbt/enron-emails/resolve/main/data/train-0000$i-of-00003.parquet"; done
  uv run python -m enronmail_rl.db > logs/db_build.log 2>&1 && echo "db built"
) &
wait

cp tasks/tasks_eval.jsonl tasks/tasks_train.jsonl data/
uv run python - <<'EOF'
import json, sys
sys.path.insert(0, "src")
from enronmail_rl.reward import score
ts = [json.loads(l) for l in open("data/tasks_eval.jsonl")]
ok = sum(score(t, json.dumps({"answer": t["answer"], "citations": t["evidence"]}))["R"] == 1.0 for t in ts)
assert ok == len(ts), f"oracle check failed: {ok}/{len(ts)} -- the DB does not match the tasks"
print(f"oracle check ok: {ok}/{len(ts)}")
import art, art.local.backend  # backend imports cleanly
print("ART", art.__version__ if hasattr(art, "__version__") else "ok")
EOF
uv run python vm/train_art.py --fake-client >/dev/null && echo "fake-client check ok"
echo "SETUP COMPLETE. Start training with:  nohup uv run python vm/train_art.py --steps 60 > logs/train.log 2>&1 &"
