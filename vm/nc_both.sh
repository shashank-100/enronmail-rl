#!/usr/bin/env bash
# Both RL runs on one 8x GPU node, concurrently, under one hard time budget (default 2.5 h including setup).
#   curl -sSL https://raw.githubusercontent.com/shashank-100/enronmail-rl/main/vm/nc_both.sh | BUDGET_HOURS=2.5 bash
# GPU 0: EnronMail-RL, Qwen2.5-14B multi-turn tool use.   GPU 1: GameLab Experiment D, Qwen2.5-7B tab model.
# Results: ~/both_results.tgz (copy it off, then release the node).
set -uo pipefail
BUDGET_HOURS="${BUDGET_HOURS:-2.5}"
T0=$(date +%s)
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
cd ~
[ -d enronmail-rl ] || git clone --depth 1 https://github.com/shashank-100/enronmail-rl.git
[ -d gamelab ] || git clone --depth 1 https://github.com/shashank-100/gamelab.git

# setup both in parallel (Enron: deps + 14B weights + DB build + oracle check; GameLab: deps + 7B weights + check)
( cd ~/enronmail-rl && bash vm/setup.sh > ~/setup_enron.log 2>&1; echo "enron setup exit $?" >> ~/setup_enron.log ) &
( cd ~/gamelab && uv sync --extra rl >/dev/null 2>&1 || uv sync; uv pip install "openpipe-art[backend]==0.5.24" "huggingface_hub[cli]" \
  && uv run huggingface-cli download Qwen/Qwen2.5-7B-Instruct >/dev/null \
  && uv run python vm/train_art.py --fake-client && echo "gamelab setup ok" ) > ~/setup_gamelab.log 2>&1 &
wait
grep -q "SETUP COMPLETE" ~/setup_enron.log || { echo "ENRON SETUP FAILED"; tail -20 ~/setup_enron.log; }
grep -q "gamelab setup ok" ~/setup_gamelab.log || { echo "GAMELAB SETUP FAILED"; tail -20 ~/setup_gamelab.log; }

LEFT_H=$(python3 -c "import time;print(round(max(0.3, $BUDGET_HOURS - (time.time()-$T0)/3600 - 0.15), 2))")
echo "training budget ${LEFT_H}h per run"
( cd ~/enronmail-rl && CUDA_VISIBLE_DEVICES=0 uv run python vm/train_art.py --steps 60 --max-hours "$LEFT_H" > ~/train_enron.log 2>&1 ) &
( cd ~/gamelab && CUDA_VISIBLE_DEVICES=1 uv run python vm/train_art.py --steps 60 --max-hours "$LEFT_H" > ~/train_gamelab.log 2>&1 ) &
# hard stop: whatever happens, nothing runs past the budget
( sleep "$(python3 -c "print(int(($BUDGET_HOURS*3600) - (__import__('time').time()-$T0) - 300))")"; pkill -f vm/train_art.py ) &
WATCHDOG=$!
wait %2 %3 2>/dev/null; kill $WATCHDOG 2>/dev/null
tar czf ~/both_results.tgz -C ~ enronmail-rl/results gamelab/results setup_enron.log setup_gamelab.log train_enron.log train_gamelab.log 2>/dev/null
echo "DONE: ~/both_results.tgz ($(du -h ~/both_results.tgz | cut -f1)). Copy it off, then release the node."
