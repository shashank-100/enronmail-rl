#!/usr/bin/env bash
# Whole GPU job on a fresh node, start to finish, with a hard time budget (default 2.5 h incl. setup).
#   curl -sSL https://raw.githubusercontent.com/shashank-100/enronmail-rl/main/vm/nc_job.sh | BUDGET_HOURS=2.5 bash
set -euo pipefail
BUDGET_HOURS="${BUDGET_HOURS:-2.5}"
T0=$(date +%s)
[ -d enronmail-rl ] || git clone --depth 1 https://github.com/shashank-100/enronmail-rl.git
cd enronmail-rl
bash vm/setup.sh 2>&1 | tee logs_setup.txt
USED_H=$(python3 -c "import time;print(round((time.time()-$T0)/3600,3))")
LEFT_H=$(python3 -c "print(max(0.3, $BUDGET_HOURS - $USED_H - 0.1))")
echo "setup took ${USED_H}h; training budget ${LEFT_H}h"
uv run python vm/train_art.py --steps 60 --max-hours "$LEFT_H" 2>&1 | tee train_log.txt
tar czf ~/enron_results.tgz results train_log.txt logs_setup.txt
echo "RESULTS: ~/enron_results.tgz  -- copy it off the node, then release the node"
