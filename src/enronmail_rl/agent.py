"""Run a tool-using agent (headless Claude Code with only ./search and ./read) on the eval tasks and score it.

    uv run python -m enronmail_rl.agent --n 100 --model claude-opus-5-5
"""
import argparse
import collections
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .reward import score

ROOT = Path(__file__).resolve().parents[2]
SYSTEM = """You answer questions about the Enron email archive (517k emails, 1998-2002) using two commands, run with Bash:
  ./search [--keywords "words"] [--from addr] [--to addr] [--after YYYY-MM-DD] [--before YYYY-MM-DD] [--limit N]
      emails matching ALL filters, newest first: id, date, from, to, subject, snippet. --to matches To and Cc.
  ./read <id>      one full email
The same email can appear several times (one copy per mailbox folder); count such copies once.
Work step by step: find the emails the question depends on, verify them, then answer.
If the emails the question assumes do not exist, the answer is "unknown".
End with ONLY this JSON on the last line:
{"answer": "<short answer: a YYYY-MM-DD date, an email subject, a number, or unknown>", "citations": [<ids of the emails that prove the answer>]}"""


def run_one(task, model, effort, root):
    d = Path(tempfile.mkdtemp(prefix="enron-", dir=root))
    py = ROOT / ".venv" / "bin" / "python"
    for name, sub in (("search", "search"), ("read", "read")):
        f = d / name
        f.write_text(f'#!/bin/sh\ncd "{ROOT}" && exec "{py}" -m enronmail_rl.cli {sub} "$@"\n')
        f.chmod(0o755)
    cmd = ["claude", "-p", task["question"], "--output-format", "json", "--model", model, "--effort", effort,
           "--system-prompt", SYSTEM, "--tools", "Bash", "--allowedTools", "Bash(./search:*),Bash(./read:*),Bash(./search),Bash(./read)",
           "--restricted", "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence", "--max-budget-usd", "1"]
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDECODE")}
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=900, env=env)
        out = json.loads(p.stdout) if p.stdout.strip().startswith("{") else {}
    except subprocess.TimeoutExpired:
        out = {}
    turns = out.get("num_turns", 0)
    s = score(task, out.get("result", ""), tool_calls=max(0, turns - 1))
    u = out.get("usage") or {}
    return {"id": task["id"], "kind": task["kind"], **s, "turns": turns, "cost_usd": out.get("total_cost_usd", 0),
            "tokens": u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0) + u.get("output_tokens", 0),
            "seconds": round(time.time() - t0, 1), "gold": task["answer"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="tasks, balanced across kinds")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    tasks = [json.loads(l) for l in open(ROOT / "data" / "tasks_eval.jsonl")]
    by = collections.defaultdict(list)
    for t in tasks:
        by[t["kind"]].append(t)
    pick = [t for k in sorted(by) for t in by[k][:a.n // len(by)]]
    out_dir = ROOT / "results" / f"agent_{a.model}_{time.strftime('%Y%m%d-%H%M')}"
    out_dir.mkdir(parents=True)
    root = tempfile.mkdtemp(prefix="enron-runs-")
    rows = []
    with ThreadPoolExecutor(a.workers) as ex, open(out_dir / "runs.jsonl", "w") as f:
        for r in ex.map(lambda t: run_one(t, a.model, a.effort, root), pick):
            rows.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"{'OK ' if r['A'] else 'BAD'} {r['id']:10s} {r['kind']:13s} R={r['R']:.2f} E={r['E']:.2f} C={r['C']:.2f} turns={r['turns']} ans={str(r['answer'])[:40]!r} gold={r['gold'][:30]!r}", flush=True)
    summ = {"model": a.model, "n": len(rows)}
    for k in ["all"] + sorted(by):
        rs = rows if k == "all" else [r for r in rows if r["kind"] == k]
        if rs:
            summ[k] = {m: round(sum(r[m] for r in rs) / len(rs), 3) for m in ("A", "E", "C", "R", "turns")}
    summ["total_cost_usd"] = round(sum(r["cost_usd"] or 0 for r in rows), 2)
    (out_dir / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    import shutil
    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
