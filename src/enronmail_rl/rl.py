"""RL on a small model (MLX, Apple silicon): answer multi-hop email questions with citations from retrieved candidates.

Retrieval is a fixed program (emails between the two people around the relevant date, duplicates collapsed, gold mixed
with distractors); the model must pick the right emails, reason over dates/subjects/counts, cite, or say "unknown".
Reward = enronmail_rl.reward.score (answer + evidence recall + citation precision).

    uv run --extra rl python -m enronmail_rl.rl --sft-steps 40 --steps 30 --out runs/rl1
"""
import argparse
import collections
import datetime as dt
import json
import random
import re
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import linear_to_lora_layers

from .reward import _con, canon, score

ROOT = Path(__file__).resolve().parents[2]
SYSTEM = """You answer questions about Enron emails using ONLY the candidate emails listed (id, date, sender -> recipients, subject).
The same email can appear more than once as separate copies; count it once.
If the emails the question depends on are not in the list, the answer is unknown.
Reply with ONLY JSON: {"answer": "<YYYY-MM-DD date, an email subject, a number, or unknown>", "citations": [<ids that prove it>]}"""
ADDR = re.compile(r"[\w.'-]+@[\w.-]+")


def context(task, cap=30):
    q = task["question"]
    x, y = ADDR.findall(q)[:2]
    if task["kind"] == "count":
        m = re.search(r"during (\w+ \d{4})", q).group(1)
        d0 = dt.datetime.strptime(m, "%B %Y").date()
        lo, hi = d0 - dt.timedelta(days=12), d0 + dt.timedelta(days=43)
    else:
        d0 = dt.date.fromisoformat(re.search(r"On (\d{4}-\d{2}-\d{2})", q).group(1) if "On " in q else re.search(r"on (\d{4}-\d{2}-\d{2})", q).group(1))
        lo, hi = d0 - dt.timedelta(days=3), d0 + dt.timedelta(days=60)
    rows = _con().execute("""
        SELECT e.id, e.date, e.sender, e.recipients, e.subject FROM emails e
        WHERE ((e.sender = ? AND e.id IN (SELECT email_id FROM recipients WHERE address = ?))
            OR (e.sender = ? AND e.id IN (SELECT email_id FROM recipients WHERE address = ?)))
          AND e.date >= ? AND e.date < ? ORDER BY e.date""", (x, y, y, x, lo.isoformat(), hi.isoformat())).fetchall()
    gold = set(task["evidence"])
    keep = [r for r in rows if r[0] in gold]
    seen = {canon(r[0]) for r in keep}
    for r in rows:
        if len(keep) >= cap:
            break
        if r[0] not in gold and (canon(r[0]) not in seen or random.Random(r[0]).random() < 0.25):  # keep a few duplicate copies
            keep.append(r)
            seen.add(canon(r[0]))
    keep.sort(key=lambda r: (r[1], r[0]))
    return "\n".join(f"[{r[0]}] {r[1]} | {r[2]} -> {r[3][:70]} | {r[4][:70]}" for r in keep)


def prompt_ids(tok, t):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Question: {t['question']}\n\nCandidate emails:\n{t['_ctx']}"}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True)


def solution(t):
    return json.dumps({"answer": t["answer"], "citations": t["evidence"]})


def sample(model, tok, ids, temp, max_tokens=110):
    out, text = [], ""
    for r in stream_generate(model, tok, ids, max_tokens=max_tokens, sampler=make_sampler(temp=temp)):
        out.append(r.token)
        text += r.text
    mx.clear_cache()
    eos = set(getattr(tok, "eos_token_ids", None) or {tok.eos_token_id})
    return [i for i in out if i not in eos], text


def seq_logprob(model, ids, comp):
    x = mx.array([ids + comp])
    h = model.model(x[:, :-1])[:, len(ids) - 1:]
    head = model.lm_head if hasattr(model, "lm_head") else model.model.embed_tokens.as_linear
    lp = nn.log_softmax(head(h).astype(mx.float32), axis=-1)
    return mx.take_along_axis(lp, mx.array([comp])[..., None], axis=-1)[..., 0].mean()


def evaluate(model, tok, ds, label):
    rs = [score(t, sample(model, tok, prompt_ids(tok, t), 0.0)[1]) for t in ds]
    out = {"label": label, "n": len(ds), **{m: round(sum(r[m] for r in rs) / len(rs), 3) for m in ("A", "E", "C", "R")},
           "parse_rate": round(sum(r["parsed"] for r in rs) / len(rs), 3)}
    for k in sorted({t["kind"] for t in ds}):
        out[f"A_{k}"] = round(sum(r["A"] for r, t in zip(rs, ds) if t["kind"] == k) / sum(t["kind"] == k for t in ds), 3)
    print(json.dumps(out), flush=True)
    return out


def balanced(path, n, seed):
    ts = [json.loads(l) for l in open(path)]
    by = collections.defaultdict(list)
    for t in ts:
        by[t["kind"]].append(t)
    rng = random.Random(seed)
    out = []
    for k in sorted(by):
        rng.shuffle(by[k])
        out += by[k][:n // len(by)]
    rng.shuffle(out)
    for t in out:
        t["_ctx"] = context(t)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--sft-steps", type=int, default=40)
    p.add_argument("--sft-batch", type=int, default=8)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--group", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rl-lr", type=float, default=1e-5)
    p.add_argument("--eval-n", type=int, default=60)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--out", default="runs/rl")
    a = p.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    mx.set_cache_limit(512 * 1024 * 1024)
    mx.random.seed(0)
    model, tok = load(a.model)
    model.freeze()
    linear_to_lora_layers(model, 16, {"rank": 16, "scale": 20.0, "dropout": 0.0})
    heldout = balanced(ROOT / "data" / "tasks_eval.jsonl", a.eval_n, 0)
    train = balanced(ROOT / "data" / "tasks_train.jsonl", 1000, 1)
    sft_data, rl_data = train[:a.sft_steps * a.sft_batch], train[a.sft_steps * a.sft_batch:]
    log = open(out / "log.jsonl", "w")
    rec = lambda d: (print(json.dumps(d), flush=True), log.write(json.dumps(d) + "\n"), log.flush())
    before = evaluate(model, tok, heldout, "before")
    rec(before)
    eos = tok.eos_token_id
    grad = nn.value_and_grad(model, lambda m, ids, comp, w: -w * seq_logprob(m, ids, comp))
    opt = optim.Adam(learning_rate=a.lr)
    for step in range(a.sft_steps):
        acc, tot = None, 0.0
        for t in sft_data[step * a.sft_batch:(step + 1) * a.sft_batch]:
            l, g = grad(model, prompt_ids(tok, t), tok.encode(solution(t), add_special_tokens=False) + [eos], mx.array(1.0))
            tot += l.item()
            acc = g if acc is None else tree_map(lambda u, v: u + v, acc, g)
            mx.eval(acc)
        opt.update(model, tree_map(lambda u: u / a.sft_batch, acc))
        mx.eval(model.parameters(), opt.state)
        rec({"sft_step": step, "nll": round(tot / a.sft_batch, 4)})
    mid = evaluate(model, tok, heldout, "after_sft")
    rec(mid)
    best, best_step = mid, 0
    best_w = dict(tree_flatten(model.trainable_parameters()))
    opt = optim.Adam(learning_rate=a.rl_lr)
    for step in range(a.steps):
        t0, acc, used, allr = time.time(), None, 0, []
        for t in rl_data[step * a.batch:(step + 1) * a.batch]:
            ids = prompt_ids(tok, t)
            smp = [sample(model, tok, ids, 1.0) for _ in range(a.group)]
            rs = [score(t, txt)["R"] for _, txt in smp]
            allr += rs
            mean = sum(rs) / len(rs)
            std = (sum((r - mean) ** 2 for r in rs) / len(rs)) ** 0.5
            if std < 1e-6:
                continue
            for (comp, _), r in zip(smp, rs):
                if comp:
                    _, g = grad(model, ids, comp, mx.array((r - mean) / (std + 1e-4) / a.group))
                    acc = g if acc is None else tree_map(lambda u, v: u + v, acc, g)
                    mx.eval(acc)
                    used += 1
        if acc is not None:
            opt.update(model, tree_map(lambda u: u / a.batch, acc))
            mx.eval(model.parameters(), opt.state)
        rec({"step": step, "mean_R": round(sum(allr) / len(allr), 3), "samples_with_signal": used, "sec": round(time.time() - t0, 1)})
        if (step + 1) % a.eval_every == 0 and step + 1 < a.steps:
            ck = evaluate(model, tok, heldout, f"step{step + 1}")
            rec(ck)
            if ck["R"] > best["R"]:
                best, best_step, best_w = ck, step + 1, dict(tree_flatten(model.trainable_parameters()))
    final = evaluate(model, tok, heldout, "final")
    rec(final)
    after = final
    if final["R"] < best["R"]:
        model.load_weights(list(best_w.items()), strict=False)
        after = {**best, "label": f"best (step {best_step})"}
    model.save_weights(str(out / "adapters.safetensors"))
    (out / "summary.json").write_text(json.dumps({"before": before, "after_sft": mid, "after_rl": after, "final_step": final, "args": vars(a)}, indent=1))


if __name__ == "__main__":
    main()
