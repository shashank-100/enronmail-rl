"""Multi-hop email questions built from metadata, so every answer and its evidence set are exact.

Kinds:
  reply_date   X emailed Y about S; when did Y first reply ("RE: S")?            evidence: original + reply
  next_subject after X's email about S, what was the subject of Y's next email to X?   evidence: original + next
  count        how many emails did X send Y in <month>?                         evidence: all of them
  unanswerable a plausible question whose premise never happened                 answer: "unknown"

    uv run python -m enronmail_rl.tasks     # writes data/tasks_eval.jsonl (200) and data/tasks_train.jsonl (1000)
"""
import json
import random
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "enron.db"
ENRON = "@enron.com"


def norm_subject(s):
    s = re.sub(r"^\s*((re|fw|fwd)\s*:\s*)+", "", s or "", flags=re.I)
    return " ".join(s.lower().split())


def month_name(ym):
    import datetime
    return datetime.date(int(ym[:4]), int(ym[5:7]), 1).strftime("%B %Y")


def candidates(con, rng, n):
    """Random (original email, recipient) pairs between two Enron people: sample ids in Python, then filter."""
    top = con.execute("SELECT max(id) FROM emails").fetchone()[0]
    out = []
    while len(out) < n * 40:
        ids = rng.sample(range(1, top + 1), 4000)
        rows = con.execute(f"""
            SELECT a.id, a.date, a.sender, r.address, a.subject
            FROM emails a JOIN recipients r ON r.email_id = a.id AND r.kind = 'to'
            WHERE a.id IN ({",".join(map(str, ids))})
              AND a.sender LIKE '%@enron.com' AND r.address LIKE '%@enron.com' AND a.sender != r.address
              AND length(a.subject) BETWEEN 12 AND 80 AND lower(a.subject) NOT LIKE 're:%' AND lower(a.subject) NOT LIKE 'fw%'""").fetchall()
        small = {e for (e,) in con.execute(f"SELECT email_id FROM recipients WHERE email_id IN ({','.join(str(r[0]) for r in rows) or '0'}) GROUP BY email_id HAVING count(*) <= 3")}
        out += [r for r in rows if r[0] in small]
    rng.shuffle(out)
    return out


def first_reply(con, a_id, a_date, x, y, subj):
    return con.execute("""
        SELECT e.id, e.date FROM emails e JOIN recipients r ON r.email_id = e.id AND r.address = ?
        WHERE e.sender = ? AND e.date > ? AND lower(e.subject) IN (?, ?) ORDER BY e.date LIMIT 1""",
                       (x, y, a_date, "re: " + subj.lower(), "re:" + subj.lower())).fetchone()


def make(n_each, seed, exclude=frozenset()):
    rng = random.Random(seed)
    con = sqlite3.connect(DB)
    con.execute("PRAGMA temp_store=MEMORY")
    out, used = {"reply_date": [], "next_subject": [], "count": [], "unanswerable": []}, set(exclude)
    for a_id, a_date, x, y, subj in candidates(con, rng, n_each * 3):
        if all(len(v) >= n_each for v in out.values()):
            break
        key = (x, y, norm_subject(subj))
        if key in used:
            continue
        rep = first_reply(con, a_id, a_date, x, y, subj)
        day = a_date[:10]
        if rep and len(out["reply_date"]) < n_each:
            used.add(key)
            out["reply_date"].append({"kind": "reply_date", "question": f'On {day}, {x} emailed {y} with the subject "{subj}". On what date did {y} first reply to that email?',
                                      "answer": rep[1][:10], "evidence": [a_id, rep[0]]})
            continue
        nxt = con.execute("""SELECT e.id, e.subject FROM emails e JOIN recipients r ON r.email_id = e.id AND r.address = ?
                             WHERE e.sender = ? AND e.date > ? AND length(e.subject) > 3 ORDER BY e.date LIMIT 1""", (x, y, a_date)).fetchone()
        if nxt and len(out["next_subject"]) < n_each:
            used.add(key)
            out["next_subject"].append({"kind": "next_subject", "question": f'After {x} emailed {y} about "{subj}" on {day}, what was the subject of the next email {y} sent to {x}?',
                                        "answer": nxt[1], "evidence": [a_id, nxt[0]]})
            continue
        ym = a_date[:7]
        ids, seen_msg = [], set()  # the same email is stored once per folder with a different Message-ID: count it once
        for eid, k in con.execute("""SELECT e.id, e.date || '|' || e.subject || '|' || substr(e.body,1,300) FROM emails e
                                     JOIN recipients r ON r.email_id = e.id AND r.address = ? AND r.kind = 'to'
                                     WHERE e.sender = ? AND substr(e.date,1,7) = ? ORDER BY e.id""", (y, x, ym)):
            if k not in seen_msg:
                seen_msg.add(k)
                ids.append(eid)
        if 2 <= len(ids) <= 12 and len(out["count"]) < n_each and (x, y, ym) not in used:
            used.add((x, y, ym))
            out["count"].append({"kind": "count", "question": f"How many emails did {x} send directly to {y} during {month_name(ym)}?",
                                 "answer": str(len(ids)), "evidence": ids})
            continue
        if len(out["unanswerable"]) < n_each:  # same people, real-looking subject, but a year with no such email
            fake_day = str(int(day[:4]) - 3) + day[4:]
            if not con.execute("SELECT 1 FROM emails WHERE sender=? AND date LIKE ? LIMIT 1", (x, fake_day[:4] + "%")).fetchone():
                used.add(key)
                out["unanswerable"].append({"kind": "unanswerable", "question": f'On {fake_day}, {x} emailed {y} with the subject "{subj}". On what date did {y} first reply to that email?',
                                            "answer": "unknown", "evidence": []})
    tasks = [t for v in out.values() for t in v]
    rng.shuffle(tasks)
    return tasks, used


def main():
    ev, used = make(50, seed=1)
    tr, _ = make(250, seed=2, exclude=frozenset(used))
    for name, ts in (("tasks_eval.jsonl", ev), ("tasks_train.jsonl", tr)):
        with open(ROOT / "data" / name, "w") as f:
            for i, t in enumerate(ts):
                f.write(json.dumps({"id": f"{name.split('_')[1].split('.')[0]}-{i}", **t}) + "\n")
        from collections import Counter
        print(name, len(ts), dict(Counter(t["kind"] for t in ts)))


if __name__ == "__main__":
    main()
