"""Score an agent's final answer: R = 0.5*A + 0.3*E + 0.2*C - 0.01*tool_calls (cost capped at 0.2).

A  answer correct (exact date / subject / count; "unknown" for unanswerable; "unknown" on an answerable task scores 0)
E  evidence recall: share of the gold evidence emails the agent cited
C  citation precision: share of cited emails that are gold evidence (padding with irrelevant emails is penalised)
Duplicate copies of the same email (Enron stores one per folder) count as the same email.
The agent's final message must contain JSON: {"answer": "...", "citations": [email ids]}.
"""
import json
import re
import sqlite3
import threading
from functools import lru_cache

from .tasks import DB, norm_subject

UNKNOWN = re.compile(r"\b(unknown|not found|no such|cannot (?:be )?(?:found|determined)|no (?:record|email|evidence)|did not happen|never)\b", re.I)


_local = threading.local()


def _con():
    """One read-only connection per thread: sqlite connections must not be shared across threads."""
    if not hasattr(_local, "con"):
        _local.con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    return _local.con



@lru_cache(maxsize=200_000)
def canon(email_id: int) -> str:
    r = _con().execute("SELECT date || '|' || subject || '|' || substr(body,1,300) FROM emails WHERE id=?", (int(email_id),)).fetchone()
    return r[0] if r else f"missing:{email_id}"


def parse(final_text: str):
    m = None
    for m in re.finditer(r"\{[^{}]*\"answer\"[^{}]*\}", final_text or "", re.S):
        pass
    if not m:
        return None, []
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, []
    cites = []
    for c in d.get("citations") or []:
        try:
            cites.append(int(c))
        except (TypeError, ValueError):
            pass
    return str(d.get("answer", "")), cites


def answer_ok(task, ans: str) -> bool:
    if ans is None:
        return False
    if task["kind"] == "unanswerable":
        return bool(UNKNOWN.search(ans))
    if UNKNOWN.search(ans) and task["kind"] != "next_subject":
        return False
    if task["kind"] == "reply_date":
        return task["answer"] in re.findall(r"\d{4}-\d{2}-\d{2}", ans)
    if task["kind"] == "count":
        nums = re.findall(r"\b\d+\b", ans)
        return bool(nums) and nums[0] == task["answer"]
    if task["kind"] == "next_subject":
        g, a = norm_subject(task["answer"]), norm_subject(ans.strip().strip('"'))
        return bool(g) and (a == g or (len(g) > 6 and g in a))
    return False


def score(task, final_text: str, tool_calls: int = 0) -> dict:
    ans, cites = parse(final_text)
    A = float(answer_ok(task, ans))
    gold = {canon(i) for i in task["evidence"]}
    cited = {canon(i) for i in cites}
    if not gold:  # unanswerable: nothing to cite
        E = 1.0 if A else 0.0
        C = 1.0 if (A and not cited) else (0.0 if cited else A)
    else:
        E = len(gold & cited) / len(gold)
        C = len(gold & cited) / len(cited) if cited else 0.0
    cost = min(0.2, 0.01 * tool_calls)
    return {"A": A, "E": round(E, 3), "C": round(C, 3), "R": round(0.5 * A + 0.3 * E + 0.2 * C - cost, 3),
            "parsed": ans is not None, "answer": ans, "citations": cites}
