"""The agent's tools over data/enron.db: keyword search with filters, and reading one email."""
import re
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parents[2] / "data" / "enron.db"
_con = None


def con():
    global _con
    if _con is None:
        _con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, check_same_thread=False)
    return _con


def search(keywords: str = "", sender: str = "", recipient: str = "", after: str = "", before: str = "", limit: int = 10):
    """Emails matching all filters, newest first. keywords: FTS over subject+body. after/before: YYYY-MM-DD (inclusive/exclusive)."""
    q, args = ["SELECT e.id, e.date, e.sender, e.recipients, e.subject, substr(e.body,1,160) FROM emails e"], []
    where = []
    if keywords.strip():
        terms = " ".join(f'"{t}"' for t in re.findall(r"[\w.@'-]+", keywords)[:8])
        q.append("JOIN emails_fts f ON f.rowid = e.id")
        where.append("emails_fts MATCH ?")
        args.append(terms)
    if sender:
        where.append("e.sender = ?")
        args.append(sender.lower())
    if recipient:
        where.append("e.id IN (SELECT email_id FROM recipients WHERE address = ?)")
        args.append(recipient.lower())
    if after:
        where.append("e.date >= ?")
        args.append(after)
    if before:
        where.append("e.date < ?")
        args.append(before)
    if where:
        q.append("WHERE " + " AND ".join(where))
    q.append("ORDER BY e.date DESC LIMIT ?")
    args.append(max(1, min(int(limit), 50)))
    rows = con().execute(" ".join(q), args).fetchall()
    return [{"id": r[0], "date": r[1], "from": r[2], "to": r[3][:120], "subject": r[4], "snippet": " ".join(r[5].split())} for r in rows]


def read(email_id: int):
    r = con().execute("SELECT id, date, sender, recipients, cc, subject, body FROM emails WHERE id = ?", (int(email_id),)).fetchone()
    if not r:
        return {"error": f"no email with id {email_id}"}
    return {"id": r[0], "date": r[1], "from": r[2], "to": r[3], "cc": r[4], "subject": r[5], "body": r[6][:4000]}
