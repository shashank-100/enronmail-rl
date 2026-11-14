"""Build data/enron.db: one row per unique email (deduplicated by Message-ID), full-text search over subject + body.

    uv run python -m enronmail_rl.db
"""
import sqlite3
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "enron.db"


def clean_list(v):
    return [x.strip().lower() for x in (v or []) if x and x.strip()]


def build():
    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(DB)
    con.executescript("""
        PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        CREATE TABLE emails (id INTEGER PRIMARY KEY, message_id TEXT UNIQUE, date TEXT, sender TEXT,
                             recipients TEXT, cc TEXT, subject TEXT, body TEXT, folder TEXT);
        CREATE TABLE recipients (email_id INTEGER, address TEXT, kind TEXT);
    """)
    seen, n = set(), 0
    for shard in sorted((ROOT / "data").glob("emails_*.parquet")):
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=20000):
            rows, recs = [], []
            for r in batch.to_pylist():
                mid = r["message_id"]
                if mid in seen:
                    continue
                seen.add(mid)
                n += 1
                to, cc = clean_list(r["to"]), clean_list(r["cc"])
                date = r["date"].strftime("%Y-%m-%d %H:%M") if r["date"] else ""
                rows.append((n, mid, date, (r["from"] or "").lower(), ",".join(to), ",".join(cc),
                             r["subject"] or "", (r["body"] or "")[:20000], r["file_name"].rsplit("/", 1)[0]))
                recs += [(n, a, "to") for a in to] + [(n, a, "cc") for a in cc]
            con.executemany("INSERT INTO emails VALUES (?,?,?,?,?,?,?,?,?)", rows)
            con.executemany("INSERT INTO recipients VALUES (?,?,?)", recs)
            print(f"{shard.name}: {n} unique emails so far", flush=True)
    con.executescript("""
        CREATE INDEX i_sender ON emails(sender); CREATE INDEX i_date ON emails(date);
        CREATE INDEX i_rec ON recipients(address); CREATE INDEX i_rec_e ON recipients(email_id);
        CREATE VIRTUAL TABLE emails_fts USING fts5(subject, body, content='emails', content_rowid='id', tokenize='porter');
        INSERT INTO emails_fts(rowid, subject, body) SELECT id, subject, body FROM emails;
    """)
    con.commit()
    print(f"DONE: {n} unique emails -> {DB} ({DB.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    sys.exit(build())
