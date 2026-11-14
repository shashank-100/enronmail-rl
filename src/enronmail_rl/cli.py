"""Command-line tools the agent calls:  search [--keywords ..] [--from ..] [--to ..] [--after ..] [--before ..] [--limit N]  |  read <id>"""
import argparse
import json

from . import tools


def main():
    p = argparse.ArgumentParser(prog="enron")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--keywords", default="")
    s.add_argument("--from", dest="sender", default="")
    s.add_argument("--to", dest="recipient", default="")
    s.add_argument("--after", default="")
    s.add_argument("--before", default="")
    s.add_argument("--limit", type=int, default=10)
    r = sub.add_parser("read")
    r.add_argument("id", type=int)
    a = p.parse_args()
    if a.cmd == "search":
        out = tools.search(a.keywords, a.sender, a.recipient, a.after, a.before, a.limit)
    else:
        out = tools.read(a.id)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
