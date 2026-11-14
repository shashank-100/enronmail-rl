"""Scripted stand-ins for ART + the OpenAI client, so train_art.py's rollout path can be checked on a Mac with no GPU.

The fake model plays a real search strategy: search the sender->recipient emails, read one, then answer with the gold
answer half the time (and a wrong one otherwise), so both reward branches and every tool are exercised."""
import json
import re
from types import SimpleNamespace


class FakeTrajectory:
    def __init__(self, messages_and_choices, reward, metadata):
        self.messages_and_choices, self.reward, self.metadata, self.metrics, self.tools = messages_and_choices, reward, metadata, {}, None

    def messages(self):
        out = []
        for m in self.messages_and_choices:
            out.append(m if isinstance(m, dict) else {"role": "assistant", "content": m.message.content,
                                                      "tool_calls": [{"id": t.id, "function": {"name": t.function.name, "arguments": t.function.arguments}} for t in (m.message.tool_calls or [])]})
        return out


class FakeArt:
    Trajectory = FakeTrajectory


class FakeModel:
    def get_inference_name(self):
        return "fake"


def _call(i, name, args):
    return SimpleNamespace(id=f"call_{i}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))


class _Completions:
    def __init__(self):
        self.n = 0

    async def create(self, model, messages, tools, temperature, max_completion_tokens):
        self.n += 1
        q = next(m["content"] for m in messages if m["role"] == "user")
        addrs = re.findall(r"[\w.'-]+@[\w.-]+", q)
        turn = sum(1 for m in messages if m["role"] == "assistant")
        if turn == 0:
            calls = [_call(self.n, "search_emails", {"sender": addrs[0], "recipient": addrs[1] if len(addrs) > 1 else "", "limit": 5})]
        elif turn == 1:
            last = json.loads(next(m["content"] for m in reversed(messages) if m["role"] == "tool"))
            calls = [_call(self.n, "read_email", {"email_id": last[0]["id"] if last else 1})]
        else:
            gold = getattr(self, "gold", {})
            ans = gold.get(q, ("unknown", []))
            if self.n % 2:
                ans = ("2099-01-01", [1])
            calls = [_call(self.n, "answer", {"answer": ans[0], "citations": ans[1]})]
        msg = SimpleNamespace(content=None, tool_calls=calls)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")])


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_Completions())
        tasks = [json.loads(l) for l in open(__import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "tasks_eval.jsonl")]
        self.chat.completions.gold = {t["question"]: (t["answer"], t["evidence"]) for t in tasks}
