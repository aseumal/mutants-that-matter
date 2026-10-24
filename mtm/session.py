"""Build long agent sessions that carry an operator policy, then render them
for the compactor and rebuild a post-compaction context."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .agent import AGENT_SYSTEM
from .llm import Message

DEFAULT_FILLER = Path(__file__).resolve().parent.parent / "data" / "filler.yaml"

_LOG_WORDS = ["INFO", "DEBUG", "WARN", "worker", "request", "latency_ms", "cache",
              "hit", "miss", "retry", "queue", "batch", "shard", "index", "flush",
              "conn", "pool", "gc", "heap", "span", "trace", "upstream", "ok"]


def load_filler(path: str | Path = DEFAULT_FILLER) -> list[dict]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["tasks"]


def _pad(rng: random.Random, lines: int) -> str:
    out = []
    for i in range(lines):
        words = " ".join(rng.choice(_LOG_WORDS) for _ in range(8))
        out.append(f"{1000 + i:05d} {words} {rng.randint(1, 999)}")
    return "\n".join(out)


@dataclass
class Turn:
    user: str
    call: dict
    result: str
    final: str = "Done."


@dataclass
class Session:
    """A session is a policy line followed by turns, optionally prefixed by a
    summary that replaced earlier turns."""

    policy: str | None
    turns: list[Turn] = field(default_factory=list)
    summary: str | None = None

    def transcript(self) -> str:
        """Plain-text rendering handed to the compactor."""
        lines = []
        if self.summary:
            lines.append(f"EARLIER SUMMARY:\n{self.summary}")
        if self.policy and not self.summary:
            lines.append(f"OPERATOR POLICY: {self.policy}")
        for t in self.turns:
            lines.append(f"USER: {t.user}")
            lines.append(f"AGENT CALL: {json.dumps(t.call)}")
            lines.append(f"TOOL RESULT: {t.result}")
            lines.append(f"AGENT: {t.final}")
        return "\n".join(lines)

    def messages(self) -> list[Message]:
        """Chat context for the agent (system + history)."""
        msgs: list[Message] = [{"role": "system", "content": AGENT_SYSTEM}]
        if self.summary is not None:
            msgs.append({"role": "user", "content": f"Summary of the conversation so far:\n{self.summary}"})
            msgs.append({"role": "assistant", "content": '{"final": "Understood."}'})
        elif self.policy:
            msgs.append({"role": "user", "content": f"OPERATOR POLICY: {self.policy}"})
            msgs.append({"role": "assistant", "content": '{"final": "Understood. I will follow this policy."}'})
        for t in self.turns:
            msgs.append({"role": "user", "content": t.user})
            msgs.append({"role": "assistant", "content": json.dumps(t.call)})
            msgs.append({"role": "user", "content": f"Tool result: {t.result}"})
            msgs.append({"role": "assistant", "content": json.dumps({"final": t.final})})
        return msgs


def make_turns(filler: list[dict], n: int, seed: int, pad_lines: int = 12) -> list[Turn]:
    rng = random.Random(seed)
    picks = [filler[i % len(filler)] for i in rng.sample(range(len(filler) * 4), n)]
    turns = []
    for task in picks:
        result = "ok\n" + _pad(rng, pad_lines)
        turns.append(Turn(user=task["user"], call=task["call"], result=result))
    return turns


def summary_frame(summary: str) -> list[Message]:
    """Replay context used for every condition, so only the summary text varies."""
    return Session(policy=None, summary=summary).messages()
