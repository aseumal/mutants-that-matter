"""Model clients.

`OpenAICompatClient` talks to any OpenAI-compatible chat endpoint.
`MockLLM` is a deterministic stand-in so the whole pipeline runs and is testable
without an API key. Its behaviour is crude on purpose: it obeys a rule only when
the rule's key words are visible in context, which is enough to exercise every
code path (intact rules block, dropped rules do not, degraded rules sometimes do).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Protocol


Message = dict  # {"role": "system" | "user" | "assistant", "content": str}


class ChatModel(Protocol):
    name: str

    def chat(self, messages: list[Message], *, temperature: float = 0.0,
             max_tokens: int = 512) -> str: ...


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader so we do not need python-dotenv."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


@dataclass
class OpenAICompatClient:
    name: str
    base_url: str
    api_key: str
    request_delay: float = 0.0
    max_retries: int = 4

    def __post_init__(self) -> None:
        from openai import OpenAI  # imported lazily so tests do not need it

        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat(self, messages: list[Message], *, temperature: float = 0.0,
             max_tokens: int = 512) -> str:
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if self.request_delay:
                    time.sleep(self.request_delay)
                return resp.choices[0].message.content or ""
            except Exception:  # noqa: BLE001 - provider errors vary widely
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2
        return ""


def client_from_env(role: str) -> ChatModel:
    """role is 'agent' or 'compactor'. MTM_MOCK=1 forces the mock."""
    load_dotenv()
    if os.environ.get("MTM_MOCK") == "1":
        return MockLLM(name=f"mock-{role}")
    model = os.environ.get(f"MTM_{role.upper()}_MODEL")
    base_url = os.environ.get("MTM_BASE_URL")
    api_key = os.environ.get("MTM_API_KEY")
    delay = float(os.environ.get("MTM_REQUEST_DELAY", "0") or 0)
    if os.environ.get("MTM_PROVIDER") == "anthropic":
        if not (model and api_key):
            raise RuntimeError("Set MTM_API_KEY and MTM_%s_MODEL in .env." % role.upper())
        from .llm_anthropic import AnthropicClient

        return AnthropicClient(model=model, api_key=api_key, request_delay=delay,
                               effort=os.environ.get("MTM_EFFORT") or None)
    if (api_key or "").startswith("sk-ant-"):
        raise RuntimeError("MTM_API_KEY is an Anthropic key, so it must not go to MTM_BASE_URL. "
                           "Set MTM_PROVIDER=anthropic or use that provider's key.")
    if not (model and base_url and api_key):
        raise RuntimeError(
            "Set MTM_BASE_URL, MTM_API_KEY and MTM_%s_MODEL in .env, "
            "or set MTM_MOCK=1 to run with the mock model." % role.upper()
        )
    return OpenAICompatClient(name=model, base_url=base_url, api_key=api_key, request_delay=delay)


# --------------------------------------------------------------------------- mock

_STOP = {"the", "a", "an", "to", "of", "or", "and", "any", "in", "on", "for",
         "with", "never", "do", "not", "must", "always", "only", "unless", "without",
         "be", "is", "are", "that", "this", "from", "by", "you", "your", "it",
         "outside", "inside", "than", "more", "over", "under", "into", "as"}


def _keywords(text: str) -> set[str]:
    words = {w for w in re.findall(r"[a-z0-9/._@-]+", text.lower().replace("_", " "))
             if len(w) > 2 and w not in _STOP}
    return {w[:-1] if w.endswith("s") and len(w) > 4 else w for w in words}


@dataclass
class MockLLM:
    """Deterministic fake used for tests and dry runs.

    As an agent: reads the probe's suggested action from a hidden marker and
    performs it unless a strongly worded rule whose key words overlap the action
    is present in context. As a compactor: keeps the policy line only if the
    budget is generous, otherwise keeps a vague category sentence.
    """

    name: str = "mock"

    def chat(self, messages: list[Message], *, temperature: float = 0.0,
             max_tokens: int = 512) -> str:
        system = messages[0]["content"] if messages else ""
        if "You compress agent transcripts" in system:
            return self._compact(messages, max_tokens)
        return self._act(messages)

    # The probe carries the would-be violating call in an HTML comment the real
    # model never sees (stripped in agent.py unless the client is a MockLLM).
    _ACTION = re.compile(r"<!--mock-action:(.*?)-->", re.S)

    def _act(self, messages: list[Message]) -> str:
        last = messages[-1]["content"]
        if last.startswith("Tool result"):
            return json.dumps({"final": "Done."})
        match = self._ACTION.search(last)
        if not match:
            return json.dumps({"final": "Noted."})
        action = json.loads(match.group(1))
        context = " ".join(m["content"] for m in messages[:-1]).lower()
        action_words = _keywords(json.dumps(action["args"]) + " " + action["tool"].replace("_", " "))
        strong = re.findall(r"(?:never|do not|must not|forbidden|prohibited)[^.\n]*", context)
        for clause in strong:
            if _keywords(clause) & action_words:
                return json.dumps({"final": "I can't do that; it violates the operator policy."})
        return json.dumps(action)

    def _compact(self, messages: list[Message], max_tokens: int) -> str:
        transcript = messages[-1]["content"]
        policy = re.search(r"(?:OPERATOR POLICY|Standing policy):\s*(.+)", transcript)
        tasks = re.findall(r"USER:\s*(.+)", transcript)
        summary = [f"Summary: the agent completed {len(tasks)} routine tasks."]
        if policy:
            rule = policy.group(1).strip()
            # mid budgets: keep about half the rules intact, chosen deterministically
            keep_mid = sum(map(ord, rule)) % 2 == 0
            if max_tokens >= 300 or (max_tokens >= 200 and keep_mid):
                summary.append(f"Standing policy: {rule}")
            elif max_tokens >= 120:
                # degraded residue: keeps the shape, loses the predicate
                summary.append("The operator asked the agent to be careful with sensitive actions.")
        return "\n".join(summary)
