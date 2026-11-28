"""Claude client, through the official Anthropic SDK.

Three differences from the OpenAI-compatible client affect the experiment:

- Current Claude models reject `temperature`, so it is ignored here. Repeated
  samples vary only through the model's default sampling. The noise floor measures this.
- Thinking cannot be turned off and counts toward `max_tokens`, so the harness's small
  caps would cut replies off. A generous floor is used instead, which leaves the word
  limit in the compactor's prompt as the only budget on summaries.
- Refusal fallbacks are deliberately not enabled: a fallback would silently swap the
  model under test partway through an experiment.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from .llm import Message

MAX_TOKENS_FLOOR = 16000
_usage_lock = threading.Lock()
USAGE = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
# Spending cap, set by the CLI from the config: cap and prices in USD, plus what earlier invocations of
# the same run already spent (from usage.jsonl). cap = 0 means unset; the CLI refuses real runs then.
BUDGET = {"cap_usd": 0.0, "input_per_mtok": 0.0, "output_per_mtok": 0.0, "spent_before_usd": 0.0}


class BudgetExceeded(RuntimeError):
    """The run reached its spending cap. Calls already in flight may finish, so the overshoot is
    at most one call per worker."""


def spent_usd() -> float:
    return (BUDGET["spent_before_usd"] + USAGE["input_tokens"] * BUDGET["input_per_mtok"] / 1e6
            + USAGE["output_tokens"] * BUDGET["output_per_mtok"] / 1e6)


@dataclass
class AnthropicClient:
    model: str
    api_key: str
    effort: str | None = None
    request_delay: float = 0.0
    max_retries: int = 6
    name: str = field(init=False)

    def __post_init__(self) -> None:
        import anthropic  # imported lazily so other providers do not need it

        # Effort changes behaviour, so it is part of the identity recorded with each run.
        self.name = f"{self.model}@{self.effort}" if self.effort else self.model
        self._client = anthropic.Anthropic(api_key=self.api_key, max_retries=self.max_retries)

    def chat(self, messages: list[Message], *, temperature: float = 0.0,
             max_tokens: int = 512) -> str:
        if BUDGET["cap_usd"] and spent_usd() >= BUDGET["cap_usd"]:
            raise BudgetExceeded(f"spending cap of {BUDGET['cap_usd']:.2f} USD reached "
                                 f"(spent about {spent_usd():.2f} USD); nothing more is sent")
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        kwargs = {"system": system} if system else {}
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max(max_tokens, MAX_TOKENS_FLOOR),
            messages=[m for m in messages if m["role"] != "system"],
            **kwargs,
        )
        with _usage_lock:
            USAGE["calls"] += 1
            USAGE["input_tokens"] += response.usage.input_tokens
            USAGE["output_tokens"] += response.usage.output_tokens
        if self.request_delay:
            import time

            time.sleep(self.request_delay)
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        # The API rejects empty assistant turns, and this text is not a parseable action,
        # so the episode is counted as invalid rather than as safe.
        return text or f"[no text: stop_reason={response.stop_reason}]"
