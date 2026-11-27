from types import SimpleNamespace as NS

import pytest

pytest.importorskip("anthropic")

from mtm.agent import run_episode  # noqa: E402
from mtm.llm import client_from_env  # noqa: E402
from mtm.llm_anthropic import MAX_TOKENS_FLOOR, AnthropicClient  # noqa: E402


class _Messages:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.replies.pop(0)


def _reply(*blocks, stop="end_turn"):
    return NS(content=list(blocks), stop_reason=stop, usage=NS(input_tokens=1, output_tokens=1))


def _client(*replies, effort="low"):
    client = AnthropicClient(model="claude-x", api_key="k", effort=effort)
    client._client = NS(messages=_Messages(replies))
    return client


def test_request_shape():
    client = _client(_reply(NS(type="thinking", thinking=""), NS(type="text", text="hi")))
    out = client.chat([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
                      temperature=0.7, max_tokens=100)
    sent = client._client.messages.calls[0]
    assert out == "hi"
    assert sent["system"] == "S" and sent["messages"] == [{"role": "user", "content": "U"}]
    assert "temperature" not in sent
    assert sent["max_tokens"] == MAX_TOKENS_FLOOR
    assert sent["output_config"] == {"effort": "low"}
    assert client.name == "claude-x@low"


def test_refusal_becomes_an_invalid_episode_not_a_safe_one():
    refused = _reply(stop="refusal")
    client = _client(refused, refused, refused)
    ep = run_episode(client, [{"role": "system", "content": "S"}], "do it", max_steps=3)
    assert ep.parse_errors == 3 and not ep.calls and ep.final is None
    # no empty assistant turn is ever sent back to the API
    for call in client._client.messages.calls:
        assert all(m["content"] for m in call["messages"])


def test_anthropic_key_cannot_reach_another_provider(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MTM_MOCK", raising=False)
    monkeypatch.delenv("MTM_PROVIDER", raising=False)
    monkeypatch.setenv("MTM_API_KEY", "sk-ant-x")
    monkeypatch.setenv("MTM_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("MTM_AGENT_MODEL", "m")
    with pytest.raises(RuntimeError, match="Anthropic key"):
        client_from_env("agent")
    monkeypatch.setenv("MTM_PROVIDER", "anthropic")
    assert isinstance(client_from_env("agent"), AnthropicClient)
