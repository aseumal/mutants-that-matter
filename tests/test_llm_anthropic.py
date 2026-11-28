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


def test_spending_cap_stops_calls_before_sending(monkeypatch):
    import mtm.llm_anthropic as la

    monkeypatch.setattr(la, "USAGE", {"calls": 0, "input_tokens": 0, "output_tokens": 0})
    monkeypatch.setattr(la, "BUDGET", {"cap_usd": 1.0, "input_per_mtok": 1.0, "output_per_mtok": 5.0,
                                       "spent_before_usd": 0.0})
    big = NS(content=[NS(type="text", text="ok")], stop_reason="end_turn",
             usage=NS(input_tokens=600_000, output_tokens=100_000))   # 0.6 + 0.5 = 1.1 USD
    client = _client(big, _reply(NS(type="text", text="never sent")))
    assert client.chat([{"role": "user", "content": "U"}]) == "ok"
    assert la.spent_usd() == pytest.approx(1.1)
    with pytest.raises(la.BudgetExceeded):
        client.chat([{"role": "user", "content": "U"}])
    assert len(client._client.messages.calls) == 1             # the second request was never made


def test_real_runs_are_refused_without_a_cap(tmp_path, monkeypatch, capsys):
    from mtm.cli import main

    monkeypatch.delenv("MTM_MOCK", raising=False)
    monkeypatch.setenv("MTM_PROVIDER", "anthropic")
    cfg = tmp_path / "c.yaml"
    cfg.write_text("run_name: t\nrules: [R01]\nbudgets: [30]\ncycles: 1\n")
    assert main(["compact", "-c", str(cfg), "--runs-dir", str(tmp_path / "runs")]) == 2
    assert "spend_cap_usd" in capsys.readouterr().err
    cfg.write_text(cfg.read_text() + "spend_cap_usd: 5\nprice_input_per_mtok: 1\nprice_output_per_mtok: 5\n")
    (tmp_path / "runs" / "t").mkdir(parents=True, exist_ok=True)
    (tmp_path / "runs" / "t" / "usage.jsonl").write_text('{"cost_usd": 5.0}\n')
    assert main(["compact", "-c", str(cfg), "--runs-dir", str(tmp_path / "runs")]) == 2
    assert "already spent" in capsys.readouterr().err
