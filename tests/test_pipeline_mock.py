from mtm.agent import parse_action
from mtm.analysis import analyze, cohens_kappa, ochiai, write_report
from mtm.experiment import Config, select_rules, stage_compact, stage_mutate, stage_replay
from mtm.llm import MockLLM
from mtm.rules import load_rules


def test_parse_action_variants():
    assert parse_action('{"tool": "deploy", "args": {"service": "api"}}')["tool"] == "deploy"
    assert parse_action('```json\n{"final": "no"}\n```')["final"] == "no"
    assert parse_action('I will do it: {"tool": "x", "args": {}} thanks')["tool"] == "x"
    assert parse_action("no json here") is None


def test_ochiai_and_kappa():
    assert ochiai({"a", "b"}, {"b"}) == 1 / 2 ** 0.5
    assert ochiai(set(), {"a"}) == 0.0
    assert cohens_kappa(["x", "y", "x"], ["x", "y", "x"]) == 1.0


def test_end_to_end_with_mock(tmp_path):
    cfg = Config(run_name="t", rules=["R01", "R11", "R15"], budgets=[60, 150], cycles=2,
                 turns_per_cycle=4, samples=1, temperature=0.0, workers=1)
    rules = select_rules(load_rules(), cfg)
    out = tmp_path / "t"
    out.mkdir()
    m = MockLLM()
    stage_compact(cfg, rules, m, out)
    stage_mutate(cfg, rules, out)
    stage_replay(cfg, rules, m, out)
    result = analyze(out)
    s = result["summary"]
    assert s["rules"] == 3 and s["residues"] == 12
    assert s["intact_violation_rate"] < s["absent_violation_rate"]
    assert s["harmful_residues"] > 0
    ops = {r["operator"]: r for r in result["operators"]}
    assert ops["weaken_modal"]["presence_pass_rate"] == 1.0   # presence check is fooled
    assert (write_report(result, out)).exists()

    # resumability: a second replay adds nothing
    before = (out / "behavior.jsonl").read_text().count("\n")
    stage_replay(cfg, rules, m, out)
    assert (out / "behavior.jsonl").read_text().count("\n") == before
