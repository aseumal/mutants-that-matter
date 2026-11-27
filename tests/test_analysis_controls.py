import json

import pytest

from mtm.analysis import analyze, chance_overlap, violation_rates
from mtm.cli import main
from mtm.experiment import Config, RunMismatch, check_manifest, conditions_for, stage_mutate
from mtm.rules import load_rules


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _behavior(rule, condition, failing, probes=("p1", "p2", "p3")):
    return [{"rule_id": rule, "condition": condition, "probe_id": p, "sample": 0,
             "violated": p in failing, "calls": [], "final": "ok", "parse_errors": 0}
            for p in probes]


def _mutant(rule, operator, family="rule"):
    return {"rule_id": rule, "operator": operator, "family": family, "text": "x",
            "condition": f"mutant:{operator}", "presence_pass": True,
            "exact_present": False, "keyword_recall": 1.0}


def _residue(rule, condition="residue:b60:c1"):
    return {"rule_id": rule, "condition": condition, "budget_words": 60, "cycle": 1, "summary": "s",
            "presence_pass": True, "exact_present": False, "keyword_recall": 1.0}


def _run(tmp_path, behavior, mutants, residues):
    _write(tmp_path / "behavior.jsonl", behavior)
    _write(tmp_path / "mutants.jsonl", mutants)
    _write(tmp_path / "residues.jsonl", residues)
    return analyze(tmp_path)


def test_manifest_refuses_another_model(tmp_path):
    check_manifest(tmp_path, agent="model-a", compactor="model-a")
    check_manifest(tmp_path, agent="model-a")  # same model resumes
    with pytest.raises(RunMismatch):
        check_manifest(tmp_path, agent="model-b")
    with pytest.raises(RunMismatch):
        check_manifest(tmp_path, compactor="model-b")


def test_mock_run_is_kept_apart_from_real_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("MTM_MOCK", "1")
    cfg = tmp_path / "c.yaml"
    cfg.write_text("run_name: t\nrules: [R01]\nbudgets: [60]\ncycles: 1\nturns_per_cycle: 2\n"
                   "samples: 1\nworkers: 1\nintact_repeats: 1\n")
    runs = tmp_path / "runs"
    assert main(["pilot", "--mock", "-c", str(cfg), "--runs-dir", str(runs)]) == 0
    assert not (runs / "t").exists()
    out = runs / "t-mock"
    manifest = json.loads((out / "run.json").read_text())
    assert {k: manifest[k] for k in ("compactor", "agent")} == {"compactor": "mock-compactor", "agent": "mock-agent"}
    assert set(manifest["fingerprint"]) >= {"rules", "probes", "graders", "filler", "prompts", "config", "code"}
    # a repeat run adds no rows
    before = (out / "behavior.jsonl").read_text()
    assert main(["pilot", "--mock", "-c", str(cfg), "--runs-dir", str(runs)]) == 0
    assert (out / "behavior.jsonl").read_text() == before


def test_unparseable_episodes_are_not_counted_as_safe(tmp_path):
    invalid = {"rule_id": "R01", "condition": "c", "probe_id": "p1", "sample": 1, "violated": False,
               "calls": [], "final": None, "parse_errors": 3}
    violated = {"rule_id": "R01", "condition": "c", "probe_id": "p1", "sample": 0, "violated": True,
                "calls": [{"tool": "x"}], "final": None, "parse_errors": 1}
    assert violation_rates([invalid, violated])[("R01", "c")]["p1"] == 1.0
    result = _run(tmp_path, _behavior("R01", "intact", set()) + [invalid], [], [])
    assert result["summary"]["invalid_episodes"] == 1


class _ChangingModel:
    name = "changing"
    calls = 0

    def chat(self, messages, **kw):
        type(self).calls += 1
        return json.dumps([f"Reworded version {self.calls}."])


def test_llm_mutants_stay_identical_across_reruns(tmp_path):
    rules = [r for r in load_rules() if r.id == "R01"]
    cfg = Config(llm_mutants=1)
    stage_mutate(cfg, rules, tmp_path, _ChangingModel())
    first = (tmp_path / "mutants.jsonl").read_text()
    assert "Reworded version 1." in first
    stage_mutate(cfg, rules, tmp_path, _ChangingModel())
    assert (tmp_path / "mutants.jsonl").read_text() == first
    assert _ChangingModel.calls == 1


def test_control_operator_does_not_count_as_detection(tmp_path):
    behavior = (_behavior("R01", "intact", set()) + _behavior("R01", "absent", set())
                + _behavior("R01", "residue:b60:c1", {"p1"})
                + _behavior("R01", "mutant:drop_scope", {"p1"})
                + _behavior("R01", "mutant:hedge", set()))
    result = _run(tmp_path, behavior, [_mutant("R01", "drop_scope"), _mutant("R01", "hedge")], [_residue("R01")])
    assert result["summary"]["residue_detection_rate"] == 0.0
    ops = {r["operator"]: r for r in result["operators"]}
    assert ops["drop_scope"]["control"] and ops["drop_scope"]["behavior_kill_rate"] == 1.0

    behavior += _behavior("R01", "mutant:weaken_modal", {"p1"})
    result = _run(tmp_path, behavior, [_mutant("R01", "drop_scope"), _mutant("R01", "weaken_modal")],
                  [_residue("R01")])
    assert result["summary"]["residue_detection_rate"] == 1.0


@pytest.mark.parametrize("a,b,n,expected", [
    (1, 1, 3, 1 / 3), (1, 2, 3, 2 / 3), (2, 2, 3, 1.0), (3, 1, 3, 1.0), (0, 2, 3, 0.0), (1, 1, 0, 0.0),
])
def test_chance_overlap(a, b, n, expected):
    assert chance_overlap(a, b, n) == pytest.approx(expected)


def test_chance_and_absent_baselines(tmp_path):
    behavior = (_behavior("R01", "intact", set()) + _behavior("R01", "absent", {"p2"})
                + _behavior("R01", "residue:b60:c1", {"p1"})
                + _behavior("R01", "mutant:hedge", {"p1"}))
    result = _run(tmp_path, behavior, [_mutant("R01", "hedge")], [_residue("R01")])
    hedge = result["operators"][0]
    assert hedge["coupling_rate"] == 1.0
    assert hedge["chance_coupling"] == pytest.approx(0.333)
    assert result["summary"]["absent_coupling_rate"] == 0.0
    assert result["summary"]["absent_coupling_chance"] == pytest.approx(0.333)
    assert result["summary"]["residue_detection_chance"] == pytest.approx(0.333)


def test_noise_floor_does_not_change_exclusions(tmp_path):
    behavior = (_behavior("R01", "intact", {"p3"})  # p3 fails with the intact rule: excluded
                + _behavior("R01", "intact:rep1", {"p1"})
                + _behavior("R01", "intact:rep2", set()))
    summary = _run(tmp_path, behavior, [], [])["summary"]
    assert summary["probes_excluded_intact_fails"] == 1
    assert summary["noise_floor"] == 0.25  # 1 of 2 eligible probes x 2 repeats


def test_intact_repeats_add_conditions():
    rule = load_rules()[0]
    keys = [k for k, _ in conditions_for(rule, [], [], intact_repeats=2)]
    assert keys == ["intact", "absent", "intact:rep1", "intact:rep2"]


def test_false_alarm_counts_harmless_residues_that_fail_presence(tmp_path):
    missing = _residue("R01") | {"presence_pass": False}
    behavior = _behavior("R01", "intact", set()) + _behavior("R01", "residue:b60:c1", set())
    summary = _run(tmp_path, behavior, [], [missing])["summary"]
    assert summary["false_alarm_rate"] == 1.0
    assert summary["false_assurance_rate"] is None
