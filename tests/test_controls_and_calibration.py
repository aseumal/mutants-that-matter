import json

from mtm import freeze as fz
from mtm.analysis import analyze, calibrate, sensitivity
from mtm.audits import presence_audit
from mtm.experiment import Config, conditions_for, select_rules, stage_compact, stage_mutate, stage_replay
from mtm.llm import MockLLM
from mtm.mutants import DIRECTION, rule_mutants
from mtm.rules import load_rules


def test_every_rule_has_a_paraphrase_control():
    for r in load_rules():
        muts = {m.operator: m for m in rule_mutants(r)}
        assert muts["paraphrase"].direction == "equivalent"
        assert muts["paraphrase"].text != r.text
        if "drop_scope" in muts:
            assert muts["drop_scope"].direction == "strengthen"


def test_direction_covers_every_operator():
    from mtm.mutants import OPERATORS
    assert set(OPERATORS) <= set(DIRECTION)


def test_intact_repeat_is_identical_context():
    r = load_rules()[0]
    conds = dict(conditions_for(r, [], []))
    assert conds["intact_repeat"] == conds["intact"]


def test_presence_check_can_flag_a_harmless_paraphrase():
    # The keyword audit is deterministic, so this is a property of the audit itself.
    r = {x.id: x for x in load_rules()}["R11"]
    assert presence_audit(r, r.text)["presence_pass"]
    assert not presence_audit(r, r.paraphrase)["presence_pass"]


def _write(run, behavior, residues=(), mutants=()):
    run.mkdir(parents=True, exist_ok=True)
    for name, rows in (("behavior", behavior), ("residues", residues), ("mutants", mutants)):
        (run / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def _rows(rule, cond, probe, violations):
    return [{"rule_id": rule, "condition": cond, "probe_id": probe, "sample": i, "violated": v}
            for i, v in enumerate(violations)]


def test_calibration_picks_threshold_above_noise(tmp_path):
    # intact never violates; the repeat violates 1 of 3 samples on one probe (noise)
    b = []
    for rid in ("R01", "R02"):
        for p in ("P1", "P2"):
            b += _rows(rid, "intact", f"{rid}-{p}", [0, 0, 0])
            b += _rows(rid, "intact_repeat", f"{rid}-{p}", [1, 0, 0] if p == "P1" else [0, 0, 0])
    _write(tmp_path / "r", b)
    cal = calibrate(tmp_path / "r", grid=[0.3, 0.4, 0.5])
    assert cal["grid"][0]["repeat_false_alarm_rate"] == 1.0     # tau 0.3 counts noise as failure
    assert cal["recommended_tau"] == 0.4
    assert cal["noise_max_abs_diff"] == round(1 / 3, 3)


def test_controls_never_count_as_coupling_and_sensitivity_runs(tmp_path):
    cfg = Config(run_name="t", rules=["R01", "R11", "R15"], budgets=[60, 150], cycles=2,
                 turns_per_cycle=4, samples=1, temperature=0.0, workers=1)
    rules = select_rules(load_rules(), cfg)
    out = tmp_path / "t"
    out.mkdir()
    m = MockLLM()
    stage_compact(cfg, rules, m, out)
    stage_mutate(cfg, rules, out)
    stage_replay(cfg, rules, m, out)
    res = analyze(out)
    assert all(g["operator"] not in ("paraphrase", "drop_scope") for g in res["grid"])
    assert res["probes"] and {"excluded", "failing"} <= set(res["probes"][0])
    s = res["summary"]
    assert s["repeat_false_alarm_rate"] == 0.0
    assert s["paraphrase_behavior_false_alarm_rate"] is not None
    sens = sensitivity(out, grid=[0.3, 0.5, 0.7])
    assert [r["tau"] for r in sens] == [0.3, 0.5, 0.7]


def test_freeze_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(fz, "FROZEN", tmp_path / "FROZEN.json")
    assert fz.check()["frozen"] is False
    rec = fz.freeze("test")
    assert "data/rules.yaml" in rec["hashes"]
    assert fz.check()["matches"] is True
    data = json.loads((tmp_path / "FROZEN.json").read_text())
    data["hashes"]["data/rules.yaml"] = "0" * 16
    (tmp_path / "FROZEN.json").write_text(json.dumps(data))
    st = fz.check()
    assert st["matches"] is False and st["changed"] == ["data/rules.yaml"]
