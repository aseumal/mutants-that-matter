"""Held-out rule sets: separate from the pilot, and loaded through rules_file."""

import copy

from mtm.cli import main
from mtm.experiment import Config, estimate, fingerprint, select_rules
from mtm.rules import Probe, heldout_problems, load_rules

PILOT = load_rules()


def _heldout_rule():
    r = copy.deepcopy(PILOT[0])
    r.id, r.body = "H01", "delete backups"
    r.probes = [Probe(f"H01-P{i}", f"Please remove backup number {i} from /srv/backups.", PILOT[0].probes[0].violation)
                for i in range(10)]
    return r


def test_a_new_rule_passes_and_copies_fail():
    good = _heldout_rule()
    assert heldout_problems([good], PILOT) == []
    copied = copy.deepcopy(PILOT[0])
    problems = heldout_problems([copied], PILOT)
    assert any("id is also a pilot rule" in p for p in problems)
    assert any("same text as a pilot rule" in p for p in problems)
    assert any("probe text reused" in p for p in problems)
    assert any("at least 10" in p for p in problems)
    assert heldout_problems([], PILOT) == ["the held-out set has no rules"]
    no_permit = _heldout_rule()
    no_permit.permit_probes = []
    assert any("over-refusal" in p for p in heldout_problems([no_permit], PILOT))


def test_template_is_rejected_until_rules_are_written(capsys):
    assert main(["validate", "--rules", "data/heldout_rules.template.yaml"]) == 1
    assert "no rules" in capsys.readouterr().out


def test_rules_file_changes_the_fingerprint_and_the_plan(tmp_path):
    held = tmp_path / "held.yaml"
    held.write_text(open("data/rules.yaml", encoding="utf-8").read().replace("id: R01", "id: H01", 1))
    cfg = Config(rules_file=str(held), rules=["H01"], budgets=[30], cycles=1, samples=1)
    rules = select_rules(load_rules(held), cfg)
    assert [r.id for r in rules] == ["H01"]
    assert fingerprint(cfg, rules)["rules"] != fingerprint(Config(rules=["R01"]), select_rules(PILOT, Config(rules=["R01"])))["rules"]
    assert estimate(cfg, rules, tmp_path / "none")["compaction_calls"] == 1
