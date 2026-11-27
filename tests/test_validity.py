import json

import pytest

from mtm.validity import (Run, analyze_validity, chance_coupling, chance_jaccard, difficulty_chance,
                          partial_correlation, selection_preview,
                          score_groups, set_metrics, spearman, suite_analysis, write_validity_report,
                          zero_event_upper_bound)


def test_set_metrics_penalize_a_mutant_that_fails_everything():
    broad = set_metrics(set("abcdefghij"), {"c", "f"})
    assert broad == {"coupled": 1.0, "recall": 1.0, "precision": 0.2, "jaccard": 0.2}
    assert set_metrics({"c", "f"}, {"c", "f"})["jaccard"] == 1.0
    assert set_metrics(set(), {"c"}) == {"coupled": 0.0, "recall": 0.0, "precision": 0.0, "jaccard": 0.0}


def test_chance_measures_match_hand_computed_cases():
    assert chance_coupling(1, 1, 3) == pytest.approx(1 / 3)
    assert chance_coupling(2, 2, 3) == 1.0
    assert chance_jaccard(1, 1, 3) == pytest.approx(1 / 3)
    # sizes 2 and 2 of 3: identical with probability 1/3 (J=1), else one shared of three (J=1/3)
    assert chance_jaccard(2, 2, 3) == pytest.approx(1 / 3 + 2 / 3 * 1 / 3)
    assert chance_jaccard(3, 3, 3) == 1.0
    assert chance_jaccard(0, 2, 3) == 0.0


def test_zero_event_bound():
    assert zero_event_upper_bound(29) == pytest.approx(0.119, abs=0.001)
    assert zero_event_upper_bound(0) is None


def test_spearman_handles_ties_and_constants():
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 2, 2], [1, 1, 2, 2]) == pytest.approx(1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) is None


def test_partial_correlation():
    assert partial_correlation(0.5, 0.0, 0.0) == pytest.approx(0.5)
    assert partial_correlation(0.8, 0.8, 1.0) is None
    assert partial_correlation(None, 0.1, 0.1) is None


def _write_run(path, table):
    """table: {(rule, condition): set of failing probes}. Three probes per rule, three samples."""
    rows, mutants, residues = [], [], []
    for (rule, cond), failing in table.items():
        for p in ("P1", "P2", "P3"):
            for s in range(3):
                rows.append({"rule_id": rule, "condition": cond, "probe_id": f"{rule}-{p}", "sample": s,
                             "violated": p in failing, "calls": [], "final": "x", "parse_errors": 0})
        if cond.startswith("residue"):
            residues.append({"rule_id": rule, "condition": cond, "presence_pass": False, "summary": ""})
        elif cond not in ("intact", "absent"):
            op = cond.split(":")[1]
            mutants.append({"rule_id": rule, "condition": cond, "operator": op,
                            "family": "llm" if op.startswith("llm") else "rule", "presence_pass": True})
    for name, data in (("behavior", rows), ("mutants", mutants), ("residues", residues)):
        (path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data))


def _table():
    table = {}
    for i, f in enumerate([{"P1"}, {"P2"}, {"P3"}, {"P1"}, {"P2"}, {"P3"}]):
        r = f"R{i}"
        table[(r, "intact")] = set()
        table[(r, "absent")] = {"P1", "P2", "P3"}
        table[(r, "mutant:flip_negation")] = {"P1", "P2", "P3"}
        table[(r, "mutant:drop_scope")] = set()
        table[(r, "mutant:hedge")] = f
        table[(r, "residue:b30:c1")] = f
    return table


def test_groups_never_count_the_control_or_absent_as_mutants(tmp_path):
    _write_run(tmp_path, _table())
    groups = score_groups(Run(tmp_path))
    assert not [c for c in groups["all_mutants"] if "drop_scope" in c[1] or c[1] == "absent"]
    assert all("flip_negation" in c[1] for c in groups["catastrophic_operators"])
    assert all("hedge" in c[1] for c in groups["decay_operators"])
    assert len(groups["rule_absent"]) == 6


def test_a_mutant_matching_each_residue_predicts_detection_where_deletion_cannot(tmp_path):
    _write_run(tmp_path, _table())
    fixed = suite_analysis(Run(tmp_path), n_suites=300, n_boot=5, boot_suites=30)["fixed_size"]
    # Each suite with one probe per rule: `hedge` fails the same probe as the residue, so
    # the two scores move together. Deleting the rule fails every probe, so it never varies.
    assert fixed["1_per_rule"]["decay_operators"]["rho"] == pytest.approx(1.0)
    assert fixed["1_per_rule"]["rule_absent"]["rho"] is None


def test_report_is_written_and_excess_can_be_negative(tmp_path):
    table = _table()
    _write_run(tmp_path, table)
    result = analyze_validity(tmp_path)
    assert result["false_assurance"] == {"observed": 0, "harmful_residues": 6, "upper_95": pytest.approx(0.459, abs=0.001),
                                         "upper_95_if_one_per_rule": pytest.approx(0.459, abs=0.001)}
    ops = {r["operator"]: r for r in result["operators"]}
    assert ops["hedge"]["jaccard"] == 1.0 and ops["hedge"]["coupling"] == 1.0
    assert ops["flip_negation"]["precision"] == pytest.approx(1 / 3, abs=0.001)
    # flip fails all three, so chance already says full coupling: no excess
    assert ops["flip_negation"]["excess_coupling_ci"] == [0.0, 0.0]
    assert ops["drop_scope"]["control"] is True and ops["drop_scope"]["coupling"] == 0.0
    assert write_validity_report(result, tmp_path).exists()


def test_difficulty_chance_reduces_to_uniform_and_rises_with_skew():
    even = {"a": 1.0, "b": 1.0, "c": 1.0}
    for a, b in [(1, 1), (1, 2), (2, 2)]:
        coupled, jaccard = difficulty_chance(a, b, even)
        assert coupled == pytest.approx(chance_coupling(a, b, 3))
        assert jaccard == pytest.approx(chance_jaccard(a, b, 3))
    skew = {"a": 8.0, "b": 1.0, "c": 1.0}
    assert difficulty_chance(1, 1, skew)[0] == pytest.approx(0.66)  # .8^2 + .1^2 + .1^2
    assert difficulty_chance(0, 1, skew) == (0.0, 0.0)


def test_selection_uses_each_strategys_own_conditions(tmp_path):
    _write_run(tmp_path, _table())
    out = selection_preview(Run(tmp_path), k=1, n_boot=20)["strategies"]
    assert out["random"]["detection"] == pytest.approx(1 / 3, abs=0.001)
    assert out["decay_operators"]["detection"] == 1.0   # hedge fails exactly the residue's probe
    assert out["rule_absent"]["detection"] == pytest.approx(1 / 3, abs=0.001)  # every probe ties
    assert out["decay_operators"]["minus_random_ci"][0] > 0
