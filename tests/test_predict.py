import csv
import hashlib
import json

import numpy as np
import pytest

pytest.importorskip("sklearn")

from mtm.analysis import label_summary  # noqa: E402
from mtm.predict import (Example, evaluate, load_examples, out_of_fold, rule_folds,  # noqa: E402
                         sentences, train_on_mutants, write_predict_report)
from mtm.experiment import Config, select_rules, stage_compact, stage_mutate, stage_replay  # noqa: E402
from mtm.llm import MockLLM  # noqa: E402
from mtm.rules import load_rules  # noqa: E402


def fake_encoder(texts):
    """Normalized bag of hashed words: deterministic, and needs no model download."""
    out = np.zeros((len(texts), 64))
    for i, text in enumerate(texts):
        for word in text.lower().split():
            out[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def _examples(n_rules=8):
    out = []
    for r in range(n_rules):
        rule = f"Never touch resource{r} outside the sandbox."
        out.append(Example(f"R{r}", rule, f"Work went fine. {rule}", False, 1.0))
        out.append(Example(f"R{r}", rule, "Work went fine. Be careful generally.", True, 0.0))
    return out


def test_folds_never_share_a_rule():
    groups = ["R1", "R1", "R2", "R3", "R3", "R3"]
    folds = list(rule_folds(groups))
    assert len(folds) == 3
    for train, test in folds:
        assert not {groups[i] for i in train} & {groups[i] for i in test}


def test_held_out_rule_cannot_influence_its_own_score():
    """Changing a held-out rule's labels must not change the score it receives."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(12, 4))
    y = np.array([0, 1] * 6)
    groups = ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E", "F", "F"]
    base = out_of_fold("logreg", x, y, groups)
    flipped = y.copy()
    flipped[:2] = 1 - flipped[:2]
    assert out_of_fold("logreg", x, flipped, groups)[:2] == pytest.approx(base[:2])


def test_mutant_training_excludes_the_tested_rule():
    rng = np.random.default_rng(0)
    x_mut = rng.normal(size=(12, 4))
    y_mut = np.array([0, 1] * 6)
    mut_groups = ["A"] * 4 + ["B"] * 4 + ["C"] * 4
    x_res = rng.normal(size=(2, 4))
    base = train_on_mutants(x_mut, y_mut, mut_groups, x_res, ["A", "A"])
    changed = y_mut.copy()
    changed[:4] = 1 - changed[:4]  # labels of rule A's own mutants
    assert train_on_mutants(x_mut, changed, mut_groups, x_res, ["A", "A"]) == pytest.approx(base)


def test_sentences_splits_summaries():
    assert sentences("One done. Two done!\nThree") == ["One done.", "Two done!", "Three"]
    assert sentences("") == [""]


def test_evaluate_finds_an_easy_signal_and_reports(tmp_path):
    examples = _examples()
    result = evaluate(examples, examples, fake_encoder)
    assert result["examples"] == 16 and result["positives"] == 8 and result["feasibility_only"]
    assert result["models"]["keyword_recall"]["auroc"] == 1.0
    assert result["models"]["closest_sentence_similarity"]["auroc"] == 1.0
    assert result["models"]["logreg_embeddings"]["auroc"] > 0.8
    assert {"mlp_embeddings", "logreg_trained_on_mutants"} <= set(result["models"])
    text = write_predict_report(result, tmp_path).read_text()
    assert "Feasibility only" in text
    assert json.loads((tmp_path / "rq4.json").read_text())["positives"] == 8


def test_single_class_is_reported_not_crashed():
    examples = [e for e in _examples() if not e.label]
    result = evaluate(examples, [], fake_encoder)
    assert result["models"]["logreg_embeddings"]["auroc"] is None


@pytest.fixture
def mock_run(tmp_path):
    cfg = Config(run_name="t", rules=["R01", "R11", "R15"], budgets=[60, 150], cycles=2,
                 turns_per_cycle=4, samples=1, temperature=0.0, workers=1)
    rules = select_rules(load_rules(), cfg)
    model = MockLLM()
    stage_compact(cfg, rules, model, tmp_path)
    stage_mutate(cfg, rules, tmp_path)
    stage_replay(cfg, rules, model, tmp_path)
    return tmp_path


def test_examples_come_from_replay_labels(mock_run):
    residues, mutants = load_examples(mock_run, {r.id: r for r in load_rules()})
    assert len(residues) == 12 and mutants
    assert any(e.label for e in residues)
    assert all("Standing operator policy" in e.text for e in mutants)


def test_label_summary(mock_run, tmp_path):
    from mtm.analysis import blinded_ids, export_labels, kappa_from_files
    from mtm.experiment import read_jsonl
    from mtm.rules import load_rules

    sheet = export_labels(mock_run, {r.id: r for r in load_rules()}, tmp_path / "sheet.csv")
    with open(sheet, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    # Blinded: nothing on the sheet reveals budget, cycle or the replay result.
    assert not {"condition", "harmful", "behavioral_label", "presence_pass"} & set(rows[0])
    assert all(r["summary"] and r["original_rule"] for r in rows)
    assert blinded_ids(read_jsonl(mock_run / "residues.jsonl")) == blinded_ids(read_jsonl(mock_run / "residues.jsonl"))

    good = {"surface_presence": "absent", "semantic_preservation": "absent", "obligation_strength": "not_applicable",
            "scope": "not_applicable", "exceptions": "not_applicable", "contradiction": "no",
            "overall_damage": "deletion", "damage_type": "Deleted", "annotation_confidence": "high"}
    rows[0].update(good)
    rows[1].update(good, semantic_preservation="intact", surface_presence="present", damage_type="none",
                   overall_damage="intact", obligation_strength="preserved", scope="preserved", exceptions="preserved")
    rows[2].update(good, damage_type="made_up")
    path = tmp_path / "labels.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
        w.writerow({**rows[0], "residue_id": "r999"})
    result = label_summary(path, mock_run)
    assert result["labelled"] == 3 and result["unmatched"] == 1
    assert result["damage_type"]["deleted"]["residues"] == 1
    assert result["invalid"] == [f"{rows[2]['residue_id']}.damage_type='made_up'"]
    assert kappa_from_files(path, path) == 1.0
