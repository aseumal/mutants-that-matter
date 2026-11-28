"""RQ4: predict from text alone whether a compacted rule still works.

Each example pairs a rule with the summary text the agent would see. The label is
whether replay found the condition harmful, so it comes only from deterministic
grading. Rules appear in several examples, so every split holds out whole rules.

Requires the optional dependencies: pip install -e ".[ml]"
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .analysis import analyze
from .experiment import policy_summary, read_jsonl

ENCODER_NAME = "all-MiniLM-L6-v2"
FEW_POSITIVES = 15  # below this, report feasibility rather than performance

Encoder = Callable[[list[str]], np.ndarray]


@dataclass
class Example:
    rule_id: str
    rule: str
    text: str
    label: bool
    keyword_recall: float


def load_examples(run_dir: str | Path, rules_by_id: dict, tau: float = 0.5) -> tuple[list[Example], list[Example]]:
    """(residue examples, mutant examples). Mutants are framed as they were replayed."""
    run_dir = Path(run_dir)
    result = analyze(run_dir, tau)
    harmful = {(r["rule_id"], r["condition"]): r["harmful"] for r in result["residues"]}
    killed = {(m["rule_id"], m["condition"]): m["killed"] for m in result["mutants"]}

    residues = [Example(r["rule_id"], rules_by_id[r["rule_id"]].text, r["summary"],
                        harmful[(r["rule_id"], r["condition"])], r["keyword_recall"])
                for r in read_jsonl(run_dir / "residues.jsonl")
                if (r["rule_id"], r["condition"]) in harmful]
    mutants = [Example(m["rule_id"], rules_by_id[m["rule_id"]].text, policy_summary(m["text"]),
                       killed[(m["rule_id"], m["condition"])], m["keyword_recall"])
               for m in read_jsonl(run_dir / "mutants.jsonl")
               if (m["rule_id"], m["condition"]) in killed]
    return residues, mutants


# ------------------------------------------------------------------ features

def sentences(text: str) -> list[str]:
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    return parts or [text]


def default_encoder() -> Encoder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(ENCODER_NAME)
    return lambda texts: np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False))


def featurize(examples: list[Example], encode: Encoder) -> tuple[np.ndarray, np.ndarray]:
    """(feature matrix, similarity of each rule to its closest summary sentence).

    Summaries hold much more than the rule, so the closest sentence stands in for
    "the compacted rule", and the whole-summary embedding keeps the rest.
    """
    rows, closest = [], []
    for ex in examples:
        sents = sentences(ex.text)
        vecs = encode([ex.rule, ex.text] + sents)
        rule, whole, sent_vecs = vecs[0], vecs[1], vecs[2:]
        sims = sent_vecs @ rule
        best = sent_vecs[int(np.argmax(sims))]
        closest.append(float(sims.max()))
        rows.append(np.concatenate([rule, best, np.abs(rule - best), rule * best, whole,
                                    [float(sims.max()), float(whole @ rule)]]))
    return np.vstack(rows), np.array(closest)


# ------------------------------------------------------------------ evaluation

def rule_folds(groups: list[str]) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Leave-one-rule-out: no rule is ever on both sides of a split."""
    from sklearn.model_selection import LeaveOneGroupOut

    yield from LeaveOneGroupOut().split(np.zeros(len(groups)), groups=groups)


def _new_model(kind: str, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "logreg":
        clf = LogisticRegression(C=0.01, class_weight="balanced", max_iter=2000)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(32,), alpha=1.0, max_iter=1000, random_state=seed)
    return make_pipeline(StandardScaler(), clf)


def _fit_score(kind: str, x_train, y_train, x_test, seed: int) -> np.ndarray:
    if len(set(y_train.tolist())) < 2:  # nothing to learn from this fold
        return np.full(len(x_test), float(y_train.mean()))
    return _new_model(kind, seed).fit(x_train, y_train).predict_proba(x_test)[:, 1]


def out_of_fold(kind: str, x: np.ndarray, y: np.ndarray, groups: list[str], seed: int = 7) -> np.ndarray:
    scores = np.zeros(len(y))
    for train, test in rule_folds(groups):
        scores[test] = _fit_score(kind, x[train], y[train], x[test], seed)
    return scores


def train_on_mutants(x_mut, y_mut, groups_mut: list[str], x_res, groups_res: list[str],
                     kind: str = "logreg", seed: int = 7) -> np.ndarray:
    """Score each residue with a model fitted on mutants of the other rules only."""
    scores = np.zeros(len(groups_res))
    res_rules = np.array(groups_res)
    mut_rules = np.array(groups_mut)
    for rule in sorted(set(groups_res)):
        test = res_rules == rule
        train = mut_rules != rule
        scores[test] = _fit_score(kind, x_mut[train], y_mut[train], x_res[test], seed)
    return scores


def metrics(y: np.ndarray, scores: np.ndarray, groups: list[str], n_boot: int = 1000, seed: int = 7) -> dict:
    """AUROC and AUPRC with a 95% bootstrap interval that resamples whole rules."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    if len(set(y.tolist())) < 2:
        return {"auroc": None, "auroc_ci": None, "auprc": None, "auprc_ci": None}
    rng = np.random.default_rng(seed)
    by_rule = {r: np.flatnonzero(np.array(groups) == r) for r in sorted(set(groups))}
    names = list(by_rule)
    roc, pr = [], []
    for _ in range(n_boot):
        idx = np.concatenate([by_rule[names[i]] for i in rng.integers(len(names), size=len(names))])
        if len(set(y[idx].tolist())) < 2:
            continue
        roc.append(roc_auc_score(y[idx], scores[idx]))
        pr.append(average_precision_score(y[idx], scores[idx]))
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] if v else None
    return {"auroc": round(float(roc_auc_score(y, scores)), 3), "auroc_ci": ci(roc),
            "auprc": round(float(average_precision_score(y, scores)), 3), "auprc_ci": ci(pr)}


def evaluate(residues: list[Example], mutants: list[Example], encode: Encoder, seed: int = 7) -> dict:
    y = np.array([e.label for e in residues], dtype=int)
    groups = [e.rule_id for e in residues]
    x, closest = featurize(residues, encode)
    recall = np.array([e.keyword_recall for e in residues])

    # Baselines need no fitting: less of the rule left should mean more harm.
    scores = {"keyword_recall": -recall, "closest_sentence_similarity": -closest,
              "logreg_embeddings": out_of_fold("logreg", x, y, groups, seed),
              "mlp_embeddings": out_of_fold("mlp", x, y, groups, seed)}
    if mutants:
        y_mut = np.array([e.label for e in mutants], dtype=int)
        x_mut, _ = featurize(mutants, encode)
        scores["logreg_trained_on_mutants"] = train_on_mutants(
            x_mut, y_mut, [e.rule_id for e in mutants], x, groups, seed=seed)

    return {
        "examples": len(y), "positives": int(y.sum()), "rules": len(set(groups)),
        "prevalence": round(float(y.mean()), 3) if len(y) else None,
        "feasibility_only": int(y.sum()) < FEW_POSITIVES or int(len(y) - y.sum()) < FEW_POSITIVES,
        "models": {name: metrics(y, s, groups, seed=seed) for name, s in scores.items()},
    }


def write_predict_report(result: dict, run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    (run_dir / "rq4.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    def cell(value, ci) -> str:
        return "n/a" if value is None else f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"

    lines = [f"# RQ4 report: {run_dir.name}", "",
             f"Residues: {result['examples']} from {result['rules']} rules; harmful: {result['positives']} "
             f"(an AUPRC of {result['prevalence']} is what a random ranking gets; an AUROC of 0.5 likewise).", ""]
    if result["feasibility_only"]:
        lines += [f"**Feasibility only:** fewer than {FEW_POSITIVES} examples in one class, "
                  "so these numbers do not support a performance claim.", ""]
    lines += ["| Model | AUROC [95% CI] | AUPRC [95% CI] |", "| --- | --- | --- |"]
    for name, m in result["models"].items():
        lines.append(f"| {name} | {cell(m['auroc'], m['auroc_ci'])} | {cell(m['auprc'], m['auprc_ci'])} |")
    lines += ["", "All models are scored on rules they were not trained on. Intervals resample whole rules. "
              "`logreg_trained_on_mutants` sees only mutants during training, so it tests whether "
              "what is learned from synthetic damage carries over to real compaction damage.", ""]
    path = run_dir / "rq4.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
