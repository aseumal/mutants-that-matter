"""Exploratory: is text-based prediction of violation-inducing residues just reading a shortcut?

Every model is the RQ4 logistic regression (balanced classes, fixed untuned settings), scored on
the real residues with leave-one-rule-out folds unless stated, labels from replay only. The pilot
run is read, never written; results go to paper/analysis/results/shortcuts.json.

  length only, keyword recall only, keyword + length    low-complexity baselines
  rule / closest sentence / whole summary / cosines     which part of the RQ4 features carries the signal
  deletion + intact training                            the synthetic model from Table 6
  leave one rule category out                           transfer across rule families
  mutant to mutant                                      are synthetic faults learnable at all?
  residues that keep rule words                         the subset where missing words cannot explain a score
Differences against keyword recall are paired bootstraps over rules. Written on 2026-11-28 after
all pilot results were known.

    python paper/analysis/shortcut_diagnostics.py     (from the repository root)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deletion_transfer import paired  # noqa: E402
from dl_transfer import RUN, SEED, Item, load  # noqa: E402

from mtm import predict as P  # noqa: E402
from mtm.experiment import policy_summary  # noqa: E402
from mtm.rules import load_rules  # noqa: E402

OUT = Path("paper/analysis/results/shortcuts.json")
H = 384  # MiniLM width; RQ4 features are rule | closest sentence | |diff| | product | whole text | 2 cosines
SLICES = {"rule embedding only": slice(0, H), "closest sentence only": slice(H, 2 * H),
          "whole summary only": slice(4 * H, 5 * H), "two cosines only": slice(5 * H, 5 * H + 2)}


def lor(x_tr, y_tr, g_tr, x_te, g_te, model="logreg"):
    out = np.zeros(len(g_te))
    for r in sorted(set(g_te)):
        tr, te = g_tr != r, g_te == r
        assert r not in set(g_tr[tr])
        out[te] = P._fit_score(model, x_tr[tr], y_tr[tr], x_te[te], SEED)
    return out


def main() -> None:
    rules = {r.id: r for r in load_rules()}
    residues, mutants, absent = load()
    intact = [Item(a.rule_id, a.rule, policy_summary(a.rule), 0) for a in absent]
    ex_res, ex_mut = P.load_examples(RUN, rules)
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(P.ENCODER_NAME, device="cpu")
    encode = lambda texts: np.asarray(st.encode(texts, normalize_embeddings=True, show_progress_bar=False))
    feats = lambda items: P.featurize([P.Example(i.rule_id, i.rule, i.text, bool(i.label), 0.0) for i in items], encode)[0]
    x_res, x_mut, x_syn = feats(residues), feats(mutants), feats(absent + intact)
    assert x_res.shape[1] == 5 * H + 2
    y = np.array([i.label for i in residues])
    y_mut = np.array([m.label for m in mutants])
    y_syn = np.array([a.label for a in absent] + [0] * len(intact))
    g = np.array([i.rule_id for i in residues])
    g_mut = np.array([m.rule_id for m in mutants])
    g_syn = np.array([a.rule_id for a in absent] + [i.rule_id for i in intact])
    kw = np.array([e.keyword_recall for e in ex_res])
    words = np.array([len(i.text.split()) for i in residues], dtype=float)
    assert len(kw) == len(residues)

    scores = {"keyword recall (1 - recall)": 1 - kw}
    simple = {"summary length only": words[:, None], "keyword recall only": kw[:, None],
              "keyword + length": np.c_[kw, words]}
    for name, x in simple.items():
        scores[name] = lor(x, y, g, x, g)
    for name, sl in SLICES.items():
        scores[name] = lor(x_res[:, sl], y, g, x_res[:, sl], g)
    scores["full RQ4 features (reference)"] = lor(x_res, y, g, x_res, g)
    scores["deletion + intact training"] = lor(x_syn, y_syn, g_syn, x_res, g)

    cats = np.array([rules[r].category for r in g])
    loco = np.zeros(len(y))
    for c in sorted(set(cats)):
        te = cats == c
        loco[te] = P._fit_score("logreg", x_res[~te], y[~te], x_res[te], SEED)
    scores["full features, leave one category out"] = loco

    mut_scores = lor(x_mut, y_mut, g_mut, x_mut, g_mut)
    keeps_words = kw > 0
    ref = scores["keyword recall (1 - recall)"]
    report = {"residues": len(y), "violation_inducing": int(y.sum()),
              "keep_rule_words": {"residues": int(keeps_words.sum()), "violation_inducing": int(y[keeps_words].sum())},
              "mutant_to_mutant": {"mutants": len(y_mut), "killed": int(y_mut.sum()),
                                   "auroc": round(float(roc_auc_score(y_mut, mut_scores)), 3),
                                   **{k: v for k, v in P.metrics(y_mut, mut_scores, list(g_mut), seed=SEED).items()
                                      if k == "auroc_ci"}},
              "models": {}}
    for name, s in scores.items():
        m = P.metrics(y, s, list(g), seed=SEED)
        sub = keeps_words & np.isin(y, [0, 1])
        m["auroc_keep_rule_words"] = (round(float(roc_auc_score(y[sub], s[sub])), 3)
                                      if len(set(y[sub].tolist())) == 2 else None)
        m["minus_keyword_ci"] = None if name.startswith("keyword recall (") else paired(y, list(g), s, ref)
        report["models"][name] = m
        print(f"{name}: AUROC {m['auroc']} {m['auroc_ci']} | vs keyword {m['minus_keyword_ci']} | "
              f"keeps-words subset {m['auroc_keep_rule_words']}", flush=True)
    print("mutant to mutant:", report["mutant_to_mutant"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
