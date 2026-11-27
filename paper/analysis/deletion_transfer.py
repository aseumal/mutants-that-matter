"""Exploratory: which synthetic training data transfers to real compaction residues?

Every model is the RQ4 logistic regression (balanced class weights, fixed untuned settings)
on the RQ4 features, trained on synthetic examples from the other rules only and scored on
the real residues of the held-out rule (leave one rule out, 30 folds). Every example of the
held-out rule, synthetic or real, is excluded from its fold; this is asserted. Labels come
from replay only. Synthetic examples are framed exactly as they were replayed:
  deletion  the neutral summary, harmful if deleting the rule made an eligible probe fail
  intact    the neutral summary plus the rule, label 0 (intact failures are excluded by construction)
  mutant    the neutral summary plus the mutant, label = killed

Conditions, all with the same classifier, features and folds:
  deletion + intact                     58 examples per fold
  mutants                               about 313 per fold
  mutants, size-matched                 58 per fold, class ratio of deletion + intact, 20 draws
  mutants + deletion, weight w          share w of total sample weight on mutants (0 = deletion only)
  mutants, operators chosen per fold    the 3 operators whose mutants best overlap harmful residues
                                        of the training rules only (nested, no test-rule data)
  real residues                         within-domain reference (out of fold)
Differences against deletion + intact use paired bootstraps over rules.
Written on 2026-11-27 after all pilot results were known, so every result is exploratory.

    python paper/analysis/deletion_transfer.py     (from the repository root)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dl_transfer import RUN, SEED, Item, load  # noqa: E402

from mtm import predict as P  # noqa: E402
from mtm.analysis import CONTROL_OPERATORS, analyze  # noqa: E402
from mtm.experiment import policy_summary, read_jsonl  # noqa: E402
from mtm.validity import Run  # noqa: E402

N_DRAWS = 20
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def fit_score(x_tr, y_tr, x_te, w=None):
    if len(set(y_tr.tolist())) < 2:
        return np.full(len(x_te), float(y_tr.mean()))
    model = P._new_model("logreg", SEED)
    kw = {"logisticregression__sample_weight": w} if w is not None else {}
    return model.fit(x_tr, y_tr, **kw).predict_proba(x_te)[:, 1]


def paired(y, groups, a, b, n_boot=2000, seed=SEED):
    """Bootstrap over rules of AUROC(a) - AUROC(b), same resamples for both."""
    rng = np.random.default_rng(seed)
    by_rule = {r: np.flatnonzero(np.array(groups) == r) for r in sorted(set(groups))}
    names, diffs = list(by_rule), []
    for _ in range(n_boot):
        idx = np.concatenate([by_rule[names[i]] for i in rng.integers(len(names), size=len(names))])
        if len(set(y[idx].tolist())) == 2:
            diffs.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    return [round(float(np.percentile(diffs, 2.5)), 3), round(float(np.percentile(diffs, 97.5)), 3)]


def main() -> None:
    rules_by_id = {r.id: r for r in __import__("mtm.rules", fromlist=["load_rules"]).load_rules()}
    residues, mutants, absent = load()
    intact = [Item(a.rule_id, a.rule, policy_summary(a.rule), 0) for a in absent]
    killed = {(m["rule_id"], m["condition"]) for m in analyze(RUN)["mutants"]}
    mut_rows = [m for m in read_jsonl(RUN / "mutants.jsonl") if (m["rule_id"], m["condition"]) in killed]
    assert len(mut_rows) == len(mutants) and all(r["rule_id"] == m.rule_id for r, m in zip(mut_rows, mutants))
    ops = np.array([m["operator"] for m in mut_rows])

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(P.ENCODER_NAME, device="cpu")
    encode = lambda texts: np.asarray(st.encode(texts, normalize_embeddings=True, show_progress_bar=False))
    feats = lambda items: P.featurize([P.Example(i.rule_id, i.rule, i.text, bool(i.label), 0.0) for i in items], encode)[0]
    x_res, x_mut, x_syn = feats(residues), feats(mutants), feats(absent + intact)
    y = np.array([i.label for i in residues])
    y_mut = np.array([m.label for m in mutants])
    y_syn = np.array([a.label for a in absent] + [0] * len(intact))
    g_res = np.array([i.rule_id for i in residues])
    g_mut = np.array([m.rule_id for m in mutants])
    g_syn = np.array([a.rule_id for a in absent] + [i.rule_id for i in intact])
    rules = sorted(set(g_res))

    # nested operator choice: overlap of each operator's failing sets with harmful residues, training rules only
    run = Run(RUN)
    harmful = defaultdict(list)
    for k in run.harmful():
        harmful[k[0]].append(run.fails[k])
    op_hits = defaultdict(lambda: defaultdict(list))
    for m in mut_rows:
        if m["family"] == "rule" and m["operator"] not in CONTROL_OPERATORS and harmful[m["rule_id"]]:
            f = run.fails.get((m["rule_id"], m["condition"]), set())
            op_hits[m["rule_id"]][m["operator"]].append(float(any(f & h for h in harmful[m["rule_id"]])))

    scores: dict[str, np.ndarray] = {}
    fold_sizes = defaultdict(list)
    chosen_ops = {}
    rng = np.random.default_rng(SEED)

    def per_fold(name, build):
        out = np.zeros(len(residues))
        for r in rules:
            te = g_res == r
            x_tr, y_tr, w, rule_ids = build(r)
            assert r not in set(rule_ids), f"{name}: held-out rule {r} in training"
            out[te] = fit_score(x_tr, y_tr, x_res[te], w)
            fold_sizes[name].append(len(y_tr))
        scores[name] = out

    syn_fold = lambda r: (x_syn[g_syn != r], y_syn[g_syn != r], None, g_syn[g_syn != r])
    mut_fold = lambda r: (x_mut[g_mut != r], y_mut[g_mut != r], None, g_mut[g_mut != r])
    per_fold("deletion + intact", syn_fold)
    per_fold("mutants", mut_fold)

    draws = []
    for d in range(N_DRAWS):
        def matched(r, d=d):
            idx = np.flatnonzero(g_mut != r)
            n, pos_share = int((g_syn != r).sum()), y_syn[g_syn != r].mean()
            pos, neg = idx[y_mut[idx] == 1], idx[y_mut[idx] == 0]
            k_pos = min(len(pos), round(n * pos_share))
            pick = np.concatenate([rng.choice(pos, k_pos, replace=False), rng.choice(neg, n - k_pos, replace=False)])
            return x_mut[pick], y_mut[pick], None, g_mut[pick]
        per_fold(f"mutants, size-matched draw {d}", matched)
        draws.append(scores.pop(f"mutants, size-matched draw {d}"))
    fold_sizes["mutants, size-matched"] = fold_sizes.pop("mutants, size-matched draw 0")
    for d in range(1, N_DRAWS):
        fold_sizes.pop(f"mutants, size-matched draw {d}")
    draw_aurocs = [roc_auc_score(y, s) for s in draws]
    scores["mutants, size-matched (mean of draws)"] = np.mean(draws, axis=0)

    for w_share in WEIGHTS:
        def weighted(r, w_share=w_share):
            ms, ss = g_mut != r, g_syn != r
            x = np.vstack([x_mut[ms], x_syn[ss]])
            yy = np.concatenate([y_mut[ms], y_syn[ss]])
            w = np.concatenate([np.full(ms.sum(), w_share / ms.sum() if w_share else 0.0),
                                np.full(ss.sum(), (1 - w_share) / ss.sum() if w_share < 1 else 0.0)])
            keep = w > 0
            return x[keep], yy[keep], w[keep] * keep.sum(), np.concatenate([g_mut[ms], g_syn[ss]])[keep]
        per_fold(f"mutants + deletion, mutant weight {w_share}", weighted)

    def best_ops(r):
        totals = defaultdict(list)
        for rid, by_op in op_hits.items():
            if rid != r:
                for op, v in by_op.items():
                    totals[op] += v
        top = sorted(totals, key=lambda op: (-np.mean(totals[op]), op))[:3]
        chosen_ops[r] = top
        keep = (g_mut != r) & np.isin(ops, top)
        return x_mut[keep], y_mut[keep], None, g_mut[keep]
    per_fold("mutants, 3 operators chosen per fold", best_ops)

    scores["real residues (within domain)"] = P.out_of_fold("logreg", x_res, y, list(g_res), SEED)

    ref = scores["deletion + intact"]
    report = {"residues": len(y), "harmful": int(y.sum()),
              "class_balance": {"deletion + intact": round(float(y_syn.mean()), 3), "mutants": round(float(y_mut.mean()), 3)},
              "fold_training_sizes": {k: [min(v), max(v)] for k, v in fold_sizes.items()},
              "size_matched_draw_aurocs": [round(min(draw_aurocs), 3), round(float(np.mean(draw_aurocs)), 3), round(max(draw_aurocs), 3)],
              "operators_chosen": {op: sum(op in v for v in chosen_ops.values()) for op in sorted({o for v in chosen_ops.values() for o in v})},
              "models": {}}
    for name, s in scores.items():
        m = P.metrics(y, s, list(g_res), seed=SEED)
        m["minus_deletion_intact_ci"] = None if name == "deletion + intact" else paired(y, list(g_res), s, ref)
        report["models"][name] = m
        print(f"{name}: AUROC {m['auroc']} {m['auroc_ci']}  vs deletion+intact {m['minus_deletion_intact_ci']}", flush=True)
    (RUN / "deletion_transfer.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "models"}, indent=1))


if __name__ == "__main__":
    main()
