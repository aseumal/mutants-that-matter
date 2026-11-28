"""Write the manuscript's result tables from the saved run, so no number is typed by hand.

    python paper/make_tables.py            (from the repository root)

Run `mtm analyze`, `mtm validity` and `mtm predict` on configs/pilot_haiku_v3.yaml first.
"""

from __future__ import annotations

import json
from pathlib import Path

from mtm.analysis import analyze

RUN = Path("runs/pilot-haiku-v3")
OUT = Path("paper/tables")
LABELS = {"rule_absent": "rule absent", "catastrophic_operators": "catastrophic operators",
          "all_mutants": "all mutants", "llm_mutants": "LLM mutants", "decay_operators": "decay operators",
          "random": "random", "all_mutants_given_absent": "all mutants, deletion removed"}
ci = lambda v: "n/a" if not v else f"[{v[0]:.2f}, {v[1]:.2f}]".replace("-0.00", "0.00")
tt = lambda s: "\\texttt{" + s.replace("_", "\\_") + "}"


def table(name: str, spec: str, header: list[str], rows: list[list[str]], caption: str, size: str = "\\small") -> None:
    body = "\n".join(" & ".join(r) + " \\\\" for r in rows)
    (OUT / f"{name}.tex").write_text(
        f"\\begin{{table}}[t]\n\\centering{size}\\setlength{{\\tabcolsep}}{{4pt}}\n"
        f"\\caption{{{caption}}}\\label{{tab:{name}}}\n"
        f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{' & '.join(header)} \\\\\n\\midrule\n{body}\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    validity = json.loads((RUN / "validity.json").read_text())
    rq4 = json.loads((RUN / "rq4.json").read_text())
    result = analyze(RUN)
    kills = {r["operator"]: r for r in result["operators"]}
    killed = lambda op: next(v for k, v in kills[op].items() if "kill" in k)

    ops = sorted(validity["operators"], key=lambda r: -r["jaccard"])
    rows = []
    for r in ops:
        name = r["operator"]
        label = "rule absent" if name == "rule_absent" else tt(name) + (" (control)" if r["control"] else "")
        rows.append([label, "--" if name == "rule_absent" else f"{killed(name):.0%}".replace("%", "\\%"),
                     f"{r['coupling']:.0%}".replace("%", "\\%"), f"{r['chance']:.0%}".replace("%", "\\%"),
                     ci(r["excess_coupling_ci"]), f"{r['hard_chance']:.0%}".replace("%", "\\%"),
                     ci(r["excess_hard_ci"]), f"{r['jaccard']:.2f}", f"{r['rate_gap']:.2f}"])
    table("operators", "lrrrcrcrr",
          ["Condition", "Killed", "Coupl.", "Chance", "Excess CI", "Chance$_d$", "Excess$_d$ CI", "Jacc.", "Gap"], rows,
          "Fidelity of each condition to harmful residues. Chance keeps failing-set sizes; "
          "Chance$_d$ also favors probes that fail often. Intervals resample rules.", "\\footnotesize")

    one, two = (validity["suites"]["fixed_size"][k] for k in ("1_per_rule", "2_per_rule"))
    free = validity["suites"]["uncontrolled"]
    cell = lambda c: f"{c['rho']:.2f} {ci(c['ci'])}"
    order = ["rule_absent", "catastrophic_operators", "all_mutants", "llm_mutants", "decay_operators",
             "all_mutants_given_absent"]
    table("suites", "lccr", ["Suite score", "1 probe per rule", "2 probes per rule", "Any size"],
          [[LABELS[n], cell(one[n]), cell(two[n]), f"{free[n]:.2f}"] for n in order],
          "Spearman correlation between a suite's score and the share of harmful residues it detects. "
          "Suites overlap and each rule has 3 probes, so this is exploratory.")

    rows = []
    for name in ["rule_absent", "catastrophic_operators", "all_mutants", "llm_mutants", "decay_operators", "random"]:
        a, b = (block["strategies"][name] for block in validity["selection"])
        rows.append([LABELS[name], f"{a['detection']:.0%}".replace("%", "\\%"), "n/a" if name == "random" else ci(a["minus_random_ci"]),
                     "n/a" if name == "rule_absent" else ci(a["minus_absent_ci"]), f"{b['detection']:.0%}".replace("%", "\\%")])
    table("selection", "lrccr", ["Strategy", "Found (k=1)", "Minus random", "Minus deletion", "Found (k=2)"], rows,
          "Share of harmful residues found when each strategy picks $k$ probes per rule without seeing residues. "
          "Differences are for $k=1$, paired, resampling rules with harm.")

    names = {"logreg_embeddings": "logistic regression", "closest_sentence_similarity": "closest-sentence similarity",
             "mlp_embeddings": "MLP", "keyword_recall": "keyword recall", "logreg_trained_on_mutants": "trained on mutants only"}
    fmt = lambda v, c: f"{v:.3f} [{c[0]:.3f}, {c[1]:.3f}]"
    table("predict", "lcc", ["Model", "AUROC", "AUPRC"],
          [[label, fmt(rq4["models"][k]["auroc"], rq4["models"][k]["auroc_ci"]),
            fmt(rq4["models"][k]["auprc"], rq4["models"][k]["auprc_ci"])] for k, label in names.items()],
          f"Predicting harm from text on unseen rules ({rq4['examples']} residues, {rq4['positives']} harmful; "
          f"random ranking gives AUROC 0.5 and AUPRC {rq4['prevalence']}). Intervals resample rules.")
    appendix_tables()
    rq1_tables()
    print("wrote", sorted(p.name for p in OUT.glob("*.tex")))


LABELS_CSV = Path("labels_a.csv")
LABEL_COLUMNS = ["damage_type", "semantic_preservation", "surface_presence", "annotation_confidence"]


def rq1_tables() -> None:
    """RQ1 results from the author's labels, written only once every row is labelled, so that no
    label is ever seen next to replay results while labeling is still going on."""
    import csv

    from mtm.analysis import DAMAGE, PRESERVATION, label_summary

    target = OUT / "rq1.tex"
    if not LABELS_CSV.exists():
        return
    with LABELS_CSV.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    done = sum(all(r.get(c, "").strip() for c in LABEL_COLUMNS) for r in rows)
    if done < len(rows) or not rows:
        target.unlink(missing_ok=True)
        print(f"RQ1: {done} of {len(rows)} residues labelled; RQ1 tables are written only when all are done")
        return
    s = label_summary(LABELS_CSV, RUN)
    if s["invalid"] or s["unmatched"]:
        raise SystemExit(f"RQ1: fix the label sheet first: invalid {s['invalid']}, unmatched {s['unmatched']}")
    body = []
    for column, values, name in (("damage_type", DAMAGE, "Damage type"),
                                 ("semantic_preservation", PRESERVATION, "Semantic preservation")):
        body.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{name}}}}} \\\\")
        for v in values:
            c = s[column].get(v)
            if c:
                body.append(f"\\quad {tt(v)} & {c['residues']} & {c['harmful']} & {c['presence_pass']} \\\\")
    (OUT / "rq1.tex").write_text(
        "\\begin{table}[t]\n\\centering\\small\\setlength{\\tabcolsep}{4pt}\n"
        f"\\caption{{RQ1: the author's labels of all {s['labelled']} residues against replay and the presence check. "
        f"One annotator, blinded to replay; descriptive only. Violation-inducing residues whose text still looks "
        f"present or preserved: {len(s['harmful_but_text_looks_preserved'])}. Harmless residues labelled as "
        f"having lost the rule: {len(s['harmless_though_rule_absent'])}.}}\\label{{tab:rq1}}\n"
        "\\begin{tabular}{lrrr}\n\\toprule\nLabel & Residues & Violation-inducing & Passed presence check \\\\\n"
        "\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n", encoding="utf-8")


def appendix_tables() -> None:
    """Diagnostics added on 2026-11-27, all post hoc: threshold sweep, selection cost,
    continuous violation probability and the synthetic-to-real transfer ablations."""
    from mtm.stage2 import Stage2, selection_cost, thresholds

    pct = lambda v: f"{v:.0%}".replace("%", "\\%")
    sweep = thresholds(RUN)
    names = {"0.3": "at least 1 of 3", "0.5": "at least 2 of 3 (main)", "1.0": "all 3"}
    table("thresholds", "lrrrrr", ["A probe fails when", "Harmful", "Same as deletion", "Deletion-guided",
                                   "All mutants", "Random"],
          [[names[k], str(v["harmful_residues"]), str(v["deletion_identical"]), pct(v["selection_k1"]["rule_absent"]),
            pct(v["selection_k1"]["all_mutants"]), pct(v["selection_k1"]["random"])] for k, v in sweep.items()],
          "Sensitivity to the failing threshold. Selection picks one probe per rule. The same three samples are "
          "reclassified, so this tests the classification choice, not new sampling.", "\\footnotesize")

    cost = selection_cost(RUN)
    order = ["rule_absent", "catastrophic_operators", "all_mutants", "llm_mutants", "decay_operators", "random"]
    rows = []
    for k in order:
        v = cost["strategies"][k]
        per = v["episodes_per_extra_residue"]
        rows.append([LABELS[k], pct(v["detection"]), f"{v['candidate_episodes']:,}".replace(",", "{,}"),
                     f"{v['extra_residues_vs_random']:.1f}", "n/a" if per is None else f"{per:.0f}"])
    table("cost", "lrrrr", ["Strategy", "Found", "Candidate episodes", "Extra vs random", "Episodes per extra"], rows,
          "Cost of choosing one probe per rule: candidate episodes are the strategy's conditions times eligible probes "
          f"times {cost['samples_per_probe']} samples. Extra residues are found beyond random selection "
          f"({cost['harmful_residues']} harmful residues).")

    prof = Stage2(RUN).rate_profiles()
    groups = ["intact", "absent", "residue", "mutant:flip_negation", "mutant:drop_predicate", "mutant:narrow_scope",
              "mutant:generalize", "mutant:llm", "mutant:add_exception", "mutant:weaken_modal", "mutant:hedge",
              "mutant:drop_scope"]
    label = lambda g: {"intact": "intact rule", "absent": "rule absent", "residue": "residues",
                       "mutant:llm": "LLM mutants"}.get(g, tt(g.split(":")[1]) if ":" in g else g)
    table("rates", "lrc", ["Condition", "Condition-probe pairs", "Mean violation probability"],
          [[label(g), str(prof[g]["mean_violation"]["n"]),
            "0 observed (bootstrap degenerate)" if prof[g]["mean_violation"]["rate"] == 0 else
            f"{prof[g]['mean_violation']['rate']:.2f} {ci(prof[g]['mean_violation']['ci'])}"] for g in groups if g in prof],
          "Continuous outcome: mean per-probe violation probability over eligible probes. The intact rule "
          "violated nothing, so each value is also the difference from intact. Intervals resample rules; where no "
          "violation was observed the bootstrap cannot produce one, so no interval is shown (Lemma~3 gives a "
          "bound only under independence).")

    cats = Stage2(RUN).categories()
    table("categories", "lrrrr", ["Rule category", "Residues", "Violation-inducing", "Deletion-like",
                                  "Presence false alarms"],
          [[k, str(v["residues"]), str(v["violation_inducing"]), str(v["deletion_like"]),
            f"{v['presence_false_alarm']} of {v['harmless']}"] for k, v in cats.items()],
          "The pooled RQ1 numbers per rule category. Presence false alarms are harmless residues the keyword "
          "check reports as missing. Categories hold 2 to 6 rules, so differences are descriptive.")

    labels = {"rule_absent": "deletion", "catastrophic_operators": "catastrophic", "all_mutants": "all mutants",
              "llm_mutants": "LLM mutants", "decay_operators": "decay", "random": "random"}
    anchor = {"rule_absent": "south west", "catastrophic_operators": "south west", "all_mutants": "north east",
              "llm_mutants": "south east", "decay_operators": "north west", "random": "west"}
    pts = "\n".join(f"    ({v['candidate_episodes']}, {100 * v['detection']:.1f}) [{labels[k]}]"
                    for k, v in cost["strategies"].items())
    nodes = "\n".join(f"  \\node[anchor={anchor[k]}, font=\\scriptsize, inner sep=2pt] at (axis cs:{v['candidate_episodes']},"
                      f"{100 * v['detection']:.1f}) {{{labels[k]}}};" for k, v in cost["strategies"].items())
    (OUT / "fig_cost.tex").write_text(
        "\\begin{figure}[t]\n\\centering\n\\begin{tikzpicture}\n\\begin{axis}[width=0.78\\textwidth, height=5.2cm,\n"
        "  xlabel={Candidate replay episodes to choose one probe per rule}, ylabel={Violation-inducing residues found (\\%)},\n"
        "  xmin=-150, xmax=2900, ymin=66, ymax=101, grid=major, grid style={gray!20},\n"
        "  tick label style={font=\\scriptsize}, label style={font=\\small}]\n"
        "\\addplot[only marks, mark=*, mark size=2.2pt, color=black] coordinates {\n" + pts + "\n};\n"
        + nodes + "\n\\end{axis}\n\\end{tikzpicture}\n"
        "\\caption{Observed detection against candidate-selection cost. Among guided strategies, choosing probes by deletion found "
        "the most violation-inducing residues for the fewest candidate episodes; mutant-based strategies cost more and found fewer. "
        "Point estimates with substantial uncertainty; mutant generation cost is not counted. Same pilot data as "
        "Tables~\\ref{tab:selection} and~\\ref{tab:cost}; probes are reused, so this is not a held-out estimate.}"
        "\\label{fig:cost}\n\\end{figure}\n", encoding="utf-8")

    sc = json.loads(Path("paper/analysis/results/shortcuts.json").read_text())
    order = ["keyword recall (1 - recall)", "summary length only", "keyword + length", "rule embedding only",
             "whole summary only", "closest sentence only", "two cosines only", "deletion + intact training",
             "full RQ4 features (reference)", "full features, leave one category out"]
    names_sc = {"keyword recall (1 - recall)": "keyword recall (baseline)"}
    rows = []
    for k in order:
        v = sc["models"][k]
        rows.append([names_sc.get(k, k), f"{v['auroc']:.3f} {ci(v['auroc_ci'])}",
                     "--" if v["minus_keyword_ci"] is None else ci(v["minus_keyword_ci"]),
                     f"{v['auroc_keep_rule_words']:.2f}"])
    mm = sc["mutant_to_mutant"]
    kw_sub = sc["keep_rule_words"]
    table("shortcuts", "lccc", ["Predictor (leave one rule out)", "AUROC", "Minus keyword", "Keeps rule words"], rows,
          "Shortcut diagnostics on the 120 residues. Minus keyword is a paired difference resampling rules. The last "
          f"column uses only the {kw_sub['residues']} residues that keep any rule word, of which only "
          f"{kw_sub['violation_inducing']} induce violations, so it is unreliable. Mutant-to-mutant prediction "
          f"of kills reached {mm['auroc']:.3f} {ci(mm['auroc_ci'])}.", "\\footnotesize")

    dt = json.loads((RUN / "deletion_transfer.json").read_text())
    m = dt["models"]
    sizes = dt["fold_training_sizes"]
    span = lambda k: f"{sizes[k][0]}" if sizes[k][0] == sizes[k][1] else f"{sizes[k][0]}--{sizes[k][1]}"
    rows = [["deletion + intact", span("deletion + intact"), f"{m['deletion + intact']['auroc']:.3f} {ci(m['deletion + intact']['auroc_ci'])}", "--"]]
    for k, lab in [("mutants", "mutants"), ("mutants + deletion, mutant weight 0.25", "mutant weight 0.25"),
                   ("mutants + deletion, mutant weight 0.5", "mutant weight 0.5"),
                   ("mutants + deletion, mutant weight 0.75", "mutant weight 0.75"),
                   ("mutants, 3 operators chosen per fold", "3 best-coupling operators"),
                   ("real residues (within domain)", "real residues of other rules")]:
        size = "--" if k.startswith("real") else span(k)
        rows.append([lab, size, f"{m[k]['auroc']:.3f} {ci(m[k]['auroc_ci'])}", ci(m[k]["minus_deletion_intact_ci"])])
    lo, mean_, hi = dt["size_matched_draw_aurocs"]
    rows.insert(2, ["mutants, size and class matched", span("mutants, size-matched"), f"{mean_:.3f} (draws {lo:.2f}--{hi:.2f})", "--"])
    table("transfer", "lrcc", ["Training data (other rules)", "Per fold", "AUROC on real residues", "Minus deletion"], rows,
          "Synthetic-to-real transfer with one classifier, features and folds. Every example of the held-out rule is "
          "excluded. Differences are paired, resampling rules. Run after all other results were known.")



if __name__ == "__main__":
    main()
