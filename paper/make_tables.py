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


def table(name: str, spec: str, header: list[str], rows: list[list[str]], caption: str) -> None:
    body = "\n".join(" & ".join(r) + " \\\\" for r in rows)
    (OUT / f"{name}.tex").write_text(
        "\\begin{table}[t]\n\\centering\\small\n"
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
          "Chance$_d$ also favors probes that fail often. Intervals resample rules.")

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
    print("wrote", sorted(p.name for p in OUT.glob("*.tex")))


if __name__ == "__main__":
    main()
