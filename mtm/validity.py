"""Does mutation testing say anything about real compaction damage?

Two analyses, both from saved replay rows, so neither calls a model:

- Operator fidelity: how closely each operator's failing probes match those of
  harmful residues, beyond any-overlap coupling, with whole rules resampled.
- Suite analysis: in the spirit of Just et al. (2014), sample probe suites and ask
  whether mutation score predicts the share of harmful residues a suite detects.
  Every suite takes the same number of probes from each rule, so suite size and rule
  coverage cannot explain the result. Residues play no part in computing the score.

With three probes per rule this is exploratory: it shows the test, not a verdict.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from itertools import combinations, permutations
from pathlib import Path
from statistics import mean

from .analysis import CONTROL_OPERATORS, failing_set, violation_rates
from .experiment import read_jsonl

# Operators that destroy or reverse the rule, as against those meant to imitate decay.
CATASTROPHIC_OPERATORS = {"flip_negation", "drop_predicate"}


# ------------------------------------------------------------------ set measures

def set_metrics(mutant: set, residue: set) -> dict[str, float]:
    both, either = len(mutant & residue), len(mutant | residue)
    return {
        "coupled": float(both > 0),
        "recall": both / len(residue) if residue else 0.0,
        "precision": both / len(mutant) if mutant else 0.0,
        "jaccard": both / either if either else 0.0,
    }


def _overlap_probability(k: int, a: int, b: int, n: int) -> float:
    return math.comb(a, k) * math.comb(n - a, b - k) / math.comb(n, b)


def chance_jaccard(a: int, b: int, n: int) -> float:
    """Expected Jaccard for random failing sets of sizes a and b among n probes."""
    if not a or not b or not n:
        return 0.0
    return sum(_overlap_probability(k, a, b, n) * k / (a + b - k)
               for k in range(max(0, a + b - n), min(a, b) + 1))


def chance_coupling(a: int, b: int, n: int) -> float:
    if not a or not b or not n:
        return 0.0
    return 1 - math.comb(n - a, b) / math.comb(n, b)


def _weighted_subsets(size: int, weights: dict[str, float]) -> dict[frozenset, float]:
    """P(subset) when probes are drawn one by one without replacement, in proportion to weight."""
    out: dict[frozenset, float] = defaultdict(float)
    for order in permutations(weights, size):
        prob, left = 1.0, sum(weights.values())
        for probe in order:
            prob *= weights[probe] / left
            left -= weights[probe]
        out[frozenset(order)] += prob
    return out


def difficulty_chance(a: int, b: int, weights: dict[str, float]) -> tuple[float, float]:
    """(expected coupling, expected Jaccard) if failing sets of these sizes favor harder-hit probes."""
    if not a or not b or not weights:
        return 0.0, 0.0
    coupled = jaccard = 0.0
    for sa, pa in _weighted_subsets(a, weights).items():
        for sb, pb in _weighted_subsets(b, weights).items():
            coupled += pa * pb * bool(sa & sb)
            jaccard += pa * pb * len(sa & sb) / len(sa | sb)
    return coupled, jaccard


def zero_event_upper_bound(n: int, alpha: float = 0.05) -> float | None:
    """Exact two-sided upper limit when 0 of n events are seen. Assumes independence."""
    return 1 - (alpha / 2) ** (1 / n) if n else None


def spearman(x: list[float], y: list[float]) -> float | None:
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=values.__getitem__)
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for t in range(i, j + 1):
                out[order[t]] = (i + j) / 2
            i = j + 1
        return out

    return _pearson(ranks(x), ranks(y))


def _pearson(x: list[float], y: list[float]) -> float | None:
    mx, my = mean(x), mean(y)
    sxx = sum((v - mx) ** 2 for v in x)
    syy = sum((v - my) ** 2 for v in y)
    if not sxx or not syy:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sxx * syy)


def partial_correlation(rxy: float | None, rxz: float | None, ryz: float | None) -> float | None:
    if None in (rxy, rxz, ryz) or abs(rxz) >= 1 or abs(ryz) >= 1:
        return None
    return (rxy - rxz * ryz) / math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


def _percentile_ci(values: list[float]) -> list[float] | None:
    if not values:
        return None
    values = sorted(values)
    at = lambda q: values[min(len(values) - 1, int(q * len(values)))]
    return [round(at(0.025), 3), round(at(0.975), 3)]


# ------------------------------------------------------------------ loading

class Run:
    """Failing sets and rates for one run directory."""

    def __init__(self, run_dir: str | Path, tau: float = 0.5):
        run_dir = Path(run_dir)
        self.rates = violation_rates(read_jsonl(run_dir / "behavior.jsonl"))
        self.mutants = read_jsonl(run_dir / "mutants.jsonl")
        self.residues = read_jsonl(run_dir / "residues.jsonl")
        self.rule_ids = sorted({rid for rid, _ in self.rates})
        self.excluded = {r: failing_set(self.rates.get((r, "intact"), {}), set(), tau)
                         for r in self.rule_ids}
        self.fails = {key: failing_set(v, self.excluded[key[0]], tau) for key, v in self.rates.items()}
        self.eligible: dict[str, set[str]] = defaultdict(set)
        for (rid, _), v in self.rates.items():
            self.eligible[rid] |= set(v) - self.excluded[rid]

    def harmful(self) -> list[tuple[str, str]]:
        keys = [(r["rule_id"], r["condition"]) for r in self.residues]
        return [k for k in keys if self.fails.get(k)]

    def conditions_by_group(self) -> dict[str, list[tuple[str, str]]]:
        groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for m in self.mutants:
            key = (m["rule_id"], m["condition"])
            if key in self.fails:
                groups[m["operator"]].append(key)
        groups["rule_absent"] = [(r, "absent") for r in self.rule_ids if (r, "absent") in self.fails]
        return groups

    def probe_weights(self, rule: str) -> dict[str, float]:
        """How often each probe fails across this rule's conditions, smoothed so none is impossible."""
        fails = [f for (rid, cond), f in self.fails.items() if rid == rule and not cond.startswith("intact")]
        return {p: sum(p in f for f in fails) + 0.5 for p in sorted(self.eligible[rule])}

    def rate_gap(self, a: tuple[str, str], b: tuple[str, str]) -> float:
        probes = self.eligible[a[0]]
        return mean(abs(self.rates[a].get(p, 0.0) - self.rates[b].get(p, 0.0)) for p in probes) if probes else 0.0


# ------------------------------------------------------------------ operator fidelity

def operator_fidelity(run: Run, n_boot: int = 2000, seed: int = 7) -> list[dict]:
    """Per operator, over (mutant, harmful residue) pairs in the same rule."""
    harmful_by_rule: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in run.harmful():
        harmful_by_rule[key[0]].append(key)
    rng = random.Random(seed)
    rows = []
    weights = {r: run.probe_weights(r) for r in run.rule_ids}
    for operator, conditions in sorted(run.conditions_by_group().items()):
        by_rule: dict[str, list[dict]] = defaultdict(list)
        for mutant in conditions:
            for residue in harmful_by_rule.get(mutant[0], []):
                fm, fr = run.fails[mutant], run.fails[residue]
                n = len(run.eligible[mutant[0]])
                pair = set_metrics(fm, fr)
                hard_coupled, hard_jaccard = difficulty_chance(len(fm), len(fr), weights[mutant[0]])
                pair.update(chance=chance_coupling(len(fm), len(fr), n),
                            chance_jaccard=chance_jaccard(len(fm), len(fr), n),
                            hard_chance=hard_coupled, hard_chance_jaccard=hard_jaccard,
                            rate_gap=run.rate_gap(mutant, residue))
                by_rule[mutant[0]].append(pair)
        pairs = [p for rule_pairs in by_rule.values() for p in rule_pairs]
        if not pairs:
            continue
        rules = sorted(by_rule)
        excess, excess_jaccard, excess_hard, excess_hard_jaccard = [], [], [], []
        for _ in range(n_boot):
            sample = [p for r in rng.choices(rules, k=len(rules)) for p in by_rule[r]]
            excess.append(mean(p["coupled"] - p["chance"] for p in sample))
            excess_jaccard.append(mean(p["jaccard"] - p["chance_jaccard"] for p in sample))
            excess_hard.append(mean(p["coupled"] - p["hard_chance"] for p in sample))
            excess_hard_jaccard.append(mean(p["jaccard"] - p["hard_chance_jaccard"] for p in sample))
        avg = lambda name: round(mean(p[name] for p in pairs), 3)
        rate_values = [mean(run.rates[c].values()) for c in conditions if run.rates.get(c)]
        rows.append({
            "operator": operator, "control": operator in CONTROL_OPERATORS,
            "pairs": len(pairs), "rules": len(rules),
            "mean_violation_rate": round(mean(rate_values), 3) if rate_values else None,
            "coupling": avg("coupled"), "chance": avg("chance"),
            "excess_coupling_ci": _percentile_ci(excess),
            "recall": avg("recall"), "precision": avg("precision"),
            "jaccard": avg("jaccard"), "chance_jaccard": avg("chance_jaccard"),
            "excess_jaccard_ci": _percentile_ci(excess_jaccard),
            "hard_chance": avg("hard_chance"), "excess_hard_ci": _percentile_ci(excess_hard),
            "hard_chance_jaccard": avg("hard_chance_jaccard"),
            "excess_hard_jaccard_ci": _percentile_ci(excess_hard_jaccard),
            "rate_gap": avg("rate_gap"),
        })
    return rows


# ------------------------------------------------------------------ suite analysis

def score_groups(run: Run) -> dict[str, list[tuple[str, str]]]:
    """Which conditions count toward each candidate score. Controls never count."""
    groups = run.conditions_by_group()
    pick = lambda test: [c for op, conds in groups.items() if test(op) for c in conds]
    is_mutant = lambda op: op != "rule_absent" and op not in CONTROL_OPERATORS
    is_llm = lambda op: op.startswith("llm")
    return {
        "all_mutants": pick(is_mutant),
        "catastrophic_operators": pick(lambda op: op in CATASTROPHIC_OPERATORS),
        "decay_operators": pick(lambda op: is_mutant(op) and not is_llm(op)
                                and op not in CATASTROPHIC_OPERATORS),
        "llm_mutants": pick(is_llm),
        "rule_absent": groups["rule_absent"],
    }


def _tallies(run: Run, rules: list[str], keys_by_score: dict[str, list[tuple[str, str]]]) -> dict:
    """{rule: {probe subset: {score: (detected, total)}}}. Only detectable conditions count,
    since the full probe pool cannot tell apart the rest from equivalent mutants."""
    by_rule: dict[str, dict[str, list[set]]] = {r: defaultdict(list) for r in rules}
    for score, keys in keys_by_score.items():
        for key in keys:
            if run.fails.get(key):
                by_rule[key[0]][score].append(run.fails[key])
    out = {}
    for rule in rules:
        probes = sorted(run.eligible[rule])
        out[rule] = {}
        for size in range(len(probes) + 1):
            for subset in combinations(probes, size):
                chosen = set(subset)
                out[rule][subset] = {score: (sum(bool(f & chosen) for f in sets), len(sets))
                                     for score, sets in by_rule[rule].items()}
    return out


def _suite_scores(tally: dict, rules: list[str], choice: list[tuple], names: list[str]) -> dict[str, float | None]:
    scores = {}
    for name in names:
        hit = total = 0
        for rule, subset in zip(rules, choice):
            d, t = tally[rule][subset].get(name, (0, 0))
            hit += d
            total += t
        scores[name] = hit / total if total else None
    return scores


def _correlate(rows: list[dict], names: list[str]) -> dict:
    """Each score uses the suites where it exists, so a run without LLM mutants keeps the rest."""
    def rho(a: str, b: str) -> float | None:
        usable = [r for r in rows if r.get(a) is not None and r.get(b) is not None]
        return spearman([r[a] for r in usable], [r[b] for r in usable]) if len(usable) >= 3 else None

    out: dict = {name: rho(name, "residues") for name in names}
    if "all_mutants" in names and "rule_absent" in names:
        out["all_mutants_given_absent"] = partial_correlation(
            out["all_mutants"], rho("all_mutants", "rule_absent"), out["rule_absent"])
    return out


def suite_analysis(run: Run, n_suites: int = 2000, n_boot: int = 200, boot_suites: int = 300,
                   seed: int = 7) -> dict:
    groups = score_groups(run)
    names = list(groups)
    keys = {**groups, "residues": run.harmful()}
    rules = [r for r in run.rule_ids if len(run.eligible[r]) >= 3]
    tally = _tallies(run, rules, keys)
    rng = random.Random(seed)
    all_names = names + ["residues"]

    def sample(rule_list: list[str], k: int, count: int, tallied: dict) -> list[dict]:
        pools = {r: list(combinations(sorted(run.eligible[r]), k)) for r in set(rule_list)}
        return [_suite_scores(tallied, rule_list, [rng.choice(pools[r]) for r in rule_list], all_names)
                for _ in range(count)]

    fixed = {}
    for k in (1, 2):
        rows = sample(rules, k, n_suites, tally)
        point = _correlate(rows, names)
        boot: dict[str, list[float]] = defaultdict(list)
        for _ in range(n_boot):
            drawn = rng.choices(rules, k=len(rules))
            for name, value in _correlate(sample(drawn, k, boot_suites, tally), names).items():
                if value is not None:
                    boot[name].append(value)
        fixed[f"{k}_per_rule"] = {
            name: {"rho": None if value is None else round(value, 3), "ci": _percentile_ci(boot.get(name, []))}
            for name, value in point.items()}
        fixed[f"{k}_per_rule"]["mean_residue_detection"] = round(
            mean(r["residues"] for r in rows if r["residues"] is not None), 3)

    # Without the size control, for comparison: larger suites score higher on everything.
    all_probes = [(r, p) for r in rules for p in sorted(run.eligible[r])]
    free_rows = []
    for _ in range(n_suites):
        chosen = set(rng.sample(all_probes, rng.randint(1, len(all_probes))))
        choice = [tuple(p for p in sorted(run.eligible[r]) if (r, p) in chosen) for r in rules]
        free_rows.append(_suite_scores(tally, rules, choice, all_names))
    uncontrolled = _correlate(free_rows, names)

    return {
        "rules": len(rules), "harmful_residues": len(run.harmful()),
        "harmful_rules": len({k[0] for k in run.harmful()}), "fixed_size": fixed,
        "uncontrolled": {k: None if v is None else round(v, 3) for k, v in uncontrolled.items()},
    }


# ------------------------------------------------------------------ probe selection

def selection_preview(run: Run, k: int = 1, n_boot: int = 2000, seed: int = 7) -> dict:
    """Share of harmful residues found when each strategy picks k probes per rule.

    Selection never looks at residues. A strategy keeps every subset that detects the most
    of its own conditions, and detection is averaged over those ties, so a strategy with no
    information per rule scores the same as random in that rule. Residues are unseen, but
    the probes are shared, so this previews the intervention and does not test it.
    """
    groups = score_groups(run)
    names = ["all_mutants", "decay_operators", "catastrophic_operators", "llm_mutants", "rule_absent"]
    rules = [r for r in run.rule_ids if len(run.eligible[r]) >= k + 1]
    tally = _tallies(run, rules, {**{n: groups[n] for n in names}, "residues": run.harmful()})

    def picked(rule: str, name: str | None) -> tuple[float, int]:
        subsets = list(combinations(sorted(run.eligible[rule]), k))
        if name is not None:
            scores = [tally[rule][s].get(name, (0, 0))[0] for s in subsets]
            subsets = [s for s, v in zip(subsets, scores) if v == max(scores)]
        hits = [tally[rule][s].get("residues", (0, 0)) for s in subsets]
        return mean(h[0] for h in hits), hits[0][1]

    per_rule = {str(n): {r: picked(r, n) for r in rules} for n in names + [None]}
    rate = lambda name, drawn: (sum(per_rule[name][r][0] for r in drawn)
                                / max(1, sum(per_rule[name][r][1] for r in drawn)))
    rng = random.Random(seed)
    harmful_rules = [r for r in rules if per_rule["None"][r][1]]
    out: dict = {"k": k, "strategies": {}}
    for name in ["None"] + names:
        vs_random, vs_absent = [], []
        for _ in range(n_boot):
            drawn = rng.choices(harmful_rules, k=len(harmful_rules))
            vs_random.append(rate(name, drawn) - rate("None", drawn))
            vs_absent.append(rate(name, drawn) - rate("rule_absent", drawn))
        out["strategies"]["random" if name == "None" else name] = {
            "detection": round(rate(name, rules), 3),
            "minus_random_ci": _percentile_ci(vs_random),
            "minus_absent_ci": _percentile_ci(vs_absent),
        }
    return out


# ------------------------------------------------------------------ report

def analyze_validity(run_dir: str | Path, tau: float = 0.5, seed: int = 7) -> dict:
    run = Run(run_dir, tau)
    harmful = run.harmful()
    harmful_rules = {k[0] for k in harmful}
    passed = {(r["rule_id"], r["condition"]) for r in run.residues if r.get("presence_pass")}
    return {
        "false_assurance": {
            "observed": sum(k in passed for k in harmful), "harmful_residues": len(harmful),
            "upper_95": round(zero_event_upper_bound(len(harmful)), 3) if harmful else None,
            "upper_95_if_one_per_rule": (round(zero_event_upper_bound(len(harmful_rules)), 3)
                                         if harmful_rules else None),
        },
        "operators": operator_fidelity(run, seed=seed),
        "suites": suite_analysis(run, seed=seed),
        "selection": [selection_preview(run, k, seed=seed) for k in (1, 2)],
    }


def write_validity_report(result: dict, run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    (run_dir / "validity.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    ci = lambda v: "n/a" if not v else f"[{v[0]:.2f}, {v[1]:.2f}]"
    pct = lambda v: "n/a" if v is None else f"{v:.0%}"
    fa = result["false_assurance"]
    suites = result["suites"]
    lines = [f"# Validity report: {run_dir.name}", "",
             f"False assurance: {fa['observed']} of {fa['harmful_residues']} harmful residues. If none are observed, "
             f"the 95% upper bound is {pct(fa['upper_95'])} treating residues as independent, and "
             f"{pct(fa['upper_95_if_one_per_rule'])} counting one per rule.", "",
             "## Operator fidelity", "",
             "| Operator | Rules | Violation rate | Coupling | Chance | Excess CI | Recall | Precision | Jaccard | Chance | Excess CI | Rate gap |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in result["operators"]:
        label = r["operator"] + (" (control)" if r["control"] else "")
        lines.append(f"| {label} | {r['rules']} | {pct(r['mean_violation_rate'])} | {pct(r['coupling'])} | {pct(r['chance'])} | "
                     f"{ci(r['excess_coupling_ci'])} | {r['recall']:.2f} | {r['precision']:.2f} | {r['jaccard']:.2f} | "
                     f"{r['chance_jaccard']:.2f} | {ci(r['excess_jaccard_ci'])} | {r['rate_gap']:.2f} |")
    lines += ["", "Excess is observed minus chance, with intervals that resample whole rules. An interval that "
              "includes 0 is not distinguishable from chance. Rate gap is the mean absolute difference in per-probe "
              "violation rate between mutant and residue; lower is closer.", "",
              "## Does mutation score predict residue detection?", "",
              f"{suites['rules']} rules, {suites['harmful_residues']} harmful residues from {suites['harmful_rules']} rules. "
              "Spearman correlation with the share of harmful residues a suite detects. "
              "Exploratory: suites share probes, and each rule has only three.", "",
              "| Score | 1 probe per rule | 2 probes per rule | Any size (no control) |", "| --- | --- | --- | --- |"]
    one, two = suites["fixed_size"]["1_per_rule"], suites["fixed_size"]["2_per_rule"]
    fmt = lambda c: "n/a" if not c or c["rho"] is None else f"{c['rho']:.2f} {ci(c['ci'])}"
    for name in ["all_mutants", "catastrophic_operators", "decay_operators", "llm_mutants", "rule_absent",
                 "all_mutants_given_absent"]:
        free = suites["uncontrolled"].get(name)
        lines.append(f"| {name} | {fmt(one.get(name))} | {fmt(two.get(name))} | {'n/a' if free is None else f'{free:.2f}'} |")
    lines += ["", "## Selection preview", "",
              "Share of harmful residues found when each strategy picks probes without seeing residues. "
              "Exploratory: the probes are shared, so this previews the intervention and does not test it.", "",
              "| Strategy | Probes per rule | Detection | Minus random | Minus deletion |", "| --- | --- | --- | --- | --- |"]
    for block in result["selection"]:
        for name, s in block["strategies"].items():
            lines.append(f"| {name} | {block['k']} | {pct(s['detection'])} | {ci(s['minus_random_ci'])} | {ci(s['minus_absent_ci'])} |")
    lines += ["", "## Difficulty-aware chance", "",
              "Chance when failing sets favor probes that fail often across the rule's conditions.", "",
              "| Operator | Coupling | Chance | Excess CI | Jaccard | Chance | Excess CI |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in result["operators"]:
        lines.append(f"| {r['operator']} | {pct(r['coupling'])} | {pct(r['hard_chance'])} | {ci(r['excess_hard_ci'])} | "
                     f"{r['jaccard']:.2f} | {r['hard_chance_jaccard']:.2f} | {ci(r['excess_hard_jaccard_ci'])} |")
    lines += ["", "`rule_absent` scores a suite only on whether it catches rule deletion, so a mutant score must beat "
              "it to add anything. `all_mutants_given_absent` is the partial correlation after removing that.", ""]
    path = run_dir / "validity.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
