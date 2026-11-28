"""Choose the 12 residues for the matched-context feasibility experiment, before any new replay.

Criteria (fixed 2026-11-28, written here before the experiment runs):
  Strata from the frozen v3 replay, three residues each:
    A  violation-inducing and deletion-like (fails exactly the probes deletion fails)
    B  violation-inducing but not deletion-like (the partial-damage candidates)
    C  harmless, and the heuristic found a policy span (rule text kept)
    D  harmless, no policy span found (rule text gone, yet no violation)
  Within each stratum: candidates sorted by (rule, condition), shuffled with seed 20261128, then
  taken in order, skipping any rule already chosen, so the 12 residues come from 12 different rules.
  Selection uses only v3 replay outcomes and the span heuristic, never any new result.

Writes configs/stage2_feasibility_residues.json (the list and the criteria). The pilot run is read
only. This is a feasibility and controlled-replication experiment, not the confirmatory study.

    python paper/analysis/select_feasibility.py     (from the repository root)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from mtm.analysis import analyze
from mtm.context import build_splits
from mtm.experiment import read_jsonl
from mtm.rules import load_rules
from mtm.validity import Run

RUN = Path("runs/pilot-haiku-v3")
OUT = Path("configs/stage2_feasibility_residues.json")
SEED, PER_STRATUM = 20261128, 3


def main() -> None:
    rules = load_rules()
    residues = read_jsonl(RUN / "residues.jsonl")
    harmful = {(r["rule_id"], r["condition"]): r["harmful"] for r in analyze(RUN)["residues"]}
    run = Run(RUN)
    spans = {(s["rule_id"], s["condition"]): bool(s["spans"]) for s in build_splits(rules, residues, RUN, write=False)}
    strata: dict[str, list[tuple[str, str]]] = {"A": [], "B": [], "C": [], "D": []}
    for key in sorted(harmful):
        if harmful[key]:
            deletion_like = run.fails[key] == run.fails.get((key[0], "absent"), set())
            strata["A" if deletion_like else "B"].append(key)
        else:
            strata["C" if spans[key] else "D"].append(key)
    rng = random.Random(SEED)
    chosen, used_rules = {}, set()
    for name in "ABCD":
        pool = list(strata[name])
        rng.shuffle(pool)
        picked = []
        for key in pool:
            if key[0] not in used_rules and len(picked) < PER_STRATUM:
                picked.append(key)
                used_rules.add(key[0])
        chosen[name] = picked
    out = {"criteria": __doc__.split("Writes")[0].strip(), "seed": SEED,
           "stratum_sizes": {k: len(v) for k, v in strata.items()},
           "strata": {k: [f"{r}|{c}" for r, c in v] for k, v in chosen.items()},
           "matched_residues": [f"{r}|{c}" for v in chosen.values() for r, c in v],
           "rules": sorted(used_rules)}
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("stratum_sizes", "strata", "rules")}, indent=1))


if __name__ == "__main__":
    main()
