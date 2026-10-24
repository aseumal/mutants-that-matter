"""Command line entry point.

    mtm validate                       check rules, graders and filler
    mtm compact  -c configs/pilot.yaml run compaction cycles, save residues
    mtm mutate   -c configs/pilot.yaml generate mutants
    mtm replay   -c configs/pilot.yaml replay probes for every condition
    mtm analyze  -c configs/pilot.yaml coupling report
    mtm pilot    -c configs/pilot.yaml all four stages
    mtm export-labels -c ... -o labels_annotator1.csv
    mtm kappa a.csv b.csv

Add --mock to run with the deterministic mock model (no API key needed).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .analysis import analyze, export_labels, kappa_from_files, write_report
from .experiment import Config, select_rules, stage_compact, stage_mutate, stage_replay
from .grading import violates
from .llm import client_from_env
from .rules import load_rules, validate_rules
from .session import load_filler


def _validate() -> int:
    rules = load_rules()
    problems = validate_rules(rules)
    filler = load_filler()
    for t in filler:
        for r in rules:
            if violates(r.grader, t["call"]):
                problems.append(f"filler task '{t['user']}' violates {r.id}")
    print(f"{len(rules)} rules, {sum(len(r.probes) for r in rules)} probes, {len(filler)} filler tasks")
    if problems:
        print("PROBLEMS:\n- " + "\n- ".join(problems))
        return 1
    print("OK: every grader flags its probes and no filler task violates any rule.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mtm", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["validate", "compact", "mutate", "replay", "analyze",
                                        "pilot", "export-labels", "kappa"])
    ap.add_argument("files", nargs="*", help="label files for kappa")
    ap.add_argument("-c", "--config", default="configs/pilot.yaml")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--mock", action="store_true", help="use the deterministic mock model")
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args(argv)

    if args.mock:
        os.environ["MTM_MOCK"] = "1"

    if args.command == "validate":
        return _validate()
    if args.command == "kappa":
        if len(args.files) != 2:
            ap.error("kappa needs two label CSV files")
        for col in ("label_status", "label_damage_type"):
            print(f"{col}: kappa = {kappa_from_files(args.files[0], args.files[1], col)}")
        return 0

    cfg = Config.load(args.config)
    rules = select_rules(load_rules(), cfg)
    out = cfg.out_dir(args.runs_dir)

    if args.command in ("compact", "pilot"):
        print(f"[compact] {len(rules)} rules x {len(cfg.budgets)} budgets x {cfg.cycles} cycles")
        stage_compact(cfg, rules, client_from_env("compactor"), out)
    if args.command in ("mutate", "pilot"):
        mutator = client_from_env("agent") if cfg.llm_mutants else None
        print(f"[mutate] operators: {cfg.operators or 'all'}; llm mutants per rule: {cfg.llm_mutants}")
        stage_mutate(cfg, rules, out, mutator)
    if args.command in ("replay", "pilot"):
        print(f"[replay] {cfg.samples} samples per probe")
        stage_replay(cfg, rules, client_from_env("agent"), out)
    if args.command in ("analyze", "pilot"):
        result = analyze(out, cfg.fail_threshold)
        path = write_report(result, out)
        print(json.dumps(result["summary"], indent=2))
        print(f"[analyze] report: {path}")
    if args.command == "export-labels":
        path = export_labels(out, {r.id: r for r in load_rules()}, args.output or str(out / "labels.csv"))
        print(f"[export-labels] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
