"""Command line entry point.

    mtm validate                       check rules, graders and filler
    mtm compact  -c configs/pilot.yaml run compaction cycles, save residues
    mtm mutate   -c configs/pilot.yaml generate mutants
    mtm replay   -c configs/pilot.yaml replay probes for every condition
    mtm analyze  -c configs/pilot.yaml coupling report
    mtm pilot    -c configs/pilot.yaml all four stages
    mtm export-labels -c ... -o labels_annotator1.csv
    mtm kappa a.csv b.csv
    mtm label-summary labels.csv -c ...  labels vs replay and presence check
    mtm predict  -c configs/pilot.yaml RQ4 classifier (needs the [ml] extra)
    mtm validity -c configs/pilot.yaml operator fidelity and probe-suite analysis
    mtm continue-capped -c new.yaml --source old-run  regrade a run and finish its cut-off episodes
    mtm plan     -c configs/stage2_matched.yaml  calls and episodes a config implies; no API use
    mtm stage2   -c configs/stage2_matched.yaml  outcomes, over-refusal, compression, matched context
    mtm import-run -c configs/stage2_matched.yaml --source pilot-haiku-v3  start a run from a finished one
    mtm label-tool [labels_a.csv] [-o labeling/label_tool.html]  local labeling page (no network, blinded)
    mtm label-check exported.csv [--original labels_a.csv]       verify an exported label sheet

Add --mock to run with the deterministic mock model (no API key needed).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .analysis import AGREEMENT_COLUMNS, analyze, export_labels, kappa_from_files, label_summary, write_report
from .experiment import (Config, PartialTrajectory, RunMismatch, check_manifest, estimate, fingerprint,
                         select_rules, stage_compact, stage_mutate, stage_replay)
from .grading import violates
from .llm import client_from_env, load_dotenv
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
    print(f"{len(rules)} rules, {sum(len(r.probes) for r in rules)} prohibited-action probes, "
          f"{sum(len(r.permit_probes) for r in rules)} permitted-action probes, {len(filler)} filler tasks")
    if problems:
        print("PROBLEMS:\n- " + "\n- ".join(problems))
        return 1
    print("OK: every grader flags its probes, no permitted call or filler task violates any rule, "
          "and every completion predicate matches its permitted call.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mtm", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["validate", "compact", "mutate", "replay", "analyze",
                                        "pilot", "export-labels", "kappa", "label-summary", "predict", "validity",
                                        "continue-capped", "plan", "stage2", "import-run",
                                        "label-tool", "label-check"])
    ap.add_argument("files", nargs="*", help="label files for kappa or label-summary")
    ap.add_argument("-c", "--config", default="configs/pilot.yaml")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--mock", action="store_true", help="use the deterministic mock model")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--source", default=None, help="run to revise, for continue-capped")
    ap.add_argument("--original", default="labels_a.csv", help="blank label sheet, for label-check")
    args = ap.parse_args(argv)

    if args.mock:
        os.environ["MTM_MOCK"] = "1"

    if args.command == "validate":
        return _validate()
    if args.command == "label-tool":
        from .labeltool import build_tool

        sheet = args.files[0] if args.files else "labels_a.csv"
        path = build_tool(sheet, args.output or "labeling/label_tool.html")
        print(f"[label-tool] wrote {path}. Open it in a browser; it works offline and autosaves locally.")
        return 0
    if args.command == "label-check":
        from .labeltool import check_labels

        if len(args.files) != 1:
            ap.error("label-check needs one exported CSV file")
        result = check_labels(args.files[0], args.original)
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1
    if args.command == "kappa":
        if len(args.files) != 2:
            ap.error("kappa needs two label CSV files")
        for col in AGREEMENT_COLUMNS:
            print(f"{col}: kappa = {kappa_from_files(args.files[0], args.files[1], col)}")
        return 0

    load_dotenv()
    cfg = Config.load(args.config)
    if os.environ.get("MTM_MOCK") == "1":
        cfg.run_name += "-mock"  # mock rows must never sit where a real run would resume them
    rules = select_rules(load_rules(), cfg)
    if args.command == "plan":
        from pathlib import Path

        print(json.dumps(estimate(cfg, rules, Path(args.runs_dir) / cfg.run_name), indent=2))
        print("Each episode makes 1 to max_steps agent calls; v3 averaged 1.45 (5,076 episodes). Agree the spend before running.")
        return 0
    out = cfg.out_dir(args.runs_dir)
    fp = fingerprint(cfg, rules)
    if args.command == "import-run":
        from .revise import ImportRefused, import_run

        if not args.source:
            ap.error("import-run needs --source <run name>")
        try:
            print(json.dumps(import_run(out.parent / args.source, out, rules, cfg), indent=2))
        except ImportRefused as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        return 0

    try:
        if args.command in ("compact", "pilot"):
            print(f"[compact] {len(rules)} rules x {len(cfg.budgets)} budgets x {cfg.cycles} cycles")
            compactor = client_from_env("compactor")
            check_manifest(out, fp, compactor=compactor.name)
            stage_compact(cfg, rules, compactor, out)
        if args.command in ("mutate", "pilot"):
            mutator = client_from_env("agent") if cfg.llm_mutants else None
            if mutator is not None:
                check_manifest(out, fp, agent=mutator.name)
            print(f"[mutate] operators: {cfg.operators or 'all'}; llm mutants per rule: {cfg.llm_mutants}")
            stage_mutate(cfg, rules, out, mutator)
        if args.command in ("replay", "pilot"):
            print(f"[replay] {cfg.samples} samples per probe")
            agent = client_from_env("agent")
            check_manifest(out, fp, agent=agent.name)
            stage_replay(cfg, rules, agent, out)
        if args.command == "continue-capped":
            if not args.source:
                ap.error("continue-capped needs --source <run name>")
            from .revise import continue_capped

            source = cfg.out_dir(args.runs_dir).parent / args.source
            print(json.dumps(continue_capped(source, out, rules, client_from_env("agent"), cfg), indent=2))
    except (RunMismatch, PartialTrajectory) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if "mtm.llm_anthropic" in sys.modules:
        print(f"[usage] {sys.modules['mtm.llm_anthropic'].USAGE}")
    if args.command in ("analyze", "pilot", "continue-capped"):
        result = analyze(out, cfg.fail_threshold)
        path = write_report(result, out)
        print(json.dumps(result["summary"], indent=2))
        print(f"[analyze] report: {path}")
    if args.command == "export-labels":
        path = export_labels(out, {r.id: r for r in load_rules()}, args.output or str(out / "labels.csv"))
        print(f"[export-labels] wrote {path}")
    if args.command == "label-summary":
        if len(args.files) != 1:
            ap.error("label-summary needs one label CSV file")
        print(json.dumps(label_summary(args.files[0], out, cfg.fail_threshold), indent=2))
    if args.command == "stage2":
        from .stage2 import analyze_stage2, write_stage2_report

        path = write_stage2_report(analyze_stage2(out, cfg.fail_threshold), out)
        print(path.read_text(encoding="utf-8"))
    if args.command == "validity":
        from .validity import analyze_validity, write_validity_report

        path = write_validity_report(analyze_validity(out, cfg.fail_threshold, cfg.seed), out)
        print(path.read_text(encoding="utf-8"))
    if args.command == "predict":
        from .predict import default_encoder, evaluate, load_examples, write_predict_report

        residues, mutants = load_examples(out, {r.id: r for r in load_rules()}, cfg.fail_threshold)
        result = evaluate(residues, mutants, default_encoder(), cfg.seed)
        path = write_predict_report(result, out)
        print(path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
