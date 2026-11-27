"""Build a corrected run from an earlier one without replaying what did not change.

Every saved episode is regraded with the current graders. Episodes that hit the old step
cap with no final reply and no violation are then continued from where they stopped,
under the new cap. Other rows stay as they are, so the comparison with the source run
shows only the effect of these two corrections. The source run is never modified.

Rows for failed parses keep no raw text, so an episode with a parse error resumes
without that turn in its conversation. Such rows are counted in the summary.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .agent import run_episode
from .experiment import (Config, RunMismatch, _run, append_jsonl, check_manifest, conditions_for,
                         read_jsonl)
from .grading import any_violation
from .llm import ChatModel
from .rules import Rule
from .session import summary_frame


def _key(row: dict) -> tuple:
    return row["rule_id"], row["condition"], row["probe_id"], row["sample"]


def is_capped(row: dict, grader: dict) -> bool:
    """Cut off with nothing decided: the agent acted, never finished, and has not violated."""
    return row["final"] is None and bool(row["calls"]) and not any_violation(grader, row["calls"])


def continue_capped(source: Path, out: Path, rules: list[Rule], agent: ChatModel, cfg: Config) -> dict:
    source, out = Path(source), Path(out)
    if source.resolve() == out.resolve():
        raise RunMismatch("continue-capped must write to a new run directory")
    manifest = json.loads((source / "run.json").read_text(encoding="utf-8"))
    check_manifest(out, **manifest)
    check_manifest(out, agent=agent.name)
    for name in ("mutants.jsonl", "residues.jsonl"):
        shutil.copy(source / name, out / name)

    rows = read_jsonl(source / "behavior.jsonl")
    by_id = {r.id: r for r in rules}
    mutants, residues = read_jsonl(out / "mutants.jsonl"), read_jsonl(out / "residues.jsonl")
    summaries = {r.id: dict(conditions_for(r, mutants, residues, cfg.intact_repeats)) for r in rules}
    probes = {p.id: p for r in rules for p in r.probes}

    path = out / "continuations.jsonl"
    done = {_key(r) for r in read_jsonl(path)}
    capped = [r for r in rows if is_capped(r, by_id[r["rule_id"]].grader)]
    jobs = [r for r in capped if _key(r) not in done]

    def work(row: dict) -> None:
        probe = probes[row["probe_id"]]
        ep = run_episode(agent, summary_frame(summaries[row["rule_id"]][row["condition"]]), probe.text,
                         mock_action=probe.violation, max_steps=cfg.max_steps, prior_calls=row["calls"],
                         prior_steps=len(row["calls"]) + row["parse_errors"],
                         temperature=cfg.temperature if cfg.samples > 1 else 0.0)
        append_jsonl(path, {
            "rule_id": row["rule_id"], "condition": row["condition"], "probe_id": row["probe_id"],
            "sample": row["sample"], "calls": ep.calls, "final": ep.final,
            "parse_errors": row["parse_errors"] + ep.parse_errors,
        })

    _run(work, jobs, cfg.workers)

    continued = {_key(r): r for r in read_jsonl(path)}
    merged, regraded = [], 0
    for row in rows:
        grader = by_id[row["rule_id"]].grader
        new = dict(row, continued=False)
        if any_violation(grader, row["calls"]) != row["violated"]:
            regraded += 1
        if _key(row) in continued:
            new.update(continued[_key(row)], continued=True)
        new["violated"] = any_violation(grader, new["calls"])
        # Undecided is not safe: analyses can treat these separately.
        new["censored"] = new["final"] is None and bool(new["calls"]) and not new["violated"]
        # A resume is an exact continuation only if no failed-parse turn had to be left out.
        new["resume_exact"] = None if not new["continued"] else row["parse_errors"] == 0
        merged.append(new)
    tmp = out / "behavior.jsonl.tmp"
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged), encoding="utf-8")
    tmp.replace(out / "behavior.jsonl")

    summary = {
        "source": source.name, "max_steps": cfg.max_steps, "episodes": len(merged),
        "regraded_to_violation": regraded, "continued": len(continued),
        "continued_with_parse_error_turn_missing": sum(r["parse_errors"] > 0 for r in rows if _key(r) in continued),
        "violations_after_continuing": sum(r["continued"] and r["violated"] for r in merged),
        "still_capped": sum(r["continued"] and r["final"] is None for r in merged),
        "censored": sum(r["censored"] for r in merged),
    }
    (out / "revision.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


class ImportRefused(RuntimeError):
    """The source run does not reproduce under the current graders."""


def import_run(source: Path, out: Path, rules: list[Rule], cfg: Config) -> dict:
    """Start a new run from a finished one, so later conditions are added without paying
    for its episodes again. Copies residues, mutants and behavior rows, and refuses unless
    regrading every copied episode with the current graders reproduces its stored verdict
    and every copied probe still exists with the same id. The source is never modified."""
    from .experiment import fingerprint

    source, out = Path(source), Path(out)
    if any((out / f).exists() for f in ("residues.jsonl", "behavior.jsonl", "mutants.jsonl")):
        raise ImportRefused(f"{out} already holds rows; import only into an empty run.")
    by_id = {r.id: r for r in rules}
    probes = {p.id for r in rules for p in r.probes}
    behavior = read_jsonl(source / "behavior.jsonl")
    mismatched = [row for row in behavior
                  if row["rule_id"] in by_id and any_violation(by_id[row["rule_id"]].grader, row["calls"]) != row["violated"]]
    unknown = sorted({row["probe_id"] for row in behavior if row["rule_id"] in by_id} - probes)
    if mismatched or unknown:
        raise ImportRefused(f"{len(mismatched)} verdicts change under the current graders; unknown probes: {unknown}")
    keep = lambda rows: [r for r in rows if r["rule_id"] in by_id]
    for name in ("residues.jsonl", "mutants.jsonl", "behavior.jsonl"):
        rows = keep(read_jsonl(source / name))
        (out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    src_manifest = json.loads((source / "run.json").read_text(encoding="utf-8")) if (source / "run.json").exists() else {}
    manifest = {k: v for k, v in src_manifest.items() if k in ("agent", "compactor")}
    manifest |= {"fingerprint": fingerprint(cfg, rules), "imported_from": source.name,
                 "imported_rows": len(keep(behavior))}
    (out / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"source": source.name, "behavior_rows": manifest["imported_rows"],
            "residues": len(keep(read_jsonl(source / "residues.jsonl"))), "verdicts_reproduced": True}
